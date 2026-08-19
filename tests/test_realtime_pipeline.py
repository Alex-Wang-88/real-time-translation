from __future__ import annotations

import asyncio

import pytest

from realtime_meeting.audio import SegmentEvent
from realtime_meeting.language import LanguageEvidenceAggregator, LanguageGuess, normalize_qwen_label
from realtime_meeting.models import Utterance
from realtime_meeting.quality import AsrQualityState, assess_asr_quality
from realtime_meeting.runtime import LiveChineseTranslator, PartialResult, TranslationResult
from realtime_meeting.scheduler import GpuResourceManager, LatestEventQueue, LatestTranslationQueue
from realtime_meeting.session import LiveMeetingSession
from realtime_meeting.storage import LocalMeetingStore


def test_language_evidence_requires_two_observations_and_ignores_unknown() -> None:
    evidence = LanguageEvidenceAggregator(window_seconds=0.6, max_confirmation_seconds=1.2)

    assert evidence.observe(LanguageGuess("en", 0.9), start=0.0, end=0.6) is None
    assert evidence.observe(None, start=0.6, end=1.2) is None
    assert evidence.observe(LanguageGuess("en", 0.9), start=0.2, end=0.8) is None
    assert evidence.stable_code == "en"

    assert evidence.observe(LanguageGuess("zh", 0.9, "mandarin"), start=1.0, end=1.6) is None
    transition = evidence.observe(LanguageGuess("zh", 0.95, "cantonese_guangdong"), start=1.2, end=1.8)
    assert transition is not None
    assert transition.previous == "en"
    assert transition.current == "zh"
    assert transition.boundary is not None

    # Dialect metadata changes in place and never creates another paragraph.
    assert evidence.observe(LanguageGuess("zh", 0.95, "sichuan"), start=1.8, end=2.4) is None
    assert evidence.stable_variant == "sichuan"


def test_language_text_fallback_recognizes_german_and_chinese() -> None:
    assert normalize_qwen_label("unknown", "Guten Tag, wir sind bereit").code == "de"
    assert normalize_qwen_label("unknown", "今天确认发布").code == "zh"


def test_segment_event_slice_preserves_absolute_timestamped_frames() -> None:
    frame = b"\x01\x00" * 320
    event = SegmentEvent(
        "partial",
        frame + frame + frame,
        0.0,
        0.06,
        7,
        frames=((0, frame), (320, frame), (640, frame)),
    )

    sliced = event.slice(0.01, 0.05)
    assert sliced.start == 0.01
    assert sliced.end == 0.05
    assert len(sliced.pcm) == 1_280
    assert sliced.frames == ((0, frame), (320, frame), (640, frame))


@pytest.mark.asyncio
async def test_latest_event_queue_coalesces_partial_but_keeps_final() -> None:
    queue = LatestEventQueue(maxsize=4)
    first = SegmentEvent("partial", b"1", 0.0, 0.2, 1)
    latest = SegmentEvent("partial", b"2", 0.0, 0.4, 2)
    final = SegmentEvent("final", b"3", 0.0, 0.5, 3)

    assert queue.put_nowait(first) is True
    assert queue.put_latest_nowait(latest) is False
    assert queue.put_nowait(final) is True
    assert queue.get_nowait() is final
    queue.task_done()
    with pytest.raises(asyncio.QueueEmpty):
        queue.get_nowait()
    await asyncio.wait_for(queue.join(), timeout=1.0)


@pytest.mark.asyncio
async def test_latest_translation_queue_merges_revisions_and_promotes_final() -> None:
    queue = LatestTranslationQueue(maxsize=2)
    provisional = {"segment_id": "p-1", "source_revision": 1, "text": "old", "final": False}
    latest = {"segment_id": "p-1", "source_revision": 2, "text": "new", "final": False}
    final = {"segment_id": "p-1", "source_revision": 2, "text": "final", "final": True}

    assert queue.put_nowait(provisional) is True
    assert queue.put_nowait(latest) is False
    assert queue.put_nowait(final) is False
    job = await queue.get()
    assert job == final
    queue.task_done(job)
    await asyncio.wait_for(queue.join(), timeout=1.0)


@pytest.mark.asyncio
async def test_final_translation_evicts_queued_provisional_when_capacity_is_full() -> None:
    queue = LatestTranslationQueue(maxsize=1)
    assert queue.put_nowait({"segment_id": "p-1", "text": "partial", "final": False}) is True
    assert queue.put_nowait({"segment_id": "p-2", "text": "final", "final": True}) is True
    assert queue.dropped_provisional == 1
    job = await queue.get()
    assert job is not None and job["segment_id"] == "p-2"
    queue.task_done(job)
    await asyncio.wait_for(queue.join(), timeout=1.0)


@pytest.mark.asyncio
async def test_latest_translation_queue_get_is_cancelable_when_idle() -> None:
    queue = LatestTranslationQueue()
    task = asyncio.create_task(queue.get())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_live_translation_never_downloads_after_startup_preflight(tmp_path, monkeypatch) -> None:
    called = False

    def unexpected_download(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("live translation must stay local")

    monkeypatch.setattr("realtime_meeting.runtime.prepare_opus_mt_model", unexpected_download)
    translator = LiveChineseTranslator(tmp_path / "models", "cpu", autodownload=True)
    result = translator.translate_many(["hello"], "en")[0]
    assert result.status == "failed"
    assert called is False


def test_gpu_resource_manager_records_sync_stage_wait_and_run() -> None:
    manager = GpuResourceManager()
    with manager.acquire_sync("final_asr", priority=5):
        pass
    assert manager.metrics["runs"] == 1
    assert manager.metrics["durations_ms"]["final_asr"] >= 0
    assert "final_asr" in manager.metrics["wait_durations_ms"]


@pytest.mark.asyncio
async def test_timestamped_language_switch_resegments_old_and_new_windows(settings) -> None:
    # This test targets timestamp slicing itself; use the two-observation
    # boundary so the synthetic sequence remains compact. Production defaults
    # to three observations for higher language precision.
    settings.language_conflict_confirmations = 2
    class SwitchingRuntime:
        ready = True
        capabilities_ready = True
        device = "cpu"
        status = "ready"
        metrics: dict[str, object] = {}

        def __init__(self) -> None:
            self.results = iter(
                [
                    ("hello", "en"),
                    ("hello world", "en"),
                    ("mixed", "zh"),
                    ("mixed", "zh"),
                    ("hello old", "en"),
                    ("zh new", "zh"),
                ]
            )
            self.windows: list[tuple[float, float, str | None]] = []

        def new_vad(self):
            return None

        def detect_language(self, _pcm: bytes, **_kwargs):
            return None

        def transcribe_partial(self, event, **kwargs):
            self.windows.append((event.start, event.end, kwargs.get("language")))
            text, language = next(self.results)
            return PartialResult(event.revision, event.start, event.end, text, language, 0.95, "fake")

        def transcribe_final(self, event, **kwargs):
            return self.transcribe_partial(event, **kwargs)

        def translate_text(self, text, language, **_kwargs):
            return TranslationResult(f"译文:{text}", "ready", "fake")

    meeting = LiveMeetingSession(settings, SwitchingRuntime(), LocalMeetingStore(settings.results_dir))
    await meeting.start()
    frame = b"\0\0" * 320
    event = SegmentEvent("partial", b"\0\0" * 32_000, 0.0, 2.0, 1, frames=((0, frame),))
    try:
        await meeting._handle_event(event)
        await meeting._handle_event(event)
        await meeting._handle_event(event)
        await meeting._handle_event(event)
        paragraphs = meeting.load_transcript()
        assert [item.language for item in paragraphs] == ["en", "zh"]
        assert [item.text for item in paragraphs] == ["hello old", "zh new"]
        assert paragraphs[0].closed is True
        assert paragraphs[0].end == pytest.approx(paragraphs[1].start)
        assert meeting.runtime.windows[-2][0] == pytest.approx(paragraphs[0].start)
        assert meeting.runtime.windows[-1][0] == pytest.approx(paragraphs[1].start)
    finally:
        for task in (meeting.worker_task, meeting.translation_worker_task):
            if task and not task.done():
                task.cancel()
        await asyncio.gather(meeting.worker_task, meeting.translation_worker_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_empty_final_asr_retries_same_model_and_records_diagnostics(settings) -> None:
    class RetryingRuntime:
        ready = True
        capabilities_ready = True
        device = "cpu"
        status = "ready"
        metrics: dict[str, object] = {}

        def __init__(self) -> None:
            self.transcribe_calls = 0

        def new_vad(self):
            return None

        def detect_language(self, _pcm: bytes, **_kwargs):
            return None

        def transcribe_final(self, event, **_kwargs):
            self.transcribe_calls += 1
            text = "" if self.transcribe_calls == 1 else "莫得问题"
            return PartialResult(event.revision, event.start, event.end, text, "zh", 0.8, "fake-1.7b")

        def transcribe_partial(self, event, **kwargs):
            return self.transcribe_final(event, **kwargs)

        def translate_text(self, text, language, **_kwargs):
            return TranslationResult(text, "not_needed", "fake") if language == "zh" else TranslationResult("", "failed", "fake")

    runtime = RetryingRuntime()
    meeting = LiveMeetingSession(settings, runtime, LocalMeetingStore(settings.results_dir))
    await meeting.start()
    event = SegmentEvent("final", b"\x01\x00" * 1_600, 0.0, 0.1, 1)
    try:
        await meeting._handle_event(event)
        paragraphs = meeting.load_transcript()
        assert runtime.transcribe_calls == 2
        assert [item.text for item in paragraphs] == ["莫得问题"]
        assert meeting.pipeline_metrics["asr_secondary_retry_attempts"] == 1
        assert meeting.pipeline_metrics["asr_secondary_retry_replaced"] == 1
        assert meeting.pipeline_metrics["asr_secondary_retry_empty_initial"] == 1
        diagnostic = meeting.pipeline_metrics["asr_segment_diagnostics"][0]
        assert diagnostic["attempted"] is True
        assert diagnostic["replaced"] is True
        assert diagnostic["initial_text_empty"] is True
    finally:
        for task in (meeting.worker_task, meeting.translation_worker_task):
            if task and not task.done():
                task.cancel()
        await asyncio.gather(meeting.worker_task, meeting.translation_worker_task, return_exceptions=True)


def test_asr_quality_flags_unstable_final_hypothesis() -> None:
    state = AsrQualityState()
    state.observe("we will ship the beta", 0.9, is_final=False, is_partial=True)
    state.observe("completely unrelated words", 0.3, is_final=True, is_partial=False)

    assessment = assess_asr_quality(
        "completely unrelated words",
        start=0.0,
        end=3.0,
        state=state,
    )

    assert assessment.is_low is True
    assert "unstable_partial" in assessment.reasons
    assert assessment.signals["stability"] < 0.6


@pytest.mark.asyncio
async def test_post_meeting_translation_only_retranslates_low_quality_segments(settings) -> None:
    settings.post_meeting_translation_enabled = True
    class BatchRuntime:
        ready = True
        capabilities_ready = True
        device = "cpu"
        status = "ready"

        def __init__(self) -> None:
            self.calls: list[tuple[list[str], str, dict[str, object]]] = []

        def translate_text_batch(self, texts, language, *, translation_settings=None, **_kwargs):
            self.calls.append((list(texts), language, dict(translation_settings or {})))
            return [TranslationResult(f"会后译文:{text}", "ready", "fake-context-opus") for text in texts]

    runtime = BatchRuntime()
    meeting = LiveMeetingSession(settings, runtime, LocalMeetingStore(settings.results_dir))
    low = Utterance(
        1,
        "p-000001",
        0.0,
        5.0,
        "en",
        None,
        0.2,
        "hello",
        translation_zh="旧译文",
        translation_status="ready",
        closed=True,
    )
    high = Utterance(
        2,
        "p-000002",
        5.0,
        9.0,
        "en",
        None,
        0.95,
        "We will ship the beta next week as planned.",
        translation_zh="保持原译文",
        translation_status="ready",
        closed=True,
    )
    meeting.transcript_store.append(low)
    meeting.transcript_store.append(high)
    meeting.paragraphs = meeting.transcript_store.load()

    changed = await meeting.run_post_meeting_translation()

    assert changed is True
    assert len(runtime.calls) == 1
    assert runtime.calls[0][0] == ["hello"]
    assert runtime.calls[0][1] == "en"
    assert runtime.calls[0][2]["_post_meeting"] is True
    assert runtime.calls[0][2]["_translation_contexts"]
    values = meeting.load_transcript()
    assert values[0].translation_zh == "会后译文:hello"
    assert values[1].translation_zh == "保持原译文"
    assert meeting.post_translation_state == "complete"
    assert meeting.pipeline_metrics["post_translation_candidates"] == 1
    assert meeting.pipeline_metrics["post_translation_retranslated"] == 1


@pytest.mark.asyncio
async def test_post_meeting_translation_does_not_overwrite_new_source(settings) -> None:
    settings.post_meeting_translation_enabled = True
    class StaleRuntime:
        ready = True
        capabilities_ready = True
        device = "cpu"
        status = "ready"

        def __init__(self) -> None:
            self.meeting: LiveMeetingSession | None = None

        def translate_text_batch(self, texts, language, *, translation_settings=None, **_kwargs):
            del texts, language, translation_settings
            assert self.meeting is not None
            self.meeting.paragraphs[0].source_revision += 1
            return [TranslationResult("过期会后译文", "ready", "fake-context-opus")]

    runtime = StaleRuntime()
    meeting = LiveMeetingSession(settings, runtime, LocalMeetingStore(settings.results_dir))
    runtime.meeting = meeting
    item = Utterance(
        1,
        "p-000001",
        0.0,
        5.0,
        "en",
        None,
        0.2,
        "hello",
        translation_zh="实时译文",
        translation_status="ready",
        closed=True,
    )
    meeting.transcript_store.append(item)
    meeting.paragraphs = meeting.transcript_store.load()

    changed = await meeting.run_post_meeting_translation()

    assert changed is False
    assert meeting.load_transcript()[0].translation_zh == "实时译文"
    assert meeting.pipeline_metrics["stale_translation_results"] == 1

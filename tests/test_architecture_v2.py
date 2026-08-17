from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from realtime_meeting.audio import SegmentEvent
from realtime_meeting.config import Settings, default_meeting_settings, normalize_meeting_settings
from realtime_meeting.exporter import export_live_result
from realtime_meeting.jimo import JimoClient, TodoGenerator, transcript_chunks
from realtime_meeting.language import OFFICIAL_SPEECH_VARIANTS, is_mixed_source_text, normalize_qwen_label
from realtime_meeting.models import TodoDocument, TodoItem, Utterance
from realtime_meeting.runtime import LiveModelRuntime, PartialResult, TranslationResult, _canonical_qwen_model
from realtime_meeting.server import create_app
from realtime_meeting.session import LiveMeetingSession
from realtime_meeting.storage import LocalMeetingStore, TranscriptStore


def test_qwen_language_and_variant_normalization() -> None:
    assert normalize_qwen_label("English").code == "en"
    assert normalize_qwen_label("German").code == "de"
    assert normalize_qwen_label("Chinese").code == "zh"
    assert len(OFFICIAL_SPEECH_VARIANTS) == 22
    official_labels = {
        "Anhui": "anhui",
        "Dongbei": "dongbei",
        "Fujian": "fujian",
        "Gansu": "gansu",
        "Guizhou": "guizhou",
        "Hebei": "hebei",
        "Henan": "henan",
        "Hubei": "hubei",
        "Hunan": "hunan",
        "Jiangxi": "jiangxi",
        "Ningxia": "ningxia",
        "Shandong": "shandong",
        "Shaanxi": "shaanxi",
        "Shanxi": "shanxi",
        "Sichuan": "sichuan",
        "Tianjin": "tianjin",
        "Yunnan": "yunnan",
        "Zhejiang": "zhejiang",
        "Cantonese (Hong Kong accent)": "cantonese_hong_kong",
        "Cantonese (Guangdong accent)": "cantonese_guangdong",
        "Wu language": "wu",
        "Minnan language": "minnan",
    }
    assert set(official_labels.values()) == set(OFFICIAL_SPEECH_VARIANTS)
    for raw_label, expected in official_labels.items():
        assert normalize_qwen_label(raw_label).speech_variant == expected
    assert normalize_qwen_label("Cantonese (Hong Kong accent)").speech_variant == "cantonese_hong_kong"
    assert normalize_qwen_label("Cantonese (Guangdong accent)").speech_variant == "cantonese_guangdong"
    assert normalize_qwen_label("Sichuanese").speech_variant == "sichuan"
    assert normalize_qwen_label("Wu language").speech_variant == "wu"
    assert normalize_qwen_label("Minnan language").speech_variant == "minnan"
    assert normalize_qwen_label("Zhejiang dialect").speech_variant == "zhejiang"
    assert normalize_qwen_label("杭州方言").speech_variant == "zhejiang"
    assert normalize_qwen_label("Anhui").speech_variant == "anhui"
    assert normalize_qwen_label("Shaanxi").speech_variant == "shaanxi"
    assert normalize_qwen_label("Shanxi").speech_variant == "shanxi"
    assert normalize_qwen_label("English", "中文 test").code == "en"
    assert is_mixed_source_text("中文 test") is True
    assert normalize_qwen_label("not-a-language").code == "unknown"


def test_settings_default_to_single_1_7b_no_lid_and_have_no_removed_controls() -> None:
    settings = Settings()
    assert settings.asr_primary == "Qwen/Qwen3-ASR-1.7B"
    assert settings.single_asr_model is True
    assert settings.asr_fallback == settings.asr_primary
    assert settings.language_id_model == settings.asr_primary
    assert settings.language_id_on_segment is True
    assert settings.language_conflict_confirmations == 3
    assert settings.post_meeting_translation_enabled is False
    assert settings.recognition_architecture == "single_1_7b_no_lid"
    values = default_meeting_settings(settings)
    assert "keep_audio" in values
    assert values["realtime_asr_model"] == "primary"
    assert values["recognition_architecture"] == "single_1_7b_no_lid"
    assert normalize_meeting_settings({"realtime_asr_model": "0.6b"}, settings)["realtime_asr_model"] == "primary"
    legacy = Settings(single_asr_model=False, asr_fallback="Qwen/Qwen3-ASR-0.6B")
    assert normalize_meeting_settings({"realtime_asr_model": "0.6b"}, legacy)["realtime_asr_model"] == "small"
    assert normalize_meeting_settings({"recognition_architecture": "router_mixed"}, settings)["recognition_architecture"] == "single_1_7b_no_lid"
    assert not any("refine" in key or "speaker" in key or "diar" in key for key in values)


def test_qwen_runtime_uses_cantonese_adapter_and_small_model_for_lid() -> None:
    calls: list[dict[str, object]] = []

    class FakeQwen:
        def transcribe(self, **kwargs):
            calls.append(kwargs)
            return [{"text": "你好嗎", "language": "Cantonese (Guangdong accent)"}]

    runtime = LiveModelRuntime("Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B", "cpu")
    fake = FakeQwen()
    runtime.primary = fake
    runtime.fallback = fake
    runtime.language_id = fake
    result = runtime.detect_language(np.zeros(16_000, dtype=np.int16).tobytes())
    assert result is not None and result.code == "zh" and result.speech_variant == "cantonese_guangdong"
    runtime._qwen_decode(b"\0\0" * 160, fake, language="zh", speech_variant="cantonese_guangdong")
    assert calls[-1]["language"] == "Cantonese"
    assert runtime.capability_snapshot()["asr_primary"]["model"] == "Qwen/Qwen3-ASR-1.7B"


def test_qwen_runtime_can_select_the_small_model_for_realtime() -> None:
    calls: list[str] = []

    class FakeQwen:
        def __init__(self, name: str) -> None:
            self.name = name

        def transcribe(self, **_kwargs):
            calls.append(self.name)
            return [{"text": "hello", "language": "English"}]

    runtime = LiveModelRuntime("primary", "small", "cpu")
    runtime.primary = FakeQwen("primary")
    runtime.fallback = FakeQwen("small")
    event = SegmentEvent("partial", b"\0\0" * 160, 0.0, 1.0, 1)
    result = runtime.transcribe_partial(event, decode_settings={"realtime_asr_model": "small"})
    assert result.model == "small"
    assert calls == ["small"]
    runtime.transcribe_partial(event, decode_settings={"realtime_asr_model": "primary"})
    assert calls == ["small", "primary"]


def test_single_qwen_runtime_shares_one_checkpoint_across_roles(monkeypatch) -> None:
    runtime = LiveModelRuntime(
        "Qwen/Qwen3-ASR-1.7B",
        "Qwen/Qwen3-ASR-1.7B",
        "cpu",
        language_id_model="Qwen/Qwen3-ASR-1.7B",
        single_model=True,
        vad_model="disabled",
    )
    fake = object()
    monkeypatch.setattr(runtime.primary_engine, "load", lambda *_args, **_kwargs: fake)
    monkeypatch.setattr(runtime.small_engine, "load", lambda *_args, **_kwargs: pytest.fail("duplicate model load"))
    runtime.load(lambda _message: None)
    assert runtime.primary is fake
    assert runtime.fallback is fake
    assert runtime.language_id is fake
    assert _canonical_qwen_model("Qwen/Qwen3-ASR-1.7B") == _canonical_qwen_model("qwen3-asr-1.7b")
    runtime.close()


def test_runtime_treats_disabled_vad_as_ready(monkeypatch) -> None:
    runtime = LiveModelRuntime(
        "Qwen/Qwen3-ASR-1.7B",
        "Qwen/Qwen3-ASR-1.7B",
        "cpu",
        language_id_model="Qwen/Qwen3-ASR-1.7B",
        vad_model="disabled",
    )
    fake_model = object()

    class ReadyTranslator:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def preflight(self):
            return {"en": {"ready": True}, "de": {"ready": True}}

        def assets_snapshot(self):
            return {"en": {"ready": True}, "de": {"ready": True}}

    monkeypatch.setattr(runtime, "_prepare_model_cache", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(runtime.primary_engine, "load", lambda *_args, **_kwargs: fake_model)
    monkeypatch.setattr(runtime.small_engine, "load", lambda *_args, **_kwargs: pytest.fail("duplicate model load"))
    monkeypatch.setattr("realtime_meeting.runtime.LiveChineseTranslator", ReadyTranslator)
    runtime.load(lambda _message: None)
    try:
        assert runtime.ready is True
        assert runtime.capabilities_ready is True
    finally:
        runtime.close()


def test_transcript_store_is_schema_2_and_has_only_paragraph_fields(tmp_path) -> None:
    store = TranscriptStore(tmp_path / "transcript.jsonl")
    item = Utterance(
        id=1,
        segment_id="p-000001",
        start=0.0,
        end=2.0,
        language="zh",
        speech_variant="cantonese_unknown",
        language_confidence=0.9,
        text="中文",
        closed=True,
    )
    store.append(item)
    loaded = store.load()[0]
    assert loaded.translation_zh == "中文"
    payload = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2.0"
    assert payload["paragraphs"][0]["segment_id"] == "p-000001"
    assert not any(key.startswith("speaker") or key.startswith("recognition") for key in payload["paragraphs"][0])
    store.delete(loaded)
    assert store.load() == []


def test_recovered_todo_and_live_threshold_settings_are_restored(settings) -> None:
    todo = TodoDocument(
        items=[TodoItem(task="发布测试版本", owner="研发", id="todo-1", meeting_id="meeting-recovered", summary_revision=2)],
        meeting_id="meeting-recovered",
        summary_revision=2,
        generated_at="2026-08-17T00:00:00+00:00",
    )
    meeting = LiveMeetingSession(
        settings,
        FakeRuntime(),
        LocalMeetingStore(settings.results_dir),
        meeting_id="meeting-recovered",
        recovered_state={
            "schema_version": "2.0",
            "id": "meeting-recovered",
            "recording_state": "complete",
            "volume_threshold_percent": 1.0,
            "meeting_settings": {"volume_threshold_percent": 4.5},
            "todo": todo.to_dict(),
        },
    )
    assert [item.task for item in meeting.todo.items] == ["发布测试版本"]
    assert meeting.volume_threshold_percent == 4.5
    assert meeting.meeting_settings["volume_threshold_percent"] == 4.5
    meeting.configure_volume_threshold(7.3)
    assert meeting.meeting_settings["volume_threshold_percent"] == 7.3


@pytest.mark.asyncio
async def test_load_transcript_rebinds_active_paragraph(settings) -> None:
    meeting = LiveMeetingSession(settings, FakeRuntime(), LocalMeetingStore(settings.results_dir))
    await meeting._upsert_source(
        SegmentEvent("partial", b"", 0.0, 1.0, 1),
        "正在记录",
        language="zh",
        speech_variant="mandarin",
        confidence=0.9,
        asr_model="fake-qwen",
        language_source="qwen",
    )
    original = meeting.active_paragraph
    loaded = meeting.load_transcript()
    assert loaded and meeting.active_paragraph is loaded[0]
    assert meeting.active_paragraph is not original


@pytest.mark.asyncio
async def test_todo_request_is_bounded_by_jimo_request_limit(settings) -> None:
    settings.jimo_max_request_chars = 1_200

    class CapturingClient:
        def __init__(self) -> None:
            self.messages = None

        async def complete(self, messages, _session_id, **_kwargs):
            self.messages = messages
            return '{"items": []}'

    client = CapturingClient()
    result = await TodoGenerator(settings, client).generate("meeting-1", 1, "行动项内容。" * 2_000)
    assert result.items == []
    assert client.messages is not None
    assert JimoClient.request_chars(client.messages) <= settings.jimo_max_request_chars


class FakeRuntime:
    ready = True
    capabilities_ready = True
    device = "cpu"
    status = "ready"
    metrics: dict[str, object] = {}

    def __init__(self, guesses: list[object] | None = None) -> None:
        self.guesses = list(guesses or [])

    def new_vad(self):
        return None

    def detect_language(self, _pcm: bytes, **_kwargs):
        return self.guesses.pop(0) if self.guesses else None

    def transcribe_partial(self, event, **_kwargs):
        text = {
            1: "We will ship",
            2: "We will ship the beta next week.",
            3: "今天确认发布节奏。",
        }.get(event.revision, "")
        language = "zh" if event.revision == 3 else "en"
        return PartialResult(event.revision, event.start, event.end, text, language, 0.9, "fake-qwen", speech_variant=None)

    def transcribe_final(self, event, **kwargs):
        return self.transcribe_partial(event, **kwargs)

    def translate_text(self, text, language, **_kwargs):
        return TranslationResult(f"译文:{text}", "ready", "fake-opus") if language in {"en", "de"} else TranslationResult(text, "not_needed")

    def capability_snapshot(self):
        return {"asr_primary": {"model": "fake", "ready": True}}


@pytest.mark.asyncio
async def test_default_single_model_strategy_probes_language_once_and_uses_primary(settings) -> None:
    class NoLidRuntime(FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.detect_calls = 0
            self.roles: list[str] = []

        def detect_language(self, _pcm: bytes, **_kwargs):
            self.detect_calls += 1
            from realtime_meeting.language import LanguageGuess

            return LanguageGuess("en", 0.95, raw_qwen_label="English")

        def transcribe_partial(self, event, **kwargs):
            self.roles.append(str(kwargs.get("decode_settings", {}).get("realtime_asr_model")))
            return PartialResult(event.revision, event.start, event.end, "hello", "en", 0.9, "fake-qwen")

    runtime = NoLidRuntime()
    meeting = LiveMeetingSession(settings, runtime, LocalMeetingStore(settings.results_dir))
    await meeting._handle_event(SegmentEvent("partial", b"\0\0" * 16_000, 0.0, 1.0, 1))
    assert runtime.detect_calls == 1
    assert runtime.roles == ["primary"]


@pytest.mark.asyncio
async def test_forced_audio_cut_is_merged_and_language_change_closes_paragraph(settings) -> None:
    from realtime_meeting.language import LanguageGuess

    runtime = FakeRuntime([
        LanguageGuess("en", 0.9, raw_qwen_label="English"),
        LanguageGuess("en", 0.9, raw_qwen_label="English"),
        LanguageGuess("zh", 0.9, "mandarin", raw_qwen_label="Chinese"),
    ])
    meeting = LiveMeetingSession(settings, runtime, LocalMeetingStore(settings.results_dir))
    await meeting.start()
    try:
        await meeting._handle_event(SegmentEvent("partial", b"\0\0" * 16_000, 0.0, 1.0, 1))
        await meeting._handle_event(SegmentEvent("final", b"\0\0" * 16_000, 1.0, 2.0, 2, True))
        await meeting._handle_event(SegmentEvent("final", b"\0\0" * 16_000, 1.0, 2.0, 2, False))
        await meeting._handle_event(SegmentEvent("final", b"\0\0" * 16_000, 2.2, 3.2, 3, False))
        await meeting.translation_queue.join()
        paragraphs = meeting.load_transcript()
        assert len(paragraphs) == 2
        assert paragraphs[0].segment_id == "p-000001"
        assert paragraphs[0].closed is True
        assert paragraphs[1].start >= paragraphs[0].end
        assert paragraphs[0].text == "We will ship the beta next week."
        # The same-model language probe may enrich Chinese with a dialect
        # variant, but it never creates a separate paragraph for that variant.
        assert paragraphs[1].speech_variant == "mandarin"
        assert all("speaker" not in key for item in paragraphs for key in item.to_dict())
    finally:
        for task in (meeting.worker_task, meeting.translation_worker_task):
            if task and not task.done():
                task.cancel()
        await asyncio_gather_cancelled(meeting.worker_task, meeting.translation_worker_task)


@pytest.mark.asyncio
async def test_consistent_asr_language_change_closes_paragraph_without_lid_confirmation(settings) -> None:
    class SwitchingRuntime(FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.results = iter([
                ("hello everyone", "en"),
                ("hello everyone", "en"),
                ("大家好", "zh"),
                ("大家好", "zh"),
                ("大家好", "zh"),
            ])

        def detect_language(self, _pcm: bytes, **_kwargs):
            # The LID can be unavailable on a busy machine; consistent ASR
            # language labels are still enough to split safely after the
            # configured three conflicts.
            return None

        def transcribe_partial(self, event, **_kwargs):
            text, language = next(self.results)
            return PartialResult(event.revision, event.start, event.end, text, language, 0.95, "fake-qwen")

    meeting = LiveMeetingSession(settings, SwitchingRuntime(), LocalMeetingStore(settings.results_dir))
    await meeting.start()
    try:
        event = SegmentEvent("partial", b"\0\0" * 16_000, 0.0, 1.0, 1)
        for _ in range(5):
            await meeting._handle_event(event)
        paragraphs = meeting.load_transcript()
        assert [item.language for item in paragraphs] == ["en", "zh"]
        assert [item.text for item in paragraphs] == ["hello everyone", "大家好"]
        assert paragraphs[0].closed is True
    finally:
        for task in (meeting.worker_task, meeting.translation_worker_task):
            if task and not task.done():
                task.cancel()
        await asyncio_gather_cancelled(meeting.worker_task, meeting.translation_worker_task)


@pytest.mark.asyncio
async def test_empty_final_asr_result_closes_existing_paragraph_without_repeating_text(settings) -> None:
    class EmptyRuntime(FakeRuntime):
        def transcribe_final(self, event, **_kwargs):
            return PartialResult(event.revision, event.start, event.end, "", "unknown", 0.0, "fake-qwen")

    meeting = LiveMeetingSession(settings, EmptyRuntime(), LocalMeetingStore(settings.results_dir))
    await meeting._upsert_source(
        SegmentEvent("partial", b"", 0.0, 1.0, 1),
        "最后一句",
        language="zh",
        speech_variant="mandarin",
        confidence=0.9,
        asr_model="fake-qwen",
        language_source="qwen",
    )
    await meeting._handle_event(SegmentEvent("final", b"\0\0" * 16_000, 1.0, 2.0, 1))
    paragraphs = meeting.load_transcript()
    assert len(paragraphs) == 1
    assert paragraphs[0].text == "最后一句"
    assert paragraphs[0].closed is True


@pytest.mark.asyncio
async def test_vad_only_noise_partial_cannot_create_a_repeated_paragraph(settings) -> None:
    meeting = LiveMeetingSession(settings, FakeRuntime(), LocalMeetingStore(settings.results_dir))
    event = SegmentEvent("partial", b"\0\0" * 16_000, 0.0, 1.0, 1, False, True, False)
    await meeting._handle_event(event)
    assert meeting.load_transcript() == []


def test_audio_source_switch_resets_sequence_without_accepting_old_release(settings) -> None:
    meeting = LiveMeetingSession(settings, FakeRuntime(), LocalMeetingStore(settings.results_dir))
    payload = {"sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"}
    meeting.configure_audio(payload, source_id="source-a")
    meeting.last_audio_sequence = 7
    meeting.configure_audio(payload, source_id="source-b")
    assert meeting.audio_source_id == "source-b"
    assert meeting.last_audio_sequence is None
    meeting.last_audio_sequence = 3
    meeting.release_audio_source("source-a")
    assert meeting.audio_source_id == "source-b"
    assert meeting.last_audio_sequence == 3


@pytest.mark.asyncio
async def test_asr_worker_survives_one_event_failure(settings) -> None:
    class FailingRuntime(FakeRuntime):
        def transcribe_partial(self, _event, **_kwargs):
            raise RuntimeError("synthetic ASR failure")

    meeting = LiveMeetingSession(settings, FailingRuntime(), LocalMeetingStore(settings.results_dir))
    await meeting.start()
    try:
        event = SegmentEvent("partial", b"\0\0" * 160, 0.0, 1.0, 1)
        await meeting.queue.put(event)
        await asyncio.wait_for(meeting.queue.join(), timeout=2.0)
        assert meeting.worker_task is not None and not meeting.worker_task.done()
        await meeting.queue.put(SegmentEvent("partial", b"\0\0" * 160, 1.0, 2.0, 2))
        await asyncio.wait_for(meeting.queue.join(), timeout=2.0)
    finally:
        for task in (meeting.worker_task, meeting.translation_worker_task):
            if task and not task.done():
                task.cancel()
        await asyncio_gather_cancelled(meeting.worker_task, meeting.translation_worker_task)


@pytest.mark.asyncio
async def test_max_duration_requests_finalize_instead_of_leaving_recording_active(settings) -> None:
    meeting = LiveMeetingSession(
        Settings(results_dir=settings.results_dir, max_recording_seconds=0.1, queue_join_timeout_seconds=1),
        FakeRuntime(),
        LocalMeetingStore(settings.results_dir),
    )
    await meeting.start()
    meeting.configure_audio({"sample_rate": 16_000, "channels": 1, "encoding": "pcm_s16le"}, source_id="source-a")
    meeting.started_at = "2000-01-01T00:00:00+00:00"
    with pytest.raises(ValueError):
        await meeting.feed_audio(b"\0\0" * 320, sequence=0, source_id="source-a")
    assert meeting.stop_task is not None
    await asyncio.wait_for(meeting.stop_task, timeout=5.0)
    assert meeting.recording_state == "complete"


@pytest.mark.asyncio
async def test_empty_partial_at_new_technical_revision_does_not_replace_paragraph(settings) -> None:
    meeting = LiveMeetingSession(settings, FakeRuntime(), LocalMeetingStore(settings.results_dir))
    fields = {
        "language": "zh",
        "speech_variant": "mandarin",
        "confidence": 0.9,
        "asr_model": "fake-qwen",
        "language_source": "qwen",
    }
    await meeting._upsert_source(SegmentEvent("partial", b"", 0.0, 1.0, 1), "第一段", **fields)
    await meeting._upsert_source(SegmentEvent("partial", b"", 1.0, 2.0, 2), "", **fields)
    await meeting._upsert_source(SegmentEvent("partial", b"", 2.0, 3.0, 2), "第二段", **fields)
    await meeting._upsert_source(SegmentEvent("partial", b"", 2.0, 3.5, 2), "第二段补充", **fields)

    paragraphs = meeting.load_transcript()
    assert len(paragraphs) == 1
    assert paragraphs[0].text == "第一段 第二段补充"
    assert paragraphs[0].closed is False


@pytest.mark.asyncio
async def test_closed_paragraph_gets_a_new_segment_after_silence(settings) -> None:
    meeting = LiveMeetingSession(settings, FakeRuntime(), LocalMeetingStore(settings.results_dir))
    fields = {
        "language": "zh",
        "speech_variant": "mandarin",
        "confidence": 0.9,
        "asr_model": "fake-qwen",
        "language_source": "qwen",
    }
    await meeting._upsert_source(SegmentEvent("partial", b"", 0.0, 1.0, 1), "第一段", **fields)
    await meeting._close_active(1.1)
    await meeting._upsert_source(SegmentEvent("partial", b"", 1.8, 2.8, 2), "第二段", **fields)

    paragraphs = meeting.load_transcript()
    assert [item.segment_id for item in paragraphs] == ["p-000001", "p-000002"]
    assert paragraphs[0].closed is True
    assert paragraphs[1].closed is False


async def asyncio_gather_cancelled(*tasks) -> None:
    import asyncio

    await asyncio.gather(*(task for task in tasks if task), return_exceptions=True)


@pytest.mark.asyncio
async def test_stale_translation_cannot_overwrite_new_source(settings) -> None:
    runtime = FakeRuntime()
    meeting = LiveMeetingSession(settings, runtime, LocalMeetingStore(settings.results_dir))
    item = Utterance(1, "p-000001", 0.0, 2.0, "en", None, 0.9, "new source", source_revision=2)
    meeting.paragraphs = [item]
    await meeting._run_translation_job({"segment_id": item.segment_id, "source_revision": 1, "text": "old source", "language": "en", "final": True, "attempt": 2})
    assert item.translation_zh == ""


@pytest.mark.asyncio
async def test_mixed_chinese_english_paragraph_is_translated_without_changing_language(settings) -> None:
    runtime = FakeRuntime()
    meeting = LiveMeetingSession(settings, runtime, LocalMeetingStore(settings.results_dir))
    await meeting.start()
    try:
        fields = {
            "language": "zh",
            "speech_variant": "mandarin",
            "confidence": 0.9,
            "asr_model": "fake-qwen",
            "language_source": "qwen",
        }
        await meeting._upsert_source(
            SegmentEvent("partial", b"", 0.0, 1.0, 1),
            "中文 test this",
            **fields,
        )
        await meeting.translation_queue.join()
        item = meeting.load_transcript()[0]
        assert item.language == "zh"
        assert item.translation_status == "ready"
        assert item.translation_zh == "译文:中文 test this"
    finally:
        for task in (meeting.worker_task, meeting.translation_worker_task):
            if task and not task.done():
                task.cancel()
        await asyncio_gather_cancelled(meeting.worker_task, meeting.translation_worker_task)


def test_jimo_reads_latest_paragraph_projection_not_revision_events(tmp_path) -> None:
    path = tmp_path / "transcript.jsonl"
    store = TranscriptStore(path)
    item = Utterance(1, "p-000001", 0.0, 2.0, "en", None, 0.9, "We ship", source_revision=1)
    store.append(item)
    item.text = "We ship the beta"
    item.source_revision = 2
    item.revision = 2
    store.append(item)
    chunks = list(transcript_chunks(path, 10_000))
    assert len(chunks) == 1
    assert "We ship the beta" in chunks[0][3]
    assert chunks[0][3].count("We ship") == 1


def test_export_is_paragraph_based_and_does_not_write_identity_artifacts(tmp_path) -> None:
    item = Utterance(1, "p-000001", 0.0, 2.0, "en", None, 0.9, "We ship", translation_zh="我们发布", translation_status="ready", closed=True)
    files = export_live_result(
        tmp_path,
        meeting_id="m1",
        title="测试",
        started_at="",
        ended_at="",
        duration_seconds=2,
        utterances=[item],
        audio_segments=[],
        recording_state="complete",
        summary_state="idle",
        todo_state="waiting_summary",
    )
    payload = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert "paragraphs" in payload and "utterances" not in payload
    assert "speaker" not in (tmp_path / "meeting_transcript.md").read_text(encoding="utf-8").casefold()
    assert "speaker_segments.json" not in files


def test_export_exposes_summary_and_todo_files_for_download(tmp_path) -> None:
    for name, content in (
        ("meeting_minutes.md", "# 会议纪要"),
        ("todo_list.json", "{\"items\": []}"),
        ("todo_list.md", "# To-do-list"),
    ):
        (tmp_path / name).write_text(content, encoding="utf-8")
    files = export_live_result(
        tmp_path,
        meeting_id="m1",
        title="测试",
        started_at="",
        ended_at="",
        duration_seconds=0,
        utterances=[],
        audio_segments=[],
        recording_state="complete",
        summary_state="complete",
        todo_state="complete",
    )
    assert {"meeting_minutes.md", "todo_list.json", "todo_list.md"} <= set(files)


def test_api_removes_old_route_and_exposes_translation_retry(tmp_path) -> None:
    settings = Settings(results_dir=tmp_path / "meetings", translation_model_root=tmp_path / "models")
    app = create_app(settings, FakeRuntime(), load_models=False, store=LocalMeetingStore(settings.results_dir))
    with TestClient(app) as client:
        created = client.post("/api/v2/meetings", json={"title": "API 测试"})
        assert created.status_code == 201
        meeting_id = created.json()["id"]
        renamed = client.patch(f"/api/v2/meetings/{meeting_id}", json={"title": "重命名后的会议"})
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "重命名后的会议"
        assert client.get(f"/api/v2/meetings/{meeting_id}").json()["title"] == "重命名后的会议"
        persisted = json.loads((tmp_path / "meetings" / meeting_id / "session_state.json").read_text(encoding="utf-8"))
        assert persisted["title"] == "重命名后的会议"
        assert client.post(f"/api/v2/meetings/{meeting_id}/postprocess").status_code == 404
        assert client.post(f"/api/v2/meetings/{meeting_id}/translation/retry").status_code == 409
        assert client.get(f"/api/v2/meetings/{meeting_id}/transcript").json()["paragraphs"] == []


def test_web_client_has_one_paragraph_event_and_no_removed_controls() -> None:
    app_js = (Path(__file__).parents[1] / "realtime_meeting" / "web" / "app.js").read_text(encoding="utf-8").casefold()
    html = (Path(__file__).parents[1] / "realtime_meeting" / "web" / "index.html").read_text(encoding="utf-8").casefold()
    assert "paragraph_update" in app_js
    assert "translation_update" not in app_js
    assert "source_revision" in app_js
    assert "meeting-entry-actions" in app_js
    assert "renamemeeting" in app_js
    assert 'method: "patch"' in app_js
    assert "window.confirm" not in app_js
    assert "window.prompt" not in app_js
    assert 'id="deletemeeting"' not in html
    assert 'id="renamemeetingdialog"' in html
    assert 'id="confirmdialog"' in html
    assert 'id="authdialog"' in html
    assert "state.audiostreamingenabled = false" in app_js
    assert "low_latency" not in app_js
    assert "quality:" not in app_js
    assert 'value="low_latency"' not in html
    assert 'value="quality"' not in html
    assert 'id="recognitionarchitecture"' not in html
    for removed in ("asrspeechstartms", "asraudioprerollms", "asrminimumspeechms", "asrspeechratio", "asrmaxutteranceseconds"):
        assert removed not in html
    for removed in ("language_lock", "dialect_hint", "postprocess", "refinement", "diarization", "speaker"):
        assert removed not in app_js
        assert removed not in html

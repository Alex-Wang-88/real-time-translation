from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from realtime_meeting.diarization import SpeakerSegment
from realtime_meeting.models import TodoDocument, TodoItem, Utterance
from realtime_meeting.audio import SAMPLE_RATE, SegmentEvent
from realtime_meeting.session import LiveMeetingSession
from realtime_meeting.session import SessionManager
from realtime_meeting.storage import LocalMeetingStore


class FakeRuntime:
    ready = True
    device = "cpu"
    status = "模型已就绪"
    metrics = {}

    def new_vad(self):
        return None

    def new_speaker_clusterer(self):
        return None

    def transcribe_partial(self, event, recent_text="", hotwords=None):
        from realtime_meeting.runtime import PartialResult

        return PartialResult(event.revision, event.start, event.end, "hello", "en", 0.9, "fake")

    def transcribe_final(self, event, **kwargs):
        stage = "refined" if kwargs.get("refined") else "fast"
        revision = 2 if kwargs.get("refined") else 1
        return [Utterance(kwargs["next_id"], event.start, event.end, 1, "en", 0.9, f"We will send the {stage} plan.", "", "pending", f"{event.revision}:0", revision, stage)]

    def translate_text(self, text, source_language):
        from realtime_meeting.runtime import TranslationResult

        return TranslationResult("我们会发送方案。", "ready", "fake")


class DiarizationRuntime(FakeRuntime):
    def __init__(self) -> None:
        self.paths_seen: list[Path] = []

    def diarize_audio(self, paths):
        self.paths_seen = list(paths)
        assert all(path.is_file() for path in paths)
        return [SpeakerSegment(0.0, 1.0, "speaker_1", 0.9)]


class FakeSummarizer:
    calls = 0
    attempts: list[str | None] = []

    async def summarize(self, transcript_path, meeting_id, started_at, ended_at, **kwargs):
        type(self).calls += 1
        type(self).attempts.append(kwargs.get("attempt_id"))
        await kwargs["on_delta"]("# 会议纪要\n\n## 5. 行动项\n\n发送方案。")
        return "# 会议纪要\n\n## 5. 行动项\n\n发送方案。"


class FakeTodoGenerator:
    calls = 0
    minutes: list[str] = []

    async def generate(self, meeting_id, summary_revision, minutes, **kwargs):
        type(self).calls += 1
        type(self).minutes.append(minutes)
        return TodoDocument(
            items=[TodoItem(task="发送方案", meeting_id=meeting_id, summary_revision=summary_revision)],
            meeting_id=meeting_id,
            summary_revision=summary_revision,
            generated_at="now",
        )


class BlockingTodoGenerator:
    started = asyncio.Event()
    release = asyncio.Event()

    async def generate(self, meeting_id, summary_revision, minutes, **kwargs):
        type(self).started.set()
        await type(self).release.wait()
        return TodoDocument(
            items=[TodoItem(task=f"旧版本 {summary_revision}")],
            meeting_id=meeting_id,
            summary_revision=summary_revision,
            generated_at="now",
        )


class FailingStreamingSummarizer:
    async def summarize(self, *args, **kwargs):
        await kwargs["on_delta"]("未完成的新纪要")
        raise RuntimeError("summary unavailable")


@pytest.mark.asyncio
async def test_summary_and_todo_are_independent_and_revisioned(settings) -> None:
    FakeSummarizer.calls = 0
    FakeSummarizer.attempts = []
    FakeTodoGenerator.calls = 0
    FakeTodoGenerator.minutes = []
    session = LiveMeetingSession(
        settings,
        FakeRuntime(),
        __import__("realtime_meeting.storage", fromlist=["LocalMeetingStore"]).LocalMeetingStore(settings.results_dir),
        title="测试会议",
        summarizer_factory=lambda _settings: FakeSummarizer(),
        todo_factory=lambda _settings: FakeTodoGenerator(),
    )
    await session.start()
    session.audio_writer = None
    speech = np.full(320, 1200, dtype=np.int16).tobytes()
    silence = np.zeros(320, dtype=np.int16).tobytes()
    for _ in range(16):
        await session.feed_audio(speech)
    for _ in range(10):
        await session.feed_audio(silence)
    await session.request_stop()
    assert session.stop_task is not None
    await session.stop_task
    if session.postprocess_task:
        await session.postprocess_task
    assert session.postprocess.state == "ready_for_summary"
    assert session.summary_state == "idle"
    assert await session.request_summary()
    if session.summary_task:
        await session.summary_task
    if session.todo_task:
        await session.todo_task

    assert session.recording_state == "complete"
    assert session.summary_state == "complete"
    assert session.todo_state == "complete"
    assert session.summary_revision == 1
    assert FakeSummarizer.calls == 1
    assert FakeTodoGenerator.calls == 1
    assert FakeTodoGenerator.minutes == [session.summary]
    assert FakeSummarizer.attempts[0] and FakeSummarizer.attempts[0] != session.id
    assert (session.output_dir / "todo_list.json").is_file()

    assert await session.request_todo()
    await session.todo_task
    assert FakeSummarizer.calls == 1
    assert FakeTodoGenerator.calls == 2


def test_recovery_promotes_saved_interrupted_recording_to_postprocess_queue(settings) -> None:
    store = LocalMeetingStore(settings.results_dir)
    meeting_id = "recovered-meeting"
    output = store.meeting_dir(meeting_id)
    output.mkdir(parents=True, exist_ok=True)
    (output / "transcript.jsonl").write_text(
        json.dumps(Utterance(1, 0, 1, 1, "zh", 0.9, "已保存内容", "已保存内容", "not_needed", "1:0").to_dict(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "session_state.json").write_text(
        json.dumps({"id": meeting_id, "title": "恢复会议", "recording_state": "recording", "summary_state": "idle", "todo_state": "waiting_summary"}, ensure_ascii=False),
        encoding="utf-8",
    )
    manager = SessionManager(settings, FakeRuntime(), store)
    recovered = manager.get(meeting_id)
    assert recovered is not None
    assert recovered.recording_state == "complete"
    assert recovered.summary_state == "idle"
    assert recovered.postprocess.state == "queued"
    assert recovered.postprocess.stages["asr_refine"]["state"] == "queued"
    assert recovered.ended_at


def test_recovery_requeues_running_postprocessing_tasks(settings) -> None:
    store = LocalMeetingStore(settings.results_dir)
    meeting_id = "recovered-tasks"
    output = store.meeting_dir(meeting_id)
    output.mkdir(parents=True, exist_ok=True)
    (output / "meeting_minutes.md").write_text("# 已保存纪要\n", encoding="utf-8")
    (output / "session_state.json").write_text(
        json.dumps({"id": meeting_id, "title": "恢复任务", "recording_state": "complete", "summary_state": "running", "todo_state": "running", "summary_revision": 1}, ensure_ascii=False),
        encoding="utf-8",
    )
    manager = SessionManager(settings, FakeRuntime(), store)
    recovered = manager.get(meeting_id)
    assert recovered is not None
    assert recovered.summary_state == "idle"
    assert recovered.todo_state == "waiting_summary"
    assert recovered.postprocess.state != "queued"


@pytest.mark.asyncio
async def test_recovery_resumes_todo_for_completed_summary(settings) -> None:
    store = LocalMeetingStore(settings.results_dir)
    meeting_id = "recovered-todo"
    output = store.meeting_dir(meeting_id)
    output.mkdir(parents=True, exist_ok=True)
    (output / "meeting_minutes.md").write_text("# 已完成纪要\n", encoding="utf-8")
    (output / "session_state.json").write_text(
        json.dumps({
            "id": meeting_id,
            "recording_state": "complete",
            "summary_state": "complete",
            "todo_state": "running",
            "summary_revision": 1,
            "postprocess": {
                "state": "running",
                "current_stage": "todo",
                "stages": {
                    "asr_refine": {"state": "complete"},
                    "diarization": {"state": "complete"},
                    "translation": {"state": "complete"},
                    "summary": {"state": "complete"},
                    "todo": {"state": "running"},
                },
            },
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    manager = SessionManager(settings, FakeRuntime(), store)
    recovered = manager.get(meeting_id)
    assert recovered is not None and recovered.todo_state == "queued"
    recovered.todo_factory = lambda _settings: FakeTodoGenerator()
    await manager.resume_pending()
    assert recovered.todo_task is not None
    assert recovered.postprocess_task is None
    await recovered.todo_task
    assert recovered.todo_state == "complete"


@pytest.mark.asyncio
async def test_model_dependent_recovery_waits_for_runtime_readiness(settings) -> None:
    store = LocalMeetingStore(settings.results_dir)
    meeting_id = "recover-postprocess"
    output = store.meeting_dir(meeting_id)
    output.mkdir(parents=True)
    from realtime_meeting.exporter import append_utterance

    append_utterance(output / "transcript.jsonl", Utterance(1, 0.0, 1.0, 1, "zh", 1.0, "恢复处理", segment_id="1:0"))
    (output / "session_state.json").write_text(json.dumps({
        "id": meeting_id,
        "title": "恢复后处理",
        "recording_state": "complete",
        "summary_state": "idle",
        "todo_state": "waiting_summary",
        "postprocess": {
            "state": "queued",
            "stages": {
                "asr_refine": {"state": "complete"},
                "diarization": {"state": "complete"},
                "translation": {"state": "complete"},
                "summary": {"state": "idle"},
                "todo": {"state": "idle"},
            },
        },
    }), encoding="utf-8")

    manager = SessionManager(settings, FakeRuntime(), store)
    recovered = manager.get(meeting_id)
    assert recovered is not None
    await manager.resume_pending(model_tasks_ready=False)
    assert recovered.postprocess_task is None
    await manager.resume_pending(model_tasks_ready=True)
    assert recovered.postprocess_task is not None
    await recovered.postprocess_task


@pytest.mark.asyncio
async def test_create_failure_releases_capacity_and_removes_directory(settings, monkeypatch) -> None:
    manager = SessionManager(settings, FakeRuntime(), LocalMeetingStore(settings.results_dir))

    async def fail_start(_session):
        raise RuntimeError("VAD initialization failed")

    monkeypatch.setattr(LiveMeetingSession, "start", fail_start)
    with pytest.raises(RuntimeError, match="VAD initialization failed"):
        await manager.create("启动失败")
    assert manager.sessions == {}
    assert list(settings.results_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_finalize_failure_marks_error_and_allows_delete(settings) -> None:
    manager = SessionManager(settings, FakeRuntime(), LocalMeetingStore(settings.results_dir))
    session = await manager.create("保存失败")

    class FailingWriter:
        def close(self):
            raise RuntimeError("FFmpeg close failed")

    session.audio_writer = FailingWriter()
    await session.request_stop()
    assert session.stop_task is not None
    await session.stop_task
    assert session.recording_state == "error"
    assert "FFmpeg close failed" in (session.error or "")
    assert await manager.delete(session.id) is True


@pytest.mark.asyncio
async def test_finalize_failure_before_queue_shutdown_cancels_worker(settings) -> None:
    manager = SessionManager(settings, FakeRuntime(), LocalMeetingStore(settings.results_dir))
    session = await manager.create("分段收尾失败")

    class FailingSegmenter:
        def flush(self):
            raise RuntimeError("segment flush failed")

    session.segmenter = FailingSegmenter()
    session.audio_writer = None
    await session.request_stop()
    assert session.stop_task is not None
    await session.stop_task
    assert session.recording_state == "error"
    assert session.worker_task is not None and session.worker_task.done()
    assert await manager.delete(session.id) is True


@pytest.mark.asyncio
async def test_unretained_audio_is_deleted_only_after_diarization(settings) -> None:
    settings.keep_audio = False
    settings.enable_refinement = False
    runtime = DiarizationRuntime()
    session = LiveMeetingSession(settings, runtime, LocalMeetingStore(settings.results_dir), title="不留录音")
    await session.start()
    audio_path = session.output_dir / "audio" / "audio-0001.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"test audio")

    class ExistingAudioWriter:
        def close(self):
            return [{"file": audio_path.name, "start_seconds": 0.0, "end_seconds": 1.0, "samples": 16000, "format": "wav"}]

    session.audio_writer = ExistingAudioWriter()
    from realtime_meeting.exporter import append_utterance

    append_utterance(
        session.transcript_path,
        Utterance(1, 0.0, 1.0, 1, "zh", 1.0, "测试说话人", segment_id="1:0"),
    )
    await session.request_stop()
    assert session.stop_task is not None
    await session.stop_task
    assert session.postprocess_task is not None
    await session.postprocess_task

    assert runtime.paths_seen == [audio_path]
    assert not audio_path.exists()
    assert session.audio_segments == []


@pytest.mark.asyncio
async def test_delete_rejects_completed_meeting_with_background_task(settings) -> None:
    manager = SessionManager(settings, FakeRuntime(), LocalMeetingStore(settings.results_dir))
    session = LiveMeetingSession(settings, FakeRuntime(), manager.store)
    session.recording_state = "complete"
    session.todo_task = asyncio.create_task(asyncio.sleep(60))
    manager.sessions[session.id] = session
    try:
        with pytest.raises(ValueError, match="后台任务"):
            await manager.delete(session.id)
        assert session.output_dir.exists()
    finally:
        session.todo_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await session.todo_task


@pytest.mark.asyncio
async def test_failed_summary_retry_preserves_previous_committed_summary(settings) -> None:
    session = LiveMeetingSession(
        settings,
        FakeRuntime(),
        LocalMeetingStore(settings.results_dir),
        summarizer_factory=lambda _settings: FailingStreamingSummarizer(),
    )
    session.recording_state = "complete"
    for stage in ("asr_refine", "diarization", "translation"):
        session.postprocess.update(stage, "complete")
    session.summary = "# 已提交旧纪要"
    session.summary_revision = 1
    session.summary_state = "complete"
    (session.output_dir / "meeting_minutes.md").write_text(session.summary + "\n", encoding="utf-8")

    assert await session.request_summary()
    assert session.summary_task is not None
    await session.summary_task

    assert session.summary_state == "error"
    assert session.summary == "# 已提交旧纪要"
    assert session.summary_revision == 1
    assert (session.output_dir / "meeting_minutes.md").read_text(encoding="utf-8") == "# 已提交旧纪要\n"


def test_refinement_event_survives_restart(settings) -> None:
    store = LocalMeetingStore(settings.results_dir)
    first = LiveMeetingSession(settings, FakeRuntime(), store, title="精修恢复")
    event = SegmentEvent("final", np.full(640, 1000, dtype=np.int16).tobytes(), 1.0, 1.04, 7)
    first._persist_refinement_event(event)

    restored = LiveMeetingSession(settings, FakeRuntime(), store, meeting_id=first.id, recovered_state={"recording_state": "complete"})
    assert restored._refinement_events[7].pcm == event.pcm
    assert restored._refinement_events[7].start == 1.0


def test_refinement_events_can_be_rebuilt_from_saved_audio(settings) -> None:
    store = LocalMeetingStore(settings.results_dir)
    session = LiveMeetingSession(settings, FakeRuntime(), store, title="录音重建")
    audio_dir = session.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / "audio-0001.wav"
    pcm = np.full(SAMPLE_RATE, 1000, dtype=np.int16).tobytes()
    with wave.open(str(audio_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm)
    session.audio_segments = [{"file": audio_path.name, "samples": SAMPLE_RATE}]
    from realtime_meeting.exporter import append_utterance
    append_utterance(session.transcript_path, Utterance(1, 0.1, 0.4, 1, "zh", 1.0, "恢复", segment_id="1:0", source_segment_id="1"))

    assert session._rebuild_refinement_events_from_audio() == 1
    assert len(session._refinement_events[1].pcm) == round(0.3 * SAMPLE_RATE) * 2


@pytest.mark.asyncio
async def test_refinement_queue_overflow_is_processed_after_stop(settings) -> None:
    settings.refinement_queue_size = 1
    session = LiveMeetingSession(settings, FakeRuntime(), LocalMeetingStore(settings.results_dir))
    pcm = np.full(640, 1000, dtype=np.int16).tobytes()
    await session._handle_final(SegmentEvent("final", pcm, 0.0, 0.04, 1))
    await session._handle_final(SegmentEvent("final", pcm, 0.1, 0.14, 2))
    assert session.refinement_queue.qsize() == 1
    assert len(session._refinement_events) == 2
    session.recording_state = "complete"
    session._set_postprocess_queue(True)
    await session.run_postprocess()
    refined_sources = {
        item.source_segment_id for item in session.recent if item.recognition_stage == "refined"
    }
    assert refined_sources == {"1", "2"}
    assert session.postprocess.state == "ready_for_summary"


@pytest.mark.asyncio
async def test_refinement_recovery_fails_without_persisted_or_saved_audio(settings) -> None:
    session = LiveMeetingSession(settings, FakeRuntime(), LocalMeetingStore(settings.results_dir))
    from realtime_meeting.exporter import append_utterance
    append_utterance(session.transcript_path, Utterance(1, 0.0, 0.5, 1, "zh", 1.0, "仅有文本", segment_id="1:0", source_segment_id="1"))
    session.recording_state = "complete"
    session._set_postprocess_queue(True)
    await session.run_postprocess()
    assert session.postprocess.state == "error"
    assert session.postprocess.stages["asr_refine"]["state"] == "error"


@pytest.mark.asyncio
async def test_new_summary_cancels_old_todo_task(settings) -> None:
    BlockingTodoGenerator.started = asyncio.Event()
    BlockingTodoGenerator.release = asyncio.Event()
    session = LiveMeetingSession(
        settings,
        FakeRuntime(),
        LocalMeetingStore(settings.results_dir),
        title="版本竞态",
        summarizer_factory=lambda _settings: FakeSummarizer(),
        todo_factory=lambda _settings: BlockingTodoGenerator(),
    )
    session.recording_state = "complete"
    session.postprocess.update("asr_refine", "complete")
    session.postprocess.update("diarization", "complete")
    session.postprocess.update("translation", "complete")
    session.summary = "旧纪要"
    session.summary_revision = 1
    session.summary_state = "complete"
    session.todo_state = "queued"
    session.todo_task = asyncio.create_task(session.run_todo())
    await BlockingTodoGenerator.started.wait()

    assert await session.request_summary() is True
    assert session.todo_task is None
    assert session.summary_task is not None
    await session.summary_task
    if session.todo_task:
        session.todo_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await session.todo_task

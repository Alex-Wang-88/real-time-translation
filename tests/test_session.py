from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from realtime_meeting.models import TodoDocument, TodoItem, Utterance
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


def test_recovery_promotes_saved_interrupted_recording_to_summary_queue(settings) -> None:
    store = LocalMeetingStore(settings.results_dir)
    meeting_id = "recovered-meeting"
    output = store.meeting_dir(meeting_id)
    output.mkdir(parents=True, exist_ok=True)
    (output / "transcript.jsonl").write_text(
        __import__("json").dumps(Utterance(1, 0, 1, 1, "zh", 0.9, "已保存内容", "已保存内容", "not_needed", "1:0").to_dict(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "session_state.json").write_text(
        __import__("json").dumps({"id": meeting_id, "title": "恢复会议", "recording_state": "recording", "summary_state": "idle", "todo_state": "waiting_summary"}, ensure_ascii=False),
        encoding="utf-8",
    )
    manager = SessionManager(settings, FakeRuntime(), store)
    recovered = manager.get(meeting_id)
    assert recovered is not None
    assert recovered.recording_state == "complete"
    assert recovered.summary_state == "queued"
    assert recovered.ended_at


def test_recovery_requeues_running_postprocessing_tasks(settings) -> None:
    store = LocalMeetingStore(settings.results_dir)
    meeting_id = "recovered-tasks"
    output = store.meeting_dir(meeting_id)
    output.mkdir(parents=True, exist_ok=True)
    (output / "meeting_minutes.md").write_text("# 已保存纪要\n", encoding="utf-8")
    (output / "session_state.json").write_text(
        __import__("json").dumps({"id": meeting_id, "title": "恢复任务", "recording_state": "complete", "summary_state": "running", "todo_state": "running", "summary_revision": 1}, ensure_ascii=False),
        encoding="utf-8",
    )
    manager = SessionManager(settings, FakeRuntime(), store)
    recovered = manager.get(meeting_id)
    assert recovered is not None
    assert recovered.summary_state == "queued"
    assert recovered.todo_state == "queued"

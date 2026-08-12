from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from realtime_meeting.audio import SAMPLE_RATE, SAMPLE_WIDTH, SegmentEvent
from realtime_meeting.config import Settings
from realtime_meeting.jimo import JimoClient, MeetingSummarizer, TodoGenerator
from realtime_meeting.models import Utterance
from realtime_meeting.runtime import PartialResult, TranslationResult
from realtime_meeting.session import LiveMeetingSession, SessionManager
from realtime_meeting.storage import LocalMeetingStore
from realtime_meeting.exporter import load_utterances


def make_loud_packet(seconds: float = 0.04) -> bytes:
    import struct

    samples = int(SAMPLE_RATE * seconds)
    values = [30000 if (i // 20) % 2 == 0 else -30000 for i in range(samples)]
    return struct.pack(f"<{samples}h", *values)


def make_silence_packet(seconds: float = 0.04) -> bytes:
    import struct

    samples = int(SAMPLE_RATE * seconds)
    return struct.pack(f"<{samples}h", *([0] * samples))


class FakeRuntime:
    ready = True
    device = "cpu"
    status = "ok"
    metrics = {}

    def new_vad(self):
        return None

    def new_speaker_clusterer(self):
        return None

    def transcribe_partial(self, event, recent_text="", hotwords=None):
        return PartialResult(event.revision, event.start, event.end, "partial draft", "en", 0.7)

    def transcribe_final(self, event, *, next_id, previous_language=None, recent_text="", hotwords=None, speaker_clusterer=None, refined=False):
        text = "Hello team this is the English test meeting content"
        return [
            Utterance(
                next_id,
                event.start,
                event.end,
                1,
                "en",
                0.9,
                text,
                "",
                "pending",
                f"{event.revision}:0",
                2 if refined else 1,
                "refined" if refined else "fast",
            )
        ]

    def translate_text(self, text, source_language):
        return TranslationResult(f"中文翻译：{text}", "ready")


class FakeJimoClient(JimoClient):
    def __init__(self, settings, *, endpoint=None, client=None):
        self.settings = settings

    @staticmethod
    def request_chars(messages):
        return sum(len(str(m.get("content", ""))) for m in messages)

    async def complete(self, messages, session_id, *, on_delta=None, on_reset=None):
        blob = "\n".join(str(m.get("content", "")) for m in messages)
        if "MODE=FINAL" in blob:
            content = "# 会议纪要\n\n## 1. 会议主题\n测试会议\n\n## 2. 核心结论\n完成集成测试\n"
        elif "请只输出符合要求的 JSON" in blob:
            content = json.dumps(
                {
                    "schema_version": "1.0",
                    "items": [
                        {
                            "task": "完成集成测试任务",
                            "owner": "张三",
                            "due_date": "2026-08-20",
                            "priority": "高",
                            "status": "未开始",
                            "evidence": "测试依据",
                        }
                    ],
                }
            )
        else:
            content = "会议状态已更新"
        if on_delta:
            result = on_delta(content)
            if asyncio.iscoroutine(result):
                await result
        return content


def make_summarizer(settings: Settings) -> MeetingSummarizer:
    return MeetingSummarizer(settings, client=FakeJimoClient(settings))


def make_todo(settings: Settings) -> TodoGenerator:
    return TodoGenerator(settings, client=FakeJimoClient(settings))


@pytest.fixture
def session(tmp_path: Path) -> LiveMeetingSession:
    settings = Settings(
        results_dir=tmp_path / "meetings",
        translation_model_root=tmp_path / "models",
        jimo_api_url="https://summary.example.test",
        jimo_todo_api_url="https://todo.example.test",
        jimo_authorization="Bearer secret",
        enable_refinement=True,
    )
    store = LocalMeetingStore(settings.results_dir)
    return LiveMeetingSession(
        settings,
        FakeRuntime(),
        store,
        title="集成测试会议",
        summarizer_factory=make_summarizer,
        todo_factory=make_todo,
    )


async def drive_audio(meeting: LiveMeetingSession, packets: int = 80, silence: int = 30) -> None:
    for _ in range(packets):
        await meeting.feed_audio(make_loud_packet())
    for _ in range(silence):
        await meeting.feed_audio(make_silence_packet())


async def wait_until(meeting: LiveMeetingSession, predicate, timeout: float = 10.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate() and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.02)
    if not predicate():
        raise AssertionError("wait_until timed out")


async def test_full_meeting_lifecycle(session: LiveMeetingSession) -> None:
    await session.start()
    assert session.recording_state == "recording"
    await drive_audio(session)
    await session.request_stop("user")
    await wait_until(session, lambda: session.recording_state == "complete")
    await wait_until(session, lambda: session.summary_state == "complete", timeout=15)
    await wait_until(session, lambda: session.todo_state == "complete", timeout=15)

    assert session.recording_state == "complete"
    assert session.summary_state == "complete"
    assert session.todo_state == "complete"
    assert session.summary.strip()
    assert session.todo is not None and session.todo.items

    files = session.files
    expected = {
        "meeting_transcript.md",
        "translated_zh.md",
        "transcript.json",
        "transcript.jsonl",
        "meeting_minutes.md",
        "todo_list.json",
        "todo_list.md",
        "original_en.md",
    }
    missing = expected - set(files)
    assert not missing, f"缺少导出文件: {missing}"

    transcript_path = session.output_dir / "transcript.jsonl"
    utterances = transcript_path.read_text(encoding="utf-8").strip().splitlines()
    assert utterances, "transcript.jsonl 应当有内容"
    first = json.loads(utterances[0])
    assert first["translation_zh"], "英文 utterance 应当有中文翻译"


async def test_meeting_without_speech_has_no_summary(session: LiveMeetingSession) -> None:
    await session.start()
    for _ in range(20):
        await session.feed_audio(make_silence_packet())
    await session.request_stop("user")
    await wait_until(session, lambda: session.recording_state == "complete")
    await wait_until(session, lambda: session.summary_state == "error", timeout=5)
    assert session.summary_state == "error"
    assert session.summary_error


async def test_recover_interrupted_meeting(tmp_path: Path) -> None:
    settings = Settings(
        results_dir=tmp_path / "meetings",
        translation_model_root=tmp_path / "models",
        jimo_api_url="https://summary.example.test",
        jimo_todo_api_url="https://todo.example.test",
        jimo_authorization="Bearer secret",
    )
    store = LocalMeetingStore(settings.results_dir)
    first = LiveMeetingSession(
        settings, FakeRuntime(), store, title="恢复测试",
        summarizer_factory=make_summarizer, todo_factory=make_todo,
    )
    await first.start()
    await drive_audio(first)
    await first.request_stop("user")
    await wait_until(first, lambda: first.recording_state == "complete")
    await wait_until(first, lambda: first.summary_state == "complete", timeout=15)
    await wait_until(first, lambda: first.todo_state == "complete", timeout=15)

    manager = SessionManager(settings, FakeRuntime(), store)
    recovered = manager.get(first.id)
    assert recovered is not None
    assert recovered.recording_state == "complete"
    assert recovered.summary_state == "complete"
    assert recovered.todo_state == "complete"
    assert recovered.todo is not None and recovered.todo.items


class UnsupportedTranslationRuntime(FakeRuntime):
    def translate_text(self, text, source_language):
        from realtime_meeting.runtime import TranslationResult

        return TranslationResult(text, "unsupported")


class RefinementErrorRuntime(FakeRuntime):
    def transcribe_final(self, event, **kwargs):
        if kwargs.get("refined"):
            raise RuntimeError("large-v3 unavailable in test")
        return super().transcribe_final(event, **kwargs)


async def test_refinement_error_keeps_recording_saved_and_marks_retryable_stage(session: LiveMeetingSession) -> None:
    session.runtime = RefinementErrorRuntime()
    await session.start()
    await drive_audio(session)
    await session.request_stop("user")
    await wait_until(session, lambda: session.recording_state == "complete")
    await wait_until(session, lambda: session.postprocess.state == "error", timeout=10)
    if session.postprocess_task:
        await session.postprocess_task

    assert session.recording_state == "complete"
    assert session.postprocess.stages["asr_refine"]["state"] == "error"
    assert load_utterances(session.transcript_path)


async def test_unsupported_translation_leaves_translation_empty(session: LiveMeetingSession) -> None:
    session.runtime = UnsupportedTranslationRuntime()
    await session.start()
    await drive_audio(session)
    await session.request_stop("user")
    await wait_until(session, lambda: session.recording_state == "complete")
    path = session.output_dir / "transcript.jsonl"
    await wait_until(
        session,
        lambda: all(
            item.translation_status != "pending"
            for item in load_utterances(path)
            if item.language != "zh"
        ),
    )
    utterances = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert utterances, "应当有转写内容"
    for item in utterances:
        if item["language"] != "zh":
            assert item["translation_status"] == "unsupported"
            assert item["translation_zh"] == "", "翻译不可用时不应把原文当作中文翻译"


def test_load_utterances_dedup(tmp_path: Path) -> None:
    from realtime_meeting.exporter import append_utterance, load_utterances

    path = tmp_path / "t.jsonl"
    # append a fast then refined version of the same segment
    append_utterance(path, Utterance(1, 0.0, 1.0, 1, "en", 0.9, "fast text", "", "pending", "rev:0", 1, "fast"))
    items = load_utterances(path)
    assert len(items) == 1
    append_utterance(path, Utterance(1, 0.0, 1.0, 1, "en", 0.95, "refined text", "中文", "ready", "rev:0", 2, "refined"))
    items = load_utterances(path)
    assert len(items) == 1, "同一 segment 不应产生重复条目"
    assert items[0].text == "refined text"
    assert items[0].translation_zh == "中文"

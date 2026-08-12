from __future__ import annotations

import asyncio
import json

import pytest

from realtime_meeting.audio import SAMPLE_RATE, StreamSegmenter
from realtime_meeting.config import Settings
from realtime_meeting.exporter import append_utterance
from realtime_meeting.jimo import (
    JimoClient,
    MeetingSummarizer,
    TodoGenerator,
    _event_content,
    parse_sse_lines,
    parse_todo_document,
    transcript_chunks,
)
from realtime_meeting.language import MultilingualDetector
from realtime_meeting.models import Utterance


def test_event_content_plain_text() -> None:
    class Ev:
        event = "message"
        data = "会议纪要的第一段内容"

    content, ended = _event_content(Ev())
    assert content == "会议纪要的第一段内容", "纯文本 SSE 应当被当作内容返回"


def test_event_content_json_delta() -> None:
    class Ev:
        event = "message"
        data = json.dumps({"choices": [{"delta": {"content": "你好"}}]})

    content, ended = _event_content(Ev())
    assert content == "你好"


def test_event_content_done_marker() -> None:
    class Ev:
        event = "message"
        data = "[DONE]"

    content, ended = _event_content(Ev())
    assert ended is True


def test_event_content_event_done() -> None:
    class Ev:
        event = "done"
        data = "ignored"

    content, ended = _event_content(Ev())
    assert ended is True


def test_parse_todo_prose_prefixed_json() -> None:
    raw = '好的，这是提取的行动项：\n```json\n{"schema_version":"1.0","items":[{"task":"完成报告","owner":"李四"}]}\n```'
    doc = parse_todo_document(raw, "m1", 1)
    assert doc.items
    assert doc.items[0].task == "完成报告"


def test_parse_todo_bare_json() -> None:
    raw = json.dumps({"schema_version": "1.0", "items": [{"task": "x"}]})
    doc = parse_todo_document(raw, "m1", 1)
    assert doc.items and doc.items[0].task == "x"


def test_parse_todo_empty_items() -> None:
    raw = json.dumps({"schema_version": "1.0", "items": []})
    doc = parse_todo_document(raw, "m1", 1)
    assert doc.items == []


def test_language_detection() -> None:
    detector = MultilingualDetector()
    assert detector.detect("这是一段中文会议内容，大家讨论一下方案。").code == "zh"
    assert detector.detect("Hello everyone, let us start the meeting now").code == "en"
    assert detector.detect("Guten Morgen, wir müssen das Projekt besprechen").code == "de"


def test_segmenter_emits_final_after_silence() -> None:
    segmenter = StreamSegmenter(minimum_rms=10.0)
    import struct

    frames = []
    # 2 seconds loud
    for _ in range(int(SAMPLE_RATE * 2 / 320)):
        pcm = struct.pack(f"<{160}h", *([20000] * 160))
        frames.extend(segmenter.feed(pcm))
    # 1 second silence
    for _ in range(int(SAMPLE_RATE * 1 / 320)):
        pcm = struct.pack(f"<{160}h", *([0] * 160))
        frames.extend(segmenter.feed(pcm))
    finals = [f for f in frames if f.kind == "final"]
    assert finals, "静音后应当产生 final 事件"


def test_transcript_chunks_basic(tmp_path) -> None:
    path = tmp_path / "t.jsonl"
    for i, text in enumerate(["a", "b", "c"]):
        u = Utterance(i + 1, float(i), float(i) + 1.0, 1, "en", 0.9, text, f"中{text}", "ready", f"{i}:0")
        from realtime_meeting.exporter import append_utterance

        append_utterance(path, u)
    chunks = list(transcript_chunks(path, 1000))
    assert chunks
    _, _, _, text = chunks[0]
    assert "a" in text


def test_summarizer_strips_markdown_fence(tmp_path) -> None:
    settings = Settings(
        results_dir=tmp_path / "meetings",
        translation_model_root=tmp_path / "models",
        jimo_api_url="https://summary.example.test",
        jimo_todo_api_url="https://todo.example.test",
        jimo_authorization="Bearer secret",
    )
    path = tmp_path / "t.jsonl"
    append = __import__("realtime_meeting.exporter", fromlist=["append_utterance"]).append_utterance
    append(path, Utterance(1, 0.0, 1.0, 1, "en", 0.9, "hello", "你好", "ready", "1:0"))

    class FenceJimo(JimoClient):
        def __init__(self, settings, *, endpoint=None, client=None):
            self.settings = settings

        async def complete(self, messages, session_id, *, on_delta=None, on_reset=None):
            blob = "\n".join(str(m.get("content", "")) for m in messages)
            if "MODE=FINAL" in blob:
                content = "```markdown\n# 会议纪要\n\n## 1. 会议主题\n测试\n```"
            else:
                content = "状态更新"
            if on_delta:
                result = on_delta(content)
                if asyncio.iscoroutine(result):
                    await result
            return content

    summarizer = MeetingSummarizer(settings, client=FenceJimo(settings))
    result = asyncio.get_event_loop().run_until_complete(
        summarizer.summarize(path, "m1", "2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z")
    )
    assert "```" not in result, "会议纪要不应包含 markdown 代码围栏"


def test_parse_sse_lines_plain_text() -> None:
    lines = [
        "data: 第一段内容",
        "",
        "data: 第二段内容",
        "",
        "event: done",
        "data: [DONE]",
        "",
    ]
    events = list(parse_sse_lines(lines))
    contents = [e.data for e in events if e.event == "message" and e.data]
    assert "第一段内容" in contents
    assert "第二段内容" in contents
    assert any(e.event == "done" for e in events)


# --- Server edge cases -------------------------------------------------


class ReadyRuntime:
    ready = True
    device = "cpu"
    status = "模型已就绪"
    metrics = {}

    def new_vad(self):
        return None

    def new_speaker_clusterer(self):
        return None


def test_delete_active_meeting_returns_409() -> None:
    from pathlib import Path

    from fastapi.testclient import TestClient
    from realtime_meeting.server import create_app
    from realtime_meeting.storage import LocalMeetingStore

    import tempfile

    settings = Settings(
        host="127.0.0.1",
        results_dir=Path(tempfile.mkdtemp()),
        translation_model_root=Path(tempfile.mkdtemp()),
        jimo_authorization="x",
    )
    app = create_app(settings, ReadyRuntime(), load_models=False, store=LocalMeetingStore(settings.results_dir))
    with TestClient(app) as client:
        created = client.post("/api/v2/meetings", json={"title": "进行中"})
        assert created.status_code == 201
        meeting_id = created.json()["id"]
        deleted = client.delete(f"/api/v2/meetings/{meeting_id}")
        assert deleted.status_code == 409, "进行中的会议不能被删除"


def test_download_missing_file_returns_404() -> None:
    from pathlib import Path

    from fastapi.testclient import TestClient
    from realtime_meeting.server import create_app
    from realtime_meeting.storage import LocalMeetingStore

    import tempfile

    settings = Settings(
        host="127.0.0.1",
        results_dir=Path(tempfile.mkdtemp()),
        translation_model_root=Path(tempfile.mkdtemp()),
        jimo_authorization="x",
    )
    app = create_app(settings, ReadyRuntime(), load_models=False, store=LocalMeetingStore(settings.results_dir))
    with TestClient(app) as client:
        created = client.post("/api/v2/meetings", json={"title": "空会议"})
        assert created.status_code == 201
        meeting_id = created.json()["id"]
        response = client.get(f"/api/v2/meetings/{meeting_id}/files/meeting_minutes.md")
        assert response.status_code == 404

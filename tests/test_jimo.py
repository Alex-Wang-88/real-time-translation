from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from realtime_meeting.jimo import (
    JimoClient,
    MeetingSummarizer,
    TodoGenerator,
    parse_sse_lines,
    parse_todo_document,
    transcript_chunks,
)
from realtime_meeting.models import TodoDocument, Utterance


def test_sse_parser_supports_multiline_data_and_end_event() -> None:
    events = list(parse_sse_lines([
        ": heartbeat\n",
        "event: data\n",
        'data: {"content":"第一段"}\n',
        "data: {\"content\":\"第二段\"}\n",
        "\n",
        "event: end\n",
        "data: {}\n",
        "\n",
    ]))
    assert events[0].event == "data"
    assert events[0].data.count("\n") == 1
    assert events[1].event == "end"


@pytest.mark.asyncio
async def test_jimo_request_keeps_legacy_body_and_streams_content(settings) -> None:
    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        seen["payload"] = json.loads(request.content)
        stream = (
            'event: data\n'
            'data: {"content":"中"}\n'
            'data: {"content":"文"}\n\n'
            'data: {"choices":[{"delta":{"content":"纪要"}}]}\n\n'
            'data: [DONE]\n\n'
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream.encode())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = JimoClient(settings, client=http_client)
        deltas: list[str] = []
        result = await client.complete(
            [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
            "meeting:m:summary:a1",
            on_delta=deltas.append,
        )

    payload = seen["payload"]
    assert result == "中文纪要"
    assert deltas == ["中文", "纪要"]
    assert payload == {
        "messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
        "sessionId": "meeting:m:summary:a1",
        "source": "api",
        "extra": {},
    }
    assert "test-secret" in seen["headers"]["authorization"]


def test_transcript_chunks_keep_utterance_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    items = [
        Utterance(1, 0, 2, 1, "en", 0.9, "We will ship the first plan.", "我们会发布第一个计划。", "ready", "s1"),
        Utterance(2, 2, 4, 1, "de", 0.9, "Wir prüfen die Risiken.", "我们检查风险。", "ready", "s2"),
        Utterance(3, 4, 6, 1, "zh", 0.9, "下周确认负责人。", "下周确认负责人。", "not_needed", "s3"),
    ]
    path.write_text("\n".join(json.dumps(item.to_dict(), ensure_ascii=False) for item in items) + "\n", encoding="utf-8")
    chunks = list(transcript_chunks(path, 80))
    assert len(chunks) == 3
    assert [chunk[0] for chunk in chunks] == [1, 2, 3]
    assert all("[" in chunk[3] and chunk[2] >= chunk[1] for chunk in chunks)
    assert "We will ship" in "\n".join(chunk[3] for chunk in chunks)


class _FakeTodoClient:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], str]] = []

    async def complete(self, messages, session_id, **_kwargs):
        self.calls.append((messages, session_id))
        return """```json
        {"schema_version":"1.0","items":[{"task":"发送方案","owner":"李雷","due_date":"2026-08-20","source_time_start":12.5,"source_time_end":14,"evidence":"会议决定由李雷发送方案"}]}
        ```"""


@pytest.mark.asyncio
async def test_todo_generator_makes_one_logical_model_call(settings) -> None:
    fake = _FakeTodoClient()
    generator = TodoGenerator(settings, client=fake)
    result = await generator.generate("meeting-1", 3, "# 会议纪要\n\n李雷发送方案，截止 2026-08-20。")
    assert len(fake.calls) == 1
    assert fake.calls[0][1] == "meeting:meeting-1:todo:3"
    assert fake.calls[0][0][0]["role"] == "system"
    assert result.items[0].task == "发送方案"
    assert result.items[0].owner == "李雷"
    assert result.items[0].source_time_start == 12.5


def test_todo_json_invalid_schema_is_a_failure() -> None:
    with pytest.raises(ValueError):
        parse_todo_document('{"items":[{"task": 3}]}', "meeting-1", 1)

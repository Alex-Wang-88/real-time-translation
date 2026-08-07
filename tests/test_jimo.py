from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from realtime_meeting.config import Settings
from realtime_meeting.exporter import append_utterance
from realtime_meeting.jimo import JimoClient, MeetingSummarizer, parse_sse_lines, transcript_chunks
from realtime_meeting.models import Utterance


def settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "results_dir": tmp_path,
        "jimo_api_url": "https://example.test/v2/chat/completions/share?shareId=test",
        "jimo_authorization": "opaque-authorization-value",
        "jimo_max_request_chars": 12_000,
        "jimo_transcript_chars": 1_200,
        "jimo_state_chars": 1_000,
    }
    values.update(overrides)
    return Settings(**values)


def test_sse_parser_supports_documented_field_order_and_end_event():
    events = list(
        parse_sse_lines(
            [
                'data: {"role":"assistant","content":"第一段"}',
                "event: data",
                "",
                "data: {'end': {}, 'role': 'assistant'}",
                "event: end",
                "",
            ]
        )
    )
    assert [(event.event, event.data) for event in events] == [
        ("data", '{"role":"assistant","content":"第一段"}'),
        ("end", "{'end': {}, 'role': 'assistant'}"),
    ]


@pytest.mark.asyncio
async def test_jimo_client_uses_raw_authorization_and_parses_sse(tmp_path: Path):
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        body = (
            'data: {"role":"assistant","content":"会议"}\n'
            "event: data\n\n"
            'data: {"role":"assistant","content":"纪要"}\n'
            "event: data\n\n"
            "data: {'end': {}, 'role': 'assistant'}\n"
            "event: end\n\n"
        )
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = JimoClient(settings(tmp_path), http_client)
        deltas = []
        result = await client.complete(
            [{"role": "user", "content": "test"}], "same-session", on_delta=deltas.append
        )
    assert result == "会议纪要"
    assert deltas == ["会议", "纪要"]
    assert captured["authorization"] == "opaque-authorization-value"
    assert captured["body"]["sessionId"] == "same-session"
    assert captured["body"]["source"] == "api"


def test_transcript_chunks_do_not_split_an_utterance(tmp_path: Path):
    path = tmp_path / "transcript.jsonl"
    for index in range(1, 5):
        append_utterance(
            path,
            Utterance(index, index, index + 1, 1, "de", 0.9, "Danke " * 20, "谢谢 " * 20),
        )
    chunks = list(transcript_chunks(path, 500))
    assert len(chunks) >= 2
    assert all("演讲人1（德文）" in chunk[3] for chunk in chunks)
    assert all("演讲人1（中文翻译）" in chunk[3] for chunk in chunks)


@pytest.mark.asyncio
async def test_meeting_summarizer_reuses_session_and_marks_start_end(tmp_path: Path):
    path = tmp_path / "transcript.jsonl"
    for index in range(1, 8):
        append_utterance(
            path,
            Utterance(index, index * 2, index * 2 + 1, 1, "zh", 0.9, "项目进展正常。" * 12, "项目进展正常。" * 8),
        )

    class FakeClient:
        def __init__(self):
            self.calls = []

        async def complete(self, messages, session_id, *, on_delta=None, on_reset=None):
            self.calls.append((messages, session_id))
            final = "MEETING_END" in messages[-1]["content"]
            output = "1. 会议主题\n项目进展" if final else "主题：项目进展；无新增风险。"
            if final and on_delta:
                for part in ("1. 会议主题\n", "项目进展"):
                    result = on_delta(part)
                    if hasattr(result, "__await__"):
                        await result
            return output

    fake = FakeClient()
    summarizer = MeetingSummarizer(settings(tmp_path), fake)
    statuses = []
    deltas = []
    result = await summarizer.summarize(
        path,
        "one-session",
        "2026-01-01T00:00:00Z",
        "2026-01-01T01:00:00Z",
        on_status=lambda kind, index, total: statuses.append((kind, index, total)),
        on_delta=deltas.append,
        on_reset=lambda: deltas.clear(),
    )
    assert result.startswith("1. 会议主题")
    assert all(session_id == "one-session" for _messages, session_id in fake.calls)
    assert "MEETING_START" in fake.calls[0][0][-1]["content"]
    assert "MEETING_END" in fake.calls[-1][0][-1]["content"]
    assert all(
        sum(len(message["content"]) for message in messages) <= 12_000
        for messages, _session in fake.calls
    )
    assert statuses[-1][0] == "final"
    assert "".join(deltas) == result

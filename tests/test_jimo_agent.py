from __future__ import annotations

import json

import httpx
import pytest

from realtime_meeting.jimo import JimoClient, MeetingAgent, parse_meeting_agent_result
from realtime_meeting.models import Utterance


def _payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "meta": {"title": "测试会议", "languages": ["zh", "en"], "duration_seconds": 4.2},
        "transcript": [
            {
                "index": 1,
                "speaker": "说话人1",
                "speaker_name": None,
                "language": "en",
                "speech_variant": None,
                "start": 0.0,
                "end": 2.1,
                "original": "We will confirm the date today.",
                "translation_zh": "我们今天确认日期。",
                "uncertain": False,
                "uncertainty_note": "",
            },
        ],
        "minutes": {
            "topic": "日期确认",
            "core_conclusions": [{"text": "今天确认日期。", "source_indices": [1]}],
            "discussion_points": [],
            "decisions": [],
            "risks_and_blockers": [],
            "open_questions": [],
        },
        "todo": [
            {
                "task": "确认日期",
                "owner": "说话人1",
                "due_date": None,
                "priority": "高",
                "status": "未开始",
                "source_time_start": 0.0,
                "source_time_end": 2.1,
                "evidence": "We will confirm the date today.",
                "notes": "",
                "source_indices": [1],
            },
        ],
    }


def test_parse_meeting_agent_result_normalizes_transcript_minutes_and_todo() -> None:
    result = parse_meeting_agent_result(json.dumps(_payload()), "meeting-1", 3)

    assert result.transcript[0]["original"].startswith("We will")
    assert result.transcript[0]["translation_zh"] == "我们今天确认日期。"
    assert "# 会议纪要" in result.summary_markdown
    assert result.todo.items[0].task == "确认日期"
    assert result.todo.items[0].summary_revision == 3


def test_parse_marked_agent_sections_into_three_result_parts() -> None:
    raw = """@@JIMO_SECTION:DATA:BEGIN@@
# 完整逐句转写（精修）

## 逐句转写

### [S001] 时间：00:00:00.000 - 00:00:02.000

- 说话人：说话人1
- 语言：英语
- 原文：We will confirm the date today.
- 中文翻译：我们今天确认日期。
- 是否存疑：否
- 存疑说明：无

### [S002] 时间：00:00:02.000 - 00:00:04.000

- 说话人：说话人2
- 语言：中文
- 原文：好的。
- 中文翻译：好的。
- 是否存疑：否
- 存疑说明：无
@@JIMO_SECTION:DATA:END@@
@@JIMO_SECTION:SUMMARY:BEGIN@@
# 会议纪要

## 会议主题

日期确认。
@@JIMO_SECTION:SUMMARY:END@@
@@JIMO_SECTION:TODOLIST:BEGIN@@
# 待办事项

## 已确认待办

### T001

- 任务：确认日期
- 负责人：说话人1
- 截止时间：今天
- 优先级：高
- 当前状态：未开始
- 原文依据：S001
- 时间范围：00:00:00.000 - 00:00:02.000
- 事实说明：会议中明确提出今天确认日期。
@@JIMO_SECTION:TODOLIST:END@@"""

    result = parse_meeting_agent_result(raw, "meeting-1", 4)

    assert len(result.transcript) == 2
    assert result.transcript[0]["language"] == "en"
    assert result.transcript[0]["original"].startswith("We will")
    assert "日期确认" in result.summary_markdown
    assert result.todo.items[0].task == "确认日期"
    assert "@@JIMO_SECTION:DATA:BEGIN@@" not in result.transcript_markdown
    assert result.todo_markdown.startswith("# 待办事项")


@pytest.mark.asyncio
async def test_jimo_client_reads_complete_json_and_file_message(settings) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Accept"] == "application/json"
        body = json.loads((await request.aread()).decode("utf-8"))
        content = body["messages"][1]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["file_url"]["fileId"] == "file-1"
        return httpx.Response(
            200,
            json={"content": json.dumps(_payload(), ensure_ascii=False)},
            request=request,
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = JimoClient(settings, client=http_client)
        agent = MeetingAgent(settings, client=client)
        result = await agent.process(
            "meeting-1",
            summary_revision=1,
            audio_files=[{"url": "https://example.test/audio.wav", "file_id": "file-1"}],
        )

    assert len(requests) == 1
    assert result.todo.items[0].task == "确认日期"


@pytest.mark.asyncio
async def test_meeting_agent_falls_back_to_local_transcript_without_audio(settings) -> None:
    class CapturingClient:
        def __init__(self) -> None:
            self.messages = None

        async def complete(self, messages, _session_id, **_kwargs):
            self.messages = messages
            return json.dumps(_payload(), ensure_ascii=False)

    client = CapturingClient()
    agent = MeetingAgent(settings, client=client)
    result = await agent.process(
        "meeting-1",
        transcript=[Utterance(1, "p-1", 0.0, 2.0, "en", None, 0.9, "We will confirm the date today.")],
    )

    assert result.transcript
    assert client.messages[1]["content"][0]["type"] == "text"
    assert "We will confirm" in client.messages[1]["content"][0]["text"]

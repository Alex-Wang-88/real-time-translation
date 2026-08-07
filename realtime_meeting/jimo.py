from __future__ import annotations

import ast
import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import Settings
from .formatting import paired_text
from .models import Utterance


SUMMARY_SYSTEM_PROMPT = """你是一名严谨的中文会议秘书。只使用会议记录中明确出现的信息，不得臆测。持续维护一份紧凑的结构化状态，必须保留：主题、结论、议题、异议、条件、决策及时间戳、行动项及负责人/期限、风险、阻塞、未决问题。人名或专有名词不确定时标记[待确认]。中间轮次只返回更新后的紧凑状态，不写寒暄。最终轮次输出中文会议纪要，结构为：1.会议主题；2.核心结论；3.讨论要点；4.决策记录；5.行动项表格；6.风险与阻塞；7.未决问题。"""


@dataclass(frozen=True, slots=True)
class SseEvent:
    event: str
    data: str


def parse_sse_lines(lines: Iterable[str]) -> Iterator[SseEvent]:
    event_name = "message"
    data_parts: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            if data_parts or event_name != "message":
                yield SseEvent(event_name, "\n".join(data_parts))
            event_name = "message"
            data_parts = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if not separator:
            continue
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event_name = value
        elif field == "data":
            data_parts.append(value)
    if data_parts or event_name != "message":
        yield SseEvent(event_name, "\n".join(data_parts))


async def parse_sse_async(lines: AsyncIterator[str]) -> AsyncIterator[SseEvent]:
    event_name = "message"
    data_parts: list[str] = []
    async for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            if data_parts or event_name != "message":
                yield SseEvent(event_name, "\n".join(data_parts))
            event_name = "message"
            data_parts = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if not separator:
            continue
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event_name = value
        elif field == "data":
            data_parts.append(value)
    if data_parts or event_name != "message":
        yield SseEvent(event_name, "\n".join(data_parts))


def _event_content(event: SseEvent) -> tuple[str, bool]:
    if event.event == "end":
        return "", True
    if event.event not in {"data", "message"} or not event.data.strip():
        return "", False
    try:
        payload = json.loads(event.data)
    except json.JSONDecodeError:
        try:
            payload = ast.literal_eval(event.data)
        except (ValueError, SyntaxError):
            return "", False
    if not isinstance(payload, dict):
        return "", False
    if "end" in payload:
        return "", True
    content = payload.get("content", "")
    return (str(content) if content is not None else ""), False


class JimoClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        if not settings.jimo_configured:
            raise ValueError("积墨 AI 尚未配置，请在 .env 设置 JIMO_API_URL 和 JIMO_AUTHORIZATION")
        self.settings = settings
        self._external_client = client

    @staticmethod
    def _request_chars(messages: list[dict[str, str]]) -> int:
        return sum(len(item.get("content", "")) for item in messages)

    async def complete(
        self,
        messages: list[dict[str, str]],
        session_id: str,
        *,
        on_delta: Callable[[str], Any] | None = None,
        on_reset: Callable[[], Any] | None = None,
    ) -> str:
        size = self._request_chars(messages)
        if size > self.settings.jimo_max_request_chars:
            raise ValueError(
                f"积墨单次请求为 {size} 字符，超过上限 {self.settings.jimo_max_request_chars}"
            )
        payload = {
            "messages": messages,
            "sessionId": session_id,
            "source": "api",
            "extra": {},
        }
        headers = {
            "Authorization": self.settings.jimo_authorization,
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(3):
            if attempt and on_reset:
                result = on_reset()
                if asyncio.iscoroutine(result):
                    await result
            try:
                return await self._complete_once(payload, headers, on_delta)
            except Exception as exc:
                last_error = exc
                if attempt >= 2:
                    break
                await asyncio.sleep(2 if attempt == 0 else 5)
        raise RuntimeError(f"积墨 AI 请求失败，已重试两次: {last_error}") from last_error

    async def _complete_once(
        self,
        payload: dict[str, object],
        headers: dict[str, str],
        on_delta: Callable[[str], Any] | None,
    ) -> str:
        owned = self._external_client is None
        client = self._external_client or httpx.AsyncClient(
            timeout=httpx.Timeout(180.0, connect=20.0), follow_redirects=True
        )
        chunks: list[str] = []
        ended = False
        try:
            async with client.stream(
                "POST", self.settings.jimo_api_url, headers=headers, json=payload
            ) as response:
                response.raise_for_status()
                async for event in parse_sse_async(response.aiter_lines()):
                    content, is_end = _event_content(event)
                    if content:
                        chunks.append(content)
                        if on_delta:
                            result = on_delta(content)
                            if asyncio.iscoroutine(result):
                                await result
                    if is_end:
                        ended = True
                        break
        finally:
            if owned:
                await client.aclose()
        if not ended:
            raise RuntimeError("积墨 AI SSE 在 end 事件之前断开")
        return "".join(chunks).strip()


def transcript_chunks(path: Path, max_chars: int) -> Iterator[tuple[int, float, float, str]]:
    lines: list[str] = []
    chars = 0
    chunk_index = 0
    start = 0.0
    end = 0.0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            item = Utterance.from_dict(json.loads(raw))
            pair = paired_text(item)
            if lines and chars + len(pair) + 1 > max_chars:
                chunk_index += 1
                yield chunk_index, start, end, "\n".join(lines)
                lines = []
                chars = 0
            if not lines:
                start = item.start
            lines.append(pair)
            chars += len(pair) + 1
            end = item.end
    if lines:
        chunk_index += 1
        yield chunk_index, start, end, "\n".join(lines)


class MeetingSummarizer:
    def __init__(self, settings: Settings, client: JimoClient | None = None) -> None:
        self.settings = settings
        self.client = client or JimoClient(settings)

    async def summarize(
        self,
        transcript_path: Path,
        session_id: str,
        started_at: str,
        ended_at: str,
        *,
        on_status: Callable[[str, int, int], Any],
        on_delta: Callable[[str], Any],
        on_reset: Callable[[], Any],
    ) -> str:
        max_raw = min(
            self.settings.jimo_transcript_chars,
            max(
                1_000,
                self.settings.jimo_max_request_chars
                - len(SUMMARY_SYSTEM_PROMPT)
                - self.settings.jimo_state_chars
                - 1_200,
            ),
        )
        total = sum(1 for _ in transcript_chunks(transcript_path, max_raw))
        if total == 0:
            raise ValueError("会议没有可总结的有效发言")
        state = ""
        for index, start, end, text in transcript_chunks(transcript_path, max_raw):
            marker = (
                f"[TRANSCRIPT_CHUNK {index} START={start:.3f} END={end:.3f}]\n{text}\n"
                f"请更新紧凑会议状态，控制在 {self.settings.jimo_state_chars} 字符以内。"
            )
            if index == 1:
                marker = f"[MEETING_START ID={session_id} STARTED_AT={started_at}]\n" + marker
            messages = [{"role": "system", "content": SUMMARY_SYSTEM_PROMPT}]
            if state:
                messages.append({"role": "assistant", "content": state})
            messages.append({"role": "user", "content": marker})
            status_result = on_status("chunk", index, total)
            if asyncio.iscoroutine(status_result):
                await status_result
            state = await self.client.complete(messages, session_id)
            if len(state) > self.settings.jimo_state_chars:
                compact_messages = [
                    {
                        "role": "system",
                        "content": "压缩下面的会议状态，保留全部决策、行动项、风险、异议和时间戳。只输出压缩结果。",
                    },
                    {"role": "user", "content": state[: self.settings.jimo_max_request_chars - 500]},
                ]
                state = await self.client.complete(compact_messages, session_id)
                state = state[: self.settings.jimo_state_chars]

        final_messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "assistant", "content": state},
            {
                "role": "user",
                "content": (
                    f"[MEETING_END ID={session_id} ENDED_AT={ended_at} TOTAL_CHUNKS={total}]\n"
                    "现在输出最终中文会议纪要。行动项必须有原稿依据，缺失负责人或期限写待确认。"
                ),
            },
        ]
        status_result = on_status("final", total, total)
        if asyncio.iscoroutine(status_result):
            await status_result
        return await self.client.complete(
            final_messages,
            session_id,
            on_delta=on_delta,
            on_reset=on_reset,
        )

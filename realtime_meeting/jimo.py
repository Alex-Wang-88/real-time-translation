from __future__ import annotations

import ast
import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import Settings
from .models import TodoDocument, TodoItem, Utterance, utc_now_iso
from .prompts import SUMMARY_SYSTEM_PROMPT, TODO_SYSTEM_PROMPT


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
            event_name, data_parts = "message", []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if not separator:
            continue
        if value.startswith(" "):
            value = value[1:]
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
            event_name, data_parts = "message", []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if not separator:
            continue
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data_parts.append(value)
    if data_parts or event_name != "message":
        yield SseEvent(event_name, "\n".join(data_parts))


def _event_content(event: SseEvent) -> tuple[str, bool]:
    if event.event.casefold() in {"end", "done"}:
        return "", True
    if event.event not in {"data", "message"} or not event.data.strip():
        return "", False
    if event.data.strip() in {"[DONE]", "DONE"}:
        return "", True
    payloads: list[dict[str, Any]] = []
    try:
        payload = json.loads(event.data)
        if isinstance(payload, dict):
            payloads.append(payload)
    except json.JSONDecodeError:
        for line in event.data.splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                try:
                    payload = ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    continue
            if isinstance(payload, dict):
                payloads.append(payload)
    if not payloads:
        text = event.data.strip()
        # Many share SSE endpoints stream plain text instead of JSON.
        # Treat non-empty, non-terminal data as a content chunk.
        if text and text not in {"[DONE]", "DONE"}:
            return text, False
        return "", False
    contents: list[str] = []
    ended = False
    for payload in payloads:
        if payload.get("end") is True or payload.get("done") is True or payload.get("finished") is True:
            ended = True
        content = payload.get("content")
        if content is None:
            choices = payload.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                delta = choices[0].get("delta", choices[0].get("message", {}))
                if isinstance(delta, dict):
                    content = delta.get("content", "")
        if content is not None:
            contents.append(str(content))
    return "".join(contents), ended


class JimoClient:
    """Compatibility client for the existing Jimo share SSE endpoint."""

    def __init__(
        self,
        settings: Settings,
        *,
        endpoint: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.endpoint = endpoint or settings.jimo_api_url
        if not self.endpoint.strip():
            raise ValueError("Jimo API URL 尚未配置")
        if not settings.jimo_authorization.strip():
            raise ValueError("JIMO_AUTHORIZATION 尚未配置")
        self._external_client = client

    @staticmethod
    def request_chars(messages: list[dict[str, str]]) -> int:
        return sum(len(str(item.get("content", ""))) for item in messages)

    async def complete(
        self,
        messages: list[dict[str, str]],
        session_id: str,
        *,
        on_delta: Callable[[str], Any] | None = None,
        on_reset: Callable[[], Any] | None = None,
    ) -> str:
        size = self.request_chars(messages)
        if size > self.settings.jimo_max_request_chars:
            raise ValueError(f"Jimo 请求为 {size} 字符，超过上限 {self.settings.jimo_max_request_chars}")
        payload = {"messages": messages, "sessionId": session_id, "source": "api", "extra": {}}
        headers = {
            "Authorization": self.settings.jimo_authorization,
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self.settings.jimo_max_retries):
            if attempt and on_reset:
                result = on_reset()
                if asyncio.iscoroutine(result):
                    await result
            try:
                return await self._complete_once(payload, headers, on_delta)
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.settings.jimo_max_retries:
                    await asyncio.sleep(min(10.0, 2.0**attempt))
        raise RuntimeError(f"Jimo 请求失败，已重试 {self.settings.jimo_max_retries - 1} 次: {last_error}") from last_error

    async def _complete_once(
        self,
        payload: dict[str, object],
        headers: dict[str, str],
        on_delta: Callable[[str], Any] | None,
    ) -> str:
        owned = self._external_client is None
        client = self._external_client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.settings.jimo_timeout_seconds, connect=self.settings.jimo_connect_timeout_seconds),
            follow_redirects=True,
        )
        chunks: list[str] = []
        response_chars = 0
        ended = False
        try:
            async with client.stream("POST", self.endpoint, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for event in parse_sse_async(response.aiter_lines()):
                    content, is_end = _event_content(event)
                    if content:
                        response_chars += len(content)
                        if response_chars > self.settings.jimo_max_response_chars:
                            raise ValueError(
                                f"Jimo response exceeds {self.settings.jimo_max_response_chars} characters"
                            )
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
            raise RuntimeError("Jimo SSE 在结束事件之前断开")
        return "".join(chunks).strip()


def paired_text(item: Utterance) -> str:
    translation = item.translation_zh.strip()
    source = item.text.strip()
    speaker = f"演讲人{item.speaker_id}"
    label = {"zh": "中文", "en": "英文", "de": "德文"}.get(item.language, item.language)
    original = f"[{item.start:.3f}-{item.end:.3f}] {speaker}（{label}）：{source}"
    if translation and item.language != "zh":
        return f"{original}\n[{item.start:.3f}-{item.end:.3f}] 中文翻译：{translation}"
    return original


def transcript_chunks(path: Path, max_chars: int) -> Iterator[tuple[int, float, float, str]]:
    lines: list[str] = []
    chars = 0
    index = 0
    start = end = 0.0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            item = Utterance.from_dict(json.loads(raw))
            pair = paired_text(item)
            if lines and chars + len(pair) + 1 > max_chars:
                index += 1
                yield index, start, end, "\n".join(lines)
                lines, chars = [], 0
            if not lines:
                start = item.start
            lines.append(pair)
            chars += len(pair) + 1
            end = item.end
    if lines:
        index += 1
        yield index, start, end, "\n".join(lines)


class MeetingSummarizer:
    def __init__(self, settings: Settings, client: JimoClient | None = None) -> None:
        self.settings = settings
        self.client = client or JimoClient(settings)

    async def summarize(
        self,
        transcript_path: Path,
        meeting_id: str,
        started_at: str,
        ended_at: str,
        *,
        on_status: Callable[[str, int, int], Any] | None = None,
        on_delta: Callable[[str], Any] | None = None,
        on_reset: Callable[[], Any] | None = None,
        attempt_id: str | None = None,
    ) -> str:
        raw_limit = min(
            self.settings.jimo_transcript_chars,
            max(1000, self.settings.jimo_max_request_chars - len(SUMMARY_SYSTEM_PROMPT) - self.settings.jimo_state_chars - 800),
        )
        chunks = list(transcript_chunks(transcript_path, raw_limit))
        if not chunks:
            raise ValueError("会议没有可总结的有效发言")
        attempt_id = attempt_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        state = ""
        for index, start, end, text in chunks:
            if on_status:
                result = on_status("chunk", index, len(chunks))
                if asyncio.iscoroutine(result):
                    await result
            marker = (
                f"MODE=STATE_UPDATE\nMEETING_ID={meeting_id}\nCHUNK_INDEX={index}\nCHUNK_TOTAL={len(chunks)}\n"
                f"TIME_RANGE={start:.3f}-{end:.3f}\n\n以下是本轮会议逐句记录：\n{text}\n\n"
                "请将本轮明确事实合并到会议状态中，只输出更新后的紧凑状态。"
            )
            messages = [{"role": "system", "content": SUMMARY_SYSTEM_PROMPT}]
            if state:
                messages.append({"role": "assistant", "content": state})
            messages.append({"role": "user", "content": marker})
            state = await self.client.complete(messages, f"meeting:{meeting_id}:summary:{attempt_id}")
            if len(state) > self.settings.jimo_state_chars:
                compact_messages = [
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "assistant", "content": state},
                    {"role": "user", "content": "MODE=STATE_UPDATE\n请压缩会议状态到 4000 个字符以内，保留所有决策、行动项、风险、异议和时间范围。只输出压缩后的状态。"},
                ]
                state = await self.client.complete(compact_messages, f"meeting:{meeting_id}:summary:{attempt_id}")
                state = state[: self.settings.jimo_state_chars]
        if on_status:
            result = on_status("final", len(chunks), len(chunks))
            if asyncio.iscoroutine(result):
                await result
        final_messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "assistant", "content": state},
            {"role": "user", "content": f"MODE=FINAL\nMEETING_ID={meeting_id}\nENDED_AT={ended_at}\nTOTAL_CHUNKS={len(chunks)}\n\n请根据当前会议状态输出最终中文会议纪要。会议开始时间：{started_at}"},
        ]
        final = await self.client.complete(
            final_messages,
            f"meeting:{meeting_id}:summary:{attempt_id}",
            on_delta=on_delta,
            on_reset=on_reset,
        )
        return _strip_markdown_fence(final)


def _strip_json_fence(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    # Fast path: the whole payload is already valid JSON.
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass
    # Real models often wrap JSON in prose or code fences. Locate the
    # outermost balanced JSON object/array and ignore the surrounding text.
    start = next((i for i, ch in enumerate(cleaned) if ch in "{["), None)
    if start is not None:
        depth = 0
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if char in "{[":
                depth += 1
            elif char in "}]":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : index + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break
    return cleaned


def _strip_markdown_fence(value: str) -> str:
    stripped = value.strip()
    stripped = re.sub(r"^```(?:markdown)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


class _TodoItemPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", str_max_length=4000)

    task: str = Field(min_length=1)
    owner: str | None = None
    due_date: str | None = None
    priority: str = "待确认"
    status: str = "未开始"
    source_time_start: float | None = None
    source_time_end: float | None = None
    evidence: str = ""
    notes: str = ""


class _TodoPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", str_max_length=4000)

    schema_version: str = "1.0"
    items: list[_TodoItemPayload] = Field(default_factory=list, max_length=100)


def parse_todo_document(raw: str, meeting_id: str, summary_revision: int) -> TodoDocument:
    try:
        payload = _TodoPayload.model_validate(json.loads(_strip_json_fence(raw)))
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        raise ValueError("To-do-list 节点返回的不是符合 schema 的合法 JSON") from exc
    now = utc_now_iso()
    items: list[TodoItem] = []
    for index, item in enumerate(payload.items):
        task = item.task.strip()
        if not task:
            continue
        owner = item.owner.strip() if isinstance(item.owner, str) else item.owner
        due_date = item.due_date.strip() if isinstance(item.due_date, str) else item.due_date
        items.append(
            TodoItem(
                task=task,
                owner=owner if owner not in {"", "待确认"} else None,
                due_date=due_date if due_date not in {"", "待确认"} else None,
                priority=item.priority or "待确认",
                status=item.status or "未开始",
                source_time_start=item.source_time_start,
                source_time_end=item.source_time_end,
                evidence=item.evidence or "",
                notes=item.notes or "",
                id=f"{meeting_id}-todo-{summary_revision}-{index + 1}",
                meeting_id=meeting_id,
                summary_revision=summary_revision,
                created_at=now,
            )
        )
    return TodoDocument(payload.schema_version or "1.0", items, meeting_id, summary_revision, now)


class TodoGenerator:
    def __init__(self, settings: Settings, client: JimoClient | None = None) -> None:
        self.settings = settings
        self.client = client or JimoClient(settings, endpoint=settings.jimo_todo_api_url)

    async def generate(
        self,
        meeting_id: str,
        summary_revision: int,
        minutes: str,
        *,
        on_status: Callable[[str], Any] | None = None,
    ) -> TodoDocument:
        if not minutes.strip():
            raise ValueError("会议纪要为空，无法生成 To-do-list")
        message = f"MEETING_ID={meeting_id}\nSUMMARY_REVISION={summary_revision}\n\n以下是完整会议纪要：\n{minutes}\n\n请只输出符合要求的 JSON。"
        if on_status:
            result = on_status("request")
            if asyncio.iscoroutine(result):
                await result
        raw = await self.client.complete(
            [{"role": "system", "content": TODO_SYSTEM_PROMPT}, {"role": "user", "content": message}],
            f"meeting:{meeting_id}:todo:{summary_revision}",
        )
        return parse_todo_document(raw, meeting_id, summary_revision)

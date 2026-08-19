from __future__ import annotations

import asyncio
import ast
import json
import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import Settings
from .exporter import render_todo_markdown
from .language import is_mixed_source_text
from .models import TodoDocument, TodoItem, Utterance, speech_variant_label, utc_now_iso
from .prompts import MEETING_AGENT_REQUEST_PROMPT


@dataclass(frozen=True, slots=True)
class SseEvent:
    event: str
    data: str


def parse_sse_lines(lines: Iterable[str]) -> Iterator[SseEvent]:
    """Read a platform event stream as complete events, without UI streaming."""

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
        return (text, False) if text and text not in {"[DONE]", "DONE"} else ("", False)

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
        if content is None:
            for key in ("answer", "text", "output", "result"):
                if payload.get(key) is not None:
                    content = payload[key]
                    break
        if content is not None:
            contents.append(str(content))
    return "".join(contents), ended


def _looks_like_sse(body: str, content_type: str) -> bool:
    normalized_type = content_type.casefold()
    if "text/event-stream" in normalized_type:
        return True
    stripped = body.lstrip()
    return stripped.startswith("event:") or stripped.startswith("data:")


def _content_for_budget(value: Any) -> str:
    """Return a stable text representation for request-size accounting."""

    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _response_content(value: Any) -> str:
    """Extract assistant text from JSON, OpenAI-like, or Jimo envelopes."""

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_response_content(item) for item in value]
        return "".join(part for part in parts if part)
    if not isinstance(value, Mapping):
        return _content_for_budget(value)

    # A single-node agent may return the structured object directly.
    if {"transcript", "minutes", "todo"}.issubset(value):
        return json.dumps(dict(value), ensure_ascii=False)

    choices = value.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, Mapping):
            message = choice.get("message") or choice.get("delta") or choice
            if isinstance(message, Mapping) and "content" in message:
                return _response_content(message.get("content"))

    for key in ("content", "answer", "text", "output", "result", "data", "message"):
        if key not in value:
            continue
        candidate = value.get(key)
        if isinstance(candidate, Mapping) and {"transcript", "minutes", "todo"}.issubset(candidate):
            return json.dumps(dict(candidate), ensure_ascii=False)
        extracted = _response_content(candidate)
        if extracted:
            return extracted
    return json.dumps(dict(value), ensure_ascii=False)


class JimoClient:
    """Client for the single configured Jimo share endpoint.

    The application consumes one complete result. Some share gateways still
    transport that result as SSE even when JSON is requested, so the response
    parser collects the stream internally and never exposes token deltas to the
    frontend.
    """

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
    def request_chars(messages: list[dict[str, Any]]) -> int:
        return sum(len(_content_for_budget(item.get("content", ""))) for item in messages)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
    ) -> str:
        size = self.request_chars(messages)
        if size > self.settings.jimo_max_request_chars:
            raise ValueError(f"Jimo 请求为 {size} 字符，超过上限 {self.settings.jimo_max_request_chars}")
        payload = {"messages": messages, "sessionId": session_id, "source": "api", "extra": {}}
        headers = {
            "Authorization": self.settings.jimo_authorization,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self.settings.jimo_max_retries):
            try:
                return await self._complete_once(payload, headers)
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.settings.jimo_max_retries:
                    await asyncio.sleep(min(10.0, 2.0**attempt))
        raise RuntimeError(f"Jimo 请求失败，已重试 {self.settings.jimo_max_retries - 1} 次: {last_error}") from last_error

    async def _complete_once(
        self,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> str:
        owned = self._external_client is None
        client = self._external_client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.settings.jimo_timeout_seconds, connect=self.settings.jimo_connect_timeout_seconds),
            follow_redirects=True,
        )
        try:
            response = await client.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            body = response.text
            if _looks_like_sse(body, response.headers.get("content-type", "")):
                chunks: list[str] = []
                ended = False
                for event in parse_sse_lines(body.splitlines()):
                    event_text, is_end = _event_content(event)
                    if event_text:
                        chunks.append(event_text)
                    if is_end:
                        ended = True
                        break
                if not ended and chunks:
                    ended = True
                content = "".join(chunks).strip()
                if not ended:
                    raise RuntimeError("Jimo SSE 响应在结束之前断开")
            else:
                try:
                    response_payload = response.json()
                except (json.JSONDecodeError, ValueError):
                    content = body.strip()
                else:
                    content = _response_content(response_payload).strip()

            if len(content) > self.settings.jimo_max_response_chars:
                raise ValueError(f"Jimo response exceeds {self.settings.jimo_max_response_chars} characters")
        finally:
            if owned:
                await client.aclose()
        return content


def paired_text(item: Utterance) -> str:
    translation = item.translation_zh.strip()
    source = item.text.strip()
    label = {"zh": "中文", "en": "英文", "de": "德文"}.get(item.language, item.language)
    variant = f"/{speech_variant_label(item.speech_variant)}" if item.speech_variant else ""
    original = f"[{item.start:.3f}-{item.end:.3f}]（{label}{variant}）：{source}"
    if translation and (item.language != "zh" or is_mixed_source_text(source)):
        return f"{original}\n[{item.start:.3f}-{item.end:.3f}] 中文翻译：{translation}"
    return original


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


_MARKED_AGENT_SECTION_RE = re.compile(
    r"@@JIMO_SECTION:(DATA|SUMMARY|TODOLIST):BEGIN@@[ \t]*\r?\n"
    r"(?P<body>.*?)"
    r"\r?\n@@JIMO_SECTION:\1:END@@",
    re.DOTALL,
)
_TRANSCRIPT_HEADING_RE = re.compile(
    r"^###\s+\[S?(?P<index>\d+)\]\s+时间\s*[:：]\s*(?P<time>.+?)\s*$",
    re.MULTILINE,
)
_TODO_HEADING_RE = re.compile(r"^###\s+(?P<id>T\d+)\s*$", re.MULTILINE)


def _marked_agent_sections(raw: str) -> dict[str, str] | None:
    """Extract the three fixed sections emitted by the platform end node."""

    text = _strip_markdown_fence(str(raw or ""))
    sections: dict[str, str] = {}
    for match in _MARKED_AGENT_SECTION_RE.finditer(text):
        name = match.group(1)
        if name in sections:
            return None
        sections[name] = match.group("body").strip()
    if set(sections) != {"DATA", "SUMMARY", "TODOLIST"}:
        return None
    return sections


def _markdown_fields(block: str, labels: tuple[str, ...]) -> dict[str, str]:
    label_expr = "|".join(re.escape(label) for label in labels)
    pattern = re.compile(
        rf"^- (?P<label>{label_expr})\s*[:：]\s*(?P<value>.*?)"
        rf"(?=^- (?:{label_expr})\s*[:：]|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    return {match.group("label"): match.group("value").strip() for match in pattern.finditer(block)}


def _markdown_time_range(value: str) -> tuple[float | None, float | None]:
    parts = re.split(r"\s+[-–—]\s+", str(value or "").strip(), maxsplit=1)
    if len(parts) != 2:
        return None, None
    return _optional_number(parts[0]), _optional_number(parts[1])


def _markdown_language(value: str) -> str:
    normalized = str(value or "").strip()
    return {
        "中文": "zh",
        "普通话": "zh",
        "英文": "en",
        "英语": "en",
        "德文": "de",
        "德语": "de",
    }.get(normalized, normalized or "unknown")


def _parse_markdown_transcript(markdown: str, meeting_id: str) -> list[dict[str, Any]]:
    headings = list(_TRANSCRIPT_HEADING_RE.finditer(markdown))
    items: list[dict[str, Any]] = []
    labels = ("说话人", "语言", "原文", "中文翻译", "是否存疑", "存疑说明")
    for position, heading in enumerate(headings):
        block_end = headings[position + 1].start() if position + 1 < len(headings) else len(markdown)
        block = markdown[heading.end() : block_end]
        fields = _markdown_fields(block, labels)
        original = fields.get("原文", "").strip()
        translation = fields.get("中文翻译", "").strip()
        if not original and not translation:
            continue
        start, end = _markdown_time_range(heading.group("time"))
        index = int(heading.group("index"))
        items.append(
            {
                "index": index,
                "speaker": fields.get("说话人", "待确认") or "待确认",
                "language": _markdown_language(fields.get("语言", "")),
                "start": start,
                "end": end,
                "original": original,
                "translation_zh": translation,
                "uncertain": fields.get("是否存疑", "").strip().lower() in {"是", "true", "yes"},
                "uncertainty_note": fields.get("存疑说明", "").strip(),
            }
        )
    return _normalise_agent_transcript(items, meeting_id)


def _parse_markdown_todo(markdown: str, meeting_id: str, summary_revision: int) -> TodoDocument:
    headings = list(_TODO_HEADING_RE.finditer(markdown))
    labels = ("任务", "负责人", "截止时间", "优先级", "当前状态", "原文依据", "时间范围", "事实说明")
    raw_items: list[dict[str, Any]] = []
    for position, heading in enumerate(headings):
        block_end = headings[position + 1].start() if position + 1 < len(headings) else len(markdown)
        fields = _markdown_fields(markdown[heading.end() : block_end], labels)
        task = fields.get("任务", "").strip()
        if not task or task in {"无", "暂无明确待办事项"}:
            continue
        start, end = _markdown_time_range(fields.get("时间范围", ""))
        source = fields.get("原文依据", "").strip()
        fact = fields.get("事实说明", "").strip()
        evidence = fact or source
        if fact and source:
            evidence = f"{fact}（依据：{source}）"
        raw_items.append(
            {
                "task": task,
                "owner": fields.get("负责人") or None,
                "due_date": fields.get("截止时间") or None,
                "priority": fields.get("优先级") or "待确认",
                "status": fields.get("当前状态") or "未开始",
                "source_time_start": start,
                "source_time_end": end,
                "evidence": evidence,
                "notes": "",
            }
        )
    payload = {"schema_version": "1.0", "items": raw_items}
    return parse_todo_document(json.dumps(payload, ensure_ascii=False), meeting_id, summary_revision)


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


def _first_text(value: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        candidate = value.get(key)
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return text
    return ""


def _optional_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:\.(\d{1,3}))?\s*", value)
        if match:
            hours = float(match.group(1) or 0)
            minutes = float(match.group(2))
            seconds = float(match.group(3))
            fraction = float(f"0.{match.group(4)}") if match.group(4) else 0.0
            value = hours * 3600 + minutes * 60 + seconds + fraction
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return round(result, 3) if math.isfinite(result) and result >= 0 else None


def _source_indices(value: Any) -> list[int]:
    values = value if isinstance(value, list) else []
    result: list[int] = []
    for item in values:
        try:
            index = int(item)
        except (TypeError, ValueError, OverflowError):
            continue
        if index > 0 and index not in result:
            result.append(index)
    return result


def _normalise_agent_transcript(raw_items: Any, meeting_id: str) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []
    result: list[dict[str, Any]] = []
    for position, raw_item in enumerate(raw_items, start=1):
        if not isinstance(raw_item, Mapping):
            continue
        try:
            index = max(1, int(raw_item.get("index", position) or position))
        except (TypeError, ValueError, OverflowError):
            index = position
        original = _first_text(raw_item, "original", "text", "source")
        translation = _first_text(raw_item, "translation_zh", "translation", "chinese_translation")
        if not original and not translation:
            continue
        result.append(
            {
                "id": str(raw_item.get("id") or f"{meeting_id}-refined-{index}"),
                "index": index,
                "speaker": _first_text(raw_item, "speaker", "speaker_id", "speaker_label") or "待确认",
                "speaker_name": raw_item.get("speaker_name"),
                "language": _first_text(raw_item, "language", "lang") or "unknown",
                "speech_variant": raw_item.get("speech_variant"),
                "start": _optional_number(raw_item.get("start", raw_item.get("time_start"))),
                "end": _optional_number(raw_item.get("end", raw_item.get("time_end"))),
                "original": original,
                "translation_zh": translation,
                "uncertain": bool(raw_item.get("uncertain", False)),
                "uncertainty_note": _first_text(raw_item, "uncertainty_note", "uncertainty"),
                "source_indices": _source_indices(raw_item.get("source_indices")) or [index],
            }
        )
    return result


def _normalise_minutes(raw_minutes: Any) -> dict[str, Any]:
    if isinstance(raw_minutes, Mapping):
        return dict(raw_minutes)
    if isinstance(raw_minutes, str) and raw_minutes.strip():
        return {"topic": "", "plain_text": raw_minutes.strip()}
    return {}


def _markdown_cell(value: Any) -> str:
    return " ".join(str(value or "").replace("|", "\\|").splitlines()).strip() or "未提及"


def _entry_text(value: Any, *keys: str) -> str:
    if isinstance(value, Mapping):
        return _first_text(value, *keys)
    return str(value or "").strip()


def render_agent_minutes(minutes: Mapping[str, Any] | str, todo: TodoDocument | None = None) -> str:
    """Render the structured agent minutes into the existing Markdown card."""

    if isinstance(minutes, str):
        return _strip_markdown_fence(minutes)
    values = dict(minutes)
    plain_text = _first_text(values, "plain_text", "summary_markdown")
    if plain_text:
        return _strip_markdown_fence(plain_text)

    topic = _first_text(values, "topic", "title") or "未提及"
    core = values.get("core_conclusions") or values.get("core_conclusion") or []
    discussions = values.get("discussion_points") or values.get("key_points") or []
    decisions = values.get("decisions") or []
    risks = values.get("risks_and_blockers") or values.get("risks") or []
    questions = values.get("open_questions") or values.get("unresolved_questions") or []
    lines = ["# 会议纪要", "", "## 1. 会议主题", topic, "", "## 2. 核心结论"]
    if isinstance(core, list) and core:
        for item in core:
            text = _entry_text(item, "text", "conclusion", "summary")
            if text:
                lines.append(f"- {text}")
    else:
        lines.append("未提及")

    lines.extend(["", "## 3. 讨论要点"])
    if isinstance(discussions, list) and discussions:
        for item in discussions:
            if isinstance(item, Mapping):
                heading = _first_text(item, "topic", "title") or "未命名议题"
                lines.append(f"### {heading}")
                points = item.get("points") or item.get("items") or []
                if isinstance(points, list) and points:
                    lines.extend(f"- {str(point).strip()}" for point in points if str(point).strip())
                else:
                    text = _first_text(item, "text", "summary")
                    lines.append(f"- {text or '未提及'}")
            else:
                text = str(item).strip()
                if text:
                    lines.append(f"- {text}")
    else:
        lines.append("未提及")

    lines.extend(["", "## 4. 决策记录", "", "| 决策 | 条件或依据 | 原文句子 |", "|---|---|---|"])
    if isinstance(decisions, list) and decisions:
        for item in decisions:
            if isinstance(item, Mapping):
                indices = ", ".join(str(index) for index in _source_indices(item.get("source_indices"))) or "待确认"
                lines.append(
                    f"| {_markdown_cell(_first_text(item, 'decision', 'text'))} | "
                    f"{_markdown_cell(_first_text(item, 'basis', 'condition', 'evidence'))} | {indices} |"
                )
            else:
                lines.append(f"| {_markdown_cell(item)} | 未提及 | 待确认 |")
    else:
        lines.append("| 未提及 | 未提及 | 待确认 |")

    lines.extend(["", "## 5. 行动项", "", "| 任务 | 负责人 | 截止时间 | 优先级 | 状态 | 依据 |", "|---|---|---|---|---|---|"])
    if todo and todo.items:
        for item in todo.items:
            lines.append(
                f"| {_markdown_cell(item.task)} | {_markdown_cell(item.owner)} | {_markdown_cell(item.due_date)} | "
                f"{_markdown_cell(item.priority)} | {_markdown_cell(item.status)} | {_markdown_cell(item.evidence)} |"
            )
    else:
        lines.append("| 未提及 | 待确认 | 待确认 | 待确认 | 未开始 | 未提及 |")

    def append_text_section(title: str, entries: Any) -> None:
        lines.extend(["", title])
        if isinstance(entries, list) and entries:
            for item in entries:
                text = _entry_text(item, "text", "question", "risk", "summary")
                if text:
                    lines.append(f"- {text}")
        else:
            lines.append("未提及")

    append_text_section("## 6. 风险与阻塞", risks)
    append_text_section("## 7. 未决问题", questions)
    return "\n".join(lines).strip()


@dataclass(frozen=True, slots=True)
class MeetingAgentResult:
    schema_version: str
    meta: dict[str, Any]
    transcript: list[dict[str, Any]]
    minutes: dict[str, Any]
    summary_markdown: str
    todo: TodoDocument
    transcript_markdown: str = ""
    todo_markdown: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "meta": dict(self.meta),
            "transcript": [dict(item) for item in self.transcript],
            "minutes": dict(self.minutes),
            "summary_markdown": self.summary_markdown,
            "todo": [item.to_dict() for item in self.todo.items],
            "transcript_markdown": self.transcript_markdown,
            "todo_markdown": self.todo_markdown,
        }


def _agent_payload_from_raw(raw: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        payload: Any = dict(raw)
    else:
        cleaned = _strip_json_fence(str(raw or ""))
        try:
            payload = json.loads(cleaned)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("单节点智能体返回的不是合法 JSON") from exc
    for _ in range(3):
        if not isinstance(payload, Mapping):
            break
        if {"transcript", "minutes", "todo"}.issubset(payload):
            return dict(payload)
        nested = next(
            (payload.get(key) for key in ("data", "result", "output", "content") if isinstance(payload.get(key), Mapping)),
            None,
        )
        if nested is None:
            break
        payload = nested
    if not isinstance(payload, Mapping):
        raise ValueError("单节点智能体 JSON 顶层必须是对象")
    values = dict(payload)
    if "transcript" not in values:
        values["transcript"] = values.get("sentences", values.get("segments", []))
    if "minutes" not in values:
        values["minutes"] = values.get("meeting_minutes", values.get("summary", {}))
    if "todo" not in values:
        values["todo"] = values.get("todo_list", values.get("action_items", []))
    return values


def parse_meeting_agent_result(
    raw: str | Mapping[str, Any],
    meeting_id: str,
    summary_revision: int,
) -> MeetingAgentResult:
    if isinstance(raw, str):
        sections = _marked_agent_sections(raw)
        if sections is not None:
            transcript_markdown = sections["DATA"]
            summary_markdown = sections["SUMMARY"]
            todo_markdown = sections["TODOLIST"]
            transcript = _parse_markdown_transcript(transcript_markdown, meeting_id)
            if not transcript:
                raise ValueError("分段智能体返回的 DATA 区块中没有可解析的逐句转写")
            todo = _parse_markdown_todo(todo_markdown, meeting_id, summary_revision)
            return MeetingAgentResult(
                schema_version="1.0",
                meta={"output_format": "marked_markdown"},
                transcript=transcript,
                minutes={"plain_text": summary_markdown},
                summary_markdown=summary_markdown,
                todo=todo,
                transcript_markdown=transcript_markdown,
                todo_markdown=todo_markdown,
            )
    payload = _agent_payload_from_raw(raw)
    transcript = _normalise_agent_transcript(payload.get("transcript"), meeting_id)
    if not transcript:
        raise ValueError("单节点智能体返回的 transcript 为空")
    minutes = _normalise_minutes(payload.get("minutes"))
    raw_todo = payload.get("todo", [])
    if isinstance(raw_todo, Mapping):
        raw_todo = raw_todo.get("items", [])
    todo_payload = {"schema_version": "1.0", "items": raw_todo if isinstance(raw_todo, list) else []}
    todo = parse_todo_document(json.dumps(todo_payload, ensure_ascii=False), meeting_id, summary_revision)
    meta = dict(payload.get("meta")) if isinstance(payload.get("meta"), Mapping) else {}
    return MeetingAgentResult(
        schema_version=str(payload.get("schema_version") or "1.0"),
        meta=meta,
        transcript=transcript,
        minutes=minutes,
        summary_markdown=render_agent_minutes(minutes, todo),
        todo=todo,
        todo_markdown=render_todo_markdown(todo),
    )


class MeetingAgent:
    """One non-streaming request for refined transcript, minutes and todos."""

    def __init__(self, settings: Settings, client: JimoClient | None = None) -> None:
        self.settings = settings
        self.client = client or JimoClient(settings, endpoint=settings.jimo_api_url)

    async def process(
        self,
        meeting_id: str,
        *,
        summary_revision: int = 0,
        started_at: str = "",
        ended_at: str = "",
        title: str = "",
        audio_files: Iterable[Mapping[str, Any]] | None = None,
        transcript: Iterable[Utterance] | None = None,
        on_status: Callable[[str], Any] | None = None,
    ) -> MeetingAgentResult:
        files = [dict(item) for item in (audio_files or []) if str(item.get("url", "")).strip()]
        context = (
            f"MEETING_ID={meeting_id}\nTITLE={title}\nSTARTED_AT={started_at}\nENDED_AT={ended_at}\n\n"
            "请完成逐句转写（精修）、会议纪要和 To-do-list 三个步骤，并严格遵循平台智能体配置的三个分隔区块输出格式。\n"
        )
        if files:
            context += "音视频文件的链接列表为：\n" + "\n".join(str(item["url"]) for item in files)
        else:
            source_lines = [paired_text(item) for item in (transcript or []) if item.text.strip()]
            source = "\n\n".join(source_lines)
            budget = max(
                1000,
                self.settings.jimo_max_request_chars - len(MEETING_AGENT_REQUEST_PROMPT) - len(context) - 256,
            )
            source_limit = min(self.settings.jimo_transcript_chars, budget)
            if len(source) > source_limit:
                marker = "\n[本地转写中段省略]"
                source = source[: max(0, source_limit - len(marker))] + marker
            if not source:
                raise ValueError("没有可发送给单节点智能体的音频链接或本地转写")
            context += "未提供公网音频链接，以下是本地实时转写上下文，请据此完成精修和整理：\n" + source

        content: list[dict[str, Any]] = [{"type": "text", "text": context}]
        for item in files:
            file_url: dict[str, Any] = {"url": str(item["url"])}
            file_id = item.get("file_id") or item.get("fileId")
            if file_id:
                file_url["fileId"] = str(file_id)
            content.append({"type": "file_url", "file_url": file_url})

        if on_status:
            status_result = on_status("request")
            if asyncio.iscoroutine(status_result):
                await status_result
        raw = await self.client.complete(
            [
                {"role": "system", "content": MEETING_AGENT_REQUEST_PROMPT},
                {"role": "user", "content": content},
            ],
            f"meeting:{meeting_id}:agent",
        )
        return parse_meeting_agent_result(raw, meeting_id, summary_revision)

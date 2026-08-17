from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from .language import CANONICAL_SPEECH_VARIANTS, VARIANT_LABELS

LANGUAGE_LABELS = {"zh": "中文", "en": "English", "de": "Deutsch", "unknown": "未知"}
SPEECH_VARIANT_LABELS = {**VARIANT_LABELS, "unknown": "方言未确认"}
SUPPORTED_LANGUAGES = ("zh", "en", "de")
SUPPORTED_SPEECH_VARIANTS = CANONICAL_SPEECH_VARIANTS

RecordingState = Literal["created", "starting", "recording", "finalizing", "complete", "error"]
SummaryState = Literal["idle", "queued", "running", "complete", "error"]
TodoState = Literal["waiting_summary", "queued", "running", "complete", "stale", "error"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def language_label(code: str | None) -> str:
    normalized = (code or "").strip().casefold()
    return LANGUAGE_LABELS.get(normalized, f"未知语言（{normalized or 'unknown'}）")


def speech_variant_label(value: str | None) -> str:
    normalized = (value or "").strip().casefold()
    return SPEECH_VARIANT_LABELS.get(normalized, normalized or "方言未确认")


@dataclass(slots=True)
class Utterance:
    """One displayed paragraph; the historical class name is kept internally."""

    id: int
    segment_id: str
    start: float
    end: float
    language: str
    speech_variant: str | None
    language_confidence: float
    text: str
    translation_zh: str = ""
    translation_status: str = "pending"
    revision: int = 1
    source_revision: int = 1
    closed: bool = False
    asr_model: str | None = None
    language_source: str = "qwen"
    translation_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Utterance":
        values = dict(payload)
        # Storage is schema 2 and does not migrate old meetings.  Unknown keys
        # are ignored so a partially written record never leaks extra fields.
        allowed = {
            "id", "segment_id", "start", "end", "language", "speech_variant",
            "language_confidence", "text", "translation_zh", "translation_status",
            "revision", "source_revision", "closed", "asr_model", "language_source",
            "translation_model",
        }
        values = {key: value for key, value in values.items() if key in allowed}
        if "segment_id" not in values:
            values["segment_id"] = f"legacy:{values.get('id', 0)}"
        values.setdefault("id", 0)
        values.setdefault("start", 0.0)
        values.setdefault("end", values["start"])
        values.setdefault("language", "unknown")
        values.setdefault("speech_variant", None)
        values.setdefault("language_confidence", 0.0)
        values.setdefault("text", "")
        values.setdefault("translation_zh", "")
        values.setdefault("translation_status", "pending")
        values.setdefault("revision", 1)
        values.setdefault("source_revision", 1)
        values.setdefault("closed", False)
        values.setdefault("asr_model", None)
        values.setdefault("language_source", "qwen")
        values.setdefault("translation_model", None)
        return cls(
            id=int(values["id"]),
            segment_id=str(values["segment_id"]),
            start=float(values["start"]),
            end=float(values["end"]),
            language=str(values["language"]),
            speech_variant=values["speech_variant"],
            language_confidence=float(values["language_confidence"] or 0.0),
            text=str(values["text"] or ""),
            translation_zh=str(values["translation_zh"] or ""),
            translation_status=str(values["translation_status"] or "pending"),
            revision=max(1, int(values["revision"] or 1)),
            source_revision=max(1, int(values["source_revision"] or 1)),
            closed=bool(values["closed"]),
            asr_model=values["asr_model"],
            language_source=str(values["language_source"] or "qwen"),
            translation_model=values["translation_model"],
        )


Paragraph = Utterance


@dataclass(slots=True)
class TodoItem:
    task: str
    owner: str | None = None
    due_date: str | None = None
    priority: str = "待确认"
    status: str = "未开始"
    source_time_start: float | None = None
    source_time_end: float | None = None
    evidence: str = ""
    notes: str = ""
    id: str = ""
    meeting_id: str = ""
    summary_revision: int = 0
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TodoItem":
        values = dict(payload)

        def optional_float(value: Any) -> float | None:
            if value is None or value == "":
                return None
            try:
                result = float(value)
                return result if math.isfinite(result) else None
            except (TypeError, ValueError, OverflowError):
                return None

        try:
            summary_revision = max(0, int(values.get("summary_revision", 0) or 0))
        except (TypeError, ValueError):
            summary_revision = 0
        return cls(
            task=str(values.get("task", "") or ""),
            owner=str(values["owner"]) if values.get("owner") is not None else None,
            due_date=str(values["due_date"]) if values.get("due_date") is not None else None,
            priority=str(values.get("priority", "待确认") or "待确认"),
            status=str(values.get("status", "未开始") or "未开始"),
            source_time_start=optional_float(values.get("source_time_start")),
            source_time_end=optional_float(values.get("source_time_end")),
            evidence=str(values.get("evidence", "") or ""),
            notes=str(values.get("notes", "") or ""),
            id=str(values.get("id", "") or ""),
            meeting_id=str(values.get("meeting_id", "") or ""),
            summary_revision=summary_revision,
            created_at=str(values.get("created_at", "") or ""),
        )


@dataclass(slots=True)
class TodoDocument:
    schema_version: str = "1.0"
    items: list[TodoItem] = field(default_factory=list)
    meeting_id: str = ""
    summary_revision: int = 0
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "meeting_id": self.meeting_id,
            "summary_revision": self.summary_revision,
            "generated_at": self.generated_at,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, default_meeting_id: str = "") -> "TodoDocument":
        values = dict(payload)
        try:
            summary_revision = max(0, int(values.get("summary_revision", 0) or 0))
        except (TypeError, ValueError):
            summary_revision = 0
        raw_items = values.get("items", [])
        items: list[TodoItem] = []
        if isinstance(raw_items, list):
            for raw_item in raw_items:
                if not isinstance(raw_item, dict):
                    continue
                try:
                    item = TodoItem.from_dict(raw_item)
                except (TypeError, ValueError):
                    continue
                if item.task.strip():
                    if not item.meeting_id:
                        item.meeting_id = str(values.get("meeting_id", default_meeting_id) or default_meeting_id)
                    if item.summary_revision <= 0:
                        item.summary_revision = summary_revision
                    items.append(item)
        return cls(
            schema_version=str(values.get("schema_version", "1.0") or "1.0"),
            items=items,
            meeting_id=str(values.get("meeting_id", default_meeting_id) or default_meeting_id),
            summary_revision=summary_revision,
            generated_at=str(values.get("generated_at", "") or ""),
        )

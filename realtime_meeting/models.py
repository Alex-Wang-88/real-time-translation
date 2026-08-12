from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

LANGUAGE_LABELS = {"zh": "中文", "en": "英文", "de": "德文"}
SUPPORTED_LANGUAGES = tuple(LANGUAGE_LABELS)

RecordingState = Literal["starting", "recording", "finalizing", "complete", "error"]
SummaryState = Literal["idle", "queued", "running", "complete", "error"]
TodoState = Literal["waiting_summary", "queued", "running", "complete", "stale", "error"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def language_label(code: str | None) -> str:
    normalized = (code or "").strip().casefold()
    return LANGUAGE_LABELS.get(normalized, f"未知语言（{normalized or 'unknown'}）")


@dataclass(slots=True)
class Utterance:
    id: int
    start: float
    end: float
    speaker_id: int
    language: str
    language_confidence: float
    text: str
    translation_zh: str = ""
    translation_status: str = "pending"
    segment_id: str = ""
    revision: int = 1
    recognition_stage: str = "fast"
    source_segment_id: str = ""
    asr_model: str | None = None
    language_source: str = "detector"
    speaker_source: str = "online"
    speaker_confidence: float = 0.0
    speaker_ids: list[int] = field(default_factory=list)
    speaker_overlap: float = 0.0
    translation_model: str | None = None
    deleted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Utterance":
        values = dict(payload)
        if "translation_zh" not in values:
            values["translation_zh"] = values.pop("translation_en", "")
        values.setdefault("translation_status", "pending")
        values.setdefault("segment_id", f"legacy:{values.get('id', 0)}")
        values.setdefault("revision", 1)
        values.setdefault("recognition_stage", "fast")
        values.setdefault("source_segment_id", str(values.get("segment_id", "")).split(":", 1)[0])
        values.setdefault("asr_model", None)
        values.setdefault("language_source", "detector")
        values.setdefault("speaker_source", "online")
        values.setdefault("speaker_confidence", 0.0)
        values.setdefault("speaker_ids", [])
        values.setdefault("speaker_overlap", 0.0)
        values.setdefault("translation_model", None)
        values.setdefault("deleted", False)
        if not isinstance(values["speaker_ids"], list):
            values["speaker_ids"] = [values["speaker_ids"]]
        return cls(**values)


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

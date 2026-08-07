from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


# Whisper can identify far more than the three languages used in the original
# demo.  Keep the labels in one place so a Russian/Spanish/Portuguese turn is
# not accidentally presented as English in the UI and exports.
LANGUAGE_LABELS = {
    "zh": "中文",
    "en": "英文",
    "de": "德文",
    "ru": "俄文",
    "es": "西班牙文",
    "pt": "葡萄牙文",
    "fr": "法文",
    "it": "意大利文",
    "ja": "日文",
    "ko": "韩文",
    "ar": "阿拉伯文",
    "uk": "乌克兰文",
    "pl": "波兰文",
    "nl": "荷兰文",
    "tr": "土耳其文",
    "vi": "越南文",
    "id": "印尼文",
    "th": "泰文",
    "cs": "捷克文",
    "sv": "瑞典文",
    "da": "丹麦文",
    "no": "挪威文",
    "fi": "芬兰文",
}


def language_label(code: str | None) -> str:
    """Return a stable Chinese label for a Whisper language code.

    Unknown codes are still shown rather than silently falling back to English;
    this matters for the user's original-language line when a meeting uses a
    language not in the compact label table above.
    """

    normalized = (code or "").strip().casefold()
    return LANGUAGE_LABELS.get(normalized, normalized or "未知语言")
SessionState = Literal[
    "starting",
    "recording",
    "finalizing",
    "summary_pending",
    "summarizing",
    "complete",
    "summary_error",
    "error",
]


@dataclass(slots=True)
class Utterance:
    id: int
    start: float
    end: float
    speaker_id: int
    language: str
    language_confidence: float
    text: str
    # The live translation target is always Simplified Chinese.  The old
    # ``translation_en`` key is accepted when recovering JSONL written by
    # earlier builds (see ``from_dict`` below), but new records are explicit.
    translation_zh: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Utterance":
        values = dict(payload)
        if "translation_zh" not in values:
            values["translation_zh"] = values.pop("translation_en", "")
        else:
            # Do not leak the legacy field into the dataclass constructor if a
            # caller sends both versions during a rolling upgrade.
            values.pop("translation_en", None)
        return cls(**values)

    @property
    def translation_en(self) -> str:
        """Compatibility read alias for integrations using the old field."""

        return self.translation_zh

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MeetingSnapshot:
    id: str
    state: SessionState
    started_at: str
    elapsed_seconds: float
    current_language: str | None
    utterance_count: int
    recent_utterances: list[dict[str, Any]]
    summary: str = ""
    error: str | None = None
    files: list[str] = field(default_factory=list)
    audio_bytes_received: int = 0
    audio_packets_received: int = 0
    audio_samples_received: int = 0
    audio_level: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

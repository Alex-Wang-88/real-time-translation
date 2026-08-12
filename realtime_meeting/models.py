from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


# Whisper can identify far more than the three languages used in the original
# demo.  Keep the labels in one place so a Russian/Spanish/Portuguese turn is
# not accidentally presented as English in the UI and exports.
LANGUAGE_LABELS = {
    "zh": "中文",
    "yue": "粤语",
    "en": "英文",
    "de": "德文",
    "af": "南非荷兰文",
    "am": "阿姆哈拉文",
    "ar": "阿拉伯文",
    "as": "阿萨姆文",
    "az": "阿塞拜疆文",
    "ba": "巴什基尔文",
    "be": "白俄罗斯文",
    "bo": "藏文",
    "bn": "孟加拉文",
    "ru": "俄文",
    "es": "西班牙文",
    "pt": "葡萄牙文",
    "fr": "法文",
    "it": "意大利文",
    "ja": "日文",
    "ko": "韩文",
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
    "bs": "波斯尼亚文",
    "ca": "加泰罗尼亚文",
    "cy": "威尔士文",
    "el": "希腊文",
    "et": "爱沙尼亚文",
    "eu": "巴斯克文",
    "fa": "波斯文",
    "fo": "法罗文",
    "gl": "加利西亚文",
    "gu": "古吉拉特文",
    "ha": "豪萨文",
    "haw": "夏威夷文",
    "he": "希伯来文",
    "hi": "印地文",
    "ht": "海地克里奥尔文",
    "hy": "亚美尼亚文",
    "is": "冰岛文",
    "jw": "爪哇文",
    "ka": "格鲁吉亚文",
    "kk": "哈萨克文",
    "km": "高棉文",
    "kn": "卡纳达文",
    "la": "拉丁文",
    "lb": "卢森堡文",
    "ln": "林加拉文",
    "lo": "老挝文",
    "lt": "立陶宛文",
    "lv": "拉脱维亚文",
    "mg": "马达加斯加文",
    "mi": "毛利文",
    "mk": "马其顿文",
    "ml": "马拉雅拉姆文",
    "mn": "蒙古文",
    "mr": "马拉地文",
    "ms": "马来文",
    "mt": "马耳他文",
    "my": "缅甸文",
    "ne": "尼泊尔文",
    "nn": "新挪威文",
    "oc": "奥克西坦文",
    "pa": "旁遮普文",
    "ps": "普什图文",
    "ro": "罗马尼亚文",
    "sa": "梵文",
    "sd": "信德文",
    "si": "僧伽罗文",
    "sk": "斯洛伐克文",
    "sl": "斯洛文尼亚文",
    "sn": "修纳文",
    "so": "索马里文",
    "sq": "阿尔巴尼亚文",
    "sr": "塞尔维亚文",
    "su": "巽他文",
    "sw": "斯瓦希里文",
    "ta": "泰米尔文",
    "te": "泰卢固文",
    "tg": "塔吉克文",
    "tk": "土库曼文",
    "tl": "菲律宾文",
    "tt": "鞑靼文",
    "uz": "乌兹别克文",
    "yi": "意第绪文",
    "yo": "约鲁巴文",
    "eo": "世界语",
    "ga": "爱尔兰文",
    "lg": "卢干达文",
    "nb": "书面挪威文",
    "st": "南索托文",
    "tn": "茨瓦纳文",
    "ts": "聪加文",
    "xh": "科萨文",
    "zu": "祖鲁文",
}

# Active live mode is intentionally limited to this small, predictable set.
# Keep the larger compatibility table above for archived records, but do not
# advertise or export long-tail languages until they are enabled again.
SUPPORTED_LANGUAGE_LABELS = {"zh": "中文", "en": "英文", "de": "德文"}


def language_label(code: str | None) -> str:
    """Return a stable Chinese label for a Whisper language code.

    Unknown codes are still shown rather than silently falling back to English;
    this matters for the user's original-language line when a meeting uses a
    language not in the compact label table above.
    """

    normalized = (code or "").strip().casefold()
    if normalized in LANGUAGE_LABELS:
        return LANGUAGE_LABELS[normalized]
    if normalized:
        return f"未知语言（{normalized}）"
    return "未知语言"


SessionState = Literal[
    "starting",
    "recording",
    "finalizing",
    "refining",
    "refinement_error",
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
    segment_revision: int = 0
    recognition_stage: Literal["fast", "refined"] = "refined"
    translation_status: Literal[
        "pending", "ready", "not_needed", "unsupported", "failed"
    ] = "ready"
    segment_id: str = ""
    revision: int = 1

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Utterance":
        values = dict(payload)
        values.setdefault("segment_revision", 0)
        values.setdefault("recognition_stage", "refined")
        values.setdefault("translation_status", "ready")
        values.setdefault("revision", 1)
        values.setdefault("segment_id", "")
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
    audio_packets_dropped: int = 0
    audio_packets_out_of_order: int = 0
    audio_samples_received: int = 0
    audio_level: float = 0.0
    pending_refinements: int = 0
    failed_refinements: int = 0
    owner_id: str = "local"
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

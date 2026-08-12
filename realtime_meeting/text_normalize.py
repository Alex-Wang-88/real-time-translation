"""Text normalization used for the Chinese source line.

Whisper sometimes emits Traditional Chinese even when the meeting language is
Chinese.  The Web client and exported originals are specified as
Simplified Chinese, so normalize the source before it is stored or rendered.
OpenCC provides the complete conversion table; the small fallback keeps the
application safe if an old environment has not installed the optional wheel
yet (the project declares it as a normal dependency).
"""

from __future__ import annotations

try:
    from opencc import OpenCC

    _T2S = OpenCC("t2s")
except Exception:  # pragma: no cover - only used by an incomplete install
    _T2S = None


_FALLBACK_T2S = str.maketrans(
    {
        "這": "这",
        "個": "个",
        "那": "那",
        "來": "来",
        "試": "试",
        "給": "给",
        "對": "对",
        "會": "会",
        "議": "议",
        "紀": "纪",
        "要": "要",
        "國": "国",
        "臺": "台",
        "裏": "里",
        "裡": "里",
        "麥": "麦",
        "克": "克",
        "風": "风",
        "權": "权",
        "許": "许",
        "靜": "静",
        "音": "音",
        "量": "量",
    }
)


def simplify_chinese(text: str) -> str:
    """Convert Traditional Chinese characters to Simplified Chinese."""

    if not text:
        return text
    if _T2S is not None:
        return _T2S.convert(text)
    return text.translate(_FALLBACK_T2S)

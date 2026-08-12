from __future__ import annotations

try:
    from opencc import OpenCC

    _converter = OpenCC("t2s")
except Exception:  # pragma: no cover - optional audio install
    _converter = None

_FALLBACK = str.maketrans(
    {
        "這": "这", "個": "个", "來": "来", "試": "试", "給": "给",
        "對": "对", "會": "会", "議": "议", "紀": "纪", "國": "国",
        "臺": "台", "裏": "里", "裡": "里", "麥": "麦", "風": "风",
        "權": "权", "許": "许", "靜": "静",
    }
)


def simplify_chinese(text: str) -> str:
    if not text:
        return text
    return _converter.convert(text) if _converter is not None else text.translate(_FALLBACK)


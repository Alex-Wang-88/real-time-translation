from __future__ import annotations

from .models import Utterance, language_label
from .text_normalize import simplify_chinese


def timestamp(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def source_line(item: Utterance) -> str:
    language = language_label(item.language)
    text = simplify_chinese(item.text) if item.language == "zh" else item.text
    return (
        f"[{timestamp(item.start)} - {timestamp(item.end)}] "
        f"演讲人{item.speaker_id}（{language}）：“{text}”"
    )


def translation_line(item: Utterance) -> str:
    return (
        f"[{timestamp(item.start)} - {timestamp(item.end)}] "
        f"演讲人{item.speaker_id}（中文翻译）：“{item.translation_zh}”"
    )


def english_line(item: Utterance) -> str:
    """Compatibility alias retained for older callers.

    The returned line is intentionally the Chinese translation line; the
    application no longer produces English as its target language.
    """

    return translation_line(item)


def paired_text(item: Utterance) -> str:
    return f"{source_line(item)}\n{translation_line(item)}"

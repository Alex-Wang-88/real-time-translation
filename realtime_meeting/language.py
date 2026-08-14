from __future__ import annotations

import re
from dataclasses import dataclass

SUPPORTED_LANGUAGE_CODES = frozenset({"zh", "en", "de"})

_CJK = re.compile(r"[\u3400-\u9fff]")
_GERMAN = re.compile(r"[äöüßÄÖÜ]|\b(?:und|aber|nicht|ist|sind|wir|ich|das|der|die|ein|eine|mit|für|auf|auch|dass|haben|wird|morgen|danke|guten|müssen|hallo|bitte|heute|treffen|willkommen)\b", re.I)
_ENGLISH = re.compile(r"\b(?:and|but|not|is|are|we|i|it|the|a|an|with|for|on|also|that|have|will|hello|good|morning|thanks|thank|please|yes|no|today|meeting|ready|everyone|welcome|you|your|this|there|can|could|would)\b", re.I)


@dataclass(frozen=True, slots=True)
class LanguageGuess:
    code: str
    confidence: float


def normalize_language_code(code: str | None) -> str | None:
    normalized = (code or "").strip().casefold().split("-")[0].split("_")[0]
    if normalized in {"cmn", "yue", "zh-cn", "zh-tw"}:
        return "zh"
    return normalized if normalized in SUPPORTED_LANGUAGE_CODES else None


class MultilingualDetector:
    def __init__(self) -> None:
        self._lingua = None
        try:
            from lingua import Language, LanguageDetectorBuilder

            self._lingua = LanguageDetectorBuilder.from_languages(
                Language.CHINESE, Language.ENGLISH, Language.GERMAN
            ).build()
        except Exception:
            self._lingua = None

    def detect(
        self,
        text: str,
        *,
        previous: str | None = None,
        whisper_language: str | None = None,
        whisper_confidence: float = 0.0,
    ) -> LanguageGuess:
        value = text.strip()
        if not value:
            return LanguageGuess(normalize_language_code(previous) or "zh", 0.0)
        hint = normalize_language_code(whisper_language)
        if hint and whisper_confidence >= 0.65 and (hint != "zh" or _CJK.search(value)):
            return LanguageGuess(hint, min(0.99, whisper_confidence))
        if _CJK.search(value):
            cjk = len(_CJK.findall(value))
            return LanguageGuess("zh", min(0.99, 0.55 + cjk / max(20, len(value))))
        german_hits = len(_GERMAN.findall(value))
        english_hits = len(_ENGLISH.findall(value))
        if german_hits > english_hits:
            return LanguageGuess("de", min(0.98, 0.55 + german_hits * 0.08))
        if english_hits > german_hits:
            return LanguageGuess("en", min(0.98, 0.55 + english_hits * 0.08))
        if self._lingua is not None:
            try:
                detected = self._lingua.detect_language_of(value)
                name = getattr(detected, "name", "")
                code = {"CHINESE": "zh", "ENGLISH": "en", "GERMAN": "de"}.get(name)
                if code:
                    return LanguageGuess(code, 0.65)
            except Exception:
                # Optional detector failures fall through to explicit language
                # hints and script-based heuristics below.
                self._lingua = None
        if hint:
            return LanguageGuess(hint, 0.5)
        prev = normalize_language_code(previous)
        return LanguageGuess(prev or "en", 0.25)

    def split_clauses(self, text: str) -> list[str]:
        pieces = [piece.strip() for piece in re.split(r"(?<=[。！？.!?])\s+|(?<=[。！？.!?])", text) if piece.strip()]
        return pieces or [text.strip()]

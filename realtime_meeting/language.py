from __future__ import annotations

import re
from dataclasses import dataclass


_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_LANGUAGE_CODE_RE = re.compile(r"^[a-z]{2,3}$", re.I)
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_HEBREW_RE = re.compile(r"[\u0590-\u05ff]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
_BENGALI_RE = re.compile(r"[\u0980-\u09ff]")
_THAI_RE = re.compile(r"[\u0e00-\u0e7f]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
_HIRAGANA_KATAKANA_RE = re.compile(r"[\u3040-\u30ff]")
_GREEK_RE = re.compile(r"[\u0370-\u03ff]")
_GERMAN_MARKERS = re.compile(
    r"[äöüßÄÖÜ]|\b(?:und|aber|nicht|ist|sind|wir|ich|das|der|die|ein|eine|mit|für|auf|auch|dass|haben|wird|egal|morgen|danke|guten|müssen|hallo|bitte|ja|nein|heute|treffen|bereit|willkommen|vielen)\b",
    re.I,
)
_ENGLISH_MARKERS = re.compile(
    r"\b(?:and|but|not|is|are|we|i|the|a|an|with|for|on|also|that|have|will|hello|good|morning|thanks|thank|please|yes|no|today|meeting|ready|everyone|welcome|you|your|this|there|can|could|would)\b",
    re.I,
)
_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")
_GERMAN_WORDS = frozenset(
    "und aber nicht ist sind wir ich das der die ein eine mit für auf auch dass haben wird "
    "egal morgen danke guten müssen hallo bitte ja nein heute treffen bereit willkommen vielen dank"
    .casefold()
    .split()
)
_ENGLISH_WORDS = frozenset(
    "and but not is are we the an with for on also that have will hello good morning thanks thank "
    "please yes no today meeting ready everyone welcome you your this there can could would"
    .casefold()
    .split()
)
# Only use unambiguous words for an intra-sentence switch.  ``meeting`` and
# ``ready`` are frequent German loanwords, so they remain useful for overall
# detection but must not split ``Wir besprechen das Meeting heute`` into an
# English clause.
_GERMAN_SWITCH_WORDS = frozenset(
    "aber nicht sind wir ich das der die ein eine für morgen danke guten müssen hallo bitte "
    "ja nein heute treffen willkommen vielen dank"
    .casefold()
    .split()
)
_ENGLISH_SWITCH_WORDS = frozenset(
    "hello good morning thanks thank please yes no today everyone welcome you your this there "
    "can could would"
    .casefold()
    .split()
)
_GERMAN_HINT_WORDS = frozenset(
    "aber nicht sind wir ich das der die ein eine mit für auf auch dass haben wird egal morgen "
    "danke guten müssen hallo bitte ja nein heute treffen bereit willkommen vielen dank"
    .casefold()
    .split()
)
_ENGLISH_HINT_WORDS = frozenset(
    "and but not is are we i the a an with for on also that have will hello good morning thanks "
    "thank please yes no today everyone welcome you your this there can could would okay"
    .casefold()
    .split()
)
_SPANISH_WORDS = frozenset(
    "tienes ganas gracias hola buenos buenas días dias que los las una uno para esto esta"
    .casefold()
    .split()
)
_PORTUGUESE_WORDS = frozenset(
    "então entao vamos não nao você voce vocês voces obrigado obrigada olá ola uma isso lá la"
    .casefold()
    .split()
)
_ITALIAN_WORDS = frozenset(
    "farlo grazie buongiorno ciao non sono questo questa perché perche allora"
    .casefold()
    .split()
)
_FRENCH_WORDS = frozenset(
    "bonjour merci vous nous avec pour les une cette pourquoi"
    .casefold()
    .split()
)
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;.,，])\s*|(?<=[。！？!?；;.,，])")


@dataclass(frozen=True, slots=True)
class LanguageGuess:
    code: str
    confidence: float


def _script_language(text: str, hint: str | None) -> LanguageGuess | None:
    """Recognize scripts that the compact three-language Lingua model lacks."""

    if _HIRAGANA_KATAKANA_RE.search(text):
        return LanguageGuess("ja", 0.92)
    if _HANGUL_RE.search(text):
        return LanguageGuess("ko", 0.92)
    if _ARABIC_RE.search(text):
        return LanguageGuess(hint if hint in {"ar", "fa", "ur", "ps"} else "ar", 0.88)
    if _HEBREW_RE.search(text):
        return LanguageGuess("he", 0.88)
    if _DEVANAGARI_RE.search(text):
        return LanguageGuess(hint if hint in {"hi", "mr", "ne", "sa"} else "hi", 0.88)
    if _BENGALI_RE.search(text):
        return LanguageGuess("bn", 0.88)
    if _THAI_RE.search(text):
        return LanguageGuess("th", 0.88)
    if _GREEK_RE.search(text):
        return LanguageGuess("el", 0.88)
    if _CYRILLIC_RE.search(text):
        if hint in {"ru", "uk", "bg", "sr", "mk", "be", "kk", "tg", "tt"}:
            return LanguageGuess(hint, 0.88)
        if re.search(r"[іїєґІЇЄҐ]", text):
            return LanguageGuess("uk", 0.84)
        return LanguageGuess("ru", 0.82)
    return None


def _latin_foreign_language(
    text: str,
    german_words: int,
    english_words: int,
) -> LanguageGuess | None:
    """Catch common Latin-script languages outside the fast Lingua set.

    This is deliberately conservative: clear English/German vocabulary wins,
    while one distinctive Spanish/Portuguese/Italian/French word or diacritic
    prevents a short clause from being labelled English merely because Lingua
    was configured for the original three meeting languages.
    """

    if german_words or english_words:
        return None
    foreign_words = [word.casefold() for word in re.findall(r"[A-Za-zÀ-ÿ]+", text)]
    if not foreign_words:
        return None
    candidates = (
        ("pt", _PORTUGUESE_WORDS),
        ("es", _SPANISH_WORDS),
        ("it", _ITALIAN_WORDS),
        ("fr", _FRENCH_WORDS),
    )
    for code, marker_words in candidates:
        hits = sum(word in marker_words for word in foreign_words)
        if hits:
            return LanguageGuess(code, min(0.86, 0.68 + 0.08 * hits))
    if re.search(r"[ãõçÃÕÇ]", text):
        return LanguageGuess("pt", 0.82)
    if re.search(r"[ñÑ¿¡]", text):
        return LanguageGuess("es", 0.82)
    return None


class TrilingualDetector:
    def __init__(self) -> None:
        from lingua import Language, LanguageDetectorBuilder

        self._mapping = {
            Language.CHINESE: "zh",
            Language.ENGLISH: "en",
            Language.GERMAN: "de",
        }
        self._detector = (
            LanguageDetectorBuilder.from_languages(*self._mapping)
            .with_minimum_relative_distance(0.05)
            .build()
        )

    def detect(
        self,
        text: str,
        previous: str | None = None,
        *,
        whisper_language: str | None = None,
        whisper_confidence: float | None = None,
    ) -> LanguageGuess:
        """Classify a short clause while keeping Whisper's segment hint.

        Whisper detects a language for the complete audio window, while the
        UI can display smaller clauses from that window.  For an unusual
        German word (a name, product term, or ASR spelling) Lingua may call
        the isolated text English.  The Whisper hint is therefore used only
        for low-evidence conflicts; clear lexical evidence still wins so
        genuine English/German code switching remains visible.
        """
        text = text.strip()
        if not text:
            return LanguageGuess(previous or "zh", 0.0)
        hint = whisper_language.casefold().strip() if whisper_language else None
        if hint and not _LANGUAGE_CODE_RE.fullmatch(hint):
            hint = None
        try:
            hint_confidence = float(whisper_confidence or 0.0)
        except (TypeError, ValueError):
            hint_confidence = 0.0
        scripted = _script_language(text, hint)
        if scripted is not None:
            return scripted
        cjk_count = len(_CJK_RE.findall(text))
        letter_count = sum(ch.isalpha() for ch in text)
        if cjk_count >= 2 or (cjk_count and cjk_count / max(letter_count, 1) >= 0.3):
            return LanguageGuess("zh", min(0.99, 0.75 + cjk_count / 40))

        # Whisper can return a one-word segment at a language boundary (for
        # example ``Hello`` immediately after German). A short-text fallback
        # that blindly inherits ``previous`` makes those switches invisible.
        # Resolve clear vocabulary before considering the neighbouring turn.
        words = [word.casefold() for word in _WORD_RE.findall(text)]
        german_words = sum(word in _GERMAN_WORDS for word in words)
        english_words = sum(word in _ENGLISH_WORDS for word in words)
        german_hint_words = sum(word in _GERMAN_HINT_WORDS for word in words)
        english_hint_words = sum(word in _ENGLISH_HINT_WORDS for word in words)
        if german_words > english_words:
            return LanguageGuess("de", min(0.98, 0.72 + 0.06 * german_words))
        if english_words > german_words:
            # German speech commonly contains one English loanword (meeting,
            # ready, project, etc.).  If Whisper strongly identifies the
            # containing audio as German and there is no distinctive English
            # phrase, keep that loanword in the German source line.
            if (
                hint == "de"
                and hint_confidence >= 0.75
                and english_hint_words == 0
                and german_hint_words == 0
            ):
                return LanguageGuess("de", min(0.90, hint_confidence))
            return LanguageGuess("en", min(0.98, 0.72 + 0.06 * english_words))

        de_hits = len(_GERMAN_MARKERS.findall(text))
        en_hits = len(_ENGLISH_MARKERS.findall(text))
        foreign = _latin_foreign_language(text, german_words, english_words)
        if foreign is not None:
            return foreign

        # Lingua is deliberately limited to the three meeting languages for
        # fast code-switch decisions.  For every other Whisper language, use
        # its segment-level prediction when the clause has no contradictory
        # German/English evidence.  This keeps Cyrillic, Spanish, Portuguese,
        # Japanese, etc. in their real source-language line instead of
        # labelling them as English.
        if (
            hint
            and hint not in {"zh", "en", "de"}
            and hint_confidence >= 0.55
            and not (de_hits or en_hits or german_words or english_words)
        ):
            return LanguageGuess(hint, min(0.96, hint_confidence))

        values = self._detector.compute_language_confidence_values(text)
        if values:
            best = values[0]
            code = self._mapping.get(best.language)
            # Lingua is reliable for short Latin phrases such as ``Hello``
            # and ``Morgen``; use it before a previous-language fallback.
            if code and best.value >= 0.60:
                # A segment-level Whisper hint is especially useful for a
                # one-word German ASR result such as a name or a technical
                # term.  Do not override a strong text decision: this keeps
                # an actual English clause inside a German audio window in
                # English.  The 0.80 threshold is deliberately conservative
                # and only repairs low-evidence conflicts.
                if (
                    hint
                    and hint != code
                    and hint_confidence >= 0.75
                    and best.value < 0.80
                    and not (de_hits or en_hits or german_words or english_words)
                ):
                    return LanguageGuess(hint, min(0.90, hint_confidence))
                return LanguageGuess(code, float(best.value))
        if de_hits > en_hits:
            return LanguageGuess("de", 0.55)
        if en_hits > de_hits:
            return LanguageGuess("en", 0.55)
        # Only genuinely ambiguous short backchannels inherit the previous
        # language. Longer text must be classified independently.
        if previous and letter_count <= 8:
            return LanguageGuess(previous, 0.45)
        if values:
            best = values[0]
            code = self._mapping.get(best.language)
            if code and best.value >= 0.40:
                return LanguageGuess(code, float(best.value))
        return LanguageGuess(previous or "en", 0.35)

    def split_clauses(self, text: str) -> list[str]:
        clauses: list[str] = []
        for part in _CLAUSE_SPLIT_RE.split(text.strip()):
            part = part.strip()
            if not part:
                continue
            runs = re.split(
                r"(?<=[\u3400-\u9fff])\s*(?=[A-Za-zÄÖÜäöüß])|"
                r"(?<=[A-Za-zÄÖÜäöüß])\s*(?=[\u3400-\u9fff])",
                part,
            )
            for run in runs:
                run = run.strip()
                if not run:
                    continue
                # German and English share the Latin script, so a script
                # boundary alone cannot split ``Guten Morgen Good morning``.
                # Use explicit vocabulary transitions as additional
                # code-switch boundaries while leaving unknown words in the
                # surrounding run.
                current: list[str] = []
                current_code: str | None = None
                tokens = run.split()
                token_codes: list[str | None] = []
                for token in tokens:
                    word_match = _WORD_RE.search(token)
                    word = word_match.group(0).casefold() if word_match else ""
                    token_codes.append(
                        "de" if word in _GERMAN_SWITCH_WORDS else
                        "en" if word in _ENGLISH_SWITCH_WORDS else
                        None
                    )
                for index, token in enumerate(tokens):
                    code = token_codes[index]
                    next_code = next(
                        (candidate for candidate in token_codes[index + 1 :] if candidate),
                        None,
                    )
                    # A single loanword is not a language switch.  For
                    # example, ``Wir besprechen das Meeting heute`` must stay
                    # German, while ``Guten Morgen Good morning`` should split
                    # because the opposing language appears in a run of at
                    # least two recognised words.
                    switch = bool(code and current_code and code != current_code and next_code == code)
                    if switch and current:
                        clauses.append(" ".join(current).strip())
                        current = []
                        current_code = code
                    elif code and not current_code:
                        current_code = code
                    current.append(token)
                if current:
                    clauses.append(" ".join(current).strip())
        return clauses or [text.strip()]

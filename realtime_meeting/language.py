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
_GURMUKHI_RE = re.compile(r"[\u0a00-\u0a7f]")
_GUJARATI_RE = re.compile(r"[\u0a80-\u0aff]")
_TAMIL_RE = re.compile(r"[\u0b80-\u0bff]")
_TELUGU_RE = re.compile(r"[\u0c00-\u0c7f]")
_KANNADA_RE = re.compile(r"[\u0c80-\u0cff]")
_MALAYALAM_RE = re.compile(r"[\u0d00-\u0d7f]")
_SINHALA_RE = re.compile(r"[\u0d80-\u0dff]")
_THAI_RE = re.compile(r"[\u0e00-\u0e7f]")
_LAO_RE = re.compile(r"[\u0e80-\u0eff]")
_TIBETAN_RE = re.compile(r"[\u0f00-\u0fff]")
_MYANMAR_RE = re.compile(r"[\u1000-\u109f]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
_HIRAGANA_KATAKANA_RE = re.compile(r"[\u3040-\u30ff]")
_GREEK_RE = re.compile(r"[\u0370-\u03ff]")
_ARMENIAN_RE = re.compile(r"[\u0530-\u058f]")
_GEORGIAN_RE = re.compile(r"[\u10a0-\u10ff]")
_ETHIOPIC_RE = re.compile(r"[\u1200-\u137f]")
_KHMER_RE = re.compile(r"[\u1780-\u17ff]")
_MONGOLIAN_RE = re.compile(r"[\u1800-\u18af]")
_GERMAN_MARKERS = re.compile(
    r"[äöüßÄÖÜ]|\b(?:und|aber|nicht|ist|sind|wir|ich|das|der|die|ein|eine|mit|für|auf|auch|dass|haben|wird|egal|morgen|danke|guten|müssen|hallo|bitte|ja|nein|heute|treffen|bereit|willkommen|vielen)\b",
    re.I,
)
_ENGLISH_MARKERS = re.compile(
    r"\b(?:and|but|not|is|are|we|i|it|the|a|an|with|for|on|also|that|have|will|hello|good|morning|thanks|thank|please|yes|no|today|meeting|ready|everyone|welcome|you|your|this|there|can|could|would|very|easy|to|sell)\b",
    re.I,
)
# Include common Latin extension blocks so accents are not discarded before
# text-level language detection.
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ž]+")
_GERMAN_WORDS = frozenset(
  "und aber nicht ist sind wir ich das der die ein eine mit für auf auch dass haben wird "
  "egal morgen danke guten müssen hallo bitte ja nein heute treffen bereit willkommen vielen dank "
    "den seit sturm deutschland millionen existenzen stehen sehr einfach zu sagen passierte möchte "
    "könnte sollte wieso warum welche welcher welches"
    .casefold()
    .split()
)
_ENGLISH_WORDS = frozenset(
    "and but not is are we i it the an with for on also that have will hello good morning thanks thank "
    "please yes no today meeting ready everyone welcome you your this there can could would very easy to sell"
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
  "danke guten müssen hallo bitte ja nein heute treffen bereit willkommen vielen dank "
    "den seit sturm deutschland millionen existenzen stehen sehr einfach zu sagen passierte möchte "
    "könnte sollte wieso warum welche welcher welches"
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
    "bonjour merci vous nous avec pour les une cette pourquoi comment allez "
    "être suis sont dans des qui que quoi mais oui non aujourd'hui"
    .casefold()
    .split()
)
_DUTCH_WORDS = frozenset(
    "hallo goedemorgen dank bedankt en niet een het de van voor met wij zijn "
    "morgen vandaag hoe gaat u jullie"
    .casefold()
    .split()
)
_SWEDISH_WORDS = frozenset(
    "hej tack god morgon och inte det den en ett vi är jag du hur mår idag"
    .casefold()
    .split()
)
_DANISH_WORDS = frozenset(
    "hej tak godmorgen og ikke det den en et vi er jeg du hvordan i dag"
    .casefold()
    .split()
)
_NORWEGIAN_WORDS = frozenset(
    "hei takk god morgen og ikke det den en et vi er jeg du hvordan i dag"
    .casefold()
    .split()
)
_TURKISH_WORDS = frozenset(
    "merhaba teşekkür teşekkürler günaydın ve değil bir bu için nasıl bugün"
    .casefold()
    .split()
)
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;.,，])\s*|(?<=[。！？!?；;.,，])")

# Keep the live product focused on the three languages currently supported by
# the UI. Whisper may report Cantonese as ``yue``; it is normalized to Chinese
# so Chinese dialects stay in the same recognition and translation path.
SUPPORTED_LANGUAGE_CODES = frozenset({"zh", "en", "de"})
_CHINESE_LANGUAGE_ALIASES = frozenset(
    {"zh", "zho", "cmn", "yue", "zh-cn", "zh-tw", "zh-hk"}
)


@dataclass(frozen=True, slots=True)
class LanguageGuess:
    code: str
    confidence: float


def normalize_language_code(code: str | None) -> str | None:
    normalized = (code or "").strip().casefold().replace("_", "-")
    if normalized in _CHINESE_LANGUAGE_ALIASES or normalized.startswith("zh-"):
        return "zh"
    if normalized in SUPPORTED_LANGUAGE_CODES:
        return normalized
    return None


def _script_language(text: str, hint: str | None) -> LanguageGuess | None:
    """Recognize writing systems before statistical language detection.

    Script detection is deterministic and is especially valuable for short
    utterances where Whisper's language probability is underdetermined. The
    hint is used only to distinguish languages sharing a script; it can never
    force an English label onto another script.
    """

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
        return LanguageGuess(hint if hint in {"bn", "as"} else "bn", 0.88)
    if _GURMUKHI_RE.search(text):
        return LanguageGuess("pa", 0.88)
    if _GUJARATI_RE.search(text):
        return LanguageGuess("gu", 0.88)
    if _TAMIL_RE.search(text):
        return LanguageGuess("ta", 0.88)
    if _TELUGU_RE.search(text):
        return LanguageGuess("te", 0.88)
    if _KANNADA_RE.search(text):
        return LanguageGuess("kn", 0.88)
    if _MALAYALAM_RE.search(text):
        return LanguageGuess("ml", 0.88)
    if _SINHALA_RE.search(text):
        return LanguageGuess("si", 0.88)
    if _LAO_RE.search(text):
        return LanguageGuess("lo", 0.88)
    if _THAI_RE.search(text):
        return LanguageGuess("th", 0.88)
    if _TIBETAN_RE.search(text):
        return LanguageGuess("bo", 0.88)
    if _MYANMAR_RE.search(text):
        return LanguageGuess("my", 0.88)
    if _GREEK_RE.search(text):
        return LanguageGuess("el", 0.88)
    if _ARMENIAN_RE.search(text):
        return LanguageGuess("hy", 0.88)
    if _GEORGIAN_RE.search(text):
        return LanguageGuess("ka", 0.88)
    if _ETHIOPIC_RE.search(text):
        return LanguageGuess("am", 0.88)
    if _KHMER_RE.search(text):
        return LanguageGuess("km", 0.88)
    if _MONGOLIAN_RE.search(text):
        return LanguageGuess("mn", 0.88)
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
    foreign_words = [word.casefold() for word in _WORD_RE.findall(text)]
    if not foreign_words:
        return None
    candidates = (
        ("pt", _PORTUGUESE_WORDS),
        ("es", _SPANISH_WORDS),
        ("it", _ITALIAN_WORDS),
        ("fr", _FRENCH_WORDS),
        ("nl", _DUTCH_WORDS),
        ("sv", _SWEDISH_WORDS),
        ("da", _DANISH_WORDS),
        ("no", _NORWEGIAN_WORDS),
        ("tr", _TURKISH_WORDS),
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


class MultilingualDetector:
    """Detect the three live languages supported by the current product.

    Lingua still supplies a text-level second opinion, while Whisper supplies
    the audio-level hint. Only Chinese, English, and German can leave this
    class as usable results; unsupported guesses become ``unknown`` instead
    of appearing as a confident but misleading language label.
    """

    def __init__(self) -> None:
        from lingua import Language, LanguageDetectorBuilder

        self._mapping = {
            language: str(language.iso_code_639_1).split(".")[-1].casefold()
            for language in Language.all()
        }
        self._detector = (
            LanguageDetectorBuilder.from_all_languages()
            .with_minimum_relative_distance(0.05)
            .build()
        )
        # Lingua loads model data lazily. Do that while the UI is still in its
        # model-loading state instead of on the first live utterance.
        self._detector.compute_language_confidence_values("hello guten morgen")

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
            return LanguageGuess(normalize_language_code(previous) or "zh", 0.0)
        hint = normalize_language_code(whisper_language)
        raw_hint = (whisper_language or "").casefold().strip()
        if raw_hint and not _LANGUAGE_CODE_RE.fullmatch(raw_hint):
            hint = None
        previous_code = normalize_language_code(previous)
        try:
            hint_confidence = float(whisper_confidence or 0.0)
        except (TypeError, ValueError):
            hint_confidence = 0.0
        scripted = _script_language(text, hint)
        if scripted is not None:
            return (
                scripted
                if scripted.code in SUPPORTED_LANGUAGE_CODES
                else LanguageGuess("unknown", scripted.confidence)
            )
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
                and english_words <= 1
                and english_hint_words == 0
                and german_hint_words == 0
            ):
                return LanguageGuess("de", min(0.90, hint_confidence))
            return LanguageGuess("en", min(0.98, 0.72 + 0.06 * english_words))

        de_hits = len(_GERMAN_MARKERS.findall(text))
        en_hits = len(_ENGLISH_MARKERS.findall(text))
        foreign = _latin_foreign_language(text, german_words, english_words)
        if foreign is not None:
            return (
                foreign
                if foreign.code in SUPPORTED_LANGUAGE_CODES
                else LanguageGuess("unknown", foreign.confidence)
            )

        # Full-language Lingua is used as a text-level second opinion after
        # deterministic script and switch-word checks.
        values = self._detector.compute_language_confidence_values(text)
        if values:
            best = values[0]
            code = self._mapping.get(best.language)
            if code in SUPPORTED_LANGUAGE_CODES and best.value >= 0.50:
                if (
                    hint in SUPPORTED_LANGUAGE_CODES
                    and hint != code
                    and hint_confidence >= 0.75
                    and not (de_hits or en_hits or german_words or english_words)
                ):
                    return LanguageGuess(hint, min(0.92, hint_confidence))
                return LanguageGuess(code, float(best.value))

            # A strong supported Whisper hint is safer than presenting the
            # best unsupported Lingua result. This is useful for Chinese
            # dialects whose ASR text may contain few CJK characters.
            if (
                hint in SUPPORTED_LANGUAGE_CODES
                and hint_confidence >= 0.75
                and not (de_hits or en_hits or german_words or english_words)
            ):
                return LanguageGuess(hint, min(0.92, hint_confidence))

        # Whisper remains useful for short or noisy clauses, but only the
        # three supported language hints are allowed through.
        if (
            hint in SUPPORTED_LANGUAGE_CODES
            and hint_confidence >= 0.35
            and not (de_hits or en_hits or german_words or english_words)
        ):
            return LanguageGuess(hint, min(0.96, hint_confidence))

        if de_hits > en_hits:
            return LanguageGuess("de", 0.55)
        if en_hits > de_hits:
            return LanguageGuess("en", 0.55)
        if values:
            best = values[0]
            code = self._mapping.get(best.language)
            # Longer clauses should not inherit a previous language just
            # because the statistical confidence is modest.
            if code in SUPPORTED_LANGUAGE_CODES and (best.value >= 0.35 or letter_count > 8):
                return LanguageGuess(code, float(best.value))
        # Only genuinely ambiguous short backchannels inherit the previous
        # language.
        if previous_code and letter_count <= 8:
            return LanguageGuess(previous_code, 0.45)
        return LanguageGuess("unknown", 0.25)

    def split_clauses(self, text: str) -> list[str]:
        clauses: list[str] = []
        for part in _CLAUSE_SPLIT_RE.split(text.strip()):
            part = part.strip()
            if not part:
                continue
            runs = re.split(
                r"(?<=[\u3400-\u9fff])\s*(?=[A-Za-zÀ-ÖØ-öø-ÿĀ-ž])|"
                r"(?<=[A-Za-zÀ-ÖØ-öø-ÿĀ-ž])\s*(?=[\u3400-\u9fff])",
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


# Backwards-compatible import name used by older callers and saved plugins.
TrilingualDetector = MultilingualDetector

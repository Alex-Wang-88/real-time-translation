from realtime_meeting.language import TrilingualDetector


def test_long_english_text_does_not_inherit_chinese() -> None:
    detector = TrilingualDetector()
    assert detector.detect("communist in history", previous="zh").code == "en"


def test_short_backchannel_can_inherit_previous_language() -> None:
    detector = TrilingualDetector()
    assert detector.detect("okay", previous="en").code == "en"


def test_clear_short_words_override_previous_language_at_a_switch() -> None:
    detector = TrilingualDetector()
    assert detector.detect("Hello", previous="de").code == "en"
    assert detector.detect("Hallo", previous="en").code == "de"
    assert detector.detect("Good", previous="de").code == "en"
    assert detector.detect("Guten", previous="en").code == "de"


def test_latin_code_switch_is_split_into_independent_translation_units() -> None:
    detector = TrilingualDetector()
    assert detector.split_clauses("Guten Morgen Good morning") == [
        "Guten Morgen",
        "Good morning",
    ]


def test_whisper_german_hint_repairs_unknown_short_asr_word() -> None:
    detector = TrilingualDetector()
    # Names and technical terms have little textual evidence.  Whisper's
    # high-confidence German prediction for the containing audio segment must
    # prevent Lingua from labelling the isolated token as English.
    assert (
        detector.detect(
            "Kamianism.",
            previous="de",
            whisper_language="de",
            whisper_confidence=0.99,
        ).code
        == "de"
    )


def test_clear_english_clause_still_overrides_german_segment_hint() -> None:
    detector = TrilingualDetector()
    assert (
        detector.detect(
            "It's very easy to sell.",
            whisper_language="de",
            whisper_confidence=0.99,
        ).code
        == "en"
    )


def test_german_loanword_does_not_create_false_english_clause() -> None:
    detector = TrilingualDetector()
    assert detector.split_clauses("Wir besprechen das Meeting heute") == [
        "Wir besprechen das Meeting heute"
    ]
    assert (
        detector.detect(
            "Meeting",
            whisper_language="de",
            whisper_confidence=0.99,
        ).code
        == "de"
    )
    assert (
        detector.detect(
            "the meeting",
            whisper_language="de",
            whisper_confidence=0.99,
        ).code
        == "en"
    )


def test_non_trilingual_whisper_hints_are_preserved() -> None:
    detector = TrilingualDetector()
    assert detector.detect("паньями", whisper_language="ru", whisper_confidence=0.95).code == "ru"
    assert detector.detect("Tienes ganas.", whisper_language="es", whisper_confidence=0.95).code == "es"
    assert detector.detect("Então, vamos lá.", whisper_language="pt", whisper_confidence=0.95).code == "pt"


def test_script_and_foreign_markers_repair_an_english_whisper_hint() -> None:
    detector = TrilingualDetector()
    assert detector.detect("Ой, мой голос", whisper_language="en", whisper_confidence=0.9).code == "ru"
    assert detector.detect("Tienes ganas.", whisper_language="en", whisper_confidence=0.9).code == "es"
    assert detector.detect("e farlo", whisper_language="en", whisper_confidence=0.9).code == "it"
    assert detector.detect("João", whisper_language="en", whisper_confidence=0.9).code == "pt"

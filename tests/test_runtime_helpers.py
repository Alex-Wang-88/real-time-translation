from realtime_meeting.runtime import LiveChineseTranslator, is_boundary_duplicate


def test_forced_cut_fuzzy_suffix_is_deduplicated() -> None:
    assert is_boundary_duplicate(
        "from now on you don't have to pay any rent",
        "penny rent",
    )


def test_distinct_short_followup_is_not_deduplicated() -> None:
    assert not is_boundary_duplicate("the budget is approved", "next topic")


def test_live_translator_always_targets_simplified_chinese() -> None:
    class SentencePiece:
        def encode(self, text, out_type=str):
            return [text]

        def decode(self, tokens):
            return "你好"

    class Generated:
        hypotheses = [["zho_Hans", "你好"]]

    class Model:
        def __init__(self):
            self.batch = None

        def translate_batch(self, batch, **kwargs):
            self.batch = (batch, kwargs)
            return [Generated()]

    translator = object.__new__(LiveChineseTranslator)
    translator.sp = SentencePiece()
    translator.model = Model()
    assert translator.translate("Guten Morgen", "de") == "你好"
    assert translator.model.batch[0][0] == ["deu_Latn", "Guten Morgen", "</s>"]
    assert translator.model.batch[1]["target_prefix"] == [["zho_Hans"]]
    assert translator.translate("你好", "zh") == "你好"

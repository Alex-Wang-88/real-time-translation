from realtime_meeting.text_normalize import simplify_chinese


def test_traditional_chinese_source_is_converted_to_simplified() -> None:
    assert simplify_chinese("這段那個") == "这段那个"
    assert simplify_chinese("對啊我是這個不行啊") == "对啊我是这个不行啊"


def test_non_chinese_transcript_text_is_unchanged() -> None:
    assert simplify_chinese("Guten Morgen / Good morning") == "Guten Morgen / Good morning"

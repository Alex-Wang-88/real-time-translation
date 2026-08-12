import sys

import numpy as np
import pytest

from faster_whisper.tokenizer import _LANGUAGE_CODES

from realtime_meeting.runtime import (
    LiveChineseTranslator,
    LiveModelRuntime,
    NLLB_CODES,
    choose_device,
    is_boundary_duplicate,
)


def test_explicit_cuda_does_not_silently_fallback_to_cpu(monkeypatch):
    class FakeCTranslate2:
        @staticmethod
        def get_cuda_device_count():
            return 0

    monkeypatch.setitem(sys.modules, "ctranslate2", FakeCTranslate2)
    with pytest.raises(RuntimeError, match="CUDA"):
        choose_device("cuda")


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


def test_runtime_forces_multilingual_transcribe_task() -> None:
    class Info:
        language = "de"
        language_probability = 0.91

    class Segment:
        start = 0.0
        end = 1.0
        text = "Guten Morgen"

    class ASR:
        def __init__(self):
            self.kwargs = None

        def transcribe(self, audio, **kwargs):
            self.kwargs = kwargs
            return iter([Segment()]), Info()

    runtime = LiveModelRuntime("large-v3-turbo", "translation", "cpu")
    runtime.ready = True
    runtime.asr = ASR()
    runtime._recognize(np.zeros(16000, dtype=np.int16).tobytes(), "", None, False)
    assert runtime.asr.kwargs["task"] == "transcribe"
    assert runtime.asr.kwargs["language"] is None
    assert runtime.asr.kwargs["multilingual"] is True


def test_runtime_does_not_force_previous_language_into_whisper() -> None:
    class Info:
        language = "zh"
        language_probability = 0.91

    class Segment:
        start = 0.0
        end = 1.0
        text = "中文内容"

    class ASR:
        def __init__(self):
            self.kwargs = None

        def transcribe(self, audio, **kwargs):
            self.kwargs = kwargs
            return iter([Segment()]), Info()

    runtime = LiveModelRuntime("large-v3-turbo", "translation", "cpu")
    runtime.ready = True
    runtime.asr = ASR()
    runtime._recognize(
        np.zeros(16000, dtype=np.int16).tobytes(),
        "",
        None,
        False,
        language_hint="pt",
    )
    assert runtime.asr.kwargs["language"] is None


def test_funasr_primary_route_accepts_supported_language_without_whisper() -> None:
    class FunASR:
        def generate(self, **kwargs):
            assert kwargs["batch_size"] == 1
            assert kwargs["itn"] is True
            return [
                {
                    "text": "你好，世界",
                    "language": "zh",
                    "confidence": 0.93,
                }
            ]

    class Whisper:
        def transcribe(self, *_args, **_kwargs):
            raise AssertionError("supported FunASR input should not reach Whisper")

    runtime = LiveModelRuntime("funasr-nano", "translation", "cpu")
    runtime.ready = True
    runtime.fun_asr = FunASR()
    runtime.asr = Whisper()
    segments, language, confidence, model = runtime._recognize(
        np.zeros(16000, dtype=np.int16).tobytes(),
        "最近一句",
        None,
        False,
    )

    assert [segment.text for segment in segments] == ["你好，世界"]
    assert language == "zh"
    assert confidence == 0.93
    assert model == "funasr"


@pytest.mark.parametrize(
    ("refine_model", "refinement_enabled", "expected_loads"),
    [
        ("large-v3-turbo", True, 1),
        # The low-priority refine model is lazy-loaded after recording so it
        # cannot compete with realtime ASR during startup.
        ("large-v3", True, 1),
        ("large-v3", False, 1),
    ],
)
def test_runtime_deduplicates_identical_asr_weights(
    monkeypatch,
    refine_model: str,
    refinement_enabled: bool,
    expected_loads: int,
) -> None:
    import faster_whisper
    import realtime_meeting.runtime as runtime_module
    import resemblyzer

    loaded: list[str] = []

    class ASR:
        def __init__(self, name, **_kwargs):
            loaded.append(name)

    class Translator:
        def __init__(self, *_args, **_kwargs):
            pass

    class Encoder:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(faster_whisper, "WhisperModel", ASR)
    monkeypatch.setattr(runtime_module, "LiveChineseTranslator", Translator)
    monkeypatch.setattr(resemblyzer, "VoiceEncoder", Encoder)
    monkeypatch.setattr(LiveModelRuntime, "_warmup", lambda self: None)
    runtime = LiveModelRuntime(
        "large-v3-turbo", "translation", "cpu", refine_model, refinement_enabled
    )
    runtime.load()
    assert len(loaded) == expected_loads
    if refine_model == "large-v3-turbo":
        assert runtime.refine_asr is runtime.asr
    elif refinement_enabled:
        assert runtime.refine_asr is None


def test_nllb_mapping_covers_whispers_supported_languages_except_known_gaps() -> None:
    unsupported = {"br", "haw", "la"}
    assert unsupported.isdisjoint(NLLB_CODES)
    assert {code for code in _LANGUAGE_CODES if code not in unsupported} <= NLLB_CODES.keys()

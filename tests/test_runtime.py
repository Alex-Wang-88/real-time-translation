from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from realtime_meeting.runtime import MODEL_REVISIONS, LiveModelRuntime, OPUS_MT_TARGET_TAGS, opus_mt_repository


def test_runtime_load_uses_refinement_model_name(settings, monkeypatch) -> None:
    class FakeWhisperModel:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class DisabledFunAsrModel:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("disabled in unit test")

    monkeypatch.setattr("realtime_meeting.runtime.choose_device", lambda _requested: ("cpu", "int8"))
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.setitem(sys.modules, "funasr", types.SimpleNamespace(AutoModel=DisabledFunAsrModel))

    runtime = LiveModelRuntime(
        settings.asr_primary,
        settings.asr_fallback,
        settings.asr_refine,
        "cpu",
        refinement_enabled=True,
        translation_model_root=settings.translation_model_root,
    )
    runtime.load()

    assert runtime.ready is True
    assert runtime.asr_refine_name == settings.asr_refine
    assert not any(event.get("event") == "load_error" for event in runtime.metrics["model_events"])


def test_runtime_reports_resemblyzer_diarization(settings) -> None:
    runtime = LiveModelRuntime(
        settings.asr_primary,
        settings.asr_fallback,
        settings.asr_refine,
        "cpu",
        asr_autodownload=False,
    )

    runtime.diarization.device = "cpu"
    assert runtime.diarization.preflight() is True

    snapshot = runtime.capability_snapshot()["diarization"]
    assert snapshot["available"] is True
    assert snapshot["backend"] == "resemblyzer"
    assert snapshot["model"] == "resemblyzer/voice_encoder"
    assert snapshot["parameters"]["sample_rate"] == 16000
    assert runtime.metrics["fallback_count"] == 0


def test_opus_mt_repository_and_target_tags_are_explicit() -> None:
    assert opus_mt_repository("en") == "Helsinki-NLP/opus-mt-en-zh"
    assert opus_mt_repository("de") == "Helsinki-NLP/opus-mt-de-ZH"
    assert OPUS_MT_TARGET_TAGS == {"en": ">>cmn_Hans<<", "de": ">>zh_cn<<"}
    assert all(len(revision) == 40 for revision in MODEL_REVISIONS.values())


def test_whisper_receives_recent_context(settings) -> None:
    runtime = LiveModelRuntime(
        settings.asr_primary,
        settings.asr_fallback,
        settings.asr_refine,
        "cpu",
        translation_model_root=settings.translation_model_root,
    )

    class Info:
        language = "en"
        language_probability = 0.9

    class Model:
        def __init__(self) -> None:
            self.kwargs = {}

        def transcribe(self, _audio, **kwargs):
            self.kwargs = kwargs
            return iter(()), Info()

    model = Model()
    runtime._whisper(b"\x00\x00" * 20, model, prompt=runtime._asr_prompt("previous decision"))
    assert model.kwargs["initial_prompt"] == "最近内容：previous decision"
    assert model.kwargs["beam_size"] == settings.asr_realtime_beam_size
    assert model.kwargs["best_of"] == settings.asr_best_of
    assert model.kwargs["temperature"] == 0.0
    assert model.kwargs["log_prob_threshold"] == settings.asr_log_prob_threshold
    assert model.kwargs["no_speech_threshold"] == settings.asr_no_speech_threshold
    assert model.kwargs["compression_ratio_threshold"] == settings.asr_compression_ratio_threshold
    assert len(runtime._asr_prompt("x" * 1000) or "") <= 520


def test_whisper_filters_punctuation_only_segments(settings) -> None:
    runtime = LiveModelRuntime(
        settings.asr_primary,
        settings.asr_fallback,
        settings.asr_refine,
        "cpu",
        translation_model_root=settings.translation_model_root,
    )

    class Info:
        language = "en"
        language_probability = 0.9

    class Model:
        def transcribe(self, _audio, **_kwargs):
            return iter([SimpleNamespace(text="...", avg_logprob=-0.2, no_speech_prob=0.1, compression_ratio=1.0)]), Info()

    result = runtime._whisper_decode(b"\x00\x00" * 20, Model())
    assert result.text == ""
    assert runtime._discard_decode_result(result) == "empty"


def test_low_quality_decode_retries_without_poisoned_context(settings) -> None:
    runtime = LiveModelRuntime(
        settings.asr_primary,
        settings.asr_fallback,
        settings.asr_refine,
        "cpu",
        translation_model_root=settings.translation_model_root,
    )
    runtime.primary = object()
    calls = []

    class Info:
        language = "en"
        language_probability = 0.9

    class Model:
        def transcribe(self, _audio, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                segment = SimpleNamespace(text="repeated repeated", avg_logprob=-1.4, no_speech_prob=0.1, compression_ratio=3.2, temperature=0.0)
            else:
                segment = SimpleNamespace(text="clear speech", avg_logprob=-0.2, no_speech_prob=0.05, compression_ratio=1.1, temperature=0.2)
            return iter([segment]), Info()

    runtime.primary = Model()
    text, _guess, _confidence, _model, _source = runtime._recognize(
        b"\x00\x00" * 20,
        recent_text="previous decision",
    )
    assert text == "clear speech"
    assert len(calls) == 2
    assert calls[0]["beam_size"] == settings.asr_realtime_beam_size
    assert calls[1]["beam_size"] == 5
    assert calls[1]["best_of"] == 5
    assert calls[1]["temperature"] == settings.asr_retry_temperature
    assert calls[0]["initial_prompt"] == "最近内容：previous decision"
    assert calls[1]["initial_prompt"] is None
    assert runtime.metrics["asr_decode_retries"] == 1


def test_high_confidence_repetitive_hallucination_is_discarded(settings) -> None:
    runtime = LiveModelRuntime(
        settings.asr_primary,
        settings.asr_fallback,
        settings.asr_refine,
        "cpu",
        translation_model_root=settings.translation_model_root,
    )
    result = SimpleNamespace(
        text="看着" * 100,
        avg_logprob=-0.02,
        no_speech_prob=0.01,
        compression_ratio=31.57,
    )

    assert runtime._decode_needs_retry(result) is True
    assert runtime._discard_decode_result(result) == "repetitive_hallucination"
    assert runtime._asr_prompt(result.text) is None

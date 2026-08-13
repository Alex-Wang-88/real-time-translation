from __future__ import annotations

import sys
import types

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


def test_whisper_receives_bounded_hotwords_and_recent_context(settings) -> None:
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
    runtime._whisper(b"\x00\x00" * 20, model, prompt=runtime._asr_prompt("previous decision", "Codex, Luna"))
    assert model.kwargs["initial_prompt"] == "专业词和姓名：Codex, Luna\n最近内容：previous decision"
    assert len(runtime._asr_prompt("x" * 1000, "y" * 2000) or "") <= 1520

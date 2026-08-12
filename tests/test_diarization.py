from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from realtime_meeting.diarization import DiarizationEngine, SpeakerSegment, align_speakers
from realtime_meeting.models import Utterance


def _write_pcm_wav(path: Path, seconds: float = 2.0) -> None:
    sample_rate = 16_000
    samples = int(sample_rate * seconds)
    timeline = np.arange(samples, dtype=np.float32) / sample_rate
    waveform = (np.sin(2 * np.pi * 220 * timeline) * 0.25 * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(waveform.tobytes())


def test_resemblyzer_is_the_only_backend() -> None:
    engine = DiarizationEngine(required=True)

    assert engine.preflight() is True
    assert engine.backend == "resemblyzer"
    assert engine.model_name == "resemblyzer/voice_encoder"
    assert engine.capability_ready() is True
    assert engine.model_parameters()["cluster_threshold"] == 0.68


def test_missing_resemblyzer_does_not_switch_backend(monkeypatch) -> None:
    engine = DiarizationEngine(required=True)
    monkeypatch.setattr(engine, "_resemblyzer_available", lambda: False)

    assert engine.preflight() is False
    assert engine.backend == "resemblyzer"
    assert engine.status == "dependency_missing"
    assert engine.error


def test_resemblyzer_produces_segments_from_local_audio(tmp_path: Path) -> None:
    pytest.importorskip("librosa")
    audio = tmp_path / "speech.wav"
    _write_pcm_wav(audio)
    engine = DiarizationEngine()
    engine.ready = True
    engine.backend = "resemblyzer"
    # Exercise the deterministic local VAD/interval path without loading a
    # second neural model in the unit suite.
    engine.encoder = None
    segments = engine.diarize(audio)

    assert segments
    assert segments[0].start >= 0
    assert segments[-1].end <= 2.01
    assert all(item.label.startswith("speaker_") for item in segments)


def test_align_speakers_preserves_overlap_metadata() -> None:
    item = Utterance(1, 1.0, 3.0, 1, "en", 0.9, "hello")
    aligned = align_speakers(
        [item],
        [
            SpeakerSegment(1.0, 2.4, "speaker_a", 0.8),
            SpeakerSegment(2.0, 3.0, "speaker_b", 0.7),
        ],
    )[0]

    assert aligned.speaker_source == "diarization"
    assert aligned.speaker_id == 1
    assert aligned.speaker_ids == [1, 2]
    assert 0 < aligned.speaker_overlap <= 1
    assert aligned.speaker_confidence > 0

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from realtime_meeting.audio import RotatingAudioWriter, SAMPLE_RATE, StreamSegmenter


def pcm_tone(seconds: float, amplitude: int = 6000) -> bytes:
    samples = int(seconds * SAMPLE_RATE)
    timeline = np.arange(samples, dtype=np.float32) / SAMPLE_RATE
    audio = np.sin(2 * np.pi * 220 * timeline) * amplitude
    return audio.astype(np.int16).tobytes()


def pcm_silence(seconds: float) -> bytes:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.int16).tobytes()


def test_stream_segmenter_commits_after_silence_and_emits_partial():
    segmenter = StreamSegmenter(partial_interval_ms=500)
    audio = pcm_silence(0.4) + pcm_tone(2.0) + pcm_silence(0.7)
    events = []
    for offset in range(0, len(audio), 777):
        events.extend(segmenter.feed(audio[offset : offset + 777]))
    partials = [event for event in events if event.kind == "partial"]
    finals = [event for event in events if event.kind == "final"]
    assert partials
    assert len(finals) == 1
    assert 0.0 <= finals[0].start < 0.5
    assert 2.3 <= finals[0].end <= 2.7
    assert segmenter.elapsed_seconds >= 3.0


def test_stream_segmenter_forces_long_utterance_cut():
    segmenter = StreamSegmenter(max_utterance_ms=1000, partial_interval_ms=5000)
    events = segmenter.feed(pcm_tone(2.4))
    forced = [event for event in events if event.kind == "final" and event.forced]
    assert len(forced) >= 2
    assert all(event.end > event.start for event in forced)


def test_stream_segmenter_accepts_quiet_microphone_signal():
    # A laptop/Bluetooth microphone can deliver valid speech well below the
    # old fixed 220-RMS gate. It should still produce a stable segment.
    quiet = pcm_tone(1.6, amplitude=100)
    events = StreamSegmenter().feed(quiet + pcm_silence(0.7))
    assert any(event.kind == "final" for event in events)


def test_audio_writer_rotates_bounded_segments_with_wav_fallback(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("realtime_meeting.audio.shutil.which", lambda _name: None)
    writer = RotatingAudioWriter(tmp_path, segment_minutes=1)
    writer.segment_samples = 1600
    writer.write(pcm_tone(0.25))
    segments = writer.close()
    assert len(segments) == 3
    assert [item["samples"] for item in segments] == [1600, 1600, 800]
    for item in segments:
        path = tmp_path / str(item["file"])
        with wave.open(str(path), "rb") as handle:
            assert handle.getframerate() == SAMPLE_RATE
            assert handle.getnchannels() == 1

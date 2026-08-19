from __future__ import annotations

import numpy as np
import pytest

from realtime_meeting.audio import (
    FRAME_BYTES,
    StreamSegmenter,
    apply_volume_gate,
    volume_threshold_percent_to_rms,
)


def test_volume_threshold_converts_to_rms_without_modifying_packets() -> None:
    threshold = volume_threshold_percent_to_rms(2.2)
    quiet = np.full(FRAME_BYTES // 2, 100, dtype=np.int16).tobytes()
    loud = np.full(FRAME_BYTES // 2, 1000, dtype=np.int16).tobytes()

    assert 200 < threshold < 300
    assert apply_volume_gate(quiet, threshold) == quiet
    assert apply_volume_gate(loud, threshold) == loud


def test_volume_threshold_rejects_values_outside_ui_range() -> None:
    with pytest.raises(ValueError):
        volume_threshold_percent_to_rms(-0.1)
    with pytest.raises(ValueError):
        volume_threshold_percent_to_rms(30.1)


def test_segmenter_handles_odd_packet_and_emits_final_segment() -> None:
    segmenter = StreamSegmenter(speech_start_ms=40, silence_ms=160, pre_roll_ms=40, partial_interval_ms=200)
    speech = np.full(FRAME_BYTES // 2, 1200, dtype=np.int16).tobytes()
    silence = np.zeros(FRAME_BYTES // 2, dtype=np.int16).tobytes()
    events = []
    events.extend(segmenter.feed(speech + b"\x00"))
    for _ in range(15):
        events.extend(segmenter.feed(speech))
    for _ in range(10):
        events.extend(segmenter.feed(silence))
    finals = [event for event in events if event.kind == "final"]
    assert finals
    assert finals[0].start == 0
    assert finals[0].end > finals[0].start


def test_segmenter_does_not_emit_repeated_partials_during_silence() -> None:
    segmenter = StreamSegmenter(speech_start_ms=40, silence_ms=800, pre_roll_ms=40, partial_interval_ms=200)
    speech = np.full(FRAME_BYTES // 2, 1200, dtype=np.int16).tobytes()
    silence = np.zeros(FRAME_BYTES // 2, dtype=np.int16).tobytes()
    events = []
    for _ in range(70):
        events.extend(segmenter.feed(speech))
    partial_count_before_silence = len([event for event in events if event.kind == "partial"])
    for _ in range(120):
        events.extend(segmenter.feed(silence))
    partial_count_after_silence = len([event for event in events if event.kind == "partial"])
    assert partial_count_before_silence > 0
    assert partial_count_after_silence == partial_count_before_silence


def test_model_vad_can_recall_quiet_speech_below_energy_threshold() -> None:
    segmenter = StreamSegmenter(
        speech_start_ms=40,
        silence_ms=160,
        minimum_rms=240.0,
        vad=lambda _frame: True,
    )
    room_noise = np.full(FRAME_BYTES // 2, 180, dtype=np.int16).tobytes()
    events = []
    for _ in range(500):
        events.extend(segmenter.feed(room_noise))
    events.extend(segmenter.flush())
    assert [event for event in events if event.kind == "partial"]
    assert not [event for event in events if event.kind == "final"]


def test_explicit_vad_silence_can_end_a_paragraph_even_with_room_noise() -> None:
    decisions = iter([True] * 10 + [False] * 10)

    def vad(_frame: bytes) -> bool:
        return next(decisions, False)

    segmenter = StreamSegmenter(
        speech_start_ms=40,
        silence_ms=160,
        pre_roll_ms=40,
        minimum_rms=240.0,
        vad=vad,
    )
    room_noise = np.full(FRAME_BYTES // 2, 500, dtype=np.int16).tobytes()
    events = []
    for _ in range(20):
        events.extend(segmenter.feed(room_noise))
    assert [event for event in events if event.kind == "final"]


def test_near_zero_audio_cannot_open_vad() -> None:
    segmenter = StreamSegmenter(
        speech_start_ms=40,
        silence_ms=160,
        minimum_rms=240.0,
        vad=lambda _frame: True,
    )
    near_zero = np.full(FRAME_BYTES // 2, 4, dtype=np.int16).tobytes()
    events = []
    for _ in range(100):
        events.extend(segmenter.feed(near_zero))
    events.extend(segmenter.flush())
    assert not events


def test_short_noise_burst_does_not_pass_utterance_admission() -> None:
    segmenter = StreamSegmenter(
        speech_start_ms=40,
        silence_ms=160,
        pre_roll_ms=40,
        minimum_rms=240.0,
        minimum_speech_ms=300,
        vad=lambda _frame: True,
    )
    loud = np.full(FRAME_BYTES // 2, 1200, dtype=np.int16).tobytes()
    silence = np.zeros(FRAME_BYTES // 2, dtype=np.int16).tobytes()
    events = []
    for _ in range(5):  # 100 ms click/bump-like burst
        events.extend(segmenter.feed(loud))
    for _ in range(10):
        events.extend(segmenter.feed(silence))
    assert not [event for event in events if event.kind == "final"]
    diagnostics = segmenter.diagnostics_snapshot()
    assert diagnostics["segments_opened"] == 1
    assert diagnostics["segments_emitted"] == 0
    assert diagnostics["admission_rejections"] >= 1

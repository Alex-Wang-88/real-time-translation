from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
HARNESS = Path(__file__).with_name("audio_worklet_harness.js")
WORKLET = ROOT / "realtime_meeting" / "web" / "audio-worklet.js"


def run_worklet(input_rate: int, frequency: int, amplitude: float = 0.5) -> dict[str, float]:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for AudioWorklet signal tests")
    result = subprocess.run(
        [node, str(HARNESS), str(WORKLET), str(input_rate), str(frequency), str(amplitude)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("input_rate", [16_000, 44_100, 48_000])
def test_worklet_resampling_preserves_voice_band_and_block_continuity(input_rate: int) -> None:
    result = run_worklet(input_rate, 1_000)
    assert abs(result["count"] - 16_000) <= 320
    assert 0.30 <= result["rms"] <= 0.40
    assert result["maxJump"] < 0.25


def test_worklet_resampler_attenuates_above_nyquist_input() -> None:
    voice = run_worklet(48_000, 1_000)
    alias = run_worklet(48_000, 12_000)
    assert alias["rms"] < voice["rms"] * 0.2


def test_worklet_volume_threshold_never_zeros_pcm() -> None:
    quiet = run_worklet(48_000, 1_000, amplitude=0.01)
    assert quiet["rms"] > 0.005

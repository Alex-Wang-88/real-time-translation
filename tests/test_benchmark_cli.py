from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_checked_in_benchmark_fixture_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "benchmark.py"), str(ROOT / "tests" / "fixtures" / "benchmark.json")],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    report = json.loads(result.stdout)
    assert report["samples"] == 2
    assert report["asr"]["wer_or_cer_mean"] == 0.0
    assert report["translation"]["chrf_mean"] == 1.0


def test_model_smoke_has_help_and_requires_explicit_audio() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_test_local_models.py"), "--help"],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    assert "audio" in result.stdout

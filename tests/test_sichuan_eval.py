from __future__ import annotations

import wave
from pathlib import Path

from realtime_meeting.sichuan_eval import (
    build_wsc_eval_manifest,
    load_wsc_eval_subset,
    validate_wsc_eval_manifest,
)


def _write_wav(path: Path, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(b"\0\0" * round(16_000 * seconds))


def _write_wsc_fixture(root: Path) -> None:
    subset = root / "Easy"
    (subset / "wav").mkdir(parents=True)
    (subset / "text").write_text(
        "clip-a 今天把这个事情整起\nclip-b 莫得问题，明天给你回话\n",
        encoding="utf-8",
    )
    (subset / "wav.scp").write_text(
        "clip-a Eval/Easy/wav/clip-a.wav\nclip-b Eval/Easy/wav/clip-b.wav\n",
        encoding="utf-8",
    )
    (subset / "key.txt").write_text("clip-b\nclip-a\n", encoding="utf-8")
    _write_wav(subset / "wav" / "clip-a.wav", 1.0)
    _write_wav(subset / "wav" / "clip-b.wav", 2.0)


def test_wsc_kaldi_export_becomes_ordered_dual_text_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "WSC-Eval-ASR"
    _write_wsc_fixture(dataset_root)
    output_path = tmp_path / "manifests" / "sichuan.json"

    manifest = build_wsc_eval_manifest(dataset_root, "Easy", output_path=output_path)

    assert [item["source_id"] for item in manifest["samples"]] == ["clip-b", "clip-a"]
    assert manifest["samples"][0]["text_sichuan"] == "莫得问题，明天给你回话"
    assert manifest["samples"][0]["text_mandarin"] is None
    assert manifest["samples"][0]["speech_variant"] == "sichuan"
    assert manifest["samples"][0]["duration_seconds"] == 2.0
    assert manifest["samples"][0]["audio_path"] == "../WSC-Eval-ASR/Easy/wav/clip-b.wav"
    assert manifest["recording_seconds"] == 3.0
    assert validate_wsc_eval_manifest(manifest, require_audio=True, base_dir=output_path.parent) == []


def test_wsc_manifest_limit_keeps_surface_text_and_external_audio_reference(tmp_path: Path) -> None:
    dataset_root = tmp_path / "WSC-Eval-ASR"
    _write_wsc_fixture(dataset_root)

    samples = load_wsc_eval_subset(dataset_root, "Easy", limit=1)

    assert len(samples) == 1
    assert samples[0]["source_id"] == "clip-b"
    assert samples[0]["reference_text"] == samples[0]["text_sichuan"]
    assert samples[0]["text_mandarin"] is None

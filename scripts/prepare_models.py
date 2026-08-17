from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from realtime_meeting.runtime import (
    MODEL_REVISIONS,
    OPUS_MT_REPOSITORIES,
    QWEN_ASR_PRIMARY_MODEL,
    _is_qwen_asr_model,
    _model_snapshot,
    prepare_opus_mt_model,
)


def _prepare_translation(root: Path, autodownload: bool) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for source in ("en", "de"):
        target = root / f"{source}-zh"
        if not (target / "model.bin").is_file() and autodownload:
            prepare_opus_mt_model(source, root)
        spm_ready = bool(list(target.glob("*.spm")) or list(target.glob("*.model"))) or any((target / name).is_file() for name in ("source.spm", "target.spm", "sentencepiece.bpe.model", "spm.model"))
        metadata_ready = any((target / name).is_file() for name in ("config.json", "model.json", "meeting_model.json", "config.yml", "config.yaml"))
        results[source] = {
            "path": str(target),
            "model_bin": (target / "model.bin").is_file(),
            "sentencepiece": spm_ready,
            "metadata": metadata_ready,
            "ready": (target / "model.bin").is_file() and spm_ready and metadata_ready,
            "repository": OPUS_MT_REPOSITORIES[source],
        }
    return results


def _check_vad(autodownload: bool) -> dict[str, Any]:
    model = os.getenv("MEETING_VAD", "fsmn-vad")
    result: dict[str, Any] = {
        "model": model,
        "package_ready": importlib.util.find_spec("funasr") is not None,
        "torchaudio_ready": importlib.util.find_spec("torchaudio") is not None,
        "local_snapshot": False,
        "ready": False,
    }
    if model in {"", "disabled"}:
        result["ready"] = True
        return result
    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            repo_id="funasr/fsmn-vad",
            revision=MODEL_REVISIONS["funasr/fsmn-vad"],
            local_files_only=not autodownload,
        )
        result["snapshot"] = str(snapshot)
        result["local_snapshot"] = True
    except Exception as exc:
        result["error"] = str(exc)
    result["ready"] = bool(result["package_ready"] and result["torchaudio_ready"] and result["local_snapshot"])
    return result


def _check_asr(autodownload: bool) -> dict[str, Any]:
    primary = os.getenv("MEETING_ASR_PRIMARY", QWEN_ASR_PRIMARY_MODEL)
    single_model = os.getenv("MEETING_SINGLE_ASR_MODEL", "1").strip().casefold() in {"1", "true", "yes", "on"}
    models = {
        "primary": primary,
        "fallback": primary if single_model else os.getenv("MEETING_ASR_FALLBACK", QWEN_ASR_PRIMARY_MODEL),
        "language_id": primary if single_model else os.getenv("MEETING_ASR_LANGUAGE_ID", QWEN_ASR_PRIMARY_MODEL),
    }
    result: dict[str, Any] = {"models": models, "snapshots": {}, "ready": False, "local_only": not autodownload}
    try:
        if importlib.util.find_spec("qwen_asr") is None:
            raise RuntimeError("qwen-asr package is not installed")
        for key, model_name in models.items():
            if not _is_qwen_asr_model(model_name):
                raise RuntimeError(f"not a supported Qwen3-ASR model: {model_name}")
            result["snapshots"][key] = str(_model_snapshot(model_name, local_files_only=not autodownload))
        result["ready"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Qwen ASR, VAD and translation models")
    parser.add_argument("--root", type=Path, default=Path("models"))
    parser.add_argument("--translation-root", type=Path, default=None)
    parser.add_argument("--download-translation", action="store_true")
    parser.add_argument("--skip-translation", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    translation_root = args.translation_root or args.root / "opus-mt"
    report: dict[str, Any] = {
        "asr": {
            **_check_asr(autodownload=not args.check_only),
            "policy": "single Qwen3-ASR-1.7B for realtime ASR, segment language confirmation and conflict re-decoding",
        },
        "vad": _check_vad(autodownload=not args.check_only),
        "translation": {} if args.skip_translation else _prepare_translation(translation_root, args.download_translation and not args.check_only),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.check_only and (
        not report["asr"]["ready"]
        or not report["vad"]["ready"]
        or (not args.skip_translation and not all(item["ready"] for item in report["translation"].values()))
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

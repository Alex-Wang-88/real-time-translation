from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from realtime_meeting.diarization import DiarizationEngine
from realtime_meeting.runtime import OPUS_MT_REPOSITORIES, prepare_opus_mt_model


def _download(repo: str, target: Path) -> Path:
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    path = Path(snapshot_download(repo_id=repo, local_dir=str(target)))
    return path


def _prepare_translation(root: Path, autodownload: bool) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for source in ("en", "de"):
        target = root / f"{source}-zh"
        if not (target / "model.bin").is_file() and autodownload:
            prepare_opus_mt_model(source, root)
        spm_ready = any((target / name).is_file() for name in ("source.spm", "target.spm", "sentencepiece.bpe.model", "spm.model")) or bool(list(target.glob("*.spm"))) or bool(list(target.glob("*.model")))
        metadata_ready = any((target / name).is_file() for name in ("config.json", "model.json", "meeting_model.json", "config.yml", "config.yaml"))
        results[source] = {
            "path": str(target),
            "model_bin": (target / "model.bin").is_file(),
            "sentencepiece": spm_ready,
            "metadata": metadata_ready,
            "ready": (target / "model.bin").is_file() and spm_ready and metadata_ready,
        }
    return results


def _check_vad(autodownload: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": os.getenv("MEETING_VAD", "fsmn-vad"),
        "package_ready": importlib.util.find_spec("funasr") is not None,
        "torchaudio_ready": importlib.util.find_spec("torchaudio") is not None,
        "local_snapshot": False,
        "ready": False,
        "policy": "FunASR VAD adapter; runtime resolves a local HF snapshot",
    }
    if result["model"] in {"", "disabled"}:
        result["ready"] = True
        return result
    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            repo_id="funasr/fsmn-vad",
            local_files_only=not autodownload,
        )
        result["snapshot"] = str(snapshot)
        result["local_snapshot"] = True
    except Exception as exc:
        result["error"] = str(exc)
    result["ready"] = bool(
        result["package_ready"]
        and result["torchaudio_ready"]
        and result["local_snapshot"]
    )
    return result


def _check_asr(autodownload: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "realtime": os.getenv("MEETING_ASR_REALTIME", "large-v3-turbo"),
        "refine": os.getenv("MEETING_ASR_REFINE", "large-v3"),
        "ready": False,
        "local_only": not autodownload,
    }
    try:
        from faster_whisper.utils import download_model

        paths: dict[str, str] = {}
        for key, model_name, repository in (
            ("realtime", result["realtime"], "mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
            ("refine", result["refine"], "Systran/faster-whisper-large-v3"),
        ):
            paths[key] = str(download_model(repository, local_files_only=not autodownload))
        result["snapshots"] = paths
        result["ready"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local meeting models")
    parser.add_argument("--root", type=Path, default=Path("models"))
    parser.add_argument("--translation-root", type=Path, default=None)
    parser.add_argument("--download-translation", action="store_true")
    parser.add_argument("--skip-translation", action="store_true", help="Only check ASR/VAD/speaker assets")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    translation_root = args.translation_root or args.root / "opus-mt"
    diarization = DiarizationEngine(
        os.getenv("MEETING_DEVICE", "cpu"),
        required=True,
    )
    diarization_ready = diarization.preflight()
    report: dict[str, Any] = {
        "asr": {**_check_asr(autodownload=not args.check_only), "policy": "faster-whisper local snapshot preflight"},
        "vad": _check_vad(autodownload=not args.check_only),
        "translation": {} if args.skip_translation else _prepare_translation(translation_root, args.download_translation and not args.check_only),
        "diarization": {
            "model": diarization.model_name,
            "weight_bytes": diarization.model_size_bytes,
            "ready": diarization_ready,
            "backend": diarization.backend,
            "status": diarization.status,
            "parameters": diarization.model_parameters(),
            "error": diarization.error,
            "policy": "Resemblyzer only; local bundled weights; no external authorization or runtime download",
        },
    }
    for source, asset in report.get("translation", {}).items():
        asset["repository"] = OPUS_MT_REPOSITORIES[source]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.check_only and (
        not report["asr"]["ready"]
        or not report["vad"]["ready"]
        or (not args.skip_translation and not all(item["ready"] for item in report["translation"].values()))
        or not report["diarization"]["ready"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import snapshot_download

from realtime_meeting.runtime import (
    MODEL_REVISIONS,
    QWEN_ASR_PRIMARY_MODEL,
    QWEN_ASR_SMALL_MODEL,
    LiveChineseTranslator,
    _model_snapshot,
    choose_device,
)


def _qwen_model(model_name: str, device: str):
    import torch
    from qwen_asr import Qwen3ASRModel

    snapshot = _model_snapshot(model_name, local_files_only=True)
    return Qwen3ASRModel.from_pretrained(
        str(snapshot),
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="cuda:0" if device == "cuda" else "cpu",
        max_inference_batch_size=1,
        max_new_tokens=256,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Qwen ASR, VAD and translation smoke checks")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--translation-root", type=Path, default=Path("models/opus-mt"))
    args = parser.parse_args()
    audio = args.audio.resolve()
    if not audio.is_file():
        parser.error(f"audio file does not exist: {audio}")
    device, compute_type = choose_device(args.device)
    print(json.dumps({"device": device, "compute_type": compute_type}, ensure_ascii=True))

    for model_name in (QWEN_ASR_PRIMARY_MODEL, QWEN_ASR_SMALL_MODEL):
        started = time.perf_counter()
        model = _qwen_model(model_name, device)
        loaded = time.perf_counter()
        results = model.transcribe(str(audio))
        first = results[0] if results else None
        print(json.dumps({
            "asr_model": model_name,
            "asr_loaded": True,
            "load_seconds": round(loaded - started, 3),
            "transcribe_seconds": round(time.perf_counter() - loaded, 3),
            "language": getattr(first, "language", None) if first is not None else None,
            "text": str(getattr(first, "text", "") if first is not None else ""),
        }, ensure_ascii=True))
        del model

    from funasr import AutoModel

    vad_snapshot = snapshot_download("funasr/fsmn-vad", revision=MODEL_REVISIONS["funasr/fsmn-vad"], local_files_only=True)
    vad = AutoModel(model=vad_snapshot, hub="hf", trust_remote_code=True, disable_update=True, device=f"{device}:0" if device == "cuda" else device)
    print(json.dumps({"vad_loaded": True, "vad_result": vad.generate(input=str(audio))}, ensure_ascii=True))

    translator = LiveChineseTranslator(args.translation_root, device)
    print(json.dumps({"translation_assets": translator.preflight()}, ensure_ascii=True))


if __name__ == "__main__":
    main()

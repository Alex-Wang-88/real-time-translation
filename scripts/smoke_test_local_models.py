from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model

from realtime_meeting.runtime import MODEL_REVISIONS, LiveChineseTranslator, choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local ASR, VAD and translation smoke checks")
    parser.add_argument("audio", type=Path, help="Existing speech audio file used for ASR and VAD")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--translation-root", type=Path, default=Path("models/opus-mt"))
    args = parser.parse_args()
    audio = args.audio.resolve()
    if not audio.is_file():
        parser.error(f"audio file does not exist: {audio}")
    device, compute_type = choose_device(args.device)
    print(json.dumps({"device": device, "compute_type": compute_type}, ensure_ascii=True))

    for model_name, repository in (
        ("large-v3-turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
        ("large-v3", "Systran/faster-whisper-large-v3"),
    ):
        started = time.perf_counter()
        snapshot = download_model(
            repository,
            revision=MODEL_REVISIONS[repository],
            local_files_only=True,
        )
        model = WhisperModel(snapshot, device=device, compute_type=compute_type)
        loaded = time.perf_counter()
        segments, info = model.transcribe(
            str(audio),
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            language="en",
        )
        rows = list(segments)
        print(json.dumps({
            "asr_model": model_name,
            "asr_loaded": True,
            "load_seconds": round(loaded - started, 3),
            "transcribe_seconds": round(time.perf_counter() - loaded, 3),
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "segments": len(rows),
            "text": " ".join(item.text.strip() for item in rows),
        }, ensure_ascii=True))
        del rows, segments, model
        try:
            import gc
            import torch

            gc.collect()
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            print(json.dumps({"cleanup_warning": str(exc)}, ensure_ascii=True))

    from funasr import AutoModel

    started = time.perf_counter()
    vad_snapshot = snapshot_download(
        "funasr/fsmn-vad",
        revision=MODEL_REVISIONS["funasr/fsmn-vad"],
        local_files_only=True,
    )
    vad = AutoModel(
        model=vad_snapshot,
        hub="hf",
        trust_remote_code=True,
        disable_update=True,
        device=f"{device}:0" if device == "cuda" else device,
    )
    print(json.dumps({"vad_loaded": True, "seconds": round(time.perf_counter() - started, 3)}, ensure_ascii=True))
    started = time.perf_counter()
    vad_result = vad.generate(input=str(audio))
    print(json.dumps({"vad_inference_seconds": round(time.perf_counter() - started, 3), "vad_result": vad_result}, ensure_ascii=True))

    translator = LiveChineseTranslator(args.translation_root, device)
    assets = translator.preflight()
    print(json.dumps({"translation_assets": assets}, ensure_ascii=True))
    for source, sentence in (
        ("en", "We will ship the beta on September 15."),
        ("de", "Anna prueft die Sicherheitsrisiken bis Freitag."),
    ):
        started = time.perf_counter()
        result = translator.translate_many([sentence], source)[0]
        print(json.dumps({
            "source": source,
            "seconds": round(time.perf_counter() - started, 3),
            "text": result.text,
            "status": result.status,
            "model": result.model,
            "error": result.error,
        }, ensure_ascii=True))


if __name__ == "__main__":
    main()

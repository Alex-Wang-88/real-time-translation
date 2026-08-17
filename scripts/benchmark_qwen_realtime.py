from __future__ import annotations

import argparse
import json
import statistics
import time
import wave
from pathlib import Path

from realtime_meeting.runtime import _model_snapshot, choose_device


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * ratio))]


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as source:
        return source.getnframes() / max(1, source.getframerate())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed Qwen3-ASR realtime promotion gate on a local WAV file"
    )
    parser.add_argument("audio", type=Path, help="Representative 16-bit WAV speech recording")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--device", choices=("auto", "cuda"), default="auto")
    parser.add_argument(
        "--language",
        choices=("Chinese", "English", "German", "Cantonese"),
        help="Optional Qwen language hint; Sichuan/Wu remain Chinese plus context hints.",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    audio = args.audio.resolve()
    if not audio.is_file():
        parser.error(f"audio file does not exist: {audio}")
    duration = wav_duration(audio)
    if duration <= 0:
        parser.error("audio file is empty")

    import torch
    from qwen_asr import Qwen3ASRModel

    device, _compute_type = choose_device(args.device)
    if device != "cuda" or not torch.cuda.is_available():
        parser.error("the 1.7B promotion gate requires CUDA")
    snapshot = _model_snapshot(args.model, local_files_only=not args.allow_download)
    model = Qwen3ASRModel.from_pretrained(
        str(snapshot),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=256,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    latencies: list[float] = []
    decoded_audio_seconds = 0.0
    started = time.perf_counter()
    deadline = started + max(1.0, args.minutes * 60.0)
    oom = False
    error: str | None = None
    iterations = 0
    try:
        while time.perf_counter() < deadline:
            decode_started = time.perf_counter()
            kwargs = {"audio": str(audio), "context": ""}
            if args.language:
                kwargs["language"] = args.language
            model.transcribe(**kwargs)
            latency = time.perf_counter() - decode_started
            latencies.append(latency)
            decoded_audio_seconds += duration
            iterations += 1
    except torch.cuda.OutOfMemoryError as exc:
        oom = True
        error = str(exc)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - started
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    rtf = sum(latencies) / decoded_audio_seconds if decoded_audio_seconds else float("inf")
    p95 = percentile(latencies, 0.95)
    passed = (
        not oom
        and error is None
        and elapsed >= args.minutes * 60.0
        and peak_gib <= 7.5
        and rtf <= 0.75
        and p95 <= 2.5
    )
    report = {
        "model": args.model,
        "stress_minutes_requested": args.minutes,
        "stress_seconds_completed": round(elapsed, 3),
        "iterations": iterations,
        "audio_seconds_decoded": round(decoded_audio_seconds, 3),
        "latency_seconds": {
            "p50": round(percentile(latencies, 0.5), 3),
            "p95_final": round(p95, 3),
            "mean": round(statistics.mean(latencies), 3) if latencies else None,
        },
        "rtf": round(rtf, 4) if rtf != float("inf") else None,
        "peak_vram_gib": round(peak_gib, 3),
        "oom": oom,
        "error": error,
        "promotion_thresholds": {
            "peak_vram_gib_max": 7.5,
            "rtf_max": 0.75,
            "stress_minutes_min": 30,
            "p95_final_seconds_max": 2.5,
        },
        "eligible_for_default": passed,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()

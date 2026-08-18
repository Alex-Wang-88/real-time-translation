"""Run the production ASR/LID/translation path on a generated meeting fixture."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from realtime_meeting.audio import SegmentEvent
from realtime_meeting.config import load_settings
from realtime_meeting.runtime import LiveModelRuntime


def tokens(text: str, language: str) -> list[str]:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    cleaned: list[str] = []
    for char in value:
        category = unicodedata.category(char)
        if char.isspace() or category.startswith("P") or category.startswith("S"):
            cleaned.append(" ")
        else:
            cleaned.append(char)
    value = "".join(cleaned)
    if language == "zh" or any("\u3400" <= char <= "\u9fff" for char in value):
        return [char for char in value if not char.isspace()]
    return value.split()


def edit_distance(reference: Iterable[str], hypothesis: Iterable[str]) -> int:
    left, right = list(reference), list(hypothesis)
    previous = list(range(len(right) + 1))
    for row, token in enumerate(left, 1):
        current = [row]
        for column, other in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (token != other),
                )
            )
        previous = current
    return previous[-1]


def score(reference: str, hypothesis: str, language: str) -> dict[str, float | int]:
    expected = tokens(reference, language)
    actual = tokens(hypothesis, language)
    distance = edit_distance(expected, actual)
    return {
        "reference_tokens": len(expected),
        "edit_distance": distance,
        "error_rate": round(distance / max(1, len(expected)), 6),
    }


def chrf(reference: str, hypothesis: str, order: int = 6) -> float:
    scores = []
    for size in range(1, order + 1):
        ref = {reference[index:index + size] for index in range(max(0, len(reference) - size + 1))}
        hyp = {hypothesis[index:index + size] for index in range(max(0, len(hypothesis) - size + 1))}
        if not ref and not hyp:
            scores.append(1.0)
        elif not ref or not hyp:
            scores.append(0.0)
        else:
            scores.append(len(ref & hyp) / len(ref | hyp))
    return round(sum(scores) / max(1, len(scores)), 6)


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * ratio))
    return round(ordered[index], 3)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def read_audio(path: Path) -> tuple[bytes, float]:
    import wave

    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getframerate() != 16_000:
            raise ValueError(f"{path} is not 16 kHz mono PCM16 WAV")
        pcm = source.readframes(source.getnframes())
        duration = source.getnframes() / source.getframerate()
    return pcm, duration


def normalize_variant(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    return text or None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"samples": 0}
    reference_total = sum(int(item.get("reference_tokens", 0)) for item in records)
    distance_total = sum(int(item.get("edit_distance", 0)) for item in records)
    language_records = [item for item in records if item.get("language_expected") in {"zh", "en", "de"}]
    language_correct = sum(bool(item.get("language_correct")) for item in language_records)
    translation_records = [item for item in records if item.get("translation_expected")]
    translated = [item for item in translation_records if item.get("translation_status") == "ready"]
    return {
        "samples": len(records),
        "asr_macro_error_rate": round(statistics.mean(safe_float(item.get("error_rate")) for item in records), 6),
        "asr_token_weighted_error_rate": round(distance_total / max(1, reference_total), 6),
        "language_accuracy": round(language_correct / len(language_records), 6) if language_records else None,
        "language_accuracy_samples": len(language_records),
        "sichuan_variant_emission_rate": round(
            sum(bool(item.get("speech_variant")) for item in records if item.get("language_expected") == "zh" and item.get("speech_variant_expected") == "sichuan")
            / max(1, sum(item.get("speech_variant_expected") == "sichuan" for item in records)),
            6,
        ),
        "sichuan_variant_accuracy": round(
            sum(bool(item.get("variant_correct")) for item in records if item.get("speech_variant_expected") == "sichuan")
            / max(1, sum(item.get("speech_variant_expected") == "sichuan" for item in records)),
            6,
        ),
        "translation_samples": len(translation_records),
        "translation_success_rate": round(len(translated) / len(translation_records), 6) if translation_records else None,
        "translation_chrf_mean": round(statistics.mean(safe_float(item.get("translation_chrf")) for item in translated), 6) if translated else None,
        "latency_ms": {
            "language_id_p50": percentile([safe_float(item.get("language_id_latency_ms")) for item in records], 0.5),
            "language_id_p95": percentile([safe_float(item.get("language_id_latency_ms")) for item in records], 0.95),
            "asr_p50": percentile([safe_float(item.get("asr_latency_ms")) for item in records], 0.5),
            "asr_p95": percentile([safe_float(item.get("asr_latency_ms")) for item in records], 0.95),
            "e2e_p50": percentile([safe_float(item.get("e2e_latency_ms")) for item in records], 0.5),
            "e2e_p95": percentile([safe_float(item.get("e2e_latency_ms")) for item in records], 0.95),
        },
        "rtf": round(statistics.mean(safe_float(item.get("e2e_latency_ms")) / 1000 / max(0.001, safe_float(item.get("duration_seconds"))) for item in records), 6),
        "errors": sum(bool(item.get("error")) for item in records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload.get("samples") or []
    if not samples:
        raise ValueError(f"manifest has no samples: {manifest_path}")

    settings = load_settings()
    runtime = LiveModelRuntime(
        settings.asr_primary,
        settings.asr_fallback,
        args.device,
        language_id_model=settings.language_id_model,
        asr_autodownload=settings.asr_autodownload,
        translation_model_root=settings.translation_model_root,
        translation_autodownload=False,
        translation_warmup=settings.translation_warmup,
        vad_model="disabled",
        single_model=settings.single_asr_model,
    )
    load_started = time.perf_counter()
    runtime.load(print)
    load_seconds = time.perf_counter() - load_started
    if not runtime.ready or runtime.primary is None:
        raise RuntimeError(f"ASR runtime is not ready: {runtime.status}")

    records: list[dict[str, Any]] = []
    try:
        if not args.no_warmup:
            warmup_pcm = b"\0\0" * 16_000
            warmup_event = SegmentEvent("final", warmup_pcm, 0.0, 1.0, 1, False)
            runtime.detect_language(warmup_pcm)
            runtime.transcribe_final(warmup_event, decode_settings={"realtime_asr_model": "primary"})

        for index, sample in enumerate(samples, 1):
            audio_path = manifest_path.parent / str(sample["audio_path"])
            pcm, duration = read_audio(audio_path)
            lid_started = time.perf_counter()
            lid_error = None
            lid = None
            try:
                lid = runtime.detect_language(pcm)
            except Exception as exc:  # noqa: BLE001
                lid_error = f"{type(exc).__name__}: {exc}"
            lid_latency_ms = (time.perf_counter() - lid_started) * 1000
            language_hint = getattr(lid, "code", None) if lid else None
            variant_hint = getattr(lid, "speech_variant", None) if lid else None
            event = SegmentEvent("final", pcm, 0.0, duration, 1, False)
            asr_started = time.perf_counter()
            error = lid_error
            result = None
            try:
                result = runtime.transcribe_final(
                    event,
                    language=language_hint if language_hint in {"zh", "en", "de"} else None,
                    speech_variant=variant_hint,
                    decode_settings={"realtime_asr_model": "primary"},
                )
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
            asr_latency_ms = (time.perf_counter() - asr_started) * 1000
            text = str(getattr(result, "text", "") or "").strip() if result is not None else ""
            language = str(getattr(result, "language", "unknown") or "unknown") if result is not None else "unknown"
            variant = normalize_variant(getattr(result, "speech_variant", None) if result is not None else None)
            quality = score(str(sample["reference_text"]), text, str(sample["language"]))

            translation_text = ""
            translation_status = "not_needed"
            translation_error = None
            translation_latency_ms = 0.0
            translation_expected = sample.get("reference_translation")
            if language in {"en", "de"} and text:
                translation_started = time.perf_counter()
                try:
                    translated = runtime.translate_text(text, language)
                    translation_text = str(getattr(translated, "text", "") or "").strip()
                    translation_status = str(getattr(translated, "status", "failed") or "failed")
                    translation_error = getattr(translated, "error", None)
                except Exception as exc:  # noqa: BLE001
                    translation_status = "failed"
                    translation_error = f"{type(exc).__name__}: {exc}"
                translation_latency_ms = (time.perf_counter() - translation_started) * 1000
            elif language in {"en", "de"}:
                translation_status = "skipped_empty_asr"

            item = {
                "sample_id": sample["sample_id"],
                "speaker_name": sample.get("speaker_name"),
                "group": sample.get("group"),
                "audio_path": sample["audio_path"],
                "duration_seconds": duration,
                "language_expected": sample["language"],
                "speech_variant_expected": sample.get("speech_variant"),
                "reference_text": sample["reference_text"],
                "reference_translation": translation_expected,
                "language_detected": getattr(lid, "code", "unknown") if lid else "unknown",
                "speech_variant_detected": getattr(lid, "speech_variant", None) if lid else None,
                "text": text,
                "language": language,
                "speech_variant": variant,
                "translation_expected": translation_expected,
                "raw_qwen_label": str(getattr(result, "raw_qwen_label", "") or "") if result is not None else "",
                "confidence": safe_float(getattr(result, "confidence", 0.0) if result is not None else 0.0),
                "language_correct": language == sample["language"],
                "variant_correct": variant == sample.get("speech_variant") if sample["language"] == "zh" else None,
                "language_id_latency_ms": round(lid_latency_ms, 3),
                "asr_latency_ms": round(asr_latency_ms, 3),
                "translation_latency_ms": round(translation_latency_ms, 3),
                "e2e_latency_ms": round(lid_latency_ms + asr_latency_ms + translation_latency_ms, 3),
                "translation": translation_text,
                "translation_status": translation_status,
                "translation_error": translation_error,
                "translation_chrf": chrf(str(translation_expected), translation_text) if translation_expected and translation_text else None,
                "error": error,
                **quality,
            }
            records.append(item)
            print(f"[{index}/{len(samples)}] {sample['sample_id']} {language} {asr_latency_ms / 1000:.2f}s", flush=True)

        report = {
            "schema_version": "1.0-generated-meeting",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "manifest": str(manifest_path),
            "runtime": {
                "device": runtime.device,
                "requested_device": args.device,
                "status": runtime.status,
                "load_seconds": round(load_seconds, 3),
                "capabilities": runtime.capability_snapshot(),
                "metrics": runtime.metrics,
            },
            "summary": summarize(records),
            "by_group": {
                group: summarize([item for item in records if item.get("group") == group])
                for group in sorted({str(item.get("group")) for item in records})
            },
            "records": records,
        }
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())

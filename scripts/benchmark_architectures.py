"""Run the two current realtime ASR architectures on a local manifest."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
import unicodedata
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from realtime_meeting.audio import SegmentEvent
from realtime_meeting.config import load_settings
from realtime_meeting.runtime import LiveModelRuntime


ARCHITECTURES = {
    "quality_first": {
        "label": "大+小千问（质量优先）",
        "realtime_asr_model": "primary",
        "primary_model": "Qwen/Qwen3-ASR-1.7B",
        "fallback_model": "Qwen/Qwen3-ASR-0.6B",
    },
    "latency_first": {
        "label": "小千问主识别（低延迟）",
        "realtime_asr_model": "small",
        "primary_model": "Qwen/Qwen3-ASR-0.6B",
        "fallback_model": "Qwen/Qwen3-ASR-1.7B",
    },
}


def _tokens(text: str, language: str) -> list[str]:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    if language == "mixed":
        # Mixed Mandarin-English references need a tokenization that keeps a
        # Chinese character as one token while treating an English phrase as
        # words.  This avoids charging every English word as several CJK-like
        # character errors in the long code-switch benchmark.
        return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[\u3400-\u9fff]", value)
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


def _edit_distance(reference: Iterable[str], hypothesis: Iterable[str]) -> int:
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


def _score(reference: str, hypothesis: str, language: str) -> dict[str, float | int]:
    expected = _tokens(reference, language)
    actual = _tokens(hypothesis, language)
    distance = _edit_distance(expected, actual)
    return {
        "reference_tokens": len(expected),
        "edit_distance": distance,
        "error_rate": distance / max(1, len(expected)),
    }


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * ratio))
    return round(ordered[index], 6)


def _read_audio(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getframerate() != 16_000:
            raise ValueError(f"{path} is not 16 kHz mono PCM16 WAV")
        pcm = source.readframes(source.getnframes())
        duration = source.getnframes() / max(1, source.getframerate())
    return pcm, duration


def _limit_audio(pcm: bytes, duration: float, maximum_seconds: float | None) -> tuple[bytes, float]:
    if maximum_seconds is None or maximum_seconds >= duration:
        return pcm, duration
    frame_count = max(1, round(maximum_seconds * 16_000))
    return pcm[: frame_count * 2], min(duration, maximum_seconds)


def _sync_cuda(device: str) -> None:
    if device != "cuda":
        return
    import torch

    torch.cuda.synchronize()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _load_manifest(path: Path, limit: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"manifest has no samples: {path}")
    root = path.parent
    checked: list[dict[str, Any]] = []
    for item in samples[:limit] if limit else samples:
        if not isinstance(item, dict):
            continue
        audio_path = root / str(item.get("audio_path", ""))
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        enriched = dict(item)
        enriched["_audio_path"] = audio_path
        checked.append(enriched)
    if not checked:
        raise ValueError("manifest limit removed all samples")
    return payload, checked


def _normalize_variant(value: Any, raw_label: Any = "") -> str | None:
    text = str(value or "").strip().casefold()
    raw = str(raw_label or "").strip().casefold()
    if text:
        return text
    if "cantonese" in raw:
        return "cantonese"
    if "mandarin" in raw:
        return "mandarin"
    return None


def _is_cantonese_variant(value: str | None, raw_label: Any = "") -> bool:
    return "cantonese" in str(value or "").casefold() or "cantonese" in str(raw_label or "").casefold()


def _run_one(
    runtime: LiveModelRuntime,
    sample: dict[str, Any],
    *,
    architecture_id: str,
    lid: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    pcm, duration = _read_audio(Path(sample["_audio_path"]))
    event = SegmentEvent("final", pcm, 0.0, duration, 1, False)
    language = str(lid.get("language", "unknown"))
    language_hint_used = bool(sample.get("decode_language_hint", True))
    hinted_language = language if language_hint_used and language in {"zh", "en", "de"} else None
    speech_variant = lid.get("speech_variant") if language_hint_used else None
    started = time.perf_counter()
    _sync_cuda(device)
    error: str | None = None
    result: Any | None = None
    try:
        result = runtime.transcribe_final(
            event,
            recent_text="",
            previous_language=hinted_language,
            language=hinted_language,
            speech_variant=speech_variant,
            decode_settings={"realtime_asr_model": ARCHITECTURES[architecture_id]["realtime_asr_model"]},
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    _sync_cuda(device)
    elapsed = time.perf_counter() - started
    text = str(getattr(result, "text", "") or "").strip() if result is not None else ""
    language_result = str(getattr(result, "language", "unknown") or "unknown") if result is not None else "unknown"
    raw_label = str(getattr(result, "raw_qwen_label", "") or "") if result is not None else ""
    variant = _normalize_variant(getattr(result, "speech_variant", None) if result is not None else None, raw_label)
    score = _score(str(sample["reference_text"]), text, str(sample["language"]))
    return {
        "architecture": architecture_id,
        "text": text,
        "language": language_result,
        "speech_variant": variant,
        "raw_qwen_label": raw_label,
        "model_used": getattr(result, "model", None) if result is not None else None,
        "language_hint_used": language_hint_used,
        "confidence": _safe_float(getattr(result, "confidence", 0.0) if result is not None else 0.0),
        "decode_latency_ms": round(elapsed * 1000, 3),
        "decode_rtf": round(elapsed / max(duration, 0.001), 6),
        "error": error,
        **score,
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"samples": 0}
    error_rates = [_safe_float(item.get("error_rate")) for item in records]
    latencies = [_safe_float(item.get("decode_latency_ms")) for item in records]
    e2e_latencies = [_safe_float(item.get("e2e_latency_ms")) for item in records]
    rtfs = [_safe_float(item.get("decode_rtf")) for item in records]
    total_distance = sum(int(item.get("edit_distance", 0)) for item in records)
    total_reference = sum(int(item.get("reference_tokens", 0)) for item in records)
    language_records = [item for item in records if item.get("language_correct") is not None]
    language_correct = sum(bool(item.get("language_correct")) for item in language_records)
    variant_records = [item for item in records if item.get("language") == "zh"]
    variant_emitted = sum(bool(item.get("speech_variant")) for item in variant_records)
    cantonese_records = [item for item in records if item.get("expected_group") == "Cantonese"]
    cantonese_correct = sum(bool(item.get("variant_correct")) for item in cantonese_records)
    model_usage = Counter(str(item.get("model_used") or "error") for item in records)
    errors = sum(bool(item.get("error")) for item in records)
    return {
        "samples": len(records),
        "macro_error_rate": round(statistics.mean(error_rates), 6),
        "token_weighted_error_rate": round(total_distance / max(1, total_reference), 6),
        "language_accuracy": round(language_correct / len(language_records), 6) if language_records else None,
        "language_accuracy_samples": len(language_records),
        "speech_variant_emission_rate_zh": round(variant_emitted / max(1, len(variant_records)), 6) if variant_records else None,
        "cantonese_variant_accuracy": round(cantonese_correct / len(cantonese_records), 6) if cantonese_records else None,
        "latency_ms": {
            "decode_p50": _percentile(latencies, 0.5),
            "decode_p95": _percentile(latencies, 0.95),
            "e2e_p50": _percentile(e2e_latencies, 0.5),
            "e2e_p95": _percentile(e2e_latencies, 0.95),
        },
        "rtf": {
            "decode_mean": round(statistics.mean(rtfs), 6),
            "decode_p95": _percentile(rtfs, 0.95),
        },
        "model_usage": dict(model_usage),
        "errors": errors,
        "context": {
            "duration_seconds_mean": round(
                statistics.mean(_safe_float(item.get("duration_seconds")) for item in records), 3
            ),
            "turn_count_mean": round(
                statistics.mean(_safe_float(item.get("turn_count")) for item in records), 3
            ) if any(item.get("turn_count") is not None for item in records) else None,
            "speaker_count_mean": round(
                statistics.mean(_safe_float(item.get("speaker_count")) for item in records), 3
            ) if any(item.get("speaker_count") is not None for item in records) else None,
            "language_switches_mean": round(
                statistics.mean(_safe_float(item.get("reference_language_switches")) for item in records), 3
            ) if any(item.get("reference_language_switches") is not None for item in records) else None,
        },
    }


def _comparison(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    quality = summaries["quality_first"]
    latency = summaries["latency_first"]
    quality_error = _safe_float(quality.get("token_weighted_error_rate"))
    latency_error = _safe_float(latency.get("token_weighted_error_rate"))
    quality_p95 = _safe_float(quality.get("latency_ms", {}).get("e2e_p95"))
    latency_p95 = _safe_float(latency.get("latency_ms", {}).get("e2e_p95"))
    error_delta_pp = (latency_error - quality_error) * 100
    latency_reduction_pct = (quality_p95 - latency_p95) / max(quality_p95, 1e-9) * 100
    if error_delta_pp <= 2.0 and latency_reduction_pct >= 30.0:
        recommendation = "latency_first"
        rationale = "低延迟架构的错误率增加不超过 2 个百分点，且端到端 P95 延迟至少降低 30%。"
    elif quality_error <= latency_error:
        recommendation = "quality_first"
        rationale = "质量优先架构的 token 加权错误率不高于低延迟架构；会议记录场景优先保留识别质量。"
    else:
        recommendation = "latency_first"
        rationale = "低延迟架构错误率更低，且本次数据未显示质量优先架构有质量优势。"
    return {
        "error_delta_latency_minus_quality_pp": round(error_delta_pp, 3),
        "e2e_p95_latency_reduction_latency_vs_quality_pct": round(latency_reduction_pct, 3),
        "selection_rule": "latency_first only when error-rate increase <= 2 pp and e2e P95 reduction >= 30%; otherwise prefer lower token-weighted error rate",
        "recommended_architecture": recommendation,
        "rationale": rationale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--lid-seconds",
        type=float,
        help="use only the first N seconds for shared language ID; ASR still decodes the complete sample",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.lid_seconds is not None and args.lid_seconds <= 0:
        parser.error("--lid-seconds must be positive")

    manifest_path = args.manifest.resolve()
    manifest, samples = _load_manifest(manifest_path, args.limit)
    settings = load_settings()
    runtime = LiveModelRuntime(
        settings.asr_primary,
        settings.asr_fallback,
        args.device,
        language_id_model=settings.language_id_model,
        asr_autodownload=args.allow_download,
        translation_model_root=settings.translation_model_root,
        translation_autodownload=False,
        vad_model="disabled",
    )
    load_started = time.perf_counter()
    runtime.load(print)
    load_seconds = time.perf_counter() - load_started
    if not runtime.ready or not runtime.fallback:
        raise RuntimeError(f"ASR runtime is not ready: {runtime.status}")
    device = runtime.device

    try:
        if not args.no_warmup:
            warmup_pcm = b"\0\0" * 16_000
            warmup_event = SegmentEvent("final", warmup_pcm, 0.0, 1.0, 1, False)
            for architecture_id in ARCHITECTURES:
                runtime.transcribe_final(
                    warmup_event,
                    language=None,
                    speech_variant=None,
                    decode_settings={"realtime_asr_model": ARCHITECTURES[architecture_id]["realtime_asr_model"]},
                )
            runtime.detect_language(warmup_pcm)
            _sync_cuda(device)

        predictions: list[dict[str, Any]] = []
        for index, sample in enumerate(samples):
            pcm, duration = _read_audio(Path(sample["_audio_path"]))
            lid_pcm, lid_duration = _limit_audio(pcm, duration, args.lid_seconds)
            lid_started = time.perf_counter()
            _sync_cuda(device)
            lid_error: str | None = None
            lid_result: Any | None = None
            try:
                lid_result = runtime.detect_language(lid_pcm)
            except Exception as exc:  # noqa: BLE001
                lid_error = f"{type(exc).__name__}: {exc}"
            _sync_cuda(device)
            lid_elapsed = time.perf_counter() - lid_started
            lid = {
                "language": str(getattr(lid_result, "code", "unknown") or "unknown") if lid_result else "unknown",
                "speech_variant": _normalize_variant(
                    getattr(lid_result, "speech_variant", None) if lid_result else None,
                    getattr(lid_result, "raw_qwen_label", "") if lid_result else "",
                ),
                "raw_qwen_label": str(getattr(lid_result, "raw_qwen_label", "") or "") if lid_result else "",
                "confidence": _safe_float(getattr(lid_result, "confidence", 0.0) if lid_result else 0.0),
                "latency_ms": round(lid_elapsed * 1000, 3),
                "audio_seconds": round(lid_duration, 3),
                "error": lid_error,
            }
            lid["language_correct"] = (
                None if sample["language"] == "mixed" else lid["language"] == sample["language"]
            )
            lid["variant_correct"] = (
                sample["group"] == "Cantonese" and _is_cantonese_variant(lid["speech_variant"], lid["raw_qwen_label"])
            )
            order = ("quality_first", "latency_first") if index % 2 == 0 else ("latency_first", "quality_first")
            for architecture_id in order:
                record = _run_one(
                    runtime,
                    sample,
                    architecture_id=architecture_id,
                    lid=lid,
                    device=device,
                )
                record.update(
                    {
                        "sample_id": sample["sample_id"],
                        "group": sample["group"],
                        "scenario": sample.get("scenario", sample["group"]),
                        "source_dataset": sample.get("source_dataset"),
                        "expected_group": sample["group"],
                        "language_expected": sample["language"],
                        "decode_language_hint": bool(sample.get("decode_language_hint", True)),
                        "reference_language_switches": sample.get("reference_language_switches"),
                        "speaker_count": sample.get("speaker_count"),
                        "turn_count": sample.get("turn_count"),
                        "reference_text": sample["reference_text"],
                        "audio_path": sample["audio_path"],
                        "duration_seconds": duration,
                        "lid": lid,
                        "lid_language_correct": lid["language_correct"],
                        "lid_variant_correct": lid["variant_correct"],
                        "language_correct": (
                            None if sample["language"] == "mixed" else record["language"] == sample["language"]
                        ),
                        "variant_correct": (
                            sample["group"] == "Cantonese"
                            and _is_cantonese_variant(record["speech_variant"], record["raw_qwen_label"])
                        ),
                        "e2e_latency_ms": round(lid_elapsed * 1000 + _safe_float(record["decode_latency_ms"]), 3),
                        "e2e_rtf": round((lid_elapsed + _safe_float(record["decode_latency_ms"]) / 1000) / max(duration, 0.001), 6),
                    }
                )
                predictions.append(record)
            print(f"[{index + 1}/{len(samples)}] {sample['sample_id']}")

        summaries: dict[str, dict[str, Any]] = {}
        for architecture_id in ARCHITECTURES:
            records = [item for item in predictions if item["architecture"] == architecture_id]
            summaries[architecture_id] = _summarize(records)
            by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in records:
                by_group[str(item["group"])].append(item)
            summaries[architecture_id]["by_group"] = {
                group: _summarize(group_records) for group, group_records in sorted(by_group.items())
            }
            by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in records:
                by_scenario[str(item["scenario"])].append(item)
            summaries[architecture_id]["by_scenario"] = {
                scenario: _summarize(scenario_records)
                for scenario, scenario_records in sorted(by_scenario.items())
            }

        lid_records = [
            item["lid"]
            | {
                "group": item["group"],
                "language_expected": item["language_expected"],
                "language_correct": item["lid_language_correct"],
            }
            for item in predictions
            if item["architecture"] == "quality_first"
        ]
        lid_applicable = [item for item in lid_records if item.get("language_correct") is not None]
        lid_summary = {
            "samples": len(lid_records),
            "language_accuracy": (
                round(sum(bool(item["language_correct"]) for item in lid_applicable) / len(lid_applicable), 6)
                if lid_applicable
                else None
            ),
            "language_accuracy_samples": len(lid_applicable),
            "latency_ms": {
                "p50": _percentile([_safe_float(item["latency_ms"]) for item in lid_records], 0.5),
                "p95": _percentile([_safe_float(item["latency_ms"]) for item in lid_records], 0.95),
            },
            "by_group": {},
        }
        lid_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in lid_records:
            lid_by_group[str(item["group"])].append(item)
        for group, group_records in sorted(lid_by_group.items()):
            applicable = [item for item in group_records if item.get("language_correct") is not None]
            lid_summary["by_group"][group] = {
                "samples": len(group_records),
                "language_accuracy": (
                    round(sum(bool(item["language_correct"]) for item in applicable) / len(applicable), 6)
                    if applicable
                    else None
                ),
                "language_accuracy_samples": len(applicable),
                "latency_ms_p50": _percentile([_safe_float(item["latency_ms"]) for item in group_records], 0.5),
            }

        report = {
            "schema_version": "1.1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest_path),
            "manifest_schema_version": manifest.get("schema_version"),
            "runtime": {
                "device": device,
                "requested_device": args.device,
                "status": runtime.status,
                "load_seconds": round(load_seconds, 3),
                "capabilities": runtime.capability_snapshot(),
                "metrics": runtime.metrics,
            },
            "evaluation": {
                "language_id_seconds": args.lid_seconds,
                "asr_decodes_full_sample": True,
            },
            "architectures": ARCHITECTURES,
            "language_id": lid_summary,
            "summaries": summaries,
            "comparison": _comparison(summaries),
            "predictions": predictions,
        }
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))
        print(json.dumps({name: summaries[name] for name in ARCHITECTURES}, ensure_ascii=False, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()

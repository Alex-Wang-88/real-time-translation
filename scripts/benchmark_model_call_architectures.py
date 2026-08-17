"""Compare model-call strategies on the existing ASR benchmark manifests.

The two original architecture reports are loaded as baselines.  This runner
adds call strategies that cannot be represented by a single ``realtime_asr_model``
setting: a true one-checkpoint 1.7B runtime, a no-LID one-checkpoint runtime, a
0.6B draft followed by a 1.7B final pass, a draft-based mixed-language router,
and fixed 12-second 1.7B chunk-and-merge decoding.

The router never reads reference labels.  It detects a likely code-switched
draft from the draft text itself, so it is a deployable policy rather than an
oracle selection.  All ASR scores are computed against the same reference as
the baseline reports.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from realtime_meeting.audio import SegmentEvent
from realtime_meeting.config import load_settings
from realtime_meeting.language import is_likely_code_switched
from realtime_meeting.runtime import LiveModelRuntime

from scripts.benchmark_architectures import (
    ARCHITECTURES as BASE_ARCHITECTURES,
    _is_cantonese_variant,
    _limit_audio,
    _load_manifest,
    _normalize_variant,
    _safe_float,
    _score,
    _summarize,
    _sync_cuda,
)


NEW_ARCHITECTURES: dict[str, dict[str, Any]] = {
    "single_1_7b": {
        "label": "单一 1.7B（1.7B 同时承担 LID/ASR，无 0.6B）",
        "runtime": "single",
        "policy": "1.7B language ID -> 1.7B full-sample ASR",
    },
    "single_1_7b_no_lid": {
        "label": "单一 1.7B 无 LID",
        "runtime": "single",
        "policy": "1.7B full-sample ASR，不做语言识别/语言强制",
    },
    "cascade_small_to_1_7b": {
        "label": "0.6B 草稿 -> 1.7B 最终",
        "runtime": "dual",
        "policy": "0.6B 全段草稿作为 1.7B context，再输出 1.7B 最终文本",
    },
    "router_mixed": {
        "label": "草稿驱动中英混合路由",
        "runtime": "dual",
        "policy": "0.6B 草稿检测到中英混合则直接采用草稿，否则升级 1.7B",
    },
    "chunked_1_7b": {
        "label": "1.7B 12 秒分块合并",
        "runtime": "single",
        "policy": "完整长段切为 12 秒窗口，逐窗调用 1.7B 并合并",
    },
}


def _load_samples(paths: list[Path], limit: int | None) -> tuple[list[dict[str, Any]], list[str]]:
    samples: list[dict[str, Any]] = []
    manifests: list[str] = []
    for path in paths:
        manifest, items = _load_manifest(path.resolve(), limit)
        del manifest
        manifests.append(str(path.resolve()))
        for item in items:
            enriched = dict(item)
            enriched["_manifest_path"] = str(path.resolve())
            if not enriched.get("scenario"):
                enriched["scenario"] = "short_multilingual_dialect"
            samples.append(enriched)
    if not samples:
        raise ValueError("no benchmark samples found")
    return samples, manifests


def _load_baselines(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for raw in payload.get("predictions", []) if isinstance(payload, dict) else []:
            if not isinstance(raw, dict):
                continue
            key = (str(raw.get("architecture", "")), str(raw.get("sample_id", "")))
            if key in seen:
                continue
            seen.add(key)
            record = dict(raw)
            group = str(record.get("group", ""))
            record["scenario"] = str(
                record.get("scenario")
                or (
                    "multi_person_meeting"
                    if group == "AMI multi-person meeting"
                    else "mandarin_english_code_switch"
                    if group == "ASCEND Mandarin-English long mixed"
                    else "short_multilingual_dialect"
                )
            )
            record.setdefault("model_calls", 1)
            record.setdefault("asr_model_calls", 1)
            records.append(record)
    return records


def _new_runtime(settings: Any, device: str, *, single: bool) -> LiveModelRuntime:
    if single:
        primary = settings.asr_primary
        return LiveModelRuntime(
            primary,
            primary,
            device,
            language_id_model=primary,
            asr_autodownload=False,
            translation_model_root=settings.translation_model_root,
            translation_autodownload=False,
            vad_model="disabled",
        )
    return LiveModelRuntime(
        settings.asr_primary,
        settings.asr_fallback,
        device,
        language_id_model=settings.language_id_model,
        asr_autodownload=False,
        translation_model_root=settings.translation_model_root,
        translation_autodownload=False,
        vad_model="disabled",
    )


def _warm_runtime(runtime: LiveModelRuntime, *, single: bool) -> None:
    warmup_pcm = b"\0\0" * 16_000
    event = SegmentEvent("final", warmup_pcm, 0.0, 1.0, 1, False)
    runtime.transcribe_final(event, language=None, speech_variant=None, decode_settings={"realtime_asr_model": "primary"})
    if not single:
        runtime.transcribe_final(event, language=None, speech_variant=None, decode_settings={"realtime_asr_model": "small"})
    runtime.detect_language(warmup_pcm)
    _sync_cuda(runtime.device)


def _lid(runtime: LiveModelRuntime, pcm: bytes, duration: float, maximum_seconds: float | None) -> dict[str, Any]:
    lid_pcm, lid_duration = _limit_audio(pcm, duration, maximum_seconds)
    started = time.perf_counter()
    _sync_cuda(runtime.device)
    result: Any | None = None
    error: str | None = None
    try:
        result = runtime.detect_language(lid_pcm)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    _sync_cuda(runtime.device)
    elapsed = time.perf_counter() - started
    raw_label = str(getattr(result, "raw_qwen_label", "") or "") if result else ""
    return {
        "available": True,
        "language": str(getattr(result, "code", "unknown") or "unknown") if result else "unknown",
        "speech_variant": _normalize_variant(getattr(result, "speech_variant", None) if result else None, raw_label),
        "raw_qwen_label": raw_label,
        "confidence": _safe_float(getattr(result, "confidence", 0.0) if result else 0.0),
        "latency_ms": round(elapsed * 1000, 3),
        "audio_seconds": round(lid_duration, 3),
        "error": error,
    }


def _hint(sample: dict[str, Any], lid: dict[str, Any], *, enabled: bool) -> tuple[str | None, str | None, bool]:
    used = bool(enabled and sample.get("decode_language_hint", True))
    language = str(lid.get("language", "unknown")) if used else "unknown"
    language = language if language in {"zh", "en", "de"} else None
    variant = lid.get("speech_variant") if language else None
    return language, variant, used


def _call(
    runtime: LiveModelRuntime,
    sample: dict[str, Any],
    pcm: bytes,
    duration: float,
    *,
    role: str,
    lid: dict[str, Any],
    recent_text: str = "",
    language_enabled: bool = True,
) -> dict[str, Any]:
    language, variant, hint_used = _hint(sample, lid, enabled=language_enabled)
    event = SegmentEvent("final", pcm, 0.0, duration, 1, False)
    started = time.perf_counter()
    _sync_cuda(runtime.device)
    result: Any | None = None
    error: str | None = None
    try:
        result = runtime.transcribe_final(
            event,
            recent_text=recent_text,
            previous_language=language,
            language=language,
            speech_variant=variant,
            decode_settings={"realtime_asr_model": role},
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    _sync_cuda(runtime.device)
    elapsed = time.perf_counter() - started
    return {
        "text": str(getattr(result, "text", "") or "").strip() if result is not None else "",
        "language": str(getattr(result, "language", "unknown") or "unknown") if result is not None else "unknown",
        "speech_variant": _normalize_variant(
            getattr(result, "speech_variant", None) if result is not None else None,
            getattr(result, "raw_qwen_label", "") if result is not None else "",
        ),
        "raw_qwen_label": str(getattr(result, "raw_qwen_label", "") or "") if result is not None else "",
        "model_used": getattr(result, "model", None) if result is not None else None,
        "confidence": _safe_float(getattr(result, "confidence", 0.0) if result is not None else 0.0),
        "latency_ms": round(elapsed * 1000, 3),
        "error": error,
        "language_hint_used": hint_used,
    }


def _aggregate(
    sample: dict[str, Any],
    architecture: str,
    lid: dict[str, Any],
    calls: list[dict[str, Any]],
    *,
    selected_role: str,
    routing_decision: str | None = None,
    chunk_count: int = 1,
) -> dict[str, Any]:
    final = calls[-1]
    text = final.get("text", "")
    if chunk_count > 1:
        text = " ".join(str(item.get("text", "")).strip() for item in calls if str(item.get("text", "")).strip()).strip()
    score = _score(str(sample["reference_text"]), text, str(sample["language"]))
    decode_ms = sum(_safe_float(item.get("latency_ms")) for item in calls)
    errors = [str(item["error"]) for item in calls if item.get("error")]
    expected_language = str(sample["language"])
    output_language = str(final.get("language", "unknown"))
    return {
        "architecture": architecture,
        "sample_id": sample["sample_id"],
        "scenario": sample.get("scenario", "short_multilingual_dialect"),
        "source_dataset": sample.get("source_dataset"),
        "group": sample.get("group", ""),
        "expected_group": sample.get("group", ""),
        "language_expected": expected_language,
        "language_correct": None if expected_language == "mixed" else output_language == expected_language,
        "decode_language_hint": bool(sample.get("decode_language_hint", True)),
        "language_hint_used": bool(final.get("language_hint_used", False)),
        "reference_language_switches": sample.get("reference_language_switches"),
        "speaker_count": sample.get("speaker_count"),
        "turn_count": sample.get("turn_count"),
        "reference_text": sample["reference_text"],
        "audio_path": sample["audio_path"],
        "duration_seconds": sample["duration_seconds"],
        "text": text,
        "language": output_language,
        "speech_variant": final.get("speech_variant"),
        "raw_qwen_label": final.get("raw_qwen_label", ""),
        "model_used": final.get("model_used"),
        "selected_role": selected_role,
        "routing_decision": routing_decision,
        "chunk_count": chunk_count,
        "model_calls": len(calls),
        "asr_model_calls": len(calls),
        "confidence": _safe_float(final.get("confidence")),
        "decode_latency_ms": round(decode_ms, 3),
        "decode_rtf": round(decode_ms / 1000 / max(_safe_float(sample["duration_seconds"]), 0.001), 6),
        "e2e_latency_ms": round(_safe_float(lid.get("latency_ms")) + decode_ms, 3),
        "e2e_rtf": round(
            (_safe_float(lid.get("latency_ms")) + decode_ms) / 1000 / max(_safe_float(sample["duration_seconds"]), 0.001),
            6,
        ),
        "lid": lid,
        "lid_language_correct": (
            None
            if expected_language == "mixed" or not lid.get("available", True)
            else lid.get("language") == expected_language
        ),
        "lid_variant_correct": (
            sample.get("group") == "Cantonese"
            and _is_cantonese_variant(lid.get("speech_variant"), lid.get("raw_qwen_label", ""))
        ),
        "variant_correct": (
            sample.get("group") == "Cantonese"
            and _is_cantonese_variant(final.get("speech_variant"), final.get("raw_qwen_label", ""))
        ),
        "error": "; ".join(errors) if errors else None,
        **score,
    }


def _run_chunked(
    runtime: LiveModelRuntime,
    sample: dict[str, Any],
    pcm: bytes,
    duration: float,
    lid: dict[str, Any],
    chunk_seconds: float,
) -> tuple[list[dict[str, Any]], int]:
    bytes_per_chunk = max(2, round(chunk_seconds * 16_000) * 2)
    calls: list[dict[str, Any]] = []
    previous_text = ""
    for index, start in enumerate(range(0, len(pcm), bytes_per_chunk), 1):
        chunk = pcm[start : start + bytes_per_chunk]
        chunk_duration = len(chunk) / 2 / 16_000
        if chunk_duration <= 0:
            continue
        call = _call(
            runtime,
            sample,
            chunk,
            chunk_duration,
            role="primary",
            lid=lid,
            recent_text=previous_text[-240:],
        )
        call["chunk_index"] = index
        call["chunk_start_seconds"] = round(start / 2 / 16_000, 3)
        previous_text = f"{previous_text} {call['text']}".strip()
        calls.append(call)
    return calls, len(calls)


def _run_single_architectures(
    runtime: LiveModelRuntime,
    samples: list[dict[str, Any]],
    architecture_ids: list[str],
    *,
    lid_seconds: float | None,
    chunk_seconds: float,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    lid_by_sample: dict[str, dict[str, Any]] = {}
    needs_lid = any(name == "single_1_7b" or name == "chunked_1_7b" for name in architecture_ids)
    for index, sample in enumerate(samples, 1):
        pcm, duration = __import__("scripts.benchmark_architectures", fromlist=["_read_audio"])._read_audio(Path(sample["_audio_path"]))
        if needs_lid and sample["sample_id"] not in lid_by_sample:
            lid_by_sample[sample["sample_id"]] = _lid(runtime, pcm, duration, lid_seconds)
        for architecture in architecture_ids:
            if architecture == "single_1_7b_no_lid":
                lid = {
                    "available": False,
                    "language": "unknown",
                    "speech_variant": None,
                    "latency_ms": 0.0,
                    "error": None,
                }
                call = _call(runtime, sample, pcm, duration, role="primary", lid=lid, language_enabled=False)
                predictions.append(_aggregate(sample, architecture, lid, [call], selected_role="primary"))
            elif architecture == "single_1_7b":
                lid = lid_by_sample[sample["sample_id"]]
                call = _call(runtime, sample, pcm, duration, role="primary", lid=lid)
                predictions.append(_aggregate(sample, architecture, lid, [call], selected_role="primary"))
            elif architecture == "chunked_1_7b":
                lid = lid_by_sample[sample["sample_id"]]
                calls, chunk_count = _run_chunked(runtime, sample, pcm, duration, lid, chunk_seconds)
                predictions.append(
                    _aggregate(sample, architecture, lid, calls, selected_role="primary", chunk_count=chunk_count)
                )
            else:
                raise ValueError(f"unsupported single-runtime architecture: {architecture}")
        print(f"single [{index}/{len(samples)}] {sample['sample_id']}")
    return predictions


def _run_dual_architectures(
    runtime: LiveModelRuntime,
    samples: list[dict[str, Any]],
    architecture_ids: list[str],
    *,
    lid_seconds: float | None,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, 1):
        from scripts.benchmark_architectures import _read_audio

        pcm, duration = _read_audio(Path(sample["_audio_path"]))
        lid = _lid(runtime, pcm, duration, lid_seconds)
        for architecture in architecture_ids:
            if architecture == "cascade_small_to_1_7b":
                draft = _call(runtime, sample, pcm, duration, role="small", lid=lid)
                final = _call(runtime, sample, pcm, duration, role="primary", lid=lid, recent_text=draft["text"])
                predictions.append(
                    _aggregate(
                        sample,
                        architecture,
                        lid,
                        [draft, final],
                        selected_role="primary",
                        routing_decision="always_upgrade",
                    )
                )
            elif architecture == "router_mixed":
                draft = _call(runtime, sample, pcm, duration, role="small", lid=lid)
                if is_likely_code_switched(draft["text"]):
                    predictions.append(
                        _aggregate(
                            sample,
                            architecture,
                            lid,
                            [draft],
                            selected_role="small",
                            routing_decision="draft_detected_code_switch",
                        )
                    )
                else:
                    final = _call(runtime, sample, pcm, duration, role="primary", lid=lid, recent_text=draft["text"])
                    predictions.append(
                        _aggregate(
                            sample,
                            architecture,
                            lid,
                            [draft, final],
                            selected_role="primary",
                            routing_decision="draft_not_code_switch",
                        )
                    )
            else:
                raise ValueError(f"unsupported dual-runtime architecture: {architecture}")
        print(f"dual [{index}/{len(samples)}] {sample['sample_id']}")
    return predictions


def _add_by_scenario(summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("scenario", "unknown"))].append(record)
    summary["by_scenario"] = {name: _summarize(values) for name, values in sorted(grouped.items())}


def _rank(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for architecture, summary in summaries.items():
        rows.append(
            {
                "architecture": architecture,
                "token_weighted_error_rate": summary.get("token_weighted_error_rate"),
                "decode_p50_ms": summary.get("latency_ms", {}).get("decode_p50"),
                "e2e_p95_ms": summary.get("latency_ms", {}).get("e2e_p95"),
                "mean_model_calls": summary.get("model_calls_mean"),
            }
        )
    return sorted(rows, key=lambda row: (_safe_float(row["token_weighted_error_rate"]), _safe_float(row["e2e_p95_ms"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--baseline-report", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, default=Path("result/architecture_call_benchmark/report.json"))
    parser.add_argument("--architectures", nargs="+", choices=sorted(NEW_ARCHITECTURES), default=list(NEW_ARCHITECTURES))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--lid-seconds", type=float, default=12.0)
    parser.add_argument("--chunk-seconds", type=float, default=12.0)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.lid_seconds <= 0 or args.chunk_seconds <= 0:
        parser.error("--lid-seconds and --chunk-seconds must be positive")

    samples, manifest_paths = _load_samples(args.manifest, args.limit)
    settings = load_settings()
    baseline_paths = args.baseline_report or [
        Path("result/architecture_benchmark/report.json"),
        Path("result/architecture_benchmark_long/report.json"),
    ]
    predictions = _load_baselines(baseline_paths)
    runtime_snapshots: list[dict[str, Any]] = []

    single_ids = [name for name in args.architectures if NEW_ARCHITECTURES[name]["runtime"] == "single"]
    dual_ids = [name for name in args.architectures if NEW_ARCHITECTURES[name]["runtime"] == "dual"]
    if single_ids:
        runtime = _new_runtime(settings, args.device, single=True)
        try:
            load_started = time.perf_counter()
            runtime.load(print)
            if not runtime.ready or runtime.primary is None or runtime.fallback is None:
                raise RuntimeError(f"single 1.7B runtime is not ready: {runtime.status}")
            if not args.no_warmup:
                _warm_runtime(runtime, single=True)
            predictions.extend(
                _run_single_architectures(
                    runtime,
                    samples,
                    single_ids,
                    lid_seconds=args.lid_seconds,
                    chunk_seconds=args.chunk_seconds,
                )
            )
            runtime_snapshots.append(
                {
                    "runtime": "single_1_7b",
                    "device": runtime.device,
                    "load_seconds": round(time.perf_counter() - load_started, 3),
                    "status": runtime.status,
                    "capabilities": runtime.capability_snapshot(),
                    "metrics": runtime.metrics,
                }
            )
        finally:
            runtime.close()
    if dual_ids:
        runtime = _new_runtime(settings, args.device, single=False)
        try:
            load_started = time.perf_counter()
            runtime.load(print)
            if not runtime.ready or runtime.primary is None or runtime.fallback is None:
                raise RuntimeError(f"dual runtime is not ready: {runtime.status}")
            if not args.no_warmup:
                _warm_runtime(runtime, single=False)
            predictions.extend(_run_dual_architectures(runtime, samples, dual_ids, lid_seconds=args.lid_seconds))
            runtime_snapshots.append(
                {
                    "runtime": "dual_1.7b_plus_0.6b",
                    "device": runtime.device,
                    "load_seconds": round(time.perf_counter() - load_started, 3),
                    "status": runtime.status,
                    "capabilities": runtime.capability_snapshot(),
                    "metrics": runtime.metrics,
                }
            )
        finally:
            runtime.close()

    summaries: dict[str, dict[str, Any]] = {}
    by_architecture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in predictions:
        by_architecture[str(record["architecture"])].append(record)
    for architecture, records in sorted(by_architecture.items()):
        summary = _summarize(records)
        summary["model_calls_mean"] = round(statistics.mean(_safe_float(item.get("model_calls")) for item in records), 3)
        summary["model_calls_total"] = sum(int(item.get("model_calls", 0)) for item in records)
        summary["routing_decisions"] = dict(Counter(str(item.get("routing_decision")) for item in records if item.get("routing_decision")))
        _add_by_scenario(summary, records)
        summaries[architecture] = summary

    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifests": manifest_paths,
        "baseline_reports": [str(path.resolve()) for path in baseline_paths if path.is_file()],
        "evaluation": {
            "language_id_seconds": args.lid_seconds,
            "chunk_seconds": args.chunk_seconds,
            "asr_full_sample_for_non_chunked": True,
            "router_reference_free": True,
        },
        "architectures": BASE_ARCHITECTURES | NEW_ARCHITECTURES,
        "runtime_snapshots": runtime_snapshots,
        "summaries": summaries,
        "ranking_by_token_error": _rank(summaries),
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "architectures": list(summaries), "ranking": report["ranking_by_token_error"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

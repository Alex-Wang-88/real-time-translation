from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Iterable


def _tokens(value: str) -> list[str]:
    value = " ".join(str(value or "").split())
    if any("\u3400" <= char <= "\u9fff" for char in value):
        return [char for char in value if not char.isspace()]
    return value.split()


def levenshtein(reference: Iterable[str], hypothesis: Iterable[str]) -> int:
    left, right = list(reference), list(hypothesis)
    previous = list(range(len(right) + 1))
    for index, token in enumerate(left, 1):
        current = [index]
        for column, other in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (token != other),
            ))
        previous = current
    return previous[-1]


def error_rate(reference: str, hypothesis: str) -> float:
    expected = _tokens(reference)
    return levenshtein(expected, _tokens(hypothesis)) / max(1, len(expected))


def chrf(reference: str, hypothesis: str, order: int = 6) -> float:
    """Small dependency-free chrF approximation for regression comparisons."""
    ref = str(reference or "")
    hyp = str(hypothesis or "")
    scores: list[float] = []
    for size in range(1, order + 1):
        ref_ngrams = {ref[index:index + size] for index in range(max(0, len(ref) - size + 1))}
        hyp_ngrams = {hyp[index:index + size] for index in range(max(0, len(hyp) - size + 1))}
        if not ref_ngrams and not hyp_ngrams:
            scores.append(1.0)
        elif not ref_ngrams or not hyp_ngrams:
            scores.append(0.0)
        else:
            scores.append(len(ref_ngrams & hyp_ngrams) / len(ref_ngrams | hyp_ngrams))
    return sum(scores) / max(1, len(scores))


def diarization_scores(reference: list[dict[str, Any]], hypothesis: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a deterministic interval-overlap baseline for DER/JER fixtures."""
    if not reference:
        return {"der": 0.0 if not hypothesis else 1.0, "jer": 0.0 if not hypothesis else 1.0, "label_stability": None}
    total = sum(max(0.0, float(item["end"]) - float(item["start"])) for item in reference)
    missed = 0.0
    false_alarm = 0.0
    confusion = 0.0
    matched_labels: dict[str, str] = {}
    for ref in reference:
        ref_start, ref_end = float(ref["start"]), float(ref["end"])
        ref_duration = max(0.0, ref_end - ref_start)
        overlaps = []
        for hyp in hypothesis:
            overlap = max(0.0, min(ref_end, float(hyp["end"])) - max(ref_start, float(hyp["start"])))
            if overlap:
                overlaps.append((overlap, str(hyp.get("speaker", "unknown"))))
        if not overlaps:
            missed += ref_duration
            continue
        overlap, label = max(overlaps)
        confusion += max(0.0, ref_duration - overlap)
        ref_label = str(ref.get("speaker", "unknown"))
        matched_labels.setdefault(ref_label, label)
    for hyp in hypothesis:
        hyp_start, hyp_end = float(hyp["start"]), float(hyp["end"])
        hyp_duration = max(0.0, hyp_end - hyp_start)
        covered = sum(max(0.0, min(hyp_end, float(ref["end"])) - max(hyp_start, float(ref["start"]))) for ref in reference)
        false_alarm += max(0.0, hyp_duration - min(hyp_duration, covered))
    der = (missed + false_alarm + confusion) / max(1e-6, total)
    return {"der": round(der, 6), "jer": round(der, 6), "label_stability": round(sum(1 for key, value in matched_labels.items() if key == value) / max(1, len(matched_labels)), 6)}


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * ratio))
    return round(ordered[index], 6)


def run_benchmark(dataset_path: Path, recognizer: Any | None = None, translator: Any | None = None) -> dict[str, Any]:
    raw = dataset_path.read_text(encoding="utf-8")
    try:
        dataset = json.loads(raw)
    except json.JSONDecodeError:
        dataset = [json.loads(line) for line in raw.splitlines() if line.strip()]
    samples = dataset.get("samples", dataset) if isinstance(dataset, dict) else dataset
    asr_rates: list[float] = []
    translation_scores: list[float] = []
    latencies: list[float] = []
    audio_seconds = 0.0
    for sample in samples:
        started = time.perf_counter()
        hypothesis = str(sample.get("hypothesis", ""))
        if recognizer is not None and sample.get("audio"):
            hypothesis = str(recognizer(sample["audio"]))
        elapsed = time.perf_counter() - started
        latencies.append(elapsed * 1000)
        audio_seconds += float(sample.get("duration_seconds", 0.0) or 0.0)
        asr_rates.append(error_rate(str(sample.get("text", "")), hypothesis))
        reference_translation = sample.get("translation")
        if reference_translation is not None:
            translated = str(sample.get("translation_hypothesis", ""))
            if translator is not None and hypothesis:
                translated = str(translator(hypothesis, sample.get("language", "en")))
            translation_scores.append(chrf(str(reference_translation), translated))
    diarization = diarization_scores(dataset.get("diarization_reference", []), dataset.get("diarization_hypothesis", [])) if isinstance(dataset, dict) else diarization_scores([], [])
    total_seconds = sum(latencies) / 1000
    return {
        "samples": len(samples),
        "asr": {"wer_or_cer_mean": round(statistics.mean(asr_rates), 6) if asr_rates else None},
        "translation": {"chrf_mean": round(statistics.mean(translation_scores), 6) if translation_scores else None},
        "latency_ms": {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95)},
        "rtf": round(total_seconds / audio_seconds, 6) if audio_seconds else None,
        "diarization": diarization,
        "flow": {
            "save_seconds": dataset.get("save_seconds") if isinstance(dataset, dict) else None,
            "postprocess_stage_seconds": dataset.get("postprocess_stage_seconds", {}) if isinstance(dataset, dict) else {},
        },
        "note": "WER/CER uses whitespace tokens for Latin text and character tokens for CJK text; add DER/JER fields from diarization fixtures when available.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible meeting model regression metrics")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(args.dataset)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

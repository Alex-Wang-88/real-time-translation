"""Reference-based evaluation for a completed real-time meeting replay.

The live pipeline is intentionally evaluated after the meeting has finished.
This keeps the evaluator independent from model execution and makes it usable
against saved replay reports, including reports produced by older revisions of
the replay script.
"""

from __future__ import annotations

import math
import statistics
import unicodedata
from functools import lru_cache
from typing import Any, Iterable, Sequence


SUPPORTED_LANGUAGES = {"zh", "en", "de"}
_LANGUAGE_ALIASES = {
    "zh-cn": "zh",
    "zh-sc": "zh",
    "中文": "zh",
    "普通话": "zh",
    "四川方言": "zh",
    "english": "en",
    "英语": "en",
    "german": "de",
    "deutsch": "de",
    "德语": "de",
}
_VARIANT_ALIASES = {
    "none": None,
    "unknown": None,
    "": None,
    "普通话": "mandarin",
    "mandarin": "mandarin",
    "四川话": "sichuan",
    "四川方言": "sichuan",
    "sichuan": "sichuan",
}


def normalize_language(value: Any) -> str:
    text = str(value or "").strip().casefold()
    return _LANGUAGE_ALIASES.get(text, text)


def normalize_variant(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    return _VARIANT_ALIASES.get(text, text or None)


def tokens(text: str, language: str) -> list[str]:
    """Tokenize CJK by character and Latin languages by whitespace words."""

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


def chrf(reference: str, hypothesis: str, order: int = 6) -> float:
    """Return a small dependency-free character n-gram overlap score."""

    scores: list[float] = []
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


def score_text(reference: str, hypothesis: str, language: str) -> dict[str, float | int]:
    expected = tokens(reference, language)
    actual = tokens(hypothesis, language)
    distance = edit_distance(expected, actual)
    normalized_reference = "".join(expected)
    normalized_hypothesis = "".join(actual)
    return {
        "reference_tokens": len(expected),
        "hypothesis_tokens": len(actual),
        "edit_distance": distance,
        "error_rate": round(distance / max(1, len(expected)), 6),
        "chrf": chrf(normalized_reference, normalized_hypothesis),
    }

def _reference_surface_text(sample: dict[str, Any]) -> str:
    """Return the spoken surface form, keeping Sichuan wording intact."""

    return str(sample.get("text_sichuan") or sample.get("reference_text") or sample.get("text") or "")


def _reference_text(sample: dict[str, Any]) -> str:
    return _reference_surface_text(sample)


def _reference_mandarin_text(sample: dict[str, Any]) -> str:
    """Return an optional meaning-normalized Mandarin reference.

    WSC-Eval-ASR does not publish this field.  It is therefore optional and is
    never inferred from the Sichuan surface transcript.
    """

    return str(sample.get("text_mandarin") or sample.get("reference_mandarin_text") or "")


def _reference_translation(sample: dict[str, Any]) -> str:
    return str(sample.get("reference_translation") or sample.get("translation") or "")


def _paragraph_id(paragraph: dict[str, Any], fallback: int) -> str:
    value = paragraph.get("segment_id") or paragraph.get("id")
    return str(value if value is not None else fallback)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _duration(paragraph: dict[str, Any]) -> float:
    start = _safe_float(paragraph.get("start"))
    end = _safe_float(paragraph.get("end"), start)
    return max(0.0, end - start)


def _group_text(paragraphs: Sequence[dict[str, Any]], field: str) -> str:
    return " ".join(str(item.get(field) or "").strip() for item in paragraphs if str(item.get(field) or "").strip())


def _group_first_available_text(paragraphs: Sequence[dict[str, Any]], fields: Sequence[str]) -> str:
    for field in fields:
        value = _group_text(paragraphs, field)
        if value:
            return value
    return ""


def _group_duration(paragraphs: Sequence[dict[str, Any]]) -> float:
    if not paragraphs:
        return 0.0
    starts = [_safe_float(item.get("start")) for item in paragraphs]
    ends = [_safe_float(item.get("end"), start) for item, start in zip(paragraphs, starts)]
    return max(0.0, max(ends) - min(starts))


def _group_languages(paragraphs: Sequence[dict[str, Any]]) -> list[str]:
    return [normalize_language(item.get("language")) for item in paragraphs]


def _group_variants(paragraphs: Sequence[dict[str, Any]]) -> list[str | None]:
    return [normalize_variant(item.get("speech_variant")) for item in paragraphs]


def _match_cost(sample: dict[str, Any], paragraphs: Sequence[dict[str, Any]]) -> float:
    """Cost for mapping one reference segment to one or more actual paragraphs."""

    if not paragraphs:
        return 1.5
    expected_language = normalize_language(sample.get("language"))
    actual_languages = _group_languages(paragraphs)
    if actual_languages and all(value == expected_language for value in actual_languages):
        language_cost = 0.0
    elif expected_language in actual_languages:
        language_cost = 0.35
    else:
        language_cost = 1.0

    expected_duration = max(0.1, _safe_float(sample.get("duration_seconds"), 0.0))
    actual_duration = _group_duration(paragraphs)
    duration_cost = min(1.5, abs(actual_duration - expected_duration) / expected_duration)

    text_score = score_text(_reference_text(sample), _group_text(paragraphs, "text"), expected_language)
    text_cost = min(2.0, _safe_float(text_score["error_rate"], 2.0)) / 2.0
    grouping_cost = 0.16 * max(0, len(paragraphs) - 1)
    return 0.55 * text_cost + 0.2 * language_cost + 0.15 * duration_cost + grouping_cost


def align_samples(
    samples: Sequence[dict[str, Any]],
    paragraphs: Sequence[dict[str, Any]],
    *,
    max_group_size: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Align ordered reference samples to ordered actual paragraphs.

    The dynamic program allows one reference segment to map to up to three
    adjacent actual paragraphs.  This is important for live ASR, where a
    language boundary or a late final result can split one speaker turn.
    """

    references = list(samples)
    actual = list(paragraphs)
    group_limit = max(1, min(max_group_size, 5))

    @lru_cache(maxsize=None)
    def solve(reference_index: int, actual_index: int) -> tuple[float, tuple[tuple[str, int, int], ...]]:
        if reference_index >= len(references):
            remaining = tuple(("extra", reference_index, index) for index in range(actual_index, len(actual)))
            return 0.85 * len(remaining), remaining
        if actual_index >= len(actual):
            remaining = tuple(("missing", index, actual_index) for index in range(reference_index, len(references)))
            return 1.5 * len(remaining), remaining

        candidates: list[tuple[float, tuple[tuple[str, int, int], ...]]] = []
        for size in range(1, min(group_limit, len(actual) - actual_index) + 1):
            tail_cost, tail_path = solve(reference_index + 1, actual_index + size)
            group = actual[actual_index:actual_index + size]
            candidates.append(
                (
                    _match_cost(references[reference_index], group) + tail_cost,
                    (("match", reference_index, size),) + tail_path,
                )
            )

        missing_cost, missing_path = solve(reference_index + 1, actual_index)
        candidates.append((1.5 + missing_cost, (("missing", reference_index, 0),) + missing_path))

        extra_cost, extra_path = solve(reference_index, actual_index + 1)
        candidates.append((0.85 + extra_cost, (("extra", reference_index, 1),) + extra_path))
        return min(candidates, key=lambda item: item[0])

    total_cost, path = solve(0, 0)
    reference_entries: list[dict[str, Any]] = []
    extras: list[dict[str, Any]] = []
    actual_index = 0
    for operation, reference_index, size in path:
        if operation == "match":
            group = actual[actual_index:actual_index + size]
            reference_entries.append(
                {
                    "expected_index": reference_index + 1,
                    "sample": references[reference_index],
                    "actual_indices": list(range(actual_index, actual_index + size)),
                    "actual_paragraphs": group,
                    "alignment_status": "matched",
                    "alignment_cost": round(_match_cost(references[reference_index], group), 6),
                }
            )
            actual_index += size
        elif operation == "missing":
            reference_entries.append(
                {
                    "expected_index": reference_index + 1,
                    "sample": references[reference_index],
                    "actual_indices": [],
                    "actual_paragraphs": [],
                    "alignment_status": "missing",
                    "alignment_cost": 1.5,
                }
            )
        else:
            extras.append(
                {
                    "actual_index": actual_index,
                    "paragraph": actual[actual_index],
                    "alignment_cost": 0.85,
                }
            )
            actual_index += 1
    return reference_entries, extras, round(total_cost, 6)


def _language_result(sample: dict[str, Any], paragraphs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    expected = normalize_language(sample.get("language"))
    actual = _group_languages(paragraphs)
    return {
        "expected": expected,
        "actual": actual,
        "correct": bool(actual) and all(value == expected for value in actual),
        "present": expected in actual,
    }


def _variant_result(sample: dict[str, Any], paragraphs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    expected = normalize_variant(sample.get("speech_variant"))
    actual = _group_variants(paragraphs)
    emitted = [value for value in actual if value]
    if expected == "sichuan":
        correct = "sichuan" in emitted
    elif expected == "mandarin":
        correct = bool(actual) and all(value in {None, "mandarin"} for value in actual)
    else:
        correct = True
    return {
        "expected": expected,
        "actual": actual,
        "emitted": bool(emitted),
        "correct": correct,
    }


def _translation_result(sample: dict[str, Any], paragraphs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reference = _reference_translation(sample)
    needed = bool(reference)
    actual = _group_text(paragraphs, "translation_zh")
    statuses = [str(item.get("translation_status") or "").strip() for item in paragraphs]
    ready = needed and bool(actual) and bool(statuses) and all(value == "ready" for value in statuses)
    status = "not_needed" if not needed else "ready" if ready else "partial" if actual else "missing"
    return {
        "needed": needed,
        "reference": reference,
        "actual": actual,
        "statuses": statuses,
        "status": status,
        "success": ready,
        "chrf": chrf(reference, actual) if needed else None,
    }


def _mandarin_semantic_result(sample: dict[str, Any], paragraphs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Score optional Mandarin meaning text without requiring another model.

    This is a character-level proxy, not a semantic-model judgment.  The
    report clearly distinguishes the result so a later human or embedding
    based evaluator can replace it without changing the reference schema.
    """

    reference = _reference_mandarin_text(sample)
    if not reference:
        return {
            "needed": False,
            "scored": False,
            "reference": "",
            "actual": "",
            "status": "not_annotated",
            "error_rate": None,
            "chrf": None,
        }
    actual = _group_first_available_text(paragraphs, ("mandarin_text", "semantic_text"))
    if not actual:
        return {
            "needed": True,
            "scored": False,
            "reference": reference,
            "actual": "",
            "status": "missing_hypothesis",
            "error_rate": None,
            "chrf": None,
        }
    score = score_text(reference, actual, "zh")
    return {
        "needed": True,
        "scored": True,
        "reference": reference,
        "actual": actual,
        "status": "scored",
        "error_rate": score["error_rate"],
        "chrf": score["chrf"],
    }


def _evaluate_entry(entry: dict[str, Any]) -> dict[str, Any]:
    sample = entry["sample"]
    paragraphs = entry["actual_paragraphs"]
    language = normalize_language(sample.get("language"))
    reference = _reference_text(sample)
    hypothesis = _group_text(paragraphs, "text")
    asr = score_text(reference, hypothesis, language)
    if entry["alignment_status"] == "missing":
        asr["edit_distance"] = int(asr["reference_tokens"])
        asr["error_rate"] = 1.0
        asr["chrf"] = 0.0
    actual_ids = [_paragraph_id(item, index + 1) for index, item in zip(entry["actual_indices"], paragraphs)]
    starts = [_safe_float(item.get("start")) for item in paragraphs]
    ends = [_safe_float(item.get("end"), start) for item, start in zip(paragraphs, starts)]
    language_result = _language_result(sample, paragraphs)
    variant_result = _variant_result(sample, paragraphs)
    translation_result = _translation_result(sample, paragraphs)
    mandarin_semantic_result = _mandarin_semantic_result(sample, paragraphs)
    return {
        "expected_index": entry["expected_index"],
        "sample_id": str(sample.get("sample_id") or sample.get("segment_id") or entry["expected_index"]),
        "alignment_status": entry["alignment_status"],
        "alignment_cost": entry["alignment_cost"],
        "actual_paragraph_ids": actual_ids,
        "actual_start": round(min(starts), 3) if starts else None,
        "actual_end": round(max(ends), 3) if ends else None,
        "reference_duration_seconds": round(_safe_float(sample.get("duration_seconds")), 3),
        "actual_duration_seconds": round(_group_duration(paragraphs), 3) if paragraphs else 0.0,
        "reference_language": language,
        "reference_speech_variant": normalize_variant(sample.get("speech_variant")),
        "asr": asr,
        "surface_asr": asr,
        "language": language_result,
        "speech_variant": variant_result,
        "translation": translation_result,
        "mandarin_semantic": mandarin_semantic_result,
    }


def evaluate_postprocess_api(postprocess_api: dict[str, Any] | None) -> dict[str, Any]:
    """Check that exactly the two requested post-processing APIs completed."""

    payload = postprocess_api or {}
    summary = payload.get("summary") or {}
    todo = payload.get("todo") or {}
    summary_request = summary.get("request") or {}
    todo_request = todo.get("request") or {}
    final_state = payload.get("final_state") or {}
    checks = {
        "request_count_is_two": payload.get("request_count") == 2,
        "summary_http_202": summary_request.get("status_code") == 202,
        "todo_http_202": todo_request.get("status_code") == 202,
        "summary_task_complete": summary.get("task", {}).get("status") == "complete",
        "todo_task_complete": todo.get("task", {}).get("status") == "complete",
        "summary_state_complete": final_state.get("summary_state") == "complete",
        "todo_state_complete": final_state.get("todo_state") == "complete",
        "summary_has_no_error": not final_state.get("summary_error"),
        "todo_has_no_error": not final_state.get("todo_error"),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "request_count": payload.get("request_count"),
        "request_statuses": {
            "summary": summary_request.get("status_code"),
            "todo": todo_request.get("status_code"),
        },
        "task_statuses": {
            "summary": (summary.get("task") or {}).get("status"),
            "todo": (todo.get("task") or {}).get("status"),
        },
        "final_states": {
            "summary": final_state.get("summary_state"),
            "todo": final_state.get("todo_state"),
        },
    }


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(statistics.mean(values), 6) if values else None


def evaluate_realtime_replay(
    manifest: dict[str, Any],
    replay_report: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a saved replay report against its generated meeting manifest."""

    samples = [item for item in (manifest.get("samples") or []) if isinstance(item, dict)]
    paragraphs = [item for item in (replay_report.get("paragraphs") or []) if isinstance(item, dict)]
    entries, extras, alignment_cost = align_samples(samples, paragraphs)
    scored = [_evaluate_entry(item) for item in entries]

    reference_token_total = sum(int(item["asr"]["reference_tokens"]) for item in scored)
    edit_distance_total = sum(int(item["asr"]["edit_distance"]) for item in scored)
    language_records = [item for item in scored if item["reference_language"] in SUPPORTED_LANGUAGES]
    translation_records = [item for item in scored if item["translation"]["needed"]]
    sichuan_records = [item for item in scored if item["reference_speech_variant"] == "sichuan"]
    mandarin_reference_records = [item for item in sichuan_records if item["mandarin_semantic"]["needed"]]
    mandarin_scored_records = [item for item in mandarin_reference_records if item["mandarin_semantic"]["scored"]]
    matched = [item for item in scored if item["alignment_status"] == "matched"]

    summary = {
        "reference_samples": len(samples),
        "actual_paragraphs": len(paragraphs),
        "paragraph_count_delta": len(paragraphs) - len(samples),
        "matched_samples": len(matched),
        "missing_samples": sum(item["alignment_status"] == "missing" for item in scored),
        "extra_paragraphs": len(extras),
        "sample_coverage": round(len(matched) / max(1, len(samples)), 6),
        "one_to_one_alignment_rate": round(
            sum(len(item["actual_paragraph_ids"]) == 1 for item in matched) / max(1, len(samples)),
            6,
        ),
        "alignment_cost": alignment_cost,
        "asr_macro_error_rate": _mean(float(item["asr"]["error_rate"]) for item in scored),
        "asr_token_weighted_error_rate": round(edit_distance_total / max(1, reference_token_total), 6),
        "asr_chrf_mean": _mean(float(item["asr"]["chrf"]) for item in scored),
        "language_accuracy": round(sum(bool(item["language"]["correct"]) for item in language_records) / max(1, len(language_records)), 6),
        "language_presence_accuracy": round(sum(bool(item["language"]["present"]) for item in language_records) / max(1, len(language_records)), 6),
        "language_samples": len(language_records),
        "sichuan_expected_samples": len(sichuan_records),
        "sichuan_variant_emission_rate": round(
            sum(bool(item["speech_variant"]["emitted"]) for item in sichuan_records) / max(1, len(sichuan_records)),
            6,
        ),
        "sichuan_variant_accuracy": round(
            sum(bool(item["speech_variant"]["correct"]) for item in sichuan_records) / max(1, len(sichuan_records)),
            6,
        ),
        "sichuan_surface_error_rate": _mean(float(item["surface_asr"]["error_rate"]) for item in sichuan_records),
        "sichuan_surface_chrf_mean": _mean(float(item["surface_asr"]["chrf"]) for item in sichuan_records),
        "sichuan_mandarin_reference_samples": len(mandarin_reference_records),
        "sichuan_mandarin_scored_samples": len(mandarin_scored_records),
        "sichuan_mandarin_error_rate": _mean(
            float(item["mandarin_semantic"]["error_rate"]) for item in mandarin_scored_records
        ),
        "sichuan_mandarin_chrf_mean": _mean(
            float(item["mandarin_semantic"]["chrf"]) for item in mandarin_scored_records
        ),
        "translation_samples": len(translation_records),
        "translation_success_rate": round(
            sum(bool(item["translation"]["success"]) for item in translation_records) / max(1, len(translation_records)),
            6,
        ) if translation_records else None,
        "translation_chrf_mean": _mean(
            float(item["translation"]["chrf"] or 0.0) for item in translation_records
        ) if translation_records else None,
    }
    api = evaluate_postprocess_api(replay_report.get("postprocess_api"))
    runtime_metrics = replay_report.get("runtime_metrics") or {}
    evaluation_contract = manifest.get("evaluation_contract") or {}
    postprocess_api_required = bool(evaluation_contract.get("postprocess_api_required", True))
    contract_checks = {
        "recording_complete": replay_report.get("recording_state") == "complete",
        "reference_samples_aligned": len(matched) == len(samples),
        "postprocess_api_complete": api["passed"] if postprocess_api_required else True,
        "runtime_stage_failures_empty": not runtime_metrics.get("stage_failures"),
    }
    return {
        "schema_version": "1.0-realtime-automatic-evaluation",
        "evaluator": {
            "alignment": "ordered_dynamic_programming",
            "max_actual_paragraphs_per_reference": 3,
            "reference_text_field": "text_sichuan_or_reference_text_or_text",
            "mandarin_reference_field": "text_mandarin_or_reference_mandarin_text",
            "translation_reference_field": "reference_translation_or_translation",
        },
        "summary": summary,
        "contract": {
            "passed": all(contract_checks.values()),
            "postprocess_api_required": postprocess_api_required,
            "checks": contract_checks,
        },
        "postprocess_api": api,
        "segments": scored,
        "extra_paragraphs": [
            {
                "actual_index": item["actual_index"] + 1,
                "paragraph_id": _paragraph_id(item["paragraph"], item["actual_index"] + 1),
                "language": normalize_language(item["paragraph"].get("language")),
                "start": _safe_float(item["paragraph"].get("start")),
                "end": _safe_float(item["paragraph"].get("end")),
                "text_preview": str(item["paragraph"].get("text") or "")[:160],
            }
            for item in extras
        ],
    }

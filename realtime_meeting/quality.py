from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any


_UNIT_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-zÀ-ÖØ-öø-ÿ0-9]+")
_REPEATED_FRAGMENT_RE = re.compile(r"(.{2,24})(?:\1){2,}", re.IGNORECASE)


def _units(text: str) -> list[str]:
    return [value.casefold() for value in _UNIT_RE.findall(str(text or ""))]


def _normalized_text(text: str) -> str:
    return " ".join(_units(text))


def normalized_change(previous: str, current: str) -> float:
    """Return a language-neutral edit/churn ratio in the range 0..1."""

    left = _normalized_text(previous)
    right = _normalized_text(current)
    if not left and not right:
        return 0.0
    if not left or not right:
        return 1.0
    return max(0.0, min(1.0, 1.0 - SequenceMatcher(None, left, right).ratio()))


@dataclass(slots=True)
class AsrQualityState:
    """Small in-memory history used to score a final paragraph."""

    previous_text: str = ""
    max_partial_change: float = 0.0
    final_change: float = 0.0
    update_count: int = 0
    partial_count: int = 0
    confidence_sum: float = 0.0
    confidence_count: int = 0
    language_conflicts: int = 0
    empty_results: int = 0

    def observe(
        self,
        text: str,
        confidence: float,
        *,
        is_final: bool,
        is_partial: bool,
        language_conflict: bool = False,
    ) -> None:
        current = str(text or "").strip()
        if self.previous_text and current:
            change = normalized_change(self.previous_text, current)
            self.max_partial_change = max(self.max_partial_change, change)
            if is_final:
                self.final_change = change
        self.previous_text = current
        self.update_count += 1
        self.partial_count += int(is_partial)
        value = float(confidence or 0.0)
        if value > 0.0:
            self.confidence_sum += max(0.0, min(1.0, value))
            self.confidence_count += 1
        if not current:
            self.empty_results += 1
        if language_conflict:
            self.language_conflicts += 1


@dataclass(frozen=True, slots=True)
class AsrQualityAssessment:
    score: float
    reasons: tuple[str, ...] = ()
    signals: dict[str, float] = field(default_factory=dict)

    @property
    def is_low(self) -> bool:
        return self.score < 0.62 or bool(
            set(self.reasons) & {"unstable_partial", "final_changed", "repetition", "language_conflict"}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
            "signals": {key: round(value, 4) for key, value in self.signals.items()},
        }


def assess_asr_quality(
    text: str,
    *,
    start: float,
    end: float,
    confidence: float = 0.0,
    state: AsrQualityState | None = None,
) -> AsrQualityAssessment:
    """Score final ASR text using conservative, language-neutral signals.

    The score is intentionally a routing signal for a post-meeting pass, not a
    claim that the transcript is objectively correct. Thresholds should be
    calibrated with labelled recordings once production samples are available.
    """

    history = state or AsrQualityState()
    units = _units(text)
    duration = max(0.0, float(end) - float(start))
    average_confidence = (
        history.confidence_sum / history.confidence_count
        if history.confidence_count
        else max(0.0, min(1.0, float(confidence or 0.0)))
    )
    confidence_signal = average_confidence if average_confidence > 0.0 else 0.65

    if history.update_count <= 1:
        stability_signal = 0.68
    else:
        stability_signal = max(0.0, 1.0 - min(1.0, history.max_partial_change))

    expected_units = max(1.0, duration * 1.2)
    coverage_signal = max(0.0, min(1.0, len(units) / expected_units)) if units else 0.0
    if duration <= 0.9 and units:
        coverage_signal = max(coverage_signal, 0.7)

    normalized = _normalized_text(text)
    repeated = bool(normalized and _REPEATED_FRAGMENT_RE.search(normalized.replace(" ", "")))
    sanity_signal = 0.2 if repeated else 1.0 if units else 0.0
    language_signal = max(0.0, 1.0 - min(1.0, history.language_conflicts / 3.0))

    score = (
        confidence_signal * 0.30
        + stability_signal * 0.25
        + coverage_signal * 0.15
        + language_signal * 0.15
        + sanity_signal * 0.15
    )
    score = max(0.0, min(1.0, score))
    reasons: list[str] = []
    if history.confidence_count and confidence_signal < 0.55:
        reasons.append("low_confidence")
    if history.max_partial_change > 0.45:
        reasons.append("unstable_partial")
    if history.final_change > 0.35:
        reasons.append("final_changed")
    if coverage_signal < 0.35:
        reasons.append("sparse_transcript")
    if repeated:
        reasons.append("repetition")
    if history.language_conflicts:
        reasons.append("language_conflict")
    if not units:
        reasons.append("empty_text")

    return AsrQualityAssessment(
        score=score,
        reasons=tuple(reasons),
        signals={
            "confidence": confidence_signal,
            "stability": stability_signal,
            "coverage": coverage_signal,
            "language_consistency": language_signal,
            "text_sanity": sanity_signal,
        },
    )

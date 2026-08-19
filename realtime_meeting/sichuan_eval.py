"""Utilities for preparing the public WenetSpeech-Chuan ASR benchmark.

The upstream WSC-Eval-ASR release uses Kaldi-style ``text`` and ``wav.scp``
files.  This module converts those files into the manifest shape used by this
project while keeping two text fields separate:

* ``text_sichuan``: the surface transcription used for dialect ASR scoring;
* ``text_mandarin``: an optional meaning-normalized Mandarin reference.

The public benchmark only provides the first field.  The second field is
intentionally left empty until a human annotator supplies it; inventing a
Mandarin paraphrase from the ASR transcript would turn the semantic score into
an evaluation of an annotation heuristic.
"""

from __future__ import annotations

import json
import os
import re
import wave
from pathlib import Path
from typing import Any


WSC_EVAL_REPO_ID = "ASLP-lab/WSC-Eval"
WSC_EVAL_URL = "https://huggingface.co/datasets/ASLP-lab/WSC-Eval"
WSC_EVAL_ASR_URL = "https://huggingface.co/datasets/ASLP-lab/WSC-Eval/blob/main/WSC-Eval-ASR/readme.md"
WSC_SUBSETS = ("Easy", "Hard", "Long", "Short")
WSC_NORMALIZATION = "wsc-eval-asr-v1"


class WscEvalError(ValueError):
    """Raised when a WSC-Eval-ASR export cannot be converted safely."""


def _read_kaldi_map(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise WscEvalError(f"缺少 WSC-Eval 文件: {path}")
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise WscEvalError(f"{path}:{line_number} 不是 key value 格式")
        key, value = parts
        if key in result:
            raise WscEvalError(f"{path}:{line_number} 出现重复 key: {key}")
        result[key] = value.strip()
    return result


def _read_kaldi_keys(path: Path) -> list[str]:
    if not path.is_file():
        return []
    keys: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        key = raw_line.strip().split(maxsplit=1)[0] if raw_line.strip() else ""
        if key and key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def _wav_spec_path(spec: str) -> str:
    """Extract a path from a Kaldi wav.scp value.

    WSC-Eval-ASR currently contains plain paths.  Keeping the first command
    separator here makes failures explicit if a future export contains a
    shell command instead of a local file path.
    """

    value = spec.strip()
    if not value:
        raise WscEvalError("wav.scp 中存在空音频路径")
    if "|" in value:
        raise WscEvalError(f"暂不支持 wav.scp 音频命令: {value}")
    return value


def _resolve_wav_path(dataset_root: Path, subset_dir: Path, spec: str) -> Path:
    raw = Path(_wav_spec_path(spec))
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(dataset_root / raw)
        parts = list(raw.parts)
        if parts and parts[0].casefold() == "eval":
            candidates.append(dataset_root.joinpath(*parts[1:]))
        candidates.append(subset_dir / "wav" / raw.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    # Metadata-only preparation is supported.  Use the canonical local path
    # so a later audio download produces a manifest with the same references.
    if raw.is_absolute():
        return raw
    return (subset_dir / "wav" / raw.name).resolve()


def _wav_duration_seconds(path: Path) -> float | None:
    if not path.is_file():
        return None
    try:
        with wave.open(str(path), "rb") as source:
            frame_rate = source.getframerate()
            frame_count = source.getnframes()
    except (wave.Error, OSError):
        return None
    if frame_rate <= 0:
        return None
    return round(frame_count / frame_rate, 6)


def _relative_path(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
        return relative.as_posix()
    except ValueError:
        return path.as_posix()


def _sample_id(subset: str, source_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id).strip("_") or "sample"
    return f"wsc_{subset.casefold()}_{safe_id}"


def load_wsc_eval_subset(dataset_root: Path, subset: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Read one WSC-Eval-ASR subset into project-compatible reference samples."""

    subset_name = str(subset).strip().title()
    if subset_name not in WSC_SUBSETS:
        raise WscEvalError(f"不支持的 WSC-Eval 子集: {subset}; 可选值: {', '.join(WSC_SUBSETS)}")
    root = Path(dataset_root).expanduser().resolve()
    subset_dir = root / subset_name
    text_map = _read_kaldi_map(subset_dir / "text")
    wav_map = _read_kaldi_map(subset_dir / "wav.scp")
    ordered_keys = _read_kaldi_keys(subset_dir / "key.txt") or list(text_map)
    missing_text = [key for key in ordered_keys if key not in text_map]
    missing_wav = [key for key in ordered_keys if key not in wav_map]
    if missing_text or missing_wav:
        details = []
        if missing_text:
            details.append(f"text 缺少 {missing_text[:3]}")
        if missing_wav:
            details.append(f"wav.scp 缺少 {missing_wav[:3]}")
        raise WscEvalError(f"{subset_name} 的 Kaldi 文件不一致: {'; '.join(details)}")

    selected_keys = ordered_keys[:limit] if limit is not None else ordered_keys
    samples: list[dict[str, Any]] = []
    for index, source_id in enumerate(selected_keys, 1):
        audio_path = _resolve_wav_path(root, subset_dir, wav_map[source_id])
        duration = _wav_duration_seconds(audio_path)
        sample: dict[str, Any] = {
            "sample_id": _sample_id(subset_name, source_id),
            "source_id": source_id,
            "dataset": "WSC-Eval-ASR",
            "dataset_repo": WSC_EVAL_REPO_ID,
            "subset": subset_name,
            "audio_path": _relative_path(audio_path, root),
            "language": "zh",
            "speech_variant": "sichuan",
            "reference_text": text_map[source_id],
            "text_sichuan": text_map[source_id],
            "text_mandarin": None,
            "reference_text_normalization": WSC_NORMALIZATION,
            "annotation_status": "surface_only",
            "annotation": {
                "surface_text": "WSC-Eval-ASR/text",
                "mandarin_text": "not_provided",
                "uncertainty_marker": "*",
                "non_primary_speaker_marker": "parentheses",
                "punctuation": "omitted_by_upstream",
            },
            "source_order": index,
        }
        if duration is not None:
            sample["duration_seconds"] = duration
        samples.append(sample)
    return samples


def build_wsc_eval_manifest(
    dataset_root: Path,
    subset: str,
    *,
    output_path: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build a manifest without copying or embedding third-party audio."""

    if limit is not None and limit <= 0:
        raise WscEvalError("limit 必须大于 0")
    root = Path(dataset_root).expanduser().resolve()
    samples = load_wsc_eval_subset(root, subset, limit=limit)
    if output_path is not None:
        output_parent = Path(output_path).expanduser().resolve().parent
        for sample in samples:
            audio_file = (root / str(sample["audio_path"])).resolve()
            sample["audio_path"] = Path(os.path.relpath(audio_file, output_parent)).as_posix()
    durations = [float(item["duration_seconds"]) for item in samples if item.get("duration_seconds") is not None]
    manifest: dict[str, Any] = {
        "schema_version": "1.0-sichuan-wsc-eval",
        "dataset": {
            "name": "WSC-Eval-ASR",
            "repo_id": WSC_EVAL_REPO_ID,
            "source_url": WSC_EVAL_URL,
            "annotation_url": WSC_EVAL_ASR_URL,
            "subset": str(subset).strip().title(),
            "audio_is_external": True,
        },
        "reference_schema": {
            "surface_text_field": "text_sichuan",
            "mandarin_text_field": "text_mandarin",
            "semantic_score": "optional_until_mandarin_reference_is_added",
            "wsc_normalization": WSC_NORMALIZATION,
        },
        "evaluation_contract": {
            "postprocess_api_required": False,
            "audio_replay_required": True,
        },
        "sample_count": len(samples),
        "recording_seconds": round(sum(durations), 6) if len(durations) == len(samples) else None,
        "samples": samples,
    }
    return manifest


def write_wsc_eval_manifest(
    dataset_root: Path,
    subset: str,
    output_path: Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    output = Path(output_path).expanduser().resolve()
    manifest = build_wsc_eval_manifest(dataset_root, subset, output_path=output, limit=limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def validate_wsc_eval_manifest(
    manifest: dict[str, Any], *, require_audio: bool = False, base_dir: Path | None = None
) -> list[str]:
    """Return validation errors instead of silently accepting incomplete data."""

    errors: list[str] = []
    if manifest.get("schema_version") != "1.0-sichuan-wsc-eval":
        errors.append("schema_version 不是 1.0-sichuan-wsc-eval")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append("samples 为空或不是数组")
        return errors
    for index, sample in enumerate(samples, 1):
        if not isinstance(sample, dict):
            errors.append(f"samples[{index}] 不是对象")
            continue
        for field in ("sample_id", "audio_path", "text_sichuan"):
            if not str(sample.get(field) or "").strip():
                errors.append(f"samples[{index}] 缺少 {field}")
        if sample.get("language") != "zh":
            errors.append(f"samples[{index}] language 必须是 zh")
        if sample.get("speech_variant") != "sichuan":
            errors.append(f"samples[{index}] speech_variant 必须是 sichuan")
        if require_audio:
            audio_path = Path(str(sample.get("audio_path") or ""))
            if base_dir is not None and not audio_path.is_absolute():
                audio_path = Path(base_dir) / audio_path
            if not audio_path.is_file():
                errors.append(f"samples[{index}] 音频不存在: {sample.get('audio_path')}")
    return errors

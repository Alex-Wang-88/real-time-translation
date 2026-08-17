"""Collect long meeting and Mandarin-English code-switching benchmark audio.

The source rows are transcript-backed public samples from the Hugging Face
Dataset Viewer.  AMI IHM rows are ordered by meeting time and are stitched
into speech-only multi-speaker blocks.  ASCEND rows are ordered by session and
are stitched into long Chinese-English code-switching blocks.  No noise is
added; the benchmark intentionally focuses on long-context and speaker/language
turn changes.

The source clips remain separately cached so a failed or interrupted run can
resume without downloading the same audio again.  The final manifest records
stable dataset coordinates and the exact source turns used in each block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
AMI_DATASET = "edinburghcstr/ami"
AMI_CONFIG = "ihm"
AMI_SPLIT = "test"
ASCEND_DATASET = "CAiRE/ASCEND"
ASCEND_CONFIG = "main"
ASCEND_SPLIT = "test"
USER_AGENT = "real-time-translation-long-architecture-benchmark/1.0"


def _slug(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z\u3400-\u9fff]+", "_", str(value)).strip("_")
    return value.casefold() or "sample"


def _get_json(url: str, *, retries: int = 10) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError(f"unexpected JSON payload from {url}")
            return payload
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {exc.code} while fetching {url}") from exc
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if retry_after:
                try:
                    delay = min(60.0, max(1.0, float(retry_after)))
                except ValueError:
                    delay = min(60.0, 2.0**attempt)
            else:
                delay = min(60.0, 2.0**attempt)
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            delay = min(60.0, 1.5**attempt)
        if attempt + 1 < retries:
            time.sleep(delay)
    raise RuntimeError(f"failed to fetch {url}: {last_error}") from last_error


def _rows_url(dataset: str, config: str, split: str, offset: int, length: int = 100) -> str:
    query = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    )
    return f"{HF_ROWS_API}?{query}"


def _fetch_rows(dataset: str, config: str, split: str, offset: int) -> tuple[list[dict[str, Any]], str, int | None]:
    url = _rows_url(dataset, config, split, offset)
    payload = _get_json(url)
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise RuntimeError(f"Dataset Viewer returned no rows for {url}: {payload}")
    rows: list[dict[str, Any]] = []
    for item in raw_rows:
        if not isinstance(item, dict) or not isinstance(item.get("row"), dict):
            continue
        row = dict(item["row"])
        row["_row_idx"] = int(item.get("row_idx", offset + len(rows)))
        rows.append(row)
    total = payload.get("num_rows_total")
    return rows, url, int(total) if isinstance(total, (int, float)) else None


def _load_or_fetch_rows(
    dataset: str,
    config: str,
    split: str,
    cache_path: Path,
    *,
    refresh: bool,
) -> list[dict[str, Any]]:
    if cache_path.is_file() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(payload, list) and payload:
            return [dict(item) for item in payload if isinstance(item, dict)]

    partial_path = cache_path.with_suffix(cache_path.suffix + ".partial")
    rows: list[dict[str, Any]] = []
    if partial_path.is_file() and not refresh:
        payload = json.loads(partial_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = [dict(item) for item in payload if isinstance(item, dict)]
    offset = len(rows)
    total: int | None = None
    while total is None or offset < total:
        page, _api_url, page_total = _fetch_rows(dataset, config, split, offset)
        if total is None:
            total = page_total
        if not page:
            break
        rows.extend(page)
        offset += len(page)
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(f"fetched {dataset}/{config}/{split}: {len(rows)}/{total or '?'} rows")
        # Dataset Viewer applies a fairly small per-client request budget.
        # A modest cadence prevents losing a long metadata crawl to 429s.
        time.sleep(2.0)
        if len(page) < 100:
            break
    if not rows:
        raise RuntimeError(f"no rows fetched for {dataset}/{config}/{split}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    partial_path.unlink(missing_ok=True)
    return rows


def _audio_src(row: dict[str, Any]) -> str:
    audio = row.get("audio")
    if isinstance(audio, list) and audio and isinstance(audio[0], dict):
        return str(audio[0].get("src", ""))
    if isinstance(audio, dict):
        return str(audio.get("src", ""))
    return ""


def _text(row: dict[str, Any], field: str) -> str:
    return re.sub(r"\s+", " ", str(row.get(field, "") or "")).strip()


def _number(row: dict[str, Any], field: str, default: float = 0.0) -> float:
    try:
        return float(row.get(field, default))
    except (TypeError, ValueError):
        return default


def _language_label(row: dict[str, Any], text: str) -> str:
    label = str(row.get("language", "") or "").strip().casefold()
    if label in {"zh", "en", "mixed"}:
        return label
    has_zh = bool(re.search(r"[\u3400-\u9fff]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    if has_zh and has_latin:
        return "mixed"
    if has_zh:
        return "zh"
    return "en" if has_latin else "unknown"


def _language_switches(rows: Iterable[dict[str, Any]], text_field: str) -> tuple[list[str], int]:
    labels = [_language_label(row, _text(row, text_field)) for row in rows]
    switches = sum(left != right for left, right in zip(labels, labels[1:]) if left != "unknown" and right != "unknown")
    return labels, switches


def _download(url: str, destination: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(6):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=180) as response, destination.open("wb") as target:
                shutil.copyfileobj(response, target)
            return
        except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            last_error = exc
            destination.unlink(missing_ok=True)
            if attempt + 1 < 6:
                time.sleep(min(12.0, 1.5**attempt))
    raise RuntimeError(f"failed to download audio after retries: {url}") from last_error


def _convert_to_wav(source: Path, destination: Path, ffmpeg: str) -> None:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed for {source.name}: {result.stderr[-1000:]}")


def _wav_info(path: Path) -> tuple[float, str]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getframerate() != 16_000:
            raise RuntimeError(f"unexpected WAV format: {path}")
        duration = source.getnframes() / 16_000
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return duration, digest


def _combine_wavs(paths: list[Path], destination: Path, silence_ms: int = 120) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    silence = b"\0\0" * round(16_000 * silence_ms / 1000)
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        for index, path in enumerate(paths):
            with wave.open(str(path), "rb") as source:
                if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getframerate() != 16_000:
                    raise RuntimeError(f"unexpected segment WAV format: {path}")
                output.writeframes(source.readframes(source.getnframes()))
            if index + 1 < len(paths):
                output.writeframes(silence)


def _speaker_key(row: dict[str, Any]) -> str:
    return str(row.get("speaker_id") or row.get("original_speaker_id") or "unknown")


def _ami_candidates(rows: list[dict[str, Any]], target_seconds: float) -> list[dict[str, Any]]:
    by_meeting: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        text = _text(row, "text")
        if _audio_src(row) and text and _number(row, "end_time") > _number(row, "begin_time"):
            by_meeting[str(row.get("meeting_id", "unknown"))].append(row)

    candidates: list[dict[str, Any]] = []
    for meeting_id, meeting_rows in sorted(by_meeting.items()):
        meeting_rows.sort(key=lambda row: (_number(row, "begin_time"), int(row.get("_row_idx", -1))))
        stride = max(8, round(target_seconds / 2.0))
        for start in range(0, len(meeting_rows), stride):
            selected: list[dict[str, Any]] = []
            speech_seconds = 0.0
            last_end: float | None = None
            for row in meeting_rows[start:]:
                begin = _number(row, "begin_time")
                end = _number(row, "end_time")
                if last_end is not None:
                    if begin < last_end - 0.10:
                        continue
                    if begin - last_end > 8.0 and speech_seconds < target_seconds * 0.75:
                        break
                selected.append(row)
                speech_seconds += max(0.05, end - begin)
                last_end = end
                if speech_seconds >= target_seconds:
                    break
            speakers = {_speaker_key(row) for row in selected}
            if speech_seconds < target_seconds * 0.75 or len(selected) < 8 or len(speakers) < 3:
                continue
            labels, switches = _language_switches(selected, "text")
            candidates.append(
                {
                    "scenario": "multi_person_meeting",
                    "group": "AMI multi-person meeting",
                    "language": "en",
                    "decode_language_hint": True,
                    "source_dataset": AMI_DATASET,
                    "source_config": AMI_CONFIG,
                    "source_split": AMI_SPLIT,
                    "source_dataset_url": f"https://huggingface.co/datasets/{AMI_DATASET}",
                    "meeting_id": meeting_id,
                    "reference_text": " ".join(_text(row, "text") for row in selected),
                    "reference_language_segments": labels,
                    "reference_language_switches": switches,
                    "speaker_count": len(speakers),
                    "turn_count": len(selected),
                    "speaker_switches": sum(
                        _speaker_key(left) != _speaker_key(right) for left, right in zip(selected, selected[1:])
                    ),
                    "_segments": [
                        {
                            "source_row_idx": int(row.get("_row_idx", -1)),
                            "source_id": str(row.get("audio_id", row.get("_row_idx", ""))),
                            "source_audio_url": _audio_src(row),
                            "speaker_id": _speaker_key(row),
                            "begin_time": _number(row, "begin_time"),
                            "end_time": _number(row, "end_time"),
                            "reference_text": _text(row, "text"),
                        }
                        for row in selected
                    ],
                    "source_speech_seconds": round(speech_seconds, 4),
                }
            )
    return candidates


def _ascend_candidates(rows: list[dict[str, Any]], target_seconds: float) -> list[dict[str, Any]]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        text = _text(row, "transcription")
        if _audio_src(row) and text and _number(row, "duration") > 0:
            by_session[str(row.get("session_id", "unknown"))].append(row)

    candidates: list[dict[str, Any]] = []
    for session_id, session_rows in sorted(by_session.items()):
        session_rows.sort(key=lambda row: int(row.get("_row_idx", 0)))
        stride = max(8, round(target_seconds / 3.0))
        for start in range(0, len(session_rows), stride):
            selected: list[dict[str, Any]] = []
            duration = 0.0
            for row in session_rows[start:]:
                row_duration = max(0.05, _number(row, "duration"))
                if duration >= target_seconds * 0.75 and duration + row_duration > target_seconds * 1.35:
                    break
                selected.append(row)
                duration += row_duration
                if duration >= target_seconds:
                    break
            labels, switches = _language_switches(selected, "transcription")
            label_set = set(labels)
            mixed_rows = sum(label == "mixed" for label in labels)
            speakers = {_speaker_key(row) for row in selected}
            if (
                duration < target_seconds * 0.75
                or len(selected) < 10
                or not {"zh", "en"}.issubset(label_set)
                or mixed_rows < 2
            ):
                continue
            candidates.append(
                {
                    "scenario": "mandarin_english_code_switch",
                    "group": "ASCEND Mandarin-English long mixed",
                    "language": "mixed",
                    "decode_language_hint": False,
                    "source_dataset": ASCEND_DATASET,
                    "source_config": ASCEND_CONFIG,
                    "source_split": ASCEND_SPLIT,
                    "source_dataset_url": f"https://huggingface.co/datasets/{ASCEND_DATASET}",
                    "session_id": session_id,
                    "topic": str(selected[0].get("topic", "")),
                    "reference_text": " ".join(_text(row, "transcription") for row in selected),
                    "reference_language_segments": labels,
                    "reference_language_switches": switches,
                    "mixed_segment_count": mixed_rows,
                    "speaker_count": len(speakers),
                    "turn_count": len(selected),
                    "speaker_switches": sum(
                        _speaker_key(left) != _speaker_key(right) for left, right in zip(selected, selected[1:])
                    ),
                    "_segments": [
                        {
                            "source_row_idx": int(row.get("_row_idx", -1)),
                            "source_id": str(row.get("id", row.get("_row_idx", ""))),
                            "source_audio_url": _audio_src(row),
                            "speaker_id": _speaker_key(row),
                            "language": label,
                            "duration_seconds": round(_number(row, "duration"), 4),
                            "reference_text": _text(row, "transcription"),
                        }
                        for row, label in zip(selected, labels)
                    ],
                    "source_speech_seconds": round(duration, 4),
                }
            )
    return candidates


def _select_diverse(candidates: list[dict[str, Any]], count: int, key: str) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    ordered = sorted(
        candidates,
        key=lambda item: (
            -int(item.get("speaker_count", 0)),
            -int(item.get("reference_language_switches", 0)),
            abs(_number(item, "source_speech_seconds") - 60.0),
        ),
    )
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    used_ranges: dict[str, list[tuple[int, int]]] = defaultdict(list)

    def row_range(item: dict[str, Any]) -> tuple[int, int] | None:
        values = [int(segment.get("source_row_idx", -1)) for segment in item.get("_segments", [])]
        values = [value for value in values if value >= 0]
        return (min(values), max(values)) if values else None

    def overlaps(identity: str, item: dict[str, Any]) -> bool:
        current = row_range(item)
        if current is None:
            return False
        return any(current[0] <= right and left <= current[1] for left, right in used_ranges[identity])

    for require_new_group in (True, False):
        for item in ordered:
            identity = str(item.get(key, "unknown"))
            if require_new_group and identity in used:
                continue
            if id(item) in {id(value) for value in chosen}:
                continue
            if not require_new_group and overlaps(identity, item):
                continue
            chosen.append(item)
            used.add(identity)
            current = row_range(item)
            if current is not None:
                used_ranges[identity].append(current)
            if len(chosen) >= count:
                return chosen
    return chosen


def _materialize_block(item: dict[str, Any], output_dir: Path, ffmpeg: str, index: int) -> None:
    source_dir = output_dir / "segments" / _slug(str(item["scenario"]))
    source_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    public_segments: list[dict[str, Any]] = []
    for segment in item.pop("_segments"):
        source_id = _slug(str(segment.get("source_id", segment.get("source_row_idx", ""))))
        segment_path = source_dir / f"{int(segment.get('source_row_idx', index)):06d}_{source_id}.wav"
        if not segment_path.is_file():
            with tempfile.NamedTemporaryFile(prefix="long-asr-benchmark-", suffix=".download", delete=False) as temporary:
                temporary_path = Path(temporary.name)
            try:
                _download(str(segment["source_audio_url"]), temporary_path)
                _convert_to_wav(temporary_path, segment_path, ffmpeg)
            finally:
                temporary_path.unlink(missing_ok=True)
        _wav_info(segment_path)
        paths.append(segment_path)
        public_segment = {key: value for key, value in segment.items() if key != "source_audio_url"}
        public_segments.append(public_segment)

    sample_id = f"{_slug(str(item['scenario']))}_{index:03d}"
    audio_path = output_dir / "audio" / f"{sample_id}.wav"
    _combine_wavs(paths, audio_path)
    duration, digest = _wav_info(audio_path)
    item["sample_id"] = sample_id
    item["audio_path"] = str(audio_path.relative_to(output_dir)).replace("\\", "/")
    item["duration_seconds"] = round(duration, 4)
    item["sha256"] = digest
    item["segments"] = public_segments
    item.pop("source_speech_seconds", None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("result/architecture_benchmark_long"))
    parser.add_argument("--meeting-blocks", type=int, default=6)
    parser.add_argument("--code-switch-blocks", type=int, default=6)
    parser.add_argument("--block-seconds", type=float, default=60.0)
    parser.add_argument("--force", action="store_true", help="rebuild block WAVs and refresh cached source rows")
    parser.add_argument("--refresh-source", action="store_true", help="re-fetch Dataset Viewer metadata")
    args = parser.parse_args()
    if args.meeting_blocks < 1 or args.code_switch_blocks < 1:
        parser.error("block counts must be positive")
    if args.block_seconds < 20 or args.block_seconds > 180:
        parser.error("--block-seconds must be between 20 and 180")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        parser.error("ffmpeg is required to normalize downloaded audio")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        for path in (output_dir / "audio").glob("*.wav"):
            path.unlink()

    cache_dir = output_dir / "source_cache"
    ami_rows = _load_or_fetch_rows(
        AMI_DATASET,
        AMI_CONFIG,
        AMI_SPLIT,
        cache_dir / "ami_ihm_test_rows.json",
        refresh=args.refresh_source or args.force,
    )
    ascend_rows = _load_or_fetch_rows(
        ASCEND_DATASET,
        ASCEND_CONFIG,
        ASCEND_SPLIT,
        cache_dir / "ascend_test_rows.json",
        refresh=args.refresh_source or args.force,
    )
    meeting_candidates = _ami_candidates(ami_rows, args.block_seconds)
    mixed_candidates = _ascend_candidates(ascend_rows, args.block_seconds)
    meetings = _select_diverse(meeting_candidates, args.meeting_blocks, "meeting_id")
    mixed = _select_diverse(mixed_candidates, args.code_switch_blocks, "session_id")
    if len(meetings) < args.meeting_blocks:
        raise RuntimeError(f"AMI could not provide {args.meeting_blocks} diverse blocks; got {len(meetings)}")
    if len(mixed) < args.code_switch_blocks:
        raise RuntimeError(f"ASCEND could not provide {args.code_switch_blocks} diverse blocks; got {len(mixed)}")

    samples = meetings + mixed
    for index, sample in enumerate(samples, 1):
        _materialize_block(sample, output_dir, ffmpeg, index)
        print(f"materialized {index}/{len(samples)} {sample['sample_id']} ({sample['duration_seconds']:.1f}s)")

    manifest = {
        "schema_version": "1.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collection": {
            "meeting_blocks": args.meeting_blocks,
            "code_switch_blocks": args.code_switch_blocks,
            "target_speech_seconds": args.block_seconds,
            "audio_format": "16 kHz mono PCM s16le WAV",
            "noise_policy": "No artificial noise; AMI IHM near-field source and clean ASCEND source clips",
            "viewer_api": HF_ROWS_API,
        },
        "sources": [
            {
                "dataset": AMI_DATASET,
                "config": AMI_CONFIG,
                "split": AMI_SPLIT,
                "purpose": "Multi-person English meeting turns, stitched by meeting time",
                "license_note": "Follow the AMI corpus and dataset repository terms for local use",
            },
            {
                "dataset": ASCEND_DATASET,
                "config": ASCEND_CONFIG,
                "split": ASCEND_SPLIT,
                "purpose": "Long Mandarin-English code-switching dialogue blocks",
                "license_note": "ASCEND is released under the dataset repository license; retain attribution",
            },
        ],
        "samples": samples,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = Counter(str(sample["scenario"]) for sample in samples)
    print(json.dumps({"output": str(manifest_path), "samples": len(samples), "scenarios": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

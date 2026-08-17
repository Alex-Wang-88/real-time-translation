"""Collect a small, balanced, reproducible ASR architecture benchmark set.

The benchmark intentionally uses public, transcript-backed test material from
several sources instead of relying on a single language or a single recording
condition.  Audio is fetched through the Hugging Face Dataset Viewer API and
converted locally to the format used by the meeting runtime.

The resulting manifest contains the source dataset, row index, reference text,
download URL and a SHA-256 checksum for every local WAV file.  The signed
viewer URLs are kept for provenance; they may expire, so the dataset/row fields
are the stable re-fetch coordinates.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
KE_SPEECH_DATASET = "miaocongxin/KeSpeech"
KE_SPEECH_PAGE_OFFSETS = [
    0,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    10000,
    11000,
    12000,
    13000,
    14000,
    15000,
    16000,
    17000,
    18000,
    19000,
]
KE_SPEECH_GROUPS = [
    "Mandarin",
    "Northeastern",
    "Southwestern",
    "Zhongyuan",
    "Jiang-Huai",
    "Lan-Yin",
    "Ji-Lu",
    "Jiao-Liao",
]


def _slug(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z\u3400-\u9fff]+", "_", str(value)).strip("_")
    return value.casefold() or "sample"


def _get_json(url: str, *, retries: int = 5) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "real-time-translation-architecture-benchmark/1.0",
        },
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {exc.code} while fetching {url}") from exc
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt + 1 < retries:
            time.sleep(min(8.0, 1.5**attempt))
    raise RuntimeError(f"failed to fetch {url}: {last_error}") from last_error


def _rows_url(dataset: str, config: str, split: str, offset: int, length: int = 100) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    return f"{HF_ROWS_API}?{query}"


def _fetch_rows(dataset: str, config: str, split: str, offset: int) -> tuple[list[dict[str, Any]], str]:
    url = _rows_url(dataset, config, split, offset)
    payload = _get_json(url)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"Dataset Viewer returned no rows for {url}: {payload}")
    return [item for item in rows if isinstance(item, dict)], url


def _audio_src(row: dict[str, Any]) -> str:
    audio = row.get("audio")
    if isinstance(audio, list) and audio and isinstance(audio[0], dict):
        return str(audio[0].get("src", ""))
    if isinstance(audio, dict):
        return str(audio.get("src", ""))
    return ""


def _text_is_useful(text: str, *, minimum: int = 6) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    letters = sum(char.isalnum() or "\u3400" <= char <= "\u9fff" for char in compact)
    return letters >= minimum


def _ke_speech_sample(row_item: dict[str, Any], api_url: str) -> dict[str, Any] | None:
    row = row_item.get("row")
    if not isinstance(row, dict):
        return None
    dialect = str(row.get("Dialect", "")).strip()
    text = str(row.get("Text", "")).strip()
    audio_url = _audio_src(row)
    if dialect not in KE_SPEECH_GROUPS or not audio_url or not _text_is_useful(text, minimum=6):
        return None
    sample_key = str(row.get("ID", row_item.get("row_idx", "")))
    return {
        "source_dataset": KE_SPEECH_DATASET,
        "source_config": "default",
        "source_split": "test",
        "source_row_idx": int(row_item.get("row_idx", -1)),
        "source_api_url": api_url,
        "source_dataset_url": f"https://huggingface.co/datasets/{KE_SPEECH_DATASET}",
        "source_audio_url": audio_url,
        "source_id": sample_key,
        "group": dialect,
        "language": "zh",
        "speech_variant": None,
        "reference_text": text,
        "speaker_key": sample_key.split("_", 1)[0],
    }


def _generic_sample(
    row_item: dict[str, Any],
    api_url: str,
    *,
    dataset: str,
    config: str,
    split: str,
    group: str,
    language: str,
    text_field: str,
    id_field: str,
    speaker_field: str | None = None,
    speech_variant: str | None = None,
    minimum_text_length: int = 8,
) -> dict[str, Any] | None:
    row = row_item.get("row")
    if not isinstance(row, dict):
        return None
    text = str(row.get(text_field, "") or "").strip()
    audio_url = _audio_src(row)
    if not audio_url or not _text_is_useful(text, minimum=minimum_text_length):
        return None
    source_id = str(row.get(id_field, row_item.get("row_idx", "")))
    speaker = str(row.get(speaker_field, "")) if speaker_field else ""
    return {
        "source_dataset": dataset,
        "source_config": config,
        "source_split": split,
        "source_row_idx": int(row_item.get("row_idx", -1)),
        "source_api_url": api_url,
        "source_dataset_url": f"https://huggingface.co/datasets/{dataset}",
        "source_audio_url": audio_url,
        "source_id": source_id,
        "group": group,
        "language": language,
        "speech_variant": speech_variant,
        "reference_text": text,
        "speaker_key": speaker,
    }


def _iter_offsets(preferred: Iterable[int], *, maximum: int) -> Iterable[int]:
    seen: set[int] = set()
    for offset in list(preferred) + list(range(0, maximum, 100)):
        if offset not in seen:
            seen.add(offset)
            yield offset


def _collect_ke_speech(per_group: int) -> list[dict[str, Any]]:
    chosen: dict[str, list[dict[str, Any]]] = {group: [] for group in KE_SPEECH_GROUPS}
    seen_ids: set[str] = set()
    for offset in _iter_offsets(KE_SPEECH_PAGE_OFFSETS, maximum=20_000):
        rows, api_url = _fetch_rows(KE_SPEECH_DATASET, "default", "test", offset)
        for row_item in rows:
            item = _ke_speech_sample(row_item, api_url)
            if item is None or item["source_id"] in seen_ids:
                continue
            group = str(item["group"])
            if len(chosen[group]) >= per_group:
                continue
            # Prefer different speakers so a single recording condition does
            # not dominate the tiny evaluation set.
            speaker_keys = {str(value["speaker_key"]) for value in chosen[group]}
            if item["speaker_key"] in speaker_keys and len(speaker_keys) >= per_group:
                continue
            chosen[group].append(item)
            seen_ids.add(str(item["source_id"]))
        if all(len(values) >= per_group for values in chosen.values()):
            break
    missing = {group: len(values) for group, values in chosen.items() if len(values) < per_group}
    if missing:
        raise RuntimeError(f"KeSpeech could not provide {per_group} samples per group: {missing}")
    return [item for group in KE_SPEECH_GROUPS for item in chosen[group][:per_group]]


def _collect_source(
    *,
    dataset: str,
    config: str,
    split: str,
    group: str,
    language: str,
    text_field: str,
    id_field: str,
    per_group: int,
    total_rows: int,
    speaker_field: str | None = None,
    speech_variant: str | None = None,
    minimum_text_length: int = 8,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_speakers: set[str] = set()
    for offset in range(0, total_rows, 100):
        rows, api_url = _fetch_rows(dataset, config, split, offset)
        for row_item in rows:
            item = _generic_sample(
                row_item,
                api_url,
                dataset=dataset,
                config=config,
                split=split,
                group=group,
                language=language,
                text_field=text_field,
                id_field=id_field,
                speaker_field=speaker_field,
                speech_variant=speech_variant,
                minimum_text_length=minimum_text_length,
            )
            if item is None:
                continue
            speaker_key = str(item.get("speaker_key", ""))
            if speaker_key and speaker_key in seen_speakers and len(seen_speakers) < per_group:
                continue
            selected.append(item)
            if speaker_key:
                seen_speakers.add(speaker_key)
            if len(selected) >= per_group:
                return selected[:per_group]
    raise RuntimeError(f"{dataset}/{config}/{split} could not provide {per_group} samples")


def _download(url: str, destination: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(5):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "real-time-translation-architecture-benchmark/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response, destination.open("wb") as target:
                shutil.copyfileobj(response, target)
            return
        except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            last_error = exc
            destination.unlink(missing_ok=True)
            if attempt + 1 < 5:
                time.sleep(min(10.0, 1.5**attempt))
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
        duration = source.getnframes() / max(1, source.getframerate())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return duration, digest


def _materialize(samples: list[dict[str, Any]], output_dir: Path, ffmpeg: str) -> None:
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(samples, 1):
        sample_id = f"{_slug(item['language'])}_{_slug(item['group'])}_{index:03d}"
        destination = audio_dir / f"{sample_id}.wav"
        if not destination.is_file():
            with tempfile.NamedTemporaryFile(prefix="asr-benchmark-", suffix=".download", delete=False) as temporary:
                temporary_path = Path(temporary.name)
            try:
                _download(str(item["source_audio_url"]), temporary_path)
                _convert_to_wav(temporary_path, destination, ffmpeg)
            finally:
                temporary_path.unlink(missing_ok=True)
        duration, digest = _wav_info(destination)
        item["sample_id"] = sample_id
        item["audio_path"] = str(destination.relative_to(output_dir)).replace("\\", "/")
        item["duration_seconds"] = round(duration, 4)
        item["sha256"] = digest
        item.pop("source_audio_url", None)


def _collect(per_group: int) -> list[dict[str, Any]]:
    samples = _collect_ke_speech(per_group)
    samples.extend(
        _collect_source(
            dataset="openslr/librispeech_asr",
            config="clean",
            split="test",
            group="English",
            language="en",
            text_field="text",
            id_field="id",
            speaker_field="speaker_id",
            per_group=per_group,
            total_rows=2620,
            minimum_text_length=12,
        )
    )
    samples.extend(
        _collect_source(
            dataset="facebook/multilingual_librispeech",
            config="german",
            split="test",
            group="German",
            language="de",
            text_field="transcript",
            id_field="id",
            speaker_field="speaker_id",
            per_group=per_group,
            total_rows=3394,
            minimum_text_length=12,
        )
    )
    samples.extend(
        _collect_source(
            dataset="shunyalabs/cantonese-speech-dataset",
            config="default",
            split="test",
            group="Cantonese",
            language="zh",
            text_field="transcript",
            id_field="id",
            per_group=per_group,
            total_rows=819,
            speech_variant="cantonese_hong_kong",
            minimum_text_length=8,
        )
    )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("result/architecture_benchmark"))
    parser.add_argument("--per-group", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="re-download and reconvert existing WAV files")
    args = parser.parse_args()
    if args.per_group < 1 or args.per_group > 50:
        parser.error("--per-group must be between 1 and 50")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        parser.error("ffmpeg is required to normalize downloaded audio")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        for path in (output_dir / "audio").glob("*.wav"):
            path.unlink()

    samples = _collect(args.per_group)
    _materialize(samples, output_dir, ffmpeg)
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collection": {
            "per_group": args.per_group,
            "audio_format": "16 kHz mono PCM s16le WAV",
            "viewer_api": HF_ROWS_API,
        },
        "sources": [
            {
                "dataset": "miaocongxin/KeSpeech",
                "config": "default",
                "split": "test",
                "purpose": "Mandarin plus seven Mandarin subdialect groups",
                "license_note": "KeSpeech custom license; paper states free academic/noncommercial use",
            },
            {
                "dataset": "openslr/librispeech_asr",
                "config": "clean",
                "split": "test",
                "purpose": "English",
                "license_note": "Use the dataset's current license terms",
            },
            {
                "dataset": "facebook/multilingual_librispeech",
                "config": "german",
                "split": "test",
                "purpose": "German",
                "license_note": "CC BY 4.0",
            },
            {
                "dataset": "shunyalabs/cantonese-speech-dataset",
                "config": "default",
                "split": "test",
                "purpose": "Cantonese",
                "license_note": "No license field was declared on the dataset page; keep this local unless separately cleared",
            },
        ],
        "samples": samples,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(manifest_path), "samples": len(samples)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

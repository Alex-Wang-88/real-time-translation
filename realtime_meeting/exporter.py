from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .formatting import paired_text, source_line
from .models import LANGUAGE_LABELS, Utterance, language_label
from .text_normalize import simplify_chinese


def load_utterances(path: Path) -> list[Utterance]:
    items: list[Utterance] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = Utterance.from_dict(json.loads(line))
                if item.language == "zh":
                    item.text = simplify_chinese(item.text)
                items.append(item)
    return items


def append_utterance(path: Path, item: Utterance) -> None:
    if item.language == "zh":
        item.text = simplify_chinese(item.text)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
        handle.flush()


def _write(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def export_live_result(
    output_dir: Path,
    *,
    session_id: str,
    started_at: str,
    ended_at: str,
    duration_seconds: float,
    utterances: Iterable[Utterance],
    audio_segments: list[dict[str, object]],
    status: str,
    summary_error: str | None = None,
    processing_error: str | None = None,
) -> list[str]:
    items = list(utterances)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": round(duration_seconds, 3),
        "utterances": [item.to_dict() for item in items],
    }
    (output_dir / "transcript.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # A recovered meeting may have been recorded by an older build that
    # stored Traditional Chinese in JSONL.  Rewrite the completed JSONL from
    # the normalized in-memory items so every downloadable transcript agrees.
    (output_dir / "transcript.jsonl").write_text(
        "".join(json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
    )
    (output_dir / "audio_manifest.json").write_text(
        json.dumps({"segments": audio_segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pairs = "\n".join(paired_text(item) for item in items)
    _write(output_dir / "meeting_transcript.md", "# 会议实时逐句转译\n\n" + (pairs or "（未检测到有效发言）"))
    _write(output_dir / "translated_zh.md", "# 整场会议中文译稿\n\n" + (pairs or "（未检测到有效发言）"))
    # Always keep the three original-language files for the original product
    # contract, then add any language actually detected in this meeting.
    languages = list(LANGUAGE_LABELS)
    for item in items:
        if item.language not in languages:
            languages.append(item.language)
    for language in languages:
        label = language_label(language)
        selected = [source_line(item) for item in items if item.language == language]
        body = "\n".join(selected) if selected else "（未检测到该语言发言）"
        _write(output_dir / f"original_{language}.md", f"# {label}原稿\n\n{body}")
    manifest = {
        "session_id": session_id,
        "status": status,
        "transcript": "meeting_transcript.md",
        "transcript_json": "transcript.json",
        "transcript_jsonl": "transcript.jsonl",
        "audio_manifest": "audio_manifest.json",
        "minutes": "meeting_minutes.md" if (output_dir / "meeting_minutes.md").exists() else None,
        "summary_error": summary_error,
        "processing_error": processing_error,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return [
        "meeting_transcript.md",
        "translated_zh.md",
        "transcript.json",
        "transcript.jsonl",
        "audio_manifest.json",
        "manifest.json",
    ] + [f"original_{language}.md" for language in languages]

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .jimo import paired_text
from .models import LANGUAGE_LABELS, TodoDocument, Utterance, language_label
from .storage import TranscriptStore, atomic_write_json, atomic_write_text
from .text_normalize import simplify_chinese


def load_utterances(path: Path) -> list[Utterance]:
    return TranscriptStore(path).load()


def append_utterance(path: Path, item: Utterance) -> None:
    TranscriptStore(path).append(item)


def delete_utterance(path: Path, item: Utterance) -> None:
    TranscriptStore(path).delete(item)


def _write(path: Path, content: str) -> None:
    atomic_write_text(path, content.rstrip() + "\n")


def render_todo_markdown(todo: TodoDocument) -> str:
    lines = ["# To-do-list", "", f"会议：{todo.meeting_id}", f"纪要版本：{todo.summary_revision}", ""]
    if not todo.items:
        lines.append("未提取到明确行动项。")
        return "\n".join(lines) + "\n"
    lines.extend([
        "| 任务 | 负责人 | 截止时间 | 优先级 | 状态 | 原文时间 | 依据 |",
        "|---|---|---|---|---|---|---|",
    ])
    for item in todo.items:
        owner = item.owner or "待确认"
        due = item.due_date or "待确认"
        start = "待确认" if item.source_time_start is None else f"{item.source_time_start:.3f}"
        end = "待确认" if item.source_time_end is None else f"{item.source_time_end:.3f}"
        values = [item.task, owner, due, item.priority, item.status, f"{start}-{end}", item.evidence]
        lines.append("| " + " | ".join(value.replace("|", "\\|").replace("\n", " ") for value in values) + " |")
    return "\n".join(lines) + "\n"


def export_live_result(
    output_dir: Path,
    *,
    meeting_id: str,
    title: str,
    started_at: str,
    ended_at: str,
    duration_seconds: float,
    utterances: Iterable[Utterance],
    audio_segments: list[dict[str, object]],
    recording_state: str,
    summary_state: str,
    todo_state: str,
    summary_error: str | None = None,
    todo_error: str | None = None,
    postprocess: dict[str, object] | None = None,
    model_metadata: dict[str, object] | None = None,
) -> list[str]:
    items = list(utterances)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "transcript.json", {
        "meeting_id": meeting_id,
        "title": title,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": round(duration_seconds, 3),
        "utterances": [item.to_dict() for item in items],
    })
    atomic_write_text(output_dir / "transcript.jsonl", "".join(json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in items))
    atomic_write_json(output_dir / "audio_manifest.json", {"segments": audio_segments})
    speaker_segments_path = output_dir / "speaker_segments.json"
    if not speaker_segments_path.exists():
        atomic_write_json(speaker_segments_path, {"segments": []})
    pairs = "\n".join(paired_text(item) for item in items)
    _write(output_dir / "meeting_transcript.md", f"# {title or '会议逐句转写'}\n\n" + (pairs or "（未检测到有效发言）"))
    _write(output_dir / "translated_zh.md", "# 整场会议中文译稿\n\n" + (pairs or "（未检测到有效发言）"))
    languages = list(LANGUAGE_LABELS)
    for item in items:
        if item.language not in languages:
            languages.append(item.language)
    for language in languages:
        selected = [
            f"[{item.start:.3f}-{item.end:.3f}] 演讲人{item.speaker_id}：{item.text}"
            for item in items if item.language == language
        ]
        _write(output_dir / f"original_{language}.md", f"# {language_label(language)}原稿\n\n" + ("\n".join(selected) or "（未检测到该语言发言）"))
    manifest = {
        "meeting_id": meeting_id,
        "title": title,
        "recording_state": recording_state,
        "summary_state": summary_state,
        "todo_state": todo_state,
        "transcript": "meeting_transcript.md",
        "transcript_json": "transcript.json",
        "transcript_jsonl": "transcript.jsonl",
        "audio_manifest": "audio_manifest.json",
        "transcript_events": "transcript_events.jsonl" if (output_dir / "transcript_events.jsonl").is_file() else None,
        "speaker_segments": "speaker_segments.json" if speaker_segments_path.is_file() else None,
        "minutes": "meeting_minutes.md" if (output_dir / "meeting_minutes.md").is_file() else None,
        "todo_json": "todo_list.json" if (output_dir / "todo_list.json").is_file() else None,
        "todo_markdown": "todo_list.md" if (output_dir / "todo_list.md").is_file() else None,
        "summary_error": summary_error,
        "todo_error": todo_error,
        "postprocess": postprocess or {},
        "model_metadata": model_metadata or {},
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    files = [
        "meeting_transcript.md", "translated_zh.md", "transcript.json", "transcript.jsonl",
        "audio_manifest.json", "manifest.json",
    ] + [f"original_{language}.md" for language in languages]
    if (output_dir / "transcript_events.jsonl").is_file():
        files.append("transcript_events.jsonl")
    if speaker_segments_path.is_file():
        files.append("speaker_segments.json")
    for segment in audio_segments:
        if isinstance(segment, dict) and segment.get("file"):
            audio_name = str(segment["file"]).lstrip("/\\")
            files.append(f"audio/{audio_name}")
    for name in ("meeting_minutes.md", "todo_list.json", "todo_list.md"):
        if (output_dir / name).is_file():
            files.append(name)
    return files

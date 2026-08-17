from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .models import TodoDocument, Utterance, language_label, speech_variant_label
from .storage import TranscriptStore, atomic_write_json, atomic_write_text


def load_utterances(path: Path) -> list[Utterance]:
    return TranscriptStore(path).load()


def load_paragraphs(path: Path) -> list[Utterance]:
    return load_utterances(path)


def append_utterance(path: Path, item: Utterance) -> None:
    TranscriptStore(path).append(item)


def delete_utterance(path: Path, item: Utterance) -> None:
    TranscriptStore(path).delete(item)


def _write(path: Path, content: str) -> None:
    atomic_write_text(path, content.rstrip() + "\n")


def _language_filename(language: str) -> str:
    return {"zh": "original_zh.md", "en": "original_en.md", "de": "original_de.md"}.get(language, "original_unknown.md")


def _variant_label(item: Utterance) -> str:
    if item.language != "zh":
        return language_label(item.language)
    return speech_variant_label(item.speech_variant) if item.speech_variant else "普通话/中文未细分"


def _paragraph_markdown(item: Utterance, *, include_translation: bool = True) -> str:
    lines = [
        f"### [{item.start:.3f}-{item.end:.3f}] {_variant_label(item)}",
        f"原文：{item.text.strip()}",
    ]
    if include_translation and item.translation_zh.strip():
        lines.append(f"中文：{item.translation_zh.strip()}")
    return "\n".join(lines)


def render_todo_markdown(todo: TodoDocument) -> str:
    lines = ["# To-do-list", "", f"生成时间：{todo.generated_at or '待确认'}", ""]
    if not todo.items:
        lines.append("暂无明确行动项。")
        return "\n".join(lines)
    lines.extend(["| 任务 | 负责人 | 截止时间 | 优先级 | 状态 | 原文时间 | 依据 |", "|---|---|---|---|---|---|---|"])
    for item in todo.items:
        time_range = "待确认"
        if item.source_time_start is not None:
            time_range = f"{item.source_time_start:.3f}-{(item.source_time_end if item.source_time_end is not None else item.source_time_start):.3f}"
        cells = [item.task, item.owner or "待确认", item.due_date or "待确认", item.priority, item.status, time_range, item.evidence]
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in cells) + " |")
    return "\n".join(lines)


def export_live_result(
    output_dir: Path,
    *,
    meeting_id: str,
    title: str,
    started_at: str,
    ended_at: str,
    duration_seconds: float,
    utterances: Iterable[Utterance],
    audio_segments: Iterable[dict[str, Any]],
    recording_state: str,
    summary_state: str,
    todo_state: str,
    summary_error: str | None = None,
    todo_error: str | None = None,
    model_metadata: dict[str, Any] | None = None,
) -> list[str]:
    """Write schema 2 paragraph exports."""

    output_dir.mkdir(parents=True, exist_ok=True)
    items = sorted(list(utterances), key=lambda item: (item.start, item.end, item.id))
    transcript = {
        "schema_version": "2.0",
        "meeting_id": meeting_id,
        "title": title,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration_seconds,
        "paragraphs": [item.to_dict() for item in items],
    }
    atomic_write_json(output_dir / "transcript.json", transcript)

    meeting_lines = [f"# {title or '会议记录'}", "", f"会议 ID：{meeting_id}", f"开始：{started_at}", f"结束：{ended_at}", ""]
    meeting_lines.extend(_paragraph_markdown(item) for item in items)
    _write(output_dir / "meeting_transcript.md", "\n\n".join(meeting_lines))

    translated_lines = [f"# {title or '中文译文'}", ""]
    translated_lines.extend(
        f"### [{item.start:.3f}-{item.end:.3f}]\n{item.translation_zh.strip()}"
        for item in items
        if item.translation_zh.strip()
    )
    _write(output_dir / "translated_zh.md", "\n\n".join(translated_lines))

    language_files: list[str] = []
    for language in ("zh", "en", "de"):
        language_items = [item for item in items if item.language == language]
        if not language_items:
            continue
        name = _language_filename(language)
        language_lines = [f"# {language_label(language)}原文", ""]
        language_lines.extend(
            f"### [{item.start:.3f}-{item.end:.3f}] {_variant_label(item)}\n{item.text.strip()}"
            for item in language_items
        )
        _write(output_dir / name, "\n\n".join(language_lines))
        language_files.append(name)

    audio_manifest = [dict(segment) for segment in audio_segments]
    atomic_write_json(output_dir / "audio_manifest.json", {"segments": audio_manifest})
    manifest = {
        "schema_version": "2.0",
        "meeting_id": meeting_id,
        "title": title,
        "recording_state": recording_state,
        "summary_state": summary_state,
        "todo_state": todo_state,
        "summary_error": summary_error,
        "todo_error": todo_error,
        "paragraph_count": len(items),
        "model_metadata": model_metadata or {},
    }
    atomic_write_json(output_dir / "manifest.json", manifest)

    files = [
        "meeting_transcript.md", "translated_zh.md", "transcript.json", "transcript.jsonl",
        "transcript_events.jsonl", "audio_manifest.json", "manifest.json",
        "meeting_minutes.md", "todo_list.json", "todo_list.md",
    ] + language_files
    return [name for name in files if (output_dir / name).is_file()]

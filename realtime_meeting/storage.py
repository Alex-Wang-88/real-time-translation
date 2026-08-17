from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from weakref import WeakValueDictionary

from .language import is_mixed_source_text
from .models import Utterance
from .text_normalize import simplify_chinese


class MeetingStore(Protocol):
    def meeting_dir(self, meeting_id: str) -> Path: ...

    def list_states(self) -> list[dict[str, Any]]: ...

    def delete(self, meeting_id: str) -> None: ...


class TranscriptStore:
    """Schema 2 append-only paragraph event log and latest projection."""

    schema_version = "2.0"
    _locks: WeakValueDictionary[str, threading.RLock] = WeakValueDictionary()
    _locks_guard = threading.Lock()

    def __init__(self, path: Path) -> None:
        self.path = path
        self.events_path = path.with_name("transcript_events.jsonl")
        self.projection_path = path.with_suffix(".json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._latest: list[Utterance] | None = None
        self._event_signature: tuple[int, int] | None = None
        self._projection_pending = 0
        key = str(path.resolve())
        with self._locks_guard:
            self._lock = self._locks.setdefault(key, threading.RLock())

    @staticmethod
    def _normalize(item: Utterance) -> Utterance:
        item.segment_id = str(item.segment_id or f"p-{item.id:06d}")
        item.language = str(item.language or "unknown")
        if item.language == "zh" and not is_mixed_source_text(item.text):
            item.text = simplify_chinese(item.text)
            item.translation_zh = simplify_chinese(item.text)
            item.translation_status = "not_needed"
            item.translation_model = None
        item.revision = max(1, int(item.revision or 1))
        item.source_revision = max(1, int(item.source_revision or 1))
        return item

    def append(self, item: Utterance, *, event_type: str = "upsert") -> None:
        item = self._normalize(item)
        record = {
            "schema_version": self.schema_version,
            "event_type": event_type,
            "paragraph": item.to_dict(),
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            latest = self._ensure_latest_locked()
            for event_path in (self.path, self.events_path):
                with event_path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
            stored = Utterance.from_dict(item.to_dict())
            if event_type == "delete":
                latest[:] = [current for current in latest if current.segment_id != stored.segment_id]
            else:
                replaced = False
                for index, current in enumerate(latest):
                    if current.segment_id == stored.segment_id:
                        latest[index] = stored
                        replaced = True
                        break
                if not replaced:
                    latest.append(stored)
            self._event_signature = self._signature(self.path)
            self._projection_pending += 1
            if stored.closed or event_type == "delete" or len(latest) <= 1 or self._projection_pending >= 32:
                self._write_projection(latest)
                self._projection_pending = 0

    def delete(self, item: Utterance) -> None:
        self.append(item, event_type="delete")

    def replace_all(self, items: list[Utterance], *, reason: str = "rewrite") -> None:
        normalized = [self._normalize(item) for item in items]
        record = {
            "schema_version": self.schema_version,
            "event_type": "replace_all",
            "reason": reason,
            "paragraphs": [item.to_dict() for item in normalized],
        }
        with self._lock:
            line = json.dumps(record, ensure_ascii=False) + "\n"
            for event_path in (self.path, self.events_path):
                with event_path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
            self._latest = [Utterance.from_dict(item.to_dict()) for item in normalized]
            self._event_signature = self._signature(self.path)
            self._write_projection(normalized)
            self._projection_pending = 0

    def load(self) -> list[Utterance]:
        with self._lock:
            signature = self._signature(self.path)
            if self._latest is None or signature != self._event_signature:
                self._latest = self._load_from_disk_locked()
                self._event_signature = signature
                self._projection_pending = 0
            return [Utterance.from_dict(item.to_dict()) for item in self._latest]

    @staticmethod
    def _signature(path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        return stat.st_size, stat.st_mtime_ns

    def _ensure_latest_locked(self) -> list[Utterance]:
        signature = self._signature(self.path)
        if self._latest is None or signature != self._event_signature:
            self._latest = self._load_from_disk_locked()
            self._event_signature = signature
            self._projection_pending = 0
        return self._latest

    def _load_from_disk_locked(self) -> list[Utterance]:
        event_path = self.path if self.path.is_file() else self.events_path
        if not event_path.is_file():
            if not self.projection_path.is_file():
                return []
            try:
                payload = json.loads(self.projection_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                return []
            values = payload.get("paragraphs", []) if isinstance(payload, dict) else []
            return [Utterance.from_dict(item) for item in values if isinstance(item, dict)]
        latest: dict[str, Utterance] = {}
        try:
            lines = event_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict) or str(record.get("schema_version")) != self.schema_version:
                continue
            event_type = record.get("event_type")
            if event_type == "replace_all":
                latest = {
                    item.segment_id: item
                    for item in (
                        Utterance.from_dict(value)
                        for value in record.get("paragraphs", [])
                        if isinstance(value, dict)
                    )
                }
                continue
            value = record.get("paragraph")
            if not isinstance(value, dict):
                continue
            item = Utterance.from_dict(value)
            if event_type == "delete":
                latest.pop(item.segment_id, None)
            else:
                previous = latest.get(item.segment_id)
                if previous is None or (item.revision, item.source_revision) >= (previous.revision, previous.source_revision):
                    latest[item.segment_id] = self._normalize(item)
        return sorted(latest.values(), key=lambda item: (item.start, item.end, item.id))

    def _write_projection(self, items: list[Utterance]) -> None:
        atomic_write_json(
            self.projection_path,
            {
                "schema_version": self.schema_version,
                "paragraphs": [self._normalize(item).to_dict() for item in sorted(items, key=lambda value: (value.start, value.end, value.id))],
            },
        )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


class LocalMeetingStore:
    """Filesystem store for schema 2 meetings; old state files are ignored."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def meeting_dir(self, meeting_id: str) -> Path:
        if not meeting_id or "/" in meeting_id or "\\" in meeting_id or meeting_id in {".", ".."}:
            raise ValueError("非法会议 ID")
        candidate = (self.root / meeting_id).resolve()
        candidate.relative_to(self.root)
        return candidate

    def list_states(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        with self._lock:
            for directory in self.root.iterdir():
                if not directory.is_dir():
                    continue
                path = directory / "session_state.json"
                if not path.is_file():
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                if (
                    isinstance(payload, dict)
                    and payload.get("schema_version") == "2.0"
                    and str(payload.get("id", "")) == directory.name
                    and "recording_state" in payload
                ):
                    results.append(payload)
        return sorted(results, key=lambda item: str(item.get("started_at", "")), reverse=True)

    def delete(self, meeting_id: str) -> None:
        target = self.meeting_dir(meeting_id).resolve()
        with self._lock:
            if not target.exists():
                return
            target.relative_to(self.root)
            shutil.rmtree(target)

    def purge_expired(self, retention_days: int) -> list[str]:
        if retention_days <= 0:
            return []
        cutoff = datetime.now(timezone.utc).timestamp() - retention_days * 86400
        removed: list[str] = []
        with self._lock:
            for directory in self.root.iterdir():
                if not directory.is_dir():
                    continue
                state_path = directory / "session_state.json"
                if not state_path.is_file():
                    continue
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                if not isinstance(state, dict) or state.get("schema_version") != "2.0":
                    continue
                if state.get("recording_state") in {"starting", "recording", "finalizing"} or not state.get("ended_at"):
                    continue
                try:
                    ended_at = datetime.fromisoformat(str(state["ended_at"]).replace("Z", "+00:00"))
                    if ended_at.tzinfo is None:
                        ended_at = ended_at.replace(tzinfo=timezone.utc)
                except (TypeError, ValueError):
                    continue
                if ended_at.timestamp() >= cutoff:
                    continue
                removed.append(directory.name)
                self.delete(directory.name)
        return removed

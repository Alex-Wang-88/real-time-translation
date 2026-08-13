from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .models import Utterance
from .text_normalize import simplify_chinese


class MeetingStore(Protocol):
    def meeting_dir(self, meeting_id: str) -> Path: ...

    def list_states(self) -> list[dict[str, Any]]: ...

    def delete(self, meeting_id: str) -> None: ...


class TranscriptStore:
    """Append-only transcript storage with a backwards-compatible projection."""

    schema_version = "1.0"
    _locks: dict[str, threading.RLock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, path: Path) -> None:
        self.path = path
        self.events_path = path.with_name("transcript_events.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        key = str(path.resolve())
        with self._locks_guard:
            self._lock = self._locks.setdefault(key, threading.RLock())

    @staticmethod
    def _normalize(item: Utterance) -> Utterance:
        if not item.segment_id:
            item.segment_id = f"legacy:{item.id}"
        if not item.source_segment_id or item.source_segment_id == item.segment_id:
            item.source_segment_id = item.segment_id.split(":", 1)[0]
        if item.language == "zh":
            item.text = simplify_chinese(item.text)
            item.translation_zh = item.text
            item.translation_status = "not_needed"
        return item

    @staticmethod
    def _same_segment_key(item: Utterance) -> str:
        return item.segment_id or item.source_segment_id or f"legacy:{item.id}"

    def append(self, item: Utterance, *, event_type: str = "upsert") -> None:
        item = self._normalize(item)
        payload = item.to_dict()
        event = {
            "schema_version": self.schema_version,
            "event": event_type,
            "segment_id": item.segment_id,
            "revision": item.revision,
            "utterance": payload,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
            with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                handle.flush()

    def delete(self, item: Utterance) -> None:
        item.deleted = True
        item.revision = max(2, item.revision + 1)
        item.recognition_stage = "deleted"
        self.append(item, event_type="delete")

    def replace_segment(self, segment_id: str, items: list[Utterance]) -> None:
        """Emit tombstones for removed revisions, then append replacements."""
        current = [item for item in self.load() if item.segment_id == segment_id]
        replacement_ids = {item.id for item in items}
        for previous in current:
            if previous.id not in replacement_ids:
                self.delete(previous)
        for item in items:
            self.append(item, event_type="replace")

    def load(self) -> list[Utterance]:
        latest: dict[str, Utterance] = {}
        if not self.path.exists():
            return []
        with self._lock:
            with self.path.open("r", encoding="utf-8") as handle:
                for raw in handle:
                    if not raw.strip():
                        continue
                    try:
                        item = Utterance.from_dict(json.loads(raw))
                    except (ValueError, TypeError, json.JSONDecodeError):
                        continue
                    key = self._same_segment_key(item)
                    previous = latest.get(key)
                    if previous is None or (item.revision, item.recognition_stage == "refined") >= (
                        previous.revision,
                        previous.recognition_stage == "refined",
                    ):
                        latest[key] = item
        return [item for item in latest.values() if not item.deleted]


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
    """Filesystem implementation used by Windows v2 and replaceable in enterprise deployments."""

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
                # Older releases stored a different schema in timestamped
                # directories (``state`` instead of ``recording_state``).
                # Loading those files with v2 defaults turns an old
                # recording into a new active meeting after every restart.
                # A v2 state must identify the directory it lives in and
                # carry the v2 recording state explicitly.
                if (
                    isinstance(payload, dict)
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
                if not isinstance(state, dict) or (
                    state.get("recording_state") in {"starting", "recording", "finalizing"}
                    or not state.get("ended_at")
                ):
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

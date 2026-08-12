from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .audio import SegmentEvent


class SessionRepository(Protocol):
    def write_state(self, output_dir: Path, payload: dict[str, object]) -> None: ...


class AudioStore(Protocol):
    def write_refinement_pcm(self, output_dir: Path, revision: int, pcm: bytes) -> Path: ...

    def delete(self, path: Path) -> None: ...


class LocalSessionRepository:
    def write_state(self, output_dir: Path, payload: dict[str, object]) -> None:
        path = output_dir / "session_state.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)


class LocalAudioStore:
    def write_refinement_pcm(self, output_dir: Path, revision: int, pcm: bytes) -> Path:
        directory = output_dir / "refinement_jobs"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"revision-{revision:08d}.pcm"
        temporary = path.with_suffix(".pcm.tmp")
        temporary.write_bytes(pcm)
        temporary.replace(path)
        return path

    def delete(self, path: Path) -> None:
        path.unlink(missing_ok=True)


@dataclass(slots=True)
class RefinementJob:
    session_id: str
    revision: int
    start: float
    end: float
    forced: bool
    pcm_path: str
    draft_text: str
    draft_language: str | None
    attempts: int = 0
    status: str = "queued"
    error: str | None = None

    def event(self) -> SegmentEvent:
        return SegmentEvent(
            kind="final",
            revision=self.revision,
            start=self.start,
            end=self.end,
            pcm=Path(self.pcm_path).read_bytes(),
            forced=self.forced,
        )


class RefinementJobStore:
    """SQLite WAL queue metadata with atomically spooled PCM payloads."""

    def __init__(self, database_path: Path, audio_store: AudioStore | None = None) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path
        self.audio_store = audio_store or LocalAudioStore()
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS refinement_jobs (
                    session_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    start REAL NOT NULL,
                    end REAL NOT NULL,
                    forced INTEGER NOT NULL,
                    pcm_path TEXT NOT NULL,
                    draft_text TEXT NOT NULL,
                    draft_language TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'queued',
                    error TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (session_id, revision)
                )
                """
            )
            # A process can die between claim and completion. Reclaim those
            # jobs on startup; idempotent transcript commits suppress repeats.
            connection.execute(
                "UPDATE refinement_jobs SET status='queued', updated_at=CURRENT_TIMESTAMP "
                "WHERE status='running'"
            )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> RefinementJob:
        return RefinementJob(
            session_id=row["session_id"],
            revision=int(row["revision"]),
            start=float(row["start"]),
            end=float(row["end"]),
            forced=bool(row["forced"]),
            pcm_path=row["pcm_path"],
            draft_text=row["draft_text"],
            draft_language=row["draft_language"],
            attempts=int(row["attempts"]),
            status=row["status"],
            error=row["error"],
        )

    def enqueue(
        self,
        *,
        session_id: str,
        output_dir: Path,
        event: SegmentEvent,
        draft_text: str,
        draft_language: str | None,
    ) -> RefinementJob:
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM refinement_jobs WHERE session_id=? AND revision=?",
                (session_id, event.revision),
            ).fetchone()
            if existing is not None:
                return self._from_row(existing)
            pcm_path = self.audio_store.write_refinement_pcm(
                output_dir, event.revision, event.pcm
            )
            connection.execute(
                """
                INSERT INTO refinement_jobs
                    (session_id, revision, start, end, forced, pcm_path,
                     draft_text, draft_language, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued')
                """,
                (
                    session_id,
                    event.revision,
                    event.start,
                    event.end,
                    int(event.forced),
                    str(pcm_path),
                    draft_text,
                    draft_language,
                ),
            )
        return RefinementJob(
            session_id=session_id,
            revision=event.revision,
            start=event.start,
            end=event.end,
            forced=event.forced,
            pcm_path=str(pcm_path),
            draft_text=draft_text,
            draft_language=draft_language,
        )

    def pending(self, session_id: str) -> list[RefinementJob]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM refinement_jobs WHERE session_id=? "
                "AND status IN ('queued','running') ORDER BY revision",
                (session_id,),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def mark_running(self, job: RefinementJob) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE refinement_jobs SET status='running', attempts=attempts+1, "
                "error=NULL, updated_at=CURRENT_TIMESTAMP WHERE session_id=? AND revision=?",
                (job.session_id, job.revision),
            )

    def mark_done(self, job: RefinementJob) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE refinement_jobs SET status='done', error=NULL, "
                "updated_at=CURRENT_TIMESTAMP WHERE session_id=? AND revision=?",
                (job.session_id, job.revision),
            )
        self.audio_store.delete(Path(job.pcm_path))

    def mark_failed(self, job: RefinementJob, error: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE refinement_jobs SET status='failed', error=?, "
                "updated_at=CURRENT_TIMESTAMP WHERE session_id=? AND revision=?",
                (error[:2000], job.session_id, job.revision),
            )

    def requeue(self, job: RefinementJob, error: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE refinement_jobs SET status='queued', error=?, "
                "updated_at=CURRENT_TIMESTAMP WHERE session_id=? AND revision=?",
                (error[:2000], job.session_id, job.revision),
            )

    def retry_failed(self, session_id: str) -> int:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE refinement_jobs SET status='queued', error=NULL, "
                "updated_at=CURRENT_TIMESTAMP WHERE session_id=? AND status='failed'",
                (session_id,),
            )
            return cursor.rowcount

    def counts(self, session_id: str | None = None) -> dict[str, int]:
        query = "SELECT status, COUNT(*) count FROM refinement_jobs"
        parameters: tuple[object, ...] = ()
        if session_id is not None:
            query += " WHERE session_id=?"
            parameters = (session_id,)
        query += " GROUP BY status"
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        counts = {"queued": 0, "running": 0, "failed": 0, "done": 0}
        counts.update({row["status"]: int(row["count"]) for row in rows})
        return counts

    def pending_total(self) -> int:
        counts = self.counts()
        return counts["queued"] + counts["running"]

    def spool_bytes(self) -> int:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT pcm_path FROM refinement_jobs WHERE status != 'done'"
            ).fetchall()
        total = 0
        for row in rows:
            try:
                total += Path(row["pcm_path"]).stat().st_size
            except OSError:
                pass
        return total

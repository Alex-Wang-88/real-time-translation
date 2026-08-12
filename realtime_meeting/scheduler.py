from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator


class GpuResourceManager:
    """Serializes GPU-heavy stages and exposes lightweight timing metrics."""

    def __init__(self) -> None:
        self._thread_lock = threading.Lock()
        self.metrics: dict[str, Any] = {"waits": 0, "runs": 0, "durations_ms": {}}

    @asynccontextmanager
    async def acquire(self, stage: str) -> AsyncIterator[None]:
        if not self._thread_lock.acquire(blocking=False):
            self.metrics["waits"] = int(self.metrics.get("waits", 0)) + 1
            await asyncio.to_thread(self._thread_lock.acquire)
        started = time.perf_counter()
        try:
            yield
        finally:
            durations = self.metrics.setdefault("durations_ms", {})
            durations[stage] = round(float(durations.get(stage, 0.0)) + (time.perf_counter() - started) * 1000, 3)
            self.metrics["runs"] = int(self.metrics.get("runs", 0)) + 1
            self._thread_lock.release()

    @contextmanager
    def acquire_sync(self, stage: str) -> Any:
        started = time.perf_counter()
        with self._thread_lock:
            try:
                yield
            finally:
                durations = self.metrics.setdefault("durations_ms", {})
                durations[stage] = round(float(durations.get(stage, 0.0)) + (time.perf_counter() - started) * 1000, 3)
                self.metrics["runs"] = int(self.metrics.get("runs", 0)) + 1


class PostprocessTracker:
    STAGES = ("asr_refine", "diarization", "translation", "summary", "todo")

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        self.state = str(payload.get("state", "idle"))
        self.current_stage = payload.get("current_stage")
        self.overall_percent = int(payload.get("overall_percent", 0) or 0)
        self.error = payload.get("error")
        self._stage_started: dict[str, float] = {}
        self.stage_durations_ms: dict[str, float] = {
            str(key): float(value) for key, value in (payload.get("stage_durations_ms") or {}).items()
        }
        self.stages: dict[str, dict[str, Any]] = {}
        source = payload.get("stages") if isinstance(payload.get("stages"), dict) else {}
        for stage in self.STAGES:
            self.stages[stage] = {
                "state": "idle" if stage not in source else str(source[stage].get("state", "queued")),
                "current": int(source.get(stage, {}).get("current", 0) or 0) if isinstance(source.get(stage), dict) else 0,
                "total": int(source.get(stage, {}).get("total", 0) or 0) if isinstance(source.get(stage), dict) else 0,
                "error": source.get(stage, {}).get("error") if isinstance(source.get(stage), dict) else None,
            }

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "current_stage": self.current_stage,
            "overall_percent": self.overall_percent,
            "stages": self.stages,
            "stage_durations_ms": self.stage_durations_ms,
            "error": self.error,
        }

    def start(self) -> None:
        self.state = "running"

    def update(self, stage: str, state: str, *, current: int | None = None, total: int | None = None, error: str | None = None) -> None:
        if stage not in self.stages:
            self.stages[stage] = {"state": state, "current": 0, "total": 0, "error": error}
        record = self.stages[stage]
        if state == "running" and stage not in self._stage_started:
            self._stage_started[stage] = time.perf_counter()
        if state in {"complete", "error", "partial"} and stage in self._stage_started:
            self.stage_durations_ms[stage] = round(
                self.stage_durations_ms.get(stage, 0.0)
                + (time.perf_counter() - self._stage_started.pop(stage)) * 1000,
                3,
            )
        record["state"] = state
        if current is not None:
            record["current"] = current
        if total is not None:
            record["total"] = total
        if error is not None:
            record["error"] = error
        self.current_stage = stage if state in {"queued", "running"} else self.current_stage
        finished = sum(1 for value in self.stages.values() if value.get("state") == "complete")
        self.overall_percent = round(finished / max(1, len(self.STAGES)) * 100)

    def fail(self, stage: str, error: str) -> None:
        self.update(stage, "error", error=error)
        self.state = "error"
        self.error = error

    def complete(self) -> None:
        self.state = "complete"
        self.current_stage = None
        self.overall_percent = 100


class BackgroundTaskRegistry:
    """Small task registry used by the single-process local deployment."""

    def __init__(self) -> None:
        self.tasks: dict[str, asyncio.Task[Any]] = {}

    def add(self, key: str, task: asyncio.Task[Any]) -> None:
        self.tasks[key] = task
        task.add_done_callback(lambda _task: self.tasks.pop(key, None))

    def cancel_all(self) -> None:
        for task in tuple(self.tasks.values()):
            if not task.done():
                task.cancel()

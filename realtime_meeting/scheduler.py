from __future__ import annotations

import asyncio
import threading
import time
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
            # A cancelled ``to_thread(lock.acquire)`` keeps running and can
            # acquire the lock after its coroutine has disappeared, leaking it
            # forever. Polling non-blockingly keeps cancellation safe.
            while not self._thread_lock.acquire(blocking=False):
                await asyncio.sleep(0.01)
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

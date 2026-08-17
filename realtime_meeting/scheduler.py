from __future__ import annotations

import asyncio
import heapq
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator


class LatestEventQueue:
    """Priority queue for realtime audio events.

    Partial ASR hypotheses are replaceable.  Keeping only a marker for the
    latest partial prevents a slow model call from turning every 800 ms audio
    update into a backlog.  Final events always have priority over partials.
    The small queue-like surface deliberately mirrors ``asyncio.Queue`` so
    existing session lifecycle and tests remain compatible.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self.maxsize = max(0, int(maxsize))
        self._queue: asyncio.PriorityQueue[tuple[int, int, object]] = asyncio.PriorityQueue(maxsize=self.maxsize)
        self._sequence = 0
        self._pending_partial: Any | None = None
        self._partial_marker = False
        self._last_final_revision = -1

    @staticmethod
    def _priority(value: Any) -> int:
        if value is None:
            return 100
        return 0 if getattr(value, "kind", "") == "final" else 10

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def full(self) -> bool:
        return self._queue.full()

    async def put(self, value: Any) -> None:
        if getattr(value, "kind", "") == "partial":
            await self.put_latest(value)
            return
        if getattr(value, "kind", "") == "final":
            self._pending_partial = None
            self._last_final_revision = max(self._last_final_revision, int(getattr(value, "revision", -1)))
        self._sequence += 1
        await self._queue.put((self._priority(value), self._sequence, value))

    async def put_latest(self, value: Any) -> bool:
        """Replace a queued partial and return whether a queue slot was used."""

        if getattr(value, "kind", "") != "partial":
            await self.put(value)
            return True
        self._pending_partial = value
        if self._partial_marker:
            return False
        self._sequence += 1
        try:
            self._queue.put_nowait((10, self._sequence, "__latest_partial__"))
        except asyncio.QueueFull:
            self._pending_partial = None
            raise
        self._partial_marker = True
        return True

    def put_nowait(self, value: Any) -> bool:
        if getattr(value, "kind", "") == "partial":
            return self.put_latest_nowait(value)
        if getattr(value, "kind", "") == "final":
            self._pending_partial = None
            self._last_final_revision = max(self._last_final_revision, int(getattr(value, "revision", -1)))
        self._sequence += 1
        self._queue.put_nowait((self._priority(value), self._sequence, value))
        return True

    def put_latest_nowait(self, value: Any) -> bool:
        if getattr(value, "kind", "") != "partial":
            self.put_nowait(value)
            return True
        self._pending_partial = value
        if self._partial_marker:
            return False
        self._sequence += 1
        try:
            self._queue.put_nowait((10, self._sequence, "__latest_partial__"))
        except asyncio.QueueFull:
            self._pending_partial = None
            raise
        self._partial_marker = True
        return True

    async def get(self) -> Any:
        while True:
            _priority, _sequence, value = await self._queue.get()
            if value == "__latest_partial__":
                self._partial_marker = False
                event = self._pending_partial
                self._pending_partial = None
                if event is None or int(getattr(event, "revision", -1)) <= self._last_final_revision:
                    self._queue.task_done()
                    continue
                return event
            return value

    def get_nowait(self) -> Any:
        while True:
            _priority, _sequence, value = self._queue.get_nowait()
            if value == "__latest_partial__":
                self._partial_marker = False
                event = self._pending_partial
                self._pending_partial = None
                if event is None or int(getattr(event, "revision", -1)) <= self._last_final_revision:
                    self._queue.task_done()
                    continue
                return event
            return value

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()


class LatestTranslationQueue:
    """Latest-wins, final-priority queue keyed by paragraph id."""

    def __init__(self, maxsize: int = 0) -> None:
        self.maxsize = max(0, int(maxsize))
        self._heap: list[tuple[int, int, str, int]] = []
        self._jobs: dict[str, dict[str, Any]] = {}
        self._queued: dict[str, tuple[int, int]] = {}
        self._inflight: set[str] = set()
        self._sequence = 0
        self._generation: dict[str, int] = {}
        self._unfinished = 0
        self._join_event = asyncio.Event()
        self._join_event.set()
        self._sentinel_inflight = 0
        self._changed = asyncio.Event()
        self._space_available = asyncio.Event()
        self._space_available.set()
        self.dropped_provisional = 0
        self.dropped_provisional_jobs: list[dict[str, Any]] = []

    def qsize(self) -> int:
        return len(self._queued) + len(self._inflight)

    def empty(self) -> bool:
        return not self._queued and not self._inflight

    def full(self) -> bool:
        return self.maxsize > 0 and self._unfinished >= self.maxsize

    @staticmethod
    def _priority(job: dict[str, Any]) -> int:
        return 0 if bool(job.get("final")) else 10

    def _evict_provisional(self) -> bool:
        """Free one queued provisional slot for an authoritative final."""

        for segment_id, (priority, _generation) in tuple(self._queued.items()):
            if priority <= 0:
                continue
            self._queued.pop(segment_id, None)
            dropped_job = self._jobs.pop(segment_id, None)
            self._unfinished = max(0, self._unfinished - 1)
            self.dropped_provisional += 1
            if dropped_job is not None:
                self.dropped_provisional_jobs.append(dropped_job)
                del self.dropped_provisional_jobs[:-128]
            if self._unfinished == 0:
                self._join_event.set()
            self._space_available.set()
            return True
        return False

    async def put(self, job: dict[str, Any] | None) -> None:
        if job is None:
            while self.full():
                self._space_available.clear()
                await self._space_available.wait()
            self._sequence += 1
            heapq.heappush(self._heap, (100, self._sequence, "__sentinel__", 0))
            self._unfinished += 1
            self._join_event.clear()
            self._changed.set()
            return
        await self.put_latest(job)

    async def put_latest(self, job: dict[str, Any]) -> bool:
        segment_id = str(job.get("segment_id", ""))
        if not segment_id:
            raise ValueError("translation job requires segment_id")
        already_queued = segment_id in self._queued
        already_inflight = segment_id in self._inflight
        while not already_queued and not already_inflight and self.full() and not bool(job.get("final")):
            self._space_available.clear()
            await self._space_available.wait()
            already_queued = segment_id in self._queued
            already_inflight = segment_id in self._inflight
        if not already_queued and not already_inflight and self.full() and bool(job.get("final")):
            self._evict_provisional()
        if already_queued:
            old_priority, _old_generation = self._queued[segment_id]
            # A final source is authoritative.  A late provisional update
            # must not downgrade a final job that has not started yet.
            if old_priority == 0 and self._priority(job) > old_priority:
                return False
            self._jobs[segment_id] = dict(job)
            new_priority = min(old_priority, self._priority(job))
            if new_priority < old_priority:
                generation = self._generation.get(segment_id, 0) + 1
                self._generation[segment_id] = generation
                self._sequence += 1
                self._queued[segment_id] = (new_priority, generation)
                heapq.heappush(self._heap, (new_priority, self._sequence, segment_id, generation))
            self._changed.set()
            return False
        self._jobs[segment_id] = dict(job)
        self._sequence += 1
        generation = self._generation.get(segment_id, 0) + 1
        self._generation[segment_id] = generation
        priority = self._priority(job)
        self._queued[segment_id] = (priority, generation)
        heapq.heappush(self._heap, (priority, self._sequence, segment_id, generation))
        self._unfinished += 1
        self._join_event.clear()
        self._changed.set()
        if self.full():
            self._space_available.clear()
        return True

    def put_nowait(self, job: dict[str, Any] | None) -> bool:
        if job is None:
            if self.full():
                raise asyncio.QueueFull
            self._sequence += 1
            heapq.heappush(self._heap, (100, self._sequence, "__sentinel__", 0))
            self._unfinished += 1
            self._join_event.clear()
            return True
        segment_id = str(job.get("segment_id", ""))
        if not segment_id:
            raise ValueError("translation job requires segment_id")
        already_queued = segment_id in self._queued
        already_inflight = segment_id in self._inflight
        if not already_queued and not already_inflight and self.full() and bool(job.get("final")):
            self._evict_provisional()
        if not already_queued and not already_inflight and self.full() and not bool(job.get("final")):
            raise asyncio.QueueFull
        if already_queued:
            old_priority, _old_generation = self._queued[segment_id]
            if old_priority == 0 and self._priority(job) > old_priority:
                return False
            self._jobs[segment_id] = dict(job)
            new_priority = min(old_priority, self._priority(job))
            if new_priority < old_priority:
                generation = self._generation.get(segment_id, 0) + 1
                self._generation[segment_id] = generation
                self._sequence += 1
                self._queued[segment_id] = (new_priority, generation)
                heapq.heappush(self._heap, (new_priority, self._sequence, segment_id, generation))
            return False
        self._jobs[segment_id] = dict(job)
        self._sequence += 1
        generation = self._generation.get(segment_id, 0) + 1
        self._generation[segment_id] = generation
        priority = self._priority(job)
        self._queued[segment_id] = (priority, generation)
        heapq.heappush(self._heap, (priority, self._sequence, segment_id, generation))
        self._unfinished += 1
        self._join_event.clear()
        return True

    async def get(self) -> dict[str, Any] | None:
        while True:
            try:
                return self.get_nowait()
            except asyncio.QueueEmpty:
                self._changed.clear()
                await self._changed.wait()

    def get_nowait(self) -> dict[str, Any] | None:
        while self._heap:
            _priority, _sequence, segment_id, generation = heapq.heappop(self._heap)
            if segment_id == "__sentinel__":
                self._sentinel_inflight += 1
                return None
            current = self._queued.get(segment_id)
            if current is None or current[1] != generation:
                continue
            self._queued.pop(segment_id, None)
            self._inflight.add(segment_id)
            return self._jobs.pop(segment_id, None)
        raise asyncio.QueueEmpty

    def task_done(self, job: dict[str, Any] | None = None) -> None:
        if job is not None:
            self._inflight.discard(str(job.get("segment_id", "")))
        elif self._sentinel_inflight:
            self._sentinel_inflight -= 1
        elif self._inflight:
            self._inflight.pop()
        if self._unfinished <= 0:
            raise ValueError("task_done() called too many times")
        self._unfinished -= 1
        if self._unfinished == 0:
            self._join_event.set()
        self._changed.set()
        self._space_available.set()

    def abort(self) -> None:
        """Drop queued/in-flight bookkeeping after a canceled worker.

        This is used only during session finalization after the worker has
        already been canceled.  It prevents ``join`` from waiting forever on
        a model call that was intentionally abandoned.
        """

        self._heap.clear()
        self._jobs.clear()
        self._queued.clear()
        self._inflight.clear()
        self._sentinel_inflight = 0
        self._unfinished = 0
        self._join_event.set()
        self._changed.set()
        self._space_available.set()

    async def join(self) -> None:
        await self._join_event.wait()


class GpuResourceManager:
    """Serializes GPU-heavy stages and exposes lightweight timing metrics."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._held = False
        self._waiters: list[tuple[int, int, threading.Event]] = []
        self._sequence = 0
        self.metrics: dict[str, Any] = {"waits": 0, "runs": 0, "durations_ms": {}, "wait_durations_ms": {}}

    @asynccontextmanager
    async def acquire(self, stage: str, *, priority: int = 50) -> AsyncIterator[None]:
        acquire_task = asyncio.create_task(asyncio.to_thread(self._acquire_sync, stage, priority))
        try:
            await asyncio.shield(acquire_task)
        except asyncio.CancelledError:
            async def release_after_acquire() -> None:
                try:
                    await acquire_task
                except Exception:
                    return
                self._release(stage, time.perf_counter())

            asyncio.create_task(release_after_acquire())
            raise
        started = time.perf_counter()
        try:
            yield
        finally:
            self._release(stage, started)

    def _acquire_sync(self, stage: str, priority: int) -> None:
        wait_started = time.perf_counter()
        with self._condition:
            self._sequence += 1
            ticket = (int(priority), self._sequence, threading.Event())
            self._waiters.append(ticket)
            self._waiters.sort(key=lambda item: (item[0], item[1]))
            if len(self._waiters) > 1:
                self.metrics["waits"] = int(self.metrics.get("waits", 0)) + 1
            while self._held or self._waiters[0] is not ticket:
                self._condition.wait()
            self._waiters.pop(0)
            self._held = True
        waits = self.metrics.setdefault("wait_durations_ms", {})
        waits[stage] = round(float(waits.get(stage, 0.0)) + (time.perf_counter() - wait_started) * 1000, 3)

    def _release(self, stage: str, started: float) -> None:
        durations = self.metrics.setdefault("durations_ms", {})
        durations[stage] = round(float(durations.get(stage, 0.0)) + (time.perf_counter() - started) * 1000, 3)
        self.metrics["runs"] = int(self.metrics.get("runs", 0)) + 1
        with self._condition:
            self._held = False
            self._condition.notify_all()

    @contextmanager
    def acquire_sync(self, stage: str, *, priority: int = 50) -> Any:
        self._acquire_sync(stage, priority)
        started = time.perf_counter()
        try:
            yield
        finally:
            self._release(stage, started)


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

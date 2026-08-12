from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, TypeVar


Stage = Literal["fast", "translation", "speaker", "refine", "export"]
T = TypeVar("T")


class ASRBackend(Protocol):
    async def transcribe(
        self,
        audio: Any,
        *,
        language: str | None,
        hotwords: list[str],
        partial: bool,
    ) -> Any:
        """Transcribe one audio segment without imposing a model choice."""


class TranslationBackend(Protocol):
    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> Any:
        """Translate text, returning a result with text and status."""


@dataclass(slots=True)
class InferenceJob:
    stage: Stage | str
    function: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


class InferenceScheduler(Protocol):
    async def submit(self, job: InferenceJob) -> Any:
        """Submit a job to the shared priority scheduler."""


class InferenceCapacityError(RuntimeError):
    """Raised when a process-local inference slot cannot be acquired in time."""


_PRIORITY: dict[str, int] = {
    "fast": 0,
    "translation": 1,
    "speaker": 2,
    "refine": 3,
    "export": 4,
}


@dataclass(order=True, slots=True)
class _QueuedJob:
    priority: int
    sequence: int
    stage: str = field(compare=False)
    function: Callable[..., Any] = field(compare=False)
    args: tuple[Any, ...] = field(compare=False)
    kwargs: dict[str, Any] = field(compare=False)
    future: asyncio.Future[Any] = field(compare=False)


class InferenceCoordinator:
    """Coordinate process-local model work with priority-aware GPU access.

    The normal single-GPU configuration uses one shared priority worker, so a
    refinement job can never occupy the same GPU concurrently with live ASR.
    The legacy multi-worker path is retained for CPU-oriented tests and future
    deployments that explicitly request more than one worker.
    """

    def __init__(
        self,
        *,
        fast_workers: int,
        refine_workers: int,
        wait_timeout_seconds: float,
        gpu_workers: int | None = None,
        gpu_memory_budget_mb: int = 7_200,
    ) -> None:
        self.fast_workers = max(1, fast_workers)
        self.refine_workers = max(1, refine_workers)
        self.wait_timeout_seconds = max(0.1, wait_timeout_seconds)
        self.gpu_workers = max(1, gpu_workers if gpu_workers is not None else 1)
        self.gpu_memory_budget_mb = max(1_024, gpu_memory_budget_mb)
        # Omitting gpu_workers is the legacy extension seam used by older
        # embedded callers and tests. The application passes it explicitly.
        self.priority_mode = gpu_workers is not None and self.gpu_workers == 1

        self._active = {stage: 0 for stage in _PRIORITY}
        self._waiting = {stage: 0 for stage in _PRIORITY}
        self._sequence = 0
        self._queue: asyncio.PriorityQueue[_QueuedJob] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._semaphores = {
            "fast": asyncio.Semaphore(self.fast_workers),
            "refine": asyncio.Semaphore(self.refine_workers),
        }

    def _ensure_priority_worker(self) -> None:
        if not self.priority_mode:
            return
        if self._queue is None:
            self._queue = asyncio.PriorityQueue()
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(
                self._priority_worker(), name="inference-priority-worker"
            )

    async def _priority_worker(self) -> None:
        assert self._queue is not None
        while True:
            job = await self._queue.get()
            self._waiting[job.stage] = max(0, self._waiting[job.stage] - 1)
            try:
                if job.future.cancelled():
                    continue
                self._active[job.stage] += 1
                try:
                    result = await asyncio.to_thread(
                        job.function, *job.args, **job.kwargs
                    )
                    if not job.future.cancelled():
                        job.future.set_result(result)
                except BaseException as exc:  # propagate worker failures to caller
                    if not job.future.cancelled():
                        job.future.set_exception(exc)
                finally:
                    self._active[job.stage] = max(0, self._active[job.stage] - 1)
            finally:
                self._queue.task_done()

    async def _run_legacy(
        self, stage: str, function: Callable[..., T], *args: Any, **kwargs: Any
    ) -> T:
        semaphore = self._semaphores["refine" if stage == "refine" else "fast"]
        counter_stage = "refine" if stage == "refine" else "fast"
        self._waiting[counter_stage] += 1
        try:
            try:
                await asyncio.wait_for(
                    semaphore.acquire(), timeout=self.wait_timeout_seconds
                )
            except TimeoutError as exc:
                raise InferenceCapacityError(
                    f"{stage} inference capacity wait exceeded "
                    f"{self.wait_timeout_seconds:.1f}s"
                ) from exc
        finally:
            self._waiting[counter_stage] -= 1

        self._active[counter_stage] += 1
        try:
            return await asyncio.to_thread(function, *args, **kwargs)
        finally:
            self._active[counter_stage] -= 1
            semaphore.release()

    async def run(
        self, stage: Stage | str, function: Callable[..., T], /, *args: Any, **kwargs: Any
    ) -> T:
        """Run a blocking model call under the configured priority policy."""

        normalized = stage if stage in _PRIORITY else "fast"
        if not self.priority_mode:
            return await self._run_legacy(normalized, function, *args, **kwargs)

        self._ensure_priority_worker()
        assert self._queue is not None
        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()
        self._sequence += 1
        self._waiting[normalized] += 1
        await self._queue.put(
            _QueuedJob(
                _PRIORITY[normalized],
                self._sequence,
                normalized,
                function,
                args,
                kwargs,
                future,
            )
        )
        try:
            return await asyncio.wait_for(
                asyncio.shield(future), timeout=self.wait_timeout_seconds
            )
        except TimeoutError as exc:
            future.cancel()
            raise InferenceCapacityError(
                f"{normalized} inference capacity wait exceeded "
                f"{self.wait_timeout_seconds:.1f}s"
            ) from exc

    async def submit(
        self,
        job: InferenceJob | Stage | str,
        function: Callable[..., T] | None = None,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Submit either an :class:`InferenceJob` or the legacy call shape."""

        if isinstance(job, InferenceJob):
            return await self.run(job.stage, job.function, *job.args, **job.kwargs)
        if function is None:
            raise TypeError("submit() requires a callable when stage is passed directly")
        return await self.run(job, function, *args, **kwargs)

    async def close(self) -> None:
        worker = self._worker
        self._worker = None
        if worker and not worker.done():
            worker.cancel()
            with __import__("contextlib").suppress(asyncio.CancelledError):
                await worker

    def snapshot(self) -> dict[str, dict[str, int] | int]:
        snapshot: dict[str, dict[str, int] | int] = {
            "fast": {
                "workers": 1 if self.priority_mode else self.fast_workers,
                "active": self._active["fast"],
                "waiting": self._waiting["fast"],
            },
            "refine": {
                "workers": 1 if self.priority_mode else self.refine_workers,
                "active": self._active["refine"],
                "waiting": self._waiting["refine"],
            },
        }
        if self.priority_mode:
            snapshot["translation"] = {
                "workers": 1,
                "active": self._active["translation"],
                "waiting": self._waiting["translation"],
            }
            snapshot["speaker"] = {
                "workers": 1,
                "active": self._active["speaker"],
                "waiting": self._waiting["speaker"],
            }
            snapshot["queue_depth"] = sum(self._waiting.values())
            snapshot["gpu_memory_budget_mb"] = self.gpu_memory_budget_mb
        return snapshot

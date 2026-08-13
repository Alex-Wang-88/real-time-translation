from __future__ import annotations

import asyncio

import pytest

from realtime_meeting.scheduler import GpuResourceManager, PostprocessTracker


@pytest.mark.asyncio
async def test_cancelled_async_gpu_waiter_does_not_leak_thread_lock() -> None:
    manager = GpuResourceManager()
    acquired = asyncio.Event()
    release = asyncio.Event()

    async def holder() -> None:
        async with manager.acquire("holder"):
            acquired.set()
            await release.wait()

    holder_task = asyncio.create_task(holder())
    await acquired.wait()
    waiter = asyncio.create_task(manager.acquire("cancelled").__aenter__())
    await asyncio.sleep(0.02)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    release.set()
    await holder_task

    async with manager.acquire("next"):
        pass


def test_postprocess_tracker_tolerates_malformed_persisted_fields() -> None:
    tracker = PostprocessTracker({
        "overall_percent": "bad",
        "stage_durations_ms": "bad",
        "stages": {"asr_refine": "bad", "translation": {"current": "bad"}},
    })

    assert tracker.overall_percent == 0
    assert tracker.stage_durations_ms == {}
    assert tracker.stages["asr_refine"]["state"] == "idle"
    assert tracker.stages["translation"]["current"] == 0

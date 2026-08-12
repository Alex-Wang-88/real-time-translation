from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import numpy as np
import pytest

from realtime_meeting.audio import SegmentEvent
from realtime_meeting.config import Settings
from realtime_meeting.inference import InferenceCoordinator
from realtime_meeting.models import Utterance
from realtime_meeting.session import CapacityLimitError, SessionManager
from realtime_meeting.speaker import OnlineSpeakerClusterer
from realtime_meeting.storage import RefinementJobStore


class ReadyRuntime:
    ready = True
    status = "ready"
    device = "cpu"


def test_legacy_utterance_defaults_to_refined_stage() -> None:
    item = Utterance.from_dict(
        {
            "id": 1,
            "start": 0.0,
            "end": 1.0,
            "speaker_id": 1,
            "language": "en",
            "language_confidence": 0.9,
            "text": "hello",
            "translation_zh": "你好",
        }
    )
    assert item.segment_revision == 0
    assert item.recognition_stage == "refined"


def test_refinement_store_is_idempotent_and_recovers_running_jobs(tmp_path: Path) -> None:
    store = RefinementJobStore(tmp_path / "jobs.sqlite3")
    output_dir = tmp_path / "meeting"
    event = SegmentEvent("final", b"\x01\x00" * 160, 0.0, 1.0, 7)
    first = store.enqueue(
        session_id="meeting-1",
        output_dir=output_dir,
        event=event,
        draft_text="draft",
        draft_language="en",
    )
    store.enqueue(
        session_id="meeting-1",
        output_dir=output_dir,
        event=event,
        draft_text="new draft",
        draft_language="en",
    )
    assert len(store.pending("meeting-1")) == 1
    store.mark_running(first)

    recovered = RefinementJobStore(tmp_path / "jobs.sqlite3")
    jobs = recovered.pending("meeting-1")
    assert len(jobs) == 1
    assert jobs[0].status == "queued"
    recovered.mark_done(jobs[0])
    assert recovered.pending_total() == 0
    assert not Path(jobs[0].pcm_path).exists()


@pytest.mark.asyncio
async def test_coordinator_enforces_worker_limit_across_many_jobs() -> None:
    coordinator = InferenceCoordinator(
        fast_workers=3, refine_workers=2, wait_timeout_seconds=5
    )
    active = 0
    peak = 0
    lock = threading.Lock()

    async def job() -> None:
        nonlocal active, peak

        def blocking() -> None:
            import time

            time.sleep(0.01)

            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.01)
            finally:
                with lock:
                    active -= 1

        await coordinator.run("refine", blocking)

    await asyncio.gather(*(job() for _ in range(50)))
    assert peak == 2
    assert coordinator.snapshot()["refine"] == {
        "workers": 2,
        "active": 0,
        "waiting": 0,
    }


@pytest.mark.asyncio
async def test_priority_coordinator_keeps_realtime_work_ahead_of_refinement() -> None:
    coordinator = InferenceCoordinator(
        fast_workers=1,
        refine_workers=1,
        wait_timeout_seconds=5,
        gpu_workers=1,
    )
    started = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def blocking(label: str) -> str:
        order.append(label)
        if label == "refine-1":
            started.set()
            release.wait(2)
        return label

    first = asyncio.create_task(coordinator.run("refine", blocking, "refine-1"))
    await asyncio.to_thread(started.wait, 2)
    second = asyncio.create_task(coordinator.run("refine", blocking, "refine-2"))
    translation = asyncio.create_task(
        coordinator.run("translation", blocking, "translation")
    )
    realtime = asyncio.create_task(coordinator.run("fast", blocking, "fast"))
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(first, realtime, translation, second)
    await coordinator.close()
    assert order == ["refine-1", "fast", "translation", "refine-2"]


def test_speaker_clusters_are_isolated_while_encoder_is_shared(monkeypatch) -> None:
    class Encoder:
        def embed_utterance(self, _wav):
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr("realtime_meeting.speaker.np.frombuffer", lambda *_a, **_k: np.ones(64_000))
    monkeypatch.setattr("resemblyzer.preprocess_wav", lambda wav, source_sr: wav)
    encoder = Encoder()
    first = OnlineSpeakerClusterer("cpu", encoder=encoder)
    second = OnlineSpeakerClusterer("cpu", encoder=encoder)
    assert first.encoder is second.encoder
    assert first.assign(b"pcm", 4.0) == 1
    assert len(first.clusters) == 1
    assert second.clusters == []


@pytest.mark.asyncio
async def test_session_manager_allows_configured_concurrency(tmp_path: Path) -> None:
    manager = SessionManager(
        Settings(
            results_dir=tmp_path,
            max_concurrent_meetings=2,
            disk_warn_bytes=0,
            disk_stop_bytes=0,
        ),
        ReadyRuntime(),
    )
    first = await manager.create(owner_id="alice")
    second = await manager.create(owner_id="bob")
    with pytest.raises(CapacityLimitError):
        await manager.create(owner_id="carol")
    assert first.owner_id == "alice"
    assert second.owner_id == "bob"
    await asyncio.gather(first.stop(), second.stop())

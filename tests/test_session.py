from __future__ import annotations

import json
from pathlib import Path

import pytest

from realtime_meeting.config import Settings
from realtime_meeting.models import Utterance
from realtime_meeting.session import LiveMeetingSession, SessionManager


class ReadyRuntime:
    ready = True
    status = "ready"
    device = "cpu"


@pytest.mark.asyncio
async def test_feed_audio_rejects_oversized_and_odd_pcm_packets(tmp_path: Path):
    settings = Settings(
        results_dir=tmp_path,
        max_audio_packet_bytes=4,
        disk_warn_bytes=0,
        disk_stop_bytes=0,
    )
    session = LiveMeetingSession(settings, ReadyRuntime())
    await session.start()
    try:
        with pytest.raises(ValueError, match="4"):
            await session.feed_audio(b"\x00" * 6)
        with pytest.raises(ValueError, match="偶数"):
            await session.feed_audio(b"\x00")
        assert session.audio_packets_received == 0
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_stop_failure_is_persisted_without_losing_error(tmp_path: Path, monkeypatch):
    settings = Settings(results_dir=tmp_path, disk_warn_bytes=0, disk_stop_bytes=0)
    session = LiveMeetingSession(settings, ReadyRuntime())
    await session.start()

    def fail_export(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("realtime_meeting.session.export_live_result", fail_export)
    await session.stop()

    assert session.state == "error"
    assert "disk full" in (session.error or "")
    saved = json.loads((session.output_dir / "session_state.json").read_text(encoding="utf-8"))
    assert saved["state"] == "error"
    assert "disk full" in saved["error"]


def test_summary_claim_is_single_use(tmp_path: Path):
    session = LiveMeetingSession(
        Settings(results_dir=tmp_path, disk_warn_bytes=0, disk_stop_bytes=0),
        ReadyRuntime(),
    )
    session.state = "summary_pending"
    assert session.begin_summary() is True
    assert session.state == "summarizing"
    assert session.begin_summary() is False


def test_recovery_restores_terminal_state_audio_and_safe_files(tmp_path: Path):
    output_dir = tmp_path / "20260808-120000-recovered"
    output_dir.mkdir()
    item = Utterance(1, 0.5, 2.0, 1, "en", 0.9, "hello", "你好")
    (output_dir / "transcript.jsonl").write_text(
        json.dumps(item.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "audio_manifest.json").write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "file": "audio/audio-0001.wav",
                        "start_seconds": 0,
                        "end_seconds": 4.0,
                        "samples": 64_000,
                        "format": "wav",
                    },
                    {"file": "bad.wav", "samples": "not-a-number"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "meeting_transcript.md").write_text("saved", encoding="utf-8")
    (output_dir / "session_state.json").write_text(
        json.dumps(
            {
                "id": "recovered-id",
                "state": "summary_pending",
                "started_at": "2026-08-08T12:00:00+00:00",
                "ended_at": "2026-08-08T12:01:00+00:00",
                "error": None,
                "processing_error": "one segment failed",
                "audio_bytes_received": 128_000,
                "audio_packets_received": 12,
                "audio_samples_received": 64_000,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "session_state.json.tmp").write_text("stale", encoding="utf-8")

    settings = Settings(results_dir=tmp_path, disk_warn_bytes=0, disk_stop_bytes=0)
    manager = SessionManager(settings, ReadyRuntime())
    session = manager.active()

    assert session is not None
    assert session.id == "recovered-id"
    assert session.state == "summary_pending"
    assert session.processing_error == "one segment failed"
    assert session.audio_segments[0]["file"] == "audio-0001.wav"
    assert session.audio_bytes_received == 128_000
    assert session.audio_packets_received == 12
    assert session.audio_samples_received == 64_000
    assert session.elapsed_seconds == 4.0
    assert "meeting_transcript.md" in session.files
    assert "session_state.json" not in session.files
    assert "session_state.json.tmp" not in session.files

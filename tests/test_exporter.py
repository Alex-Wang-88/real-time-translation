from __future__ import annotations

from realtime_meeting.exporter import export_live_result
from realtime_meeting.models import Utterance


def test_exporter_sanitizes_recovered_language_and_audio_filenames(tmp_path) -> None:
    files = export_live_result(
        tmp_path,
        meeting_id="safe",
        title="safe",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:01+00:00",
        duration_seconds=1.0,
        utterances=[
            Utterance(1, 0.0, 1.0, 1, "../../outside", 1.0, "text", segment_id="1:0")
        ],
        audio_segments=[{"file": "../outside.flac"}],
        recording_state="complete",
        summary_state="idle",
        todo_state="waiting_summary",
    )

    assert "original_outside.md" in files
    assert "audio/outside.flac" in files
    assert not (tmp_path.parent / "outside.md").exists()

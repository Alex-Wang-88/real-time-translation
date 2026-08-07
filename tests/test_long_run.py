from pathlib import Path

import numpy as np

from realtime_meeting.audio import RotatingAudioWriter
from realtime_meeting.config import Settings
from realtime_meeting.exporter import append_utterance, load_utterances
from realtime_meeting.models import Utterance
from realtime_meeting.session import LiveMeetingSession


class ReadyRuntime:
    ready = True


def test_accelerated_four_hour_rotation_and_bounded_recent_memory(
    tmp_path: Path, monkeypatch
) -> None:
    # One synthetic sample represents one second, so four hours needs only 28 KB.
    monkeypatch.setattr("realtime_meeting.audio.SAMPLE_RATE", 1)
    monkeypatch.setattr("realtime_meeting.audio.shutil.which", lambda _name: None)
    writer = RotatingAudioWriter(tmp_path / "audio", segment_minutes=30)
    writer.write(np.zeros(4 * 60 * 60, dtype=np.int16).tobytes())
    segments = writer.close()
    assert len(segments) == 8
    assert all(item["samples"] == 30 * 60 for item in segments)

    session = LiveMeetingSession(
        Settings(results_dir=tmp_path / "results"), ReadyRuntime()
    )
    for index in range(1, 1_001):
        item = Utterance(index, index, index + 0.5, 1, "en", 1.0, f"line {index}", f"line {index}")
        append_utterance(session.transcript_path, item)
        session.recent.append(item)
    assert len(session.recent) == 500
    assert session.recent[0].id == 501
    assert len(load_utterances(session.transcript_path)) == 1_000

from __future__ import annotations

import json
from pathlib import Path

from realtime_meeting.exporter import append_utterance, export_live_result, load_utterances
from realtime_meeting.models import Utterance


def test_live_export_contract(tmp_path: Path):
    jsonl = tmp_path / "transcript.jsonl"
    items = [
        Utterance(1, 0.1, 2.2, 1, "zh", 0.98, "你好", "你好"),
        Utterance(2, 2.3, 4.8, 2, "de", 0.91, "Guten Morgen", "早上好"),
    ]
    for item in items:
        append_utterance(jsonl, item)
    loaded = load_utterances(jsonl)
    files = export_live_result(
        tmp_path,
        session_id="session-1",
        started_at="start",
        ended_at="end",
        duration_seconds=5,
        utterances=loaded,
        audio_segments=[{"file": "audio-0001.flac"}],
        status="summary_pending",
    )
    assert "meeting_transcript.md" in files
    transcript = (tmp_path / "meeting_transcript.md").read_text(encoding="utf-8")
    assert "演讲人1（中文）：“你好”" in transcript
    assert "演讲人1（中文翻译）：“你好”" in transcript
    assert "演讲人2（德文）：“Guten Morgen”" in transcript
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["status"] == "summary_pending"

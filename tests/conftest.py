from __future__ import annotations

from pathlib import Path

import pytest

from realtime_meeting.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        results_dir=tmp_path / "meetings",
        translation_model_root=tmp_path / "models",
        jimo_api_url="https://summary.example.test/v2/chat/completions/share?shareId=summary",
        jimo_todo_api_url="https://todo.example.test/v2/chat/completions/share?shareId=todo",
        jimo_authorization="Bearer test-secret",
        audio_pre_roll_ms=40,
        speech_start_ms=40,
        silence_ms=160,
        # Unit-test audio uses short synthetic utterances; production
        # Settings defaults to 450 ms to reject click/noise bursts.
        vad_minimum_speech_ms=300,
        partial_interval_ms=200,
        max_utterance_seconds=4,
        max_active_meetings=1,
    )

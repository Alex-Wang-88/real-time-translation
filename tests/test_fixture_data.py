from __future__ import annotations

import json
from pathlib import Path

from realtime_meeting.jimo import paired_text, transcript_chunks
from realtime_meeting.models import Utterance


FIXTURE = Path(__file__).parent / "fixtures" / "sample_meeting.jsonl"


def _load_fixture() -> list[Utterance]:
    return [
        Utterance.from_dict(json.loads(line))
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_sample_meeting_fixture_covers_supported_languages_and_translations() -> None:
    items = _load_fixture()
    assert len(items) == 8
    assert {item.language for item in items} == {"zh", "en", "de"}
    assert all(item.segment_id.startswith("sample-") for item in items)
    assert all(item.end > item.start for item in items)
    assert all(item.translation_zh for item in items if item.language != "zh")
    assert "中文翻译" in paired_text(next(item for item in items if item.language == "en"))


def test_sample_meeting_fixture_can_be_chunked_without_splitting_utterances() -> None:
    chunks = list(transcript_chunks(FIXTURE, max_chars=360))
    assert len(chunks) >= 2
    assert [chunk[0] for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert all(start <= end for _, start, end, _ in chunks)
    combined = "\n".join(text for _, _, _, text in chunks)
    assert "Q3" in combined
    assert "Anna prüft" in combined
    assert "Li Lei" in combined

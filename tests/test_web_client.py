from pathlib import Path


APP_JS = Path(__file__).parents[1] / "realtime_meeting" / "web" / "app.js"


def test_postprocess_does_not_open_or_reconnect_audio_websocket() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert 'if (["recording", "starting", "finalizing"].includes(snapshot.recording_state))' in source
    assert "const postProcessing =" not in source
    assert 'closeStream(true);\n    setConnection("后台处理中", "warning");' in source


def test_audio_worklet_keeps_resampling_phase_across_callbacks() -> None:
    source = (APP_JS.parent / "audio-worklet.js").read_text(encoding="utf-8")
    assert "this.resamplePosition" in source
    assert "this.resampleInput" in source
    assert "Math.floor(mono.length / ratio)" not in source


def test_web_client_supports_cookie_auth_and_full_transcript_pages() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    assert 'credentials: "same-origin"' in source
    assert 'fetch("/api/v2/auth/session"' in source
    assert "async function loadFullTranscript(id)" in source
    assert "/transcript?offset=${offset}&limit=${limit}" in source


def test_summary_button_waits_for_automatic_refinement() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    assert 'const preprocessingReady = ["asr_refine", "diarization", "translation"].every' in source
    assert '"生成纪要和 To-do-list"' in source
    assert 'value.state === "ready_for_summary"' in source


def test_failed_summary_stream_restores_committed_summary() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    assert "summary: payload.summary ?? state.meeting.summary" in source
    assert "summary_revision: payload.summary_revision ?? state.meeting.summary_revision" in source

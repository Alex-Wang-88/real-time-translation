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


def test_threshold_slider_overlays_live_microphone_level() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")

    assert 'id="microphoneLevelFill"' in html
    assert 'id="microphoneLevelMarker"' in html
    assert "function renderMicrophoneLevel" in source
    assert "const MICROPHONE_METER_MAX_PERCENT = 100;" in source
    assert "const meterMax = MICROPHONE_METER_MAX_PERCENT;" in source
    assert "level / meterMax * 100" in source
    assert '"低于阈值，将被过滤"' in source
    assert "state.audioStreamingEnabled && state.ws?.readyState" in source
    assert "await startMicrophonePreview()" in source


def test_web_client_supports_cookie_auth_and_full_transcript_pages() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    assert 'credentials: "same-origin"' in source
    assert 'fetch("/api/v2/auth/session"' in source
    assert "async function loadFullTranscript(id)" in source
    assert "/transcript?offset=${offset}&limit=${limit}" in source
    assert "state.transcript = transcript" in source
    assert "snapshot.snapshot_revision || 0) > previousRevision" in source


def test_create_requires_manual_start_backend_contract() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    create_section = source.split("async function createMeeting(title)", 1)[1].split("async function startRecording()", 1)[0]

    assert 'health.meeting_start_mode !== "manual"' in create_section
    assert 'snapshot.recording_state !== "created"' in create_section
    assert "prepareMicrophone" not in create_section
    assert "connectStream" not in create_section
    assert '/start`, { method: "POST" }' in source


def test_summary_button_waits_for_automatic_refinement() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    assert 'const preprocessingReady = ["asr_refine", "diarization", "translation"].every' in source
    assert '"生成纪要和 To-do-list"' in source
    assert 'value.state === "ready_for_summary"' in source


def test_failed_summary_stream_restores_committed_summary() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    assert "summary: payload.summary ?? state.meeting.summary" in source
    assert "summary_revision: payload.summary_revision ?? state.meeting.summary_revision" in source


def test_live_transcript_updates_are_incremental() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    event_section = source.split("async function handleEvent(payload)", 1)[1]

    assert "transcriptNodes: new Map()" in source
    assert "function renderTranscriptItem(segmentId" in source
    assert "if (changed) renderTranscriptItem(payload.utterance.segment_id, true);" in event_section
    assert "removeTranscriptItem(payload.segment_id);" in event_section
    assert "renderTranscriptItem(payload.segment_id);" in event_section


def test_asr_settings_panel_uses_sliders_and_refresh_buttons_keep_hover_only() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
    css = (APP_JS.parent / "styles.css").read_text(encoding="utf-8")

    assert 'id="openAsrSettings"' in html
    assert 'id="asrSettingsDialog"' in html
    assert 'id="settingsAudioSection"' in html
    assert 'id="inputDeviceSummary"' in html
    assert 'class="settings-slider threshold-slider"' in html
    assert "/settings`, {" in source
    assert "识别设置只能在录音开始前调整" in source
    assert ".icon-button:hover { border-color" in css
    assert ".icon-button:hover { transform" not in css
    assert ".sidebar .icon-button:hover { color" in css
    assert "transform: rotate" not in css


def test_action_buttons_share_the_settings_button_visual_spec() -> None:
    css = (APP_JS.parent / "styles.css").read_text(encoding="utf-8")
    html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")

    assert ".primary-button, .ghost-button, .text-button, .settings-button, .record-button" in css
    assert "min-width: 96px; height: 38px; min-height: 38px;" in css
    assert 'id="openAsrSettings" class="settings-button"' in html
    assert 'id="deleteMeeting" class="ghost-button danger"' in html
    assert 'id="saveAsrSettings" class="primary-button"' in html
    assert ".dialog-close:hover:not(:disabled)" in css
    assert "width: 38px; min-width: 38px; height: 38px; min-height: 38px;" in css

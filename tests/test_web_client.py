from pathlib import Path


APP_JS = Path(__file__).parents[1] / "realtime_meeting" / "web" / "app.js"


def test_postprocess_does_not_open_or_reconnect_audio_websocket() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert 'if (["recording", "starting", "finalizing"].includes(snapshot.recording_state))' in source
    assert "const postProcessing =" not in source
    assert 'closeStream(true);\n    setConnection("后台处理中", "warning");' in source

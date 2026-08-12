from pathlib import Path

from realtime_meeting.config import Settings, persist_device_preference


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "realtime_meeting" / "web"


def test_web_client_contains_complete_browser_capture_controls() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    app = (WEB / "app.js").read_text(encoding="utf-8")

    for marker in (
        'id="inputDeviceSelect"',
        'id="refreshInputDevices"',
        'id="inputWarning"',
        'id="deviceSelect"',
        'id="retrySummary"',
    ):
        assert marker in html
    for marker in (
        "enumerateDevices",
        "getUserMedia",
        "AudioWorkletNode",
        "stream-ticket",
        "audio_config",
        "translation_update",
        "utterance_update",
        "retry-summary",
        "summary_delta",
        "renderDownloads",
    ):
        assert marker in app
    for marker in (
        "requestAnimationFrame",
        "passive: true",
        "MAX_RENDERED_UTTERANCES",
        "content-visibility: auto",
    ):
        assert marker in app or marker in (WEB / "styles.css").read_text(encoding="utf-8")
    assert "ui.transcriptList.scrollTop = ui.transcriptList.scrollHeight" in app
    assert "ui.transcriptList.addEventListener(\"scroll\", () => {" in app

    worklet = (WEB / "audio-worklet.js").read_text(encoding="utf-8")
    for marker in ("_sinc", "halfTaps", "packetSamples", "targetRate", "this.output.splice(0, this.packetSamples)"):
        assert marker in worklet


def test_pyqt_client_and_desktop_entrypoint_are_removed() -> None:
    assert not (ROOT / "realtime_meeting" / "desktop.py").exists()
    assert not (ROOT / "tests" / "test_desktop.py").exists()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "PyQt6" not in pyproject
    assert "meeting-desktop" not in pyproject


def test_web_device_switch_persistence_is_opt_in(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("JIMO_AUTHORIZATION=keep-me\nMEETING_DEVICE=auto\n", encoding="utf-8")

    persist_device_preference(Settings(environment_file=env_file), "cuda")

    assert env_file.read_text(encoding="utf-8") == (
        "JIMO_AUTHORIZATION=keep-me\nMEETING_DEVICE=cuda\n"
    )

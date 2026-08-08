from __future__ import annotations

import time
from pathlib import Path

import realtime_meeting.server as server_module
from fastapi.testclient import TestClient

from realtime_meeting.audio import SAMPLE_RATE
from realtime_meeting.config import Settings
from realtime_meeting.models import Utterance
from realtime_meeting.runtime import PartialResult
from realtime_meeting.server import create_app
from tests.test_audio import pcm_silence, pcm_tone


class FakeRuntime:
    ready = True
    status = "模型已就绪"
    device = "cuda"

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True
        self.ready = False

    def transcribe_partial(self, event, recent_text="", hotwords=None):
        return PartialResult(event.revision, event.start, event.end, "临时字幕", "zh")

    def transcribe_final(
        self, event, *, next_id, previous_language, recent_text="", hotwords=None
    ):
        return [
            Utterance(
                next_id,
                event.start,
                event.end,
                1,
                "zh",
                0.99,
                "测试会议内容",
                "测试会议内容",
            )
        ]


def test_health_does_not_expose_authorization_and_single_meeting(tmp_path: Path):
    config = Settings(
        results_dir=tmp_path,
        disk_warn_bytes=0,
        disk_stop_bytes=0,
        jimo_api_url="https://example.test/api",
        jimo_authorization="super-secret-value",
    )
    app = create_app(config, FakeRuntime(), load_models=False)
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ready"
        assert "super-secret-value" not in health.text
        first = client.post("/api/meetings", json={})
        assert first.status_code == 201
        second = client.post("/api/meetings", json={})
        assert second.status_code == 409
        client.post(f"/api/meetings/{first.json()['id']}/stop")


def test_non_loopback_requires_configured_token_and_accepts_query_token(tmp_path: Path):
    config = Settings(
        host="0.0.0.0",
        api_token="test-token",
        results_dir=tmp_path,
        disk_warn_bytes=0,
        disk_stop_bytes=0,
    )
    app = create_app(config, FakeRuntime(), load_models=False)
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 401
        assert client.get(
            "/api/health", headers={"Authorization": "Bearer test-token"}
        ).status_code == 200
        assert client.get("/api/health?token=test-token").status_code == 200


def test_device_switch_updates_runtime_used_by_new_meetings(tmp_path: Path, monkeypatch):
    class SwitchRuntime:
        instances = []

        def __init__(self, _asr_model, _translation_model, requested_device):
            self.device = requested_device
            self.status = "等待加载"
            self.ready = False
            self.closed = False
            self.__class__.instances.append(self)

        def load(self):
            self.ready = True
            self.status = "模型已就绪"

        def close(self):
            self.closed = True
            self.ready = False

    monkeypatch.setattr(server_module, "LiveModelRuntime", SwitchRuntime)
    previous = FakeRuntime()
    config = Settings(results_dir=tmp_path, disk_warn_bytes=0, disk_stop_bytes=0)
    app = create_app(config, previous, load_models=False)
    with TestClient(app) as client:
        response = client.post("/api/device", json={"device": "cpu"})
        assert response.status_code == 202
        deadline = time.monotonic() + 2
        while app.state.switching and time.monotonic() < deadline:
            time.sleep(0.01)
        health = client.get("/api/health").json()
        assert health["device"] == "cpu"
        assert health["status"] == "ready"
        assert previous.closed is True

        created = client.post("/api/meetings", json={})
        assert created.status_code == 201
        assert app.state.manager.active().runtime is app.state.runtime
        client.post(f"/api/meetings/{created.json()['id']}/stop")


def test_new_meeting_is_rejected_while_device_switch_is_in_flight(tmp_path: Path):
    config = Settings(results_dir=tmp_path, disk_warn_bytes=0, disk_stop_bytes=0)
    app = create_app(config, FakeRuntime(), load_models=False)
    app.state.switching = True
    with TestClient(app) as client:
        response = client.post("/api/meetings", json={})
    assert response.status_code == 409


def test_websocket_stream_emits_strict_utterance_and_saves_on_stop(tmp_path: Path):
    config = Settings(results_dir=tmp_path, disk_warn_bytes=0, disk_stop_bytes=0)
    app = create_app(config, FakeRuntime(), load_models=False)
    with TestClient(app) as client:
        created = client.post("/api/meetings", json={}).json()
        session_id = created["id"]
        with client.websocket_connect(f"/api/meetings/{session_id}/stream") as websocket:
            assert websocket.receive_json()["type"] == "snapshot"
            websocket.send_bytes(pcm_silence(0.3) + pcm_tone(1.2) + pcm_silence(0.7))
            utterance = None
            audio_input = None
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                event = websocket.receive_json()
                if event["type"] == "audio_input":
                    audio_input = event
                if event["type"] == "utterance":
                    utterance = event["utterance"]
                if utterance is not None and audio_input is not None:
                    break
            assert audio_input and audio_input["packets_received"] >= 1
            assert utterance is not None
            assert utterance["language"] == "zh"
            assert utterance["translation_zh"] == "测试会议内容"
            response = client.post(f"/api/meetings/{session_id}/stop")
            assert response.status_code == 202
            summary_ready = None
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                event = websocket.receive_json()
                if event["type"] == "summary_pending":
                    summary_ready = event
                    break
            assert summary_ready and summary_ready["session_id"] == session_id
            snapshot = client.get(f"/api/meetings/{session_id}").json()
            assert snapshot["state"] == "summary_pending"

            # AI is not called during stop. It starts only after the explicit
            # manual request, and this unconfigured test client then fails in
            # the same retryable way as the production UI will surface.
            response = client.post(f"/api/meetings/{session_id}/retry-summary")
            assert response.status_code == 202
            summary_error = None
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                event = websocket.receive_json()
                if event["type"] == "error" and event["code"] == "summary_failed":
                    summary_error = event
                    break
            assert summary_error and summary_error["retryable"] is True
        snapshot = client.get(f"/api/meetings/{session_id}").json()
        assert snapshot["state"] == "summary_error"
        result_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
        transcript = (result_dir / "meeting_transcript.md").read_text(encoding="utf-8")
        assert "演讲人1（中文）：“测试会议内容”" in transcript
        assert "演讲人1（中文翻译）：“测试会议内容”" in transcript

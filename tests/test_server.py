from __future__ import annotations

import time
from pathlib import Path

import realtime_meeting.server as server_module
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

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

    def transcribe_draft(self, event, recent_text="", hotwords=None):
        return self.transcribe_partial(event, recent_text, hotwords)

    def transcribe_final(
        self,
        event,
        *,
        next_id,
        previous_language,
        recent_text="",
        hotwords=None,
        speaker_clusterer=None,
        refined=True,
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
                segment_revision=event.revision,
                recognition_stage="refined" if refined else "fast",
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
        assert second.status_code == 429
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


def test_trusted_gateway_identity_enforces_meeting_ownership(tmp_path: Path):
    config = Settings(
        host="0.0.0.0",
        results_dir=tmp_path,
        disk_warn_bytes=0,
        disk_stop_bytes=0,
        trusted_proxy_auth=True,
        trusted_proxy_service_token="gateway-secret",
        trusted_proxy_cidrs=("0.0.0.0/0", "::/0"),
    )
    app = create_app(config, FakeRuntime(), load_models=False)

    def headers(user: str) -> dict[str, str]:
        return {
            "x-meeting-service-token": "gateway-secret",
            "x-meeting-user": user,
        }

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        created = client.post("/api/meetings", json={}, headers=headers("alice"))
        assert created.status_code == 201
        session_id = created.json()["id"]
        assert client.get(
            f"/api/meetings/{session_id}", headers=headers("alice")
        ).status_code == 200
        assert client.get(
            f"/api/meetings/{session_id}", headers=headers("bob")
        ).status_code == 404
        assert client.post(
            f"/api/meetings/{session_id}/stop", headers=headers("alice")
        ).status_code == 202


def test_device_switch_updates_runtime_used_by_new_meetings(tmp_path: Path, monkeypatch):
    class SwitchRuntime:
        instances = []

        def __init__(
            self,
            _asr_model,
            _translation_model,
            requested_device,
            _refine_asr_model=None,
            _refinement_enabled=True,
        ):
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
            draft = None
            event_order = []
            audio_input = None
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                event = websocket.receive_json()
                event_order.append(event["type"])
                if event["type"] == "audio_input":
                    audio_input = event
                if event["type"] == "draft":
                    draft = event
                if event["type"] == "utterance":
                    utterance = event["utterance"]
                if utterance is not None and audio_input is not None:
                    break
            assert audio_input and audio_input["packets_received"] >= 1
            assert draft is not None
            assert "translation_zh" not in draft
            assert utterance is not None
            assert event_order.index("draft") < event_order.index("utterance")
            assert utterance["segment_revision"] == draft["revision"]
            assert utterance["recognition_stage"] == "refined"
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


def test_stream_ticket_authenticates_once_and_audio_config_is_acknowledged(tmp_path: Path):
    config = Settings(
        host="0.0.0.0",
        api_token="api-secret",
        results_dir=tmp_path,
        disk_warn_bytes=0,
        disk_stop_bytes=0,
    )
    app = create_app(config, FakeRuntime(), load_models=False)
    headers = {"Authorization": "Bearer api-secret"}
    with TestClient(app) as client:
        created = client.post("/api/meetings", json={}, headers=headers)
        assert created.status_code == 201
        session_id = created.json()["id"]
        ticket_response = client.post(
            f"/api/meetings/{session_id}/stream-ticket", headers=headers
        )
        assert ticket_response.status_code == 201
        ticket = ticket_response.json()["ticket"]

        with client.websocket_connect(f"/api/meetings/{session_id}/stream") as websocket:
            websocket.send_json({"type": "auth", "ticket": ticket})
            assert websocket.receive_json()["type"] == "auth_ok"
            assert websocket.receive_json()["type"] == "snapshot"
            websocket.send_json(
                {
                    "type": "audio_config",
                    "sample_rate": 16000,
                    "channels": 1,
                    "encoding": "pcm_s16le",
                    "packet_ms": 40,
                    "sequence_header": True,
                }
            )
            acknowledgement = websocket.receive_json()
            assert acknowledgement["type"] == "audio_config_ack"
            assert acknowledgement["sequence_header"] is True
            websocket.send_bytes((0).to_bytes(4, "little") + pcm_silence(0.04))

        client.post(f"/api/meetings/{session_id}/stop", headers=headers)
        with client.websocket_connect(f"/api/meetings/{session_id}/stream") as replay:
            replay.send_json({"type": "auth", "ticket": ticket})
            with pytest.raises(WebSocketDisconnect) as error:
                replay.receive_json()
            assert error.value.code == 4403


def test_translation_is_emitted_after_source_without_blocking_it(tmp_path: Path):
    class AsyncTranslationRuntime(FakeRuntime):
        def transcribe_final(
            self,
            event,
            *,
            next_id,
            previous_language,
            recent_text="",
            hotwords=None,
            speaker_clusterer=None,
            refined=True,
        ):
            return [
                Utterance(
                    next_id,
                    event.start,
                    event.end,
                    1,
                    "de",
                    0.95,
                    "Guten Morgen",
                    "",
                    segment_revision=event.revision,
                    recognition_stage="refined" if refined else "fast",
                    translation_status="pending",
                )
            ]

        def translate_text_batch(self, texts, source_language):
            return [
                {"text": "早上好", "status": "ready"}
                for _text in texts
            ]

    config = Settings(
        results_dir=tmp_path,
        refinement_enabled=False,
        disk_warn_bytes=0,
        disk_stop_bytes=0,
    )
    app = create_app(config, AsyncTranslationRuntime(), load_models=False)
    with TestClient(app) as client:
        created = client.post("/api/meetings", json={}).json()
        session_id = created["id"]
        with client.websocket_connect(f"/api/meetings/{session_id}/stream") as websocket:
            assert websocket.receive_json()["type"] == "snapshot"
            websocket.send_bytes(pcm_silence(0.3) + pcm_tone(1.2) + pcm_silence(0.7))
            source_event = None
            translation_event = None
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and translation_event is None:
                event = websocket.receive_json()
                if event["type"] == "utterance":
                    source_event = event
                elif event["type"] == "translation_update":
                    translation_event = event
            assert source_event is not None
            assert source_event["utterance"]["translation_status"] == "pending"
            assert translation_event is not None
            assert translation_event["segment_id"] == source_event["utterance"]["segment_id"]
            assert translation_event["translation_zh"] == "早上好"
            assert translation_event["translation_status"] == "ready"
            client.post(f"/api/meetings/{session_id}/stop")

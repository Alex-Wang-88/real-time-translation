from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from realtime_meeting.config import Settings
from realtime_meeting.exporter import append_utterance
from realtime_meeting.models import Utterance
from realtime_meeting.server import create_app
from realtime_meeting.storage import LocalMeetingStore


class ReadyRuntime:
    ready = True
    device = "cpu"
    status = "模型已就绪"
    metrics = {}

    def new_vad(self):
        return None

    def new_speaker_clusterer(self):
        return None


def test_startup_purges_only_confirmed_expired_meetings(tmp_path) -> None:
    settings = Settings(
        results_dir=tmp_path / "meetings",
        translation_model_root=tmp_path / "models",
        retention_days=1,
    )
    store = LocalMeetingStore(settings.results_dir)
    expired = store.meeting_dir("expired")
    expired.mkdir(parents=True)
    (expired / "session_state.json").write_text(json.dumps({
        "id": "expired",
        "recording_state": "complete",
        "ended_at": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
    }), encoding="utf-8")
    corrupt = store.meeting_dir("corrupt")
    corrupt.mkdir(parents=True)
    (corrupt / "session_state.json").write_text("not-json", encoding="utf-8")
    app = create_app(settings, ReadyRuntime(), load_models=False, store=store)
    with TestClient(app):
        assert not expired.exists()
        assert app.state.manager.get("expired") is None
        assert corrupt.exists()


def test_health_does_not_expose_authorization_and_delete_recovers(tmp_path) -> None:
    settings = Settings(
        host="0.0.0.0",
        api_token="local-token",
        results_dir=tmp_path / "meetings",
        translation_model_root=tmp_path / "models",
        jimo_api_url="https://summary.test",
        jimo_todo_api_url="https://todo.test",
        jimo_authorization="super-secret-authorization",
    )
    app = create_app(settings, ReadyRuntime(), load_models=False, store=LocalMeetingStore(settings.results_dir))
    headers = {"Authorization": "Bearer local-token"}
    with TestClient(app) as client:
        health = client.get("/api/v2/health", headers=headers)
        assert health.status_code == 200
        assert "super-secret" not in health.text
        created = client.post("/api/v2/meetings", headers=headers, json={"title": "API 测试"})
        assert created.status_code == 201
        meeting_id = created.json()["id"]
        ticket = client.post(f"/api/v2/meetings/{meeting_id}/stream-ticket", headers=headers)
        assert ticket.status_code == 201
        with client.websocket_connect(f"/api/v2/meetings/{meeting_id}/stream") as websocket:
            websocket.send_json({"type": "auth", "ticket": ticket.json()["ticket"]})
            assert websocket.receive_json()["type"] == "auth_ok"
            assert websocket.receive_json()["type"] == "snapshot"
            websocket.send_json({"type": "audio_config", "sample_rate": 16000, "channels": 1, "encoding": "pcm_s16le", "packet_ms": 40, "sequence_header": True})
            assert websocket.receive_json()["type"] == "audio_config_ack"
        assert client.post(f"/api/v2/meetings/{meeting_id}/stop", headers=headers).status_code == 202
        with pytest.raises(WebSocketDisconnect) as disconnect:
            with client.websocket_connect(f"/api/v2/meetings/{meeting_id}/stream") as reused:
                reused.send_json({"type": "auth", "ticket": ticket.json()["ticket"]})
                reused.receive_json()
        assert disconnect.value.code == 4403
        for _ in range(30):
            snapshot = client.get(f"/api/v2/meetings/{meeting_id}", headers=headers).json()
            if snapshot["recording_state"] == "complete":
                break
            time.sleep(0.02)
        assert snapshot["recording_state"] == "complete"
        traversal = client.get(
            f"/api/v2/meetings/{meeting_id}/files/{quote('../session_state.json', safe='')}" ,
            headers=headers,
        )
        assert traversal.status_code in {403, 404}
        deleted = client.delete(f"/api/v2/meetings/{meeting_id}", headers=headers)
        assert deleted.status_code == 200
        assert client.get(f"/api/v2/meetings/{meeting_id}", headers=headers).status_code == 404


def test_browser_session_cookie_authenticates_api(tmp_path) -> None:
    settings = Settings(host="0.0.0.0", api_token="browser-secret", results_dir=tmp_path / "meetings", translation_model_root=tmp_path / "models")
    app = create_app(settings, ReadyRuntime(), load_models=False, store=LocalMeetingStore(settings.results_dir))
    with TestClient(app) as client:
        assert client.get("/api/v2/health").status_code == 401
        assert client.post("/api/v2/auth/session", json={"token": "wrong"}).status_code == 401
        login = client.post("/api/v2/auth/session", json={"token": "browser-secret"})
        assert login.status_code == 204
        assert "HttpOnly" in login.headers["set-cookie"] and "SameSite=strict" in login.headers["set-cookie"]
        assert client.get("/api/v2/health").status_code == 200
        created = client.post("/api/v2/meetings", json={"title": "浏览器鉴权"})
        assert created.status_code == 201
        assert client.post(f"/api/v2/meetings/{created.json()['id']}/stop").status_code == 202


def test_transcript_endpoint_pages_complete_history(tmp_path) -> None:
    settings = Settings(results_dir=tmp_path / "meetings", translation_model_root=tmp_path / "models")
    app = create_app(settings, ReadyRuntime(), load_models=False, store=LocalMeetingStore(settings.results_dir))
    with TestClient(app) as client:
        created = client.post("/api/v2/meetings", json={"title": "长会议"}).json()
        meeting = app.state.manager.get(created["id"])
        assert meeting is not None
        for index in range(501):
            append_utterance(meeting.transcript_path, Utterance(index + 1, float(index), float(index + 1), 1, "zh", 1.0, f"发言{index}", segment_id=f"{index}:0"))
        first = client.get(f"/api/v2/meetings/{meeting.id}/transcript?offset=0&limit=500").json()
        second = client.get(f"/api/v2/meetings/{meeting.id}/transcript?offset=500&limit=500").json()
        assert first["total"] == 501 and first["has_more"] is True and len(first["items"]) == 500
        assert second["has_more"] is False and [item["text"] for item in second["items"]] == ["发言500"]
        assert client.post(f"/api/v2/meetings/{meeting.id}/stop").status_code == 202

from __future__ import annotations

import time
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from realtime_meeting.config import Settings
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

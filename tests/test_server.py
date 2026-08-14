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


def test_store_ignores_legacy_state_schema(tmp_path) -> None:
    settings = Settings(results_dir=tmp_path / "meetings", translation_model_root=tmp_path / "models")
    store = LocalMeetingStore(settings.results_dir)
    legacy = settings.results_dir / "20260808-003026-legacy"
    legacy.mkdir(parents=True)
    (legacy / "session_state.json").write_text(json.dumps({
        "id": "legacy-meeting",
        "state": "recording",
        "started_at": "2026-08-07T16:30:26+00:00",
    }), encoding="utf-8")

    assert store.list_states() == []


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
        assert health.json()["meeting_start_mode"] == "manual"
        assert "super-secret" not in health.text
        created = client.post("/api/v2/meetings", headers=headers, json={"title": "API 测试"})
        assert created.status_code == 201
        meeting_id = created.json()["id"]
        assert created.json()["recording_state"] == "created"
        assert client.post(f"/api/v2/meetings/{meeting_id}/stop", headers=headers).status_code == 409
        started = client.post(f"/api/v2/meetings/{meeting_id}/start", headers=headers)
        assert started.status_code == 202
        assert started.json()["recording_state"] == "recording"
        ticket = client.post(f"/api/v2/meetings/{meeting_id}/stream-ticket", headers=headers)
        assert ticket.status_code == 201
        with client.websocket_connect(f"/api/v2/meetings/{meeting_id}/stream") as websocket:
            websocket.send_json({"type": "auth", "ticket": ticket.json()["ticket"]})
            assert websocket.receive_json()["type"] == "auth_ok"
            assert websocket.receive_json()["type"] == "snapshot"
            websocket.send_json({"type": "audio_config", "sample_rate": 16000, "channels": 1, "encoding": "pcm_s16le", "packet_ms": 40, "sequence_header": True})
            assert websocket.receive_json()["type"] == "audio_config_ack"
            websocket.send_json({"type": "audio_threshold", "percent": 8.0})
            threshold_ack = websocket.receive_json()
            assert threshold_ack == {"type": "audio_threshold_ack", "percent": 8.0}
            assert app.state.manager.get(meeting_id).volume_threshold_percent == 8.0
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


def test_stream_rejects_binary_audio_before_audio_config(tmp_path) -> None:
    settings = Settings(
        results_dir=tmp_path / "meetings",
        translation_model_root=tmp_path / "models",
    )
    app = create_app(settings, ReadyRuntime(), load_models=False, store=LocalMeetingStore(settings.results_dir))
    with TestClient(app) as client:
        created = client.post("/api/v2/meetings", json={"title": "音频协议测试"})
        meeting_id = created.json()["id"]
        assert client.post(f"/api/v2/meetings/{meeting_id}/start").status_code == 202
        ticket = client.post(f"/api/v2/meetings/{meeting_id}/stream-ticket").json()["ticket"]
        with pytest.raises(WebSocketDisconnect) as disconnect:
            with client.websocket_connect(f"/api/v2/meetings/{meeting_id}/stream") as websocket:
                websocket.send_json({"type": "auth", "ticket": ticket})
                assert websocket.receive_json()["type"] == "auth_ok"
                assert websocket.receive_json()["type"] == "snapshot"
                websocket.send_bytes(b"\x00\x00")
                websocket.receive_json()
        assert disconnect.value.code == 1003
        assert client.post(f"/api/v2/meetings/{meeting_id}/stop").status_code == 202


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
        meeting_id = created.json()["id"]
        assert client.post(f"/api/v2/meetings/{meeting_id}/start").status_code == 202
        assert client.post(f"/api/v2/meetings/{meeting_id}/stop").status_code == 202


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
        assert client.post(f"/api/v2/meetings/{meeting.id}/start").status_code == 202
        assert client.post(f"/api/v2/meetings/{meeting.id}/stop").status_code == 202


def test_asr_settings_are_persisted_clamped_and_locked_after_start(tmp_path) -> None:
    settings = Settings(results_dir=tmp_path / "meetings", translation_model_root=tmp_path / "models")
    app = create_app(settings, ReadyRuntime(), load_models=False, store=LocalMeetingStore(settings.results_dir))
    with TestClient(app) as client:
        created = client.post("/api/v2/meetings", json={"title": "识别设置"})
        assert created.status_code == 201
        meeting_id = created.json()["id"]
        assert created.json()["asr_settings"] == {
            "realtime_beam_size": 5,
            "refine_beam_size": 6,
            "best_of": 5,
            "silence_ms": 700,
            "vad_minimum_speech_ms": 450,
        }

        updated = client.patch(
            f"/api/v2/meetings/{meeting_id}/settings",
            json={
                "asr_settings": {
                    "realtime_beam_size": 99,
                    "refine_beam_size": -1,
                    "best_of": 7,
                    "silence_ms": 9999,
                    "vad_minimum_speech_ms": -20,
                }
            },
        )
        assert updated.status_code == 200
        assert updated.json()["asr_settings"] == {
            "realtime_beam_size": 10,
            "refine_beam_size": 1,
            "best_of": 7,
            "silence_ms": 2000,
            "vad_minimum_speech_ms": 0,
        }

        state_file = app.state.manager.get(meeting_id).output_dir / "session_state.json"
        persisted = json.loads(state_file.read_text(encoding="utf-8"))
        assert persisted["asr_settings"] == updated.json()["asr_settings"]

        assert client.post(f"/api/v2/meetings/{meeting_id}/start").status_code == 202
        locked = client.patch(
            f"/api/v2/meetings/{meeting_id}/settings",
            json={"asr_settings": {"best_of": 1}},
        )
        assert locked.status_code == 409
        assert client.post(f"/api/v2/meetings/{meeting_id}/stop").status_code == 202


def test_full_meeting_settings_are_clamped_and_template_ready(tmp_path) -> None:
    settings = Settings(results_dir=tmp_path / "meetings", translation_model_root=tmp_path / "models")
    app = create_app(settings, ReadyRuntime(), load_models=False, store=LocalMeetingStore(settings.results_dir))
    with TestClient(app) as client:
        created = client.post("/api/v2/meetings", json={"title": "完整设置"})
        meeting_id = created.json()["id"]
        updated = client.patch(
            f"/api/v2/meetings/{meeting_id}/settings",
            json={
                "settings": {
                    "speech_start_ms": 9999,
                    "max_utterance_seconds": 0,
                    "retry_temperature": 0.8,
                    "translation_beam_size": 99,
                    "speaker_cluster_threshold": 0.1,
                    "enable_refinement": False,
                    "keep_audio": False,
                }
            },
        )
        assert updated.status_code == 200
        values = updated.json()["meeting_settings"]
        assert values["speech_start_ms"] == 1000
        assert values["max_utterance_seconds"] == 2
        assert values["retry_temperature"] == 0.8
        assert values["translation_beam_size"] == 8
        assert values["speaker_cluster_threshold"] == 0.4
        assert values["enable_refinement"] is False
        assert values["keep_audio"] is False
        assert updated.json()["asr_settings"]["realtime_beam_size"] == 5

        state_file = app.state.manager.get(meeting_id).output_dir / "session_state.json"
        persisted = json.loads(state_file.read_text(encoding="utf-8"))
        assert persisted["meeting_settings"] == values

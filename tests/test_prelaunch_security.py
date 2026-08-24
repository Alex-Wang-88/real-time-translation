"""上线前专项测试：音频恢复回归、文件下载路径穿越、鉴权强制。

这些用例覆盖 2026-08-24 上线前全检中识别出的测试缺口（对应行动清单 #4）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from realtime_meeting.config import Settings
from realtime_meeting.server import create_app
from realtime_meeting.session import LiveMeetingSession
from realtime_meeting.storage import LocalMeetingStore


class _FakeRuntime:
    """最小可用运行时桩，仅供 create_app / LiveMeetingSession 构造使用。"""

    ready = True
    capabilities_ready = True
    device = "cpu"
    status = "ready"

    def load(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover - 桩
        return None

    def close(self) -> None:  # pragma: no cover - 桩
        return None

    def capability_snapshot(self) -> dict[str, object]:
        return {"asr_primary": {"model": "fake", "ready": True}}

    @property
    def metrics(self) -> dict[str, object]:
        return {}


def test_recover_audio_segments_after_restart(tmp_path: Path) -> None:
    """重启后，已完成会议应能从 audio/ 目录恢复音频片段（此前会 404）。"""
    settings = Settings(results_dir=tmp_path / "meetings", translation_model_root=tmp_path / "models")
    store = LocalMeetingStore(settings.results_dir)
    meeting_id = "rec-meeting"
    meeting_dir = store.meeting_dir(meeting_id)
    (meeting_dir / "audio").mkdir(parents=True)
    (meeting_dir / "audio" / "seg1.wav").write_bytes(b"RIFF")
    payload = {
        "schema_version": "2.0",
        "id": meeting_id,
        "recording_state": "complete",
        "ended_at": "2026-01-01T00:00:00+00:00",
        "files": [],
    }
    (meeting_dir / "session_state.json").write_text(json.dumps(payload), encoding="utf-8")

    session = LiveMeetingSession(
        settings,
        _FakeRuntime(),
        store,
        meeting_id=meeting_id,
        recovered_state=payload,
    )

    assert session.audio_segments, "重启后 audio_segments 应能从磁盘恢复"
    assert "seg1.wav" in {segment["file"] for segment in session.audio_segments}


def test_download_file_blocks_path_traversal(tmp_path: Path) -> None:
    """文件下载端点必须拒绝穿越到会议目录之外的文件。"""
    settings = Settings(results_dir=tmp_path / "meetings", translation_model_root=tmp_path / "models")
    app = create_app(settings, _FakeRuntime(), load_models=False, store=LocalMeetingStore(settings.results_dir))
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET", encoding="utf-8")

    with TestClient(app) as client:
        created = client.post("/api/v2/meetings", json={"title": "遍历测试"})
        assert created.status_code == 201
        meeting_id = created.json()["id"]

        # 尝试用 ../ 越过会议目录读取外部 secret.txt
        response = client.get(f"/api/v2/meetings/{meeting_id}/files/..%2f..%2f..%2f..%2f..%2fsecret.txt")
        assert response.status_code != 200
        assert "TOP-SECRET" not in response.text


def test_api_auth_required_enforces_token_when_exposed() -> None:
    """监听非本地地址且未配置 token 时，API 必须拒绝服务（防止暴露到网络）。"""
    settings = Settings(
        host="0.0.0.0",
        api_token="",
        results_dir=Path("result/meetings"),
        translation_model_root=Path("models"),
    )
    app = create_app(settings, _FakeRuntime(), load_models=False, store=LocalMeetingStore(settings.results_dir))

    with TestClient(app) as client:
        # 未配置 token + 非本地监听 -> 必须 503
        assert client.get("/api/v2/health").status_code == 503

    # 配置 token 后，无凭证应 401，带凭证应放行
    settings.api_token = "s3cr3t"
    app = create_app(settings, _FakeRuntime(), load_models=False, store=LocalMeetingStore(settings.results_dir))
    with TestClient(app) as client:
        assert client.get("/api/v2/health").status_code == 401
        assert client.get("/api/v2/health", headers={"Authorization": "Bearer s3cr3t"}).status_code == 200
        assert client.get("/api/v2/health", headers={"X-Meeting-Token": "s3cr3t"}).status_code == 200

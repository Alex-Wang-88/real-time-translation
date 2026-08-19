from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from realtime_meeting.jimo_upload import JimoUploadClient


@pytest.mark.asyncio
async def test_upload_audio_mirrors_jimo_cos_put_and_registration(settings, tmp_path: Path) -> None:
    audio_path = tmp_path / "会议 录音.wav"
    audio_path.write_bytes(b"RIFF-test-audio")
    settings.jimo_upload_share_id = "share-123"
    settings.jimo_upload_tenant_id = "tenant-1"
    settings.jimo_upload_create_by = "624"
    settings.jimo_upload_cos_authorization = "public-cos-header"
    settings.jimo_upload_max_retries = 1
    settings.jimo_upload_download_ttl_seconds = 600

    now = int(time.time())
    requests: list[httpx.Request] = []
    registered: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/settings/byKeyword":
            assert request.url.params["keyword"] == "resource.cos"
            return httpx.Response(
                200,
                json={"content": json.dumps({"tenantIds": ["tenant-1"]})},
                request=request,
            )
        if request.url.path == "/v1/tencent/cos/record/img":
            assert request.headers["Authorization"] == "public-cos-header"
            assert request.url.params["tenantId"] == "tenant-1"
            assert request.url.params["module"] == "web"
            assert request.url.params["contentType"] == "audio"
            assert request.url.params["fileName"] == "会议录音.wav"
            assert json.loads(request.url.params["content"]) == {"shareId": "share-123"}
            return httpx.Response(
                200,
                json={
                    "access": "tmp-access",
                    "secret": "tmp-secret",
                    "sessionToken": "session-token",
                    "bucketName": "bucket-1250000000",
                    "region": "ap-shanghai",
                    "path": "tenant-1/web/audio/test.wav",
                    "startTime": now - 30,
                    "expiresAt": now + 900,
                },
                request=request,
            )
        if request.method == "PUT":
            assert "q-signature" in request.url.params
            assert request.url.params["q-header-list"] == "host"
            assert request.url.params["x-cos-security-token"] == "session-token"
            assert await request.aread() == b"RIFF-test-audio"
            return httpx.Response(200, request=request)
        if request.method == "POST" and request.url.path == "/v2/upload/file/share":
            registered.update(json.loads((await request.aread()).decode("utf-8")))
            return httpx.Response(200, json={"fileId": "file-1"}, request=request)
        return httpx.Response(404, request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        async with JimoUploadClient(settings, client=http_client) as uploader:
            result = await uploader.upload_audio(audio_path)

    assert [request.method for request in requests] == ["GET", "GET", "PUT", "POST"]
    assert registered["url"] == result.object_url
    assert registered["name"] == "会议 录音.wav"
    assert registered["sizeType"] == "wav"
    assert registered["type"] == "wav"
    assert registered["shareId"] == "share-123"
    assert result.file_id == "file-1"
    assert result.url == result.object_url
    assert result.expires_at is None


def test_signed_cos_url_contains_no_secret(settings) -> None:
    from realtime_meeting.jimo_upload import _CosCredentials, _signed_cos_url

    credentials = _CosCredentials(
        access="access-id",
        secret="do-not-leak-this-secret",
        session_token="session-token",
        bucket_name="bucket-1250000000",
        region="ap-shanghai",
        object_key="tenant/audio.wav",
        start_time=int(time.time()) - 10,
        expires_at=int(time.time()) + 600,
    )
    url, _expiry = _signed_cos_url(
        "https://bucket-1250000000.cos.ap-shanghai.myqcloud.com/tenant/audio.wav",
        credentials,
        method="GET",
        ttl_seconds=300,
    )
    assert "do-not-leak-this-secret" not in url
    assert "q-ak=access-id" in url
    assert "q-signature=" in url

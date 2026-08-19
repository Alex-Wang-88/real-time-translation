"""Upload recorded audio through Jimo's public-share COS flow.

The Jimo web client does not send the local file to ``/v2/upload/file/share``
directly.  It first obtains temporary Tencent COS credentials, uploads the
bytes with a signed PUT request, and then registers the resulting object URL
with Jimo.  This module mirrors that flow without requiring a browser.

The URL returned by :meth:`JimoUploadClient.upload_audio` is the bare COS
object URL registered with Jimo.  Jimo's transcription node can fetch this
URL directly after registration; in live testing, passing the equivalent
short-lived ``q-sign-*`` URL to the node caused its audio tool to return
``FAILED`` even though the URL worked in a normal HTTP client.  Callers that
need a signed URL can opt into it with ``sign_download_url=True``.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import mimetypes
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote, unquote, urlencode, urlsplit, urlunsplit

import httpx


class JimoUploadError(RuntimeError):
    """Raised when a Jimo/COS upload cannot be completed."""


_AUDIO_SUFFIXES = {
    ".aac",
    ".amr",
    ".flac",
    ".m4a",
    ".mp3",
    ".mpeg",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
    ".3gp",
    ".mpeg4",
}
_TRANSIENT_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


class _AsyncFileStream(httpx.AsyncByteStream):
    """Stream a local file without loading a long recording into RAM."""

    def __init__(self, path: Path, *, chunk_size: int = 1024 * 1024) -> None:
        self.path = path
        self.chunk_size = chunk_size

    async def __aiter__(self):
        with self.path.open("rb") as source:
            while True:
                chunk = await asyncio.to_thread(source.read, self.chunk_size)
                if not chunk:
                    break
                yield chunk

    async def aclose(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class JimoUploadResult:
    """The URLs and Jimo file identifier produced by one upload."""

    url: str
    object_url: str
    name: str
    size: int
    object_key: str
    share_id: str
    file_id: str | None = None
    expires_at: int | None = None

    @property
    def download_url(self) -> str:
        return self.url

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "download_url": self.download_url,
            "object_url": self.object_url,
            "name": self.name,
            "size": self.size,
            "object_key": self.object_key,
            "share_id": self.share_id,
            "file_id": self.file_id,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True, slots=True)
class _CosCredentials:
    access: str
    secret: str
    session_token: str
    bucket_name: str
    region: str
    object_key: str
    start_time: int
    expires_at: int


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _parse_json_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _find_value(payload: Any, names: set[str]) -> Any:
    """Find a field in the slightly varying envelopes used by Jimo APIs."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key).casefold() in names and value not in (None, ""):
                return value
        for value in payload.values():
            found = _find_value(value, names)
            if found not in (None, ""):
                return found
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            found = _find_value(value, names)
            if found not in (None, ""):
                return found
    elif isinstance(payload, str):
        parsed = _parse_json_mapping(payload)
        if parsed is not None:
            return _find_value(parsed, names)
    return None


def _epoch_seconds(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        number = int(value)
        return number // 1000 if number > 10**12 else number
    text = str(value).strip()
    try:
        number = int(float(text))
    except ValueError:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return default
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())
    return number // 1000 if number > 10**12 else number


def _extension(path: Path) -> str:
    return path.suffix.casefold().lstrip(".")


def _content_type(path: Path) -> str:
    explicit = {
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".opus": "audio/opus",
        ".wav": "audio/wav",
    }.get(path.suffix.casefold())
    return explicit or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _normalise_file_name(name: str) -> str:
    # This is what the Jimo browser client sends as the COS metadata name.
    return "".join(str(name).split())


def _quote(value: str) -> str:
    return quote(str(value), safe="-_.~")


def _canonical_path(path: str) -> str:
    decoded = unquote(path or "/")
    if not decoded.startswith("/"):
        decoded = "/" + decoded
    return quote(decoded, safe="/-_.~")


def _strip_query(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _object_url(credentials: Mapping[str, Any]) -> tuple[str, str]:
    object_key = str(_find_value(credentials, {"path", "key", "objectkey", "object_key"}) or "").strip()
    if not object_key:
        raise JimoUploadError("积墨 COS 响应缺少对象路径 path")
    object_key = object_key.lstrip("/")

    supplied_url = _find_value(credentials, {"objecturl", "object_url", "location"})
    if isinstance(supplied_url, str) and supplied_url.startswith(("http://", "https://")):
        return _strip_query(supplied_url), object_key

    bucket_name = str(_find_value(credentials, {"bucketname", "bucket", "bucket_name"}) or "").strip()
    region = str(_find_value(credentials, {"region", "regionname", "region_name"}) or "").strip()
    if not bucket_name or not region:
        raise JimoUploadError("积墨 COS 响应缺少 bucketName 或 region")
    if bucket_name.startswith(("http://", "https://")):
        host = urlsplit(bucket_name).netloc
        scheme = urlsplit(bucket_name).scheme
    else:
        host = f"{bucket_name}.cos.{region}.myqcloud.com"
        scheme = "https"
    path = "/" + quote(unquote(object_key), safe="/-_.~")
    return urlunsplit((scheme, host, path, "", "")), object_key


def _signed_cos_url(
    object_url: str,
    credentials: _CosCredentials,
    *,
    method: str,
    ttl_seconds: int,
) -> tuple[str, int]:
    """Create a Tencent COS v5 temporary-credential URL.

    The Jimo browser client uses the same q-sign algorithm through the
    Tencent COS SDK.  Only ``host`` is signed, matching the public client and
    keeping the URL usable by a downstream HTTP client.
    """

    now = int(time.time())
    start = min(credentials.start_time, now)
    requested_expiry = now + max(1, int(ttl_seconds))
    expiry = min(credentials.expires_at - 1, requested_expiry)
    if expiry <= now:
        raise JimoUploadError("积墨 COS 临时凭证已过期，无法生成下载链接")

    parts = urlsplit(object_url)
    path = _canonical_path(parts.path)
    host = parts.netloc.casefold()
    header_list = "host"
    header_string = f"host={_quote(host)}\n"
    http_string = f"{method.casefold()}\n{path}\n\n{header_string}"
    key_time = f"{start};{expiry}"
    sign_key = hmac.new(credentials.secret.encode("utf-8"), key_time.encode("ascii"), hashlib.sha1).hexdigest()
    http_string_sha1 = hashlib.sha1(http_string.encode("utf-8")).hexdigest()
    string_to_sign = f"sha1\n{key_time}\n{http_string_sha1}\n"
    signature = hmac.new(sign_key.encode("ascii"), string_to_sign.encode("utf-8"), hashlib.sha1).hexdigest()

    auth_params = [
        ("q-sign-algorithm", "sha1"),
        ("q-ak", credentials.access),
        ("q-sign-time", key_time),
        ("q-key-time", key_time),
        ("q-header-list", header_list),
        ("q-url-param-list", ""),
        ("q-signature", signature),
    ]
    query = urlencode(auth_params, quote_via=quote)
    if credentials.session_token:
        query += "&" + urlencode(
            [("x-cos-security-token", credentials.session_token)],
            quote_via=quote,
        )
    return urlunsplit((parts.scheme, parts.netloc, path, query, "")), expiry


class JimoUploadClient:
    """Async client for uploading audio to a Jimo public share."""

    def __init__(
        self,
        settings: Any,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
    ) -> None:
        self.settings = settings
        self.base_url = (base_url or getattr(settings, "jimo_upload_base_url", "")).rstrip("/")
        if not self.base_url:
            self.base_url = "https://jimoai-bot-api.xiaohuodui.cn"
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> "JimoUploadClient":
        await self._get_client()
        return self

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        await self.close()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(
                max(1.0, float(getattr(self.settings, "jimo_upload_timeout_seconds", 180.0))),
                connect=max(1.0, float(getattr(self.settings, "jimo_upload_connect_timeout_seconds", 20.0))),
            )
            self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _cos_headers(self) -> dict[str, str]:
        value = str(getattr(self.settings, "jimo_upload_cos_authorization", "") or "").strip()
        return {"Authorization": value} if value else {}

    def _max_attempts(self) -> int:
        return max(1, min(8, int(getattr(self.settings, "jimo_upload_max_retries", 3))))

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        client = await self._get_client()
        url = path if path.startswith(("http://", "https://")) else f"{self.base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(self._max_attempts()):
            try:
                response = await client.request(method, url, params=params, json=json_body, headers=headers)
                if response.status_code in _TRANSIENT_STATUS_CODES and attempt + 1 < self._max_attempts():
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, Mapping):
                    raise JimoUploadError(f"积墨接口返回格式异常：{path}")
                return payload
            except JimoUploadError:
                raise
            except httpx.HTTPStatusError as exc:
                body = exc.response.text[:300].replace("\n", " ")
                last_error = JimoUploadError(
                    f"积墨接口 {path} 返回 HTTP {exc.response.status_code}: {body}"
                )
                if exc.response.status_code not in _TRANSIENT_STATUS_CODES or attempt + 1 >= self._max_attempts():
                    raise last_error from exc
            except (httpx.RequestError, ValueError) as exc:
                last_error = exc
                if attempt + 1 >= self._max_attempts():
                    raise JimoUploadError(f"请求积墨接口 {path} 失败: {exc}") from exc
        raise JimoUploadError(f"请求积墨接口 {path} 失败: {last_error}") from last_error

    async def _put_file(self, signed_url: str, path: Path) -> None:
        client = await self._get_client()
        last_error: Exception | None = None
        content_type = _content_type(path)
        for attempt in range(self._max_attempts()):
            try:
                # Re-open for each retry; an exhausted file object cannot be
                # reused for a second PUT.
                response = await client.put(
                    signed_url,
                    content=_AsyncFileStream(path),
                    headers={
                        "Content-Type": content_type,
                        "Content-Length": str(path.stat().st_size),
                    },
                )
                if response.status_code in _TRANSIENT_STATUS_CODES and attempt + 1 < self._max_attempts():
                    continue
                response.raise_for_status()
                return
            except httpx.HTTPStatusError as exc:
                last_error = JimoUploadError(
                    f"上传到腾讯 COS 失败，HTTP {exc.response.status_code}"
                )
                if exc.response.status_code not in _TRANSIENT_STATUS_CODES or attempt + 1 >= self._max_attempts():
                    raise last_error from exc
            except (OSError, httpx.RequestError) as exc:
                last_error = exc
                if attempt + 1 >= self._max_attempts():
                    raise JimoUploadError(f"上传到腾讯 COS 失败: {exc}") from exc
        raise JimoUploadError(f"上传到腾讯 COS 失败: {last_error}") from last_error

    async def _resolve_share_context(
        self,
        share_id: str,
        tenant_id: str,
        create_by: str,
    ) -> tuple[str, str]:
        if tenant_id and create_by:
            return tenant_id, create_by
        payload = await self._request_json("GET", "/v1/share/info", params={"shareId": share_id})
        if not tenant_id:
            tenant_id = str(_find_value(payload, {"tenantid", "tenant_id"}) or "").strip()
        if not create_by:
            create_by = str(
                _find_value(payload, {"adminid", "admin_id", "createby", "create_by"}) or ""
            ).strip()
        if not tenant_id or not create_by:
            raise JimoUploadError(
                "无法从积墨分享页解析 tenantId/createBy；请配置 JIMO_UPLOAD_TENANT_ID 和 JIMO_UPLOAD_CREATE_BY"
            )
        return tenant_id, create_by

    async def _resource_allows_tenant(self, tenant_id: str) -> bool:
        try:
            payload = await self._request_json(
                "GET",
                "/v1/settings/byKeyword",
                params={"keyword": "resource.cos"},
            )
        except JimoUploadError:
            # The browser falls back to the generic COS endpoint when this
            # optional setting is absent.  Let the credential request decide.
            return True
        content = _find_value(payload, {"content"})
        setting = _parse_json_mapping(content)
        if setting is None:
            return True
        tenant_ids = setting.get("tenantIds")
        if tenant_ids is None:
            return True
        if isinstance(tenant_ids, str):
            tenant_ids = [tenant_ids]
        if not isinstance(tenant_ids, (list, tuple, set)):
            return True
        return "*" in {str(item) for item in tenant_ids} or tenant_id in {str(item) for item in tenant_ids}

    async def _fetch_credentials(
        self,
        *,
        file_name: str,
        tenant_id: str,
        create_by: str,
        share_id: str,
    ) -> _CosCredentials:
        allowed = await self._resource_allows_tenant(tenant_id)
        metadata = {
            "tenantId": tenant_id,
            "module": "web",
            "contentType": "audio",
            "fileName": _normalise_file_name(file_name),
            "createBy": create_by,
            "content": json.dumps({"shareId": share_id}, ensure_ascii=False, separators=(",", ":")),
        }
        endpoint = "/v1/tencent/cos/record/img" if allowed else "/v1/tencent/cos/img"
        params = metadata if allowed else None
        headers = self._cos_headers()
        try:
            payload = await self._request_json("GET", endpoint, params=params, headers=headers)
        except JimoUploadError:
            if endpoint != "/v1/tencent/cos/record/img":
                raise
            # This mirrors the browser's generic fallback when the tenant is
            # not enabled for the recorded-object path.
            payload = await self._request_json(
                "GET",
                "/v1/tencent/cos/img",
                headers=headers,
            )

        now = int(time.time())
        access = str(_find_value(payload, {"access", "tmpsecretid", "secretid", "accesskeyid"}) or "").strip()
        secret = str(_find_value(payload, {"secret", "tmpsecretkey", "secretkey"}) or "").strip()
        bucket_name = str(_find_value(payload, {"bucketname", "bucket", "bucket_name"}) or "").strip()
        region = str(_find_value(payload, {"region", "regionname", "region_name"}) or "").strip()
        object_key = str(_find_value(payload, {"path", "key", "objectkey", "object_key"}) or "").strip().lstrip("/")
        if not access or not secret or not bucket_name or not region or not object_key:
            raise JimoUploadError("积墨 COS 响应缺少临时凭证或对象路径")
        start_time = _epoch_seconds(_find_value(payload, {"starttime", "start_time"}), now - 1)
        expires_at = _epoch_seconds(
            _find_value(payload, {"expiresat", "expires_at", "expiration", "expiretime", "expiredtime"}),
            now + 900,
        )
        if expires_at <= now:
            raise JimoUploadError("积墨 COS 临时凭证已过期")
        session_token = str(
            _find_value(payload, {"sessiontoken", "session_token", "securitytoken", "security_token", "token"})
            or ""
        ).strip()
        return _CosCredentials(
            access=access,
            secret=secret,
            session_token=session_token,
            bucket_name=bucket_name,
            region=region,
            object_key=object_key,
            start_time=start_time,
            expires_at=expires_at,
        )

    async def upload_audio(
        self,
        file_path: str | Path,
        *,
        share_id: str | None = None,
        tenant_id: str | None = None,
        create_by: str | None = None,
        sign_download_url: bool = False,
    ) -> JimoUploadResult:
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise JimoUploadError(f"音频文件不存在: {path}")
        if path.stat().st_size <= 0:
            raise JimoUploadError(f"音频文件为空: {path}")
        if path.suffix.casefold() not in _AUDIO_SUFFIXES:
            raise JimoUploadError(f"不支持的音频格式: {path.suffix or '无扩展名'}")

        resolved_share_id = str(share_id or getattr(self.settings, "jimo_upload_share_id", "") or "").strip()
        if not resolved_share_id:
            raise JimoUploadError("未配置积墨分享 ID，请设置 JIMO_UPLOAD_SHARE_ID")
        resolved_tenant_id = str(tenant_id or getattr(self.settings, "jimo_upload_tenant_id", "") or "").strip()
        resolved_create_by = str(create_by or getattr(self.settings, "jimo_upload_create_by", "") or "").strip()
        resolved_tenant_id, resolved_create_by = await self._resolve_share_context(
            resolved_share_id,
            resolved_tenant_id,
            resolved_create_by,
        )
        credentials = await self._fetch_credentials(
            file_name=path.name,
            tenant_id=resolved_tenant_id,
            create_by=resolved_create_by,
            share_id=resolved_share_id,
        )
        object_url, object_key = _object_url(
            {
                "bucketName": credentials.bucket_name,
                "region": credentials.region,
                "path": credentials.object_key,
            }
        )
        upload_url, _upload_expiry = _signed_cos_url(
            object_url,
            credentials,
            method="PUT",
            ttl_seconds=max(60, credentials.expires_at - int(time.time()) - 5),
        )
        await self._put_file(upload_url, path)

        extension = _extension(path)
        register_payload: dict[str, Any] = {
            "url": object_url,
            "name": path.name,
            "size": path.stat().st_size,
            # Jimo's browser client uses the extension for both fields, not
            # the MIME string returned by File.type.
            "sizeType": extension,
            "type": extension,
            "source": "web",
            "shareId": resolved_share_id,
        }
        registration = await self._request_json(
            "POST",
            "/v2/upload/file/share",
            json_body=register_payload,
        )
        file_id_value = _find_value(registration, {"fileid", "file_id"})
        file_id = str(file_id_value).strip() if file_id_value not in (None, "") else None

        if sign_download_url:
            ttl = max(60, int(getattr(self.settings, "jimo_upload_download_ttl_seconds", 900)))
            download_url, expiry = _signed_cos_url(
                object_url,
                credentials,
                method="GET",
                ttl_seconds=ttl,
            )
        else:
            download_url = object_url
            expiry = None
        return JimoUploadResult(
            url=download_url,
            object_url=object_url,
            name=path.name,
            size=path.stat().st_size,
            object_key=object_key,
            share_id=resolved_share_id,
            file_id=file_id,
            expires_at=expiry,
        )

    async def upload_many(
        self,
        file_paths: list[str | Path] | tuple[str | Path, ...],
        **kwargs: Any,
    ) -> list[JimoUploadResult]:
        results: list[JimoUploadResult] = []
        for file_path in file_paths:
            results.append(await self.upload_audio(file_path, **kwargs))
        return results

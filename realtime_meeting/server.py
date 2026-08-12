from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import secrets
import shutil
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings, load_settings, persist_device_preference
from .runtime import LiveModelRuntime
from .models import SUPPORTED_LANGUAGE_LABELS
from .session import CapacityLimitError, TERMINAL_STATES, SessionManager


class MeetingCreate(BaseModel):
    hotwords: str | None = Field(default=None, max_length=1_000)


class DeviceSwitch(BaseModel):
    device: str  # "auto" | "cpu" | "cuda"


def create_app(
    settings: Settings | None = None,
    runtime: LiveModelRuntime | None = None,
    *,
    load_models: bool = True,
) -> FastAPI:
    config = settings or load_settings()

    def build_runtime(requested_device: str) -> LiveModelRuntime:
        try:
            return LiveModelRuntime(
                config.asr_model,
                config.translation_model,
                requested_device,
                config.refine_asr_model,
                config.refinement_enabled,
                config.asr_fallback_model,
                translation_model_root=config.translation_model_root,
                translation_autodownload=config.translation_autodownload,
                vad_model=config.vad_model,
                gpu_memory_budget_mb=config.gpu_memory_budget_mb,
            )
        except TypeError:
            # Embedded callers may still provide a five-argument runtime
            # factory. Keep that extension seam compatible during rollout.
            return LiveModelRuntime(
                config.asr_model,
                config.translation_model,
                requested_device,
                config.refine_asr_model,
                config.refinement_enabled,
            )

    model_runtime = runtime or build_runtime(config.device)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config.results_dir.mkdir(parents=True, exist_ok=True)
        app.state.load_error = None
        app.state.model_task = None
        if load_models and not app.state.runtime.ready:
            async def load_runtime() -> None:
                current_runtime = app.state.runtime
                try:
                    await asyncio.to_thread(current_runtime.load)
                    app.state.manager.resume_recoverable()
                except Exception as exc:
                    current_runtime.status = "模型加载失败"
                    app.state.load_error = str(exc)
                finally:
                    app.state.model_task = None

            app.state.model_task = asyncio.create_task(load_runtime(), name="load-models")
        elif app.state.runtime.ready:
            app.state.manager.resume_recoverable()
        yield
        load_task = app.state.model_task
        if load_task and not load_task.done():
            load_task.cancel()
            with suppress(asyncio.CancelledError):
                await load_task
        active_sessions = [
            session
            for session in app.state.manager.sessions.values()
            if session.state not in TERMINAL_STATES
        ]
        await asyncio.gather(
            *(session.stop("server_shutdown") for session in active_sessions),
            return_exceptions=True,
        )
        if app.state.manager.recovery_tasks:
            await asyncio.gather(
                *tuple(app.state.manager.recovery_tasks), return_exceptions=True
            )
        await app.state.manager.coordinator.close()
        switch_task = app.state.model_switch_task
        if switch_task and not switch_task.done():
            switch_task.cancel()
            with suppress(asyncio.CancelledError):
                await switch_task

    app = FastAPI(title="本机实时会议转译", version="0.1.0", lifespan=lifespan)
    manager = SessionManager(config, model_runtime)
    app.state.settings = config
    app.state.runtime = model_runtime
    app.state.manager = manager
    # Device switching (set by POST /api/device). Kept on app.state so the
    # health endpoint can report progress without holding a session lock.
    app.state.switching = False
    app.state.switch_error = None
    app.state.model_task = None
    app.state.model_switch_task = None
    app.state.stream_tickets: dict[str, tuple[str, str, float]] = {}
    web_dir = Path(__file__).resolve().parent / "web"
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    def _header_token(headers: Any) -> str:
        authorization = str(headers.get("authorization", ""))
        if authorization.casefold().startswith("bearer "):
            return authorization[7:].strip()
        return str(headers.get("x-meeting-token", "")).strip()

    def _trusted_proxy_source(host: str | None) -> bool:
        try:
            address = ipaddress.ip_address((host or "").strip())
        except ValueError:
            return False
        for cidr in config.trusted_proxy_cidrs:
            try:
                if address in ipaddress.ip_network(cidr, strict=False):
                    return True
            except ValueError:
                continue
        return False

    def require_api_auth(request: Request) -> str:
        if config.trusted_proxy_auth:
            expected_service = config.trusted_proxy_service_token
            provided_service = str(request.headers.get("x-meeting-service-token", ""))
            user_id = str(request.headers.get(config.trusted_proxy_user_header, "")).strip()
            if (
                not _trusted_proxy_source(request.client.host if request.client else None)
                or not expected_service
                or not provided_service
                or not hmac.compare_digest(provided_service, expected_service)
                or not user_id
            ):
                raise HTTPException(status_code=401, detail="需要可信企业网关身份")
            return user_id[:200]
        if not config.api_auth_required:
            return "local"
        expected = config.api_token.strip()
        if not expected:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="非本机监听必须配置 MEETING_API_TOKEN",
            )
        provided = _header_token(request.headers) or request.query_params.get("token", "")
        if not provided or not hmac.compare_digest(provided, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="需要有效的会议服务访问令牌",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return "local"

    def websocket_principal(websocket: WebSocket) -> str | None:
        if config.trusted_proxy_auth:
            expected_service = config.trusted_proxy_service_token
            provided_service = str(websocket.headers.get("x-meeting-service-token", ""))
            user_id = str(
                websocket.headers.get(config.trusted_proxy_user_header, "")
            ).strip()
            if (
                _trusted_proxy_source(websocket.client.host if websocket.client else None)
                and expected_service
                and provided_service
                and hmac.compare_digest(provided_service, expected_service)
                and user_id
            ):
                return user_id[:200]
            return None
        if not config.api_auth_required:
            return "local"
        expected = config.api_token.strip()
        if not expected:
            return None
        provided = _header_token(websocket.headers) or websocket.query_params.get("token", "")
        return "local" if provided and hmac.compare_digest(provided, expected) else None

    def require_owner(meeting: Any, owner_id: str) -> None:
        if meeting.owner_id != owner_id:
            raise HTTPException(status_code=404, detail="会议不存在")

    def _issue_stream_ticket(session_id: str, owner_id: str) -> dict[str, str | float]:
        now = time.time()
        for token, (_sid, _owner, expires) in tuple(app.state.stream_tickets.items()):
            if expires <= now:
                app.state.stream_tickets.pop(token, None)
        token = secrets.token_urlsafe(32)
        expires_at = now + config.stream_ticket_ttl_seconds
        app.state.stream_tickets[token] = (session_id, owner_id, expires_at)
        return {"ticket": token, "expires_at": expires_at}

    def _consume_stream_ticket(
        token: str, session_id: str
    ) -> str | None:
        record = app.state.stream_tickets.pop(token, None)
        if record is None:
            return None
        ticket_session, owner_id, expires_at = record
        if ticket_session != session_id or expires_at <= time.time():
            return None
        return owner_id

    @app.get("/health/live", include_in_schema=False)
    async def liveness() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready", include_in_schema=False)
    async def readiness() -> dict[str, str]:
        if not app.state.runtime.ready or app.state.load_error:
            raise HTTPException(status_code=503, detail="models_not_ready")
        if shutil.disk_usage(config.results_dir).free <= config.disk_stop_bytes:
            raise HTTPException(status_code=503, detail="disk_critical")
        return {"status": "ready"}

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    @app.get("/api/health", dependencies=[Depends(require_api_auth)])
    async def health() -> dict[str, Any]:
        runtime = app.state.runtime
        free = shutil.disk_usage(config.results_dir).free
        active = manager.active()
        if app.state.switching:
            status = "loading"
            message = "正在切换推理设备"
        elif not runtime.ready:
            status = "error" if app.state.load_error else "loading"
            message = app.state.load_error or runtime.status
        elif app.state.switch_error:
            # The previous device keeps working; only the switch failed.
            status = "ready"
            message = f"切换失败：{app.state.switch_error}（继续使用原设备）"
        else:
            status = "ready"
            message = runtime.status
        return {
            "status": status,
            "message": message,
            "device": runtime.device,
            "switching": app.state.switching,
            "switch_error": app.state.switch_error,
            "asr_model": config.asr_model,
            "asr_fallback_model": config.asr_fallback_model,
            "refine_asr_model": config.refine_asr_model,
            "refinement_enabled": config.refinement_enabled,
            "vad_model": config.vad_model,
            "translation_profile": config.translation_profile,
            "translation_model": config.translation_model,
            "translation_target": config.translation_target,
            "runtime_metrics": getattr(runtime, "metrics", {}),
            "gpu_memory_budget_mb": config.gpu_memory_budget_mb,
            "language_labels": SUPPORTED_LANGUAGE_LABELS,
            "jimo_configured": config.jimo_configured,
            "disk_free_bytes": free,
            "active_session": active.snapshot().to_dict() if active else None,
            "capacity": manager.capacity_snapshot(),
        }

    @app.post(
        "/api/device",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_api_auth)],
    )
    async def switch_device(body: DeviceSwitch) -> dict[str, str]:
        """Hot-swap the inference device (auto/cpu/cuda) without restarting.

        Models are loaded once at startup on a fixed device, so changing it
        requires rebuilding the runtime and reloading weights. We do that on a
        background task and only swap the live reference once the new runtime is
        ready, so an in-flight failure leaves the previous device intact.
        """
        requested = (body.device or "").strip().lower()
        if requested not in {"auto", "cpu", "cuda"}:
            raise HTTPException(status_code=400, detail="device 必须是 auto、cpu 或 cuda")
        if manager.active_count():
            raise HTTPException(status_code=409, detail="会议进行中，无法切换设备；请先停止当前会议")
        if app.state.switching:
            raise HTTPException(status_code=409, detail="正在切换设备，请稍候")
        if app.state.model_task and not app.state.model_task.done():
            raise HTTPException(status_code=409, detail="模型正在初次加载，请稍候再切换设备")

        previous_runtime = app.state.runtime
        target = build_runtime(requested)
        app.state.switching = True
        app.state.switch_error = None
        persist_device_preference(config, requested)

        async def _do_switch() -> None:
            swapped = False
            try:
                await asyncio.to_thread(target.load)
                if not target.ready:
                    raise RuntimeError("目标推理设备加载后仍未就绪")
                app.state.runtime = target
                manager.runtime = target
                swapped = True
                if previous_runtime is not target:
                    try:
                        await asyncio.to_thread(previous_runtime.close)
                    except Exception as exc:  # noqa: BLE001 - cleanup is non-fatal
                        app.state.switch_error = f"旧模型释放失败：{exc}"
            except asyncio.CancelledError:
                if not swapped:
                    with suppress(Exception):
                        await asyncio.to_thread(target.close)
                raise
            except Exception as exc:  # noqa: BLE001 - surface any load failure to the UI
                if not swapped:
                    with suppress(Exception):
                        await asyncio.to_thread(target.close)
                app.state.switch_error = str(exc)
            finally:
                app.state.switching = False
                app.state.model_switch_task = None

        app.state.model_switch_task = asyncio.create_task(_do_switch(), name="switch-device")
        return {"status": "switching", "device": requested}

    @app.post(
        "/api/meetings",
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_api_auth)],
    )
    async def create_meeting(
        body: MeetingCreate, owner_id: str = Depends(require_api_auth)
    ) -> dict[str, Any]:
        if app.state.switching:
            raise HTTPException(status_code=409, detail="正在切换推理设备，请稍候再开始会议")
        current_runtime = app.state.runtime
        if not current_runtime.ready:
            raise HTTPException(
                status_code=503,
                detail=app.state.load_error or current_runtime.status,
            )
        try:
            meeting = await manager.create(
                body.hotwords.strip() if body.hotwords else None,
                owner_id=owner_id,
            )
        except CapacityLimitError as exc:
            raise HTTPException(
                status_code=429, detail=str(exc), headers={"Retry-After": "30"}
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return meeting.snapshot().to_dict()

    @app.post(
        "/api/meetings/{session_id}/stream-ticket",
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_api_auth)],
    )
    async def create_stream_ticket(
        session_id: str, owner_id: str = Depends(require_api_auth)
    ) -> dict[str, str | float]:
        meeting = manager.get(session_id)
        if meeting is None:
            raise HTTPException(status_code=404, detail="会议不存在")
        require_owner(meeting, owner_id)
        return _issue_stream_ticket(session_id, owner_id)

    @app.get("/api/metrics", dependencies=[Depends(require_api_auth)])
    async def metrics() -> dict[str, Any]:
        return {
            "capacity": manager.capacity_snapshot(),
            "runtime": getattr(app.state.runtime, "metrics", {}),
            "active_session": manager.active().snapshot().to_dict()
            if manager.active()
            else None,
        }

    @app.get("/api/meetings/{session_id}", dependencies=[Depends(require_api_auth)])
    async def get_meeting(
        session_id: str, owner_id: str = Depends(require_api_auth)
    ) -> dict[str, Any]:
        meeting = manager.get(session_id)
        if meeting is None:
            raise HTTPException(status_code=404, detail="会议不存在")
        require_owner(meeting, owner_id)
        return meeting.snapshot().to_dict()

    @app.post(
        "/api/meetings/{session_id}/stop",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_api_auth)],
    )
    async def stop_meeting(
        session_id: str, owner_id: str = Depends(require_api_auth)
    ) -> dict[str, str]:
        meeting = manager.get(session_id)
        if meeting is None:
            raise HTTPException(status_code=404, detail="会议不存在")
        require_owner(meeting, owner_id)
        await meeting.request_stop("user")
        return {"status": "accepted", "session_id": session_id}

    @app.post(
        "/api/meetings/{session_id}/retry-refinement",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_api_auth)],
    )
    async def retry_refinement(
        session_id: str, owner_id: str = Depends(require_api_auth)
    ) -> dict[str, str]:
        meeting = manager.get(session_id)
        if meeting is None:
            raise HTTPException(status_code=404, detail="会议不存在")
        require_owner(meeting, owner_id)
        if meeting.state != "refinement_error":
            raise HTTPException(status_code=409, detail="当前会议没有可重试的精修任务")
        asyncio.create_task(
            meeting.retry_refinement(), name=f"retry-refinement-{session_id}"
        )
        return {"status": "accepted", "session_id": session_id}

    @app.post(
        "/api/meetings/{session_id}/retry-summary",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_api_auth)],
    )
    async def retry_summary(
        session_id: str, owner_id: str = Depends(require_api_auth)
    ) -> dict[str, str]:
        meeting = manager.get(session_id)
        if meeting is None:
            raise HTTPException(status_code=404, detail="会议不存在")
        require_owner(meeting, owner_id)
        if not meeting.begin_summary():
            raise HTTPException(status_code=409, detail="当前会议不需要生成纪要")
        asyncio.create_task(
            meeting.retry_summary(claimed=True), name=f"retry-summary-{session_id}"
        )
        return {"status": "accepted", "session_id": session_id}

    @app.get(
        "/api/meetings/{session_id}/files/{file_path:path}",
        dependencies=[Depends(require_api_auth)],
    )
    async def download_file(
        session_id: str,
        file_path: str,
        owner_id: str = Depends(require_api_auth),
    ) -> FileResponse:
        meeting = manager.get(session_id)
        if meeting is None:
            raise HTTPException(status_code=404, detail="会议不存在")
        require_owner(meeting, owner_id)
        root = meeting.output_dir.resolve()
        candidate = (root / file_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="不允许访问该文件") from exc
        allowed = file_path in meeting.files
        if file_path.startswith("audio/"):
            allowed = any(
                file_path == f"audio/{item.get('file')}" for item in meeting.audio_segments
            )
        if not allowed or not candidate.is_file():
            raise HTTPException(status_code=404, detail="文件不存在")
        return FileResponse(candidate, filename=candidate.name)

    @app.websocket("/api/meetings/{session_id}/stream")
    async def meeting_stream(websocket: WebSocket, session_id: str) -> None:
        meeting = manager.get(session_id)
        if meeting is None:
            await websocket.close(code=4404, reason="会议不存在")
            return
        await websocket.accept()
        websocket_owner = websocket_principal(websocket)
        ticket_authenticated = False
        if websocket_owner is None:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(), timeout=config.websocket_auth_timeout_seconds
                )
                if message.get("type") != "websocket.receive" or not message.get("text"):
                    await websocket.close(code=4401, reason="需要先完成 WebSocket 认证")
                    return
                payload = json.loads(message["text"])
                if payload.get("type") != "auth":
                    await websocket.close(code=4401, reason="需要先完成 WebSocket 认证")
                    return
                websocket_owner = _consume_stream_ticket(
                    str(payload.get("ticket", "")), session_id
                )
                ticket_authenticated = websocket_owner is not None
            except (asyncio.TimeoutError, json.JSONDecodeError, WebSocketDisconnect):
                await websocket.close(code=4401, reason="WebSocket 认证超时或无效")
                return
        if websocket_owner is None or meeting.owner_id != websocket_owner:
            await websocket.close(code=4403, reason="需要有效的会议服务访问令牌")
            return
        await meeting.add_client(websocket)
        if ticket_authenticated:
            await websocket.send_json(
                {"type": "auth_ok", "ticket_authenticated": ticket_authenticated}
            )
        await websocket.send_json({"type": "snapshot", "meeting": meeting.snapshot().to_dict()})
        framed_audio = False
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                audio = message.get("bytes")
                if audio:
                    sequence: int | None = None
                    if framed_audio:
                        if len(audio) < 4:
                            await websocket.close(code=1003, reason="音频包缺少序号")
                            break
                        sequence = int.from_bytes(audio[:4], "little", signed=False)
                        audio = audio[4:]
                    try:
                        await meeting.feed_audio(audio, sequence=sequence)
                    except ValueError as exc:
                        await websocket.close(code=1003, reason=str(exc)[:120])
                        break
                    continue
                text = message.get("text")
                if text:
                    if len(text) > 4_096:
                        await websocket.close(code=1009, reason="控制消息过大")
                        break
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif payload.get("type") == "auth":
                        # Local and trusted-proxy clients may send the same
                        # first message as ticket-authenticated clients.
                        await websocket.send_json({"type": "auth_ok", "ticket_authenticated": ticket_authenticated})
                    elif payload.get("type") == "audio_config":
                        try:
                            meeting.configure_audio(payload)
                        except ValueError as exc:
                            await websocket.close(code=1003, reason=str(exc)[:120])
                            break
                        framed_audio = bool(payload.get("sequence_header", False))
                        await websocket.send_json(
                            {
                                "type": "audio_config_ack",
                                "sample_rate": meeting.audio_sample_rate,
                                "channels": meeting.audio_channels,
                                "encoding": meeting.audio_encoding,
                                "sequence_header": framed_audio,
                            }
                        )
        except WebSocketDisconnect:
            pass
        finally:
            meeting.remove_client(websocket)
            if not meeting.clients and meeting.state == "recording":
                await meeting.request_stop("websocket_disconnect")

    return app


def create_default_app() -> FastAPI:
    return create_app()

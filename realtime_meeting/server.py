from __future__ import annotations

import asyncio
import hmac
import json
import shutil
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings, load_settings
from .runtime import LiveModelRuntime
from .session import TERMINAL_STATES, SessionManager


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
    model_runtime = runtime or LiveModelRuntime(
        config.asr_model, config.translation_model, config.device
    )

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
                except Exception as exc:
                    current_runtime.status = "模型加载失败"
                    app.state.load_error = str(exc)
                finally:
                    app.state.model_task = None

            app.state.model_task = asyncio.create_task(load_runtime(), name="load-models")
        yield
        load_task = app.state.model_task
        if load_task and not load_task.done():
            load_task.cancel()
            with suppress(asyncio.CancelledError):
                await load_task
        active = app.state.manager.active()
        if active and active.state not in TERMINAL_STATES:
            await active.stop("server_shutdown")
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
    web_dir = Path(__file__).resolve().parent / "web"
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    def _header_token(headers: Any) -> str:
        authorization = str(headers.get("authorization", ""))
        if authorization.casefold().startswith("bearer "):
            return authorization[7:].strip()
        return str(headers.get("x-meeting-token", "")).strip()

    def require_api_auth(request: Request) -> None:
        if not config.api_auth_required:
            return
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

    def websocket_authorized(websocket: WebSocket) -> bool:
        if not config.api_auth_required:
            return True
        expected = config.api_token.strip()
        if not expected:
            return False
        provided = _header_token(websocket.headers) or websocket.query_params.get("token", "")
        return bool(provided) and hmac.compare_digest(provided, expected)

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
            "translation_model": config.translation_model,
            "jimo_configured": config.jimo_configured,
            "disk_free_bytes": free,
            "active_session": active.snapshot().to_dict() if active else None,
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
        active = manager.active()
        if active is not None and (
            active.state not in TERMINAL_STATES
            or (active.worker_task and not active.worker_task.done())
            or (active.stop_task and not active.stop_task.done())
        ):
            raise HTTPException(status_code=409, detail="会议进行中，无法切换设备；请先停止当前会议")
        if app.state.switching:
            raise HTTPException(status_code=409, detail="正在切换设备，请稍候")
        if app.state.model_task and not app.state.model_task.done():
            raise HTTPException(status_code=409, detail="模型正在初次加载，请稍候再切换设备")

        previous_runtime = app.state.runtime
        target = LiveModelRuntime(config.asr_model, config.translation_model, requested)
        app.state.switching = True
        app.state.switch_error = None

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
    async def create_meeting(body: MeetingCreate) -> dict[str, Any]:
        if app.state.switching:
            raise HTTPException(status_code=409, detail="正在切换推理设备，请稍候再开始会议")
        current_runtime = app.state.runtime
        if not current_runtime.ready:
            raise HTTPException(
                status_code=503,
                detail=app.state.load_error or current_runtime.status,
            )
        try:
            meeting = await manager.create(body.hotwords.strip() if body.hotwords else None)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return meeting.snapshot().to_dict()

    @app.get("/api/meetings/{session_id}", dependencies=[Depends(require_api_auth)])
    async def get_meeting(session_id: str) -> dict[str, Any]:
        meeting = manager.get(session_id)
        if meeting is None:
            raise HTTPException(status_code=404, detail="会议不存在")
        return meeting.snapshot().to_dict()

    @app.post(
        "/api/meetings/{session_id}/stop",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_api_auth)],
    )
    async def stop_meeting(session_id: str) -> dict[str, str]:
        meeting = manager.get(session_id)
        if meeting is None:
            raise HTTPException(status_code=404, detail="会议不存在")
        await meeting.request_stop("user")
        return {"status": "accepted", "session_id": session_id}

    @app.post(
        "/api/meetings/{session_id}/retry-summary",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_api_auth)],
    )
    async def retry_summary(session_id: str) -> dict[str, str]:
        meeting = manager.get(session_id)
        if meeting is None:
            raise HTTPException(status_code=404, detail="会议不存在")
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
    async def download_file(session_id: str, file_path: str) -> FileResponse:
        meeting = manager.get(session_id)
        if meeting is None:
            raise HTTPException(status_code=404, detail="会议不存在")
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
        if not websocket_authorized(websocket):
            await websocket.close(code=4403, reason="需要有效的会议服务访问令牌")
            return
        await websocket.accept()
        await meeting.add_client(websocket)
        await websocket.send_json({"type": "snapshot", "meeting": meeting.snapshot().to_dict()})
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                audio = message.get("bytes")
                if audio:
                    try:
                        await meeting.feed_audio(audio)
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
        except WebSocketDisconnect:
            pass
        finally:
            meeting.remove_client(websocket)
            if not meeting.clients and meeting.state == "recording":
                await meeting.request_stop("websocket_disconnect")

    return app


def create_default_app() -> FastAPI:
    return create_app()

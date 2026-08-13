from __future__ import annotations

import asyncio
import hmac
import json
import secrets
import shutil
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings, load_settings
from .models import SUPPORTED_LANGUAGES
from .runtime import LiveModelRuntime
from .session import CapacityLimitError, SessionManager
from .storage import LocalMeetingStore


class MeetingCreate(BaseModel):
    title: str = Field(default="未命名会议", max_length=200)


class MeetingSettingsUpdate(BaseModel):
    asr_settings: dict[str, Any] = Field(default_factory=dict)


class DeviceSwitch(BaseModel):
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda)$")


class BrowserLogin(BaseModel):
    token: str = Field(min_length=1, max_length=4096)


def create_app(
    settings: Settings | None = None,
    runtime: Any | None = None,
    *,
    load_models: bool = True,
    store: LocalMeetingStore | None = None,
) -> FastAPI:
    config = settings or load_settings()
    config.prepare_directories()
    web_dir = Path(__file__).with_name("web")

    def build_runtime(requested: str) -> LiveModelRuntime:
        return LiveModelRuntime(
            config.asr_primary,
            config.asr_fallback,
            config.asr_refine,
            requested,
            asr_autodownload=config.asr_autodownload,
            refinement_enabled=config.enable_refinement,
            asr_realtime_beam_size=config.asr_realtime_beam_size,
            asr_refine_beam_size=config.asr_refine_beam_size,
            asr_best_of=config.asr_best_of,
            asr_retry_temperature=config.asr_retry_temperature,
            asr_log_prob_threshold=config.asr_log_prob_threshold,
            asr_no_speech_threshold=config.asr_no_speech_threshold,
            asr_compression_ratio_threshold=config.asr_compression_ratio_threshold,
            translation_model_root=config.translation_model_root,
            translation_autodownload=config.translation_autodownload,
            vad_model=config.vad_model,
            diarization_required=config.diarization_required,
        )

    active_runtime = runtime or build_runtime(config.device)
    repository = store or LocalMeetingStore(config.results_dir)
    manager = SessionManager(config, active_runtime, repository)

    def purge_browser_sessions() -> None:
        now = time.time()
        for session_id, expires_at in tuple(app.state.browser_sessions.items()):
            if expires_at <= now:
                app.state.browser_sessions.pop(session_id, None)

    def authenticate_request(request: Request) -> str:
        if not config.api_auth_required:
            return "local"
        expected = config.api_token.strip()
        if not expected:
            raise HTTPException(status_code=503, detail="非本机监听必须配置 MEETING_API_TOKEN")
        provided = request.headers.get("authorization", "")
        if provided.casefold().startswith("bearer "):
            provided = provided[7:].strip()
        if not provided:
            provided = request.headers.get("x-meeting-token", "")
        if not provided:
            purge_browser_sessions()
            session_id = request.cookies.get("meeting_session", "")
            if session_id and app.state.browser_sessions.get(session_id, 0) > time.time():
                return "browser"
        if not provided or not hmac.compare_digest(provided, expected):
            raise HTTPException(status_code=401, detail="需要有效的会议服务访问令牌")
        return "local"

    def purge_tickets() -> None:
        now = time.time()
        for token, (_meeting_id, expires_at) in tuple(app.state.stream_tickets.items()):
            if expires_at <= now:
                app.state.stream_tickets.pop(token, None)

    def issue_ticket(meeting_id: str) -> dict[str, str | float]:
        purge_tickets()
        token = secrets.token_urlsafe(32)
        expires_at = time.time() + config.stream_ticket_ttl_seconds
        app.state.stream_tickets[token] = (meeting_id, expires_at)
        return {"ticket": token, "expires_at": expires_at}

    def consume_ticket(token: str, meeting_id: str) -> bool:
        record = app.state.stream_tickets.pop(token, None)
        return bool(record and record[0] == meeting_id and record[1] > time.time())

    def require_meeting(meeting_id: str):
        meeting = manager.get(meeting_id)
        if meeting is None:
            raise HTTPException(status_code=404, detail="会议不存在")
        return meeting

    async def startup() -> None:
        removed = await asyncio.to_thread(repository.purge_expired, config.retention_days)
        for meeting_id in removed:
            manager.sessions.pop(meeting_id, None)
        if load_models and not getattr(app.state.runtime, "ready", False):
            async def load_runtime() -> None:
                try:
                    await asyncio.to_thread(
                        app.state.runtime.load,
                        lambda message: setattr(app.state, "model_message", message),
                    )
                    if not app.state.runtime.ready or not getattr(app.state.runtime, "capabilities_ready", True):
                        app.state.model_error = getattr(app.state.runtime, "status", "模型未就绪")
                    else:
                        await manager.resume_pending(model_tasks_ready=True)
                except Exception as exc:  # noqa: BLE001 - expose readiness failure
                    app.state.model_error = str(exc)
                    app.state.model_message = app.state.model_error
            app.state.model_task = asyncio.create_task(load_runtime(), name="load-v2-models")
            # To-do recovery does not require local models and can resume while
            # the runtime loads. Model-dependent postprocess waits for success.
            await manager.resume_pending(model_tasks_ready=False)
        else:
            await manager.resume_pending(
                model_tasks_ready=bool(
                    getattr(app.state.runtime, "ready", False)
                    and getattr(app.state.runtime, "capabilities_ready", True)
                )
            )

    async def shutdown() -> None:
        manager.begin_shutdown()
        task = app.state.model_task
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        # Finish active recordings before tearing down workers and models. The
        # bounded wait preserves normal shutdown data while still allowing a
        # stuck inference backend to terminate.
        finalizers: list[asyncio.Task[Any]] = []
        for meeting in manager.sessions.values():
            if meeting.active:
                await meeting.request_stop("server_shutdown")
                if meeting.stop_task:
                    finalizers.append(meeting.stop_task)
        if finalizers:
            _done, pending = await asyncio.wait(finalizers, timeout=30)
            for pending_task in pending:
                pending_task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        cancelled: list[asyncio.Task[Any]] = []
        for meeting in manager.sessions.values():
            for task in (
                meeting.worker_task,
                meeting.refinement_worker_task,
                meeting.summary_task,
                meeting.todo_task,
                meeting.postprocess_task,
                meeting.stop_task,
                meeting.disconnect_stop_task,
                *meeting.translation_tasks,
            ):
                if task and not task.done():
                    task.cancel()
                    cancelled.append(task)
        if cancelled:
            await asyncio.gather(*cancelled, return_exceptions=True)
        close = getattr(app.state.runtime, "close", None)
        if close:
            with suppress(Exception):
                await asyncio.to_thread(close)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await startup()
        try:
            yield
        finally:
            await shutdown()

    app = FastAPI(title="实时会议记录 v2", version="0.1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=web_dir), name="static")
    app.state.config = config
    app.state.runtime = active_runtime
    app.state.manager = manager
    app.state.repository = repository
    app.state.model_task = None
    app.state.model_error = None
    app.state.model_message = getattr(active_runtime, "status", "等待加载")
    app.state.stream_tickets: dict[str, tuple[str, float]] = {}
    app.state.browser_sessions: dict[str, float] = {}

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    @app.post("/api/v2/auth/session", status_code=status.HTTP_204_NO_CONTENT)
    async def browser_login(body: BrowserLogin, request: Request, response: Response) -> Response:
        expected = config.api_token.strip()
        if not config.api_auth_required:
            response.status_code = status.HTTP_204_NO_CONTENT
            return response
        if not expected or not hmac.compare_digest(body.token.strip(), expected):
            raise HTTPException(status_code=401, detail="访问令牌无效")
        purge_browser_sessions()
        session_id = secrets.token_urlsafe(32)
        app.state.browser_sessions[session_id] = time.time() + 12 * 60 * 60
        response.set_cookie(
            "meeting_session",
            session_id,
            max_age=12 * 60 * 60,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="strict",
            path="/",
        )
        response.status_code = status.HTTP_204_NO_CONTENT
        return response

    @app.get("/health/live", include_in_schema=False)
    async def liveness() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready", include_in_schema=False)
    async def readiness() -> dict[str, str]:
        if not getattr(app.state.runtime, "ready", False) or not getattr(app.state.runtime, "capabilities_ready", True):
            raise HTTPException(status_code=503, detail="models_not_ready")
        if shutil.disk_usage(config.results_dir).free <= 512 * 1024 * 1024:
            raise HTTPException(status_code=503, detail="disk_critical")
        return {"status": "ready"}

    @app.get("/api/v2/health")
    async def health(_principal: str = Depends(authenticate_request)) -> dict[str, Any]:
        runtime = app.state.runtime
        return {
            "status": (
                "ready"
                if getattr(runtime, "ready", False) and getattr(runtime, "capabilities_ready", True)
                else "error" if app.state.model_error else "loading"
            ),
            "message": app.state.model_error or app.state.model_message or getattr(runtime, "status", ""),
            "device": getattr(runtime, "device", "unknown"),
            "languages": list(SUPPORTED_LANGUAGES),
            "translation_target": "zh-CN",
            "meeting_start_mode": "manual",
            "jimo_configured": config.jimo_configured,
            "todo_configured": config.todo_configured,
            "asr_primary": config.asr_primary,
            "asr_fallback": config.asr_fallback,
            "capabilities": getattr(runtime, "capability_snapshot", lambda: {})(),
            "active_meetings": manager.active_count(),
            "max_active_meetings": config.max_active_meetings,
            "disk_free_bytes": shutil.disk_usage(config.results_dir).free,
            "runtime_metrics": getattr(runtime, "metrics", {}),
        }

    @app.get("/api/v2/metrics")
    async def metrics(_principal: str = Depends(authenticate_request)) -> dict[str, Any]:
        runtime_metrics = dict(getattr(app.state.runtime, "metrics", {}) or {})
        gpu = getattr(getattr(app.state.runtime, "gpu_manager", None), "metrics", {})
        return {
            "active_meetings": manager.active_count(),
            "meeting_count": len(manager.sessions),
            "runtime": runtime_metrics,
            "gpu": gpu,
            "meetings": [
                {
                    "id": meeting.id,
                    "recording_state": meeting.recording_state,
                    "summary_state": meeting.summary_state,
                    "todo_state": meeting.todo_state,
                    "postprocess": meeting.postprocess.to_dict(),
                    "queue_size": meeting.queue.qsize(),
                    "refinement_queue_size": meeting.refinement_queue.qsize(),
                    "postprocess_stage_durations_ms": meeting.postprocess.stage_durations_ms,
                }
                for meeting in manager.sessions.values()
            ],
        }

    @app.get("/api/v2/meetings")
    async def list_meetings(_principal: str = Depends(authenticate_request)) -> list[dict[str, Any]]:
        return manager.list()

    @app.post("/api/v2/meetings", status_code=status.HTTP_201_CREATED)
    async def create_meeting(body: MeetingCreate, _principal: str = Depends(authenticate_request)) -> dict[str, Any]:
        if not getattr(app.state.runtime, "ready", False) or not getattr(app.state.runtime, "capabilities_ready", True):
            raise HTTPException(status_code=503, detail=app.state.model_error or "模型尚未就绪")
        try:
            meeting = await manager.create(body.title.strip() or "未命名会议")
        except CapacityLimitError as exc:
            raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "30"}) from exc
        return meeting.snapshot()

    @app.post("/api/v2/meetings/{meeting_id}/start", status_code=status.HTTP_202_ACCEPTED)
    async def start_meeting(meeting_id: str, _principal: str = Depends(authenticate_request)) -> dict[str, Any]:
        meeting = require_meeting(meeting_id)
        if not getattr(app.state.runtime, "ready", False) or not getattr(app.state.runtime, "capabilities_ready", True):
            raise HTTPException(status_code=503, detail=app.state.model_error or "models_not_ready")
        try:
            meeting = await manager.start(meeting_id)
        except CapacityLimitError as exc:
            raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "30"}) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return meeting.snapshot()

    @app.get("/api/v2/meetings/{meeting_id}")
    async def get_meeting(meeting_id: str, _principal: str = Depends(authenticate_request)) -> dict[str, Any]:
        return require_meeting(meeting_id).snapshot()

    @app.patch("/api/v2/meetings/{meeting_id}/settings")
    async def update_meeting_settings(
        meeting_id: str,
        body: MeetingSettingsUpdate,
        _principal: str = Depends(authenticate_request),
    ) -> dict[str, Any]:
        meeting = require_meeting(meeting_id)
        try:
            meeting.configure_asr_settings(body.asr_settings)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return meeting.snapshot()

    @app.get("/api/v2/meetings/{meeting_id}/transcript")
    async def get_transcript(
        meeting_id: str,
        offset: int = 0,
        limit: int = 500,
        _principal: str = Depends(authenticate_request),
    ) -> dict[str, Any]:
        meeting = require_meeting(meeting_id)
        safe_offset = max(0, offset)
        safe_limit = min(1000, max(1, limit))
        items = meeting.load_transcript()
        return {
            "items": [item.to_dict() for item in items[safe_offset:safe_offset + safe_limit]],
            "offset": safe_offset,
            "limit": safe_limit,
            "total": len(items),
            "has_more": safe_offset + safe_limit < len(items),
        }

    @app.delete("/api/v2/meetings/{meeting_id}")
    async def delete_meeting(meeting_id: str, _principal: str = Depends(authenticate_request)) -> dict[str, Any]:
        try:
            deleted = await manager.delete(meeting_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="会议不存在")
        return {"deleted": True, "meeting_id": meeting_id}

    @app.post("/api/v2/meetings/{meeting_id}/stream-ticket", status_code=status.HTTP_201_CREATED)
    async def stream_ticket(meeting_id: str, _principal: str = Depends(authenticate_request)) -> dict[str, str | float]:
        meeting = require_meeting(meeting_id)
        if meeting.recording_state == "created":
            raise HTTPException(status_code=409, detail="会议尚未开始录音")
        return issue_ticket(meeting_id)

    @app.post("/api/v2/meetings/{meeting_id}/stop", status_code=status.HTTP_202_ACCEPTED)
    async def stop_meeting(meeting_id: str, _principal: str = Depends(authenticate_request)) -> dict[str, str]:
        meeting = require_meeting(meeting_id)
        if meeting.recording_state == "created":
            raise HTTPException(status_code=409, detail="会议尚未开始录音")
        await meeting.request_stop("user")
        return {"status": "accepted", "meeting_id": meeting_id}

    @app.post("/api/v2/meetings/{meeting_id}/summary", status_code=status.HTTP_202_ACCEPTED)
    async def request_summary(meeting_id: str, _principal: str = Depends(authenticate_request)) -> dict[str, str]:
        meeting = require_meeting(meeting_id)
        if not await meeting.request_summary():
            raise HTTPException(status_code=409, detail="当前会议不可生成或重试会议纪要")
        return {"status": "accepted", "meeting_id": meeting_id}

    @app.post("/api/v2/meetings/{meeting_id}/todo", status_code=status.HTTP_202_ACCEPTED)
    async def request_todo(meeting_id: str, _principal: str = Depends(authenticate_request)) -> dict[str, str]:
        meeting = require_meeting(meeting_id)
        if not await meeting.request_todo():
            raise HTTPException(status_code=409, detail="当前会议不可生成或重试 To-do-list")
        return {"status": "accepted", "meeting_id": meeting_id}

    @app.post("/api/v2/meetings/{meeting_id}/postprocess", status_code=status.HTTP_202_ACCEPTED)
    async def request_postprocess(meeting_id: str, _principal: str = Depends(authenticate_request)) -> dict[str, str]:
        meeting = require_meeting(meeting_id)
        if not await meeting.request_postprocess():
            raise HTTPException(status_code=409, detail="当前会议不可重试后台处理")
        return {"status": "accepted", "meeting_id": meeting_id}

    @app.get("/api/v2/meetings/{meeting_id}/files/{file_path:path}")
    async def download_file(meeting_id: str, file_path: str, _principal: str = Depends(authenticate_request)) -> FileResponse:
        meeting = require_meeting(meeting_id)
        root = meeting.output_dir.resolve()
        candidate = (root / file_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="不允许访问该文件") from exc
        audio_files = set()
        for segment in meeting.audio_segments:
            if isinstance(segment, dict) and segment.get("file"):
                audio_name = str(segment["file"]).lstrip("/\\")
                audio_files.add(f"audio/{audio_name}")
        allowed = file_path in meeting.files or file_path in audio_files
        if not allowed or not candidate.is_file():
            raise HTTPException(status_code=404, detail="文件不存在")
        return FileResponse(candidate, filename=candidate.name)

    @app.websocket("/api/v2/meetings/{meeting_id}/stream")
    async def meeting_stream(websocket: WebSocket, meeting_id: str) -> None:
        meeting = manager.get(meeting_id)
        if meeting is None:
            await websocket.close(code=4404, reason="会议不存在")
            return
        await websocket.accept()
        authenticated = False
        joined = False
        try:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=config.websocket_auth_timeout_seconds)
            except (asyncio.TimeoutError, WebSocketDisconnect):
                await websocket.close(code=4401, reason="WebSocket 认证超时")
                return
            if message.get("type") != "websocket.receive" or not message.get("text"):
                await websocket.close(code=4401, reason="需要 ticket 认证")
                return
            try:
                payload = json.loads(message["text"])
            except json.JSONDecodeError:
                payload = {}
            authenticated = payload.get("type") == "auth" and consume_ticket(str(payload.get("ticket", "")), meeting_id)
            if not authenticated:
                await websocket.close(code=4403, reason="ticket 无效或已使用")
                return
            await meeting.add_client(websocket)
            joined = True
            await websocket.send_json({"type": "auth_ok", "ticket_authenticated": authenticated})
            await websocket.send_json({"type": "snapshot", "meeting": meeting.snapshot()})
            framed_audio = False
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                raw_audio = message.get("bytes")
                if raw_audio:
                    sequence = None
                    if framed_audio:
                        if len(raw_audio) < 4:
                            await websocket.close(code=1003, reason="音频包缺少序号")
                            break
                        sequence = int.from_bytes(raw_audio[:4], "little", signed=False)
                        raw_audio = raw_audio[4:]
                    try:
                        await meeting.feed_audio(raw_audio, sequence=sequence)
                    except ValueError as exc:
                        await websocket.close(code=1003, reason=str(exc)[:120])
                        break
                    continue
                text = message.get("text")
                if not text or len(text) > 4096:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                message_type = payload.get("type")
                if message_type == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message_type == "auth":
                    await websocket.send_json({"type": "auth_ok", "ticket_authenticated": authenticated})
                elif message_type == "audio_config":
                    try:
                        meeting.configure_audio(payload)
                    except ValueError as exc:
                        await websocket.close(code=1003, reason=str(exc)[:120])
                        break
                    framed_audio = bool(payload.get("sequence_header", False))
                    await websocket.send_json({"type": "audio_config_ack", "sample_rate": 16000, "channels": 1, "encoding": "pcm_s16le", "sequence_header": framed_audio})
                elif message_type == "audio_threshold":
                    try:
                        meeting.configure_volume_threshold(payload.get("percent"))
                    except ValueError as exc:
                        await websocket.send_json({"type": "audio_threshold_error", "message": str(exc)})
                        continue
                    await websocket.send_json({"type": "audio_threshold_ack", "percent": meeting.volume_threshold_percent})
        except WebSocketDisconnect:
            pass
        finally:
            meeting.remove_client(websocket)
            if joined and not meeting.clients and meeting.recording_state == "recording":
                meeting.schedule_disconnect_stop()

    return app


def create_default_app() -> FastAPI:
    return create_app()


app = create_default_app()

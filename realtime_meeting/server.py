from __future__ import annotations

import asyncio
import hmac
import json
import secrets
import shutil
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .audio import SAMPLE_RATE
from .config import RECOGNITION_ARCHITECTURES, Settings, load_settings
from .models import SUPPORTED_LANGUAGES, SUPPORTED_SPEECH_VARIANTS
from .runtime import LiveModelRuntime
from .session import CapacityLimitError, SessionManager
from .storage import LocalMeetingStore


class MeetingCreate(BaseModel):
    title: str = Field(default="未命名会议", max_length=200)


class MeetingRename(BaseModel):
    title: str = Field(default="未命名会议", max_length=200)


class MeetingSettingsUpdate(BaseModel):
    settings: dict[str, Any] = Field(default_factory=dict)


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
        # Production uses one resident 1.7B checkpoint for ASR, segment-level
        # language confirmation and conflict re-decoding.  Legacy fallback
        # names are still exposed in the API, but they do not load a second
        # model when single_asr_model is enabled.
        return LiveModelRuntime(
            config.asr_primary,
            config.asr_fallback,
            requested,
            language_id_model=config.language_id_model,
            single_model=config.single_asr_model,
            asr_autodownload=config.asr_autodownload,
            translation_model_root=config.translation_model_root,
            translation_autodownload=config.translation_autodownload,
            translation_warmup=config.translation_warmup,
            vad_model=config.vad_model,
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
                except Exception as exc:  # noqa: BLE001
                    app.state.model_error = str(exc)
                    app.state.model_message = app.state.model_error

            app.state.model_task = asyncio.create_task(load_runtime(), name="load-qwen-models")
        else:
            await manager.resume_pending(model_tasks_ready=True)

    async def shutdown() -> None:
        manager.begin_shutdown()
        model_task = app.state.model_task
        if model_task and not model_task.done():
            model_task.cancel()
            with suppress(asyncio.CancelledError):
                await model_task
        finalizers: list[asyncio.Task[Any]] = []
        for meeting in manager.sessions.values():
            if meeting.active:
                await meeting.request_stop("server_shutdown")
                if meeting.stop_task:
                    finalizers.append(meeting.stop_task)
        if finalizers:
            _done, pending = await asyncio.wait(finalizers, timeout=30)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        cancelled: list[asyncio.Task[Any]] = []
        for meeting in manager.sessions.values():
            for task in (
                meeting.worker_task,
                meeting.translation_worker_task,
                meeting.post_translation_task,
                meeting.summary_task,
                meeting.todo_task,
                meeting.stop_task,
                meeting.disconnect_stop_task,
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

    app = FastAPI(title="实时会议记录 v2", version="0.2.0", lifespan=lifespan)
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
            "status": "ready" if getattr(runtime, "ready", False) and getattr(runtime, "capabilities_ready", True) else "error" if app.state.model_error else "loading",
            "message": app.state.model_error or app.state.model_message or getattr(runtime, "status", ""),
            "device": getattr(runtime, "device", "unknown"),
            "languages": list(SUPPORTED_LANGUAGES),
            "speech_variants": list(SUPPORTED_SPEECH_VARIANTS),
            "realtime_asr_models": [
                {"id": "primary", "model": config.asr_primary, "label": "Qwen 1.7B"},
                {"id": "small", "model": config.asr_fallback, "label": "同一 Qwen 1.7B（兼容别名）"},
            ],
            "recognition_architecture": config.recognition_architecture,
            "recognition_architectures": [
                {"id": key, **value} for key, value in RECOGNITION_ARCHITECTURES.items()
            ],
            "recognition_architecture_note": "识别、分段级语言确认和冲突重识别统一使用 Qwen 1.7B；默认不加载第二个 ASR 模型。",
            "translation_target": "zh-CN",
            "meeting_start_mode": "manual",
            "jimo_configured": config.jimo_configured,
            "todo_configured": config.todo_configured,
            "asr_primary": config.asr_primary,
            "asr_fallback": config.asr_fallback,
            "language_id_model": config.language_id_model,
            "single_asr_model": config.single_asr_model,
            "capabilities": getattr(runtime, "capability_snapshot", lambda: {})(),
            "active_meetings": manager.active_count(),
            "max_active_meetings": config.max_active_meetings,
            "disk_free_bytes": shutil.disk_usage(config.results_dir).free,
            "runtime_metrics": getattr(runtime, "metrics", {}),
        }

    @app.get("/api/v2/metrics")
    async def metrics(_principal: str = Depends(authenticate_request)) -> dict[str, Any]:
        meetings = []
        for meeting in manager.sessions.values():
            pipeline = dict(getattr(meeting, "pipeline_metrics", {}) or {})
            pipeline["asr_queue_depth"] = meeting.queue.qsize()
            pipeline["translation_queue_depth"] = meeting.translation_queue.qsize()
            meetings.append(
                {
                    "id": meeting.id,
                    "recording_state": meeting.recording_state,
                    "summary_state": meeting.summary_state,
                    "todo_state": meeting.todo_state,
                    "queue_size": meeting.queue.qsize(),
                    "translation_queue_size": meeting.translation_queue.qsize(),
                    "pipeline": pipeline,
                }
            )
        return {
            "active_meetings": manager.active_count(),
            "meeting_count": len(manager.sessions),
            "runtime": dict(getattr(app.state.runtime, "metrics", {}) or {}),
            "gpu": getattr(getattr(app.state.runtime, "gpu_manager", None), "metrics", {}),
            "meetings": meetings,
        }

    @app.get("/api/v2/meetings")
    async def list_meetings(_principal: str = Depends(authenticate_request)) -> list[dict[str, Any]]:
        return manager.list()

    @app.post("/api/v2/meetings", status_code=status.HTTP_201_CREATED)
    async def create_meeting(body: MeetingCreate, _principal: str = Depends(authenticate_request)) -> dict[str, Any]:
        if load_models and (not getattr(app.state.runtime, "ready", False) or not getattr(app.state.runtime, "capabilities_ready", True)):
            raise HTTPException(status_code=503, detail=app.state.model_error or "模型尚未就绪")
        try:
            meeting = await manager.create(body.title.strip() or "未命名会议")
        except CapacityLimitError as exc:
            raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "30"}) from exc
        return meeting.snapshot()

    @app.post("/api/v2/meetings/{meeting_id}/start", status_code=status.HTTP_202_ACCEPTED)
    async def start_meeting(meeting_id: str, _principal: str = Depends(authenticate_request)) -> dict[str, Any]:
        meeting = require_meeting(meeting_id)
        if load_models and (not getattr(app.state.runtime, "ready", False) or not getattr(app.state.runtime, "capabilities_ready", True)):
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

    @app.patch("/api/v2/meetings/{meeting_id}")
    async def rename_meeting(meeting_id: str, body: MeetingRename, _principal: str = Depends(authenticate_request)) -> dict[str, Any]:
        meeting = require_meeting(meeting_id)
        meeting.rename(body.title)
        snapshot = meeting.snapshot()
        await meeting.broadcast("meeting_renamed", meeting=snapshot)
        return snapshot

    @app.patch("/api/v2/meetings/{meeting_id}/settings")
    async def update_meeting_settings(meeting_id: str, body: MeetingSettingsUpdate, _principal: str = Depends(authenticate_request)) -> dict[str, Any]:
        meeting = require_meeting(meeting_id)
        try:
            meeting.configure_meeting_settings(body.settings)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return meeting.snapshot()

    @app.get("/api/v2/meetings/{meeting_id}/transcript")
    async def get_transcript(meeting_id: str, offset: int = 0, limit: int = 500, _principal: str = Depends(authenticate_request)) -> dict[str, Any]:
        meeting = require_meeting(meeting_id)
        safe_offset = max(0, offset)
        safe_limit = min(1000, max(1, limit))
        paragraphs = meeting.load_transcript()
        page = paragraphs[safe_offset : safe_offset + safe_limit]
        return {
            "schema_version": "2.0",
            "paragraphs": [item.to_dict() for item in page],
            "offset": safe_offset,
            "limit": safe_limit,
            "total": len(paragraphs),
            "has_more": safe_offset + safe_limit < len(paragraphs),
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
            raise HTTPException(status_code=409, detail="当前会议不可生成或重试会议纪要；请先等待翻译队列完成")
        return {"status": "accepted", "meeting_id": meeting_id}

    @app.post("/api/v2/meetings/{meeting_id}/todo", status_code=status.HTTP_202_ACCEPTED)
    async def request_todo(meeting_id: str, _principal: str = Depends(authenticate_request)) -> dict[str, str]:
        meeting = require_meeting(meeting_id)
        if not await meeting.request_todo():
            raise HTTPException(status_code=409, detail="当前会议不可生成或重试 To-do-list")
        return {"status": "accepted", "meeting_id": meeting_id}

    @app.post("/api/v2/meetings/{meeting_id}/translation/retry", status_code=status.HTTP_202_ACCEPTED)
    async def retry_translation(meeting_id: str, _principal: str = Depends(authenticate_request)) -> dict[str, str]:
        meeting = require_meeting(meeting_id)
        if not await meeting.retry_translation():
            raise HTTPException(status_code=409, detail="当前会议没有可重试的英文或德文段落")
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
        audio_files: set[str] = set()
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
        connection_id = uuid.uuid4().hex
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
                auth_payload = json.loads(message["text"])
            except json.JSONDecodeError:
                auth_payload = {}
            if auth_payload.get("type") != "auth" or not consume_ticket(str(auth_payload.get("ticket", "")), meeting_id):
                await websocket.close(code=4403, reason="ticket 无效或已使用")
                return
            await meeting.add_client(websocket)
            await websocket.send_json({"type": "auth_ok", "ticket_authenticated": True})
            await websocket.send_json({"type": "snapshot", "meeting": meeting.snapshot()})
            framed_audio = False
            audio_configured = False
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                raw_audio = message.get("bytes")
                if raw_audio:
                    if not audio_configured:
                        await websocket.close(code=1003, reason="audio_config required before binary audio")
                        break
                    sequence = None
                    if framed_audio:
                        if len(raw_audio) < 4:
                            await websocket.close(code=1003, reason="音频包缺少序号")
                            break
                        sequence = int.from_bytes(raw_audio[:4], "little", signed=False)
                        raw_audio = raw_audio[4:]
                    try:
                        await meeting.feed_audio(raw_audio, sequence=sequence, source_id=connection_id)
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
                elif message_type == "audio_config":
                    try:
                        meeting.configure_audio(payload, source_id=connection_id)
                    except ValueError as exc:
                        await websocket.close(code=1003, reason=str(exc)[:120])
                        break
                    framed_audio = bool(payload.get("sequence_header", False))
                    audio_configured = True
                    await websocket.send_json({"type": "audio_config_ack", "sample_rate": SAMPLE_RATE, "channels": 1, "encoding": "pcm_s16le", "sequence_header": framed_audio})
                elif message_type == "audio_flush":
                    drained = await meeting.wait_for_audio_drain(meeting.settings.audio_drain_timeout_seconds)
                    await websocket.send_json({
                        "type": "audio_flush_ack",
                        "request_id": payload.get("request_id"),
                        "drained": drained,
                        "audio_samples_received": meeting.audio_samples_received,
                    })
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
            meeting.release_audio_source(connection_id)
            meeting.remove_client(websocket)
            if not meeting.clients and meeting.recording_state in {"starting", "recording"}:
                meeting.schedule_disconnect_stop()

    return app


def create_default_app() -> FastAPI:
    return create_app()


app = create_default_app()

from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
from collections import deque
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import WebSocket

from .audio import RotatingAudioWriter, SegmentEvent, StreamSegmenter
from .config import Settings
from .exporter import append_utterance, export_live_result, load_utterances
from .jimo import MeetingSummarizer
from .models import MeetingSnapshot, SessionState, Utterance, utc_now_iso
from .runtime import LiveModelRuntime, is_boundary_duplicate


TERMINAL_STATES: set[SessionState] = {"complete", "summary_pending", "summary_error", "error"}


class LiveMeetingSession:
    def __init__(
        self,
        settings: Settings,
        runtime: LiveModelRuntime,
        *,
        session_id: str | None = None,
        output_dir: Path | None = None,
        started_at: str | None = None,
        recovered: bool = False,
        hotwords: str | None = None,
    ) -> None:
        self.settings = settings
        self.runtime = runtime
        self.id = session_id or str(uuid.uuid4())
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir = output_dir or settings.results_dir / f"{stamp}-{self.id[:8]}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = started_at or utc_now_iso()
        self.ended_at: str | None = None
        self.started_monotonic = time.monotonic()
        self.state: SessionState = "error" if recovered else "starting"
        self.error: str | None = "服务重启后已恢复现有转写；可下载记录或开始新会议" if recovered else None
        self.processing_error: str | None = None
        self.hotwords = hotwords
        self.current_language: str | None = None
        self.summary = ""
        self.files: list[str] = []
        self.recent: deque[Utterance] = deque(maxlen=500)
        self.utterance_count = 0
        self.transcript_path = self.output_dir / "transcript.jsonl"
        self.audio_writer: RotatingAudioWriter | None = None
        self.segmenter: StreamSegmenter | None = None
        self.audio_segments: list[dict[str, object]] = []
        self.audio_bytes_received = 0
        self.audio_packets_received = 0
        self.audio_samples_received = 0
        self.audio_level = 0.0
        self._last_audio_event = 0.0
        self.clients: set[WebSocket] = set()
        self.queue: asyncio.Queue[SegmentEvent | None] = asyncio.Queue(
            maxsize=settings.inference_queue_size
        )
        self.worker_task: asyncio.Task[None] | None = None
        self.disk_task: asyncio.Task[None] | None = None
        self.stop_task: asyncio.Task[None] | None = None
        self.stop_lock = asyncio.Lock()
        self.recent_text = ""
        self.previous_language: str | None = None
        self.last_final_revision = 0
        if recovered:
            recovered_items = load_utterances(self.transcript_path)
            self.utterance_count = len(recovered_items)
            self.recent.extend(recovered_items[-500:])
            if recovered_items:
                self.current_language = recovered_items[-1].language
                self.previous_language = recovered_items[-1].language
                self.recent_text = " ".join(item.text for item in recovered_items[-20:])[-1000:]
        if not recovered:
            self._write_state()

    @property
    def elapsed_seconds(self) -> float:
        if self.audio_writer is not None:
            return self.audio_writer.total_samples / 16_000
        recent_end = self.recent[-1].end if self.recent else 0.0
        audio_end = 0.0
        for item in self.audio_segments:
            try:
                audio_end = max(audio_end, float(item.get("end_seconds", 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
        if recent_end or audio_end:
            return max(recent_end, audio_end)
        return max(0.0, time.monotonic() - self.started_monotonic) if self.state not in TERMINAL_STATES else 0.0

    def _write_state(self) -> None:
        payload = {
            "id": self.id,
            "state": self.state,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "error": self.error,
            "processing_error": self.processing_error,
            "audio_bytes_received": self.audio_bytes_received,
            "audio_packets_received": self.audio_packets_received,
            "audio_samples_received": self.audio_samples_received,
        }
        state_path = self.output_dir / "session_state.json"
        temporary_path = state_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(state_path)

    async def start(self) -> None:
        if not self.runtime.ready:
            raise RuntimeError("实时模型尚未就绪")
        self.audio_writer = RotatingAudioWriter(
            self.output_dir / "audio", self.settings.audio_segment_minutes
        )
        self.segmenter = StreamSegmenter(
            max_utterance_ms=int(self.settings.max_utterance_seconds * 1000)
        )
        self.state = "recording"
        self._write_state()
        self.worker_task = asyncio.create_task(self._inference_worker(), name=f"inference-{self.id}")
        self.disk_task = asyncio.create_task(self._disk_monitor(), name=f"disk-{self.id}")

    async def add_client(self, websocket: WebSocket) -> None:
        self.clients.add(websocket)

    def remove_client(self, websocket: WebSocket) -> None:
        self.clients.discard(websocket)

    async def broadcast(self, event_type: str, **payload: Any) -> None:
        event = {"type": event_type, **payload}
        stale: list[WebSocket] = []
        for client in tuple(self.clients):
            try:
                await client.send_json(event)
            except Exception:
                stale.append(client)
        for client in stale:
            self.clients.discard(client)

    async def status(self, stage: str, message: str, **extra: Any) -> None:
        await self.broadcast("status", stage=stage, message=message, state=self.state, **extra)

    async def _record_processing_error(
        self, code: str, message: str, *, retryable: bool = False
    ) -> None:
        if self.processing_error is None:
            self.processing_error = message
        if self.error is None:
            self.error = message
        self._write_state()
        await self.broadcast(
            "error",
            code=code,
            message=message,
            retryable=retryable,
            state=self.state,
        )

    async def _enqueue_event(self, event: SegmentEvent) -> None:
        try:
            if event.kind == "partial":
                if self.queue.full():
                    return
                self.queue.put_nowait(event)
            else:
                self.queue.put_nowait(event)
        except asyncio.QueueFull:
            message = "实时推理速度落后，当前语音片段未能全部排队；会议将保存已有记录"
            await self._record_processing_error("inference_backlog", message)
            await self.request_stop("inference_backlog")

    async def feed_audio(self, pcm: bytes) -> None:
        if self.state != "recording" or not pcm:
            return
        assert self.audio_writer is not None and self.segmenter is not None
        if len(pcm) > self.settings.max_audio_packet_bytes:
            raise ValueError(
                f"音频包过大，单包不能超过 {self.settings.max_audio_packet_bytes} 字节"
            )
        if len(pcm) % 2:
            raise ValueError("音频包必须是偶数长度的 PCM16 数据")
        self.audio_bytes_received += len(pcm)
        self.audio_packets_received += 1
        self.audio_samples_received += len(pcm) // 2
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        self.audio_level = float(np.sqrt(np.mean(samples * samples)) / 32768.0) if len(samples) else 0.0
        self.audio_writer.write(pcm)
        for event in self.segmenter.feed(pcm):
            await self._enqueue_event(event)
            if self.stop_task and not self.stop_task.done():
                break
        now = time.monotonic()
        if now - self._last_audio_event >= 0.8:
            self._last_audio_event = now
            await self.broadcast(
                "audio_input",
                bytes_received=self.audio_bytes_received,
                packets_received=self.audio_packets_received,
                samples_received=self.audio_samples_received,
                level=round(self.audio_level, 5),
                vad_active=self.segmenter.active,
            )

    async def _inference_worker(self) -> None:
        while True:
            event = await self.queue.get()
            try:
                if event is None:
                    return
                if event.kind == "partial":
                    await self.status("transcribing", "正在生成临时字幕")
                    result = await asyncio.to_thread(
                        self.runtime.transcribe_partial,
                        event,
                        self.recent_text,
                        self.hotwords,
                    )
                    if result and result.revision > self.last_final_revision and self.state == "recording":
                        await self.broadcast(
                            "partial",
                            revision=result.revision,
                            start=result.start,
                            end=result.end,
                            text=result.text,
                            language=result.language,
                        )
                    continue
                await self.status("transcribing", "正在转写稳定片段")
                items = await asyncio.to_thread(
                    self.runtime.transcribe_final,
                    event,
                    next_id=self.utterance_count + 1,
                    previous_language=self.previous_language,
                    recent_text=self.recent_text,
                    hotwords=self.hotwords,
                )
                self.last_final_revision = max(self.last_final_revision, event.revision)
                await self.broadcast("partial_clear", revision=event.revision)
                for item in items:
                    previous = self.recent[-1] if self.recent else None
                    if (
                        previous
                        and item.start <= previous.end + 0.6
                        and is_boundary_duplicate(previous.text, item.text)
                    ):
                        continue
                    item.id = self.utterance_count + 1
                    append_utterance(self.transcript_path, item)
                    self.utterance_count += 1
                    self.recent.append(item)
                    self.current_language = item.language
                    self.previous_language = item.language
                    self.recent_text = (self.recent_text + " " + item.text)[-1000:]
                    await self.status("translating", "已识别语言，正在生成中文翻译")
                    await self.broadcast("utterance", utterance=item.to_dict())
                if self.state == "recording":
                    await self.status("listening", "正在监听语音")
            except Exception as exc:
                await self._record_processing_error(
                    "inference_failed", f"语音片段处理失败：{exc}"
                )
            finally:
                self.queue.task_done()

    async def request_stop(self, reason: str = "user") -> None:
        if self.stop_task and not self.stop_task.done():
            return
        self.stop_task = asyncio.create_task(self.stop(reason), name=f"stop-{self.id}")

    async def stop(self, reason: str = "user") -> None:
        async with self.stop_lock:
            if self.state not in {"recording", "starting", "error"}:
                return
            self.state = "finalizing"
            self._write_state()
            try:
                await self.status("finalizing", "正在处理最后一段语音", reason=reason)
                disk_task = self.disk_task
                self.disk_task = None
                if disk_task:
                    disk_task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await disk_task
                worker_task = self.worker_task
                if worker_task and not worker_task.done():
                    if self.segmenter:
                        for event in self.segmenter.flush():
                            await self.queue.put(event)
                    await self.queue.join()
                    await self.queue.put(None)
                    await worker_task
                elif worker_task:
                    # A failed/cancelled worker cannot consume a stale sentinel
                    # on a later stop retry. Preserve completed transcript data
                    # and reset the queue bookkeeping before continuing export.
                    self.queue = asyncio.Queue(maxsize=self.settings.inference_queue_size)
                self.worker_task = None
                await self.status("saving", "正在保存录音和完整逐句稿")
                if self.audio_writer:
                    self.audio_segments = await asyncio.to_thread(self.audio_writer.close)
                    self.audio_writer = None
                self.ended_at = utc_now_iso()
                utterances = load_utterances(self.transcript_path)
                self.files = export_live_result(
                    self.output_dir,
                    session_id=self.id,
                    started_at=self.started_at,
                    ended_at=self.ended_at,
                    duration_seconds=self.elapsed_seconds,
                    utterances=utterances,
                    audio_segments=self.audio_segments,
                    status="summary_pending",
                    processing_error=self.processing_error,
                )
                if not utterances:
                    self.summary = "本次会议未检测到有效发言，无法生成会议纪要。"
                    (self.output_dir / "meeting_minutes.md").write_text(
                        self.summary + "\n", encoding="utf-8"
                    )
                    self.state = "complete"
                    self.files.append("meeting_minutes.md")
                    self._finish_export(utterances)
                    await self.broadcast(
                        "summary_complete", content=self.summary, files=self.files
                    )
                    return
                # Stopping a meeting only finalizes and exports the recording. AI
                # summarization is deliberately a separate user action so a user
                # can review the transcript before sending it to the configured
                # service.
                self.state = "summary_pending"
                self.summary = ""
                self._write_state()
                await self.status("summary_pending", "会议已保存，可手动生成会议纪要")
                await self.broadcast(
                    "summary_pending",
                    session_id=self.id,
                    files=self.files,
                    utterance_count=len(utterances),
                    error=self.processing_error,
                )
            except Exception as exc:
                self.state = "error"
                self.error = f"会议保存失败：{exc}"
                try:
                    self._write_state()
                finally:
                    await self.broadcast(
                        "error",
                        code="stop_failed",
                        message=self.error,
                        retryable=False,
                        state=self.state,
                    )

    def begin_summary(self) -> bool:
        """Atomically claim a pending summary before spawning its task."""

        if self.state not in {"summary_pending", "summary_error"}:
            return False
        self.state = "summarizing"
        self.summary = ""
        self.error = None
        self._write_state()
        return True

    async def _run_summary(self, utterances: list[Utterance] | None = None) -> None:
        await self.status("summarizing", "正在准备积墨 AI 分块")
        try:
            summarizer = MeetingSummarizer(self.settings)

            async def summary_status(kind: str, index: int, total: int) -> None:
                if kind == "chunk":
                    await self.status(
                        "summarizing_chunks",
                        f"积墨 AI 正在分析第 {index}/{total} 块",
                        current=index,
                        total=total,
                    )
                else:
                    await self.status("summarizing_final", "正在生成最终会议纪要")

            async def summary_delta(content: str) -> None:
                self.summary += content
                await self.broadcast("summary_delta", content=content)

            async def summary_reset() -> None:
                self.summary = ""
                await self.broadcast("summary_reset")

            self.summary = await summarizer.summarize(
                self.transcript_path,
                self.id,
                self.started_at,
                self.ended_at or utc_now_iso(),
                on_status=summary_status,
                on_delta=summary_delta,
                on_reset=summary_reset,
            )
            (self.output_dir / "meeting_minutes.md").write_text(
                self.summary.rstrip() + "\n", encoding="utf-8"
            )
            self.state = "complete"
            if "meeting_minutes.md" not in self.files:
                self.files.append("meeting_minutes.md")
            final_items = utterances or load_utterances(self.transcript_path)
            self._finish_export(final_items)
            await self.status("complete", "会议纪要已生成")
            await self.broadcast("summary_complete", content=self.summary, files=self.files)
        except Exception as exc:
            self.state = "summary_error"
            self.error = str(exc)
            items = utterances or load_utterances(self.transcript_path)
            self._finish_export(items, summary_error=self.error)
            await self.broadcast(
                "error",
                code="summary_failed",
                message=self.error,
                retryable=True,
            )

    def _finish_export(self, utterances: list[Utterance], summary_error: str | None = None) -> None:
        self.files = export_live_result(
            self.output_dir,
            session_id=self.id,
            started_at=self.started_at,
            ended_at=self.ended_at or utc_now_iso(),
            duration_seconds=self.elapsed_seconds,
            utterances=utterances,
            audio_segments=self.audio_segments,
            status=self.state,
            summary_error=summary_error,
            processing_error=self.processing_error,
        )
        if (self.output_dir / "meeting_minutes.md").exists():
            self.files.append("meeting_minutes.md")
        self._write_state()

    async def retry_summary(self, *, claimed: bool = False) -> None:
        if not claimed and not self.begin_summary():
            raise ValueError("当前会议不需要生成纪要")
        await self._run_summary()

    async def _disk_monitor(self) -> None:
        try:
            while self.state == "recording":
                free = shutil.disk_usage(self.output_dir).free
                if self.settings.disk_warn_bytes and free < self.settings.disk_warn_bytes:
                    await self.broadcast(
                        "warning",
                        code="disk_low",
                        message="可用磁盘空间不足 2GB，请尽快停止会议",
                        free_bytes=free,
                    )
                if self.settings.disk_stop_bytes and free < self.settings.disk_stop_bytes:
                    await self.broadcast(
                        "error",
                        code="disk_critical",
                        message="磁盘空间不足，系统将自动停止并保存会议",
                        retryable=False,
                    )
                    await self.request_stop("disk_low")
                    return
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            return

    def snapshot(self) -> MeetingSnapshot:
        return MeetingSnapshot(
            id=self.id,
            state=self.state,
            started_at=self.started_at,
            elapsed_seconds=round(self.elapsed_seconds, 3),
            current_language=self.current_language,
            utterance_count=self.utterance_count,
            recent_utterances=[item.to_dict() for item in self.recent],
            summary=self.summary,
            error=self.error or self.processing_error,
            files=self.files,
            audio_bytes_received=self.audio_bytes_received,
            audio_packets_received=self.audio_packets_received,
            audio_samples_received=self.audio_samples_received,
            audio_level=round(self.audio_level, 5),
        )


class SessionManager:
    def __init__(self, settings: Settings, runtime: LiveModelRuntime) -> None:
        self.settings = settings
        self.runtime = runtime
        self.sessions: dict[str, LiveMeetingSession] = {}
        self.active_id: str | None = None
        self.lock = asyncio.Lock()
        self._recover_existing()

    def _recover_existing(self) -> None:
        root = self.settings.results_dir
        if not root.exists():
            return
        state_files = sorted(
            root.glob("*/session_state.json"), key=lambda path: path.stat().st_mtime
        )[-10:]
        safe_files = {
            "meeting_transcript.md",
            "translated_zh.md",
            "transcript.json",
            "transcript.jsonl",
            "audio_manifest.json",
            "manifest.json",
            "meeting_minutes.md",
        }
        for state_file in state_files:
            try:
                payload = json.loads(state_file.read_text(encoding="utf-8"))
                session = LiveMeetingSession(
                    self.settings,
                    self.runtime,
                    session_id=payload["id"],
                    output_dir=state_file.parent,
                    started_at=payload.get("started_at"),
                    recovered=True,
                )
                session.ended_at = payload.get("ended_at")
                session.processing_error = payload.get("processing_error")
                for attribute in (
                    "audio_bytes_received",
                    "audio_packets_received",
                    "audio_samples_received",
                ):
                    try:
                        setattr(session, attribute, max(0, int(payload.get(attribute, 0) or 0)))
                    except (TypeError, ValueError):
                        pass
                recovered_state = payload.get("state")
                if recovered_state in {"complete", "summary_pending", "summary_error"}:
                    session.state = recovered_state
                    session.error = payload.get("error")
                elif recovered_state in {"recording", "starting", "finalizing", "summarizing"}:
                    session.state = "error"
                    session.error = payload.get("error") or session.error
                else:
                    session.error = payload.get("error") or session.error
                minutes = session.output_dir / "meeting_minutes.md"
                if minutes.exists():
                    session.summary = minutes.read_text(encoding="utf-8")
                audio_manifest = session.output_dir / "audio_manifest.json"
                if audio_manifest.exists():
                    try:
                        manifest_payload = json.loads(audio_manifest.read_text(encoding="utf-8"))
                        segments = (
                            manifest_payload.get("segments", [])
                            if isinstance(manifest_payload, dict)
                            else []
                        )
                    except (OSError, TypeError, ValueError, json.JSONDecodeError):
                        segments = []
                    if isinstance(segments, list):
                        recovered_segments: list[dict[str, object]] = []
                        for segment in segments:
                            if not isinstance(segment, dict) or not segment.get("file"):
                                continue
                            try:
                                normalized = dict(segment)
                                normalized["file"] = Path(str(normalized["file"])).name
                                normalized["start_seconds"] = max(
                                    0.0, float(normalized.get("start_seconds", 0.0) or 0.0)
                                )
                                normalized["end_seconds"] = max(
                                    0.0, float(normalized.get("end_seconds", 0.0) or 0.0)
                                )
                                normalized["samples"] = max(0, int(normalized.get("samples", 0) or 0))
                            except (TypeError, ValueError):
                                continue
                            recovered_segments.append(normalized)
                        session.audio_segments = recovered_segments
                        if not session.audio_samples_received:
                            session.audio_samples_received = sum(
                                int(segment.get("samples", 0) or 0)
                                for segment in recovered_segments
                            )
                        if not session.audio_bytes_received:
                            session.audio_bytes_received = session.audio_samples_received * 2
                session.files = [
                    path.name
                    for path in session.output_dir.iterdir()
                    if path.is_file()
                    and (
                        path.name in safe_files
                        or (path.name.startswith("original_") and path.suffix == ".md")
                    )
                ]
                self.sessions[session.id] = session
                self.active_id = session.id
            except Exception:
                continue

    async def create(self, hotwords: str | None = None) -> LiveMeetingSession:
        async with self.lock:
            if self.active_id:
                active = self.sessions.get(self.active_id)
                if active and (
                    active.state not in TERMINAL_STATES
                    or (active.worker_task and not active.worker_task.done())
                    or (active.stop_task and not active.stop_task.done())
                ):
                    raise RuntimeError("当前已有一场会议正在进行")
            session = LiveMeetingSession(self.settings, self.runtime, hotwords=hotwords)
            await session.start()
            self.sessions[session.id] = session
            self.active_id = session.id
            return session

    def get(self, session_id: str) -> LiveMeetingSession | None:
        return self.sessions.get(session_id)

    def active(self) -> LiveMeetingSession | None:
        return self.sessions.get(self.active_id) if self.active_id else None

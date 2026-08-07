from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
from collections import deque
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
        self.queue: asyncio.Queue[SegmentEvent | None] = asyncio.Queue()
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
        if self.recent:
            return self.recent[-1].end
        return max(0.0, time.monotonic() - self.started_monotonic) if self.state not in TERMINAL_STATES else 0.0

    def _write_state(self) -> None:
        payload = {
            "id": self.id,
            "state": self.state,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "error": self.error,
        }
        (self.output_dir / "session_state.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

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

    async def feed_audio(self, pcm: bytes) -> None:
        if self.state != "recording" or not pcm:
            return
        assert self.audio_writer is not None and self.segmenter is not None
        self.audio_bytes_received += len(pcm)
        self.audio_packets_received += 1
        self.audio_samples_received += len(pcm) // 2
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        self.audio_level = float(np.sqrt(np.mean(samples * samples)) / 32768.0) if len(samples) else 0.0
        self.audio_writer.write(pcm)
        for event in self.segmenter.feed(pcm):
            if event.kind == "partial":
                if self.queue.empty():
                    await self.queue.put(event)
            else:
                await self.queue.put(event)
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
                self.error = str(exc)
                await self.broadcast("error", code="inference_failed", message=str(exc), retryable=False)
            finally:
                self.queue.task_done()

    async def request_stop(self, reason: str = "user") -> None:
        if self.stop_task and not self.stop_task.done():
            return
        self.stop_task = asyncio.create_task(self.stop(reason), name=f"stop-{self.id}")

    async def stop(self, reason: str = "user") -> None:
        async with self.stop_lock:
            if self.state not in {"recording", "starting"}:
                return
            self.state = "finalizing"
            self._write_state()
            await self.status("finalizing", "正在处理最后一段语音", reason=reason)
            if self.disk_task:
                self.disk_task.cancel()
            if self.segmenter:
                for event in self.segmenter.flush():
                    await self.queue.put(event)
            await self.queue.join()
            await self.queue.put(None)
            if self.worker_task:
                await self.worker_task
            await self.status("saving", "正在保存录音和完整逐句稿")
            if self.audio_writer:
                self.audio_segments = await asyncio.to_thread(self.audio_writer.close)
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
            )
            if not utterances:
                self.summary = "本次会议未检测到有效发言，无法生成会议纪要。"
                (self.output_dir / "meeting_minutes.md").write_text(self.summary + "\n", encoding="utf-8")
                self.state = "complete"
                self.files.append("meeting_minutes.md")
                self._finish_export(utterances)
                await self.broadcast("summary_complete", content=self.summary, files=self.files)
                return
            # Stopping a meeting only finalizes and exports the recording. AI
            # summarization is deliberately a separate user action so a user
            # can review the transcript before sending it to the configured
            # service.
            self.state = "summary_pending"
            self.summary = ""
            self.error = None
            self._write_state()
            await self.status("summary_pending", "会议已保存，可手动生成会议纪要")
            await self.broadcast(
                "summary_pending",
                session_id=self.id,
                files=self.files,
                utterance_count=len(utterances),
            )

    async def _run_summary(self, utterances: list[Utterance] | None = None) -> None:
        self.state = "summarizing"
        self.summary = ""
        self.error = None
        self._write_state()
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
        )
        if (self.output_dir / "meeting_minutes.md").exists():
            self.files.append("meeting_minutes.md")
        self._write_state()

    async def retry_summary(self) -> None:
        if self.state not in {"summary_pending", "summary_error"}:
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
            error=self.error,
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
        state_files = sorted(root.glob("*/session_state.json"), key=lambda path: path.stat().st_mtime)[-10:]
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
                recovered_state = payload.get("state")
                if recovered_state in {"complete", "summary_pending", "summary_error"}:
                    session.state = recovered_state
                    session.error = payload.get("error")
                    minutes = session.output_dir / "meeting_minutes.md"
                    if minutes.exists():
                        session.summary = minutes.read_text(encoding="utf-8")
                session.files = [path.name for path in session.output_dir.iterdir() if path.is_file()]
                self.sessions[session.id] = session
            except Exception:
                continue

    async def create(self, hotwords: str | None = None) -> LiveMeetingSession:
        async with self.lock:
            if self.active_id:
                active = self.sessions.get(self.active_id)
                if active and active.state not in TERMINAL_STATES:
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

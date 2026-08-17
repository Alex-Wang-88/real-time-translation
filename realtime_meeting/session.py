from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import math
import shutil
import time
import uuid
from collections import deque
from contextlib import suppress
from datetime import datetime
from typing import Any, Callable

from fastapi import WebSocket

from .audio import (
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    RotatingAudioWriter,
    SegmentEvent,
    StreamSegmenter,
    volume_threshold_percent_to_rms,
)
from .config import DEFAULT_RECOGNITION_ARCHITECTURE, Settings, normalize_meeting_settings, normalize_recognition_architecture
from .exporter import export_live_result, render_todo_markdown
from .jimo import MeetingSummarizer, TodoGenerator
from .language import LanguageEvidenceAggregator, LanguageGuess, is_mixed_source_text, normalize_qwen_label, reconcile_language_guess
from .models import TodoDocument, Utterance, utc_now_iso
from .quality import AsrQualityAssessment, AsrQualityState, assess_asr_quality
from .scheduler import LatestEventQueue, LatestTranslationQueue
from .storage import LocalMeetingStore, TranscriptStore, atomic_write_json, atomic_write_text
from .text_normalize import simplify_chinese


LOGGER = logging.getLogger(__name__)
PARTIAL_ASR_WINDOW_SECONDS = 6.0
STATE_PERSIST_INTERVAL_SECONDS = 2.0


class CapacityLimitError(RuntimeError):
    pass


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _same_language_key(left: tuple[str, str], right: tuple[str, str]) -> bool:
    return left[0] == right[0] and left[1] == right[1]


class LiveMeetingSession:
    """One meeting lifecycle and one paragraph-oriented realtime stream."""

    schema_version = "2.0"

    def __init__(
        self,
        settings: Settings,
        runtime: Any,
        store: LocalMeetingStore,
        *,
        meeting_id: str | None = None,
        title: str = "未命名会议",
        recovered_state: dict[str, Any] | None = None,
        summarizer_factory: Callable[[Settings], Any] | None = None,
        todo_factory: Callable[[Settings], Any] | None = None,
        **_legacy_options: Any,
    ) -> None:
        self.settings = settings
        self.runtime = runtime
        self.store = store
        self.id = meeting_id or f"meeting-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.title = title or "未命名会议"
        self.output_dir = store.meeting_dir(self.id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path = self.output_dir / "transcript.jsonl"
        self.transcript_store = TranscriptStore(self.transcript_path)
        self.state_path = self.output_dir / "session_state.json"

        self.recording_state = "created"
        self.summary_state = "idle"
        self.todo_state = "waiting_summary"
        self.summary = ""
        self.summary_revision = 0
        self.snapshot_revision = 0
        self.todo = TodoDocument(meeting_id=self.id)
        self.summary_error: str | None = None
        self.todo_error: str | None = None
        self.error: str | None = None
        self.post_translation_state = "idle"
        self.post_translation_error: str | None = None
        self.post_translation_task: asyncio.Task[Any] | None = None
        self.started_at = ""
        self.ended_at = ""
        self.files: list[str] = []
        self.model_metadata: dict[str, Any] = {}

        self.meeting_settings = normalize_meeting_settings({}, settings)
        self.volume_threshold_percent = float(self.meeting_settings["volume_threshold_percent"])
        self.audio_sample_rate = SAMPLE_RATE
        self.audio_channels = 1
        self.audio_samples_received = 0
        self.last_audio_sequence: int | None = None
        self.audio_source_id: str | None = None
        self._audio_stop_requested = False
        self._audio_inflight = 0
        self._audio_condition = asyncio.Condition()
        self._last_state_persist_at = 0.0
        self.audio_segments: list[dict[str, Any]] = []
        self.audio_writer: RotatingAudioWriter | None = None
        self.segmenter: StreamSegmenter | None = None

        self.paragraphs: list[Utterance] = self.transcript_store.load()
        self.recent: deque[Utterance] = deque(self.paragraphs[-500:], maxlen=500)
        self.transcript_revision = max((_safe_int(item.revision) for item in self.paragraphs), default=0)
        self.next_paragraph_id = max((_safe_int(item.id) for item in self.paragraphs), default=0) + 1
        self.active_paragraph: Utterance | None = next((item for item in reversed(self.paragraphs) if not item.closed), None)
        if self.active_paragraph is not None:
            self.paragraphs = [item for item in self.paragraphs if item.segment_id != self.active_paragraph.segment_id]
            self.paragraphs.append(self.active_paragraph)

        # Language stability is intentionally per meeting, not per user.
        self.current_language = "unknown"
        self.current_variant: str | None = None
        self._stable_key: tuple[str, str] | None = None
        self._candidate_key: tuple[str, str] | None = None
        self._candidate_count = 0
        self._candidate_confidence = 0.0
        self._lid_checked_revisions: set[int] = set()
        self._language_conflicts = 0
        self._language_boundary: float | None = None
        self.language_evidence = LanguageEvidenceAggregator(
            window_seconds=settings.language_switch_window_ms / 1000.0,
            max_confirmation_seconds=settings.language_switch_max_wait_ms / 1000.0,
            confirmations=settings.language_conflict_confirmations,
        )
        self._active_technical_revision: int | None = None
        # A paragraph may span several technical audio chunks.  Keep the
        # text committed before the current chunk separate from the current
        # chunk's replaceable ASR hypothesis.
        self._active_technical_base: str | None = None
        self._forced_chunks_for_active = 0
        self._last_state_persist_at = time.monotonic()
        self._started_perf = 0.0
        self._previous_partial_text: dict[str, str] = {}
        self._stable_prefixes: dict[str, str] = {}
        self._asr_quality: dict[str, AsrQualityState] = {}

        self.clients: set[WebSocket] = set()
        self.realtime_pipeline = bool(getattr(settings, "realtime_pipeline", True))
        self.queue = (
            LatestEventQueue(maxsize=max(1, settings.inference_queue_size))
            if self.realtime_pipeline
            else asyncio.Queue(maxsize=max(1, settings.inference_queue_size))
        )
        self.translation_queue = (
            LatestTranslationQueue(maxsize=max(1, settings.translation_queue_size))
            if self.realtime_pipeline
            else asyncio.Queue(maxsize=max(1, settings.translation_queue_size))
        )
        self.worker_task: asyncio.Task[Any] | None = None
        self.translation_worker_task: asyncio.Task[Any] | None = None
        self.stop_task: asyncio.Task[Any] | None = None
        self.disconnect_stop_task: asyncio.Task[Any] | None = None
        self.summary_task: asyncio.Task[Any] | None = None
        self.todo_task: asyncio.Task[Any] | None = None
        self._translation_pending = 0
        self.translation_errors: dict[str, str] = {}
        self.pipeline_metrics: dict[str, Any] = {
            "asr_partial_dropped": 0,
            "expired_partial_dropped": 0,
            "translation_jobs_coalesced": 0,
            "translation_jobs_dropped": 0,
            "expired_translation_dropped": 0,
            "stale_translation_results": 0,
            "language_switches": 0,
            "first_partial_ms": [],
            "language_switch_ms": [],
            "asr_queue_wait_ms": [],
            "translation_queue_wait_ms": [],
            "asr_duration_ms": [],
            "language_id_duration_ms": [],
            "translation_duration_ms": [],
            "final_translation_ms": [],
            "post_translation_duration_ms": [],
            "post_translation_candidates": 0,
            "post_translation_retranslated": 0,
            "post_translation_skipped": 0,
            "post_translation_failures": 0,
            "asr_quality_low_count": 0,
            "queue_max_depth": 0,
            "asr_queue_max_depth": 0,
            "translation_queue_max_depth": 0,
        }
        self._asr_enqueue_times: dict[tuple[str, int, float, float], float] = {}
        self._translation_enqueue_times: dict[tuple[str, int, bool], float] = {}
        self._language_candidate_wall_at: float | None = None
        self._first_partial_recorded = False
        self._shutting_down = False

        self.summarizer_factory = summarizer_factory or (lambda value: MeetingSummarizer(value))
        self.todo_factory = todo_factory or (lambda value: TodoGenerator(value))
        if recovered_state:
            self._restore_state(recovered_state)

    def _restore_state(self, payload: dict[str, Any]) -> None:
        if payload.get("schema_version") != self.schema_version:
            return
        for name in (
            "title", "recording_state", "summary_state", "todo_state", "summary", "summary_revision",
            "summary_error", "todo_error", "error", "started_at", "ended_at", "files", "model_metadata", "snapshot_revision",
            "audio_samples_received", "volume_threshold_percent", "current_language", "current_variant",
            "transcript_revision", "post_translation_state", "post_translation_error",
        ):
            if name not in payload:
                continue
            value = payload[name]
            if name == "summary_revision":
                value = _safe_int(value)
            if name == "snapshot_revision":
                value = _safe_int(value)
            if name in {"audio_samples_received"}:
                value = _safe_int(value)
            if name in {"transcript_revision"}:
                value = _safe_int(value)
            if name in {"volume_threshold_percent"}:
                value = round(max(0.0, min(30.0, _safe_float(value, self.volume_threshold_percent))), 1)
            if name == "files" and not isinstance(value, list):
                value = []
            if name == "model_metadata" and not isinstance(value, dict):
                value = {}
            setattr(self, name, value)
        # Meetings written before the post-meeting pass was introduced are
        # already complete and must not be silently reprocessed on recovery.
        if "post_translation_state" not in payload:
            self.post_translation_state = "complete" if self.recording_state == "complete" else "idle"
        settings = payload.get("meeting_settings")
        if isinstance(settings, dict):
            self.meeting_settings = normalize_meeting_settings(settings, self.settings)
            if "volume_threshold_percent" in settings:
                self.volume_threshold_percent = _safe_float(
                    self.meeting_settings.get("volume_threshold_percent"),
                    self.volume_threshold_percent,
                )
        self.meeting_settings["volume_threshold_percent"] = self.volume_threshold_percent
        todo = payload.get("todo")
        if isinstance(todo, dict):
            try:
                self.todo = TodoDocument.from_dict(todo, default_meeting_id=self.id)
            except Exception:
                self.todo = TodoDocument(meeting_id=self.id)
        self.paragraphs = self.transcript_store.load()
        self.recent = deque(self.paragraphs[-500:], maxlen=500)
        self.transcript_revision = max(
            self.transcript_revision,
            max((_safe_int(item.revision) for item in self.paragraphs), default=0),
        )
        self.next_paragraph_id = max((_safe_int(item.id) for item in self.paragraphs), default=0) + 1
        self.active_paragraph = next((item for item in reversed(self.paragraphs) if not item.closed), None)

    @property
    def elapsed_seconds(self) -> float:
        if not self.started_at:
            return self.audio_samples_received / SAMPLE_RATE
        if self.recording_state == "recording":
            try:
                started = datetime.fromisoformat(self.started_at.replace("Z", "+00:00")).timestamp()
                return max(0.0, time.time() - started)
            except (TypeError, ValueError, OverflowError):
                pass
        return self.audio_samples_received / SAMPLE_RATE

    @property
    def active(self) -> bool:
        return self.recording_state in {"starting", "recording", "finalizing"}

    @property
    def has_active_tasks(self) -> bool:
        tasks = (
            self.worker_task,
            self.translation_worker_task,
            self.stop_task,
            self.post_translation_task,
            self.summary_task,
            self.todo_task,
        )
        return any(task is not None and not task.done() for task in tasks)

    @property
    def translation_pending(self) -> bool:
        return (
            self._translation_pending > 0
            or not self.translation_queue.empty()
            or self.post_translation_state == "running"
            or self.post_translation_task is not None and not self.post_translation_task.done()
        )

    @property
    def translation_state(self) -> str:
        if self.translation_pending:
            return "pending"
        if self.post_translation_state == "error":
            return "error"
        if self.translation_errors:
            return "error"
        if any(item.translation_status in {"ready", "not_needed"} for item in self.paragraphs):
            return "complete"
        return "idle"

    async def add_client(self, websocket: WebSocket) -> None:
        self.clients.add(websocket)

    def remove_client(self, websocket: WebSocket) -> None:
        self.clients.discard(websocket)

    def schedule_disconnect_stop(self) -> None:
        if self.disconnect_stop_task and not self.disconnect_stop_task.done():
            return
        self.disconnect_stop_task = asyncio.create_task(self._stop_after_disconnect(), name=f"disconnect-stop-{self.id}")

    async def _stop_after_disconnect(self) -> None:
        await asyncio.sleep(max(0.0, self.settings.websocket_disconnect_grace_seconds))
        if not self.clients and self.recording_state in {"starting", "recording"}:
            await self.request_stop("disconnect")

    async def broadcast(self, event_type: str, **payload: Any) -> None:
        message = {"type": event_type, **payload}
        async def send(websocket: WebSocket) -> WebSocket | None:
            try:
                await asyncio.wait_for(
                    websocket.send_json(message),
                    timeout=max(0.5, self.settings.websocket_send_timeout_seconds),
                )
                return None
            except Exception:
                return websocket

        results = await asyncio.gather(*(send(websocket) for websocket in tuple(self.clients)))
        stale = [websocket for websocket in results if websocket is not None]
        for websocket in stale:
            self.clients.discard(websocket)

    async def status(self, message: str, **payload: Any) -> None:
        await self.broadcast("status", message=message, **payload)

    def _state_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "title": self.title,
            "recording_state": self.recording_state,
            "summary_state": self.summary_state,
            "todo_state": self.todo_state,
            "summary": self.summary,
            "summary_revision": self.summary_revision,
            "snapshot_revision": self.snapshot_revision,
            "todo": self.todo.to_dict(),
            "summary_error": self.summary_error,
            "todo_error": self.todo_error,
            "error": self.error,
            "post_translation_state": self.post_translation_state,
            "post_translation_error": self.post_translation_error,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "audio_samples_received": self.audio_samples_received,
            "transcript_revision": self.transcript_revision,
            "volume_threshold_percent": self.volume_threshold_percent,
            "meeting_settings": self.meeting_settings,
            "active_paragraph": self.active_paragraph.to_dict() if self.active_paragraph else None,
            "current_language": self.current_language,
            "current_variant": self.current_variant,
            "translation_state": self.translation_state,
            "translation_pending": self.translation_pending,
            "realtime_pipeline": self.realtime_pipeline,
            "pipeline_metrics": self.pipeline_metrics,
            "language_evidence": self.language_evidence.snapshot(),
            "paragraph_count": len(self.paragraphs),
            "files": self.files,
            "model_metadata": self.model_metadata,
        }

    def _write_state(self) -> None:
        atomic_write_json(self.state_path, self._state_payload())

    def _record_metric(self, name: str, value: float) -> None:
        values = self.pipeline_metrics.setdefault(name, [])
        if not isinstance(values, list):
            values = []
            self.pipeline_metrics[name] = values
        values.append(round(float(value), 3))
        del values[:-2000]

    def rename(self, title: str) -> None:
        self.title = str(title).strip() or "未命名会议"
        self._write_state()

    def snapshot(self) -> dict[str, Any]:
        self.snapshot_revision += 1
        payload = self._state_payload()
        payload["duration_seconds"] = self.elapsed_seconds
        payload["active"] = self.active
        return payload

    def load_transcript(self) -> list[Utterance]:
        active_segment_id = self.active_paragraph.segment_id if self.active_paragraph else None
        self.paragraphs = self.transcript_store.load()
        self.recent = deque(self.paragraphs[-500:], maxlen=500)
        if active_segment_id is not None:
            self.active_paragraph = next(
                (item for item in self.paragraphs if item.segment_id == active_segment_id and not item.closed),
                None,
            )
        if self.active_paragraph is None:
            self.active_paragraph = next((item for item in reversed(self.paragraphs) if not item.closed), None)
        return list(self.paragraphs)

    def _recent_asr_context(self) -> str:
        active_id = self.active_paragraph.segment_id if self.active_paragraph else None
        source = [
            item.text
            for item in self.recent
            if item.text.strip() and item.closed and item.segment_id != active_id
        ][-2:]
        return " ".join(source)[-240:]

    async def start(self) -> None:
        if self.recording_state not in {"created"}:
            return
        self.recording_state = "starting"
        self._audio_stop_requested = False
        self._audio_inflight = 0
        self._started_perf = time.perf_counter()
        self._first_partial_recorded = False
        self.audio_source_id = None
        self.last_audio_sequence = None
        self._forced_chunks_for_active = 0
        self.started_at = utc_now_iso()
        self.error = None
        self.audio_writer = RotatingAudioWriter(self.output_dir / "audio", int(self.meeting_settings["audio_segment_minutes"]))
        self.segmenter = StreamSegmenter(
            pre_roll_ms=int(self.meeting_settings["audio_pre_roll_ms"]),
            speech_start_ms=int(self.meeting_settings["speech_start_ms"]),
            silence_ms=int(self.meeting_settings["silence_ms"]),
            partial_interval_ms=int(self.meeting_settings["partial_interval_ms"]),
            max_utterance_ms=int(float(self.meeting_settings["max_utterance_seconds"]) * 1000),
            minimum_rms=volume_threshold_percent_to_rms(self.volume_threshold_percent),
            minimum_speech_ms=int(self.meeting_settings["vad_minimum_speech_ms"]),
            minimum_speech_ratio=float(self.meeting_settings["vad_minimum_speech_ratio"]),
            vad=getattr(self.runtime, "new_vad", lambda: None)(),
        )
        self.worker_task = asyncio.create_task(self._worker(), name=f"asr-worker-{self.id}")
        self.translation_worker_task = asyncio.create_task(self._translation_worker(), name=f"translation-worker-{self.id}")
        self.recording_state = "recording"
        self._write_state()
        await self.status("录音已开始", meeting=self.snapshot())

    def configure_volume_threshold(self, value: Any) -> None:
        percent = _safe_float(value, self.volume_threshold_percent)
        self.volume_threshold_percent = round(max(0.0, min(30.0, percent)), 1)
        self.meeting_settings["volume_threshold_percent"] = self.volume_threshold_percent
        if self.segmenter:
            self.segmenter.minimum_rms = volume_threshold_percent_to_rms(self.volume_threshold_percent)
        self._write_state()

    def configure_meeting_settings(self, values: Any) -> None:
        if self.recording_state not in {"created"}:
            raise ValueError("录音开始后不能修改会议设置")
        self.meeting_settings = normalize_meeting_settings(values, self.settings)
        self.volume_threshold_percent = float(self.meeting_settings["volume_threshold_percent"])
        self._write_state()

    def configure_audio(self, payload: dict[str, Any], *, source_id: str | None = None) -> None:
        sample_rate = _safe_int(payload.get("sample_rate"), SAMPLE_RATE)
        channels = _safe_int(payload.get("channels"), 1)
        encoding = str(payload.get("encoding", "pcm_s16le"))
        if sample_rate != SAMPLE_RATE or channels != 1 or encoding.casefold() != "pcm_s16le":
            raise ValueError("音频必须是 16kHz、单声道、pcm_s16le")
        if source_id is not None and source_id != self.audio_source_id:
            self.audio_source_id = source_id
            self.last_audio_sequence = None
        self.audio_sample_rate = sample_rate
        self.audio_channels = channels

    def release_audio_source(self, source_id: str) -> None:
        if source_id == self.audio_source_id:
            self.audio_source_id = None
            self.last_audio_sequence = None

    async def wait_for_audio_drain(self, timeout: float | None = None) -> bool:
        async with self._audio_condition:
            waiter = self._audio_condition.wait_for(lambda: self._audio_inflight == 0)
            try:
                if timeout is None:
                    await waiter
                else:
                    await asyncio.wait_for(waiter, timeout=max(0.1, timeout))
            except asyncio.TimeoutError:
                return False
            return True

    async def feed_audio(
        self,
        pcm: bytes,
        sequence: int | None = None,
        *,
        source_id: str | None = None,
    ) -> None:
        async with self._audio_condition:
            if self.recording_state != "recording" or self._audio_stop_requested:
                raise ValueError("褰撳墠浼氳娌℃湁褰曢煶")
            if source_id is not None and self.audio_source_id not in {None, source_id}:
                raise ValueError("闊抽鏉ユ簮宸茶鏂版祦鏇挎崲")
            if source_id is not None and self.audio_source_id is None:
                self.audio_source_id = source_id
                self.last_audio_sequence = None
            self._audio_inflight += 1
        try:
            if self.elapsed_seconds >= self.settings.max_recording_seconds:
                await self.request_stop("max_duration")
                raise ValueError("浼氳褰曢煶宸茶揪鍒版渶澶ф椂闀?")
            await self._feed_audio_unsafe(pcm, sequence)
        finally:
            async with self._audio_condition:
                self._audio_inflight = max(0, self._audio_inflight - 1)
                self._audio_condition.notify_all()

    async def _feed_audio_unsafe(self, pcm: bytes, sequence: int | None = None) -> None:
        if self.recording_state != "recording":
            raise ValueError("当前会议没有录音")
        if not pcm or len(pcm) > self.settings.max_audio_packet_bytes:
            raise ValueError("音频包大小无效")
        if sequence is not None:
            if self.last_audio_sequence is not None and sequence <= self.last_audio_sequence:
                raise ValueError("音频包序号必须递增")
            self.last_audio_sequence = sequence
        if self.elapsed_seconds > self.settings.max_recording_seconds:
            raise ValueError("会议录音已达到最大时长")
        if len(pcm) % SAMPLE_WIDTH:
            pcm = pcm[:-1]
        if not pcm or not self.segmenter or not self.audio_writer:
            return
        self.audio_writer.write(pcm)
        self.audio_samples_received += len(pcm) // SAMPLE_WIDTH
        now = time.monotonic()
        if now - self._last_state_persist_at >= STATE_PERSIST_INTERVAL_SECONDS:
            self._last_state_persist_at = now
            with suppress(Exception):
                self._write_state()
        events = self.segmenter.feed(pcm)
        for event in events:
            await self._enqueue_event(event)

    async def _enqueue_event(self, event: SegmentEvent) -> None:
        key = (event.kind, int(event.revision), float(event.start), float(event.end))
        self._asr_enqueue_times[key] = time.perf_counter()
        try:
            if not self.realtime_pipeline:
                self.queue.put_nowait(event)
            elif event.kind == "partial":
                replaced = self.queue.put_latest_nowait(event)
                if not replaced:
                    self.pipeline_metrics["asr_partial_dropped"] = int(self.pipeline_metrics.get("asr_partial_dropped", 0)) + 1
                    self.pipeline_metrics["expired_partial_dropped"] = int(self.pipeline_metrics.get("expired_partial_dropped", 0)) + 1
                    for old_key in tuple(self._asr_enqueue_times):
                        if old_key[0] == "partial" and old_key[1] == int(event.revision) and old_key != key:
                            self._asr_enqueue_times.pop(old_key, None)
            elif event.kind == "final":
                self.queue.put_nowait(event)
                # The final event makes every older partial for this
                # technical revision obsolete, including its wait timer.
                for old_key in tuple(self._asr_enqueue_times):
                    if old_key[0] == "partial" and old_key[1] <= int(event.revision):
                        self._asr_enqueue_times.pop(old_key, None)
            self.pipeline_metrics["queue_max_depth"] = max(
                int(self.pipeline_metrics.get("queue_max_depth", 0)),
                self.queue.qsize(),
            )
            self.pipeline_metrics["asr_queue_max_depth"] = max(
                int(self.pipeline_metrics.get("asr_queue_max_depth", 0)),
                self.queue.qsize(),
            )
        except asyncio.QueueFull:
            if event.kind == "partial":
                self.pipeline_metrics["asr_partial_dropped"] = int(self.pipeline_metrics.get("asr_partial_dropped", 0)) + 1
                self.pipeline_metrics["expired_partial_dropped"] = int(self.pipeline_metrics.get("expired_partial_dropped", 0)) + 1
                return
            await self.queue.put(event)

    @staticmethod
    def _compatible_kwargs(method: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            return kwargs
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in parameters}

    def _recognition_architecture(self) -> str:
        return normalize_recognition_architecture(
            self.meeting_settings.get("recognition_architecture", self.settings.recognition_architecture),
            DEFAULT_RECOGNITION_ARCHITECTURE,
        )

    def _uses_language_id(self) -> bool:
        # ``single_1_7b_no_lid`` means no separate LID checkpoint.  A single
        # segment-level probe is still useful for language accuracy and uses
        # the same resident 1.7B model as ASR.
        return bool(getattr(self.settings, "language_id_on_segment", True))

    async def _invoke_transcriber(self, method: Any, event: SegmentEvent, kwargs: dict[str, Any]) -> Any:
        call_kwargs = self._compatible_kwargs(method, kwargs)
        call = functools.partial(method, event, **call_kwargs)
        return await self._invoke_model_call(
            call,
            timeout=max(1.0, self.settings.asr_timeout_seconds),
            executor_name="inference_executor",
        )

    async def _invoke_model_call(
        self,
        call: Callable[[], Any],
        *,
        timeout: float,
        executor_name: str,
    ) -> Any:
        """Run a synchronous model call through one bounded stage worker."""

        # LiveModelRuntime owns one executor for ASR/LID and one for
        # translation.  A timed-out future therefore cannot spawn another
        # model thread while the previous call is unwinding.
        executor = getattr(self.runtime, executor_name, None)
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(executor, call)
        # Shield the executor future so cancellation of a session worker does
        # not wait for ThreadPoolExecutor's non-cancelable callable to settle.
        # The runtime's single-worker executor still serializes the next call.
        return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)

    async def _transcribe_with_strategy(
        self,
        method: Any,
        event: SegmentEvent,
        kwargs: dict[str, Any],
        *,
        is_final: bool,
    ) -> Any:
        architecture = self._recognition_architecture()
        if architecture == "single_1_7b_no_lid":
            direct_kwargs = dict(kwargs)
            decode_settings = dict(kwargs.get("decode_settings") or {})
            # The default strategy always uses 1.7B, even if an older client
            # still sends the legacy ``realtime_asr_model`` key.
            decode_settings["realtime_asr_model"] = "primary"
            direct_kwargs["decode_settings"] = decode_settings
            return await self._invoke_transcriber(method, event, direct_kwargs)

        # The application currently has one active strategy; keep this direct
        # branch explicit so future experimental strategies can be added
        # without changing the default path.
        direct_kwargs = dict(kwargs)
        direct_settings = dict(kwargs.get("decode_settings") or {})
        direct_settings["realtime_asr_model"] = "primary"
        direct_kwargs["decode_settings"] = direct_settings
        return await self._invoke_transcriber(method, event, direct_kwargs)

    async def _worker(self) -> None:
        while True:
            event = await self.queue.get()
            try:
                if event is None:
                    return
                try:
                    key = (event.kind, int(event.revision), float(event.start), float(event.end))
                    queued_at = self._asr_enqueue_times.pop(key, None)
                    if queued_at is not None:
                        self._record_metric("asr_queue_wait_ms", (time.perf_counter() - queued_at) * 1000)
                    await self._handle_event(event)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Realtime ASR event failed for %s", self.id)
                    with suppress(Exception):
                        await self.broadcast("warning", code="asr_event_failed", message=str(exc))
            finally:
                self.queue.task_done()

    async def _maybe_detect_language(self, event: SegmentEvent, *, final: bool) -> bool:
        if not self._uses_language_id():
            return False
        # One same-model language call after the first stable window of each
        # technical speech chunk. Forced max-duration chunks get their own
        # check; partials do not repeatedly invoke a second ASR model.
        if event.revision in self._lid_checked_revisions:
            return False
        if not final and event.end - event.start < self.settings.language_id_min_seconds:
            return False
        self._lid_checked_revisions.add(event.revision)
        detector = getattr(self.runtime, "detect_language", None)
        if not detector:
            return False
        kwargs = self._compatible_kwargs(detector, {"previous_language": self.current_language})
        started_lid = time.perf_counter()
        try:
            guess = await self._invoke_model_call(
                functools.partial(detector, event.pcm, **kwargs),
                timeout=max(1.0, self.settings.asr_timeout_seconds),
                executor_name="inference_executor",
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Realtime language detection failed for %s", self.id)
            return False
        finally:
            self._record_metric("language_id_duration_ms", (time.perf_counter() - started_lid) * 1000)
        return self._accept_language(guess, final=final, start=event.start, end=event.end, source="lid")

    def _accept_language(
        self,
        guess: LanguageGuess | None,
        *,
        final: bool,
        text: str = "",
        start: float | None = None,
        end: float | None = None,
        source: str = "asr",
    ) -> bool:
        self._language_boundary = None
        if self.language_evidence.stable_code is None and self.active_paragraph is not None:
            active_code = str(self.active_paragraph.language or "").strip().casefold()
            if active_code in {"zh", "en", "de"}:
                self.language_evidence.seed_stable(active_code, self.active_paragraph.speech_variant)
        candidate_code = getattr(guess, "code", None) if guess is not None else None
        if self.language_evidence.stable_code is not None and candidate_code == self.language_evidence.stable_code:
            self._language_candidate_wall_at = None
        elif self.language_evidence.stable_code is not None and candidate_code in {"zh", "en", "de"} and (
            self.language_evidence.candidate_count == 0
            or self.language_evidence.candidate_code != candidate_code
        ):
            self._language_candidate_wall_at = time.perf_counter()
        transition = self.language_evidence.observe(
            guess,
            text=text,
            start=start,
            end=end,
            final=final,
            source=source,
        )
        stable_code = self.language_evidence.stable_code
        stable_variant = self.language_evidence.stable_variant
        self.current_language = stable_code or self.current_language
        self.current_variant = stable_variant or self.current_variant
        self._stable_key = (stable_code, stable_variant or "unknown") if stable_code else None
        self._candidate_key = (
            self.language_evidence.candidate_code,
            self.language_evidence.candidate_variant or "unknown",
        ) if self.language_evidence.candidate_code else None
        self._candidate_count = self.language_evidence.candidate_count
        self._candidate_confidence = self.language_evidence.candidate_confidence
        if transition is None:
            return False
        self._language_boundary = transition.boundary
        if self._language_candidate_wall_at is not None:
            self._record_metric("language_switch_ms", (time.perf_counter() - self._language_candidate_wall_at) * 1000)
        self._language_candidate_wall_at = None
        self._language_conflicts = 0
        self.pipeline_metrics["language_switches"] = int(self.pipeline_metrics.get("language_switches", 0)) + 1
        return True

    async def _handle_event(self, event: SegmentEvent) -> None:
        is_final = event.kind == "final" and not event.forced
        paragraph_start: float | None = None
        quality_language_conflict = False
        self._language_boundary = None
        if event.kind == "partial" and (
            not getattr(event, "has_new_speech", True)
            or not getattr(event, "has_audio", True)
        ):
            return
        if not getattr(event, "has_audio", True):
            if is_final:
                await self._close_active(event.end)
            return
        self._forced_chunks_for_active = self._forced_chunks_for_active + 1 if event.forced else 0
        language_changed = await self._maybe_detect_language(event, final=is_final)
        if language_changed:
            paragraph_start = max(
                event.start,
                self._language_boundary or event.start,
                self.active_paragraph.end if self.active_paragraph else event.start,
            )
            await self._close_active(paragraph_start)

        # Do not force the previous paragraph language into Qwen while the
        # paragraph is still live. ``language=`` is a hard decode constraint in
        # the Qwen wrapper; keeping it here made a language switch look like
        # the previous language forever and prevented a paragraph boundary.
        # A language confirmed by the same-model probe for a new technical
        # segment is safe to pass through as a constraint.
        use_stable_language = self._uses_language_id() and (self.active_paragraph is not None or language_changed)
        stable_language = self.current_language if use_stable_language and self.current_language in {"zh", "en", "de"} else None
        decode_language = self.current_language if language_changed and self.current_language in {"zh", "en", "de"} else None
        decode_variant = self.current_variant if language_changed else None
        method = getattr(self.runtime, "transcribe_final" if is_final else "transcribe_partial")
        kwargs = {
            "recent_text": self._recent_asr_context(),
            "previous_language": stable_language,
            "language": decode_language,
            "speech_variant": decode_variant,
            "decode_settings": self.meeting_settings,
        }
        asr_event = event
        if language_changed and self._language_boundary is not None:
            asr_event = event.slice(self._language_boundary, event.end)
        if not is_final:
            keep_bytes = int(PARTIAL_ASR_WINDOW_SECONDS * SAMPLE_RATE) * SAMPLE_WIDTH
            if len(event.pcm) > keep_bytes:
                crop_start = max(event.start, event.end - PARTIAL_ASR_WINDOW_SECONDS)
                if language_changed and self._language_boundary is not None:
                    crop_start = max(crop_start, self._language_boundary)
                asr_event = event.slice(crop_start, event.end)
        started_asr = time.perf_counter()
        try:
            result = await self._transcribe_with_strategy(method, asr_event, kwargs, is_final=is_final)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Realtime ASR failed for %s", self.id)
            result = None
        self._record_metric("asr_duration_ms", (time.perf_counter() - started_asr) * 1000)
        if result is None:
            if is_final:
                await self._close_active(event.end)
            elif event.forced and self._forced_chunks_for_active >= 2:
                await self._close_active(event.end)
            return
        result_text = str(getattr(result, "text", "") or "").strip()
        result_guess = self._result_language_guess(result, result_text)
        if not result_text:
            # Empty ASR output is not a new paragraph and must not refresh the
            # current paragraph with a model hallucination. A final empty
            # result still closes real audio already admitted by the segmenter.
            if is_final:
                await self._close_active(event.end)
            elif event.forced and self._forced_chunks_for_active >= 2:
                await self._close_active(event.end)
            return

        if event.kind == "partial" and not self._first_partial_recorded and self._started_perf:
            self._first_partial_recorded = True
            self._record_metric("first_partial_ms", (time.perf_counter() - self._started_perf) * 1000)

        result_code = result_guess.code if result_guess.code in {"zh", "en", "de"} else None
        active_code = self.active_paragraph.language if self.active_paragraph is not None else None
        expected_code = self._stable_key[0] if self._stable_key is not None else active_code
        if result_code is not None and expected_code is not None:
            if result_code != expected_code:
                quality_language_conflict = True
                self._language_conflicts += 1
                confirmed = False
                if is_final or self._language_conflicts >= self.settings.language_conflict_confirmations:
                    # The secondary detector is useful when available, but a
                    # consistent ASR language result must be sufficient. If
                    # the detector is unavailable, requiring it leaves the
                    # old paragraph open indefinitely.
                    confirmed = await self._confirm_language_conflict(event, result_guess)
                    if not confirmed:
                        confirmed = self._accept_language(
                            result_guess,
                            final=True,
                            text=result_text,
                            start=event.start,
                            end=event.end,
                            source="asr",
                        )
                if not confirmed:
                    # Never write a conflicting hypothesis into the old
                    # paragraph. That was the source of the disappearing
                    # translation when a language changed mid-stream.
                    if is_final:
                        await self._close_active(event.end)
                    return
                # A real segment carries the absolute frame map.  Re-run the
                # old prefix and the new suffix after confirmation so one ASR
                # window cannot leak words across the paragraph boundary.
                boundary = self._language_boundary
                if (
                    event.frames is not None
                    and boundary is not None
                    and event.start < boundary < event.end
                    and self.active_paragraph is not None
                ):
                    old_code = expected_code if expected_code in {"zh", "en", "de"} else self.active_paragraph.language
                    old_event = event.slice(event.start, boundary)
                    old_kwargs = dict(kwargs)
                    old_kwargs["language"] = old_code
                    old_kwargs["previous_language"] = old_code
                    old_kwargs["speech_variant"] = self.active_paragraph.speech_variant if old_code == "zh" else None
                    try:
                        self.active_paragraph.end = min(self.active_paragraph.end, boundary)
                        old_result = await self._transcribe_with_strategy(
                            method,
                            old_event,
                            old_kwargs,
                            is_final=is_final,
                        )
                        old_text = str(getattr(old_result, "text", "") or "").strip()
                        if old_text:
                            # Discard the replaceable mixed-window hypothesis,
                            # retaining only text committed by prior chunks.
                            if self._active_technical_base is not None:
                                self.active_paragraph.text = self._active_technical_base
                            old_guess = self._result_language_guess(old_result, old_text)
                            await self._upsert_source(
                                old_event,
                                old_text,
                                language=old_code,
                                speech_variant=getattr(old_result, "speech_variant", None) or old_guess.speech_variant,
                                confidence=max(_safe_float(getattr(old_result, "confidence", 0.0)), old_guess.confidence),
                                asr_model=getattr(old_result, "model", None),
                                language_source=getattr(old_result, "language_source", "qwen"),
                            )
                    except Exception:  # noqa: BLE001
                        LOGGER.exception("Language boundary prefix re-segmentation failed for %s", self.id)
                    new_event = event.slice(boundary, event.end)
                    new_kwargs = dict(kwargs)
                    new_kwargs["language"] = result_guess.code
                    new_kwargs["previous_language"] = result_guess.code
                    new_kwargs["speech_variant"] = result_guess.speech_variant
                    try:
                        new_result = await self._transcribe_with_strategy(
                            method,
                            new_event,
                            new_kwargs,
                            is_final=is_final,
                        )
                        new_text = str(getattr(new_result, "text", "") or "").strip()
                        if new_text:
                            result = new_result
                            result_text = new_text
                            result_guess = self._result_language_guess(new_result, new_text)
                    except Exception:  # noqa: BLE001
                        LOGGER.exception("Language boundary suffix re-segmentation failed for %s", self.id)
                paragraph_start = max(
                    event.start,
                    self._language_boundary or event.start,
                    self.active_paragraph.end if self.active_paragraph else event.start,
                )
                await self._close_active(paragraph_start)
                language_changed = True
            else:
                self._language_conflicts = 0
                self._accept_language(
                    result_guess,
                    final=is_final,
                    text=result_text,
                    start=event.start,
                    end=event.end,
                    source="asr",
                )
        elif result_code is not None and self._stable_key is None:
            self._accept_language(result_guess, final=is_final, text=result_text, start=event.start, end=event.end)

        language = self.current_language if self.current_language in {"zh", "en", "de"} else result_guess.code
        if language not in {"zh", "en", "de"}:
            language = "unknown"
        variant = self.current_variant if language == "zh" else getattr(result, "speech_variant", None)
        if language == "zh" and not variant:
            variant = result_guess.speech_variant
        await self._upsert_source(
            event,
            result_text,
            start_override=paragraph_start,
            language=language,
            speech_variant=variant,
            confidence=max(_safe_float(getattr(result, "confidence", 0.0)), self._candidate_confidence),
            asr_model=getattr(result, "model", None),
            language_source=getattr(result, "language_source", "qwen"),
            language_conflict=quality_language_conflict,
        )
        if is_final:
            await self._close_active(event.end)
        elif event.forced:
            if self._forced_chunks_for_active >= 2:
                await self._close_active(event.end)

    @staticmethod
    def _result_language_guess(result: Any, text: str) -> LanguageGuess:
        raw_language = str(
            getattr(result, "raw_qwen_label", "")
            or getattr(result, "language", "unknown")
            or "unknown"
        )
        normalized_result = normalize_qwen_label(raw_language, text)
        reconciled = reconcile_language_guess(normalized_result, text)
        if reconciled is not None:
            normalized_result = reconciled
        return LanguageGuess(
            normalized_result.code,
            max(_safe_float(getattr(result, "confidence", 0.0)), normalized_result.confidence),
            getattr(result, "speech_variant", None) or normalized_result.speech_variant,
            raw_language,
        )

    async def _confirm_language_conflict(self, event: SegmentEvent, guess: LanguageGuess) -> bool:
        detector = getattr(self.runtime, "detect_language", None)
        if not self.settings.language_id_on_conflict or not detector:
            self._language_conflicts = 0
            return self._accept_language(guess, final=True, start=event.start, end=event.end, source="asr")
        kwargs = self._compatible_kwargs(detector, {"previous_language": self.current_language})
        call = functools.partial(detector, event.pcm, **kwargs)
        started_lid = time.perf_counter()
        try:
            confirmed = await self._invoke_model_call(
                call,
                timeout=max(1.0, self.settings.asr_timeout_seconds),
                executor_name="inference_executor",
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Realtime language conflict detection failed for %s", self.id)
            return False
        finally:
            self._record_metric("language_id_duration_ms", (time.perf_counter() - started_lid) * 1000)
        self._language_conflicts = 0
        # Consecutive ASR conflicts are the trigger; the same 1.7B model's
        # agreeing result is the confirmation that commits the new boundary.
        return bool(
            confirmed
            and getattr(confirmed, "code", "") == getattr(guess, "code", "")
            and self._accept_language(
                confirmed,
                final=True,
                start=event.start,
                end=event.end,
                source="lid",
            )
        )

    @staticmethod
    def _merge_technical_text(previous: str, current: str) -> str:
        left, right = previous.strip(), current.strip()
        if not left:
            return right
        if not right:
            return left
        if right.casefold() in left.casefold():
            return left
        if left.casefold() in right.casefold():
            return right
        for size in range(min(80, len(left), len(right)), 1, -1):
            if left[-size:].casefold() == right[:size].casefold():
                return left + right[size:]
        left_words, right_words = left.split(), right.split()
        for size in range(min(12, len(left_words), len(right_words)), 0, -1):
            if [word.casefold() for word in left_words[-size:]] == [word.casefold() for word in right_words[:size]]:
                return " ".join(left_words + right_words[size:])
        return (left + " " + right).strip()

    @staticmethod
    def _needs_translation(language: str, text: str) -> bool:
        return language in {"en", "de"} or language == "zh" and is_mixed_source_text(text)

    @staticmethod
    def _translation_language(language: str, text: str) -> str:
        if language in {"en", "de"}:
            return language
        if language == "zh" and is_mixed_source_text(text):
            guessed = normalize_qwen_label("unknown", text).code
            return guessed if guessed in {"en", "de"} else "en"
        return language

    def _observe_asr_quality(
        self,
        item: Utterance,
        event: SegmentEvent,
        *,
        confidence: float,
        language_conflict: bool = False,
    ) -> AsrQualityAssessment:
        state = self._asr_quality.setdefault(item.segment_id, AsrQualityState())
        state.observe(
            item.text,
            confidence,
            is_final=event.kind == "final",
            is_partial=event.kind == "partial",
            language_conflict=language_conflict,
        )
        assessment = assess_asr_quality(
            item.text,
            start=item.start,
            end=item.end,
            confidence=confidence,
            state=state,
        )
        if assessment.is_low:
            self.pipeline_metrics["asr_quality_low_count"] = int(self.pipeline_metrics.get("asr_quality_low_count", 0)) + 1
        return assessment

    async def _upsert_source(
        self,
        event: SegmentEvent,
        text: str,
        *,
        start_override: float | None = None,
        language: str,
        speech_variant: str | None,
        confidence: float,
        asr_model: str | None,
        language_source: str,
        language_conflict: bool = False,
    ) -> None:
        if not text and self.active_paragraph is None:
            return
        # Dialect metadata may change while the speaker continues speaking;
        # only a language-code change is a paragraph boundary.
        key = language
        if self.active_paragraph is not None:
            active_key = self.active_paragraph.language
            if active_key != key and self.active_paragraph.text.strip():
                boundary = max(event.start, self.active_paragraph.end)
                await self._close_active(max(boundary, start_override) if start_override is not None else boundary)
        if self.active_paragraph is None:
            segment_id = f"p-{self.next_paragraph_id:06d}"
            needs_translation = self._needs_translation(language, text)
            item = Utterance(
                id=self.next_paragraph_id,
                segment_id=segment_id,
                start=float(start_override if start_override is not None else event.start),
                end=event.end,
                language=language,
                speech_variant=speech_variant,
                language_confidence=max(0.0, min(1.0, confidence)),
                text=text,
                translation_zh="" if needs_translation else simplify_chinese(text) if language == "zh" else "",
                translation_status="streaming" if needs_translation else "not_needed" if language == "zh" else "pending",
                asr_model=asr_model,
                language_source=language_source,
            )
            self.next_paragraph_id += 1
            self.active_paragraph = item
            self._active_technical_revision = event.revision
            self._active_technical_base = ""
        else:
            item = self.active_paragraph
            item.end = max(item.end, event.end)
            item.language_confidence = max(item.language_confidence, confidence)
            item.asr_model = asr_model or item.asr_model
            item.language_source = language_source or item.language_source
            item.speech_variant = speech_variant or item.speech_variant
            if self._active_technical_revision != event.revision or self._active_technical_base is None:
                self._active_technical_revision = event.revision
                self._active_technical_base = item.text.strip()
            # Qwen's result for one technical chunk is a replaceable
            # hypothesis.  Merge that hypothesis with the text committed by
            # earlier chunks, rather than replacing the entire paragraph.
            # Empty results must not erase a good hypothesis or advance the
            # paragraph text to an empty value.
            merged_text = (
                self._merge_technical_text(self._active_technical_base, text)
                if str(text or "").strip()
                else item.text
            )
            if merged_text != item.text:
                item.text = merged_text
                item.source_revision += 1
                item.revision += 1
                if self._needs_translation(item.language, item.text):
                    item.translation_zh = ""
                    item.translation_status = "streaming"
                elif item.language == "zh":
                    item.translation_zh = simplify_chinese(item.text)
                    item.translation_status = "not_needed"
        if item.language == "zh" and not self._needs_translation(item.language, item.text):
            item.text = simplify_chinese(item.text)
            item.translation_zh = item.text
            item.translation_status = "not_needed"
        self._observe_asr_quality(
            item,
            event,
            confidence=confidence,
            language_conflict=language_conflict,
        )
        self._upsert_recent(item)
        self._persist_paragraph(item)
        await self.broadcast("paragraph_update", paragraph=item.to_dict(), transcript_revision=self.transcript_revision)
        if self._needs_translation(item.language, item.text) and item.text.strip():
            await self._schedule_translation(item, final=False)

    def _upsert_recent(self, item: Utterance) -> None:
        for index, current in enumerate(self.paragraphs):
            if current.segment_id == item.segment_id:
                self.paragraphs[index] = item
                break
        else:
            self.paragraphs.append(item)
        self.recent = deque(self.paragraphs[-500:], maxlen=500)

    def _persist_paragraph(self, item: Utterance) -> None:
        self.transcript_revision += 1
        self.transcript_store.append(item)

    async def _close_active(self, end: float | None = None) -> None:
        item = self.active_paragraph
        if item is None:
            self._active_technical_revision = None
            self._active_technical_base = None
            self._language_conflicts = 0
            self._forced_chunks_for_active = 0
            return
        if not item.text.strip():
            self.active_paragraph = None
            self._active_technical_revision = None
            self._active_technical_base = None
            self._language_conflicts = 0
            self._forced_chunks_for_active = 0
            return
        if end is not None:
            item.end = max(item.end, float(end))
        if not item.closed:
            item.closed = True
            item.revision += 1
            self._upsert_recent(item)
            self._persist_paragraph(item)
            await self.broadcast("paragraph_update", paragraph=item.to_dict(), transcript_revision=self.transcript_revision)
        if self._needs_translation(item.language, item.text):
            await self._schedule_translation(item, final=True)
        self.active_paragraph = None
        self._active_technical_revision = None
        self._active_technical_base = None
        self._language_conflicts = 0
        self._forced_chunks_for_active = 0
        self._previous_partial_text.pop(item.segment_id, None)
        self._stable_prefixes.pop(item.segment_id, None)

    @staticmethod
    def _stable_prefix(previous: str, current: str, *, minimum: int) -> str:
        if not previous or not current:
            return ""
        size = 0
        for left, right in zip(previous, current):
            if left != right:
                break
            size += 1
        common = current[:size]
        if len(common) < minimum:
            return ""
        if common[-1:].isspace() or common[-1:] in "，。！？；：,.!?;:":
            return common.strip()
        boundary = max(common.rfind(" "), common.rfind("\n"))
        return common[:boundary].strip() if boundary >= minimum // 2 else common.strip()

    async def _schedule_translation(self, item: Utterance, *, final: bool) -> None:
        previous = self._previous_partial_text.get(item.segment_id, "")
        if final:
            prefix = item.text
        elif not previous and len(item.text.strip()) >= 4:
            # Give the first useful partial a provisional translation.  Later
            # revisions supersede it by source_revision, so this improves the
            # first-response latency without allowing stale text to win.
            prefix = item.text.strip()
        else:
            prefix = self._stable_prefix(previous, item.text, minimum=self.settings.stable_prefix_min_chars)
        self._previous_partial_text[item.segment_id] = item.text
        if not prefix:
            return
        if prefix == self._stable_prefixes.get(item.segment_id) and not final:
            return
        now = time.perf_counter()
        last_scheduled = getattr(self, "_translation_last_schedule_time", {}).get(item.segment_id, 0.0)
        if not final and now - last_scheduled < self.settings.translation_partial_debounce_ms / 1000.0:
            return
        if not hasattr(self, "_translation_last_schedule_time"):
            self._translation_last_schedule_time = {}
        self._translation_last_schedule_time[item.segment_id] = now
        self._stable_prefixes[item.segment_id] = prefix
        if item.translation_status != "streaming":
            item.translation_status = "streaming"
            # Translation status is visible to the client. Give it a newer
            # revision so delayed websocket events cannot overwrite it.
            item.revision += 1
        self._upsert_recent(item)
        self._persist_paragraph(item)
        await self.broadcast("paragraph_update", paragraph=item.to_dict(), transcript_revision=self.transcript_revision)
        job = {
            "segment_id": item.segment_id,
            "source_revision": item.source_revision,
            "text": prefix,
            "language": self._translation_language(item.language, prefix),
            "final": final,
            "attempt": 0,
        }
        try:
            dropped_before = int(getattr(self.translation_queue, "dropped_provisional", 0))
            if not self.realtime_pipeline:
                if final:
                    await asyncio.wait_for(
                        self.translation_queue.put(job),
                        timeout=max(1.0, self.settings.translation_deadline_seconds),
                    )
                else:
                    self.translation_queue.put_nowait(job)
                queued = True
            elif final:
                queued = await asyncio.wait_for(
                    self.translation_queue.put_latest(job),
                    timeout=max(1.0, self.settings.translation_deadline_seconds),
                )
            else:
                queued = self.translation_queue.put_nowait(job)
            if queued:
                self._translation_pending += 1
                self._translation_enqueue_times[(item.segment_id, item.source_revision, final)] = time.perf_counter()
            else:
                self.pipeline_metrics["translation_jobs_coalesced"] = int(self.pipeline_metrics.get("translation_jobs_coalesced", 0)) + 1
            self.pipeline_metrics["queue_max_depth"] = max(
                int(self.pipeline_metrics.get("queue_max_depth", 0)),
                self.translation_queue.qsize(),
            )
            self.pipeline_metrics["translation_queue_max_depth"] = max(
                int(self.pipeline_metrics.get("translation_queue_max_depth", 0)),
                self.translation_queue.qsize(),
            )
            dropped_after = int(getattr(self.translation_queue, "dropped_provisional", 0))
            if dropped_after > dropped_before:
                dropped = dropped_after - dropped_before
                dropped_jobs = list(getattr(self.translation_queue, "dropped_provisional_jobs", ()))
                dropped_jobs = dropped_jobs[-dropped:]
                for dropped_job in dropped_jobs:
                    dropped_key = (
                        str(dropped_job.get("segment_id", "")),
                        _safe_int(dropped_job.get("source_revision")),
                        bool(dropped_job.get("final")),
                    )
                    self._translation_enqueue_times.pop(dropped_key, None)
                self._translation_pending = max(0, self._translation_pending - dropped)
                self.pipeline_metrics["translation_jobs_dropped"] = int(self.pipeline_metrics.get("translation_jobs_dropped", 0)) + dropped
                self.pipeline_metrics["expired_translation_dropped"] = int(self.pipeline_metrics.get("expired_translation_dropped", 0)) + dropped
        except asyncio.QueueFull:
            # A partial translation is only a hint; the next final paragraph
            # will enqueue the authoritative source text.
            self.pipeline_metrics["translation_jobs_dropped"] = int(self.pipeline_metrics.get("translation_jobs_dropped", 0)) + 1
            self.pipeline_metrics["expired_translation_dropped"] = int(self.pipeline_metrics.get("expired_translation_dropped", 0)) + 1
        except asyncio.TimeoutError:
            self.pipeline_metrics["translation_jobs_dropped"] = int(self.pipeline_metrics.get("translation_jobs_dropped", 0)) + 1
            self.pipeline_metrics["expired_translation_dropped"] = int(self.pipeline_metrics.get("expired_translation_dropped", 0)) + 1
            LOGGER.warning("Translation queue remained full for %s", self.id)

    async def _translation_worker(self) -> None:
        while True:
            job = await self.translation_queue.get()
            try:
                if job is None:
                    return
                try:
                    await self._run_translation_job(job)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    LOGGER.exception("Translation worker failed for %s", self.id)
            finally:
                try:
                    self.translation_queue.task_done(job)
                except TypeError:
                    self.translation_queue.task_done()

    async def _run_translation_job(self, job: dict[str, Any]) -> None:
        segment_id = str(job["segment_id"])
        source_revision = _safe_int(job["source_revision"])
        final = bool(job.get("final"))
        queued_at = self._translation_enqueue_times.pop((segment_id, source_revision, final), None)
        if queued_at is not None:
            self._record_metric("translation_queue_wait_ms", (time.perf_counter() - queued_at) * 1000)
        started = time.perf_counter()
        try:
            result = await self._translate(job["text"], job["language"], final=final)
            self._record_metric("translation_duration_ms", (time.perf_counter() - started) * 1000)
            item = next((value for value in self.paragraphs if value.segment_id == segment_id), None)
            if item is None or item.source_revision != source_revision:
                self.pipeline_metrics["stale_translation_results"] = int(self.pipeline_metrics.get("stale_translation_results", 0)) + 1
                return
            status = str(getattr(result, "status", "failed"))
            if status in {"ready", "not_needed"}:
                item.translation_zh = str(getattr(result, "text", "") or "")
                item.translation_status = "ready" if status == "ready" else status
                item.translation_model = getattr(result, "model", None)
                self.translation_errors.pop(item.segment_id, None)
            else:
                # A deadline failure is terminal for this revision.  The
                # source remains visible and the explicit retry endpoint can
                # enqueue a fresh final job without extending the live ASR
                # critical path beyond the 4.5 s translation budget.
                item.translation_status = "failed"
                self.translation_errors[item.segment_id] = str(getattr(result, "error", None) or status)
            item.revision += 1
            self._upsert_recent(item)
            self._persist_paragraph(item)
            await self.broadcast("paragraph_update", paragraph=item.to_dict(), transcript_revision=self.transcript_revision)
            if final and queued_at is not None:
                self._record_metric("final_translation_ms", (time.perf_counter() - queued_at) * 1000)
        except Exception as exc:
            item = next((value for value in self.paragraphs if value.segment_id == segment_id), None)
            if item is not None and item.source_revision == source_revision:
                item.translation_status = "failed"
                item.revision += 1
                self.translation_errors[item.segment_id] = str(exc)
                self._upsert_recent(item)
                self._persist_paragraph(item)
                await self.broadcast("paragraph_update", paragraph=item.to_dict(), transcript_revision=self.transcript_revision)
        finally:
            self._translation_pending = max(0, self._translation_pending - 1)
            with suppress(Exception):
                self._write_state()

    async def _translate(self, text: str, language: str, *, final: bool = False) -> Any:
        method = getattr(self.runtime, "translate_text")
        settings = dict(self.meeting_settings)
        settings["_gpu_priority"] = 20 if final else 60
        kwargs = self._compatible_kwargs(method, {"translation_settings": settings})
        call = functools.partial(method, text, language, **kwargs)
        return await self._invoke_model_call(
            call,
            timeout=max(
                1.0,
                min(
                    self.settings.translation_timeout_seconds,
                    self.settings.translation_deadline_seconds,
                ),
            ),
            executor_name="translation_executor",
        )

    def _post_translation_context(self, items: list[Utterance], index: int) -> str:
        radius = max(0, _safe_int(getattr(self.settings, "post_meeting_translation_context_paragraphs", 2), 2))
        if radius == 0:
            return ""
        parts: list[str] = []
        start = max(0, index - radius)
        end = min(len(items), index + radius + 1)
        for position in range(start, end):
            if position == index:
                continue
            value = items[position].text.strip()
            if not value:
                continue
            label = "上文" if position < index else "下文"
            parts.append(f"[{label}] {value}")
        context = "\n".join(parts)
        limit = max(64, _safe_int(getattr(self.settings, "post_meeting_translation_context_chars", 240), 240))
        return context[-limit:]

    def _post_translation_candidates(
        self,
        items: list[Utterance],
    ) -> tuple[list[tuple[Utterance, AsrQualityAssessment, str, int]], int]:
        threshold = max(
            0.0,
            min(1.0, _safe_float(getattr(self.settings, "post_meeting_translation_quality_threshold", 0.62), 0.62)),
        )
        candidates: list[tuple[Utterance, AsrQualityAssessment, str, int]] = []
        target_count = 0
        for index, item in enumerate(items):
            if not self._needs_translation(item.language, item.text) or not item.text.strip():
                continue
            target_count += 1
            quality = assess_asr_quality(
                item.text,
                start=item.start,
                end=item.end,
                confidence=item.language_confidence,
                state=self._asr_quality.get(item.segment_id),
            )
            translation_failed = item.translation_status in {"failed", "pending", "streaming"} or item.segment_id in self.translation_errors
            if translation_failed or quality.score < threshold or quality.is_low:
                candidates.append((item, quality, self._post_translation_context(items, index), item.source_revision))
        return candidates, target_count

    async def _translate_post_batch(
        self,
        texts: list[str],
        language: str,
        contexts: list[str],
    ) -> list[Any]:
        settings = dict(self.meeting_settings)
        settings["_gpu_priority"] = 30
        settings["_post_meeting"] = True
        settings["_translation_contexts"] = list(contexts)
        # The resident OPUS-MT engine remains local and sentence-level.  A
        # context-capable runtime may consume _translation_contexts; the
        # current engine still benefits from the more conservative final-pass
        # decode settings and batched invocation.
        settings["translation_beam_size"] = max(
            4,
            _safe_int(settings.get("translation_beam_size"), 2),
        )
        settings["translation_max_decoding_length"] = max(
            512,
            _safe_int(settings.get("translation_max_decoding_length"), 384),
        )
        settings["translation_repetition_penalty"] = max(
            1.05,
            _safe_float(settings.get("translation_repetition_penalty"), 1.05),
        )
        batch_method = getattr(self.runtime, "translate_text_batch", None)
        if callable(batch_method):
            kwargs = self._compatible_kwargs(batch_method, {"translation_settings": settings})
            call = functools.partial(batch_method, texts, language, **kwargs)
            timeout = max(5.0, _safe_float(getattr(self.settings, "post_meeting_translation_timeout_seconds", 180.0), 180.0))
            result = await self._invoke_model_call(
                call,
                timeout=timeout,
                executor_name="translation_executor",
            )
            values = list(result or [])
            if len(values) != len(texts):
                raise RuntimeError("post-translation batch returned an unexpected result count")
            return values

        # Compatibility fallback for test/runtime adapters that only expose
        # the original single-text method.
        values: list[Any] = []
        for text in texts:
            values.append(await self._translate(text, language, final=True))
        return values

    async def run_post_meeting_translation(self, *, force: bool = False) -> bool:
        """Run the quality-filtered final translation pass once per meeting."""

        enabled = bool(getattr(self.settings, "post_meeting_translation_enabled", False))
        if not enabled and not force:
            self.post_translation_state = "skipped"
            self._write_state()
            return False
        if self.post_translation_state == "running":
            return False
        if self.post_translation_state == "complete" and not force:
            return False

        self.post_translation_state = "running"
        self.post_translation_error = None
        self._write_state()
        started = time.perf_counter()
        try:
            items = sorted(self.load_transcript(), key=lambda value: (value.start, value.end, value.id))
            candidates, target_count = self._post_translation_candidates(items)
            self.pipeline_metrics["post_translation_candidates"] = int(
                self.pipeline_metrics.get("post_translation_candidates", 0)
            ) + len(candidates)
            self.pipeline_metrics["post_translation_skipped"] = int(
                self.pipeline_metrics.get("post_translation_skipped", 0)
            ) + max(0, target_count - len(candidates))
            if not candidates:
                self.post_translation_state = "complete"
                self._record_metric("post_translation_duration_ms", (time.perf_counter() - started) * 1000)
                self._write_state()
                return False
            await self.status("会议结束，正在进行质量复核翻译", post_translation=True, candidate_count=len(candidates))

            groups: dict[str, list[tuple[Utterance, AsrQualityAssessment, str, int]]] = {}
            for candidate in candidates:
                groups.setdefault(self._translation_language(candidate[0].language, candidate[0].text), []).append(candidate)

            changed = 0
            failures = 0
            stale = 0
            for language, group in groups.items():
                texts = [item.text.strip() for item, _quality, _context, _source_revision in group]
                contexts = [context for _item, _quality, context, _source_revision in group]
                try:
                    results = await self._translate_post_batch(texts, language, contexts)
                except Exception as exc:  # noqa: BLE001
                    failures += len(group)
                    self.post_translation_error = str(exc)
                    LOGGER.warning("Post-meeting translation failed for %s: %s", self.id, exc)
                    continue
                for (item, _quality, _context, source_revision), result in zip(group, results):
                    current = next((value for value in self.paragraphs if value.segment_id == item.segment_id), None)
                    if current is None or current.source_revision != source_revision:
                        stale += 1
                        self.pipeline_metrics["stale_translation_results"] = int(
                            self.pipeline_metrics.get("stale_translation_results", 0)
                        ) + 1
                        continue
                    status = str(getattr(result, "status", "failed"))
                    translated = str(getattr(result, "text", "") or "").strip()
                    if status not in {"ready", "not_needed"} or not translated:
                        failures += 1
                        continue
                    if translated == current.translation_zh and current.translation_status == "ready":
                        continue
                    current.translation_zh = translated
                    current.translation_status = "ready" if status == "ready" else status
                    current.translation_model = getattr(result, "model", None) or current.translation_model
                    self.translation_errors.pop(current.segment_id, None)
                    current.revision += 1
                    self._upsert_recent(current)
                    self._persist_paragraph(current)
                    await self.broadcast("paragraph_update", paragraph=current.to_dict(), transcript_revision=self.transcript_revision)
                    changed += 1

            self.pipeline_metrics["post_translation_retranslated"] = int(
                self.pipeline_metrics.get("post_translation_retranslated", 0)
            ) + changed
            self.pipeline_metrics["post_translation_failures"] = int(
                self.pipeline_metrics.get("post_translation_failures", 0)
            ) + failures
            if failures and not changed and not stale:
                self.post_translation_state = "error"
            else:
                self.post_translation_state = "complete"
            self._record_metric("post_translation_duration_ms", (time.perf_counter() - started) * 1000)
            if self.recording_state == "complete":
                self._export_current_files()
            self._write_state()
            await self.status(
                "会后翻译完成" if self.post_translation_state == "complete" else "会后翻译部分失败",
                post_translation=True,
                retranslated=changed,
                failures=failures,
            )
            return bool(changed)
        except asyncio.CancelledError:
            self.post_translation_state = "error"
            self.post_translation_error = "post-meeting translation cancelled"
            self._write_state()
            raise
        except Exception as exc:  # noqa: BLE001
            self.post_translation_state = "error"
            self.post_translation_error = str(exc)
            self.pipeline_metrics["post_translation_failures"] = int(
                self.pipeline_metrics.get("post_translation_failures", 0)
            ) + 1
            self._record_metric("post_translation_duration_ms", (time.perf_counter() - started) * 1000)
            self._write_state()
            LOGGER.exception("Post-meeting translation failed for %s", self.id)
            return False

    async def retry_translation(self) -> bool:
        if self.recording_state not in {"complete", "recording", "finalizing"}:
            return False
        queued = False
        for item in list(self.paragraphs):
            if not self._needs_translation(item.language, item.text) or not item.text.strip():
                continue
            self.translation_errors.pop(item.segment_id, None)
            await self._schedule_translation(item, final=True)
            queued = True
        self._write_state()
        return queued

    async def _wait_for_queue(
        self,
        queue: Any,
        worker: asyncio.Task[Any] | None,
        *,
        label: str,
    ) -> None:
        try:
            await asyncio.wait_for(
                queue.join(),
                timeout=max(1.0, self.settings.queue_join_timeout_seconds),
            )
            return
        except asyncio.TimeoutError:
            LOGGER.error("%s queue did not drain before finalization for %s", label, self.id)
        if worker and not worker.done():
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker
        abort = getattr(queue, "abort", None)
        if callable(abort):
            abort()
            if label == "translation":
                self._translation_pending = 0
                self._translation_enqueue_times.clear()
            return
        while True:
            try:
                queued = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                if label == "translation" and isinstance(queued, dict):
                    self._translation_pending = max(0, self._translation_pending - 1)
                try:
                    queue.task_done(queued)
                except TypeError:
                    queue.task_done()

    async def request_stop(self, reason: str = "user") -> None:
        if self.recording_state in {"created", "complete", "error"}:
            return
        self._audio_stop_requested = True
        if self.stop_task is None or self.stop_task.done():
            self.stop_task = asyncio.create_task(self._finalize(reason), name=f"finalize-{self.id}")
        await asyncio.sleep(0)

    async def _finalize(self, reason: str) -> None:
        try:
            if not await self.wait_for_audio_drain(self.settings.audio_drain_timeout_seconds):
                LOGGER.warning("Audio ingress did not drain before finalization for %s", self.id)
            self.recording_state = "finalizing"
            self._write_state()
            if self.segmenter:
                for event in self.segmenter.flush():
                    await self._enqueue_event(event)
            await self._wait_for_queue(self.queue, self.worker_task, label="ASR")
            if self.worker_task and not self.worker_task.done():
                self.worker_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.worker_task
            await self._close_active(self.audio_samples_received / SAMPLE_RATE)
            await self._wait_for_queue(self.translation_queue, self.translation_worker_task, label="translation")
            if self.translation_worker_task and not self.translation_worker_task.done():
                self.translation_worker_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.translation_worker_task
            if self.audio_writer:
                self.audio_segments = await asyncio.to_thread(self.audio_writer.close)
            await self.run_post_meeting_translation()
            self.ended_at = utc_now_iso()
            self.recording_state = "complete"
            self.todo_state = "waiting_summary"
            self._export_current_files()
            self._delete_audio_if_unretained()
            self._export_current_files()
            self._write_state()
            await self.broadcast("recording_complete", meeting=self.snapshot(), reason=reason)
            await self.status("录音已完成，可生成会议纪要")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = str(exc)
            self.recording_state = "error"
            self.ended_at = self.ended_at or utc_now_iso()
            for task in (self.worker_task, self.translation_worker_task):
                if task and not task.done():
                    task.cancel()
            self._write_state()
            await self.broadcast("error", code="finalize_failed", message=self.error, retryable=False)

    def _export_current_files(self) -> None:
        snapshot = getattr(self.runtime, "capability_snapshot", None)
        if callable(snapshot):
            with suppress(Exception):
                self.model_metadata = snapshot()
        self.files = export_live_result(
            self.output_dir,
            meeting_id=self.id,
            title=self.title,
            started_at=self.started_at,
            ended_at=self.ended_at or utc_now_iso(),
            duration_seconds=self.audio_samples_received / SAMPLE_RATE,
            utterances=self.load_transcript(),
            audio_segments=self.audio_segments,
            recording_state=self.recording_state,
            summary_state=self.summary_state,
            todo_state=self.todo_state,
            summary_error=self.summary_error,
            todo_error=self.todo_error,
            model_metadata=self.model_metadata,
        )

    def _delete_audio_if_unretained(self) -> None:
        if self.meeting_settings.get("keep_audio", self.settings.keep_audio):
            return
        audio_dir = (self.output_dir / "audio").resolve()
        audio_dir.relative_to(self.output_dir.resolve())
        if audio_dir.exists():
            shutil.rmtree(audio_dir)
        self.audio_segments = []

    def begin_summary(self) -> bool:
        if self.recording_state != "complete" or self.translation_pending:
            return False
        if self.summary_state not in {"idle", "error", "complete"}:
            return False
        self.summary_state = "running"
        self.summary_error = None
        self.todo_state = "stale" if self.summary_revision else "waiting_summary"
        self.todo_error = None
        self._write_state()
        return True

    async def request_summary(self) -> bool:
        if not self.begin_summary():
            return False
        if self.todo_task and not self.todo_task.done():
            self.todo_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.todo_task
        if self.summary_task and not self.summary_task.done():
            return True
        self.summary_task = asyncio.create_task(self.run_summary(True), name=f"summary-{self.id}")
        return True

    async def run_summary(self, claimed: bool = False) -> None:
        if not claimed and self.summary_state != "queued":
            return
        if not claimed and not self.begin_summary():
            return
        try:
            await self.status("正在生成会议纪要")
            summarizer = self.summarizer_factory(self.settings)
            candidate = ""

            async def on_status(kind: str, index: int, total: int) -> None:
                await self.broadcast("summary_progress", phase=kind, current=index, total=total)

            async def on_delta(content: str) -> None:
                nonlocal candidate
                candidate += content
                await self.broadcast("summary_delta", content=content)

            async def on_reset() -> None:
                nonlocal candidate
                candidate = ""
                await self.broadcast("summary_reset")

            result = await summarizer.summarize(
                self.transcript_path,
                self.id,
                self.started_at,
                self.ended_at or utc_now_iso(),
                on_status=on_status,
                on_delta=on_delta,
                on_reset=on_reset,
                attempt_id=f"{self.id}:{self.summary_revision + 1}:{uuid.uuid4().hex}",
            )
            self.summary = str(result or candidate).strip()
            self.summary_revision += 1
            self.summary_state = "complete"
            self.todo_state = "queued"
            self.summary_error = None
            atomic_write_text(self.output_dir / "meeting_minutes.md", self.summary + "\n")
            self._export_current_files()
            self._write_state()
            await self.broadcast("summary_complete", content=self.summary, summary_revision=self.summary_revision, files=self.files)
            self.todo_task = asyncio.create_task(self.run_todo(True), name=f"todo-{self.id}-{self.summary_revision}")
        except Exception as exc:
            self.summary_state = "error"
            self.summary_error = str(exc)
            self._write_state()
            await self.broadcast("error", code="summary_failed", message=self.summary_error, retryable=True, summary=self.summary, summary_revision=self.summary_revision)

    def begin_todo(self) -> bool:
        if self.recording_state != "complete" or self.summary_state != "complete" or not self.summary.strip():
            return False
        if self.todo_state not in {"queued", "error", "stale", "complete"}:
            return False
        self.todo_state = "running"
        self.todo_error = None
        self._write_state()
        return True

    async def request_todo(self) -> bool:
        if not self.begin_todo():
            return False
        if self.todo_task and not self.todo_task.done():
            return True
        self.todo_task = asyncio.create_task(self.run_todo(True), name=f"todo-{self.id}-{self.summary_revision}")
        return True

    async def run_todo(self, claimed: bool = False) -> None:
        if not claimed and self.todo_state != "queued":
            return
        if not claimed and not self.begin_todo():
            return
        target_revision, target_summary = self.summary_revision, self.summary
        try:
            generator = self.todo_factory(self.settings)
            todo = await generator.generate(self.id, target_revision, target_summary, on_status=lambda phase: self.broadcast("todo_progress", phase=phase, summary_revision=target_revision))
            if target_revision != self.summary_revision or target_summary != self.summary:
                return
            self.todo = todo
            self.todo_state = "complete"
            self.todo_error = None
            atomic_write_text(self.output_dir / "todo_list.json", json.dumps(todo.to_dict(), ensure_ascii=False, indent=2))
            atomic_write_text(self.output_dir / "todo_list.md", render_todo_markdown(todo))
            self._export_current_files()
            self._write_state()
            await self.broadcast("todo_complete", todo=todo.to_dict(), files=self.files, summary_revision=self.summary_revision)
        except Exception as exc:
            self.todo_state = "error"
            self.todo_error = str(exc)
            self._write_state()
            await self.broadcast("error", code="todo_failed", message=self.todo_error, retryable=True)


class SessionManager:
    def __init__(self, settings: Settings, runtime: Any, store: LocalMeetingStore | None = None) -> None:
        self.settings = settings
        self.runtime = runtime
        self.store = store or LocalMeetingStore(settings.results_dir)
        self.sessions: dict[str, LiveMeetingSession] = {}
        self._create_lock = asyncio.Lock()
        self._shutting_down = False
        self._load_recovered()

    def _load_recovered(self) -> None:
        for payload in self.store.list_states():
            try:
                meeting_id = str(payload.get("id", ""))
                if not meeting_id:
                    continue
                if payload.get("recording_state") in {"starting", "recording", "finalizing"}:
                    payload["recording_state"] = "complete"
                    payload["ended_at"] = payload.get("ended_at") or utc_now_iso()
                self.sessions[meeting_id] = LiveMeetingSession(
                    self.settings,
                    self.runtime,
                    self.store,
                    meeting_id=meeting_id,
                    recovered_state=payload,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Skipping invalid recovered meeting state: %s", exc)

    def active_count(self) -> int:
        return sum(session.active for session in self.sessions.values())

    def get(self, meeting_id: str) -> LiveMeetingSession | None:
        return self.sessions.get(meeting_id)

    def list(self) -> list[dict[str, Any]]:
        return [session.snapshot() for session in sorted(self.sessions.values(), key=lambda item: item.started_at, reverse=True)]

    async def create(self, title: str = "") -> LiveMeetingSession:
        async with self._create_lock:
            if self.active_count() >= self.settings.max_active_meetings:
                raise CapacityLimitError("当前已达到实时会议并发上限")
            session = LiveMeetingSession(self.settings, self.runtime, self.store, title=title or "未命名会议")
            self.sessions[session.id] = session
            try:
                session._write_state()
            except Exception:
                self.sessions.pop(session.id, None)
                self.store.delete(session.id)
                raise
            return session

    async def start(self, meeting_id: str) -> LiveMeetingSession:
        async with self._create_lock:
            session = self.sessions.get(meeting_id)
            if session is None:
                raise KeyError(meeting_id)
            if session.recording_state in {"starting", "recording"}:
                return session
            if session.recording_state != "created":
                raise ValueError("当前会议不可开始录音")
            if self.active_count() >= self.settings.max_active_meetings:
                raise CapacityLimitError("当前已达到实时会议并发上限")
            await session.start()
            return session

    def begin_shutdown(self) -> None:
        self._shutting_down = True

    async def delete(self, meeting_id: str) -> bool:
        async with self._create_lock:
            session = self.sessions.get(meeting_id)
            if session and (session.active or session.has_active_tasks):
                raise ValueError("会议或后台任务进行中，不能删除")
            try:
                exists_on_disk = self.store.meeting_dir(meeting_id).exists()
            except ValueError:
                return False
            if meeting_id not in self.sessions and not exists_on_disk:
                return False
            self.sessions.pop(meeting_id, None)
            self.store.delete(meeting_id)
            return True

    async def resume_pending(self, *, model_tasks_ready: bool = True) -> None:
        if not model_tasks_ready or self._shutting_down:
            return
        # A process can exit after final ASR but before the post-meeting pass
        # writes its completed state.  Resume only that explicitly persisted
        # pass; ordinary failed live translations still require the existing
        # retry endpoint.
        for session in self.sessions.values():
            if session.recording_state != "complete" or session.post_translation_state not in {"idle", "running"}:
                continue
            if session.post_translation_task and not session.post_translation_task.done():
                continue

            async def run_post_pass(value: LiveMeetingSession = session) -> None:
                try:
                    await value.run_post_meeting_translation(force=True)
                finally:
                    value.post_translation_task = None

            session.post_translation_task = asyncio.create_task(
                run_post_pass(),
                name=f"post-translation-recovery-{session.id}",
            )

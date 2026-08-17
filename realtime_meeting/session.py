from __future__ import annotations

import asyncio
import inspect
import json
import logging
import shutil
import time
import uuid
from collections import deque
from contextlib import suppress
from dataclasses import replace
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
from .config import Settings, normalize_meeting_settings
from .exporter import export_live_result, render_todo_markdown
from .jimo import MeetingSummarizer, TodoGenerator
from .language import LanguageGuess, is_mixed_source_text, language_key, normalize_qwen_label
from .models import TodoDocument, Utterance, utc_now_iso
from .storage import LocalMeetingStore, TranscriptStore, atomic_write_json, atomic_write_text
from .text_normalize import simplify_chinese


LOGGER = logging.getLogger(__name__)
PARTIAL_ASR_WINDOW_SECONDS = 6.0
STATE_PERSIST_INTERVAL_SECONDS = 2.0


class CapacityLimitError(RuntimeError):
    pass


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
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
        self._active_technical_revision: int | None = None
        # A paragraph may span several technical audio chunks.  Keep the
        # text committed before the current chunk separate from the current
        # chunk's replaceable ASR hypothesis.
        self._active_technical_base: str | None = None
        self._forced_chunks_for_active = 0
        self._last_state_persist_at = time.monotonic()
        self._previous_partial_text: dict[str, str] = {}
        self._stable_prefixes: dict[str, str] = {}

        self.clients: set[WebSocket] = set()
        self.queue: asyncio.Queue[SegmentEvent | None] = asyncio.Queue(maxsize=max(1, settings.inference_queue_size))
        self.translation_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=max(1, settings.translation_queue_size))
        self.worker_task: asyncio.Task[Any] | None = None
        self.translation_worker_task: asyncio.Task[Any] | None = None
        self.stop_task: asyncio.Task[Any] | None = None
        self.disconnect_stop_task: asyncio.Task[Any] | None = None
        self.summary_task: asyncio.Task[Any] | None = None
        self.todo_task: asyncio.Task[Any] | None = None
        self._translation_pending = 0
        self._translation_keys: set[tuple[str, int, bool]] = set()
        self.translation_errors: dict[str, str] = {}
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
            "transcript_revision",
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
                value = _safe_float(value, self.volume_threshold_percent)
            if name == "files" and not isinstance(value, list):
                value = []
            if name == "model_metadata" and not isinstance(value, dict):
                value = {}
            setattr(self, name, value)
        settings = payload.get("meeting_settings")
        if isinstance(settings, dict):
            self.meeting_settings = normalize_meeting_settings(settings, self.settings)
        todo = payload.get("todo")
        if isinstance(todo, dict):
            try:
                self.todo = TodoDocument(
                    schema_version=str(todo.get("schema_version", "1.0")),
                    meeting_id=str(todo.get("meeting_id", self.id)),
                    summary_revision=_safe_int(todo.get("summary_revision")),
                    generated_at=str(todo.get("generated_at", "")),
                    items=[],
                )
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
        tasks = (self.worker_task, self.translation_worker_task, self.stop_task, self.summary_task, self.todo_task)
        return any(task is not None and not task.done() for task in tasks)

    @property
    def translation_pending(self) -> bool:
        return self._translation_pending > 0 or not self.translation_queue.empty()

    @property
    def translation_state(self) -> str:
        if self.translation_pending:
            return "pending"
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
            "paragraph_count": len(self.paragraphs),
            "files": self.files,
            "model_metadata": self.model_metadata,
        }

    def _write_state(self) -> None:
        atomic_write_json(self.state_path, self._state_payload())

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
        self.paragraphs = self.transcript_store.load()
        self.recent = deque(self.paragraphs[-500:], maxlen=500)
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
        if event.kind == "partial":
            try:
                self.queue.put_nowait(event)
            except asyncio.QueueFull:
                # Partials are replaceable hypotheses. Keep ingress moving and
                # retain the next final event for durable transcript state.
                return
        else:
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

    async def _worker(self) -> None:
        while True:
            event = await self.queue.get()
            try:
                if event is None:
                    return
                try:
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
        # One tiny-model call after the first ~0.8 s of each technical speech
        # chunk.  Forced max-duration chunks get their own check; partials do
        # not repeatedly invoke the small model.
        if event.revision in self._lid_checked_revisions:
            return False
        if not final and event.end - event.start < self.settings.language_id_min_seconds:
            return False
        self._lid_checked_revisions.add(event.revision)
        detector = getattr(self.runtime, "detect_language", None)
        if not detector:
            return False
        kwargs = self._compatible_kwargs(detector, {"previous_language": self.current_language})
        try:
            guess = await asyncio.wait_for(
                asyncio.to_thread(detector, event.pcm, **kwargs),
                timeout=max(1.0, self.settings.asr_timeout_seconds),
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Realtime language detection failed for %s", self.id)
            return False
        return self._accept_language(guess, final=final)

    def _accept_language(self, guess: LanguageGuess | None, *, final: bool) -> bool:
        if guess is None or guess.code not in {"zh", "en", "de"}:
            return False
        key = language_key(guess)
        if key[0] == "zh" and key[1] == "unknown":
            # Generic Chinese does not force a dialect boundary.
            key = ("zh", "unknown")
        old_stable = self._stable_key
        if self._stable_key and self._stable_key[0] == "zh" and key == ("zh", "unknown") and self._stable_key[1] != "unknown":
            # A generic Qwen ``Chinese`` result is not evidence that a known
            # dialect changed; preserve the last confirmed variant.
            self.current_language = "zh"
            self.current_variant = self._stable_key[1]
            self._candidate_key = None
            self._candidate_count = 0
            return False
        if self._stable_key is not None and _same_language_key(key, self._stable_key):
            self.current_language, self.current_variant = key[0], None if key[1] == "unknown" else key[1]
            self._candidate_key = key
            self._candidate_count = 0
            self._candidate_confidence = guess.confidence
            return False
        if self._candidate_key == key:
            self._candidate_count += 1
            self._candidate_confidence = max(self._candidate_confidence, guess.confidence)
        else:
            self._candidate_key = key
            self._candidate_count = 1
            self._candidate_confidence = guess.confidence
        if self._stable_key is None:
            confirmed = self._candidate_count >= 2 or final
        else:
            confirmed = self._candidate_count >= self.settings.language_conflict_confirmations or final
        if not confirmed:
            return False
        self._stable_key = key
        self.current_language, self.current_variant = key[0], None if key[1] == "unknown" else key[1]
        self._candidate_count = 0
        self._language_conflicts = 0
        return old_stable is not None and old_stable != key

    async def _handle_event(self, event: SegmentEvent) -> None:
        is_final = event.kind == "final" and not event.forced
        paragraph_start: float | None = None
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
            paragraph_start = max(event.start, self.active_paragraph.end) if self.active_paragraph else event.start
            await self._close_active(paragraph_start)

        # Do not force the previous paragraph language into Qwen while the
        # paragraph is still live. ``language=`` is a hard decode constraint in
        # the Qwen wrapper; keeping it here made a language switch look like
        # the previous language forever and prevented a paragraph boundary.
        # A language confirmed by the small LID for a new technical segment is
        # safe to pass through as a constraint.
        use_stable_language = self.active_paragraph is not None or language_changed
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
        if not is_final:
            keep_bytes = int(PARTIAL_ASR_WINDOW_SECONDS * SAMPLE_RATE) * SAMPLE_WIDTH
            if len(event.pcm) > keep_bytes:
                asr_event = replace(
                    event,
                    pcm=event.pcm[-keep_bytes:],
                    start=max(event.start, event.end - PARTIAL_ASR_WINDOW_SECONDS),
                )
        call_kwargs = self._compatible_kwargs(method, kwargs)
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(method, asr_event, **call_kwargs),
                timeout=max(1.0, self.settings.asr_timeout_seconds),
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Realtime ASR failed for %s", self.id)
            result = None
        if result is None:
            if is_final:
                await self._close_active(event.end)
            elif event.forced and self._forced_chunks_for_active >= 2:
                await self._close_active(event.end)
            return
        result_text = str(getattr(result, "text", "") or "").strip()
        result_guess = LanguageGuess(
            str(getattr(result, "language", "unknown") or "unknown"),
            _safe_float(getattr(result, "confidence", 0.0)),
            getattr(result, "speech_variant", None),
            str(getattr(result, "raw_qwen_label", "") or ""),
        )
        if result_guess.code not in {"zh", "en", "de"}:
            result_guess = normalize_qwen_label(getattr(result, "raw_qwen_label", ""), result_text)
        if not result_text:
            # Empty ASR output is not a new paragraph and must not refresh the
            # current paragraph with a model hallucination. A final empty
            # result still closes real audio already admitted by the segmenter.
            if is_final:
                await self._close_active(event.end)
            elif event.forced and self._forced_chunks_for_active >= 2:
                await self._close_active(event.end)
            return

        result_key = language_key(result_guess) if result_guess.code in {"zh", "en", "de"} else None
        active_key = (
            (self.active_paragraph.language, self.active_paragraph.speech_variant or "unknown")
            if self.active_paragraph is not None
            else None
        )
        expected_key = self._stable_key or active_key
        if result_key is not None and expected_key is not None:
            comparable_result_key = result_key
            if expected_key[0] == "zh" and result_key == ("zh", "unknown") and expected_key[1] != "unknown":
                # Generic Chinese is compatible with a previously confirmed
                # dialect; it is not a language boundary by itself.
                comparable_result_key = expected_key
            if comparable_result_key != expected_key:
                self._language_conflicts += 1
                confirmed = False
                if is_final or self._language_conflicts >= self.settings.language_conflict_confirmations:
                    # The secondary detector is useful when available, but a
                    # consistent ASR language result must be sufficient. If
                    # the detector is unavailable, requiring it leaves the
                    # old paragraph open indefinitely.
                    confirmed = await self._confirm_language_conflict(event, result_guess)
                    if not confirmed:
                        confirmed = self._accept_language(result_guess, final=True)
                if not confirmed:
                    # Never write a conflicting hypothesis into the old
                    # paragraph. That was the source of the disappearing
                    # translation when a language changed mid-stream.
                    if is_final:
                        await self._close_active(event.end)
                    return
                paragraph_start = max(event.start, self.active_paragraph.end) if self.active_paragraph else event.start
                await self._close_active(paragraph_start)
                language_changed = True
            else:
                self._language_conflicts = 0
                if self._stable_key is None:
                    self._accept_language(result_guess, final=is_final)
        elif result_key is not None and self._stable_key is None:
            self._accept_language(result_guess, final=is_final)

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
        )
        if is_final:
            await self._close_active(event.end)
        elif event.forced:
            if self._forced_chunks_for_active >= 2:
                await self._close_active(event.end)

    async def _confirm_language_conflict(self, event: SegmentEvent, guess: LanguageGuess) -> bool:
        detector = getattr(self.runtime, "detect_language", None)
        if not detector:
            return False
        kwargs = self._compatible_kwargs(detector, {"previous_language": self.current_language})
        try:
            confirmed = await asyncio.wait_for(
                asyncio.to_thread(detector, event.pcm, **kwargs),
                timeout=max(1.0, self.settings.asr_timeout_seconds),
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Realtime language conflict detection failed for %s", self.id)
            return False
        self._language_conflicts = 0
        # Two consecutive ASR conflicts are the trigger; the small model's
        # agreeing result is the confirmation that commits the new boundary.
        return bool(confirmed and language_key(confirmed) == language_key(guess) and self._accept_language(confirmed, final=True))

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
        return "en" if language == "zh" and is_mixed_source_text(text) else language

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
    ) -> None:
        if not text and self.active_paragraph is None:
            return
        key = (language, speech_variant or "unknown")
        if self.active_paragraph is not None:
            active_key = (self.active_paragraph.language, self.active_paragraph.speech_variant or "unknown")
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
        self._stable_prefixes[item.segment_id] = prefix
        if item.translation_status != "streaming":
            item.translation_status = "streaming"
            # Translation status is visible to the client. Give it a newer
            # revision so delayed websocket events cannot overwrite it.
            item.revision += 1
        self._upsert_recent(item)
        self._persist_paragraph(item)
        await self.broadcast("paragraph_update", paragraph=item.to_dict(), transcript_revision=self.transcript_revision)
        key = (item.segment_id, item.source_revision, final)
        if key in self._translation_keys:
            return
        self._translation_keys.add(key)
        self._translation_pending += 1
        job = {
            "segment_id": item.segment_id,
            "source_revision": item.source_revision,
            "text": prefix,
            "language": self._translation_language(item.language, prefix),
            "final": final,
            "attempt": 0,
        }
        try:
            if final:
                await asyncio.wait_for(
                    self.translation_queue.put(job),
                    timeout=max(1.0, self.settings.translation_timeout_seconds),
                )
            else:
                self.translation_queue.put_nowait(job)
        except asyncio.QueueFull:
            # A partial translation is only a hint; the next final paragraph
            # will enqueue the authoritative source text.
            self._translation_keys.discard(key)
            self._translation_pending = max(0, self._translation_pending - 1)
        except asyncio.TimeoutError:
            self._translation_keys.discard(key)
            self._translation_pending = max(0, self._translation_pending - 1)
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
                self.translation_queue.task_done()

    async def _run_translation_job(self, job: dict[str, Any]) -> None:
        key = (str(job["segment_id"]), _safe_int(job["source_revision"]), bool(job.get("final")))
        requeued = False
        try:
            result = await self._translate(job["text"], job["language"])
            item = next((value for value in self.paragraphs if value.segment_id == job["segment_id"]), None)
            if item is None or item.source_revision != _safe_int(job["source_revision"]):
                return
            status = str(getattr(result, "status", "failed"))
            if status in {"ready", "not_needed"}:
                item.translation_zh = str(getattr(result, "text", "") or "")
                item.translation_status = "ready" if status == "ready" else status
                item.translation_model = getattr(result, "model", None)
                self.translation_errors.pop(item.segment_id, None)
            else:
                attempt = _safe_int(job.get("attempt"))
                if attempt < 2:
                    retry = dict(job)
                    retry["attempt"] = attempt + 1
                    await asyncio.sleep(0.05 * (attempt + 1))
                    await self.translation_queue.put(retry)
                    requeued = True
                    return
                item.translation_status = "failed"
                self.translation_errors[item.segment_id] = str(getattr(result, "error", None) or status)
            item.revision += 1
            self._upsert_recent(item)
            self._persist_paragraph(item)
            await self.broadcast("paragraph_update", paragraph=item.to_dict(), transcript_revision=self.transcript_revision)
        except Exception as exc:
            item = next((value for value in self.paragraphs if value.segment_id == job["segment_id"]), None)
            attempt = _safe_int(job.get("attempt"))
            if item is not None and attempt < 2:
                retry = dict(job)
                retry["attempt"] = attempt + 1
                await self.translation_queue.put(retry)
                requeued = True
                return
            if item is not None and item.source_revision == _safe_int(job["source_revision"]):
                item.translation_status = "failed"
                item.revision += 1
                self.translation_errors[item.segment_id] = str(exc)
                self._upsert_recent(item)
                self._persist_paragraph(item)
                await self.broadcast("paragraph_update", paragraph=item.to_dict(), transcript_revision=self.transcript_revision)
        finally:
            if not requeued:
                self._translation_keys.discard(key)
                self._translation_pending = max(0, self._translation_pending - 1)
            with suppress(Exception):
                self._write_state()

    async def _translate(self, text: str, language: str) -> Any:
        method = getattr(self.runtime, "translate_text")
        kwargs = self._compatible_kwargs(method, {"translation_settings": self.meeting_settings})
        return await asyncio.wait_for(
            asyncio.to_thread(method, text, language, **kwargs),
            timeout=max(1.0, self.settings.translation_timeout_seconds),
        )

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
        queue: asyncio.Queue[Any],
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
        while True:
            try:
                queued = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                if label == "translation" and isinstance(queued, dict):
                    key = (
                        str(queued.get("segment_id", "")),
                        _safe_int(queued.get("source_revision")),
                        bool(queued.get("final")),
                    )
                    self._translation_keys.discard(key)
                    self._translation_pending = max(0, self._translation_pending - 1)
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
        session = self.sessions.get(meeting_id)
        if session and (session.active or session.has_active_tasks):
            raise ValueError("会议或后台任务进行中，不能删除")
        if meeting_id not in self.sessions and not self.store.meeting_dir(meeting_id).exists():
            return False
        self.sessions.pop(meeting_id, None)
        self.store.delete(meeting_id)
        return True

    async def resume_pending(self, *, model_tasks_ready: bool = True) -> None:
        del model_tasks_ready
        # Schema 2 only resumes persisted completed meetings.  Translation is
        # never silently re-run during recovery; the explicit retry endpoint is
        # the safe recovery mechanism for a failed translation.
        return

from __future__ import annotations

import asyncio
import json
import logging
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
    apply_volume_gate,
    decode_audio_pcm,
    rms_to_volume_threshold_percent,
    volume_threshold_percent_to_rms,
)
from .config import Settings
from .diarization import align_speakers, write_segments
from .exporter import append_utterance, delete_utterance, export_live_result, load_utterances, render_todo_markdown
from .jimo import MeetingSummarizer, TodoGenerator
from .models import TodoDocument, TodoItem, Utterance, utc_now_iso
from .runtime import PartialResult
from .scheduler import PostprocessTracker
from .storage import LocalMeetingStore, atomic_write_json, atomic_write_text


class CapacityLimitError(RuntimeError):
    pass


LOGGER = logging.getLogger(__name__)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


class LiveMeetingSession:
    def __init__(
        self,
        settings: Settings,
        runtime: Any,
        store: LocalMeetingStore,
        *,
        meeting_id: str | None = None,
        title: str = "",
        recovered_state: dict[str, Any] | None = None,
        summarizer_factory: Callable[[Settings], MeetingSummarizer] | None = None,
        todo_factory: Callable[[Settings], TodoGenerator] | None = None,
        postprocess_scheduler: Callable[[], None] | None = None,
    ) -> None:
        payload = recovered_state or {}
        self.settings = settings
        self.runtime = runtime
        self.store = store
        self.id = meeting_id or str(uuid.uuid4())
        self.title = title or str(payload.get("title", "未命名会议"))
        self.output_dir = store.meeting_dir(self.id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = str(payload.get("started_at") or utc_now_iso())
        self.ended_at = payload.get("ended_at")
        self.created_monotonic = time.monotonic()
        self.recording_state = str(payload.get("recording_state", "created"))
        self.summary_state = str(payload.get("summary_state", "idle"))
        self.todo_state = str(payload.get("todo_state", "waiting_summary"))
        self.summary_revision = _safe_int(payload.get("summary_revision"))
        self.snapshot_revision = _safe_int(payload.get("snapshot_revision"))
        self.postprocess = PostprocessTracker(payload.get("postprocess") if isinstance(payload.get("postprocess"), dict) else None)
        self.summary = ""
        self.todo: TodoDocument | None = None
        self.error: str | None = payload.get("error")
        self.summary_error: str | None = payload.get("summary_error")
        self.todo_error: str | None = payload.get("todo_error")
        self.files: list[str] = []
        self.transcript_path = self.output_dir / "transcript.jsonl"
        self.audio_writer: RotatingAudioWriter | None = None
        self.segmenter: StreamSegmenter | None = None
        self.audio_segments: list[dict[str, object]] = []
        self.clients: set[WebSocket] = set()
        self.recent: deque[Utterance] = deque(maxlen=500)
        self.utterance_count = 0
        self.next_utterance_id = 1
        self.current_language: str | None = None
        self.audio_sample_rate = 16_000
        self.audio_channels = 1
        self.audio_encoding = "pcm_s16le"
        self.audio_packet_ms = 40
        self.audio_bytes_received = 0
        self.audio_packets_received = 0
        self.audio_packets_dropped = 0
        self.audio_packets_out_of_order = 0
        self.audio_samples_received = 0
        self.audio_level = 0.0
        try:
            restored_threshold = float(payload.get("volume_threshold_percent", "nan"))
            self.volume_threshold_percent = rms_to_volume_threshold_percent(
                volume_threshold_percent_to_rms(restored_threshold)
            )
        except (TypeError, ValueError):
            self.volume_threshold_percent = rms_to_volume_threshold_percent(settings.vad_minimum_rms)
        self.volume_threshold_rms = volume_threshold_percent_to_rms(self.volume_threshold_percent)
        self._last_sequence: int | None = None
        self._lock = asyncio.Lock()
        self.queue: asyncio.Queue[SegmentEvent | None] = asyncio.Queue(maxsize=settings.inference_queue_size)
        self.worker_task: asyncio.Task[None] | None = None
        self.stop_task: asyncio.Task[None] | None = None
        self.disconnect_stop_task: asyncio.Task[None] | None = None
        self.summary_task: asyncio.Task[None] | None = None
        self.todo_task: asyncio.Task[None] | None = None
        self.postprocess_task: asyncio.Task[None] | None = None
        self.refinement_queue: asyncio.Queue[SegmentEvent | None] = asyncio.Queue(maxsize=settings.refinement_queue_size)
        self.refinement_worker_task: asyncio.Task[None] | None = None
        self._refinement_events: dict[int, SegmentEvent] = {}
        self.refinement_dir = self.output_dir / "refinement_events"
        self.refinement_dropped = 0
        self._segment_first_ids: dict[int, int] = {}
        self._segment_ids: dict[int, list[str]] = {}
        self.translation_tasks: set[asyncio.Task[None]] = set()
        self.model_metadata = getattr(runtime, "capability_snapshot", lambda: {})()
        self._postprocess_error: str | None = None
        self._refinement_errors: list[str] = []
        self._refinement_progress_current = 0
        self._refinement_progress_total = 0
        self.speaker_clusterer = None
        if getattr(runtime, "ready", False) and hasattr(runtime, "new_speaker_clusterer"):
            with suppress(Exception):
                self.speaker_clusterer = runtime.new_speaker_clusterer()
        self.summarizer_factory = summarizer_factory or MeetingSummarizer
        self.todo_factory = todo_factory or TodoGenerator
        self.postprocess_scheduler = postprocess_scheduler
        self._restore_files(payload)
        self._restore_refinement_events()
        if self.postprocess.state in {"running", "queued"} and self.recording_state == "complete":
            self.postprocess.state = "queued"

    def _restore_files(self, payload: dict[str, Any]) -> None:
        self.files = [
            path.name for path in self.output_dir.iterdir()
            if path.is_file() and path.name not in {"session_state.json", "session_state.json.tmp"}
        ] if self.output_dir.exists() else []
        minutes = self.output_dir / "meeting_minutes.md"
        if minutes.is_file():
            self.summary = minutes.read_text(encoding="utf-8")
        todo_path = self.output_dir / "todo_list.json"
        if todo_path.is_file():
            try:
                raw = json.loads(todo_path.read_text(encoding="utf-8"))
                todo_items: list[TodoItem] = []
                for raw_item in raw.get("items", []) if isinstance(raw, dict) else []:
                    if isinstance(raw_item, dict):
                        todo_items.append(TodoItem(**{
                            key: raw_item.get(key, default)
                            for key, default in {
                                "task": "", "owner": None, "due_date": None,
                                "priority": "待确认", "status": "未开始",
                                "source_time_start": None, "source_time_end": None,
                                "evidence": "", "notes": "", "id": "",
                                "meeting_id": self.id, "summary_revision": self.summary_revision,
                                "created_at": "",
                            }.items()
                        }))
                self.todo = TodoDocument(
                    schema_version=str(raw.get("schema_version", "1.0")),
                    items=todo_items,
                    meeting_id=str(raw.get("meeting_id", self.id)),
                    summary_revision=int(raw.get("summary_revision", self.summary_revision)),
                    generated_at=str(raw.get("generated_at", "")),
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self.todo = None
        items = load_utterances(self.transcript_path)
        self.recent.extend(items[-500:])
        self.utterance_count = len(items)
        self.next_utterance_id = max((item.id for item in items), default=0) + 1
        self.current_language = items[-1].language if items else None
        manifest = self.output_dir / "audio_manifest.json"
        if manifest.is_file():
            try:
                value = json.loads(manifest.read_text(encoding="utf-8"))
                self.audio_segments = list(value.get("segments", [])) if isinstance(value, dict) else []
                self.audio_samples_received = sum(int(item.get("samples", 0) or 0) for item in self.audio_segments)
                self.audio_bytes_received = self.audio_samples_received * 2
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

    def _persist_refinement_event(self, event: SegmentEvent) -> None:
        self.refinement_dir.mkdir(parents=True, exist_ok=True)
        pcm_path = self.refinement_dir / f"{event.revision}.pcm"
        temporary = pcm_path.with_suffix(".pcm.tmp")
        temporary.write_bytes(event.pcm)
        temporary.replace(pcm_path)
        atomic_write_json(self.refinement_dir / f"{event.revision}.json", {
            "revision": event.revision,
            "start": event.start,
            "end": event.end,
            "forced": event.forced,
            "pcm": pcm_path.name,
            "first_utterance_id": self._segment_first_ids.get(event.revision),
        })

    def _restore_refinement_events(self) -> None:
        if not self.refinement_dir.is_dir():
            return
        for metadata_path in sorted(self.refinement_dir.glob("*.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                pcm_path = self.refinement_dir / str(metadata["pcm"])
                event = SegmentEvent(
                    "final",
                    pcm_path.read_bytes(),
                    float(metadata["start"]),
                    float(metadata["end"]),
                    int(metadata["revision"]),
                    bool(metadata.get("forced", False)),
                )
                if event.pcm:
                    self._refinement_events[event.revision] = event
                    first_id = metadata.get("first_utterance_id")
                    if first_id is not None:
                        self._segment_first_ids[event.revision] = int(first_id)
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue

    def _rebuild_refinement_events_from_audio(self) -> int:
        """Recreate missing final-segment PCM from retained audio and transcript timestamps."""
        if self._refinement_events:
            return len(self._refinement_events)
        paths = [
            self.output_dir / "audio" / str(segment.get("file"))
            for segment in self.audio_segments
            if isinstance(segment, dict) and segment.get("file")
        ]
        paths = [path for path in paths if path.is_file()]
        if not paths:
            return 0
        pcm = b"".join(decode_audio_pcm(path) for path in paths)
        items = load_utterances(self.transcript_path)
        grouped: dict[str, list[Utterance]] = {}
        for item in items:
            grouped.setdefault(item.source_segment_id or item.segment_id.split(":", 1)[0], []).append(item)
        next_revision = 1
        for source_id, group in sorted(grouped.items(), key=lambda pair: min(item.start for item in pair[1])):
            try:
                revision = int(source_id)
            except ValueError:
                revision = next_revision
            while revision in self._refinement_events:
                revision += 1
            next_revision = max(next_revision, revision + 1)
            start = min(item.start for item in group)
            end = max(item.end for item in group)
            first = max(0, round(start * SAMPLE_RATE) * SAMPLE_WIDTH)
            last = min(len(pcm), round(end * SAMPLE_RATE) * SAMPLE_WIDTH)
            if last <= first:
                continue
            event = SegmentEvent("final", pcm[first:last], start, end, revision)
            self._refinement_events[revision] = event
            self._segment_first_ids[revision] = min(item.id for item in group)
            self._persist_refinement_event(event)
        return len(self._refinement_events)

    def _clear_refinement_events(self) -> None:
        self._refinement_events.clear()
        if self.refinement_dir.is_dir():
            shutil.rmtree(self.refinement_dir)

    @property
    def elapsed_seconds(self) -> float:
        if self.recording_state == "created" and not self.ended_at:
            return 0.0
        if self.ended_at and self.audio_samples_received:
            return self.audio_samples_received / self.audio_sample_rate
        return round(time.monotonic() - self.created_monotonic, 3)

    @property
    def active(self) -> bool:
        return self.recording_state in {"starting", "recording", "finalizing"}

    @property
    def has_active_tasks(self) -> bool:
        tasks = (
            self.worker_task,
            self.stop_task,
            self.disconnect_stop_task,
            self.refinement_worker_task,
            self.postprocess_task,
            self.summary_task,
            self.todo_task,
            *self.translation_tasks,
        )
        return any(task is not None and not task.done() for task in tasks)

    async def add_client(self, websocket: WebSocket) -> None:
        if self.disconnect_stop_task and not self.disconnect_stop_task.done():
            self.disconnect_stop_task.cancel()
        self.disconnect_stop_task = None
        self.clients.add(websocket)

    def remove_client(self, websocket: WebSocket) -> None:
        self.clients.discard(websocket)

    def schedule_disconnect_stop(self) -> None:
        if self.clients or self.recording_state != "recording":
            return
        if self.disconnect_stop_task and not self.disconnect_stop_task.done():
            return
        self.disconnect_stop_task = asyncio.create_task(
            self._stop_after_disconnect(), name=f"disconnect-stop-{self.id}"
        )

    async def _stop_after_disconnect(self) -> None:
        try:
            await asyncio.sleep(self.settings.websocket_disconnect_grace_seconds)
            if not self.clients and self.recording_state == "recording":
                await self.request_stop("websocket_disconnect")
        except asyncio.CancelledError:
            return

    async def broadcast(self, event_type: str, **payload: Any) -> None:
        message = {"type": event_type, **payload}
        clients = tuple(self.clients)

        async def send(client: WebSocket) -> WebSocket | None:
            try:
                await asyncio.wait_for(client.send_json(message), timeout=2.0)
            except Exception:
                return client
            return None

        if not clients:
            return
        stale = await asyncio.gather(*(send(client) for client in clients))
        for client in stale:
            if client is not None:
                self.clients.discard(client)

    async def status(self, message: str, **payload: Any) -> None:
        await self.broadcast("meeting_state", meeting=self.snapshot(), message=message, **payload)

    def _write_state(self) -> None:
        self.snapshot_revision += 1
        atomic_write_json(self.output_dir / "session_state.json", self._state_payload())

    def _state_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "recording_state": self.recording_state,
            "summary_state": self.summary_state,
            "todo_state": self.todo_state,
            "summary_revision": self.summary_revision,
            "snapshot_revision": self.snapshot_revision,
            "postprocess": self.postprocess.to_dict(),
            "error": self.error,
            "summary_error": self.summary_error,
            "todo_error": self.todo_error,
            "audio_bytes_received": self.audio_bytes_received,
            "audio_packets_received": self.audio_packets_received,
            "audio_packets_dropped": self.audio_packets_dropped,
            "audio_packets_out_of_order": self.audio_packets_out_of_order,
            "audio_samples_received": self.audio_samples_received,
            "volume_threshold_percent": self.volume_threshold_percent,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": round(self.elapsed_seconds, 3),
            "recording_state": self.recording_state,
            "summary_state": self.summary_state,
            "todo_state": self.todo_state,
            "snapshot_revision": self.snapshot_revision,
            "postprocess": self.postprocess.to_dict(),
            "summary_revision": self.summary_revision,
            "summary": self.summary or None,
            "todo": self.todo.to_dict() if self.todo else None,
            "utterance_count": self.utterance_count,
            "recent_utterances": [item.to_dict() for item in self.recent],
            "current_language": self.current_language,
            "files": sorted(self.files),
            "error": self.error or self.summary_error or self.todo_error,
            "audio_bytes_received": self.audio_bytes_received,
            "audio_packets_received": self.audio_packets_received,
            "audio_packets_dropped": self.audio_packets_dropped,
            "audio_packets_out_of_order": self.audio_packets_out_of_order,
            "audio_samples_received": self.audio_samples_received,
            "audio_level": self.audio_level,
            "volume_threshold_percent": self.volume_threshold_percent,
            "refinement_queue_size": self.refinement_queue.qsize(),
            "refinement_dropped": self.refinement_dropped,
            "model_metadata": self.model_metadata,
        }

    def load_transcript(self) -> list[Utterance]:
        return sorted(load_utterances(self.transcript_path), key=lambda item: (item.start, item.end, item.id))

    def _recent_asr_context(self, language: str | None = None) -> str:
        """Return a small rolling prompt without leaking the whole transcript."""
        items = [item for item in self.recent if item.text and item.text.strip()]
        if language:
            same_language = [item for item in items if item.language == language]
            if same_language:
                items = same_language
        return " ".join(item.text.strip() for item in items[-3:])[-500:]

    async def start(self) -> None:
        if self.recording_state not in {"created", "starting", "recording"}:
            return
        if self.recording_state == "created":
            self.started_at = utc_now_iso()
            self.created_monotonic = time.monotonic()
        self.audio_writer = RotatingAudioWriter(self.output_dir / "audio", self.settings.audio_segment_minutes)
        vad = self.runtime.new_vad() if hasattr(self.runtime, "new_vad") else None
        self.segmenter = StreamSegmenter(
            pre_roll_ms=self.settings.audio_pre_roll_ms,
            speech_start_ms=self.settings.speech_start_ms,
            silence_ms=self.settings.silence_ms,
            minimum_rms=self.settings.vad_minimum_rms,
            minimum_speech_ms=self.settings.vad_minimum_speech_ms,
            minimum_speech_ratio=self.settings.vad_minimum_speech_ratio,
            partial_interval_ms=self.settings.partial_interval_ms,
            max_utterance_ms=int(self.settings.max_utterance_seconds * 1000),
            vad=vad,
        )
        self.segmenter.minimum_rms = self.volume_threshold_rms
        self.recording_state = "recording"
        self.error = None
        self._write_state()
        self.worker_task = asyncio.create_task(self._worker(), name=f"meeting-worker-{self.id}")
        self.refinement_worker_task = None
        await self.status("正在录音")

    def configure_volume_threshold(self, value: Any) -> None:
        try:
            threshold_percent = float(value)
            threshold_rms = volume_threshold_percent_to_rms(threshold_percent)
        except (TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from exc
        normalized_percent = round(threshold_percent, 1)
        changed = normalized_percent != self.volume_threshold_percent
        self.volume_threshold_percent = normalized_percent
        self.volume_threshold_rms = threshold_rms
        if self.segmenter:
            self.segmenter.minimum_rms = threshold_rms
        if changed and self.recording_state in {"created", "starting", "recording"}:
            self._write_state()

    def configure_audio(self, payload: dict[str, Any]) -> None:
        sample_rate = int(payload.get("sample_rate", 0) or 0)
        channels = int(payload.get("channels", 0) or 0)
        encoding = str(payload.get("encoding", ""))
        if sample_rate != 16_000 or channels != 1 or encoding != "pcm_s16le":
            raise ValueError("音频必须是 16000 Hz、单声道 PCM16")
        packet_ms = int(payload.get("packet_ms", 40) or 40)
        if not 1 <= packet_ms <= 1000:
            raise ValueError("音频包时长必须在 1 到 1000 毫秒之间")
        if "volume_threshold_percent" in payload:
            self.configure_volume_threshold(payload["volume_threshold_percent"])
        self.audio_packet_ms = packet_ms
        # Sequence numbers belong to one websocket transport. A browser reload
        # starts a new sequence at zero, while reconnecting the same worklet can
        # continue from an arbitrary value.
        self._last_sequence = None

    async def feed_audio(self, pcm: bytes, sequence: int | None = None) -> None:
        warning: str | None = None
        async with self._lock:
            # Recheck the state after acquiring the ingest lock. This prevents
            # an in-flight writer from enqueueing a segment after finalization
            # has already drained and stopped the worker.
            if self.recording_state != "recording":
                raise ValueError("会议当前不在录音状态")
            if not pcm or len(pcm) > self.settings.max_audio_packet_bytes:
                raise ValueError("音频包为空或超过大小限制")
            if len(pcm) % 2:
                raise ValueError("PCM 音频包长度必须为偶数")
            if sequence is not None:
                sequence &= 0xFFFFFFFF
                if self._last_sequence is None:
                    self._last_sequence = sequence
                else:
                    delta = (sequence - self._last_sequence) & 0xFFFFFFFF
                    if delta == 0 or delta > 0x7FFFFFFF:
                        self.audio_packets_out_of_order += 1
                    else:
                        if delta > 1:
                            self.audio_packets_dropped += delta - 1
                        self._last_sequence = sequence
            self.audio_packets_received += 1
            self.audio_bytes_received += len(pcm)
            self.audio_samples_received += len(pcm) // 2
            try:
                import numpy as np

                samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
                self.audio_level = min(1.0, float(np.sqrt(np.mean(samples * samples))) / 32768.0 * 3.0) if len(samples) else 0.0
            except Exception:
                self.audio_level = 0.0
            gated_pcm = apply_volume_gate(pcm, self.volume_threshold_rms)
            if self.audio_writer:
                await asyncio.to_thread(self.audio_writer.write, gated_pcm)
            if self.segmenter:
                for event in self.segmenter.feed(gated_pcm):
                    try:
                        self.queue.put_nowait(event)
                    except asyncio.QueueFull:
                        self.error = "实时推理队列已满，已丢弃一个语音片段"
                        warning = self.error
            packets_received = self.audio_packets_received
            audio_level = self.audio_level
        if warning:
            await self.broadcast("warning", message=warning)
        if packets_received % 10 == 0:
            await self.broadcast("audio_input", packets_received=packets_received, audio_level=audio_level)

    async def _worker(self) -> None:
        while True:
            event = await self.queue.get()
            try:
                if event is None:
                    return
                if event.kind == "partial":
                    await self._handle_partial(event)
                else:
                    await self._handle_final(event)
            except Exception as exc:
                self.error = str(exc)
                await self.broadcast("error", code="transcription_failed", message=self.error, retryable=True)
            finally:
                self.queue.task_done()

    async def _handle_partial(self, event: SegmentEvent) -> None:
        result: PartialResult = await asyncio.to_thread(
            self.runtime.transcribe_partial,
            event,
            self._recent_asr_context(self.current_language),
            self.current_language,
        )
        if result.text:
            await self.broadcast("draft", revision=event.revision, start=event.start, end=event.end, text=result.text, language=result.language, confidence=result.confidence)

    async def _handle_final(self, event: SegmentEvent) -> None:
        items: list[Utterance] = await asyncio.to_thread(
            self.runtime.transcribe_final,
            event,
            next_id=self.next_utterance_id,
            previous_language=self.current_language,
            recent_text=self._recent_asr_context(self.current_language),
            speaker_clusterer=getattr(self, "speaker_clusterer", None),
            refined=False,
        )
        self._segment_first_ids[event.revision] = items[0].id if items else self.next_utterance_id
        for item in items:
            item.source_segment_id = item.source_segment_id or str(event.revision)
            self.next_utterance_id = max(self.next_utterance_id, item.id + 1)
            self.current_language = item.language
            self.utterance_count += 1
            self.recent.append(item)
            append_utterance(self.transcript_path, item)
            await self.broadcast("utterance", utterance=item.to_dict())
            if item.language != "zh" and item.text:
                task = asyncio.create_task(self._translate_item(item), name=f"translate-{self.id}-{item.id}")
                self.translation_tasks.add(task)
                task.add_done_callback(self.translation_tasks.discard)
        if self.settings.enable_refinement:
            try:
                await asyncio.to_thread(self._persist_refinement_event, event)
                self._refinement_events[event.revision] = event
                self.refinement_queue.put_nowait(event)
            except asyncio.QueueFull:
                self.refinement_dropped += 1
                await self.broadcast("warning", message="精修队列已满，本片段已持久化并将在停止后处理")

    async def _translate_item(self, item: Utterance) -> None:
        try:
            latest = next((existing for existing in reversed(self.recent) if existing.segment_id == item.segment_id), item)
            source_text = latest.text
            source_language = latest.language
            result = await asyncio.to_thread(self.runtime.translate_text, source_text, source_language)
            latest = next(
                (existing for existing in reversed(self.recent) if existing.segment_id == item.segment_id),
                None,
            ) or next(
                (existing for existing in load_utterances(self.transcript_path) if existing.segment_id == item.segment_id),
                item,
            )
            # Refinement can replace the text while translation is in flight.
            # Its replacement task will translate the new text; never attach a
            # stale translation to a different source sentence.
            if latest.text != source_text or latest.language != source_language:
                return
            latest.translation_zh = result.text if result.status in {"ready", "not_needed"} else ""
            latest.translation_status = result.status
            latest.translation_model = getattr(result, "model", None)
            latest.revision = max(2, latest.revision + 1)
            append_utterance(self.transcript_path, latest)
            for index, existing in enumerate(self.recent):
                if existing.segment_id == latest.segment_id:
                    self.recent[index] = latest
                    break
            await self.broadcast("translation_update", segment_id=latest.segment_id, revision=latest.revision, translation_zh=latest.translation_zh, translation_status=latest.translation_status, translation_model=latest.translation_model)
        except Exception as exc:
            await self.broadcast("warning", message=f"翻译失败：{exc}")

    async def _refinement_worker(self) -> None:
        while True:
            event = await self.refinement_queue.get()
            processed = event is not None
            try:
                if event is None:
                    return
                first_id = self._segment_first_ids.get(event.revision, self.next_utterance_id)
                refined_items = await asyncio.to_thread(
                    self.runtime.transcribe_final,
                    event,
                    next_id=first_id,
                    previous_language=self.current_language,
                    recent_text=self._recent_asr_context(self.current_language),
                    speaker_clusterer=getattr(self, "speaker_clusterer", None),
                    refined=True,
                )
                await self._commit_refined_items(refined_items)
            except Exception as exc:
                self._refinement_errors.append(str(exc))
                await self.broadcast("warning", message=f"停止后精修失败，已保留实时结果：{exc}")
            finally:
                self.refinement_queue.task_done()
                if processed:
                    self._refinement_progress_current += 1
                    with suppress(Exception):
                        await self._postprocess_update(
                            "asr_refine",
                            "running",
                            current=self._refinement_progress_current,
                            total=self._refinement_progress_total,
                        )

    async def _commit_refined_items(self, items: list[Utterance]) -> None:
        if not items:
            return
        source_segment_id = items[0].source_segment_id or items[0].segment_id.split(":", 1)[0]
        previous_items = [
            existing for existing in load_utterances(self.transcript_path)
            if (existing.source_segment_id or existing.segment_id.split(":", 1)[0]) == source_segment_id
        ]
        previous_by_segment = {existing.segment_id: existing for existing in previous_items}
        previous_items.sort(key=lambda item: (item.start, item.end, item.id))
        replacement_ids: set[int] = set()
        for index, item in enumerate(items):
            item.source_segment_id = source_segment_id
            previous = previous_by_segment.get(item.segment_id)
            if previous is None and index < len(previous_items):
                previous = previous_items[index]
            if previous is None:
                previous = next((existing for existing in reversed(self.recent) if existing.segment_id == item.segment_id), None)
            if previous:
                item.id = previous.id
                replacement_ids.add(previous.id)
            item.revision = max(2, item.revision)
            item.recognition_stage = "refined"
            item.translation_zh = ""
            item.translation_status = "not_needed" if item.language == "zh" else "pending"
            append_utterance(self.transcript_path, item)
            replaced = False
            for index, existing in enumerate(self.recent):
                if existing.segment_id == item.segment_id:
                    self.recent[index] = item
                    replaced = True
                    break
            if not replaced:
                self.recent.append(item)
                self.utterance_count += 1
            await self.broadcast("utterance", utterance=item.to_dict())
            if item.language != "zh" and item.text:
                task = asyncio.create_task(self._translate_item(item), name=f"refine-translate-{self.id}-{item.id}")
                self.translation_tasks.add(task)
                task.add_done_callback(self.translation_tasks.discard)
        for previous in previous_items:
            if previous.id not in replacement_ids:
                delete_utterance(self.transcript_path, previous)
                self.recent = deque((item for item in self.recent if item.id != previous.id), maxlen=500)
                self.utterance_count = max(0, self.utterance_count - 1)
                await self.broadcast(
                    "utterance_deleted",
                    segment_id=previous.segment_id,
                    utterance_id=previous.id,
                    revision=max(previous.revision + 1, 2),
                )

    async def request_stop(self, reason: str = "user") -> None:
        if self.recording_state == "created":
            return
        if self.stop_task is None or self.stop_task.done():
            self.stop_task = asyncio.create_task(self._finalize(reason), name=f"finalize-{self.id}")
        await asyncio.sleep(0)

    async def _postprocess_update(
        self,
        stage: str | None = None,
        state: str | None = None,
        *,
        current: int | None = None,
        total: int | None = None,
        error: str | None = None,
    ) -> None:
        if stage and state:
            self.postprocess.update(stage, state, current=current, total=total, error=error)
        self._write_state()
        await self.broadcast("postprocess_update", meeting=self.snapshot())

    def _set_postprocess_queue(self, has_items: bool) -> None:
        enabled = has_items and self.settings.enable_postprocess
        self.postprocess = PostprocessTracker({
            "state": "queued" if enabled else "partial",
            "current_stage": "asr_refine" if enabled else None,
            "stages": {
                "asr_refine": {"state": "queued" if enabled and self.settings.enable_refinement else "complete"},
                "diarization": {"state": "queued" if enabled else "complete"},
                "translation": {"state": "queued" if has_items else "complete"},
                "summary": {"state": "idle" if has_items else "error"},
                "todo": {"state": "idle"},
            },
        })

    def _export_current_files(self) -> None:
        items = load_utterances(self.transcript_path)
        snapshot = getattr(self.runtime, "capability_snapshot", None)
        if snapshot:
            with suppress(Exception):
                self.model_metadata = snapshot()
        self.files = export_live_result(
            self.output_dir,
            meeting_id=self.id,
            title=self.title,
            started_at=self.started_at,
            ended_at=self.ended_at or utc_now_iso(),
            duration_seconds=self.audio_samples_received / self.audio_sample_rate,
            utterances=items,
            audio_segments=self.audio_segments,
            recording_state=self.recording_state,
            summary_state=self.summary_state,
            todo_state=self.todo_state,
            summary_error=self.summary_error,
            todo_error=self.todo_error,
            postprocess=self.postprocess.to_dict(),
            model_metadata=self.model_metadata,
        )

    def _delete_audio_if_unretained(self) -> None:
        if self.settings.keep_audio:
            return
        audio_dir = (self.output_dir / "audio").resolve()
        audio_dir.relative_to(self.output_dir.resolve())
        if audio_dir.exists():
            shutil.rmtree(audio_dir)
        self.audio_segments = []

    async def _warm_realtime_after_postprocess(self) -> None:
        warm_realtime = getattr(self.runtime, "warm_realtime", None)
        if warm_realtime:
            with suppress(Exception):
                await asyncio.to_thread(warm_realtime)

    async def _run_final_translation(self) -> bool:
        items = load_utterances(self.transcript_path)
        groups: dict[str, list[Utterance]] = {}
        for item in items:
            if item.language != "zh" and item.text:
                groups.setdefault(item.language, []).append(item)
        total = sum(len(group) for group in groups.values())
        await self._postprocess_update("translation", "running", current=0, total=total)
        completed = 0
        failures: list[str] = []
        for language, group in groups.items():
            if hasattr(self.runtime, "translate_text_batch"):
                results = await asyncio.to_thread(self.runtime.translate_text_batch, [item.text for item in group], language)
            else:
                results = [await asyncio.to_thread(self.runtime.translate_text, item.text, language) for item in group]
            for item, result in zip(group, results):
                item.translation_zh = result.text if result.status in {"ready", "not_needed"} else ""
                item.translation_status = result.status
                item.translation_model = getattr(result, "model", None)
                if result.status in {"failed", "unsupported"}:
                    failures.append(f"{item.language}:{getattr(result, 'error', None) or result.status}")
                item.revision = max(2, item.revision + 1)
                append_utterance(self.transcript_path, item)
                completed += 1
                await self.broadcast(
                    "translation_update",
                    segment_id=item.segment_id,
                    revision=item.revision,
                    translation_zh=item.translation_zh,
                    translation_status=item.translation_status,
                    translation_model=item.translation_model,
                )
            await self._postprocess_update("translation", "running", current=completed, total=total)
        if failures:
            message = "; ".join(failures[:3])
            self.postprocess.fail("translation", message)
            self._write_state()
            await self._postprocess_update()
            return False
        await self._postprocess_update("translation", "complete", current=completed, total=total)
        return True

    async def _run_diarization(self) -> None:
        if not hasattr(self.runtime, "diarize_audio"):
            await self._postprocess_update("diarization", "complete", current=0, total=0)
            return
        paths = [
            self.output_dir / "audio" / str(segment.get("file"))
            for segment in self.audio_segments
            if isinstance(segment, dict) and segment.get("file")
        ]
        paths = [path for path in paths if path.is_file()]
        if not paths:
            await self._postprocess_update("diarization", "complete", current=0, total=0)
            return
        await self._postprocess_update("diarization", "running", current=0, total=len(paths))
        # ``LiveModelRuntime.diarize_audio`` owns the shared GPU lock.  Keeping
        # the lock at the runtime boundary also makes fake/test runtimes and
        # future schedulers safe from accidental double-acquisition.
        segments = await asyncio.to_thread(self.runtime.diarize_audio, paths)
        write_segments(self.output_dir / "speaker_segments.json", segments)
        items = align_speakers(load_utterances(self.transcript_path), segments)
        for item in items:
            append_utterance(self.transcript_path, item)
            for index, existing in enumerate(self.recent):
                if existing.segment_id == item.segment_id:
                    self.recent[index] = item
                    break
            await self.broadcast("utterance", utterance=item.to_dict())
        await self._postprocess_update("diarization", "complete", current=len(paths), total=len(paths))

    async def run_postprocess(self) -> None:
        self.postprocess.start()
        await self._postprocess_update()
        try:
            if self.translation_tasks:
                await asyncio.gather(*tuple(self.translation_tasks), return_exceptions=True)
            release_realtime = getattr(self.runtime, "release_realtime_model", None)
            if release_realtime:
                await asyncio.to_thread(release_realtime)
            refine_stage = self.postprocess.stages.get("asr_refine", {})
            self._refinement_errors = []
            if self.settings.enable_refinement and refine_stage.get("state") != "complete" and not self._refinement_events:
                await asyncio.to_thread(self._rebuild_refinement_events_from_audio)
            if self.settings.enable_refinement and refine_stage.get("state") != "complete" and self._refinement_events:
                events = sorted(self._refinement_events.values(), key=lambda item: item.revision)
                self.refinement_queue = asyncio.Queue(
                    maxsize=max(self.settings.refinement_queue_size, len(events) + 1)
                )
                for event in events:
                    self.refinement_queue.put_nowait(event)
            events_total = self.refinement_queue.qsize()
            if self.settings.enable_refinement and refine_stage.get("state") != "complete" and events_total:
                self._refinement_progress_current = 0
                self._refinement_progress_total = events_total
                await self._postprocess_update("asr_refine", "running", current=0, total=events_total)
                self.refinement_worker_task = asyncio.create_task(self._refinement_worker(), name=f"refinement-{self.id}")
                await self.refinement_queue.join()
                await self.refinement_queue.put(None)
                await self.refinement_worker_task
                self.refinement_worker_task = None
                if self._refinement_errors:
                    message = "; ".join(self._refinement_errors[:3])
                    self.postprocess.fail("asr_refine", message)
                    self._export_current_files()
                    await self._postprocess_update()
                    await self._warm_realtime_after_postprocess()
                    return
                await self._postprocess_update("asr_refine", "complete", current=events_total, total=events_total)
                await asyncio.to_thread(self._clear_refinement_events)
            elif refine_stage.get("state") != "complete" and not self.settings.enable_refinement:
                await self._postprocess_update("asr_refine", "complete", current=events_total, total=events_total)
            elif refine_stage.get("state") != "complete":
                raise RuntimeError("ASR 精修输入无法从持久化片段或保留录音恢复")
            # Refinement schedules translations asynchronously. Join them before
            # diarization so a late translation cannot overwrite newer speaker
            # metadata with an older utterance revision.
            if self.translation_tasks:
                await asyncio.gather(*tuple(self.translation_tasks), return_exceptions=True)
            self._export_current_files()
            if self.postprocess.stages.get("diarization", {}).get("state") != "complete":
                await self._run_diarization()
            # Diarization is the final stage that needs the retained recording.
            # Keep it available for retries until that stage succeeds, then
            # honour MEETING_KEEP_AUDIO without weakening speaker alignment.
            self._delete_audio_if_unretained()
            self._export_current_files()
            release_postprocess = getattr(self.runtime, "release_postprocess_models", None)
            if release_postprocess:
                await asyncio.to_thread(release_postprocess)
            translation_ok = True
            if self.postprocess.stages.get("translation", {}).get("state") != "complete":
                translation_ok = await self._run_final_translation()
            self._export_current_files()
            if not translation_ok:
                self.summary_state = "idle"
                self.todo_state = "waiting_summary"
                self._write_state()
                await self._warm_realtime_after_postprocess()
                return
            if not load_utterances(self.transcript_path):
                self.postprocess.fail("summary", "meeting has no usable transcript")
                self.summary_state = "error"
                self.summary_error = "会议没有可总结的有效发言"
                await self._postprocess_update()
                return
            self.summary_state = "idle"
            self.todo_state = "waiting_summary"
            self.postprocess.state = "ready_for_summary"
            self.postprocess.current_stage = None
            self.postprocess.error = None
            self._export_current_files()
            await self._postprocess_update()
            await self._warm_realtime_after_postprocess()
        except Exception as exc:
            stage = self.postprocess.current_stage or "asr_refine"
            self.postprocess.fail(stage, str(exc))
            self._export_current_files()
            self._write_state()
            await self.broadcast("error", code="postprocess_failed", stage=stage, message=str(exc), retryable=True)
            await self.broadcast("postprocess_update", meeting=self.snapshot())
            await self._warm_realtime_after_postprocess()

    async def request_postprocess(self) -> bool:
        if self.recording_state != "complete":
            return False
        if self.postprocess_task and not self.postprocess_task.done():
            return True
        if self.postprocess.state not in {"error", "partial", "queued"}:
            return False
        self.postprocess = PostprocessTracker({
            "state": "queued",
            "current_stage": "diarization" if not self.refinement_queue.qsize() else "asr_refine",
            "stages": self.postprocess.stages,
        })
        self.postprocess.error = None
        metrics = getattr(self.runtime, "metrics", None)
        if isinstance(metrics, dict):
            metrics["retry_count"] = int(metrics.get("retry_count", 0)) + 1
        self._schedule_postprocess()
        await self._postprocess_update()
        return True

    def _schedule_postprocess(self) -> None:
        if self.postprocess_scheduler is not None:
            self.postprocess_scheduler()
            return
        if self.postprocess_task is None or self.postprocess_task.done():
            self.postprocess_task = asyncio.create_task(self.run_postprocess(), name=f"postprocess-{self.id}")

    async def _finalize_fast(self, reason: str) -> None:
        async with self._lock:
            if self.recording_state not in {"recording", "starting"}:
                return
            self.recording_state = "finalizing"
            self._write_state()
            flushed = list(self.segmenter.flush()) if self.segmenter else []
        await self.status("正在保存逐句稿", reason=reason)
        for event in flushed:
            await self.queue.put(event)
        await self.queue.join()
        if self.worker_task:
            await self.queue.put(None)
            await self.worker_task
        if self.audio_writer:
            self.audio_segments = await asyncio.to_thread(self.audio_writer.close)
        self.disconnect_stop_task = None
        self.ended_at = utc_now_iso()
        self.recording_state = "complete"
        items = load_utterances(self.transcript_path)
        self.summary_state = "idle" if items else "error"
        self.summary_error = None if items else "会议没有可总结的有效发言"
        self._set_postprocess_queue(bool(items))
        self.files = export_live_result(
            self.output_dir,
            meeting_id=self.id,
            title=self.title,
            started_at=self.started_at,
            ended_at=self.ended_at,
            duration_seconds=self.audio_samples_received / self.audio_sample_rate,
            utterances=items,
            audio_segments=self.audio_segments,
            recording_state=self.recording_state,
            summary_state=self.summary_state,
            todo_state="waiting_summary",
            postprocess=self.postprocess.to_dict(),
            model_metadata=self.model_metadata,
        )
        self._write_state()
        await self.status("录音已保存，正在后台处理")
        await self.broadcast("recording_complete", meeting=self.snapshot())
        if items:
            self._schedule_postprocess()
        else:
            self._delete_audio_if_unretained()
            self._export_current_files()
            self._write_state()

    async def _finalize(self, reason: str) -> None:
        try:
            await self._finalize_fast(reason)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = str(exc)
            if self.worker_task and not self.worker_task.done():
                self.worker_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.worker_task
            if self.audio_writer:
                with suppress(Exception):
                    self.audio_segments = await asyncio.to_thread(self.audio_writer.close)
            self.ended_at = self.ended_at or utc_now_iso()
            self.recording_state = "error"
            self.disconnect_stop_task = None
            self._write_state()
            await self.broadcast(
                "error",
                code="finalize_failed",
                message=self.error,
                retryable=False,
            )
            await self.status("会议保存失败，可删除本次会议后重新开始", reason=reason)

    def begin_summary(self) -> bool:
        preprocessing_ready = all(
            self.postprocess.stages.get(stage, {}).get("state") == "complete"
            for stage in ("asr_refine", "diarization", "translation")
        )
        if (
            self.recording_state != "complete"
            or not preprocessing_ready
            or self.summary_state not in {"idle", "error", "complete"}
        ):
            return False
        self.summary_state = "running"
        self.summary_error = None
        self.todo_state = "stale" if self.summary_revision else "waiting_summary"
        self.todo_error = None
        self.postprocess.state = "running"
        self.postprocess.error = None
        self.postprocess.update("summary", "running", current=0, total=1)
        self.postprocess.update("todo", "idle", current=0, total=0, error=None)
        self.postprocess.stages["summary"]["error"] = None
        self.postprocess.stages["todo"]["error"] = None
        self._write_state()
        return True

    async def request_summary(self) -> bool:
        if not self.begin_summary():
            return False
        if self.todo_task and not self.todo_task.done():
            self.todo_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.todo_task
            self.todo_task = None
        if self.summary_task and not self.summary_task.done():
            return True
        self.summary_task = asyncio.create_task(self.run_summary(claimed=True), name=f"summary-{self.id}")
        return True

    async def run_summary(self, claimed: bool = False) -> None:
        if not claimed and self.summary_state != "queued":
            return
        if not claimed:
            self.begin_summary()
        await self.status("正在生成会议纪要")
        candidate = ""
        try:
            summarizer = self.summarizer_factory(self.settings)

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
            candidate = result.strip()
            next_revision = self.summary_revision + 1
            atomic_write_text(self.output_dir / "meeting_minutes.md", candidate + "\n")
            self.summary = candidate
            self.summary_revision = next_revision
            self.summary_state = "complete"
            self.todo_state = "queued"
            self.summary_error = None
            self.postprocess.update("summary", "complete", current=1, total=1)
            items = load_utterances(self.transcript_path)
            self.files = export_live_result(self.output_dir, meeting_id=self.id, title=self.title, started_at=self.started_at, ended_at=self.ended_at or utc_now_iso(), duration_seconds=self.audio_samples_received / self.audio_sample_rate, utterances=items, audio_segments=self.audio_segments, recording_state=self.recording_state, summary_state=self.summary_state, todo_state=self.todo_state, postprocess=self.postprocess.to_dict(), model_metadata=self.model_metadata)
            self._write_state()
            await self.broadcast("summary_complete", content=self.summary, summary_revision=self.summary_revision, files=self.files)
            self.todo_task = asyncio.create_task(self.run_todo(), name=f"todo-{self.id}-{self.summary_revision}")
        except Exception as exc:
            self.summary_state = "error"
            self.summary_error = str(exc)
            self.postprocess.fail("summary", self.summary_error)
            self._write_state()
            await self.broadcast(
                "error",
                code="summary_failed",
                message=self.summary_error,
                retryable=True,
                summary=self.summary,
                summary_revision=self.summary_revision,
            )
            await self.broadcast("postprocess_update", meeting=self.snapshot())

    def begin_todo(self) -> bool:
        if self.recording_state != "complete" or self.summary_state != "complete" or not self.summary.strip() or self.todo_state not in {"queued", "error", "stale", "complete"}:
            return False
        self.todo_state = "running"
        self.todo_error = None
        self.postprocess.state = "running"
        self.postprocess.error = None
        self.postprocess.update("todo", "running", current=0, total=1)
        self.postprocess.stages["todo"]["error"] = None
        self._write_state()
        return True

    async def request_todo(self) -> bool:
        if not self.begin_todo():
            return False
        if self.todo_task and not self.todo_task.done():
            return True
        self.todo_task = asyncio.create_task(self.run_todo(claimed=True), name=f"todo-{self.id}-{self.summary_revision}")
        return True

    async def run_todo(self, claimed: bool = False) -> None:
        if not claimed and self.todo_state != "queued":
            return
        if not claimed:
            self.begin_todo()
        target_revision = self.summary_revision
        target_summary = self.summary
        await self.broadcast("todo_progress", phase="request", summary_revision=target_revision)
        try:
            generator = self.todo_factory(self.settings)
            todo = await generator.generate(self.id, target_revision, target_summary, on_status=lambda phase: self.broadcast("todo_progress", phase=phase, summary_revision=target_revision))
            if target_revision != self.summary_revision or target_summary != self.summary:
                return
            self.todo = todo
            atomic_write_text(self.output_dir / "todo_list.json", json.dumps(todo.to_dict(), ensure_ascii=False, indent=2))
            atomic_write_text(self.output_dir / "todo_list.md", render_todo_markdown(todo))
            self.todo_state = "complete"
            self.todo_error = None
            self.postprocess.update("todo", "complete", current=1, total=1)
            self.postprocess.complete()
            items = load_utterances(self.transcript_path)
            self.files = export_live_result(self.output_dir, meeting_id=self.id, title=self.title, started_at=self.started_at, ended_at=self.ended_at or utc_now_iso(), duration_seconds=self.audio_samples_received / self.audio_sample_rate, utterances=items, audio_segments=self.audio_segments, recording_state=self.recording_state, summary_state=self.summary_state, todo_state=self.todo_state, postprocess=self.postprocess.to_dict(), model_metadata=self.model_metadata)
            self._write_state()
            await self.broadcast("todo_complete", todo=self.todo.to_dict(), files=self.files, summary_revision=self.summary_revision)
            await self.broadcast("postprocess_update", meeting=self.snapshot())
        except Exception as exc:
            self.todo_state = "error"
            self.todo_error = str(exc)
            self.postprocess.fail("todo", self.todo_error)
            self._write_state()
            await self.broadcast("error", code="todo_failed", message=self.todo_error, retryable=True)
            await self.broadcast("postprocess_update", meeting=self.snapshot())


class SessionManager:
    def __init__(self, settings: Settings, runtime: Any, store: LocalMeetingStore | None = None) -> None:
        self.settings = settings
        self.runtime = runtime
        self.store = store or LocalMeetingStore(settings.results_dir)
        self.sessions: dict[str, LiveMeetingSession] = {}
        self._create_lock = asyncio.Lock()
        self._model_tasks_ready = bool(
            getattr(runtime, "ready", False) and getattr(runtime, "capabilities_ready", True)
        )
        self._shutting_down = False
        self._load_recovered()

    def _load_recovered(self) -> None:
        for payload in self.store.list_states():
            try:
                self._load_recovered_payload(payload)
            except Exception as exc:  # noqa: BLE001 - isolate corrupt persisted meetings
                # One damaged meeting must not prevent every other meeting (or
                # the whole service) from recovering at startup.
                LOGGER.warning("Skipping invalid recovered meeting state: %s", exc)

    def _load_recovered_payload(self, payload: dict[str, Any]) -> None:
        meeting_id = str(payload.get("id", ""))
        if not meeting_id:
            return
        state = str(payload.get("recording_state", "complete"))
        transcript_path = self.store.meeting_dir(meeting_id) / "transcript.jsonl"
        has_transcript = bool(load_utterances(transcript_path))
        if state in {"starting", "recording", "finalizing"} and not has_transcript:
            payload["recording_state"] = "error"
            payload["error"] = "服务重启时会议尚未完成，已保留已保存内容"
        if state in {"starting", "recording", "finalizing"} and has_transcript:
            payload["recording_state"] = "complete"
            payload["summary_state"] = "idle"
            payload["todo_state"] = "waiting_summary"
            payload["ended_at"] = payload.get("ended_at") or utc_now_iso()
            payload["error"] = None
            payload["postprocess"] = PostprocessTracker({
                "state": "queued",
                "current_stage": "asr_refine",
                "stages": {
                    "asr_refine": {"state": "queued"},
                    "diarization": {"state": "queued"},
                    "translation": {"state": "queued"},
                    "summary": {"state": "idle"},
                    "todo": {"state": "idle"},
                },
            }).to_dict()
        if payload.get("recording_state") == "complete":
            summary_was_running = payload.get("summary_state") == "running"
            if summary_was_running:
                payload["summary_state"] = "idle"
                payload["todo_state"] = "waiting_summary"
                postprocess = payload.get("postprocess")
                if not isinstance(postprocess, dict):
                    postprocess = PostprocessTracker({
                        "state": "ready_for_summary",
                        "stages": {
                            "asr_refine": {"state": "complete"},
                            "diarization": {"state": "complete"},
                            "translation": {"state": "complete"},
                            "summary": {"state": "idle"},
                            "todo": {"state": "idle"},
                        },
                    }).to_dict()
                    payload["postprocess"] = postprocess
                postprocess["state"] = "ready_for_summary"
                postprocess["current_stage"] = None
                postprocess["error"] = None
                stages = postprocess.get("stages")
                if isinstance(stages, dict):
                    for stage in ("summary", "todo"):
                        if isinstance(stages.get(stage), dict):
                            stages[stage]["state"] = "idle"
                            stages[stage]["error"] = None
            elif payload.get("todo_state") == "running":
                payload["todo_state"] = "queued" if payload.get("summary_state") == "complete" else "waiting_summary"
            postprocess = payload.get("postprocess")
            if isinstance(postprocess, dict) and postprocess.get("state") in {"queued", "running"} and not summary_was_running:
                stages = postprocess.get("stages")
                model_stages = ("asr_refine", "diarization", "translation")
                needs_model_work = not isinstance(stages, dict) or any(
                    not isinstance(stages.get(stage), dict)
                    or stages[stage].get("state") != "complete"
                    for stage in model_stages
                )
                postprocess["state"] = "queued" if needs_model_work else "ready_for_summary"
                postprocess["current_stage"] = "asr_refine" if needs_model_work else None
        self.sessions[meeting_id] = LiveMeetingSession(
            self.settings,
            self.runtime,
            self.store,
            meeting_id=meeting_id,
            recovered_state=payload,
            postprocess_scheduler=self._schedule_pending_postprocess,
        )

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
            if self._model_postprocess_running() or (
                self.active_count() == 0 and self._pending_model_postprocess()
            ):
                raise CapacityLimitError("上一场会议的模型后处理尚未完成")
            session = LiveMeetingSession(
                self.settings,
                self.runtime,
                self.store,
                title=title or "未命名会议",
                postprocess_scheduler=self._schedule_pending_postprocess,
            )
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
            if self._model_postprocess_running() or (
                self.active_count() == 0 and self._pending_model_postprocess()
            ):
                raise CapacityLimitError("上一场会议的模型后处理尚未完成")
            await session.start()
            return session

    def _model_postprocess_running(self) -> bool:
        return any(
            session.postprocess_task is not None and not session.postprocess_task.done()
            for session in self.sessions.values()
        )

    def _pending_model_postprocess(self) -> bool:
        return any(
            session.recording_state == "complete"
            and session.postprocess.state in {"queued", "running"}
            and any(
                session.postprocess.stages.get(stage, {}).get("state") != "complete"
                for stage in ("asr_refine", "diarization", "translation")
            )
            for session in self.sessions.values()
        )

    def _schedule_pending_postprocess(self) -> None:
        if (
            self._shutting_down
            or not self._model_tasks_ready
            or self.active_count() > 0
            or self._model_postprocess_running()
        ):
            return
        candidates = [
            session for session in self.sessions.values()
            if session.recording_state == "complete"
            and session.postprocess.state in {"queued", "running"}
            and any(
                session.postprocess.stages.get(stage, {}).get("state") != "complete"
                for stage in ("asr_refine", "diarization", "translation")
            )
            and (session.postprocess_task is None or session.postprocess_task.done())
        ]
        if not candidates:
            return

        def queue_order(item: LiveMeetingSession) -> tuple[float, str]:
            try:
                return (datetime.fromisoformat(item.started_at.replace("Z", "+00:00")).timestamp(), item.id)
            except (TypeError, ValueError, OverflowError):
                return (float("inf"), item.id)

        session = min(candidates, key=queue_order)
        session.postprocess_task = asyncio.create_task(
            session.run_postprocess(),
            name=f"postprocess-{session.id}",
        )
        session.postprocess_task.add_done_callback(lambda _task: self._schedule_pending_postprocess())

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
        self._model_tasks_ready = model_tasks_ready
        for session in self.sessions.values():
            if (
                session.todo_state == "queued"
                and session.summary_state == "complete"
                and session.summary.strip()
                and (session.todo_task is None or session.todo_task.done())
            ):
                session.todo_task = asyncio.create_task(session.run_todo(), name=f"recover-todo-{session.id}")
            elif session.summary_state == "queued" and session.recording_state == "complete":
                # Older versions queued summaries automatically. Migration keeps
                # the transcript ready but requires an explicit user action.
                session.summary_state = "idle"
                session.todo_state = "waiting_summary"
                session._write_state()
        self._schedule_pending_postprocess()

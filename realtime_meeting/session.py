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
from .inference import InferenceCapacityError, InferenceCoordinator
from .jimo import MeetingSummarizer
from .models import MeetingSnapshot, SessionState, Utterance, utc_now_iso
from .runtime import LiveModelRuntime, is_boundary_duplicate
from .storage import (
    LocalSessionRepository,
    RefinementJob,
    RefinementJobStore,
    SessionRepository,
)


TERMINAL_STATES: set[SessionState] = {
    "complete", "summary_pending", "summary_error", "refinement_error", "error"
}


class CapacityLimitError(RuntimeError):
    pass


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
        owner_id: str = "local",
        coordinator: InferenceCoordinator | None = None,
        job_store: RefinementJobStore | None = None,
        repository: SessionRepository | None = None,
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
        self.owner_id = owner_id
        self.coordinator = coordinator or InferenceCoordinator(
            fast_workers=settings.fast_inference_workers,
            refine_workers=settings.refine_inference_workers,
            wait_timeout_seconds=settings.inference_wait_timeout_seconds,
            gpu_workers=settings.gpu_workers,
            gpu_memory_budget_mb=settings.gpu_memory_budget_mb,
        )
        self.job_store = job_store or RefinementJobStore(
            settings.results_dir / "refinement_jobs.sqlite3"
        )
        self.repository = repository or LocalSessionRepository()
        self.speaker_clusterer = (
            runtime.new_speaker_clusterer()
            if runtime.ready and hasattr(runtime, "new_speaker_clusterer")
            else None
        )
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
        self.audio_packets_dropped = 0
        self.audio_packets_out_of_order = 0
        self.audio_sample_rate = 16_000
        self.audio_channels = 1
        self.audio_encoding = "pcm_s16le"
        self.audio_packet_ms = 40
        self._last_audio_sequence: int | None = None
        self.audio_level = 0.0
        self._last_audio_event = 0.0
        self.clients: set[WebSocket] = set()
        self.queue: asyncio.Queue[SegmentEvent | None] = asyncio.Queue(
            maxsize=settings.inference_queue_size
        )
        self.refine_queue: asyncio.Queue[RefinementJob | None] = asyncio.Queue(
            maxsize=settings.refinement_queue_size
        )
        self.worker_task: asyncio.Task[None] | None = None
        self.refine_worker_task: asyncio.Task[None] | None = None
        self.disk_task: asyncio.Task[None] | None = None
        self.stop_task: asyncio.Task[None] | None = None
        self.translation_queue: asyncio.Queue[tuple[str, int, str, str, float] | None] = asyncio.Queue(
            maxsize=max(16, settings.inference_queue_size)
        )
        self.translation_worker_task: asyncio.Task[None] | None = None
        self.translation_tasks: set[asyncio.Task[None]] = set()
        self.partial_latencies_ms: deque[float] = deque(maxlen=500)
        self.stable_latencies_ms: deque[float] = deque(maxlen=500)
        self.translation_latencies_ms: deque[float] = deque(maxlen=500)
        self.refinement_enqueued_at: dict[int, float] = {}
        self._partial_raw_by_revision: dict[int, str] = {}
        self._partial_stable_by_revision: dict[int, str] = {}
        self.stop_lock = asyncio.Lock()
        self.recent_text = ""
        self.previous_language: str | None = None
        self.last_final_revision = 0
        self._latest_by_segment: dict[str, Utterance] = {}
        if recovered:
            recovered_items = load_utterances(self.transcript_path)
            self.utterance_count = len(recovered_items)
            self.recent.extend(recovered_items[-500:])
            self._latest_by_segment = {
                item.segment_id or f"legacy:{item.id}": item for item in recovered_items
            }
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
            "audio_packets_dropped": self.audio_packets_dropped,
            "audio_packets_out_of_order": self.audio_packets_out_of_order,
            "audio_samples_received": self.audio_samples_received,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_channels": self.audio_channels,
            "audio_encoding": self.audio_encoding,
            "owner_id": self.owner_id,
        }
        self.repository.write_state(self.output_dir, payload)

    async def start(self) -> None:
        if not self.runtime.ready:
            raise RuntimeError("实时模型尚未就绪")
        if self.speaker_clusterer is None and hasattr(
            self.runtime, "new_speaker_clusterer"
        ):
            self.speaker_clusterer = self.runtime.new_speaker_clusterer()
        self.audio_writer = RotatingAudioWriter(
            self.output_dir / "audio", self.settings.audio_segment_minutes
        )
        vad_factory = getattr(self.runtime, "new_vad", None)
        vad = vad_factory() if callable(vad_factory) else None
        self.segmenter = StreamSegmenter(
            pre_roll_ms=self.settings.audio_pre_roll_ms,
            speech_start_ms=self.settings.speech_start_ms,
            silence_ms=self.settings.silence_ms,
            partial_interval_ms=self.settings.partial_interval_ms,
            max_utterance_ms=int(self.settings.max_utterance_seconds * 1000),
            vad=vad,
        )
        self.state = "recording"
        self._write_state()
        self.worker_task = asyncio.create_task(self._inference_worker(), name=f"inference-{self.id}")
        self.refine_worker_task = asyncio.create_task(
            self._refinement_worker(), name=f"refinement-{self.id}"
        )
        self.translation_worker_task = asyncio.create_task(
            self._translation_worker(), name=f"translation-{self.id}"
        )
        for job in self.job_store.pending(self.id):
            self.refinement_enqueued_at[job.revision] = time.monotonic()
            await self.refine_queue.put(job)
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

    def configure_audio(self, payload: dict[str, Any]) -> None:
        sample_rate = int(payload.get("sample_rate", 16_000) or 0)
        channels = int(payload.get("channels", 1) or 0)
        encoding = str(payload.get("encoding", "pcm_s16le")).strip().casefold()
        packet_ms = int(payload.get("packet_ms", 40) or 0)
        if sample_rate != 16_000:
            raise ValueError("浏览器音频必须是 16000 Hz")
        if channels != 1:
            raise ValueError("浏览器音频必须是单声道")
        if encoding not in {"pcm_s16le", "pcm16"}:
            raise ValueError("仅支持 PCM16 little-endian 音频")
        if packet_ms < 10 or packet_ms > 100:
            raise ValueError("音频包时长必须在 10–100 ms 之间")
        self.audio_sample_rate = sample_rate
        self.audio_channels = channels
        self.audio_encoding = "pcm_s16le"
        self.audio_packet_ms = packet_ms
        self._write_state()

    async def feed_audio(self, pcm: bytes, *, sequence: int | None = None) -> None:
        if self.state != "recording" or not pcm:
            return
        assert self.audio_writer is not None and self.segmenter is not None
        if len(pcm) > self.settings.max_audio_packet_bytes:
            raise ValueError(
                f"音频包过大，单包不能超过 {self.settings.max_audio_packet_bytes} 字节"
            )
        if len(pcm) % 2:
            raise ValueError("音频包必须是偶数长度的 PCM16 数据")
        if sequence is not None:
            if self._last_audio_sequence is not None:
                expected = self._last_audio_sequence + 1
                if sequence > expected:
                    self.audio_packets_dropped += sequence - expected
                elif sequence <= self._last_audio_sequence:
                    self.audio_packets_out_of_order += 1
                    # Do not move the watermark backwards when a delayed or
                    # duplicate packet arrives.
                    sequence = None
            if sequence is not None:
                self._last_audio_sequence = sequence
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
                packets_dropped=self.audio_packets_dropped,
                packets_out_of_order=self.audio_packets_out_of_order,
                samples_received=self.audio_samples_received,
                level=round(self.audio_level, 5),
                vad_active=self.segmenter.active,
                vad_speech_ratio=round(self.segmenter.speech_ratio, 4),
            )

    @staticmethod
    def _invoke_runtime(
        function: Any,
        *args: Any,
        language_hint: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if language_hint:
            try:
                return function(*args, language_hint=language_hint, **kwargs)
            except TypeError as exc:
                if "language_hint" not in str(exc):
                    raise
        return function(*args, **kwargs)

    @staticmethod
    def _trim_partial_boundary(text: str) -> str:
        """Keep a partial at a safe token boundary when the decoder revises it."""

        boundaries = " \t\r\n,，。.!！？?；;、:：)]}》」』\"'"
        for index in range(len(text) - 1, -1, -1):
            if text[index] in boundaries:
                return text[: index + 1].rstrip()
        # CJK decoders do not necessarily emit spaces. Keep a common prefix
        # there rather than dropping every partial until punctuation arrives.
        return text

    def _stable_partial_text(self, revision: int, text: str) -> str | None:
        raw_text = text
        text = text.strip()
        if not text:
            return None
        previous_raw = self._partial_raw_by_revision.get(revision)
        previous_stable = self._partial_stable_by_revision.get(revision, "")
        if previous_raw is None:
            # There is no earlier hypothesis to compare with yet. Preserve
            # the first partial; subsequent revisions will make it stable.
            candidate = text
        else:
            common_length = 0
            for left, right in zip(previous_raw, raw_text):
                if left != right:
                    break
                common_length += 1
            common_prefix = raw_text[:common_length]
            boundaries = " \t\r\n,，。.!！？?；;、:：)]}》」』\"'"
            if common_length < len(raw_text) and raw_text[common_length] in boundaries:
                candidate = common_prefix.rstrip()
            else:
                candidate = self._trim_partial_boundary(common_prefix)
        self._partial_raw_by_revision[revision] = raw_text
        if len(candidate) < len(previous_stable):
            candidate = previous_stable
        if not candidate or candidate == previous_stable:
            return None
        self._partial_stable_by_revision[revision] = candidate
        return candidate

    def _clear_partial_revision(self, revision: int) -> None:
        self._partial_raw_by_revision.pop(revision, None)
        self._partial_stable_by_revision.pop(revision, None)

    def _remember_item(self, item: Utterance, *, replace: bool = False) -> None:
        key = item.segment_id or f"legacy:{item.id}"
        if not item.segment_id:
            item.segment_id = key
        self._latest_by_segment[key] = item
        if replace:
            updated = deque(
                (item if (current.segment_id or f"legacy:{current.id}") == key else current)
                for current in self.recent
            )
            self.recent = deque(updated, maxlen=500)
        else:
            self.recent.append(item)

    async def _append_live_item(self, item: Utterance) -> None:
        if not item.segment_id:
            item.segment_id = f"{self.id}:{item.segment_revision}:{self.utterance_count + 1}"
        if item.revision < 1:
            item.revision = 1
        item.id = self.utterance_count + 1
        append_utterance(self.transcript_path, item)
        self.utterance_count += 1
        self._remember_item(item)
        self.current_language = item.language
        self.previous_language = item.language
        self.recent_text = (self.recent_text + " " + item.text)[-128:]
        await self.broadcast("utterance", utterance=item.to_dict())
        self._schedule_translation(item)

    def _schedule_translation(self, item: Utterance) -> None:
        if item.translation_status not in {"pending", "failed"}:
            return
        if not callable(getattr(self.runtime, "translate_text", None)) and not callable(
            getattr(self.runtime, "translate_text_batch", None)
        ):
            return
        if self.translation_worker_task is None or self.translation_worker_task.done():
            self.translation_worker_task = asyncio.create_task(
                self._translation_worker(), name=f"translation-{self.id}"
            )
        entry = (
            item.segment_id,
            item.revision,
            item.text,
            item.language,
            time.monotonic(),
        )
        try:
            self.translation_queue.put_nowait(entry)
        except asyncio.QueueFull:
            task = asyncio.create_task(
                self.translation_queue.put(entry),
                name=f"translation-enqueue-{self.id}-{item.segment_revision}",
            )
            self.translation_tasks.add(task)
            task.add_done_callback(self.translation_tasks.discard)

    @staticmethod
    def _translation_result_parts(result: Any, fallback_text: str) -> tuple[str, str]:
        if isinstance(result, dict):
            text = result.get("text", result.get("translation_zh", fallback_text))
            status = result.get("status", result.get("translation_status", "ready"))
            return str(text if text is not None else fallback_text), str(status)
        translated = getattr(result, "text", None)
        status = getattr(result, "status", "ready")
        if translated is None and isinstance(result, tuple):
            translated = result[0] if result else fallback_text
            status = result[1] if len(result) > 1 else "ready"
        return str(translated if translated is not None else fallback_text), str(status)

    async def _apply_translation_result(
        self,
        segment_id: str,
        revision: int,
        source_text: str,
        result: Any,
    ) -> None:
        current = self._latest_by_segment.get(segment_id)
        if current is None or current.revision != revision:
            return
        translated, status = self._translation_result_parts(result, source_text)
        current.translation_zh = translated
        current.translation_status = status  # type: ignore[assignment]
        append_utterance(self.transcript_path, current)
        await self.broadcast(
            "translation_update",
            segment_id=segment_id,
            revision=revision,
            translation_zh=current.translation_zh,
            translation_status=current.translation_status,
        )

    async def _translation_worker(self) -> None:
        """Collect at most eight items for 50 ms and translate by source language."""

        while True:
            first = await self.translation_queue.get()
            if first is None:
                self.translation_queue.task_done()
                return
            batch = [first]
            deadline = asyncio.get_running_loop().time() + 0.05
            while len(batch) < 8:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.translation_queue.get(), timeout)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    # The stop path only queues the sentinel after join(), but
                    # keep this branch safe for cancellation/retry callers.
                    self.translation_queue.task_done()
                    break
                batch.append(item)
            try:
                grouped: dict[
                    str, list[tuple[int, tuple[str, int, str, str, float]]]
                ] = {}
                for index, item in enumerate(batch):
                    grouped.setdefault(item[3], []).append((index, item))
                results: list[Any] = [None] * len(batch)
                translate_batch = getattr(self.runtime, "translate_text_batch", None)
                for language, group in grouped.items():
                    texts = [item[2] for _, item in group]
                    if callable(translate_batch):
                        group_results = await self.coordinator.run(
                            "translation", translate_batch, texts, language
                        )
                    else:
                        group_results = await asyncio.gather(
                            *(
                                self.coordinator.run(
                                    "translation",
                                    self.runtime.translate_text,
                                    item[2],
                                    language,
                                )
                                for _, item in group
                            ),
                            return_exceptions=True,
                        )
                    if not isinstance(group_results, list):
                        group_results = list(group_results)
                    if len(group_results) != len(group):
                        raise RuntimeError("翻译后端返回的微批结果数量不一致")
                    for (index, _item), result in zip(group, group_results):
                        results[index] = result
                for item, result in zip(batch, results):
                    if isinstance(result, BaseException):
                        raise result
                    await self._apply_translation_result(
                        item[0], item[1], item[2], result
                    )
                    self.translation_latencies_ms.append(
                        max(0.0, time.monotonic() - item[4]) * 1000
                    )
            except Exception as exc:  # noqa: BLE001 - preserve source transcript
                for item in batch:
                    current = self._latest_by_segment.get(item[0])
                    if current is None or current.revision != item[1]:
                        continue
                    current.translation_status = "failed"
                    current.translation_zh = item[2]
                    append_utterance(self.transcript_path, current)
                    await self.broadcast(
                        "translation_update",
                        segment_id=item[0],
                        revision=item[1],
                        translation_zh=item[2],
                        translation_status="failed",
                    )
                await self.broadcast(
                    "warning", code="translation_failed", message=f"中文翻译失败：{exc}"
                )
            finally:
                for _ in batch:
                    self.translation_queue.task_done()

    async def _translate_item(
        self,
        segment_id: str,
        revision: int,
        text: str,
        language: str,
    ) -> None:
        try:
            result = await self.coordinator.run(
                "translation", self.runtime.translate_text, text, language
            )
            current = self._latest_by_segment.get(segment_id)
            if current is None or current.revision != revision:
                return
            await self._apply_translation_result(segment_id, revision, text, result)
        except Exception as exc:  # noqa: BLE001 - source transcript remains usable
            current = self._latest_by_segment.get(segment_id)
            if current is not None and current.revision == revision:
                current.translation_status = "failed"
                append_utterance(self.transcript_path, current)
                await self.broadcast(
                    "translation_update",
                    segment_id=segment_id,
                    revision=revision,
                    translation_zh=current.text,
                    translation_status="failed",
                )
            await self.broadcast(
                "warning", code="translation_failed", message=f"中文翻译失败：{exc}"
            )

    def _replace_refined_item(self, item: Utterance, existing: Utterance) -> None:
        item.id = existing.id
        item.segment_id = existing.segment_id or f"{self.id}:{existing.segment_revision}:{existing.id}"
        item.revision = max(existing.revision + 1, item.revision)
        if item.translation_zh:
            item.translation_status = item.translation_status or "ready"
        else:
            item.translation_zh = ""
            item.translation_status = "pending"
        self._latest_by_segment[item.segment_id] = item
        self.recent = deque(
            (
                item
                if (current.segment_id or f"legacy:{current.id}") == item.segment_id
                else current
                for current in self.recent
            ),
            maxlen=500,
        )
        append_utterance(self.transcript_path, item)

    async def _inference_worker(self) -> None:
        while True:
            event = await self.queue.get()
            try:
                if event is None:
                    return
                if event.kind == "partial":
                    await self.status("transcribing", "正在生成临时字幕")
                    result = await self.coordinator.run(
                        "fast",
                        self._invoke_runtime,
                        self.runtime.transcribe_partial,
                        event,
                        self.recent_text,
                        self.hotwords,
                        language_hint=self.previous_language,
                    )
                    self.partial_latencies_ms.append(
                        max(0.0, time.monotonic() - (self.started_monotonic + event.end))
                        * 1000
                    )
                    stable_text = (
                        self._stable_partial_text(result.revision, result.text)
                        if result
                        else None
                    )
                    if (
                        result
                        and stable_text
                        and result.revision > self.last_final_revision
                        and self.state == "recording"
                    ):
                        await self.broadcast(
                            "partial",
                            revision=result.revision,
                            start=result.start,
                            end=result.end,
                            text=stable_text,
                            language=result.language,
                        )
                    continue
                await self.status("transcribing", "正在转写稳定片段")
                items = await self._handle_final_event(event)
                self.stable_latencies_ms.append(
                    max(0.0, time.monotonic() - (self.started_monotonic + event.end))
                    * 1000
                )
                self.last_final_revision = max(self.last_final_revision, event.revision)
                self._clear_partial_revision(event.revision)
                await self.broadcast("partial_clear", revision=event.revision)
                for item in items:
                    previous = self.recent[-1] if self.recent else None
                    if (
                        previous
                        and item.start <= previous.end + 0.6
                        and is_boundary_duplicate(previous.text, item.text)
                    ):
                        continue
                    await self.status("translating", "已识别语言，正在生成中文翻译")
                    await self._append_live_item(item)
                if self.state == "recording":
                    await self.status("listening", "正在监听语音")
            except Exception as exc:
                await self._record_processing_error(
                    "inference_failed", f"语音片段处理失败：{exc}"
                )
            finally:
                self.queue.task_done()

    async def _handle_final_event(self, event: SegmentEvent) -> list[Utterance]:
        if not self.settings.refinement_enabled:
            return await self.coordinator.run(
                "fast",
                self._invoke_runtime,
                self.runtime.transcribe_final,
                event,
                next_id=self.utterance_count + 1,
                previous_language=self.previous_language,
                recent_text=self.recent_text,
                hotwords=self.hotwords,
                speaker_clusterer=self.speaker_clusterer,
                refined=False,
                language_hint=self.previous_language,
            )

        draft = await self.coordinator.run(
            "fast",
            self._invoke_runtime,
            self.runtime.transcribe_draft,
            event,
            self.recent_text,
            self.hotwords,
            language_hint=self.previous_language,
        )
        if draft:
            await self.broadcast(
                "draft",
                revision=event.revision,
                start=event.start,
                end=event.end,
                text=draft.text,
                language=draft.language,
                refinement_status="queued",
            )
            draft_item = Utterance(
                id=self.utterance_count + 1,
                start=round(draft.start, 3),
                end=round(draft.end, 3),
                speaker_id=0,
                language=draft.language or self.previous_language or "unknown",
                language_confidence=round(draft.confidence, 4),
                text=draft.text,
                translation_zh="",
                segment_revision=event.revision,
                recognition_stage="fast",
                translation_status="pending",
                segment_id=f"{self.id}:{event.revision}:0",
                revision=1,
            )
            if callable(getattr(self.runtime, "translate_text", None)):
                await self._append_live_item(draft_item)
        job = self.job_store.enqueue(
            session_id=self.id,
            output_dir=self.output_dir,
            event=event,
            draft_text=draft.text if draft else "",
            draft_language=draft.language if draft else None,
        )
        self.refinement_enqueued_at.setdefault(event.revision, time.monotonic())
        if job.status != "done":
            await self.refine_queue.put(job)
        await self._broadcast_refinement_status()
        return []

    async def _commit_refined_items(self, items: list[Utterance]) -> None:
        for index, item in enumerate(items):
            existing = next(
                (
                    candidate
                    for candidate in self.recent
                    if candidate.segment_revision == item.segment_revision
                    and candidate.recognition_stage == "fast"
                ),
                None,
            )
            previous = self.recent[-1] if self.recent else None
            if (
                existing is None
                and previous
                and item.start <= previous.end + 0.6
                and is_boundary_duplicate(previous.text, item.text)
            ):
                continue
            if existing is not None:
                self._replace_refined_item(item, existing)
                await self.broadcast("utterance_update", utterance=item.to_dict())
                self._schedule_translation(item)
            else:
                if not item.segment_id:
                    item.segment_id = f"{self.id}:{item.segment_revision}:{index}"
                await self._append_live_item(item)
            self.current_language = item.language
            self.previous_language = item.language
            self.recent_text = (self.recent_text + " " + item.text)[-128:]

    async def _broadcast_refinement_status(self) -> None:
        counts = self.job_store.counts(self.id)
        ages = [
            max(0.0, time.monotonic() - created)
            for created in self.refinement_enqueued_at.values()
        ]
        await self.broadcast(
            "refinement_status",
            pending=counts["queued"] + counts["running"],
            failed=counts["failed"],
            oldest_age_seconds=round(max(ages, default=0.0), 3),
            state=self.state,
        )

    async def _refinement_worker(self) -> None:
        while True:
            job = await self.refine_queue.get()
            try:
                if job is None:
                    return
                self.job_store.mark_running(job)
                job.attempts += 1
                await self._broadcast_refinement_status()
                event = job.event()
                items = await self.coordinator.run(
                    "refine",
                    self._invoke_runtime,
                    self.runtime.transcribe_final,
                    event,
                    next_id=self.utterance_count + 1,
                    previous_language=self.previous_language,
                    recent_text=self.recent_text,
                    hotwords=self.hotwords,
                    speaker_clusterer=self.speaker_clusterer,
                    refined=True,
                    language_hint=job.draft_language or self.previous_language,
                )
                await self._commit_refined_items(items)
                self.job_store.mark_done(job)
                self.refinement_enqueued_at.pop(job.revision, None)
            except Exception as exc:
                if job is not None and job.attempts < self.settings.refinement_max_attempts:
                    self.job_store.requeue(job, str(exc))
                    await asyncio.sleep(min(4.0, float(2 ** max(0, job.attempts - 1))))
                    await self.refine_queue.put(job)
                elif job is not None:
                    self.job_store.mark_failed(job, str(exc))
                    self.refinement_enqueued_at.pop(job.revision, None)
                    self.processing_error = f"精修片段 {job.revision} 失败：{exc}"
                    self._write_state()
                    await self.broadcast(
                        "error",
                        code="refinement_failed",
                        message=self.processing_error,
                        retryable=True,
                        state=self.state,
                    )
            finally:
                await self._broadcast_refinement_status()
                self.refine_queue.task_done()

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
                refine_counts = self.job_store.counts(self.id)
                if refine_counts["queued"] + refine_counts["running"]:
                    self.state = "refining"
                    self._write_state()
                    await self.status(
                        "refining",
                        "录音已保存，正在后台生成高精度逐句稿",
                        pending=refine_counts["queued"] + refine_counts["running"],
                    )
                refine_worker = self.refine_worker_task
                if refine_worker and not refine_worker.done():
                    await self.refine_queue.join()
                    await self.refine_queue.put(None)
                    await refine_worker
                self.refine_worker_task = None
                if self.translation_tasks:
                    await asyncio.gather(*tuple(self.translation_tasks), return_exceptions=True)
                    self.translation_tasks.clear()
                translation_worker = self.translation_worker_task
                if translation_worker and not translation_worker.done():
                    await self.translation_queue.join()
                    await self.translation_queue.put(None)
                    await translation_worker
                self.translation_worker_task = None
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
                failed_refinements = self.job_store.counts(self.id)["failed"]
                if failed_refinements:
                    self.state = "refinement_error"
                    self.error = f"有 {failed_refinements} 个语音片段精修失败，可重试"
                    self._write_state()
                    await self.broadcast(
                        "error",
                        code="refinement_incomplete",
                        message=self.error,
                        retryable=True,
                        state=self.state,
                        files=self.files,
                    )
                    return
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

    async def retry_refinement(self) -> None:
        if self.state != "refinement_error":
            raise RuntimeError("当前会议没有可重试的精修任务")
        if self.speaker_clusterer is None and hasattr(
            self.runtime, "new_speaker_clusterer"
        ):
            self.speaker_clusterer = self.runtime.new_speaker_clusterer()
        retried = self.job_store.retry_failed(self.id)
        if not retried and not self.job_store.pending(self.id):
            raise RuntimeError("没有失败的精修任务")
        self.state = "refining"
        self.error = None
        self._write_state()
        self.refine_worker_task = asyncio.create_task(
            self._refinement_worker(), name=f"refinement-retry-{self.id}"
        )
        for job in self.job_store.pending(self.id):
            await self.refine_queue.put(job)
        await self._broadcast_refinement_status()
        await self.refine_queue.join()
        await self.refine_queue.put(None)
        await self.refine_worker_task
        self.refine_worker_task = None
        if self.job_store.counts(self.id)["failed"]:
            self.state = "refinement_error"
            self.error = "部分语音片段精修仍然失败"
            self._write_state()
            await self._broadcast_refinement_status()
            return
        utterances = load_utterances(self.transcript_path)
        self.files = export_live_result(
            self.output_dir,
            session_id=self.id,
            started_at=self.started_at,
            ended_at=self.ended_at or utc_now_iso(),
            duration_seconds=self.elapsed_seconds,
            utterances=utterances,
            audio_segments=self.audio_segments,
            status="summary_pending",
            processing_error=None,
        )
        self.state = "summary_pending"
        self.processing_error = None
        self._write_state()
        await self.broadcast(
            "summary_pending",
            session_id=self.id,
            files=self.files,
            utterance_count=len(utterances),
            error=None,
        )

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

    @staticmethod
    def _percentile(values: deque[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(
            len(ordered) - 1,
            max(0, int(round((percentile / 100.0) * (len(ordered) - 1)))),
        )
        return round(ordered[index], 3)

    def metrics_snapshot(self) -> dict[str, Any]:
        refinement_counts = self.job_store.counts(self.id)
        return {
            "audio_packets_received": self.audio_packets_received,
            "audio_packets_dropped": self.audio_packets_dropped,
            "audio_packets_out_of_order": self.audio_packets_out_of_order,
            "audio_duration_seconds": round(self.audio_samples_received / 16_000, 3),
            "vad_speech_ratio": round(
                self.segmenter.speech_ratio if self.segmenter else 0.0, 4
            ),
            "partial_latency_p50_ms": self._percentile(self.partial_latencies_ms, 50),
            "partial_latency_p95_ms": self._percentile(self.partial_latencies_ms, 95),
            "stable_latency_p50_ms": self._percentile(self.stable_latencies_ms, 50),
            "stable_latency_p95_ms": self._percentile(self.stable_latencies_ms, 95),
            "translation_latency_p50_ms": self._percentile(
                self.translation_latencies_ms, 50
            ),
            "translation_latency_p95_ms": self._percentile(
                self.translation_latencies_ms, 95
            ),
            "realtime_queue_depth": self.queue.qsize(),
            "refinement_queue_depth": refinement_counts["queued"]
            + refinement_counts["running"],
            "refinement_queue_oldest_age_seconds": round(
                max(
                    (
                        max(0.0, time.monotonic() - created)
                        for created in self.refinement_enqueued_at.values()
                    ),
                    default=0.0,
                ),
                3,
            ),
            "translation_queue_depth": self.translation_queue.qsize(),
        }

    def snapshot(self) -> MeetingSnapshot:
        refinement_counts = self.job_store.counts(self.id)
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
            audio_packets_dropped=self.audio_packets_dropped,
            audio_packets_out_of_order=self.audio_packets_out_of_order,
            audio_samples_received=self.audio_samples_received,
            audio_level=round(self.audio_level, 5),
            pending_refinements=refinement_counts["queued"] + refinement_counts["running"],
            failed_refinements=refinement_counts["failed"],
            owner_id=self.owner_id,
            metrics=self.metrics_snapshot(),
        )


class SessionManager:
    def __init__(self, settings: Settings, runtime: LiveModelRuntime) -> None:
        self.settings = settings
        self.runtime = runtime
        self.coordinator = InferenceCoordinator(
            fast_workers=settings.fast_inference_workers,
            refine_workers=settings.refine_inference_workers,
            wait_timeout_seconds=settings.inference_wait_timeout_seconds,
            gpu_workers=settings.gpu_workers,
            gpu_memory_budget_mb=settings.gpu_memory_budget_mb,
        )
        self.job_store = RefinementJobStore(settings.results_dir / "refinement_jobs.sqlite3")
        self.repository = LocalSessionRepository()
        self.sessions: dict[str, LiveMeetingSession] = {}
        self.recovery_tasks: set[asyncio.Task[None]] = set()
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
                    owner_id=str(payload.get("owner_id") or "local"),
                    coordinator=self.coordinator,
                    job_store=self.job_store,
                    repository=self.repository,
                )
                session.ended_at = payload.get("ended_at")
                session.processing_error = payload.get("processing_error")
                for attribute in (
                    "audio_bytes_received",
                    "audio_packets_received",
                    "audio_packets_dropped",
                    "audio_packets_out_of_order",
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
                elif recovered_state in {
                    "recording", "starting", "finalizing", "refining", "summarizing"
                }:
                    session.state = "error"
                    session.error = payload.get("error") or session.error
                    if self.job_store.pending(session.id):
                        session.state = "refinement_error"
                        session.error = "服务重启后仍有精修任务待恢复"
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

    def active_count(self) -> int:
        return sum(
            1
            for session in self.sessions.values()
            if session.state not in TERMINAL_STATES
            or (session.worker_task and not session.worker_task.done())
            or (session.stop_task and not session.stop_task.done())
        )

    def resume_recoverable(self) -> int:
        """Resume durable jobs after model loading without blocking readiness."""

        resumed = 0
        for session in self.sessions.values():
            if session.state != "refinement_error" or not self.job_store.pending(session.id):
                continue
            task = asyncio.create_task(
                session.retry_refinement(), name=f"recover-refinement-{session.id}"
            )
            self.recovery_tasks.add(task)
            task.add_done_callback(self.recovery_tasks.discard)
            resumed += 1
        return resumed

    def capacity_snapshot(self) -> dict[str, object]:
        return {
            "active_meetings": self.active_count(),
            "max_concurrent_meetings": self.settings.max_concurrent_meetings,
            "refinement_jobs": self.job_store.counts(),
            "refinement_spool_bytes": self.job_store.spool_bytes(),
            "inference": self.coordinator.snapshot(),
        }

    async def create(
        self, hotwords: str | None = None, *, owner_id: str = "local"
    ) -> LiveMeetingSession:
        async with self.lock:
            if self.active_count() >= self.settings.max_concurrent_meetings:
                raise CapacityLimitError("当前会议并发已达到容量上限")
            if self.job_store.pending_total() >= self.settings.max_pending_refinements:
                raise CapacityLimitError("精修任务队列已达到容量上限")
            if self.job_store.spool_bytes() >= self.settings.max_refinement_spool_bytes:
                raise CapacityLimitError("精修任务磁盘配额已达到上限")
            session = LiveMeetingSession(
                self.settings,
                self.runtime,
                hotwords=hotwords,
                owner_id=owner_id,
                coordinator=self.coordinator,
                job_store=self.job_store,
                repository=self.repository,
            )
            await session.start()
            self.sessions[session.id] = session
            self.active_id = session.id
            return session

    def get(self, session_id: str) -> LiveMeetingSession | None:
        return self.sessions.get(session_id)

    def active(self) -> LiveMeetingSession | None:
        return self.sessions.get(self.active_id) if self.active_id else None

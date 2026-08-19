from __future__ import annotations

import shutil
import subprocess  # nosec B404
import sys
import wave
import math
from collections import deque
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Literal

import numpy as np

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH
AUDIO_LEVEL_SCALE = 3.0
VOLUME_THRESHOLD_MIN_PERCENT = 0.0
VOLUME_THRESHOLD_MAX_PERCENT = 30.0


def volume_threshold_percent_to_rms(percent: float) -> float:
    """Convert the UI meter percentage to a PCM16 RMS threshold."""
    value = float(percent)
    if not math.isfinite(value) or not VOLUME_THRESHOLD_MIN_PERCENT <= value <= VOLUME_THRESHOLD_MAX_PERCENT:
        raise ValueError(
            f"音量阈值必须在 {VOLUME_THRESHOLD_MIN_PERCENT:g}% 到 {VOLUME_THRESHOLD_MAX_PERCENT:g}% 之间"
        )
    return value / 100.0 / AUDIO_LEVEL_SCALE * 32768.0


def rms_to_volume_threshold_percent(rms: float) -> float:
    """Convert a PCM16 RMS threshold to the clamped UI meter percentage."""
    value = float(rms)
    if not math.isfinite(value):
        value = 0.0
    percent = value / 32768.0 * AUDIO_LEVEL_SCALE * 100.0
    return round(min(VOLUME_THRESHOLD_MAX_PERCENT, max(VOLUME_THRESHOLD_MIN_PERCENT, percent)), 1)


def apply_volume_gate(pcm: bytes, minimum_rms: float) -> bytes:
    """Compatibility shim: packet-level gating is intentionally disabled."""
    return pcm


@dataclass(slots=True)
class SegmentEvent:
    kind: Literal["partial", "final"]
    pcm: bytes
    start: float
    end: float
    revision: int
    forced: bool = False
    # Keep the legacy six-field positional constructor compatible while
    # carrying whether a partial contains speech since the previous partial.
    has_new_speech: bool = True
    # Energy-gated signal evidence is separate from model VAD speech. This
    # prevents a VAD-only noise segment from reaching a hallucinating ASR.
    has_audio: bool = True
    # Retain the absolute frame timestamps used to build the event.  This is
    # intentionally optional so legacy callers constructing SegmentEvent
    # positionally continue to work.
    frames: tuple[tuple[int, bytes], ...] | None = None

    def slice(self, start: float, end: float) -> "SegmentEvent":
        """Return a timestamp-preserving PCM slice and its source frame map."""

        left = max(float(self.start), min(float(start), float(self.end)))
        right = max(left, min(float(end), float(self.end)))
        start_offset = int(round((left - self.start) * SAMPLE_RATE)) * SAMPLE_WIDTH
        end_offset = int(round((right - self.start) * SAMPLE_RATE)) * SAMPLE_WIDTH
        start_offset = max(0, min(len(self.pcm), start_offset))
        end_offset = max(start_offset, min(len(self.pcm), end_offset))
        selected_frames = None
        if self.frames is not None:
            left_sample = int(round(left * SAMPLE_RATE))
            right_sample = int(round(right * SAMPLE_RATE))
            selected_frames = tuple(
                (offset, frame)
                for offset, frame in self.frames
                if offset < right_sample and offset + len(frame) // SAMPLE_WIDTH > left_sample
            )
        return SegmentEvent(
            kind=self.kind,
            pcm=self.pcm[start_offset:end_offset],
            start=left,
            end=right,
            revision=self.revision,
            forced=self.forced,
            has_new_speech=self.has_new_speech,
            has_audio=self.has_audio,
            frames=selected_frames,
        )


class StreamSegmenter:
    def __init__(
        self,
        *,
        pre_roll_ms: int = 240,
        speech_start_ms: int = 80,
        silence_ms: int = 350,
        partial_interval_ms: int = 900,
        max_utterance_ms: int = 8000,
        minimum_rms: float = 240.0,
        minimum_speech_ms: int = 300,
        minimum_speech_ratio: float = 0.12,
        vad: Callable[[bytes], bool | None] | None = None,
    ) -> None:
        self.pre_roll_frames = max(1, pre_roll_ms // FRAME_MS)
        self.speech_start_frames = max(1, speech_start_ms // FRAME_MS)
        self.silence_frames = max(1, silence_ms // FRAME_MS)
        self.partial_interval_frames = max(1, partial_interval_ms // FRAME_MS)
        self.max_frames = max(1, max_utterance_ms // FRAME_MS)
        self.minimum_rms = minimum_rms
        self.minimum_speech_frames = max(0, minimum_speech_ms // FRAME_MS)
        self.minimum_speech_ratio = min(1.0, max(0.0, minimum_speech_ratio))
        self.vad = vad
        self.noise_floor = 15.0
        self._bytes = bytearray()
        self._pre_roll: deque[tuple[int, bytes]] = deque(maxlen=self.pre_roll_frames)
        self._active: list[tuple[int, bytes]] = []
        self._speech_run = self._silence_run = 0
        self._total_samples = self._frames_processed = self._speech_frames = 0
        self._revision = self._last_partial_frame_count = 0
        self._speech_since_partial = False
        self._audio_since_partial = False
        self._diagnostics: dict[str, int | float | bool] = {
            "segments_opened": 0,
            "segments_emitted": 0,
            "partial_events_emitted": 0,
            "final_events_emitted": 0,
            "forced_final_events_emitted": 0,
            "admission_rejections": 0,
            "max_rms": 0.0,
            "last_rms": 0.0,
        }

    @property
    def elapsed_seconds(self) -> float:
        return self._total_samples / SAMPLE_RATE

    @property
    def active(self) -> bool:
        return bool(self._active)

    @property
    def speech_ratio(self) -> float:
        return self._speech_frames / self._frames_processed if self._frames_processed else 0.0

    def _is_speech(self, frame: bytes) -> bool:
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
        self._diagnostics["last_rms"] = round(rms, 3)
        self._diagnostics["max_rms"] = max(float(self._diagnostics["max_rms"]), rms)
        near_zero = max(8.0, self.noise_floor * 1.25)
        energy_speech = rms >= max(self.minimum_rms, self.noise_floor * 3.0)

        # Near-zero input must never open a segment. When an actual VAD is
        # available, its explicit speech/silence decision wins; otherwise the
        # energy threshold is the fallback. Letting energy override an
        # explicit VAD=false decision turns room noise into continuous speech
        # and prevents the next paragraph from ever starting.
        if rms < near_zero:
            if not self._active:
                self.noise_floor = 0.98 * self.noise_floor + 0.02 * rms
            return False
        if self.vad:
            try:
                decision = self.vad(frame)
            except Exception:
                decision = None
            if decision is not None:
                return bool(decision)
        return energy_speech

    def _admitted(self, frames: list[tuple[int, bytes]]) -> bool:
        if not frames:
            return False
        speech_frames = 0
        for _, frame in frames:
            samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
            if rms >= self.minimum_rms:
                speech_frames += 1
        admitted = (
            speech_frames >= self.minimum_speech_frames
            and speech_frames / len(frames) >= self.minimum_speech_ratio
        )
        if not admitted:
            self._diagnostics["admission_rejections"] = int(self._diagnostics["admission_rejections"]) + 1
        return admitted

    def feed(self, data: bytes) -> list[SegmentEvent]:
        if len(data) % SAMPLE_WIDTH:
            data = data[:-1]
        self._bytes.extend(data)
        events: list[SegmentEvent] = []
        while len(self._bytes) >= FRAME_BYTES:
            frame = bytes(self._bytes[:FRAME_BYTES])
            del self._bytes[:FRAME_BYTES]
            events.extend(self._feed_frame(frame))
        return events

    def _feed_frame(self, frame: bytes) -> list[SegmentEvent]:
        start = self._total_samples
        self._total_samples += FRAME_SAMPLES
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
        has_audio = rms >= max(self.minimum_rms, 8.0)
        speech = self._is_speech(frame)
        self._frames_processed += 1
        self._speech_frames += int(speech)
        if not self._active:
            self._pre_roll.append((start, frame))
            self._speech_run = self._speech_run + 1 if speech else 0
            if self._speech_run >= self.speech_start_frames:
                self._revision += 1
                self._active = list(self._pre_roll)
                self._diagnostics["segments_opened"] = int(self._diagnostics["segments_opened"]) + 1
                self._last_partial_frame_count = len(self._active)
                self._silence_run = 0
                self._speech_since_partial = True
                self._audio_since_partial = has_audio
            return []
        self._active.append((start, frame))
        self._silence_run = 0 if speech else self._silence_run + 1
        if speech:
            self._speech_since_partial = True
        if has_audio:
            self._audio_since_partial = True
        events: list[SegmentEvent] = []
        count = len(self._active)
        if (
            speech
            and self._speech_since_partial
            and count >= self.partial_interval_frames
            and count - self._last_partial_frame_count >= self.partial_interval_frames
        ):
            self._last_partial_frame_count = count
            events.append(self._make_event("partial", self._active, has_new_speech=True, has_audio=self._audio_since_partial))
            self._speech_since_partial = False
            self._audio_since_partial = False
        if self._silence_run >= self.silence_frames:
            keep_tail = min(5, self._silence_run)
            useful = self._active[: count - self._silence_run + keep_tail]
            if self._admitted(useful):
                events.append(self._make_event("final", useful, has_new_speech=self._speech_since_partial))
            self._reset(self._active[-self.pre_roll_frames:])
        elif count >= self.max_frames:
            if self._admitted(self._active):
                events.append(self._make_event("final", self._active, True, has_new_speech=self._speech_since_partial))
            # A forced cut is a technical boundary, not an overlap window.
            # The current event already contains the audio up to the cut; an
            # overlap would duplicate words when a later language/paragraph
            # boundary is committed. The next speech run will build its own
            # pre-roll from frames received after this boundary.
            self._active = []
            self._pre_roll.clear()
            self._speech_run = 0
            self._silence_run = 0
            self._last_partial_frame_count = 0
            self._speech_since_partial = False
            self._audio_since_partial = False
        return events

    def _make_event(
        self,
        kind: Literal["partial", "final"],
        frames: list[tuple[int, bytes]],
        forced: bool = False,
        *,
        has_new_speech: bool = True,
        has_audio: bool | None = None,
    ) -> SegmentEvent:
        start = frames[0][0]
        end = frames[-1][0] + FRAME_SAMPLES
        if has_audio is None:
            has_audio = any(
                float(np.sqrt(np.mean(np.frombuffer(frame, dtype=np.int16).astype(np.float32) ** 2))) >= max(self.minimum_rms, 8.0)
                for _, frame in frames
            )
        self._diagnostics["segments_emitted"] = int(self._diagnostics["segments_emitted"]) + 1
        if kind == "partial":
            self._diagnostics["partial_events_emitted"] = int(self._diagnostics["partial_events_emitted"]) + 1
        else:
            self._diagnostics["final_events_emitted"] = int(self._diagnostics["final_events_emitted"]) + 1
            if forced:
                self._diagnostics["forced_final_events_emitted"] = int(self._diagnostics["forced_final_events_emitted"]) + 1
        return SegmentEvent(
            kind=kind,
            pcm=b"".join(frame for _, frame in frames),
            start=start / SAMPLE_RATE,
            end=end / SAMPLE_RATE,
            revision=self._revision,
            forced=forced,
            has_new_speech=has_new_speech,
            has_audio=bool(has_audio),
            frames=tuple(frames),
        )

    def _reset(self, pre_roll: list[tuple[int, bytes]] | None = None) -> None:
        self._active = []
        self._speech_run = self._silence_run = self._last_partial_frame_count = 0
        self._speech_since_partial = False
        self._audio_since_partial = False
        self._pre_roll.clear()
        if pre_roll:
            self._pre_roll.extend(pre_roll)

    def flush(self) -> list[SegmentEvent]:
        flush_vad = getattr(self.vad, "flush", None)
        if callable(flush_vad):
            with suppress(Exception):
                flush_vad()
        if not self._active:
            return []
        event = self._make_event("final", self._active, has_new_speech=self._speech_since_partial) if self._admitted(self._active) else None
        self._reset()
        return [event] if event else []

    def diagnostics_snapshot(self) -> dict[str, int | float | bool]:
        """Return bounded VAD/segmentation evidence for replay diagnostics."""

        return {
            **self._diagnostics,
            "frames_processed": self._frames_processed,
            "speech_frames": self._speech_frames,
            "speech_ratio": round(self.speech_ratio, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "active": self.active,
        }


@dataclass(slots=True)
class AudioSegmentInfo:
    file: str
    start_seconds: float
    end_seconds: float
    samples: int
    format: str


class RotatingAudioWriter:
    def __init__(self, output_dir: Path, segment_minutes: int = 30) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.segment_samples = max(1, segment_minutes) * 60 * SAMPLE_RATE
        self.total_samples = self.current_samples = self.index = 0
        self.segments: list[AudioSegmentInfo] = []
        self._process: subprocess.Popen[bytes] | None = None
        self._wave: wave.Wave_write | None = None
        self._sink: BinaryIO | None = None
        self._current_path: Path | None = None
        self._current_start = 0
        self._ffmpeg = shutil.which("ffmpeg")
        self._format = "flac" if self._ffmpeg else "wav"

    def _open(self) -> None:
        self.index += 1
        self.current_samples = 0
        self._current_start = self.total_samples
        suffix = ".flac" if self._format == "flac" else ".wav"
        self._current_path = self.output_dir / f"audio-{self.index:04d}{suffix}"
        if self._format == "flac":
            if not self._ffmpeg:
                raise RuntimeError("FFmpeg 不可用，无法创建 FLAC 录音")
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            self._process = subprocess.Popen(  # nosec B603
                [
                    self._ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "s16le",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    "1",
                    "-i",
                    "pipe:0",
                    "-c:a",
                    "flac",
                    str(self._current_path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
            )
            self._sink = self._process.stdin
        else:
            self._wave = wave.open(str(self._current_path), "wb")
            self._wave.setnchannels(1)
            self._wave.setsampwidth(SAMPLE_WIDTH)
            self._wave.setframerate(SAMPLE_RATE)

    def write(self, pcm: bytes) -> None:
        if len(pcm) % SAMPLE_WIDTH:
            pcm = pcm[:-1]
        offset = 0
        while offset < len(pcm):
            if self._current_path is None:
                self._open()
            room = self.segment_samples - self.current_samples
            take = min(room, (len(pcm) - offset) // SAMPLE_WIDTH)
            part = pcm[offset: offset + take * SAMPLE_WIDTH]
            if self._format == "flac":
                if not self._sink:
                    raise RuntimeError("音频写入器未打开")
                self._sink.write(part)
            else:
                if not self._wave:
                    raise RuntimeError("WAV 写入器未打开")
                self._wave.writeframesraw(part)
            offset += len(part)
            self.current_samples += take
            self.total_samples += take
            if self.current_samples >= self.segment_samples:
                self._close_current()

    def _close_current(self) -> None:
        if self._current_path is None:
            return
        try:
            if self._format == "flac":
                if self._sink:
                    self._sink.close()
                if self._process:
                    try:
                        _stdout, stderr_bytes = self._process.communicate(timeout=30)
                    except subprocess.TimeoutExpired as exc:
                        self._process.kill()
                        _stdout, stderr_bytes = self._process.communicate()
                        raise RuntimeError("FFmpeg FLAC 写入超时") from exc
                    stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
                    if self._process.returncode:
                        raise RuntimeError(f"FFmpeg FLAC 写入失败: {stderr.strip()}")
            elif self._wave:
                self._wave.close()
        except Exception:
            # Never leave a failed encoder attached to the writer. Repeated
            # close attempts would otherwise operate on a closed stdin while
            # the child process or file handle remains live.
            if self._process and self._process.poll() is None:
                self._process.kill()
                self._process.wait(timeout=5)
            if self._wave:
                with suppress(Exception):
                    self._wave.close()
            self._process = None
            self._wave = None
            self._sink = None
            self._current_path = None
            self.current_samples = 0
            raise
        self.segments.append(AudioSegmentInfo(self._current_path.name, round(self._current_start / SAMPLE_RATE, 3), round((self._current_start + self.current_samples) / SAMPLE_RATE, 3), self.current_samples, self._format))
        self._process = None
        self._wave = None
        self._sink = None
        self._current_path = None
        self.current_samples = 0

    def close(self) -> list[dict[str, object]]:
        self._close_current()
        return [asdict(item) for item in self.segments]

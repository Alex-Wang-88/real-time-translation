from __future__ import annotations

import shutil
import subprocess
import sys
import wave
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Literal

import numpy as np


SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH


@dataclass(slots=True)
class SegmentEvent:
    kind: Literal["partial", "final"]
    pcm: bytes
    start: float
    end: float
    revision: int
    forced: bool = False


class FsmnVAD:
    """Small synchronous adapter for FunASR's stateful FSMN-VAD model.

    The segmenter remains usable without FunASR: a missing model or an
    unexpected model response returns ``None`` and lets the adaptive energy
    fallback decide. This keeps local tests and CPU-only installations safe.
    """

    def __init__(self, model_name: str = "fsmn-vad", device: str = "cpu") -> None:
        from funasr import AutoModel

        runtime_device = f"{device}:0" if device == "cuda" else device
        try:
            self.model = AutoModel(
                model=model_name,
                hub="hf",
                trust_remote_code=True,
                device=runtime_device,
            )
        except TypeError:
            self.model = AutoModel(model=model_name, device=runtime_device)
        self.cache: dict[str, Any] = {}
        self.frame_start_ms = 0.0

    def __call__(self, frame: bytes) -> bool | None:
        audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            result = self.model.generate(
                input=audio,
                cache=self.cache,
                is_final=False,
            )
        except TypeError:
            result = self.model.generate(input=audio)
        except Exception:
            return None
        payload = result[0] if isinstance(result, list) and result else result
        frame_start_ms = self.frame_start_ms
        self.frame_start_ms += FRAME_MS
        if isinstance(payload, dict):
            for key in ("is_speech", "speech", "speech_prob", "score"):
                if key in payload:
                    try:
                        value = float(payload[key])
                        return value > 0.5 if key != "is_speech" else bool(value)
                    except (TypeError, ValueError):
                        return None
            info = payload.get("value")
            if isinstance(info, (list, tuple)) and info:
                # FSMN-VAD commonly returns accumulated speech intervals in
                # milliseconds: ``[[start_ms, end_ms], ...]``. Decide for the
                # current 20 ms frame while retaining the boundary format.
                intervals = []
                for interval in info:
                    if isinstance(interval, (list, tuple)) and len(interval) >= 2:
                        try:
                            intervals.append((float(interval[0]), float(interval[1])))
                        except (TypeError, ValueError):
                            continue
                if intervals:
                    frame_end_ms = frame_start_ms + FRAME_MS
                    decision = any(
                        start_ms <= frame_end_ms and end_ms >= frame_start_ms
                        for start_ms, end_ms in intervals
                    )
                    return decision
                try:
                    return float(info[-1]) > 0.5
                except (TypeError, ValueError):
                    return None
        return None


class StreamSegmenter:
    """Adaptive energy VAD with pre-roll, stable silence commits and hard cuts."""

    def __init__(
        self,
        *,
        pre_roll_ms: int = 300,
        speech_start_ms: int = 120,
        silence_ms: int = 500,
        partial_interval_ms: int = 1500,
        max_utterance_ms: int = 12_000,
        minimum_rms: float = 60.0,
        vad: Callable[[bytes], bool | None] | None = None,
    ) -> None:
        self.pre_roll_frames = max(1, pre_roll_ms // FRAME_MS)
        self.speech_start_frames = max(1, speech_start_ms // FRAME_MS)
        self.silence_frames = max(1, silence_ms // FRAME_MS)
        self.partial_interval_frames = max(1, partial_interval_ms // FRAME_MS)
        self.max_frames = max(1, max_utterance_ms // FRAME_MS)
        self.minimum_rms = minimum_rms
        self.vad = vad
        # Start with a conservative floor so a quiet microphone can trigger
        # speech detection during the first few hundred milliseconds. The
        # floor adapts upward while idle, so ordinary background noise is not
        # treated as speech.
        self.noise_floor = 15.0
        self._bytes = bytearray()
        self._pre_roll: deque[tuple[int, bytes]] = deque(maxlen=self.pre_roll_frames)
        self._active: list[tuple[int, bytes]] = []
        self._speech_run = 0
        self._silence_run = 0
        self._total_samples = 0
        self._frames_processed = 0
        self._speech_frames = 0
        self._revision = 0
        self._last_partial_frame_count = 0

    @property
    def elapsed_seconds(self) -> float:
        return self._total_samples / SAMPLE_RATE

    @property
    def active(self) -> bool:
        return bool(self._active)

    @property
    def frames_processed(self) -> int:
        return self._frames_processed

    @property
    def speech_frames(self) -> int:
        return self._speech_frames

    @property
    def speech_ratio(self) -> float:
        if not self._frames_processed:
            return 0.0
        return self._speech_frames / self._frames_processed

    def _is_speech(self, frame: bytes) -> bool:
        if self.vad is not None:
            try:
                model_decision = self.vad(frame)
            except Exception:
                model_decision = None
            if model_decision is not None:
                return bool(model_decision)
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
        threshold = max(self.minimum_rms, self.noise_floor * 3.0)
        speech = rms >= threshold
        if not speech and not self._active:
            self.noise_floor = 0.98 * self.noise_floor + 0.02 * rms
        return speech

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
        frame_start = self._total_samples
        self._total_samples += FRAME_SAMPLES
        speech = self._is_speech(frame)
        self._frames_processed += 1
        if speech:
            self._speech_frames += 1
        events: list[SegmentEvent] = []

        if not self._active:
            self._pre_roll.append((frame_start, frame))
            self._speech_run = self._speech_run + 1 if speech else 0
            if self._speech_run >= self.speech_start_frames:
                self._revision += 1
                self._active = list(self._pre_roll)
                self._last_partial_frame_count = len(self._active)
                self._silence_run = 0
            return events

        self._active.append((frame_start, frame))
        self._silence_run = 0 if speech else self._silence_run + 1
        active_frames = len(self._active)
        if (
            active_frames >= max(50, self.partial_interval_frames)
            and active_frames - self._last_partial_frame_count >= self.partial_interval_frames
        ):
            self._last_partial_frame_count = active_frames
            events.append(self._make_event("partial", self._active))

        if self._silence_run >= self.silence_frames:
            keep_tail = min(5, self._silence_run)
            useful = self._active[: active_frames - self._silence_run + keep_tail]
            if useful:
                events.append(self._make_event("final", useful))
            tail = self._active[-self.pre_roll_frames :]
            self._reset(tail)
        elif active_frames >= self.max_frames:
            events.append(self._make_event("final", self._active, forced=True))
            overlap = self._active[-self.pre_roll_frames :]
            self._revision += 1
            self._active = overlap
            self._pre_roll.clear()
            self._speech_run = self.speech_start_frames
            self._silence_run = 0
            self._last_partial_frame_count = len(self._active)
        return events

    def _make_event(
        self, kind: Literal["partial", "final"], frames: list[tuple[int, bytes]], forced: bool = False
    ) -> SegmentEvent:
        start_sample = frames[0][0]
        end_sample = frames[-1][0] + FRAME_SAMPLES
        return SegmentEvent(
            kind=kind,
            pcm=b"".join(frame for _, frame in frames),
            start=start_sample / SAMPLE_RATE,
            end=end_sample / SAMPLE_RATE,
            revision=self._revision,
            forced=forced,
        )

    def _reset(self, pre_roll: list[tuple[int, bytes]] | None = None) -> None:
        self._active = []
        self._speech_run = 0
        self._silence_run = 0
        self._last_partial_frame_count = 0
        self._pre_roll.clear()
        if pre_roll:
            self._pre_roll.extend(pre_roll)

    def flush(self) -> list[SegmentEvent]:
        if not self._active:
            return []
        event = self._make_event("final", self._active)
        self._reset()
        return [event]


@dataclass(slots=True)
class AudioSegmentInfo:
    file: str
    start_seconds: float
    end_seconds: float
    samples: int
    format: str


class RotatingAudioWriter:
    """Write an unlimited PCM stream into bounded FLAC files, with WAV fallback."""

    def __init__(self, output_dir: Path, segment_minutes: int = 30) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.segment_samples = max(1, segment_minutes) * 60 * SAMPLE_RATE
        self.total_samples = 0
        self.current_samples = 0
        self.index = 0
        self.segments: list[AudioSegmentInfo] = []
        self._process: subprocess.Popen[bytes] | None = None
        self._wave: wave.Wave_write | None = None
        self._sink: BinaryIO | None = None
        self._current_path: Path | None = None
        self._current_start = 0
        self._format = "flac" if shutil.which("ffmpeg") else "wav"

    def _open(self) -> None:
        self.index += 1
        self.current_samples = 0
        self._current_start = self.total_samples
        suffix = ".flac" if self._format == "flac" else ".wav"
        self._current_path = self.output_dir / f"audio-{self.index:04d}{suffix}"
        if self._format == "flac":
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            self._process = subprocess.Popen(
                [
                    "ffmpeg",
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
                    "-compression_level",
                    "5",
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
            room_samples = self.segment_samples - self.current_samples
            available_samples = (len(pcm) - offset) // SAMPLE_WIDTH
            take_samples = min(room_samples, available_samples)
            part = pcm[offset : offset + take_samples * SAMPLE_WIDTH]
            if self._format == "flac":
                if self._sink is None:
                    raise RuntimeError("FLAC writer is not open")
                self._sink.write(part)
            else:
                assert self._wave is not None
                self._wave.writeframesraw(part)
            offset += len(part)
            self.current_samples += take_samples
            self.total_samples += take_samples
            if self.current_samples >= self.segment_samples:
                self._close_current()

    def _close_current(self) -> None:
        if self._current_path is None:
            return
        if self._format == "flac":
            if self._sink is not None:
                self._sink.close()
            assert self._process is not None
            stderr = self._process.stderr.read().decode("utf-8", errors="replace") if self._process.stderr else ""
            code = self._process.wait(timeout=30)
            if code != 0:
                raise RuntimeError(f"FFmpeg FLAC 写入失败: {stderr.strip()}")
        elif self._wave is not None:
            self._wave.close()
        end_sample = self._current_start + self.current_samples
        self.segments.append(
            AudioSegmentInfo(
                file=self._current_path.name,
                start_seconds=round(self._current_start / SAMPLE_RATE, 3),
                end_seconds=round(end_sample / SAMPLE_RATE, 3),
                samples=self.current_samples,
                format=self._format,
            )
        )
        self._process = None
        self._wave = None
        self._sink = None
        self._current_path = None
        self.current_samples = 0

    def close(self) -> list[dict[str, object]]:
        self._close_current()
        return [asdict(item) for item in self.segments]

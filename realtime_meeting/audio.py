from __future__ import annotations

import shutil
import subprocess  # nosec B404
import sys
import wave
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


@dataclass(slots=True)
class SegmentEvent:
    kind: Literal["partial", "final"]
    pcm: bytes
    start: float
    end: float
    revision: int
    forced: bool = False


def decode_audio_pcm(path: Path) -> bytes:
    """Decode a saved audio segment to 16 kHz mono PCM16 for recovery."""
    if path.suffix.casefold() == ".wav":
        with wave.open(str(path), "rb") as source:
            if source.getnchannels() == 1 and source.getsampwidth() == SAMPLE_WIDTH and source.getframerate() == SAMPLE_RATE:
                return source.readframes(source.getnframes())
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(f"恢复精修输入需要 FFmpeg: {path.name}")
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path), "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
    )  # nosec B603
    if result.returncode:
        raise RuntimeError(f"恢复精修音频失败 {path.name}: {result.stderr.decode('utf-8', errors='replace').strip()}")
    return result.stdout


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
        threshold = max(self.minimum_rms, self.noise_floor * 3.0)

        # The model VAD can interpret steady room noise as speech.  Keep an
        # independent energy gate so low-level fans, keyboard noise and audio
        # leakage never open an utterance merely because the model says so.
        if rms < threshold:
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
        return True

    def _admitted(self, frames: list[tuple[int, bytes]]) -> bool:
        if not frames:
            return False
        speech_frames = 0
        for _, frame in frames:
            samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
            if rms >= self.minimum_rms:
                speech_frames += 1
        return (
            speech_frames >= self.minimum_speech_frames
            and speech_frames / len(frames) >= self.minimum_speech_ratio
        )

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
        speech = self._is_speech(frame)
        self._frames_processed += 1
        self._speech_frames += int(speech)
        if not self._active:
            self._pre_roll.append((start, frame))
            self._speech_run = self._speech_run + 1 if speech else 0
            if self._speech_run >= self.speech_start_frames:
                self._revision += 1
                self._active = list(self._pre_roll)
                self._last_partial_frame_count = len(self._active)
                self._silence_run = 0
            return []
        self._active.append((start, frame))
        self._silence_run = 0 if speech else self._silence_run + 1
        events: list[SegmentEvent] = []
        count = len(self._active)
        if count >= max(50, self.partial_interval_frames) and count - self._last_partial_frame_count >= self.partial_interval_frames:
            self._last_partial_frame_count = count
            events.append(self._make_event("partial", self._active))
        if self._silence_run >= self.silence_frames:
            keep_tail = min(5, self._silence_run)
            useful = self._active[: count - self._silence_run + keep_tail]
            if self._admitted(useful):
                events.append(self._make_event("final", useful))
            self._reset(self._active[-self.pre_roll_frames:])
        elif count >= self.max_frames:
            if self._admitted(self._active):
                events.append(self._make_event("final", self._active, True))
            overlap = self._active[-self.pre_roll_frames:]
            self._revision += 1
            self._active = overlap
            self._pre_roll.clear()
            self._speech_run = self.speech_start_frames
            self._silence_run = 0
            self._last_partial_frame_count = len(overlap)
        return events

    def _make_event(self, kind: Literal["partial", "final"], frames: list[tuple[int, bytes]], forced: bool = False) -> SegmentEvent:
        start = frames[0][0]
        end = frames[-1][0] + FRAME_SAMPLES
        return SegmentEvent(kind, b"".join(frame for _, frame in frames), start / SAMPLE_RATE, end / SAMPLE_RATE, self._revision, forced)

    def _reset(self, pre_roll: list[tuple[int, bytes]] | None = None) -> None:
        self._active = []
        self._speech_run = self._silence_run = self._last_partial_frame_count = 0
        self._pre_roll.clear()
        if pre_roll:
            self._pre_roll.extend(pre_roll)

    def flush(self) -> list[SegmentEvent]:
        if not self._active:
            return []
        event = self._make_event("final", self._active) if self._admitted(self._active) else None
        self._reset()
        return [event] if event else []


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

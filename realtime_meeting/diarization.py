from __future__ import annotations

import gc
import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .models import Utterance
from .storage import atomic_write_json


SAMPLE_RATE = 16_000
ENERGY_FRAME_SECONDS = 0.03
MAX_SILENCE_GAP_SECONDS = 0.25
MIN_SPEECH_SECONDS = 0.35
EMBEDDING_CONTEXT_SECONDS = 1.6
EMBEDDING_HOP_SECONDS = 0.8
CLUSTER_THRESHOLD = 0.68
OVERLAP_INCLUDE_THRESHOLD = 0.15


@dataclass(slots=True)
class SpeakerSegment:
    start: float
    end: float
    label: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DiarizationEngine:
    """Local post-meeting speaker separation using Resemblyzer only.

    The engine deliberately has one implementation path: a local Resemblyzer
    voice encoder plus energy VAD and cosine-similarity clustering. It does
    not download a model, use external credentials, or switch to another
    diarization implementation when the local asset is unavailable.
    """

    def __init__(
        self,
        device: str = "cpu",
        *,
        required: bool = False,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self.device = device
        self.required = required
        self.backend = "resemblyzer"
        self.encoder: Any | None = None
        self.ready = False
        self.status = "not_loaded"
        self.error: str | None = None
        self.model_name = "resemblyzer/voice_encoder"
        self.parameters = {
            "max_silence_gap_seconds": MAX_SILENCE_GAP_SECONDS,
            "min_speech_seconds": MIN_SPEECH_SECONDS,
            "embedding_context_seconds": EMBEDDING_CONTEXT_SECONDS,
            "embedding_hop_seconds": EMBEDDING_HOP_SECONDS,
            "cluster_threshold": CLUSTER_THRESHOLD,
            "overlap_include_threshold": OVERLAP_INCLUDE_THRESHOLD,
        }
        if isinstance(parameters, dict):
            self.parameters.update(parameters)

    @staticmethod
    def _module_available(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    @staticmethod
    def _weights_path() -> Path | None:
        try:
            import resemblyzer

            return Path(resemblyzer.__file__).with_name("pretrained.pt")
        except Exception:
            return None

    def _resemblyzer_available(self) -> bool:
        weights = self._weights_path()
        return self._module_available("resemblyzer") and bool(weights and weights.is_file())

    def capability_ready(self) -> bool:
        """Return whether the bundled local Resemblyzer asset can run."""
        return self._resemblyzer_available()

    @property
    def model_size_bytes(self) -> int | None:
        path = self._weights_path()
        return path.stat().st_size if path and path.is_file() else None

    @staticmethod
    def model_parameters() -> dict[str, Any]:
        return {
            "sample_rate": SAMPLE_RATE,
            "energy_frame_seconds": ENERGY_FRAME_SECONDS,
            "max_silence_gap_seconds": MAX_SILENCE_GAP_SECONDS,
            "min_speech_seconds": MIN_SPEECH_SECONDS,
            "embedding_context_seconds": EMBEDDING_CONTEXT_SECONDS,
            "embedding_hop_seconds": EMBEDDING_HOP_SECONDS,
            "cluster_threshold": CLUSTER_THRESHOLD,
            "overlap_include_threshold": OVERLAP_INCLUDE_THRESHOLD,
            "labels": "anonymous speaker_1, speaker_2, ...",
            "overlap_model": False,
        }

    def _load_resemblyzer(self) -> bool:
        try:
            from resemblyzer import VoiceEncoder

            encoder_device = self.device
            if encoder_device == "cuda":
                try:
                    import torch

                    if not torch.cuda.is_available():
                        encoder_device = "cpu"
                except Exception:
                    encoder_device = "cpu"
            self.encoder = VoiceEncoder(device=encoder_device, verbose=False)
            self.backend = "resemblyzer"
            self.ready = True
            self.status = "ready"
            self.error = None
            return True
        except Exception as exc:  # pragma: no cover - optional dependency/runtime
            self.encoder = None
            self.ready = False
            self.status = "error"
            self.error = f"resemblyzer unavailable: {exc}"
            return False

    def load(self) -> bool:
        if self.ready:
            return True
        if not self._resemblyzer_available():
            self.status = "dependency_missing"
            self.error = "resemblyzer package or pretrained.pt is not available"
            return False
        return self._load_resemblyzer()

    def preflight(self) -> bool:
        """Check local availability without loading a neural model."""
        if self._resemblyzer_available():
            self.backend = "resemblyzer"
            self.status = "ready"
            self.error = None
            return True
        self.status = "dependency_missing"
        self.error = "resemblyzer package or pretrained.pt is not available"
        return False

    def close(self) -> None:
        self.encoder = None
        self.ready = False
        gc.collect()

    def diarize(
        self,
        audio: Path | str | Iterable[Path | str],
        parameters: dict[str, Any] | None = None,
    ) -> list[SpeakerSegment]:
        if not self.ready and not self.load():
            raise RuntimeError(self.error or "diarization model is not ready")
        sources = [audio] if isinstance(audio, (str, Path)) else list(audio)
        active = dict(self.parameters)
        if isinstance(parameters, dict):
            active.update(parameters)
        return self._diarize_resemblyzer(sources, active)

    @staticmethod
    def _speech_intervals(
        waveform: Any,
        sample_rate: int,
        *,
        max_silence_gap_seconds: float = MAX_SILENCE_GAP_SECONDS,
        min_speech_seconds: float = MIN_SPEECH_SECONDS,
    ) -> list[tuple[float, float]]:
        import numpy as np

        frame_size = max(1, int(sample_rate * ENERGY_FRAME_SECONDS))
        frame_count = max(0, (len(waveform) - frame_size) // frame_size + 1)
        if frame_count <= 0:
            return []
        frames = waveform[: frame_count * frame_size].reshape(frame_count, frame_size)
        rms = np.sqrt(np.mean(np.square(frames), axis=1))
        nonzero = rms[rms > 1e-5]
        if nonzero.size == 0:
            return []
        noise_floor = float(np.percentile(nonzero, 20))
        upper = float(np.percentile(rms, 80))
        threshold = max(0.004, min(noise_floor * 2.5, upper * 0.6))
        active = rms >= threshold
        max_gap = max(1, int(max_silence_gap_seconds / ENERGY_FRAME_SECONDS))
        min_frames = max(1, int(min_speech_seconds / ENERGY_FRAME_SECONDS))
        intervals: list[tuple[float, float]] = []
        start: int | None = None
        gap = 0
        for index, is_active in enumerate(active):
            if bool(is_active):
                if start is None:
                    start = index
                gap = 0
                continue
            if start is None:
                continue
            gap += 1
            if gap <= max_gap:
                continue
            end = index - gap + 1
            if end - start >= min_frames:
                intervals.append((start * ENERGY_FRAME_SECONDS, min(len(waveform) / sample_rate, end * ENERGY_FRAME_SECONDS)))
            start = None
            gap = 0
        if start is not None:
            end = len(active)
            if end - start >= min_frames:
                intervals.append((start * ENERGY_FRAME_SECONDS, min(len(waveform) / sample_rate, end * ENERGY_FRAME_SECONDS)))
        return intervals

    @staticmethod
    def _cosine_similarity(left: Any, right: Any) -> float:
        import numpy as np

        if left is None or right is None:
            return -1.0
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm <= 1e-8 or right_norm <= 1e-8:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))

    @staticmethod
    def _merge_segments(
        segments: Iterable[SpeakerSegment],
        max_silence_gap_seconds: float = MAX_SILENCE_GAP_SECONDS,
    ) -> list[SpeakerSegment]:
        merged: list[SpeakerSegment] = []
        for item in sorted(segments, key=lambda value: (value.start, value.end, value.label)):
            if item.end <= item.start:
                continue
            if merged and merged[-1].label == item.label and item.start <= merged[-1].end + max_silence_gap_seconds:
                previous = merged[-1]
                previous.end = max(previous.end, item.end)
                previous.confidence = max(previous.confidence, item.confidence)
            else:
                merged.append(SpeakerSegment(item.start, item.end, item.label, item.confidence))
        return merged

    def _diarize_resemblyzer(
        self,
        sources: list[Path | str],
        parameters: dict[str, Any] | None = None,
    ) -> list[SpeakerSegment]:
        """Local no-token diarization using voice embeddings and online clustering.

        Resemblyzer supplies the embedding model locally.  A short-term
        energy VAD keeps silence out of the clusters; if an embedding cannot
        be produced, the segment is still retained as an anonymous speaker so
        post-processing remains recoverable rather than failing the recording.
        """
        import librosa
        import numpy as np

        values = dict(self.parameters)
        if isinstance(parameters, dict):
            values.update(parameters)
        max_silence_gap_seconds = max(0.05, min(1.0, float(values.get("max_silence_gap_seconds", MAX_SILENCE_GAP_SECONDS))))
        min_speech_seconds = max(0.2, min(2.0, float(values.get("min_speech_seconds", MIN_SPEECH_SECONDS))))
        embedding_context_seconds = max(0.5, min(4.0, float(values.get("embedding_context_seconds", EMBEDDING_CONTEXT_SECONDS))))
        embedding_hop_seconds = max(0.25, min(2.0, float(values.get("embedding_hop_seconds", EMBEDDING_HOP_SECONDS))))
        cluster_threshold = max(0.4, min(0.95, float(values.get("cluster_threshold", CLUSTER_THRESHOLD))))
        clusters: list[dict[str, Any]] = []
        output: list[SpeakerSegment] = []
        offset = 0.0
        encoder = self.encoder
        for source in sources:
            try:
                waveform, sample_rate = librosa.load(str(source), sr=SAMPLE_RATE, mono=True)
            except Exception as exc:
                # Skipping a middle file shifts every subsequent timestamp and
                # silently assigns speakers to the wrong utterances. Fail the
                # retryable stage while the recording is still retained.
                raise RuntimeError(f"无法读取说话人重排音频 {Path(source).name}: {exc}") from exc
            duration = len(waveform) / max(1, sample_rate)
            for interval_start, interval_end in self._speech_intervals(
                waveform,
                sample_rate,
                max_silence_gap_seconds=max_silence_gap_seconds,
                min_speech_seconds=min_speech_seconds,
            ):
                cursor = interval_start
                interval_duration = interval_end - interval_start
                while cursor < interval_end - 1e-3:
                    context_end = min(interval_end, cursor + embedding_context_seconds)
                    output_end = min(interval_end, cursor + (embedding_hop_seconds if interval_duration > embedding_context_seconds else interval_duration))
                    chunk = waveform[int(cursor * sample_rate): int(context_end * sample_rate)]
                    label_index = 0
                    confidence = 0.35
                    embedding = None
                    if encoder is not None and len(chunk) >= int(0.8 * sample_rate):
                        try:
                            from resemblyzer import preprocess_wav

                            prepared = preprocess_wav(chunk, source_sr=sample_rate)
                            if len(prepared) >= int(0.5 * sample_rate):
                                embedding = encoder.embed_utterance(prepared)
                        except Exception:
                            embedding = None
                    if embedding is not None:
                        best_similarity = -1.0
                        for index, cluster in enumerate(clusters):
                            similarity = self._cosine_similarity(embedding, cluster["centroid"])
                            if similarity > best_similarity:
                                best_similarity = similarity
                                label_index = index
                        if not clusters or best_similarity < cluster_threshold:
                            label_index = len(clusters)
                            clusters.append({"centroid": np.asarray(embedding), "count": 1})
                            confidence = 0.55
                        else:
                            cluster = clusters[label_index]
                            count = int(cluster["count"])
                            centroid = (cluster["centroid"] * count + embedding) / (count + 1)
                            norm = float(np.linalg.norm(centroid))
                            cluster["centroid"] = centroid / norm if norm > 1e-8 else centroid
                            cluster["count"] = count + 1
                            confidence = max(0.35, min(0.99, (best_similarity + 1.0) / 2.0))
                    elif clusters:
                        label_index = len(clusters) - 1
                    else:
                        clusters.append({"centroid": None, "count": 1})
                    output.append(
                        SpeakerSegment(
                            offset + cursor,
                            offset + max(cursor + 0.05, output_end),
                            f"speaker_{label_index + 1}",
                            confidence,
                        )
                    )
                    if output_end <= cursor:
                        break
                    cursor = output_end
            offset += duration
        return self._merge_segments(output, max_silence_gap_seconds)


def normalize_speaker_labels(segments: Iterable[SpeakerSegment]) -> list[SpeakerSegment]:
    mapping: dict[str, str] = {}
    ordered = sorted(segments, key=lambda item: (item.start, item.end, item.label))
    for item in ordered:
        mapping.setdefault(item.label, f"speaker_{len(mapping) + 1}")
    return [SpeakerSegment(item.start, item.end, mapping[item.label], item.confidence) for item in ordered]


def align_speakers(
    utterances: Iterable[Utterance],
    segments: Iterable[SpeakerSegment],
    overlap_include_threshold: float = OVERLAP_INCLUDE_THRESHOLD,
) -> list[Utterance]:
    """Project diarization intervals onto transcript utterances.

    The original utterance objects are copied by mutation so their stable IDs
    and segment revisions remain unchanged.  Overlapping speakers are kept in
    ``speaker_ids`` while ``speaker_id`` remains the dominant speaker for old
    clients.
    """

    items = list(utterances)
    diarized = list(segments)
    numeric: dict[str, int] = {}
    for segment in diarized:
        numeric.setdefault(segment.label, len(numeric) + 1)
    for item in items:
        duration = max(0.001, item.end - item.start)
        overlaps: list[tuple[float, SpeakerSegment]] = []
        for segment in diarized:
            overlap = max(0.0, min(item.end, segment.end) - max(item.start, segment.start))
            if overlap > 0:
                overlaps.append((overlap, segment))
        if not overlaps:
            continue
        overlaps.sort(key=lambda value: (value[0], value[1].confidence), reverse=True)
        dominant_overlap, dominant = overlaps[0]
        ids = [numeric[segment.label] for overlap, segment in overlaps if overlap / duration >= overlap_include_threshold]
        if not ids:
            ids = [numeric[dominant.label]]
        item.speaker_id = ids[0]
        item.speaker_ids = ids
        item.speaker_overlap = round(min(1.0, sum(overlap for overlap, _ in overlaps) / duration), 4)
        item.speaker_confidence = round(min(1.0, dominant_overlap / duration * dominant.confidence), 4)
        item.speaker_source = "diarization"
        item.revision = max(2, item.revision + 1)
    return items


def write_segments(path: Path, segments: Iterable[SpeakerSegment]) -> None:
    atomic_write_json(path, {"segments": [item.to_dict() for item in segments]})

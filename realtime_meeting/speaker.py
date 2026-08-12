from __future__ import annotations

import numpy as np

from .audio import SAMPLE_RATE


class OnlineSpeakerClusterer:
    """Anonymous online labels backed by the local Resemblyzer encoder."""

    def __init__(self, device: str = "cpu", threshold: float = 0.68, encoder=None) -> None:
        self.device = device
        self.threshold = threshold
        self.encoder = encoder
        if self.encoder is None:
            from resemblyzer import VoiceEncoder

            self.encoder = VoiceEncoder(device=device)
        self.clusters: list[np.ndarray] = []
        self.counts: list[int] = []
        self.last_speaker = 1

    def assign(self, pcm: bytes, content_seconds: float | None = None) -> int:
        if self.encoder is None:
            return self.last_speaker
        if content_seconds is not None and content_seconds < 1.8:
            return self.last_speaker
        try:
            from resemblyzer import preprocess_wav

            wav = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            if len(wav) < int(0.6 * SAMPLE_RATE):
                return self.last_speaker
            prepared = preprocess_wav(wav, source_sr=SAMPLE_RATE)
            embedding = self.encoder.embed_utterance(prepared)
            embedding = embedding / max(float(np.linalg.norm(embedding)), 1e-8)
        except Exception:
            return self.last_speaker
        if not self.clusters:
            self.clusters.append(embedding)
            self.counts.append(1)
            return self.last_speaker
        similarities = [float(np.dot(embedding, centroid)) for centroid in self.clusters]
        index = int(np.argmax(similarities))
        if similarities[index] < self.threshold and len(wav) >= int(3.0 * SAMPLE_RATE):
            index = len(self.clusters)
            self.clusters.append(embedding)
            self.counts.append(1)
        else:
            self.counts[index] += 1
            merged = self.clusters[index] * (self.counts[index] - 1) + embedding
            self.clusters[index] = merged / max(float(np.linalg.norm(merged)), 1e-8)
        self.last_speaker = index + 1
        return self.last_speaker

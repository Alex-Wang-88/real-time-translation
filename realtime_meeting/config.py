from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8765
    device: str = "auto"
    asr_model: str = "large-v3-turbo"
    translation_model: str = "JustFrederik/nllb-200-distilled-1.3B-ct2-int8"
    results_dir: Path = Path("result/live")
    audio_segment_minutes: int = 30
    max_utterance_seconds: float = 5.0
    disk_warn_bytes: int = 2 * 1024**3
    disk_stop_bytes: int = 512 * 1024**2
    jimo_api_url: str = ""
    jimo_authorization: str = ""
    jimo_max_request_chars: int = 12_000
    jimo_transcript_chars: int = 6_000
    jimo_state_chars: int = 4_000
    api_token: str = ""
    max_audio_packet_bytes: int = 256 * 1024
    inference_queue_size: int = 64

    @property
    def jimo_configured(self) -> bool:
        return bool(self.jimo_api_url.strip() and self.jimo_authorization.strip())

    @property
    def is_loopback_host(self) -> bool:
        host = self.host.strip().casefold().strip("[]")
        if host == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    @property
    def api_auth_required(self) -> bool:
        """Require a token for non-local listeners and explicit token configs."""

        return bool(self.api_token.strip()) or not self.is_loopback_host


def load_settings(env_file: Path | None = None) -> Settings:
    if env_file is not None:
        load_dotenv(env_file, override=False)
    else:
        load_dotenv(override=False)
    return Settings(
        host=os.getenv("MEETING_HOST", "127.0.0.1"),
        port=int(os.getenv("MEETING_PORT", "8765")),
        device=os.getenv("MEETING_DEVICE", "auto"),
        asr_model=os.getenv("MEETING_ASR_MODEL", "large-v3-turbo"),
        translation_model=os.getenv(
            "MEETING_TRANSLATION_MODEL",
            "JustFrederik/nllb-200-distilled-1.3B-ct2-int8",
        ),
        results_dir=Path(os.getenv("MEETING_RESULTS_DIR", "result/live")),
        audio_segment_minutes=max(1, int(os.getenv("MEETING_AUDIO_SEGMENT_MINUTES", "30"))),
        max_utterance_seconds=max(
            2.0, min(12.0, float(os.getenv("MEETING_MAX_UTTERANCE_SECONDS", "5")))
        ),
        disk_warn_bytes=max(0, int(os.getenv("MEETING_DISK_WARN_BYTES", str(2 * 1024**3)))),
        disk_stop_bytes=max(0, int(os.getenv("MEETING_DISK_STOP_BYTES", str(512 * 1024**2)))),
        jimo_api_url=os.getenv("JIMO_API_URL", ""),
        jimo_authorization=os.getenv("JIMO_AUTHORIZATION", ""),
        jimo_max_request_chars=max(2_000, int(os.getenv("JIMO_MAX_REQUEST_CHARS", "12000"))),
        jimo_transcript_chars=max(1_000, int(os.getenv("JIMO_TRANSCRIPT_CHARS", "6000"))),
        jimo_state_chars=max(500, int(os.getenv("JIMO_STATE_CHARS", "4000"))),
        api_token=os.getenv("MEETING_API_TOKEN", "").strip(),
        max_audio_packet_bytes=max(
            640, min(4 * 1024 * 1024, int(os.getenv("MEETING_MAX_AUDIO_PACKET_BYTES", str(256 * 1024))))
        ),
        inference_queue_size=max(
            4, min(1_024, int(os.getenv("MEETING_INFERENCE_QUEUE_SIZE", "64")))
        ),
    )

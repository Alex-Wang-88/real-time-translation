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
    asr_model: str = "FunAudioLLM/Fun-ASR-Nano-2512"
    asr_fallback_model: str = "large-v3-turbo"
    refine_asr_model: str = "large-v3"
    refinement_enabled: bool = True
    vad_model: str = "fsmn-vad"
    translation_profile: str = "opusmt-local"
    translation_model: str = "opusmt-local"
    translation_target: str = "zh-CN"
    translation_model_root: Path | None = None
    translation_autodownload: bool = False
    results_dir: Path = Path("result/live")
    audio_segment_minutes: int = 30
    max_utterance_seconds: float = 8.0
    audio_pre_roll_ms: int = 240
    speech_start_ms: int = 80
    silence_ms: int = 350
    partial_interval_ms: int = 900
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
    refinement_queue_size: int = 16
    fast_inference_workers: int = 1
    refine_inference_workers: int = 1
    gpu_workers: int = 1
    gpu_memory_budget_mb: int = 7_200
    inference_wait_timeout_seconds: float = 30.0
    max_concurrent_meetings: int = 1
    max_pending_refinements: int = 10_000
    max_refinement_spool_bytes: int = 20 * 1024**3
    refinement_max_attempts: int = 3
    stream_ticket_ttl_seconds: float = 60.0
    websocket_auth_timeout_seconds: float = 5.0
    trusted_proxy_auth: bool = False
    trusted_proxy_service_token: str = ""
    trusted_proxy_user_header: str = "x-meeting-user"
    trusted_proxy_cidrs: tuple[str, ...] = ("127.0.0.1/32", "::1/128")
    environment_file: Path | None = None

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


def persist_device_preference(settings: Settings, device: str) -> None:
    """Persist a validated device choice when this process loaded a local .env.

    Browser clients cannot write the server's configuration file themselves.
    Keep this deliberately opt-in: tests and embedded callers that construct
    ``Settings`` directly have no environment file and therefore remain
    side-effect free.
    """

    path = settings.environment_file
    if path is None or not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    replacement = f"MEETING_DEVICE={device}"
    for index, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped
            and not stripped.startswith("#")
            and stripped.split("=", 1)[0].strip().upper() == "MEETING_DEVICE"
        ):
            lines[index] = replacement
            break
    else:
        lines.append(replacement)
    try:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        return


def load_settings(env_file: Path | None = None) -> Settings:
    environment_file = env_file or (Path.cwd() / ".env")
    if env_file is not None:
        load_dotenv(env_file, override=False)
    else:
        load_dotenv(override=False)
    refinement_enabled = os.getenv("MEETING_ENABLE_REFINEMENT", "1").strip().casefold()
    trusted_proxy_auth = os.getenv("MEETING_TRUSTED_PROXY_AUTH", "0").strip().casefold()
    return Settings(
        host=os.getenv("MEETING_HOST", "127.0.0.1"),
        port=int(os.getenv("MEETING_PORT", "8765")),
        device=os.getenv("MEETING_DEVICE", "auto"),
        asr_model=os.getenv(
            "MEETING_ASR_PRIMARY",
            os.getenv("MEETING_ASR_MODEL", "FunAudioLLM/Fun-ASR-Nano-2512"),
        ),
        asr_fallback_model=os.getenv(
            "MEETING_ASR_FALLBACK",
            os.getenv("MEETING_ASR_FALLBACK_MODEL", "large-v3-turbo"),
        ),
        refine_asr_model=os.getenv(
            "MEETING_ASR_REFINE",
            os.getenv("MEETING_REFINE_ASR_MODEL", "large-v3"),
        ),
        refinement_enabled=refinement_enabled not in {"0", "false", "no", "off"},
        vad_model=os.getenv(
            "MEETING_VAD", os.getenv("MEETING_VAD_MODEL", "fsmn-vad")
        ),
        translation_profile=os.getenv(
            "MEETING_TRANSLATION_PROFILE", "opusmt-local"
        ).strip().casefold(),
        translation_model=os.getenv(
            "MEETING_TRANSLATION_MODEL",
            "opusmt-local",
        ),
        translation_target=os.getenv("MEETING_TRANSLATION_TARGET", "zh-CN").strip()
        or "zh-CN",
        translation_model_root=(
            Path(os.getenv("MEETING_TRANSLATION_MODEL_ROOT", ""))
            if os.getenv("MEETING_TRANSLATION_MODEL_ROOT", "").strip()
            else None
        ),
        translation_autodownload=os.getenv(
            "MEETING_TRANSLATION_AUTODOWNLOAD", "0"
        ).strip().casefold() in {"1", "true", "yes", "on"},
        results_dir=Path(os.getenv("MEETING_RESULTS_DIR", "result/live")),
        audio_segment_minutes=max(1, int(os.getenv("MEETING_AUDIO_SEGMENT_MINUTES", "30"))),
        max_utterance_seconds=max(
            2.0, min(12.0, float(os.getenv("MEETING_MAX_UTTERANCE_SECONDS", "8")))
        ),
        audio_pre_roll_ms=max(
            80, min(1_000, int(os.getenv("MEETING_AUDIO_PRE_ROLL_MS", "240")))
        ),
        speech_start_ms=max(
            40, min(1_000, int(os.getenv("MEETING_SPEECH_START_MS", "80")))
        ),
        silence_ms=max(
            160, min(2_000, int(os.getenv("MEETING_SILENCE_MS", "350")))
        ),
        partial_interval_ms=max(
            400, min(3_000, int(os.getenv("MEETING_PARTIAL_INTERVAL_MS", "900")))
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
        refinement_queue_size=max(
            1, min(1_024, int(os.getenv("MEETING_REFINEMENT_QUEUE_SIZE", "16")))
        ),
        fast_inference_workers=max(
            1, min(64, int(os.getenv("MEETING_FAST_INFERENCE_WORKERS", "1")))
        ),
        refine_inference_workers=max(
            1, min(64, int(os.getenv("MEETING_REFINE_INFERENCE_WORKERS", "1")))
        ),
        gpu_workers=max(1, min(1, int(os.getenv("MEETING_GPU_WORKERS", "1")))),
        gpu_memory_budget_mb=max(
            1_024,
            min(
                64 * 1_024,
                int(os.getenv("MEETING_GPU_MEMORY_BUDGET_MB", "7200")),
            ),
        ),
        inference_wait_timeout_seconds=max(
            0.1, min(600.0, float(os.getenv("MEETING_INFERENCE_WAIT_TIMEOUT_SECONDS", "30")))
        ),
        max_concurrent_meetings=max(
            1,
            min(
                10_000,
                int(
                    os.getenv(
                        "MEETING_MAX_ACTIVE_MEETINGS",
                        os.getenv("MEETING_MAX_CONCURRENT_MEETINGS", "1"),
                    )
                ),
            ),
        ),
        max_pending_refinements=max(
            1, min(10_000_000, int(os.getenv("MEETING_MAX_PENDING_REFINEMENTS", "10000")))
        ),
        max_refinement_spool_bytes=max(
            1024**3,
            int(os.getenv("MEETING_MAX_REFINEMENT_SPOOL_BYTES", str(20 * 1024**3))),
        ),
        refinement_max_attempts=max(
            1, min(20, int(os.getenv("MEETING_REFINEMENT_MAX_ATTEMPTS", "3")))
        ),
        stream_ticket_ttl_seconds=max(
            10.0,
            min(
                300.0,
                float(os.getenv("MEETING_STREAM_TICKET_TTL_SECONDS", "60")),
            ),
        ),
        websocket_auth_timeout_seconds=max(
            1.0,
            min(
                30.0,
                float(os.getenv("MEETING_WEBSOCKET_AUTH_TIMEOUT_SECONDS", "5")),
            ),
        ),
        trusted_proxy_auth=trusted_proxy_auth in {"1", "true", "yes", "on"},
        trusted_proxy_service_token=os.getenv(
            "MEETING_TRUSTED_PROXY_SERVICE_TOKEN", ""
        ).strip(),
        trusted_proxy_user_header=os.getenv(
            "MEETING_TRUSTED_PROXY_USER_HEADER", "x-meeting-user"
        ).strip().casefold(),
        trusted_proxy_cidrs=tuple(
            item.strip()
            for item in os.getenv(
                "MEETING_TRUSTED_PROXY_CIDRS", "127.0.0.1/32,::1/128"
            ).split(",")
            if item.strip()
        ),
        environment_file=environment_file if environment_file.is_file() else None,
    )

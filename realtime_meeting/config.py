from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional during import-only tests
    load_dotenv = None


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, minimum: int | None = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, value) if minimum is not None else value


def _float(name: str, default: float, minimum: float | None = None) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, value) if minimum is not None else value


@dataclass(slots=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8765
    api_token: str = ""
    device: str = "auto"
    asr_primary: str = "large-v3-turbo"
    asr_fallback: str = "large-v3-turbo"
    asr_refine: str = "large-v3"
    asr_autodownload: bool = True
    enable_refinement: bool = True
    enable_postprocess: bool = True
    vad_model: str = "fsmn-vad"
    translation_model_root: Path = Path("models/opus-mt")
    translation_autodownload: bool = True
    diarization_required: bool = True
    results_dir: Path = Path("result/meetings")
    audio_segment_minutes: int = 30
    max_utterance_seconds: float = 8.0
    audio_pre_roll_ms: int = 240
    speech_start_ms: int = 80
    silence_ms: int = 350
    vad_minimum_rms: float = 240.0
    vad_minimum_speech_ms: int = 300
    vad_minimum_speech_ratio: float = 0.12
    partial_interval_ms: int = 900
    max_audio_packet_bytes: int = 262_144
    inference_queue_size: int = 64
    refinement_queue_size: int = 16
    gpu_workers: int = 1
    gpu_memory_budget_mb: int = 7200
    max_active_meetings: int = 1
    max_pending_tasks: int = 64
    stream_ticket_ttl_seconds: int = 60
    websocket_auth_timeout_seconds: int = 5
    websocket_disconnect_grace_seconds: float = 15.0
    retention_days: int = 30
    keep_audio: bool = True
    jimo_api_url: str = ""
    jimo_todo_api_url: str = "https://jimoai-bot-api.xiaohuodui.cn/v2/chat/completions/share?shareId=jSBaVou1SZDrd4bX"
    jimo_authorization: str = ""
    jimo_max_request_chars: int = 12_000
    jimo_transcript_chars: int = 5_000
    jimo_state_chars: int = 4_000
    jimo_timeout_seconds: float = 180.0
    jimo_connect_timeout_seconds: float = 20.0
    jimo_max_retries: int = 3

    @property
    def jimo_configured(self) -> bool:
        return bool(self.jimo_api_url.strip() and self.jimo_authorization.strip())

    @property
    def todo_configured(self) -> bool:
        return bool(self.jimo_todo_api_url.strip() and self.jimo_authorization.strip())

    @property
    def api_auth_required(self) -> bool:
        return self.host not in {"127.0.0.1", "localhost", "::1"} or bool(self.api_token)

    def prepare_directories(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.translation_model_root.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    if load_dotenv:
        load_dotenv()
    defaults = Settings()
    settings = Settings(
        host=os.getenv("MEETING_HOST", "127.0.0.1"),
        port=_int("MEETING_PORT", 8765, 1),
        api_token=os.getenv("MEETING_API_TOKEN", ""),
        device=os.getenv("MEETING_DEVICE", "auto"),
        asr_primary=os.getenv("MEETING_ASR_REALTIME", os.getenv("MEETING_ASR_PRIMARY", os.getenv("MEETING_ASR_MODEL", defaults.asr_primary))),
        asr_fallback=os.getenv("MEETING_ASR_FALLBACK", os.getenv("MEETING_ASR_FALLBACK_MODEL", defaults.asr_fallback)),
        asr_refine=os.getenv("MEETING_ASR_REFINE", os.getenv("MEETING_REFINE_ASR_MODEL", defaults.asr_refine)),
        asr_autodownload=_bool("MEETING_ASR_AUTODOWNLOAD", True),
        enable_refinement=_bool("MEETING_ENABLE_REFINEMENT", True),
        enable_postprocess=_bool("MEETING_ENABLE_POSTPROCESS", True),
        vad_model=os.getenv("MEETING_VAD", "fsmn-vad"),
        translation_model_root=Path(os.getenv("MEETING_TRANSLATION_MODEL_ROOT", "models/opus-mt")),
        translation_autodownload=_bool("MEETING_TRANSLATION_AUTODOWNLOAD", True),
        diarization_required=_bool("MEETING_DIARIZATION_REQUIRED", True),
        results_dir=Path(os.getenv("MEETING_RESULTS_DIR", "result/meetings")),
        audio_segment_minutes=_int("MEETING_AUDIO_SEGMENT_MINUTES", 30, 1),
        max_utterance_seconds=min(12.0, max(2.0, _float("MEETING_MAX_UTTERANCE_SECONDS", 8.0, 2.0))),
        audio_pre_roll_ms=min(1000, max(80, _int("MEETING_AUDIO_PRE_ROLL_MS", 240, 80))),
        speech_start_ms=min(1000, max(40, _int("MEETING_SPEECH_START_MS", 80, 40))),
        silence_ms=min(2000, max(160, _int("MEETING_SILENCE_MS", 350, 160))),
        vad_minimum_rms=min(5000.0, _float("MEETING_VAD_MINIMUM_RMS", 240.0, 0.0)),
        vad_minimum_speech_ms=min(2000, _int("MEETING_VAD_MINIMUM_SPEECH_MS", 300, 0)),
        vad_minimum_speech_ratio=min(1.0, _float("MEETING_VAD_MINIMUM_SPEECH_RATIO", 0.12, 0.0)),
        partial_interval_ms=min(5000, max(400, _int("MEETING_PARTIAL_INTERVAL_MS", 900, 400))),
        max_audio_packet_bytes=_int("MEETING_MAX_AUDIO_PACKET_BYTES", 262_144, 1024),
        inference_queue_size=_int("MEETING_INFERENCE_QUEUE_SIZE", 64, 1),
        refinement_queue_size=_int("MEETING_REFINEMENT_QUEUE_SIZE", 16, 1),
        gpu_workers=_int("MEETING_GPU_WORKERS", 1, 1),
        gpu_memory_budget_mb=_int("MEETING_GPU_MEMORY_BUDGET_MB", 7200, 1024),
        max_active_meetings=_int("MEETING_MAX_ACTIVE_MEETINGS", 1, 1),
        max_pending_tasks=_int("MEETING_MAX_PENDING_TASKS", 64, 1),
        stream_ticket_ttl_seconds=_int("MEETING_STREAM_TICKET_TTL_SECONDS", 60, 10),
        websocket_auth_timeout_seconds=_int("MEETING_WEBSOCKET_AUTH_TIMEOUT_SECONDS", 5, 1),
        websocket_disconnect_grace_seconds=min(120.0, max(0.0, _float("MEETING_WEBSOCKET_DISCONNECT_GRACE_SECONDS", 15.0, 0.0))),
        retention_days=_int("MEETING_RETENTION_DAYS", 30, 0),
        keep_audio=_bool("MEETING_KEEP_AUDIO", True),
        jimo_api_url=os.getenv("JIMO_API_URL", ""),
        jimo_todo_api_url=os.getenv("JIMO_TODO_API_URL", defaults.jimo_todo_api_url),
        jimo_authorization=os.getenv("JIMO_AUTHORIZATION", ""),
        jimo_max_request_chars=_int("JIMO_MAX_REQUEST_CHARS", 12_000, 2000),
        jimo_transcript_chars=_int("JIMO_TRANSCRIPT_CHARS", 5000, 1000),
        jimo_state_chars=_int("JIMO_STATE_CHARS", 4000, 1000),
        jimo_timeout_seconds=_float("JIMO_TIMEOUT_SECONDS", 180.0, 10.0),
        jimo_connect_timeout_seconds=_float("JIMO_CONNECT_TIMEOUT_SECONDS", 20.0, 1.0),
        jimo_max_retries=min(5, max(1, _int("JIMO_MAX_RETRIES", 3, 1))),
    )
    if settings.device not in {"auto", "cpu", "cuda"}:
        settings.device = "auto"
    settings.prepare_directories()
    return settings

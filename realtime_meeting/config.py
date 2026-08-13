from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    # Match the quality-oriented live decode used by the reference pipeline.
    # The retry path still escalates to at least beam 5 when a result looks bad.
    asr_realtime_beam_size: int = 5
    asr_refine_beam_size: int = 6
    asr_best_of: int = 5
    asr_retry_temperature: float = 0.2
    asr_log_prob_threshold: float = -1.0
    asr_no_speech_threshold: float = 0.6
    asr_compression_ratio_threshold: float = 2.4
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
    # Keep sentence-final pauses in the same segment so Whisper sees more
    # complete phrases.  This is deliberately configurable for low-latency
    # deployments.
    silence_ms: int = 700
    vad_minimum_rms: float = 240.0
    vad_minimum_speech_ms: int = 450
    vad_minimum_speech_ratio: float = 0.12
    partial_interval_ms: int = 900
    max_audio_packet_bytes: int = 262_144
    inference_queue_size: int = 64
    refinement_queue_size: int = 16
    max_active_meetings: int = 1
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


ASR_SETTING_LIMITS: dict[str, tuple[int, int]] = {
    "realtime_beam_size": (1, 10),
    "refine_beam_size": (1, 12),
    "best_of": (1, 12),
    "silence_ms": (160, 2000),
    "vad_minimum_speech_ms": (0, 2000),
}

# These are the controls that are safe and meaningful to change for one
# meeting.  Deployment concerns such as model paths, API credentials, queue
# sizes and network ports intentionally stay in ``Settings``.  Keeping this
# schema in one place also gives the future template importer a stable
# contract instead of having to know about individual HTML controls.
MEETING_SETTING_LIMITS: dict[str, tuple[float, float]] = {
    "volume_threshold_percent": (0.0, 30.0),
    "speech_start_ms": (40.0, 1000.0),
    "audio_pre_roll_ms": (40.0, 1000.0),
    "silence_ms": (160.0, 2000.0),
    "vad_minimum_speech_ms": (0.0, 2000.0),
    "vad_minimum_speech_ratio": (0.0, 1.0),
    "max_utterance_seconds": (2.0, 12.0),
    "partial_interval_ms": (100.0, 5000.0),
    "audio_segment_minutes": (1.0, 120.0),
    "realtime_beam_size": (1.0, 10.0),
    "refine_beam_size": (1.0, 12.0),
    "best_of": (1.0, 12.0),
    "retry_temperature": (0.0, 1.0),
    "log_prob_threshold": (-10.0, 0.0),
    "no_speech_threshold": (0.0, 1.0),
    "compression_ratio_threshold": (1.0, 10.0),
    "translation_beam_size": (1.0, 8.0),
    "translation_max_decoding_length": (64.0, 1024.0),
    "translation_repetition_penalty": (1.0, 2.0),
    "speaker_cluster_threshold": (0.4, 0.95),
    "speaker_min_speech_seconds": (0.2, 2.0),
    "speaker_max_silence_gap_seconds": (0.05, 1.0),
    "speaker_overlap_include_threshold": (0.0, 1.0),
}

MEETING_BOOLEAN_SETTINGS = (
    "enable_refinement",
    "enable_postprocess",
    "diarization_required",
    "keep_audio",
)


def _volume_percent_from_rms(value: float) -> float:
    try:
        result = float(value) / 32768.0 * 3.0 * 100.0
    except (TypeError, ValueError):
        result = 0.0
    return round(max(0.0, min(30.0, result)), 1)


def default_asr_settings(settings: Settings) -> dict[str, int]:
    return {
        "realtime_beam_size": settings.asr_realtime_beam_size,
        "refine_beam_size": settings.asr_refine_beam_size,
        "best_of": settings.asr_best_of,
        "silence_ms": settings.silence_ms,
        "vad_minimum_speech_ms": settings.vad_minimum_speech_ms,
    }


def default_meeting_settings(settings: Settings) -> dict[str, Any]:
    """Return the user-facing settings for a newly created meeting."""
    asr = default_asr_settings(settings)
    return {
        "volume_threshold_percent": _volume_percent_from_rms(settings.vad_minimum_rms),
        "speech_start_ms": settings.speech_start_ms,
        "audio_pre_roll_ms": settings.audio_pre_roll_ms,
        "silence_ms": asr["silence_ms"],
        "vad_minimum_speech_ms": asr["vad_minimum_speech_ms"],
        "vad_minimum_speech_ratio": settings.vad_minimum_speech_ratio,
        "max_utterance_seconds": settings.max_utterance_seconds,
        "partial_interval_ms": settings.partial_interval_ms,
        "audio_segment_minutes": settings.audio_segment_minutes,
        "realtime_beam_size": asr["realtime_beam_size"],
        "refine_beam_size": asr["refine_beam_size"],
        "best_of": asr["best_of"],
        "retry_temperature": settings.asr_retry_temperature,
        "log_prob_threshold": settings.asr_log_prob_threshold,
        "no_speech_threshold": settings.asr_no_speech_threshold,
        "compression_ratio_threshold": settings.asr_compression_ratio_threshold,
        "translation_beam_size": 2,
        "translation_max_decoding_length": 384,
        "translation_repetition_penalty": 1.05,
        "speaker_cluster_threshold": 0.68,
        "speaker_min_speech_seconds": 0.35,
        "speaker_max_silence_gap_seconds": 0.25,
        "speaker_overlap_include_threshold": 0.15,
        "enable_refinement": settings.enable_refinement,
        "enable_postprocess": settings.enable_postprocess,
        "diarization_required": settings.diarization_required,
        "keep_audio": settings.keep_audio,
    }


def _flatten_meeting_settings(values: object) -> dict[str, Any]:
    if not isinstance(values, dict):
        return {}
    result = dict(values)
    # Accept grouped objects now so a later template file can be organized by
    # panel without changing the persisted flat representation.
    for group in ("audio", "segmentation", "asr", "translation", "speaker", "processing"):
        nested = values.get(group)
        if isinstance(nested, dict):
            result.update(nested)
    legacy_asr = values.get("asr_settings")
    if isinstance(legacy_asr, dict):
        result.update(legacy_asr)
    return result


def normalize_meeting_settings(values: object, settings: Settings) -> dict[str, Any]:
    """Clamp user-editable meeting settings while preserving safe defaults."""
    source = _flatten_meeting_settings(values)
    defaults = default_meeting_settings(settings)
    normalized: dict[str, Any] = {}
    integer_fields = {
        "speech_start_ms", "audio_pre_roll_ms", "silence_ms",
        "vad_minimum_speech_ms", "partial_interval_ms", "audio_segment_minutes",
        "realtime_beam_size", "refine_beam_size", "best_of",
        "translation_beam_size", "translation_max_decoding_length",
    }
    for name, (minimum, maximum) in MEETING_SETTING_LIMITS.items():
        default = defaults[name]
        try:
            value = float(source.get(name, default))
        except (TypeError, ValueError):
            value = float(default)
        value = max(minimum, min(maximum, value))
        normalized[name] = int(round(value)) if name in integer_fields else round(value, 4)
    for name in MEETING_BOOLEAN_SETTINGS:
        value = source.get(name, defaults[name])
        if isinstance(value, str):
            normalized[name] = value.strip().casefold() in {"1", "true", "yes", "on"}
        else:
            normalized[name] = bool(value)
    return normalized


def asr_settings_from_meeting(values: object, settings: Settings) -> dict[str, int]:
    normalized = normalize_meeting_settings(values, settings)
    return {
        name: int(normalized[name])
        for name in ASR_SETTING_LIMITS
    }


def normalize_asr_settings(values: object, settings: Settings) -> dict[str, int]:
    source = values if isinstance(values, dict) else {}
    defaults = default_asr_settings(settings)
    normalized: dict[str, int] = {}
    for name, (minimum, maximum) in ASR_SETTING_LIMITS.items():
        try:
            value = int(float(source.get(name, defaults[name])))
        except (TypeError, ValueError):
            value = defaults[name]
        normalized[name] = max(minimum, min(maximum, value))
    return normalized


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
        asr_realtime_beam_size=min(10, _int("MEETING_ASR_REALTIME_BEAM_SIZE", defaults.asr_realtime_beam_size, 1)),
        asr_refine_beam_size=min(12, _int("MEETING_ASR_REFINE_BEAM_SIZE", defaults.asr_refine_beam_size, 1)),
        asr_best_of=min(12, _int("MEETING_ASR_BEST_OF", defaults.asr_best_of, 1)),
        asr_retry_temperature=min(1.0, _float("MEETING_ASR_RETRY_TEMPERATURE", defaults.asr_retry_temperature, 0.0)),
        asr_log_prob_threshold=min(0.0, _float("MEETING_ASR_LOG_PROB_THRESHOLD", defaults.asr_log_prob_threshold)),
        asr_no_speech_threshold=min(1.0, _float("MEETING_ASR_NO_SPEECH_THRESHOLD", defaults.asr_no_speech_threshold, 0.0)),
        asr_compression_ratio_threshold=max(1.0, _float("MEETING_ASR_COMPRESSION_RATIO_THRESHOLD", defaults.asr_compression_ratio_threshold, 1.0)),
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
        silence_ms=min(2000, max(160, _int("MEETING_SILENCE_MS", 700, 160))),
        vad_minimum_rms=min(5000.0, _float("MEETING_VAD_MINIMUM_RMS", 240.0, 0.0)),
        vad_minimum_speech_ms=min(2000, _int("MEETING_VAD_MINIMUM_SPEECH_MS", 450, 0)),
        vad_minimum_speech_ratio=min(1.0, _float("MEETING_VAD_MINIMUM_SPEECH_RATIO", 0.12, 0.0)),
        partial_interval_ms=min(5000, max(400, _int("MEETING_PARTIAL_INTERVAL_MS", 900, 400))),
        max_audio_packet_bytes=_int("MEETING_MAX_AUDIO_PACKET_BYTES", 262_144, 1024),
        inference_queue_size=_int("MEETING_INFERENCE_QUEUE_SIZE", 64, 1),
        refinement_queue_size=_int("MEETING_REFINEMENT_QUEUE_SIZE", 16, 1),
        max_active_meetings=_int("MEETING_MAX_ACTIVE_MEETINGS", 1, 1),
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

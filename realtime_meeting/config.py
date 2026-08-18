from __future__ import annotations

import os
from dataclasses import dataclass, field
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
    except (TypeError, ValueError):
        value = default
    return max(minimum, value) if minimum is not None else value


def _float(name: str, default: float, minimum: float | None = None) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value) if minimum is not None else value


RECOGNITION_ARCHITECTURES: dict[str, dict[str, str]] = {
    "single_1_7b_no_lid": {
        "label": "默认：单 1.7B，分段级同模确认",
        "policy": "识别、语言确认和冲突重识别统一使用 1.7B；同一语音段只做一次语言探测，不加载第二个 ASR 模型。",
    },
}
DEFAULT_RECOGNITION_ARCHITECTURE = "single_1_7b_no_lid"


def normalize_recognition_architecture(value: object, default: str = DEFAULT_RECOGNITION_ARCHITECTURE) -> str:
    requested = str(value or "").strip().casefold()
    if requested in RECOGNITION_ARCHITECTURES:
        return requested
    fallback = str(default or DEFAULT_RECOGNITION_ARCHITECTURE).strip().casefold()
    return fallback if fallback in RECOGNITION_ARCHITECTURES else DEFAULT_RECOGNITION_ARCHITECTURE


@dataclass(slots=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8765
    api_token: str = ""
    device: str = "auto"
    asr_primary: str = "Qwen/Qwen3-ASR-1.7B"
    asr_hotwords: list[str] = field(default_factory=list)
    single_asr_model: bool = True
    asr_fallback: str = "Qwen/Qwen3-ASR-1.7B"
    language_id_model: str = "Qwen/Qwen3-ASR-1.7B"
    # The application currently exposes one strategy only.  The legacy model
    # fields below remain for benchmark compatibility, while production
    # defaults route every ASR/LID role to the same 1.7B checkpoint.
    recognition_architecture: str = DEFAULT_RECOGNITION_ARCHITECTURE
    asr_autodownload: bool = True
    vad_model: str = "fsmn-vad"
    translation_model_root: Path = Path("models/opus-mt")
    # Translation is local-only by default.  Enabling this only affects the
    # explicit startup preflight, never a live translation request.
    translation_autodownload: bool = False
    results_dir: Path = Path("result/meetings")
    audio_segment_minutes: int = 30
    max_utterance_seconds: float = 18.0
    audio_pre_roll_ms: int = 500
    speech_start_ms: int = 80
    silence_ms: int = 950
    vad_minimum_rms: float = 109.23
    vad_minimum_speech_ms: int = 200
    vad_minimum_speech_ratio: float = 0.06
    partial_interval_ms: int = 1000
    language_id_min_seconds: float = 1.0
    language_conflict_confirmations: int = 3
    language_id_on_segment: bool = True
    language_id_on_conflict: bool = True
    language_switch_window_ms: int = 800
    language_switch_max_wait_ms: int = 1800
    stable_prefix_min_chars: int = 8
    max_audio_packet_bytes: int = 262_144
    max_recording_seconds: float = 4 * 60 * 60
    audio_ingest_burst_seconds: float = 5.0
    inference_queue_size: int = 64
    translation_queue_size: int = 64
    # Backend-only rollback switch.  The default keeps the segmented,
    # coalescing pipeline enabled without changing the WebSocket contract.
    realtime_pipeline: bool = True
    audio_drain_timeout_seconds: float = 3.0
    asr_timeout_seconds: float = 45.0
    translation_timeout_seconds: float = 90.0
    translation_deadline_seconds: float = 4.5
    translation_warmup: bool = True
    translation_partial_debounce_ms: int = 250
    # Automatic local post-meeting retranscription/retranslation is opt-in.
    # A future higher-quality external translation agent can be requested
    # explicitly without adding another resident local model.
    post_meeting_translation_enabled: bool = False
    post_meeting_translation_quality_threshold: float = 0.62
    post_meeting_translation_context_paragraphs: int = 2
    post_meeting_translation_context_chars: int = 240
    post_meeting_translation_timeout_seconds: float = 180.0
    queue_join_timeout_seconds: float = 120.0
    websocket_send_timeout_seconds: float = 5.0
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
    jimo_max_response_chars: int = 80_000
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


MEETING_SETTING_LIMITS: dict[str, tuple[float, float]] = {
    "volume_threshold_percent": (0.0, 30.0),
    "speech_start_ms": (40.0, 1000.0),
    "audio_pre_roll_ms": (40.0, 1000.0),
    "silence_ms": (160.0, 2000.0),
    "vad_minimum_speech_ms": (0.0, 2000.0),
    "vad_minimum_speech_ratio": (0.0, 1.0),
    "max_utterance_seconds": (2.0, 18.0),
    "partial_interval_ms": (100.0, 5000.0),
    "audio_segment_minutes": (1.0, 120.0),
    "translation_beam_size": (1.0, 8.0),
    "translation_max_decoding_length": (64.0, 1024.0),
    "translation_repetition_penalty": (1.0, 2.0),
}

MEETING_BOOLEAN_SETTINGS = ("keep_audio",)
ASR_HOTWORDS_MAX_ITEMS = 100
ASR_HOTWORD_MAX_CHARS = 64
ASR_HOTWORDS_MAX_TOTAL_CHARS = 4_000


def normalize_asr_hotwords(values: object) -> list[str]:
    if isinstance(values, str):
        source = values.replace("，", ",").replace("\r", "\n").replace("\n", ",").split(",")
    elif isinstance(values, (list, tuple, set, frozenset)):
        source = list(values)
    else:
        source = []
    result: list[str] = []
    seen: set[str] = set()
    total = 0
    for item in source:
        value = " ".join(str(item or "").split())[:ASR_HOTWORD_MAX_CHARS]
        key = value.casefold()
        if not value or key in seen:
            continue
        if total + len(value) > ASR_HOTWORDS_MAX_TOTAL_CHARS or len(result) >= ASR_HOTWORDS_MAX_ITEMS:
            break
        result.append(value)
        seen.add(key)
        total += len(value)
    return result


def _volume_percent_from_rms(value: float) -> float:
    try:
        result = float(value) / 32768.0 * 3.0 * 100.0
    except (TypeError, ValueError):
        result = 0.0
    return round(max(0.0, min(30.0, result)), 1)


def default_asr_settings(settings: Settings) -> dict[str, int]:
    return {
        "silence_ms": settings.silence_ms,
        "vad_minimum_speech_ms": settings.vad_minimum_speech_ms,
    }


def default_meeting_settings(settings: Settings) -> dict[str, Any]:
    return {
        "volume_threshold_percent": _volume_percent_from_rms(settings.vad_minimum_rms),
        "speech_start_ms": settings.speech_start_ms,
        "audio_pre_roll_ms": settings.audio_pre_roll_ms,
        "silence_ms": settings.silence_ms,
        "vad_minimum_speech_ms": settings.vad_minimum_speech_ms,
        "vad_minimum_speech_ratio": settings.vad_minimum_speech_ratio,
        "max_utterance_seconds": settings.max_utterance_seconds,
        "partial_interval_ms": settings.partial_interval_ms,
        "realtime_asr_model": "primary",  # legacy field; the only active strategy always uses 1.7B
        "recognition_architecture": normalize_recognition_architecture(settings.recognition_architecture),
        "audio_segment_minutes": settings.audio_segment_minutes,
        "translation_beam_size": 2,
        "translation_max_decoding_length": 384,
        "translation_repetition_penalty": 1.05,
        "keep_audio": settings.keep_audio,
        "asr_hotwords": normalize_asr_hotwords(settings.asr_hotwords),
    }


def _flatten_meeting_settings(values: object) -> dict[str, Any]:
    if not isinstance(values, dict):
        return {}
    result = dict(values)
    for group in ("audio", "segmentation", "translation"):
        nested = values.get(group)
        if isinstance(nested, dict):
            result.update(nested)
    return result


def normalize_meeting_settings(values: object, settings: Settings) -> dict[str, Any]:
    source = _flatten_meeting_settings(values)
    defaults = default_meeting_settings(settings)
    normalized: dict[str, Any] = {}
    integer_fields = {
        "speech_start_ms", "audio_pre_roll_ms", "silence_ms", "vad_minimum_speech_ms",
        "partial_interval_ms", "audio_segment_minutes", "translation_beam_size",
        "translation_max_decoding_length",
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
        normalized[name] = value.strip().casefold() in {"1", "true", "yes", "on"} if isinstance(value, str) else bool(value)
    requested_model = str(source.get("realtime_asr_model", defaults["realtime_asr_model"]) or "").strip().casefold()
    primary_model = str(settings.asr_primary or "").strip().casefold()
    fallback_model = str(settings.asr_fallback or "").strip().casefold()
    if not getattr(settings, "single_asr_model", True) and requested_model in {"small", "fallback", "0.6b", fallback_model}:
        normalized["realtime_asr_model"] = "small"
    elif requested_model in {"primary", "1.7b", primary_model}:
        normalized["realtime_asr_model"] = "primary"
    else:
        normalized["realtime_asr_model"] = defaults["realtime_asr_model"]
    normalized["recognition_architecture"] = normalize_recognition_architecture(
        source.get("recognition_architecture", defaults["recognition_architecture"]),
        defaults["recognition_architecture"],
    )
    normalized["asr_hotwords"] = normalize_asr_hotwords(source.get("asr_hotwords", defaults["asr_hotwords"]))
    return normalized


def asr_settings_from_meeting(values: object, settings: Settings) -> dict[str, int]:
    normalized = normalize_meeting_settings(values, settings)
    return {
        "silence_ms": int(normalized["silence_ms"]),
        "vad_minimum_speech_ms": int(normalized["vad_minimum_speech_ms"]),
    }


def normalize_asr_settings(values: object, settings: Settings) -> dict[str, int]:
    return asr_settings_from_meeting(values, settings)


def load_settings() -> Settings:
    if load_dotenv:
        load_dotenv()
    defaults = Settings()
    configured_primary = os.getenv("MEETING_ASR_PRIMARY", defaults.asr_primary)
    single_asr_model = _bool("MEETING_SINGLE_ASR_MODEL", defaults.single_asr_model)
    configured_fallback = os.getenv("MEETING_ASR_FALLBACK", defaults.asr_fallback)
    configured_language_id = os.getenv("MEETING_ASR_LANGUAGE_ID", defaults.language_id_model)
    if single_asr_model:
        configured_fallback = configured_primary
        configured_language_id = configured_primary
    return Settings(
        host=os.getenv("MEETING_HOST", defaults.host),
        port=_int("MEETING_PORT", defaults.port, 1),
        api_token=os.getenv("MEETING_API_TOKEN", defaults.api_token),
        device=os.getenv("MEETING_DEVICE", defaults.device),
        asr_primary=configured_primary,
        asr_hotwords=normalize_asr_hotwords(os.getenv("MEETING_ASR_HOTWORDS", "")),
        single_asr_model=single_asr_model,
        asr_fallback=configured_fallback,
        language_id_model=configured_language_id,
        recognition_architecture=normalize_recognition_architecture(
            os.getenv("MEETING_RECOGNITION_ARCHITECTURE", defaults.recognition_architecture),
        ),
        asr_autodownload=_bool("MEETING_ASR_AUTODOWNLOAD", defaults.asr_autodownload),
        vad_model=os.getenv("MEETING_VAD", defaults.vad_model),
        translation_model_root=Path(os.getenv("MEETING_TRANSLATION_MODEL_ROOT", str(defaults.translation_model_root))),
        translation_autodownload=_bool("MEETING_TRANSLATION_AUTODOWNLOAD", defaults.translation_autodownload),
        results_dir=Path(os.getenv("MEETING_RESULTS_DIR", str(defaults.results_dir))),
        audio_segment_minutes=_int("MEETING_AUDIO_SEGMENT_MINUTES", defaults.audio_segment_minutes, 1),
        max_utterance_seconds=min(18.0, max(2.0, _float("MEETING_MAX_UTTERANCE_SECONDS", defaults.max_utterance_seconds, 2.0))),
        audio_pre_roll_ms=min(1000, max(40, _int("MEETING_AUDIO_PRE_ROLL_MS", defaults.audio_pre_roll_ms, 40))),
        speech_start_ms=min(1000, max(40, _int("MEETING_SPEECH_START_MS", defaults.speech_start_ms, 40))),
        silence_ms=min(2000, max(160, _int("MEETING_SILENCE_MS", defaults.silence_ms, 160))),
        vad_minimum_rms=min(5000.0, _float("MEETING_VAD_MINIMUM_RMS", defaults.vad_minimum_rms, 0.0)),
        vad_minimum_speech_ms=min(2000, _int("MEETING_VAD_MINIMUM_SPEECH_MS", defaults.vad_minimum_speech_ms, 0)),
        vad_minimum_speech_ratio=min(1.0, _float("MEETING_VAD_MINIMUM_SPEECH_RATIO", defaults.vad_minimum_speech_ratio, 0.0)),
        partial_interval_ms=min(5000, max(100, _int("MEETING_PARTIAL_INTERVAL_MS", defaults.partial_interval_ms, 100))),
        language_id_min_seconds=min(2.0, max(0.4, _float("MEETING_LANGUAGE_ID_MIN_SECONDS", defaults.language_id_min_seconds, 0.4))),
        language_conflict_confirmations=min(4, max(2, _int("MEETING_LANGUAGE_CONFLICT_CONFIRMATIONS", defaults.language_conflict_confirmations, 2))),
        language_id_on_segment=_bool("MEETING_LANGUAGE_ID_ON_SEGMENT", defaults.language_id_on_segment),
        language_id_on_conflict=_bool("MEETING_LANGUAGE_ID_ON_CONFLICT", defaults.language_id_on_conflict),
        language_switch_window_ms=min(1200, max(200, _int("MEETING_LANGUAGE_SWITCH_WINDOW_MS", defaults.language_switch_window_ms, 200))),
        language_switch_max_wait_ms=min(3000, max(400, _int("MEETING_LANGUAGE_SWITCH_MAX_WAIT_MS", defaults.language_switch_max_wait_ms, 400))),
        stable_prefix_min_chars=min(64, max(1, _int("MEETING_STABLE_PREFIX_MIN_CHARS", defaults.stable_prefix_min_chars, 1))),
        max_audio_packet_bytes=_int("MEETING_MAX_AUDIO_PACKET_BYTES", defaults.max_audio_packet_bytes, 1024),
        max_recording_seconds=min(24 * 60 * 60, max(60.0, _float("MEETING_MAX_RECORDING_SECONDS", defaults.max_recording_seconds, 60.0))),
        audio_ingest_burst_seconds=min(30.0, max(0.0, _float("MEETING_AUDIO_INGEST_BURST_SECONDS", defaults.audio_ingest_burst_seconds, 0.0))),
        inference_queue_size=_int("MEETING_INFERENCE_QUEUE_SIZE", defaults.inference_queue_size, 1),
        translation_queue_size=_int("MEETING_TRANSLATION_QUEUE_SIZE", defaults.translation_queue_size, 1),
        realtime_pipeline=_bool("MEETING_REALTIME_PIPELINE", defaults.realtime_pipeline),
        audio_drain_timeout_seconds=max(0.5, _float("MEETING_AUDIO_DRAIN_TIMEOUT_SECONDS", defaults.audio_drain_timeout_seconds, 0.5)),
        asr_timeout_seconds=max(1.0, _float("MEETING_ASR_TIMEOUT_SECONDS", defaults.asr_timeout_seconds, 1.0)),
        translation_timeout_seconds=max(1.0, _float("MEETING_TRANSLATION_TIMEOUT_SECONDS", defaults.translation_timeout_seconds, 1.0)),
        translation_deadline_seconds=min(30.0, max(1.0, _float("MEETING_TRANSLATION_DEADLINE_SECONDS", defaults.translation_deadline_seconds, 1.0))),
        translation_warmup=_bool("MEETING_TRANSLATION_WARMUP", defaults.translation_warmup),
        translation_partial_debounce_ms=min(2000, max(0, _int("MEETING_TRANSLATION_PARTIAL_DEBOUNCE_MS", defaults.translation_partial_debounce_ms, 0))),
        post_meeting_translation_enabled=_bool(
            "MEETING_POST_TRANSLATION_ENABLED",
            defaults.post_meeting_translation_enabled,
        ),
        post_meeting_translation_quality_threshold=min(
            0.95,
            max(
                0.0,
                _float(
                    "MEETING_POST_TRANSLATION_QUALITY_THRESHOLD",
                    defaults.post_meeting_translation_quality_threshold,
                ),
            ),
        ),
        post_meeting_translation_context_paragraphs=min(
            4,
            max(
                0,
                _int(
                    "MEETING_POST_TRANSLATION_CONTEXT_PARAGRAPHS",
                    defaults.post_meeting_translation_context_paragraphs,
                ),
            ),
        ),
        post_meeting_translation_context_chars=min(
            2000,
            max(
                64,
                _int(
                    "MEETING_POST_TRANSLATION_CONTEXT_CHARS",
                    defaults.post_meeting_translation_context_chars,
                ),
            ),
        ),
        post_meeting_translation_timeout_seconds=min(
            1800.0,
            max(
                5.0,
                _float(
                    "MEETING_POST_TRANSLATION_TIMEOUT_SECONDS",
                    defaults.post_meeting_translation_timeout_seconds,
                ),
            ),
        ),
        queue_join_timeout_seconds=max(1.0, _float("MEETING_QUEUE_JOIN_TIMEOUT_SECONDS", defaults.queue_join_timeout_seconds, 1.0)),
        websocket_send_timeout_seconds=max(0.5, _float("MEETING_WEBSOCKET_SEND_TIMEOUT_SECONDS", defaults.websocket_send_timeout_seconds, 0.5)),
        max_active_meetings=_int("MEETING_MAX_ACTIVE_MEETINGS", defaults.max_active_meetings, 1),
        stream_ticket_ttl_seconds=_int("MEETING_STREAM_TICKET_TTL_SECONDS", defaults.stream_ticket_ttl_seconds, 10),
        websocket_auth_timeout_seconds=_int("MEETING_WEBSOCKET_AUTH_TIMEOUT_SECONDS", defaults.websocket_auth_timeout_seconds, 1),
        websocket_disconnect_grace_seconds=min(120.0, max(0.0, _float("MEETING_WEBSOCKET_DISCONNECT_GRACE_SECONDS", defaults.websocket_disconnect_grace_seconds, 0.0))),
        retention_days=_int("MEETING_RETENTION_DAYS", defaults.retention_days, 0),
        keep_audio=_bool("MEETING_KEEP_AUDIO", defaults.keep_audio),
        jimo_api_url=os.getenv("JIMO_API_URL", defaults.jimo_api_url),
        jimo_todo_api_url=os.getenv("JIMO_TODO_API_URL", defaults.jimo_todo_api_url),
        jimo_authorization=os.getenv("JIMO_AUTHORIZATION", defaults.jimo_authorization),
        jimo_max_request_chars=_int("JIMO_MAX_REQUEST_CHARS", defaults.jimo_max_request_chars, 1000),
        jimo_max_response_chars=_int("JIMO_MAX_RESPONSE_CHARS", defaults.jimo_max_response_chars, 1000),
        jimo_transcript_chars=_int("JIMO_TRANSCRIPT_CHARS", defaults.jimo_transcript_chars, 1000),
        jimo_state_chars=_int("JIMO_STATE_CHARS", defaults.jimo_state_chars, 1000),
        jimo_timeout_seconds=max(1.0, _float("JIMO_TIMEOUT_SECONDS", defaults.jimo_timeout_seconds, 1.0)),
        jimo_connect_timeout_seconds=max(1.0, _float("JIMO_CONNECT_TIMEOUT_SECONDS", defaults.jimo_connect_timeout_seconds, 1.0)),
        jimo_max_retries=min(8, max(1, _int("JIMO_MAX_RETRIES", defaults.jimo_max_retries, 1))),
    )

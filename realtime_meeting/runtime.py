from __future__ import annotations

import gc
import json
import os
import sys
import threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .audio import SAMPLE_RATE, SegmentEvent
from .language import LanguageGuess, VARIANT_LABELS, is_mixed_source_text, normalize_language_code, normalize_qwen_label
from .scheduler import GpuResourceManager
from .text_normalize import simplify_chinese


OPUS_MT_REPOSITORIES = {
    "en": "Helsinki-NLP/opus-mt-en-zh",
    "de": "Helsinki-NLP/opus-mt-de-ZH",
}
OPUS_MT_TARGET_TAGS = {"en": ">>cmn_Hans<<", "de": ">>zh_cn<<"}

# Qwen's 0.6B revision was already used by the project.  The 1.7B revision is
# intentionally resolved from the local Hub cache/latest model card unless a
# deployment pins it separately; an invented hash would be worse than an
# explicit mutable default during model preparation.
MODEL_REVISIONS = {
    "funasr/fsmn-vad": "df20e6b30c653645fa4ff125cacfcabd1020a669",
    "Helsinki-NLP/opus-mt-en-zh": "408d9bc410a388e1d9aef112a2daba955b945255",
    "Helsinki-NLP/opus-mt-de-ZH": "cf77098253bb466b05d2beafd3a3c3dea92ed23b",
    "Qwen/Qwen3-ASR-0.6B": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
}
QWEN_ASR_PRIMARY_MODEL = "Qwen/Qwen3-ASR-1.7B"
QWEN_ASR_SMALL_MODEL = "Qwen/Qwen3-ASR-0.6B"
# Public compatibility constant now follows the production model.  The small
# checkpoint name remains available only for explicit legacy benchmarks.
QWEN_ASR_MODEL = QWEN_ASR_PRIMARY_MODEL
QWEN_ASR_MODELS = {
    "qwen3-asr-0.6b": QWEN_ASR_SMALL_MODEL,
    "qwen3-asr-1.7b": QWEN_ASR_PRIMARY_MODEL,
}
QWEN_ASR_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "de": "German",
}


class StreamingVadAdapter:
    """Combine optional FSMN/WebRTC decisions with the energy gate."""

    chunk_ms = 200

    def __init__(self, fsmn_model: Any | None) -> None:
        self.fsmn_model = fsmn_model
        self.cache: dict[str, Any] = {}
        self.buffer = bytearray()
        self.fsmn_active = False
        self.fsmn_failed = False
        self._rtc_silence_frames = 0
        self.webrtc: Any | None = None
        try:
            import webrtcvad

            self.webrtc = webrtcvad.Vad(2)
        except Exception:
            self.webrtc = None
        components = [name for name, ready in (("fsmn", fsmn_model is not None), ("webrtc", self.webrtc is not None)) if ready]
        self.name = "+".join(components) or "energy"

    def _parse(self, result: Any) -> None:
        payload = result[0] if isinstance(result, list) and result else result
        values = payload.get("value", []) if isinstance(payload, dict) else []
        if not isinstance(values, list):
            return
        for interval in values:
            if not isinstance(interval, (list, tuple)) or len(interval) < 2:
                continue
            try:
                begin, end = float(interval[0]), float(interval[1])
            except (TypeError, ValueError):
                continue
            if begin >= 0:
                self.fsmn_active = True
            if end >= 0:
                self.fsmn_active = False

    def _run_fsmn(self, pcm: bytes, *, is_final: bool) -> None:
        if self.fsmn_model is None or self.fsmn_failed or (not pcm and not is_final):
            return
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            result = self.fsmn_model.generate(
                input=audio,
                cache=self.cache,
                is_final=is_final,
                chunk_size=self.chunk_ms,
            )
            self._parse(result)
        except Exception:
            self.fsmn_failed = True
            self.name = "webrtc" if self.webrtc is not None else "energy"

    def __call__(self, frame: bytes) -> bool | None:
        rtc_speech: bool | None = None
        if self.webrtc is not None:
            try:
                rtc_speech = bool(self.webrtc.is_speech(frame, SAMPLE_RATE))
            except Exception:
                rtc_speech = None
        if rtc_speech:
            self._rtc_silence_frames = 0
        elif rtc_speech is False:
            self._rtc_silence_frames += 1
        # FSMN emits state transitions asynchronously in 200 ms chunks. If a
        # transition-out is lost or delayed, an old ``beg,-1`` state must not
        # keep the segmenter open forever while WebRTC has observed sustained
        # silence. The grace period is deliberately longer than one chunk so
        # quiet speech is still recalled by FSMN.
        if self.fsmn_active and self.webrtc is not None and self._rtc_silence_frames >= 40:
            self.fsmn_active = False
        self.buffer.extend(frame)
        chunk_bytes = self.chunk_ms * SAMPLE_RATE // 1000 * 2
        while len(self.buffer) >= chunk_bytes:
            chunk = bytes(self.buffer[:chunk_bytes])
            del self.buffer[:chunk_bytes]
            self._run_fsmn(chunk, is_final=False)
        if self.fsmn_model is not None and not self.fsmn_failed:
            return self.fsmn_active or bool(rtc_speech)
        return rtc_speech

    def flush(self) -> None:
        self._run_fsmn(bytes(self.buffer), is_final=True)
        self.buffer.clear()
        self.fsmn_active = False
        self._rtc_silence_frames = 0


def opus_mt_repository(source: str) -> str:
    normalized = normalize_language_code(source) or source
    try:
        return OPUS_MT_REPOSITORIES[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported OPUS-MT source language: {source}") from exc


def choose_device(requested: str) -> tuple[str, str]:
    requested = (requested or "auto").strip().casefold()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("MEETING_DEVICE must be auto, cpu or cuda")
    if requested == "cpu":
        return "cpu", "int8"
    try:
        if sys.platform == "win32":
            torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
            if torch_lib.is_dir():
                os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(torch_lib))
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "int8_float16"
    except Exception:
        if requested == "cuda":
            raise RuntimeError("CUDA is not available") from None
    if requested == "cuda":
        raise RuntimeError("CUDA is not available")
    return "cpu", "int8"


def _is_qwen_asr_model(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().casefold()
    short = normalized.rsplit("/", 1)[-1]
    return short in QWEN_ASR_MODELS


def _canonical_qwen_model(model_name: str | None) -> str:
    """Return a stable identity so one Qwen checkpoint can be shared by roles."""
    value = str(model_name or "").strip().casefold()
    short = value.rsplit("/", 1)[-1]
    return QWEN_ASR_MODELS.get(short, value).casefold()


def _model_snapshot(model_name: str, *, local_files_only: bool) -> Path:
    from huggingface_hub import snapshot_download

    short = str(model_name or "").strip().casefold().rsplit("/", 1)[-1]
    repository = QWEN_ASR_MODELS.get(short, model_name)
    kwargs: dict[str, Any] = {"repo_id": repository, "local_files_only": local_files_only}
    revision = MODEL_REVISIONS.get(repository)
    if revision:
        kwargs["revision"] = revision
    return Path(snapshot_download(**kwargs))


@dataclass(slots=True)
class TranslationResult:
    text: str
    status: str
    model: str | None = None
    error: str | None = None


def prepare_opus_mt_model(source: str, root: Path, *, progress: Callable[[str], None] | None = None) -> Path:
    source = normalize_language_code(source) or source
    target = root / f"{source}-zh"
    target.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda _message: None)
    from huggingface_hub import snapshot_download

    repository = opus_mt_repository(source)
    progress(f"downloading OPUS-MT {source}->zh")
    raw = Path(
        snapshot_download(
            repository,
            revision=MODEL_REVISIONS[repository],
            allow_patterns=[
                "config.json", "generation_config.json", "metadata.json", "pytorch_model.bin",
                "model.safetensors", "source.spm", "target.spm", "tokenizer_config.json",
                "vocab.json", "README.md",
            ],
        )
    )
    progress(f"converting OPUS-MT {source}->zh to CTranslate2")
    from ctranslate2.converters import TransformersConverter

    TransformersConverter(str(raw)).convert(str(target), quantization="int8", force=True)
    for source_path in list(raw.glob("*.spm")) + list(raw.glob("*.model")):
        target_path = target / source_path.name
        target_path.write_bytes(source_path.read_bytes())
    raw_config = next((raw / name for name in ("config.json", "model.json") if (raw / name).is_file()), None)
    if raw_config is not None:
        (target / "source_config.json").write_bytes(raw_config.read_bytes())
    raw_readme = raw / "README.md"
    if raw_readme.is_file():
        (target / "source_model_card.md").write_bytes(raw_readme.read_bytes())
    (target / "meeting_model.json").write_text(
        json.dumps(
            {
                "format": "ctranslate2",
                "source": source,
                "target": "zh",
                "target_language_tag": OPUS_MT_TARGET_TAGS[source],
                "repository": repository,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


class RealtimeAsrEngine:
    """Qwen3-ASR model holder used by both the 1.7B and 0.6B models."""

    def __init__(self, model_name: str, gpu_manager: GpuResourceManager, *, autodownload: bool = True) -> None:
        self.model_name = model_name
        self.gpu_manager = gpu_manager
        self.autodownload = autodownload
        self.model: Any | None = None
        self.device = "cpu"

    def load(self, device: str, compute_type: str = "int8") -> Any:
        del compute_type
        self.device = device
        if not _is_qwen_asr_model(self.model_name):
            raise RuntimeError(f"only Qwen3-ASR models are supported: {self.model_name}")
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError("Qwen ASR requires qwen-asr; run uv sync --extra audio") from exc
        snapshot = _model_snapshot(self.model_name, local_files_only=not self.autodownload)
        with self.gpu_manager.acquire_sync("qwen_model_load"):
            self.model = Qwen3ASRModel.from_pretrained(
                str(snapshot),
                dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="cuda:0" if device == "cuda" else "cpu",
                max_inference_batch_size=1,
                max_new_tokens=256,
            )
        return self.model

    def release(self) -> None:
        self.model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class LanguageIdEngine(RealtimeAsrEngine):
    """Qwen model holder used when language ID needs its own checkpoint."""


class LiveChineseTranslator:
    """Local OPUS-MT en/de -> zh translator."""

    CACHE_LIMIT = 4096

    def __init__(
        self,
        model_root: Path | None,
        device: str,
        progress: Callable[[str], None] | None = None,
        autodownload: bool = False,
        gpu_manager: GpuResourceManager | None = None,
    ) -> None:
        self.model_root = model_root
        self.device = device
        self.progress = progress or (lambda _message: None)
        self.autodownload = autodownload
        self.gpu_manager = gpu_manager
        self.models: dict[str, tuple[Any, Any, Any, str, str]] = {}
        self.failed: dict[str, str] = {}
        self.cache: dict[tuple[str, str, int, int, float], TranslationResult] = {}
        self._lock = threading.RLock()
        # Optional model downloads are allowed only during explicit startup
        # preflight.  A live translation request must never reach the network.
        self._startup_preflight = False

    def warmup(self) -> None:
        """Load both local translation models and run a tiny decode.

        File preflight alone does not initialize CTranslate2.  Doing that on
        the first live paragraph is the main source of the apparent first
        translation delay, so startup owns the cold-load cost instead.
        """

        for source, text in (("en", "warm up"), ("de", "Aufwärmen")):
            loaded = self._load(source)
            if loaded is None:
                continue
            source_sp, _target_sp, translator, _model_id, target_tag = loaded
            pieces = [[target_tag] + source_sp.encode(text, out_type=str) + ["</s>"]]
            context = self.gpu_manager.acquire_sync("translation_warmup", priority=50) if self.gpu_manager else _null_context()
            with context:
                translator.translate_batch(pieces, beam_size=1, max_decoding_length=32, repetition_penalty=1.0)

    def _find(self, source: str) -> Path | None:
        if not self.model_root:
            return None
        for candidate in (self.model_root / f"{source}-zh", self.model_root / f"{source}_zh", self.model_root / source):
            if (candidate / "model.bin").is_file() and self._spm_paths(candidate) and self._metadata_path(candidate):
                return candidate
        return None

    @staticmethod
    def _metadata_path(path: Path) -> Path | None:
        for name in ("config.json", "model.json", "meeting_model.json", "config.yml", "config.yaml"):
            candidate = path / name
            if candidate.is_file():
                return candidate
        return None

    def _download_if_needed(self, source: str) -> Path | None:
        path = self._find(source)
        if path is not None or not self.autodownload or self.model_root is None or not self._startup_preflight:
            return path
        try:
            prepare_opus_mt_model(source, self.model_root, progress=self.progress)
            return self._find(source)
        except Exception as exc:  # pragma: no cover - network/model dependent
            self.failed[source] = str(exc)
            self.progress(f"translation model {source} download failed: {exc}")
            return None

    def assets_snapshot(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for source in ("en", "de"):
            path = self._find(source)
            error = self.failed.get(source)
            result[source] = {
                "model": opus_mt_repository(source),
                "path": str(path or (self.model_root / f"{source}-zh" if self.model_root else "")),
                "ready": path is not None,
                "status": "ready" if path is not None else "failed" if error else "pending",
                "error": error,
            }
        return result

    def preflight(self) -> dict[str, dict[str, Any]]:
        self._startup_preflight = True
        try:
            for source in ("en", "de"):
                path = self._download_if_needed(source)
                if path is None:
                    self.failed.setdefault(source, "model not cached")
        finally:
            self._startup_preflight = False
        return self.assets_snapshot()

    @staticmethod
    def _spm_paths(path: Path) -> tuple[Path, Path] | None:
        source, target = path / "source.spm", path / "target.spm"
        if source.is_file() and target.is_file():
            return source, target
        for name in ("sentencepiece.bpe.model", "spm.model"):
            candidate = path / name
            if candidate.is_file():
                return candidate, candidate
        matches = list(path.glob("*.spm")) + list(path.glob("*.model"))
        return (matches[0], matches[0]) if matches else None

    def _load(self, source: str) -> tuple[Any, Any, Any, str, str] | None:
        if source in self.models:
            return self.models[source]
        path = self._download_if_needed(source)
        if path is None:
            self.failed[source] = self.failed.get(source, "model not cached")
            return None
        try:
            import ctranslate2
            import sentencepiece as spm

            paths = self._spm_paths(path)
            if paths is None:
                raise RuntimeError("missing SentencePiece model")
            source_sp = spm.SentencePieceProcessor(model_file=str(paths[0]))
            target_sp = spm.SentencePieceProcessor(model_file=str(paths[1]))
            translator = ctranslate2.Translator(
                str(path), device=self.device, compute_type="int8_float16" if self.device == "cuda" else "int8"
            )
            metadata_path = path / "meeting_model.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
            target_tag = str(metadata.get("target_language_tag") or OPUS_MT_TARGET_TAGS.get(source, ""))
            result = (source_sp, target_sp, translator, str(path), target_tag)
            self.models[source] = result
            return result
        except Exception as exc:  # pragma: no cover - optional dependency
            self.failed[source] = str(exc)
            self.progress(f"translation model {source} is unavailable: {exc}")
            return None

    def translate_many(self, texts: list[str], source: str, settings: dict[str, Any] | None = None) -> list[TranslationResult]:
        source = normalize_language_code(source) or source
        values = settings if isinstance(settings, dict) else {}
        try:
            beam_size = max(1, min(8, int(float(values.get("translation_beam_size", 2)))))
        except (TypeError, ValueError):
            beam_size = 2
        try:
            max_decoding_length = max(64, min(1024, int(float(values.get("translation_max_decoding_length", 384)))))
        except (TypeError, ValueError):
            max_decoding_length = 384
        try:
            repetition_penalty = max(1.0, min(2.0, float(values.get("translation_repetition_penalty", 1.05))))
        except (TypeError, ValueError):
            repetition_penalty = 1.05
        cache_suffix = (beam_size, max_decoding_length, repetition_penalty)
        results: list[TranslationResult | None] = [None] * len(texts)
        pending: list[tuple[int, str]] = []
        for index, text in enumerate(texts):
            value = str(text or "").strip()
            if not value:
                results[index] = TranslationResult("", "ready")
            elif source == "zh":
                results[index] = TranslationResult(simplify_chinese(value), "not_needed")
            elif source not in {"en", "de"}:
                results[index] = TranslationResult("", "unsupported", error=f"unsupported source language: {source}")
            else:
                cache_key = (source, value, *cache_suffix)
                with self._lock:
                    cached = self.cache.get(cache_key)
                if cached is not None:
                    results[index] = cached
                else:
                    pending.append((index, value))
        loaded = self._load(source) if pending else None
        if loaded is None and pending:
            error = self.failed.get(source, "model unavailable")
            for index, _value in pending:
                results[index] = TranslationResult("", "failed", error=error)
        elif loaded:
            source_sp, target_sp, translator, model_id, target_tag = loaded
            try:
                pieces = [[target_tag] + source_sp.encode(value, out_type=str) + ["</s>"] for _, value in pending]
                context = self.gpu_manager.acquire_sync(
                    "translation",
                    priority=max(1, int(float(values.get("_gpu_priority", 40)))),
                ) if self.gpu_manager else _null_context()
                with context:
                    with self._lock:
                        generated = translator.translate_batch(
                            pieces,
                            beam_size=beam_size,
                            max_decoding_length=max_decoding_length,
                            repetition_penalty=repetition_penalty,
                        )
                for (index, value), hypothesis in zip(pending, generated):
                    tokens = list(hypothesis.hypotheses[0])
                    translated = simplify_chinese(target_sp.decode(tokens).strip())
                    results[index] = TranslationResult(translated, "ready", model_id) if translated else TranslationResult("", "failed", model_id, "empty translation")
            except Exception as exc:  # pragma: no cover - optional dependency
                self.progress(f"translation failed: {exc}")
                for index, _value in pending:
                    results[index] = TranslationResult("", "failed", model_id, str(exc))
        final = [item or TranslationResult("", "failed", error="translator returned no result") for item in results]
        with self._lock:
            for index, value in pending:
                if final[index].status == "ready":
                    self.cache[(source, value, *cache_suffix)] = final[index]
            while len(self.cache) > self.CACHE_LIMIT:
                self.cache.pop(next(iter(self.cache)))
        return final


TranslationEngine = LiveChineseTranslator


@dataclass(slots=True)
class PartialResult:
    revision: int
    start: float
    end: float
    text: str
    language: str
    confidence: float = 0.0
    model: str | None = None
    language_source: str = "qwen"
    speech_variant: str | None = None
    raw_qwen_label: str = ""


class LiveModelRuntime:
    """Resident Qwen3-ASR runtime with explicit role sharing for model variants."""

    def __init__(
        self,
        asr_primary: str = QWEN_ASR_PRIMARY_MODEL,
        asr_fallback: str = QWEN_ASR_SMALL_MODEL,
        requested_device: str = "auto",
        *legacy_args: Any,
        asr_autodownload: bool = False,
        translation_model_root: Path | None = None,
        translation_autodownload: bool = False,
        translation_warmup: bool = True,
        vad_model: str = "fsmn-vad",
        language_id_model: str | None = None,
        single_model: bool = False,
        **_legacy_options: Any,
    ) -> None:
        if requested_device not in {"auto", "cpu", "cuda"} and legacy_args and legacy_args[0] in {"auto", "cpu", "cuda"}:
            requested_device = legacy_args[0]
        self.asr_primary_name = asr_primary or QWEN_ASR_PRIMARY_MODEL
        self.single_model = bool(single_model)
        self.asr_fallback_name = self.asr_primary_name if self.single_model else (asr_fallback or QWEN_ASR_SMALL_MODEL)
        self.language_id_name = self.asr_primary_name if self.single_model else (language_id_model or self.asr_fallback_name)
        self.requested_device = requested_device
        self.asr_autodownload = asr_autodownload
        self.translation_model_root = translation_model_root
        self.translation_autodownload = translation_autodownload
        self.translation_warmup = bool(translation_warmup)
        self.vad_model_name = vad_model
        self.device = "cpu"
        self.compute_type = "int8"
        self.gpu_manager = GpuResourceManager()
        self.primary_engine = RealtimeAsrEngine(self.asr_primary_name, self.gpu_manager, autodownload=asr_autodownload)
        self.small_engine = LanguageIdEngine(self.asr_fallback_name, self.gpu_manager, autodownload=asr_autodownload)
        self.language_id_engine = (
            self.small_engine
            if _canonical_qwen_model(self.language_id_name) == _canonical_qwen_model(self.asr_fallback_name)
            else LanguageIdEngine(self.language_id_name, self.gpu_manager, autodownload=asr_autodownload)
        )
        self.primary: Any | None = None
        self.fallback: Any | None = None
        self.language_id: Any | None = None
        self.translator: LiveChineseTranslator | None = None
        self.vad: Any | None = None
        self.ready = False
        self.capabilities_ready = False
        self.status = "waiting for model load"
        self.metrics: dict[str, Any] = {
            "asr_calls": 0,
            "translation_calls": 0,
            "language_id_calls": 0,
            "language_id_failures": 0,
            "fallback_count": 0,
            "stage_failures": 0,
            "model_events": [],
        }
        self._model_lock = threading.RLock()
        self.asr_cache_ready = False
        self.realtime_cache_ready = False
        self.language_id_cache_ready = False
        self.last_asr_error: str | None = None
        # A timed-out asyncio wrapper must never allow a second call to enter
        # the same model concurrently while the original thread is still
        # unwinding.  Dedicated one-thread executors provide that safety while
        # the GPU manager arbitrates ASR, LID and translation across stages.
        self.inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="meeting-asr")
        self.translation_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="meeting-translation")

    def _event(self, name: str, **details: Any) -> None:
        events = self.metrics.setdefault("model_events", [])
        if isinstance(events, list):
            events.append({"event": name, **details})
            del events[:-50]

    def _prepare_model_cache(self, model_name: str, progress: Callable[[str], None]) -> bool:
        try:
            progress(f"checking Qwen ASR model cache: {model_name}")
            _model_snapshot(model_name, local_files_only=not self.asr_autodownload)
            return True
        except Exception as exc:
            self._event("asr_preflight_error", model=model_name, error=str(exc))
            return False

    def _resolve_vad_source(self) -> str:
        configured = str(self.vad_model_name or "").strip()
        if configured in {"", "disabled"} or Path(configured).exists() or configured != "fsmn-vad":
            return configured
        try:
            from huggingface_hub import snapshot_download

            return str(
                snapshot_download(
                    repo_id="funasr/fsmn-vad",
                    revision=MODEL_REVISIONS["funasr/fsmn-vad"],
                    local_files_only=not self.asr_autodownload,
                )
            )
        except Exception:
            if not self.asr_autodownload:
                raise RuntimeError("fsmn-vad is not present in the local model cache") from None
            return configured

    def load(self, progress: Callable[[str], None] | None = None) -> None:
        progress = progress or (lambda _message: None)
        self.device, self.compute_type = choose_device(self.requested_device)
        try:
            primary_ready = self._prepare_model_cache(self.asr_primary_name, progress)
            primary_identity = _canonical_qwen_model(self.asr_primary_name)
            fallback_identity = _canonical_qwen_model(self.asr_fallback_name)
            fallback_ready = (
                primary_ready
                if fallback_identity == primary_identity
                else self._prepare_model_cache(self.asr_fallback_name, progress)
            )
            language_id_identity = _canonical_qwen_model(self.language_id_name)
            language_id_ready = (
                True
                if language_id_identity in {primary_identity, fallback_identity}
                else self._prepare_model_cache(self.language_id_name, progress)
            )
            if not language_id_ready:
                raise RuntimeError(f"language ID Qwen model unavailable: {self.language_id_name}")
            try:
                progress(f"loading realtime ASR: {self.asr_primary_name}")
                self.primary = self.primary_engine.load(self.device, self.compute_type)
            except Exception as exc:
                self._event("primary_load_error", model=self.asr_primary_name, error=str(exc))
                self.primary = None
            if fallback_identity == primary_identity and self.primary is not None:
                progress(f"reusing realtime ASR for fallback: {self.asr_fallback_name}")
                self.fallback = self.primary
            elif fallback_ready:
                progress(f"loading fallback Qwen ASR: {self.asr_fallback_name}")
                try:
                    self.fallback = self.small_engine.load(self.device, self.compute_type)
                except Exception as exc:
                    self._event("fallback_load_error", model=self.asr_fallback_name, error=str(exc))
                    self.fallback = None
            else:
                self._event("fallback_unavailable", model=self.asr_fallback_name)
            if language_id_identity == primary_identity and self.primary is not None:
                self.language_id = self.primary
            elif language_id_identity == fallback_identity and self.fallback is not None:
                self.language_id = self.fallback
            elif language_id_ready:
                progress(f"loading language ID Qwen ASR: {self.language_id_name}")
                try:
                    self.language_id = self.language_id_engine.load(self.device, self.compute_type)
                except Exception as exc:
                    self._event("language_id_load_error", model=self.language_id_name, error=str(exc))
                    self.language_id = None
            del primary_ready
            self.realtime_cache_ready = self.primary is not None or self.fallback is not None
            self.asr_cache_ready = self.realtime_cache_ready
            self.language_id_cache_ready = self.language_id is not None
            self.translator = LiveChineseTranslator(
                self.translation_model_root,
                self.device,
                progress,
                self.translation_autodownload,
                self.gpu_manager,
            )
            self.translator.preflight()
            if self.translation_warmup:
                try:
                    progress("warming local translation models")
                    warmup = getattr(self.translator, "warmup", None)
                    if callable(warmup):
                        warmup()
                except Exception as exc:
                    self._event("translation_warmup_error", error=str(exc))
            if self.vad_model_name != "disabled":
                try:
                    from funasr import AutoModel

                    runtime_device = f"{self.device}:0" if self.device == "cuda" else self.device
                    self.vad = AutoModel(
                        model=self._resolve_vad_source(),
                        hub="hf",
                        trust_remote_code=True,
                        disable_update=True,
                        device=runtime_device,
                    )
                except Exception as exc:
                    self._event("vad_load_error", error=str(exc))
                    self.vad = None
            self.ready = True
            vad_ready = self.vad is not None or self.vad_model_name == "disabled"
            # Translation is an optional local stage.  Missing OPUS-MT files
            # must be visible in capabilities, but cannot prevent ASR from
            # accepting a meeting or closing its final transcript.
            self.capabilities_ready = bool(self.asr_cache_ready and vad_ready)
            issues = self._capability_issues()
            self.status = "models ready" if not issues else "ASR ready; missing capabilities: " + "; ".join(issues)
            self._event("ready", device=self.device, capabilities_ready=self.capabilities_ready)
            progress(self.status)
        except Exception as exc:
            self.ready = False
            self.capabilities_ready = False
            self.status = f"model load failed: {exc}"
            self._event("load_error", error=str(exc))
            progress(self.status)

    def _capability_issues(self) -> list[str]:
        issues: list[str] = []
        if not self.primary and not self.fallback:
            issues.append("Qwen ASR unavailable")
        if self.vad is None and self.vad_model_name != "disabled":
            issues.append("VAD unavailable")
        if self.translator:
            for source, item in self.translator.assets_snapshot().items():
                if not item["ready"]:
                    issues.append(f"translation {source}->zh unavailable")
        return issues

    def close(self) -> None:
        with self._model_lock:
            released_engines: set[int] = set()
            for engine in (self.primary_engine, self.small_engine, self.language_id_engine):
                if id(engine) in released_engines:
                    continue
                released_engines.add(id(engine))
                engine.release()
            self.primary = self.fallback = self.language_id = None
            self.inference_executor.shutdown(wait=False, cancel_futures=True)
            self.translation_executor.shutdown(wait=False, cancel_futures=True)
        self.translator = None
        self.vad = None
        self.ready = False
        self.capabilities_ready = False
        self.status = "models released"

    def warm_realtime(self, progress: Callable[[str], None] | None = None) -> None:
        # The realtime models stay resident for the entire process; stopping a
        # meeting only flushes queues and never tears down model state.
        del progress
        if self.translator and self.translation_warmup:
            try:
                warmup = getattr(self.translator, "warmup", None)
                if callable(warmup):
                    warmup()
            except Exception as exc:
                self._event("translation_warmup_error", error=str(exc))
        return

    def new_vad(self) -> Callable[[bytes], bool | None] | None:
        if self.vad_model_name == "disabled":
            return None
        return StreamingVadAdapter(self.vad)

    @staticmethod
    def _audio(pcm: bytes) -> np.ndarray:
        return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    @staticmethod
    def _has_speech_text(text: str) -> bool:
        return bool(str(text or "").strip())

    @staticmethod
    def _prompt(recent_text: str, speech_variant: str | None, hotwords: list[str] | None) -> str:
        parts: list[str] = []
        if speech_variant and speech_variant != "mandarin":
            parts.append(f"请保留{VARIANT_LABELS.get(speech_variant, speech_variant)}的原意和表达。")
        if hotwords:
            parts.append("专有词：" + "、".join(hotwords[:100]))
        if recent_text.strip():
            parts.append(recent_text.strip()[-240:])
        return "\n".join(parts)

    @staticmethod
    def _forced_language(language: str | None, speech_variant: str | None) -> str | None:
        if speech_variant and speech_variant.startswith("cantonese"):
            return "Cantonese"
        return QWEN_ASR_LANGUAGE_NAMES.get(language or "")

    def _qwen_decode(self, pcm: bytes, model: Any, *, prompt: str = "", language: str | None = None, speech_variant: str | None = None) -> tuple[str, str]:
        if model is None or not hasattr(model, "transcribe"):
            raise RuntimeError("Qwen ASR model is not loaded")
        kwargs: dict[str, Any] = {"audio": (self._audio(pcm), SAMPLE_RATE)}
        forced = self._forced_language(language, speech_variant)
        if forced:
            kwargs["language"] = forced
        if prompt:
            kwargs["context"] = prompt
        try:
            results = model.transcribe(**kwargs)
        except TypeError:
            # Older qwen-asr wrappers used ``prompt`` instead of ``context``.
            if "context" not in kwargs:
                raise
            kwargs.pop("context", None)
            if prompt:
                kwargs["prompt"] = prompt
            results = model.transcribe(**kwargs)
        first = results[0] if isinstance(results, (list, tuple)) and results else results
        if isinstance(first, dict):
            return str(first.get("text", "") or "").strip(), str(first.get("language", "") or "")
        return str(getattr(first, "text", "") or "").strip(), str(getattr(first, "language", "") or "")

    def detect_language(self, pcm: bytes, *, previous_language: str | None = None) -> LanguageGuess | None:
        del previous_language
        model = self.language_id or self.fallback
        if model is None:
            return None
        try:
            with self.gpu_manager.acquire_sync("qwen_language_id", priority=10) if self.device == "cuda" else _null_context():
                text, raw = self._qwen_decode(pcm, model)
            self.metrics["language_id_calls"] = int(self.metrics.get("language_id_calls", 0)) + 1
            guess = normalize_qwen_label(raw, text)
            return guess if guess.code != "unknown" else None
        except Exception as exc:
            self.metrics["language_id_failures"] = int(self.metrics.get("language_id_failures", 0)) + 1
            self._event("language_id_error", model=self.language_id_name, error=str(exc))
            return None

    def _recognize(
        self,
        event: SegmentEvent,
        *,
        language: str | None,
        speech_variant: str | None,
        recent_text: str,
        decode_settings: dict[str, Any] | None,
    ) -> PartialResult:
        values = decode_settings if isinstance(decode_settings, dict) else {}
        hotwords = [str(item) for item in values.get("asr_hotwords", []) if str(item).strip()]
        prompt = self._prompt(recent_text, speech_variant, hotwords)
        requested_model = str(values.get("realtime_asr_model", "primary") or "").strip().casefold()
        small_requested = requested_model in {
            "small",
            "fallback",
            "0.6b",
            self.asr_fallback_name.casefold(),
        }
        if self.single_model:
            # Meeting settings from an older client may still request the
            # legacy small/fallback role. Production single-model mode must
            # never route that request to a second checkpoint.
            small_requested = False
        if small_requested:
            model = self.fallback
            model_name = self.asr_fallback_name
            alternate_model = self.primary
            alternate_name = self.asr_primary_name
            model_lock = "qwen_asr_fallback"
            alternate_lock = "qwen_asr"
        else:
            model = self.primary
            model_name = self.asr_primary_name
            alternate_model = self.fallback
            alternate_name = self.asr_fallback_name
            model_lock = "qwen_asr"
            alternate_lock = "qwen_asr_fallback"
        try:
            if model is None:
                raise RuntimeError(f"Qwen ASR model is not loaded: {model_name}")
            # Lower number means higher priority: final ASR > language
            # switch confirmation > final translation > partial ASR.
            asr_priority = 5 if event.kind == "final" else 40
            with self.gpu_manager.acquire_sync(model_lock, priority=asr_priority) if self.device == "cuda" else _null_context():
                text, raw = self._qwen_decode(model=model, pcm=event.pcm, prompt=prompt, language=language, speech_variant=speech_variant)
        except Exception as exc:
            self.metrics["fallback_count"] = int(self.metrics.get("fallback_count", 0)) + 1
            self._event("asr_fallback", from_model=model_name, to_model=alternate_name, error=str(exc))
            self.last_asr_error = str(exc)
            model = alternate_model
            model_name = alternate_name
            try:
                if model is None:
                    raise RuntimeError(f"Qwen ASR model is not loaded: {model_name}")
                with self.gpu_manager.acquire_sync(alternate_lock, priority=asr_priority) if self.device == "cuda" else _null_context():
                    text, raw = self._qwen_decode(model=model, pcm=event.pcm, prompt=prompt, language=language, speech_variant=speech_variant)
            except Exception as fallback_exc:
                self.metrics["stage_failures"] = int(self.metrics.get("stage_failures", 0)) + 1
                self._event("asr_error", model=model_name, error=str(fallback_exc))
                return PartialResult(event.revision, event.start, event.end, "", "unknown", model=model_name, raw_qwen_label=raw if "raw" in locals() else "")
        self.metrics["asr_calls"] = int(self.metrics.get("asr_calls", 0)) + 1
        model_usage = self.metrics.setdefault("asr_model_usage", {})
        if isinstance(model_usage, dict):
            model_usage[model_name] = int(model_usage.get(model_name, 0)) + 1
        guess = normalize_qwen_label(raw, text)
        if guess.code == "unknown" and language in {"zh", "en", "de"}:
            guess = LanguageGuess(language, 0.5, speech_variant, raw_qwen_label=raw)
        return PartialResult(
            event.revision,
            event.start,
            event.end,
            text,
            guess.code,
            guess.confidence,
            model_name,
            "qwen",
            guess.speech_variant or speech_variant,
            guess.raw_qwen_label,
        )

    def transcribe_partial(
        self,
        event: SegmentEvent,
        *,
        recent_text: str = "",
        previous_language: str | None = None,
        language: str | None = None,
        speech_variant: str | None = None,
        decode_settings: dict[str, Any] | None = None,
        **_unused: Any,
    ) -> PartialResult:
        del previous_language
        return self._recognize(
            event,
            language=language,
            speech_variant=speech_variant,
            recent_text=recent_text,
            decode_settings=decode_settings,
        )

    def transcribe_final(self, event: SegmentEvent, **kwargs: Any) -> PartialResult:
        return self.transcribe_partial(event, **kwargs)

    def translate_text(self, text: str, language: str, *, translation_settings: dict[str, Any] | None = None, **_unused: Any) -> TranslationResult:
        if language == "zh" and not is_mixed_source_text(text):
            return TranslationResult(simplify_chinese(text), "not_needed")
        if language == "zh" and is_mixed_source_text(text):
            guessed = normalize_qwen_label("unknown", text).code
            language = guessed if guessed in {"en", "de"} else "en"
        if self.translator is None:
            return TranslationResult("", "failed", error="translator unavailable")
        self.metrics["translation_calls"] = int(self.metrics.get("translation_calls", 0)) + 1
        return self.translator.translate_many([text], language, translation_settings)[0]

    def translate_text_batch(self, texts: list[str], language: str, *, translation_settings: dict[str, Any] | None = None, **_unused: Any) -> list[TranslationResult]:
        if language == "zh":
            return [TranslationResult(simplify_chinese(text), "not_needed") for text in texts]
        if self.translator is None:
            return [TranslationResult("", "failed", error="translator unavailable") for _ in texts]
        self.metrics["translation_calls"] = int(self.metrics.get("translation_calls", 0)) + len(texts)
        return self.translator.translate_many(texts, language, translation_settings)

    def capability_snapshot(self) -> dict[str, Any]:
        return {
            "single_model": self.single_model,
            "asr_primary": {"model": self.asr_primary_name, "ready": self.primary is not None},
            "asr_fallback": {"model": self.asr_fallback_name, "ready": self.fallback is not None},
            "language_id": {"model": self.language_id_name, "ready": self.language_id is not None},
            "vad": {"model": self.vad_model_name, "ready": self.vad is not None or self.vad_model_name == "disabled"},
            "translation": self.translator.assets_snapshot() if self.translator else {},
        }


class _null_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        return None

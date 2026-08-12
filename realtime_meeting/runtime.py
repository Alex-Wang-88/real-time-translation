from __future__ import annotations

import gc
import os
import re
import sys
import threading
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .audio import SAMPLE_RATE, SegmentEvent
from .language import (
    SUPPORTED_LANGUAGE_CODES,
    MultilingualDetector,
    normalize_language_code,
)
from .models import Utterance
from .speaker import OnlineSpeakerClusterer
from .text_normalize import simplify_chinese


# Kept as a compatibility map for old integrations and archived exports. New
# production translation uses pair-specific OPUS-MT models instead of NLLB.
NLLB_CODES = {
    "zh": "zho_Hans", "en": "eng_Latn", "de": "deu_Latn",
    "ru": "rus_Cyrl", "es": "spa_Latn", "pt": "por_Latn",
    "fr": "fra_Latn", "it": "ita_Latn", "ja": "jpn_Jpan",
    "ko": "kor_Hang", "ar": "arb_Arab", "uk": "ukr_Cyrl",
    "pl": "pol_Latn", "nl": "nld_Latn", "tr": "tur_Latn",
    "vi": "vie_Latn", "id": "ind_Latn", "th": "tha_Thai",
    "cs": "ces_Latn", "sv": "swe_Latn", "da": "dan_Latn",
    "no": "nob_Latn", "fi": "fin_Latn", "el": "ell_Grek",
    "he": "heb_Hebr", "hi": "hin_Deva", "bn": "ben_Beng",
    "fa": "pes_Arab", "ur": "urd_Arab", "ro": "ron_Latn",
    "hu": "hun_Latn", "bg": "bul_Cyrl", "sr": "srp_Cyrl",
    "hr": "hrv_Latn", "sk": "slk_Latn", "sl": "slv_Latn",
    "lt": "lit_Latn", "lv": "lvs_Latn", "et": "est_Latn",
    "ms": "zsm_Latn", "tl": "tgl_Latn", "sw": "swh_Latn",
    "af": "afr_Latn", "am": "amh_Ethi", "yo": "yor_Latn",
    "as": "asm_Beng", "az": "azj_Latn", "ba": "bak_Cyrl",
    "be": "bel_Cyrl", "bo": "bod_Tibt", "bs": "bos_Latn",
    "ca": "cat_Latn", "cy": "cym_Latn", "eu": "eus_Latn",
    "fo": "fao_Latn", "gl": "glg_Latn", "gu": "guj_Gujr",
    "ha": "hau_Latn", "ht": "hat_Latn", "hy": "hye_Armn",
    "is": "isl_Latn", "jw": "jav_Latn", "ka": "kat_Geor",
    "kk": "kaz_Cyrl", "km": "khm_Khmr", "kn": "kan_Knda",
    "lb": "ltz_Latn", "ln": "lin_Latn", "lo": "lao_Laoo",
    "mg": "plt_Latn", "mi": "mri_Latn", "mk": "mkd_Cyrl",
    "ml": "mal_Mlym", "mn": "khk_Cyrl", "mr": "mar_Deva",
    "mt": "mlt_Latn", "my": "mya_Mymr", "ne": "npi_Deva",
    "nn": "nno_Latn", "oc": "oci_Latn", "pa": "pan_Guru",
    "ps": "pbt_Arab", "sa": "san_Deva", "sd": "snd_Arab",
    "si": "sin_Sinh", "sn": "sna_Latn", "so": "som_Latn",
    "sq": "als_Latn", "su": "sun_Latn", "ta": "tam_Taml",
    "te": "tel_Telu", "tg": "tgk_Cyrl", "tk": "tuk_Latn",
    "tt": "tat_Cyrl", "uz": "uzn_Latn", "yi": "ydd_Hebr",
    "yue": "yue_Hant", "eo": "epo_Latn", "ga": "gle_Latn",
    "lg": "lug_Latn", "nb": "nob_Latn", "st": "sot_Latn",
    "tn": "tsn_Latn", "ts": "tso_Latn", "xh": "xho_Latn",
    "zu": "zul_Latn",
}


OPUS_MT_MODEL_IDS = {
    "en": "Helsinki-NLP/opus-mt-en-zh",
    "de": "Helsinki-NLP/opus-mt-de-zh",
    "ja": "Helsinki-NLP/opus-mt-ja-zh",
    "ko": "Helsinki-NLP/opus-mt-ko-zh",
    "fr": "Helsinki-NLP/opus-mt-fr-zh",
    "es": "Helsinki-NLP/opus-mt-es-zh",
    "ru": "Helsinki-NLP/opus-mt-ru-zh",
}


def choose_device(requested: str) -> tuple[str, str]:
    requested = (requested or "auto").strip().casefold()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("MEETING_DEVICE 必须是 auto、cpu 或 cuda")
    if requested == "cpu":
        return "cpu", "int8"
    if requested in {"auto", "cuda"}:
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
                raise RuntimeError(
                    "CTranslate2 未发现可用 CUDA，请检查 CUDA 12、cuBLAS 和 cuDNN 9"
                )
        if requested == "cuda":
            raise RuntimeError(
                "CTranslate2 未发现可用 CUDA，请检查 CUDA 12、cuBLAS 和 cuDNN 9"
            )
    return "cpu", "int8"


@dataclass(slots=True)
class TranslationResult:
    text: str
    status: str
    model: str | None = None


@dataclass(slots=True)
class _LoadedTranslationPair:
    source_sp: Any
    target_sp: Any
    translator: Any
    model_id: str


class LiveChineseTranslator:
    """Lazy, pair-specific OPUS-MT translator with safe pass-through fallback."""

    def __init__(
        self,
        model_name: str,
        device: str,
        progress: Callable[[str], None],
        *,
        model_root: Path | None = None,
        autodownload: bool = False,
    ) -> None:
        self.model_name = model_name or "opusmt-local"
        self.device = device
        self.progress = progress
        self.model_root = model_root
        self.autodownload = autodownload
        self.models: dict[str, _LoadedTranslationPair] = {}
        self.failed_sources: set[str] = set()
        self.cache: dict[tuple[str, str], TranslationResult] = {}
        self.cache_limit = 2_048
        progress("正在准备本地 OPUS-MT 翻译后端")

    def _local_candidates(self, source: str) -> list[Path]:
        candidates: list[Path] = []
        if self.model_root:
            candidates.extend(
                [
                    self.model_root / f"{source}-zh",
                    self.model_root / f"{source}_zh",
                    self.model_root / source,
                ]
            )
        configured = Path(self.model_name)
        if configured.exists():
            candidates.extend(
                [configured / f"{source}-zh", configured / source, configured]
            )
        return candidates

    @staticmethod
    def _sentencepiece_paths(model_path: Path) -> tuple[Path | None, Path | None]:
        source = model_path / "source.spm"
        target = model_path / "target.spm"
        if source.is_file() or target.is_file():
            source_path = source if source.is_file() else target
            target_path = target if target.is_file() else source
            return source_path, target_path
        for name in ("sentencepiece.bpe.model", "spm.model"):
            candidate = model_path / name
            if candidate.is_file():
                return candidate, candidate
        matches = list(model_path.glob("*.spm")) + list(model_path.glob("*.model"))
        if matches:
            return matches[0], matches[0]
        return None, None

    def _resolve_model_path(self, source: str) -> tuple[Path | None, str | None]:
        for candidate in self._local_candidates(source):
            if candidate.is_dir() and (candidate / "model.bin").exists():
                return candidate, str(candidate)
        configured = self.model_name.strip()
        if configured not in {"", "opusmt-local"} and "/" in configured:
            if self.autodownload:
                try:
                    from huggingface_hub import snapshot_download

                    return (
                        Path(snapshot_download(configured)),
                        configured,
                    )
                except Exception as exc:  # noqa: BLE001 - surfaced as unsupported
                    self.progress(f"OPUS-MT {source} 下载失败：{exc}")
            return None, configured
        if not self.autodownload:
            return None, None
        model_id = OPUS_MT_MODEL_IDS.get(source)
        if not model_id:
            return None, None
        try:
            from huggingface_hub import snapshot_download

            return Path(snapshot_download(model_id)), model_id
        except Exception as exc:  # noqa: BLE001 - surfaced as unsupported
            self.progress(f"OPUS-MT {source} 下载失败：{exc}")
            return None, model_id

    def _load_pair(self, source: str) -> _LoadedTranslationPair | None:
        source = source.casefold().strip()
        if source in self.models:
            return self.models[source]
        if source in self.failed_sources:
            return None
        model_path, model_id = self._resolve_model_path(source)
        if model_path is None:
            self.failed_sources.add(source)
            return None
        try:
            import ctranslate2
            import sentencepiece as spm

            source_path, target_path = self._sentencepiece_paths(model_path)
            if source_path is None or target_path is None:
                raise RuntimeError(f"OPUS-MT {source} 缺少 SentencePiece 模型")
            source_sp = spm.SentencePieceProcessor(model_file=str(source_path))
            target_sp = spm.SentencePieceProcessor(model_file=str(target_path))
            translator = ctranslate2.Translator(
                str(model_path),
                device=self.device,
                compute_type="int8_float16" if self.device == "cuda" else "int8",
            )
            loaded = _LoadedTranslationPair(
                source_sp,
                target_sp,
                translator,
                model_id or str(model_path),
            )
            self.models[source] = loaded
            return loaded
        except Exception as exc:  # noqa: BLE001 - preserve original text
            self.progress(f"OPUS-MT {source} 不可用：{exc}")
            self.failed_sources.add(source)
            return None

    def _cache_get(self, source: str, text: str) -> TranslationResult | None:
        cached = getattr(self, "cache", {}).get((source, text))
        if cached is None:
            return None
        return TranslationResult(cached.text, cached.status, cached.model)

    def _cache_put(self, source: str, text: str, result: TranslationResult) -> None:
        if not hasattr(self, "cache"):
            return
        if len(self.cache) >= self.cache_limit:
            self.cache.pop(next(iter(self.cache)))
        self.cache[(source, text)] = result

    def _translate_loaded(
        self,
        loaded: _LoadedTranslationPair,
        texts: list[str],
    ) -> list[TranslationResult]:
        try:
            pieces = [loaded.source_sp.encode(text, out_type=str) for text in texts]
            generated = loaded.translator.translate_batch(
                pieces,
                beam_size=2,
                max_decoding_length=384,
                repetition_penalty=1.05,
            )
            results: list[TranslationResult] = []
            for text, hypothesis in zip(texts, generated):
                tokens = list(hypothesis.hypotheses[0])
                while tokens and tokens[0] in {"<pad>", "</s>"}:
                    tokens.pop(0)
                translated = simplify_chinese(loaded.target_sp.decode(tokens).strip())
                results.append(
                    TranslationResult(translated or text, "ready", loaded.model_id)
                )
            if len(results) != len(texts):
                raise RuntimeError("OPUS-MT 返回的结果数量与输入不一致")
            return results
        except Exception as exc:  # noqa: BLE001 - preserve original text
            self.progress(f"OPUS-MT 翻译失败：{exc}")
            return [TranslationResult(text, "failed", loaded.model_id) for text in texts]

    def translate_many(self, texts: list[str], source: str) -> list[TranslationResult]:
        source = (source or "").casefold().strip()
        normalized = [text.strip() for text in texts]
        if source not in SUPPORTED_LANGUAGE_CODES:
            return [TranslationResult(text, "unsupported") for text in normalized]
        results: list[TranslationResult | None] = [None] * len(normalized)
        pending_texts: list[str] = []
        pending_indices: list[int] = []
        for index, text in enumerate(normalized):
            if not text:
                results[index] = TranslationResult("", "ready")
            elif source == "zh":
                results[index] = TranslationResult(simplify_chinese(text), "not_needed")
            else:
                cached = self._cache_get(source, text)
                if cached is not None:
                    results[index] = cached
                else:
                    pending_texts.append(text)
                    pending_indices.append(index)
        if not pending_texts:
            return [item or TranslationResult("", "ready") for item in results]

        # Preserve the old unit-test seam and old integrations that construct
        # the object with ``sp`` and ``model`` directly.
        if not hasattr(self, "models") and hasattr(self, "model") and hasattr(self, "sp"):
            translated: list[TranslationResult] = []
            for text in pending_texts:
                pieces = self.sp.encode(text, out_type=str)
                generated = self.model.translate_batch(
                    [[NLLB_CODES.get(source, source), *pieces, "</s>"]],
                    target_prefix=[[NLLB_CODES["zh"]]],
                    beam_size=2,
                    max_decoding_length=384,
                    repetition_penalty=1.05,
                )[0]
                tokens = generated.hypotheses[0]
                if tokens and tokens[0] == NLLB_CODES["zh"]:
                    tokens = tokens[1:]
                translated.append(
                    TranslationResult(simplify_chinese(self.sp.decode(tokens).strip()), "ready")
                )
        else:
            loaded = self._load_pair(source)
            if loaded is None:
                translated = [TranslationResult(text, "unsupported") for text in pending_texts]
            else:
                translated = self._translate_loaded(loaded, pending_texts)
        for index, text, result in zip(pending_indices, pending_texts, translated):
            results[index] = result
            self._cache_put(source, text, result)
        return [item or TranslationResult("", "ready") for item in results]

    def translate_with_status(self, text: str, source: str) -> TranslationResult:
        return self.translate_many([text], source)[0]

    def translate(self, text: str, source: str) -> str:
        return self.translate_with_status(text, source).text


@dataclass(slots=True)
class PartialResult:
    revision: int
    start: float
    end: float
    text: str
    language: str | None
    confidence: float = 0.0
    model: str | None = None


@dataclass(slots=True)
class _DecodedSegment:
    start: float
    end: float
    text: str


class LiveModelRuntime:
    """Shared ASR/translation models; mutable speaker clusters stay per meeting."""

    def __init__(
        self,
        asr_model: str,
        translation_model: str,
        requested_device: str,
        refine_asr_model: str | None = None,
        refinement_enabled: bool = True,
        fallback_asr_model: str | None = None,
        *,
        translation_model_root: Path | None = None,
        translation_autodownload: bool = False,
        vad_model: str = "fsmn-vad",
        gpu_memory_budget_mb: int = 7_200,
    ) -> None:
        self.asr_model_name = asr_model
        self.fallback_asr_model_name = fallback_asr_model or "large-v3-turbo"
        self.refine_asr_model_name = refine_asr_model or "large-v3"
        self.refinement_enabled = refinement_enabled
        self.translation_model_name = translation_model
        self.translation_model_root = translation_model_root
        self.translation_autodownload = translation_autodownload
        self.vad_model_name = vad_model
        self.gpu_memory_budget_mb = max(1_024, int(gpu_memory_budget_mb))
        self.requested_device = requested_device
        self.device = "cpu"
        self.compute_type = "int8"
        self.asr = None  # Whisper fallback; retained for compatibility.
        self.fallback_asr = None
        self.fun_asr = None
        self.refine_asr = None
        self.translator: LiveChineseTranslator | None = None
        self.detector: MultilingualDetector | None = None
        self.speaker_encoder = None
        self.speakers: OnlineSpeakerClusterer | None = None
        self.vad = None
        self._whisper_model_class: Any = None
        self._refine_model_lock = threading.Lock()
        self._refine_device = "cpu"
        self.ready = False
        self.status = "等待加载"
        self.metrics: dict[str, Any] = {
            "funasr_calls": 0,
            "whisper_calls": 0,
            "fallback_calls": 0,
            "translation_calls": 0,
            "oom_count": 0,
            "gpu_memory_peak_mb": 0,
            "model_events": [],
        }

    def _model_event(self, event: str, **details: Any) -> None:
        events = self.metrics.setdefault("model_events", [])
        if isinstance(events, list):
            events.append({"event": event, **details})
            del events[:-50]

    def _update_gpu_memory_metrics(self) -> None:
        if self.device != "cuda":
            return
        try:
            import torch

            if not torch.cuda.is_available():
                return
            allocated_mb = int(torch.cuda.memory_allocated() / 1024**2)
            peak_mb = int(torch.cuda.max_memory_allocated() / 1024**2)
            self.metrics["gpu_memory_peak_mb"] = max(
                int(self.metrics.get("gpu_memory_peak_mb", 0)), peak_mb
            )
            self.metrics["gpu_memory_allocated_mb"] = allocated_mb
        except Exception:
            return

    @staticmethod
    def _is_funasr_name(name: str) -> bool:
        normalized = (name or "").casefold()
        return "fun-asr" in normalized or "funasr" in normalized

    @staticmethod
    def _normalize_funasr_name(name: str) -> str:
        if name.casefold() in {"funasr-nano", "fun-asr-nano"}:
            return "FunAudioLLM/Fun-ASR-Nano-2512"
        return name

    def load(self, progress: Callable[[str], None] = lambda _message: None) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
        from faster_whisper import WhisperModel

        self._whisper_model_class = WhisperModel
        self.device, self.compute_type = choose_device(self.requested_device)
        self.status = "正在加载语音识别模型"
        progress(self.status)

        if self._is_funasr_name(self.asr_model_name):
            try:
                from funasr import AutoModel

                funasr_kwargs = {
                    "model": self._normalize_funasr_name(self.asr_model_name),
                    "hub": "hf",
                    "trust_remote_code": True,
                    "device": f"{self.device}:0" if self.device == "cuda" else self.device,
                }
                try:
                    self.fun_asr = AutoModel(**funasr_kwargs)
                except TypeError:
                    funasr_kwargs.pop("hub", None)
                    funasr_kwargs.pop("trust_remote_code", None)
                    self.fun_asr = AutoModel(**funasr_kwargs)
                self._model_event("loaded", model=self.asr_model_name, role="primary")
            except ImportError as exc:
                progress(f"FunASR 未安装，使用 Whisper 回退：{exc}")
                self.fun_asr = None
            except Exception as exc:  # noqa: BLE001 - fallback keeps service usable
                progress(f"FunASR 加载失败，使用 Whisper 回退：{exc}")
                self.fun_asr = None
        else:
            self.asr = WhisperModel(
                self.asr_model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            self._model_event("loaded", model=self.asr_model_name, role="primary")

        whisper_name = self.fallback_asr_model_name
        if self.asr is None:
            self.status = "正在加载多语言回退模型"
            progress(self.status)
            self.asr = WhisperModel(
                whisper_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            self._model_event("loaded", model=whisper_name, role="fallback")
        self.fallback_asr = self.asr

        if not self.refinement_enabled or self.refine_asr_model_name == whisper_name:
            self.refine_asr = self.asr
            self._refine_device = self.device
        else:
            # large-v3 is loaded on the first stop/refinement job. This keeps
            # the realtime path responsive and makes the low-priority model
            # the first candidate for unloading when the GPU budget is tight.
            self.refine_asr = None
            self._refine_device = self.device
            self._model_event("deferred", model=self.refine_asr_model_name, role="refine")

        self.status = "正在加载多语言识别"
        progress(self.status)
        self.detector = MultilingualDetector()
        if self.vad_model_name.casefold() in {"", "energy", "rms"}:
            self.vad = None
            progress("使用能量 VAD（按配置启用）")
        else:
            try:
                from .audio import FsmnVAD

                self.vad = FsmnVAD(self.vad_model_name, self.device)
                self._model_event("loaded", model=self.vad_model_name, role="vad")
            except Exception as exc:  # noqa: BLE001 - energy fallback remains valid
                self.vad = None
                progress(f"FSMN-VAD 不可用，使用能量 VAD 回退：{exc}")
        translator_kwargs = {
            "model_root": self.translation_model_root,
            "autodownload": self.translation_autodownload,
        }
        try:
            self.translator = LiveChineseTranslator(
                self.translation_model_name, self.device, progress, **translator_kwargs
            )
        except TypeError:
            # Keep compatibility with small embedded test doubles and external
            # plugins implementing the original three-argument constructor.
            self.translator = LiveChineseTranslator(
                self.translation_model_name, self.device, progress
            )

        self.status = "正在加载说话人模型"
        progress(self.status)
        from resemblyzer import VoiceEncoder

        self.speaker_encoder = VoiceEncoder(device=self.device)
        self.speakers = OnlineSpeakerClusterer(self.device, encoder=self.speaker_encoder)
        self.status = "正在预热 GPU 推理"
        progress(self.status)
        self._warmup()
        self._update_gpu_memory_metrics()
        self.ready = True
        self.status = "模型已就绪"
        progress(self.status)

    def close(self) -> None:
        self._model_event("unloaded", role="all")
        self.speaker_encoder = None
        self.ready = False
        self.asr = None
        self.fallback_asr = None
        self.fun_asr = None
        self.refine_asr = None
        self.translator = None
        self.detector = None
        self.speakers = None
        self.vad = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        self.status = "模型已释放"

    def _ensure_refine_model(self) -> Any:
        if not self.refinement_enabled:
            return self.asr
        if self.refine_asr is not None:
            return self.refine_asr
        if self._whisper_model_class is None:
            raise RuntimeError("精修模型加载器尚未就绪")
        with self._refine_model_lock:
            if self.refine_asr is not None:
                return self.refine_asr
            self.status = "正在加载高精度语音识别模型"
            try:
                self.refine_asr = self._whisper_model_class(
                    self.refine_asr_model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
                self._refine_device = self.device
                self._model_event(
                    "loaded", model=self.refine_asr_model_name, role="refine"
                )
                self._update_gpu_memory_metrics()
                if (
                    self.device == "cuda"
                    and int(self.metrics.get("gpu_memory_allocated_mb", 0))
                    > self.gpu_memory_budget_mb
                ):
                    # Keep real-time ASR healthy. If the optional refine model
                    # would exceed the configured budget, release it and retry
                    # the low-priority job on CPU instead of risking an OOM.
                    self.refine_asr = None
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self._model_event(
                        "unloaded",
                        model=self.refine_asr_model_name,
                        role="refine",
                        reason="gpu_budget",
                    )
                    self.refine_asr = self._whisper_model_class(
                        self.refine_asr_model_name,
                        device="cpu",
                        compute_type="int8",
                    )
                    self._refine_device = "cpu"
                    self._model_event(
                        "loaded",
                        model=self.refine_asr_model_name,
                        role="refine",
                        device="cpu",
                    )
                return self.refine_asr
            except RuntimeError as exc:
                if "out of memory" in str(exc).casefold():
                    self.metrics["oom_count"] = int(self.metrics.get("oom_count", 0)) + 1
                    self._model_event(
                        "oom", model=self.refine_asr_model_name, role="refine"
                    )
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                self.refine_asr = None
                raise

    def _warmup(self) -> None:
        silent = np.zeros(SAMPLE_RATE, dtype=np.float32)
        if self.fun_asr is not None:
            try:
                self.fun_asr.generate(input=silent, language="中文", itn=True)
            except Exception:
                pass
        if self.asr is not None:
            segments, _ = self.asr.transcribe(
                silent,
                task="transcribe",
                language=None,
                beam_size=1,
                best_of=1,
                multilingual=True,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            list(segments)
        if self.refine_asr is not None and self.refine_asr is not self.asr:
            refined_segments, _ = self.refine_asr.transcribe(
                silent,
                task="transcribe",
                language=None,
                beam_size=1,
                best_of=1,
                multilingual=True,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            list(refined_segments)
        if self.translator is not None:
            self.translator.translate("Guten Morgen", "de")

    def new_speaker_clusterer(self) -> OnlineSpeakerClusterer:
        if self.speaker_encoder is None:
            return OnlineSpeakerClusterer(self.device)
        return OnlineSpeakerClusterer(self.device, encoder=self.speaker_encoder)

    def new_vad(self) -> Any:
        """Return the shared stateful VAD adapter, or ``None`` for fallback."""

        return self.vad

    @staticmethod
    def _float_audio(pcm: bytes) -> np.ndarray:
        return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    def _whisper_recognize(
        self,
        pcm: bytes,
        recent_text: str,
        hotwords: str | None,
        partial: bool,
        *,
        refined: bool,
        language_hint: str | None,
    ) -> tuple[list[_DecodedSegment], str | None, float, str]:
        model = self._ensure_refine_model() if refined else self.asr
        if not self.ready and model is None:
            raise RuntimeError("模型尚未就绪")
        if model is None:
            raise RuntimeError("Whisper 回退模型尚未就绪")
        context = recent_text.strip()[-128:]
        prompt_parts = [part.strip() for part in (hotwords or "", context) if part.strip()]
        prompt = " ".join(prompt_parts)[-512:]
        # Do not force the previous segment's language into Whisper. A short
        # partial can be misclassified, and forcing that guess makes later
        # Chinese audio decode as the wrong language. Whisper's own audio-level
        # detection is more reliable for each window; ``language_hint`` is
        # reserved for the text-level detector below.
        segments, info = model.transcribe(
            self._float_audio(pcm),
            task="transcribe",
            language=None,
            beam_size=1 if partial else 3,
            best_of=1 if partial else 3,
            multilingual=True,
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=not partial,
            initial_prompt=prompt or None,
            hotwords=hotwords,
        )
        whisper_language = getattr(info, "language", None)
        try:
            whisper_confidence = float(getattr(info, "language_probability", 0.0) or 0.0)
        except (TypeError, ValueError):
            whisper_confidence = 0.0
        decoded = [
            _DecodedSegment(float(item.start), float(item.end), item.text.strip())
            for item in list(segments)
            if getattr(item, "text", "").strip()
        ]
        self.metrics["whisper_calls"] = int(self.metrics["whisper_calls"]) + 1
        return decoded, whisper_language, whisper_confidence, "whisper"

    def _funasr_recognize(
        self,
        pcm: bytes,
        *,
        partial: bool,
        language_hint: str | None,
        hotwords: str | None,
    ) -> tuple[list[_DecodedSegment], str | None, float, str]:
        if self.fun_asr is None:
            return [], None, 0.0, "funasr"
        audio = self._float_audio(pcm)
        kwargs: dict[str, Any] = {
            "input": audio,
            "itn": True,
            "batch_size": 1,
        }
        if hotwords:
            kwargs["hotword"] = hotwords
        if not partial:
            kwargs["is_final"] = True
        try:
            result = self.fun_asr.generate(**kwargs)
        except TypeError:
            kwargs.pop("is_final", None)
            kwargs.pop("hotword", None)
            try:
                result = self.fun_asr.generate(**kwargs)
            except Exception:  # noqa: BLE001 - route an unsupported response to Whisper
                self.metrics["fallback_calls"] = int(self.metrics["fallback_calls"]) + 1
                return [], None, 0.0, "funasr"
        except Exception:  # noqa: BLE001 - route model/API failures to Whisper
            # Some Fun-ASR releases require an explicit Chinese language for
            # the Nano checkpoint. Retry once with that documented default
            # before routing the segment to Whisper.
            if "language" not in kwargs:
                kwargs["language"] = "中文"
                try:
                    result = self.fun_asr.generate(**kwargs)
                except Exception:  # noqa: BLE001
                    self.metrics["fallback_calls"] = int(self.metrics["fallback_calls"]) + 1
                    return [], None, 0.0, "funasr"
            else:
                self.metrics["fallback_calls"] = int(self.metrics["fallback_calls"]) + 1
                return [], None, 0.0, "funasr"
        payload = result[0] if isinstance(result, list) and result else result
        if not isinstance(payload, dict):
            return [], None, 0.0, "funasr"
        language = payload.get("language") or payload.get("lang")
        try:
            confidence = float(
                payload.get("confidence", payload.get("score", 0.0)) or 0.0
            )
        except (TypeError, ValueError):
            confidence = 0.0
        decoded: list[_DecodedSegment] = []
        sentence_info = payload.get("sentence_info")
        if isinstance(sentence_info, list):
            for item in sentence_info:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or item.get("sentence") or "").strip()
                if not text:
                    continue
                start = float(item.get("start", 0.0) or 0.0)
                end = float(item.get("end", start) or start)
                if start > 100 or end > 100:
                    start /= 1000.0
                    end /= 1000.0
                decoded.append(_DecodedSegment(start, end, text))
        if not decoded:
            text = str(payload.get("text") or payload.get("sentence") or "").strip()
            if text:
                decoded.append(_DecodedSegment(0.0, len(audio) / SAMPLE_RATE, text))
        self.metrics["funasr_calls"] = int(self.metrics["funasr_calls"]) + 1
        return decoded, str(language) if language else None, confidence, "funasr"

    def _recognize(
        self,
        pcm: bytes,
        recent_text: str,
        hotwords: str | None,
        partial: bool,
        *,
        refined: bool = False,
        language_hint: str | None = None,
    ) -> tuple[list[_DecodedSegment], str | None, float, str]:
        if self.fun_asr is not None and not refined:
            fun_segments, fun_language, fun_confidence, model_name = self._funasr_recognize(
                pcm,
                partial=partial,
                language_hint=language_hint,
                hotwords=hotwords,
            )
            text = " ".join(item.text for item in fun_segments).strip()
            if text:
                detected = self.detector.detect(
                    text,
                    whisper_language=fun_language,
                    whisper_confidence=fun_confidence,
                ) if self.detector else None
                detected_code = (
                    detected.code
                    if detected
                    else normalize_language_code(fun_language)
                )
                # Fun-ASR is the Chinese/English fast path. German goes to
                # Whisper, which is the stronger multilingual model for that
                # language and avoids returning a plausible-looking but wrong
                # Fun-ASR transcript.
                if detected_code in {"zh", "en"}:
                    return fun_segments, detected_code, max(fun_confidence, detected.confidence if detected else 0.0), model_name
                if detected_code == "de" or (language_hint or "").casefold() == "de":
                    self.metrics["fallback_calls"] = int(self.metrics["fallback_calls"]) + 1
                    return self._whisper_recognize(
                        pcm,
                        recent_text,
                        hotwords,
                        partial,
                        refined=False,
                        language_hint="de",
                    )
        return self._whisper_recognize(
            pcm,
            recent_text,
            hotwords,
            partial,
            refined=refined,
            language_hint=language_hint,
        )

    def transcribe_partial(
        self,
        event: SegmentEvent,
        recent_text: str = "",
        hotwords: str | None = None,
        language_hint: str | None = None,
    ) -> PartialResult | None:
        segments, whisper_language, whisper_confidence, model_name = self._recognize(
            event.pcm,
            recent_text,
            hotwords,
            partial=True,
            language_hint=language_hint,
        )
        raw_text = " ".join(item.text for item in segments if item.text).strip()
        if not raw_text:
            return None
        language = (
            self.detector.detect(
                raw_text,
                normalize_language_code(language_hint),
                whisper_language=whisper_language,
                whisper_confidence=whisper_confidence,
            ).code
            if self.detector
            else whisper_language
        )
        if language not in SUPPORTED_LANGUAGE_CODES:
            return None
        text = simplify_chinese(raw_text) if language == "zh" else raw_text
        return PartialResult(
            event.revision,
            event.start,
            event.end,
            text,
            language,
            whisper_confidence,
            model_name,
        )

    def transcribe_draft(
        self,
        event: SegmentEvent,
        recent_text: str = "",
        hotwords: str | None = None,
        language_hint: str | None = None,
    ) -> PartialResult | None:
        return self.transcribe_partial(
            event, recent_text, hotwords, language_hint=language_hint
        )

    def transcribe_final(
        self,
        event: SegmentEvent,
        *,
        next_id: int,
        previous_language: str | None,
        recent_text: str = "",
        hotwords: str | None = None,
        speaker_clusterer: OnlineSpeakerClusterer | None = None,
        refined: bool = True,
        language_hint: str | None = None,
    ) -> list[Utterance]:
        if self.detector is None:
            raise RuntimeError("语言检测器尚未就绪")
        clusterer = speaker_clusterer or self.speakers
        if clusterer is None:
            raise RuntimeError("说话人模型尚未就绪")
        raw_segments, whisper_language, whisper_confidence, _model_name = self._recognize(
            event.pcm,
            recent_text,
            hotwords,
            partial=False,
            refined=refined,
            language_hint=language_hint or previous_language,
        )
        if not raw_segments:
            return []
        content_start = min(float(raw.start) for raw in raw_segments)
        content_end = max(float(raw.end) for raw in raw_segments)
        speaker_id = clusterer.assign(event.pcm, max(0.0, content_end - content_start))
        utterances: list[Utterance] = []
        language = previous_language
        for raw in raw_segments:
            text = raw.text.strip()
            if not text:
                continue
            clauses = self.detector.split_clauses(text)
            weights = [max(1, len(clause)) for clause in clauses]
            total_weight = sum(weights)
            raw_start = event.start + float(raw.start)
            raw_end = min(event.end, event.start + float(raw.end))
            cursor = raw_start
            duration = max(0.01, raw_end - raw_start)
            for clause, weight in zip(clauses, weights):
                end = min(raw_end, cursor + duration * weight / total_weight)
                if not any(character.isalnum() for character in clause):
                    cursor = end
                    continue
                guess = self.detector.detect(
                    clause,
                    language,
                    whisper_language=whisper_language,
                    whisper_confidence=whisper_confidence,
                )
                if guess.code not in SUPPORTED_LANGUAGE_CODES:
                    cursor = end
                    continue
                source_text = simplify_chinese(clause) if guess.code == "zh" else clause
                utterances.append(
                    Utterance(
                        id=next_id + len(utterances),
                        start=round(cursor, 3),
                        end=round(end, 3),
                        speaker_id=speaker_id,
                        language=guess.code,
                        language_confidence=round(guess.confidence, 4),
                        text=source_text,
                        translation_zh="",
                        segment_revision=event.revision,
                        recognition_stage="refined" if refined else "fast",
                        translation_status="pending",
                        revision=2 if refined else 1,
                    )
                )
                language = guess.code
                cursor = end
        return utterances

    def translate_text(self, text: str, source_language: str) -> TranslationResult:
        if self.translator is None:
            return TranslationResult(text, "unsupported")
        self.metrics["translation_calls"] = int(self.metrics["translation_calls"]) + 1
        return self.translator.translate_with_status(text, source_language)

    def translate_text_batch(
        self, texts: list[str], source_language: str
    ) -> list[TranslationResult]:
        if self.translator is None:
            return [TranslationResult(text, "unsupported") for text in texts]
        self.metrics["translation_calls"] = int(self.metrics["translation_calls"]) + len(texts)
        return self.translator.translate_many(texts, source_language)


_NORMALIZE_RE = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)


def is_boundary_duplicate(previous: str, current: str) -> bool:
    left = _NORMALIZE_RE.sub("", previous).casefold()
    right = _NORMALIZE_RE.sub("", current).casefold()
    if not left or not right:
        return False
    if left == right:
        return True
    if min(len(left), len(right)) >= 8 and (left.endswith(right) or right.startswith(left)):
        return True
    if min(len(left), len(right)) >= 12 and SequenceMatcher(None, left, right).ratio() >= 0.88:
        return True
    if 6 <= len(right) <= 20 and len(left) >= len(right):
        suffix = left[-len(right):]
        return SequenceMatcher(None, suffix, right).ratio() >= 0.65
    return False

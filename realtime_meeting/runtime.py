from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import SAMPLE_RATE, SegmentEvent
from .language import TrilingualDetector
from .models import Utterance
from .speaker import OnlineSpeakerClusterer
from .text_normalize import simplify_chinese


# ISO-639-1 language codes returned by Whisper -> NLLB language tags.  This
# covers common meeting languages and prevents non-English text (for example
# Russian, Spanish or Portuguese) from being silently treated as English.
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
    # Additional Whisper languages (NLLB tags use a few non-obvious ISO
    # aliases, so keep the mapping explicit instead of guessing a suffix).
    "as": "asm_Beng", "az": "azj_Latn", "ba": "bak_Cyrl",
    "be": "bel_Cyrl", "bo": "bod_Tibt",
    "bs": "bos_Latn", "ca": "cat_Latn", "cy": "cym_Latn",
    "eu": "eus_Latn", "fo": "fao_Latn", "gl": "glg_Latn",
    "gu": "guj_Gujr", "ha": "hau_Latn",
    "ht": "hat_Latn", "hy": "hye_Armn", "is": "isl_Latn",
    "jw": "jav_Latn", "ka": "kat_Geor", "kk": "kaz_Cyrl",
    "km": "khm_Khmr", "kn": "kan_Knda",
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
    "yue": "yue_Hant",
}


def choose_device(requested: str) -> tuple[str, str]:
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
                raise RuntimeError("CTranslate2 未发现可用 CUDA，请检查 CUDA 12、cuBLAS 和 cuDNN 9")
    return "cpu", "int8"


class LiveChineseTranslator:
    def __init__(self, model_name: str, device: str, progress: Callable[[str], None]) -> None:
        import ctranslate2
        import sentencepiece as spm
        from huggingface_hub import snapshot_download

        progress("正在加载中文翻译模型（NLLB 1.3B）")
        model_path = Path(
            snapshot_download(
                model_name,
                allow_patterns=[
                    "model.bin",
                    "config.json",
                    "shared_vocabulary.json",
                    "shared_vocabulary.txt",
                    "sentencepiece.bpe.model",
                ],
            )
        )
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path / "sentencepiece.bpe.model"))
        self.model = ctranslate2.Translator(
            str(model_path),
            device=device,
            compute_type="int8_float16" if device == "cuda" else "int8",
        )

    def translate(self, text: str, source: str) -> str:
        text = text.strip()
        if not text:
            return ""
        if source == "zh":
            return simplify_chinese(text)
        source_code = NLLB_CODES.get(source.casefold().strip())
        if not source_code:
            # Keep an unsupported source visible instead of silently claiming
            # that it was English.  The original-language line remains useful
            # and the mapping can be extended without data loss.
            return text
        pieces = self.sp.encode(text, out_type=str)
        generated = self.model.translate_batch(
            [[source_code, *pieces, "</s>"]],
            target_prefix=[[NLLB_CODES["zh"]]],
            beam_size=2,
            max_decoding_length=384,
            repetition_penalty=1.05,
        )[0]
        tokens = generated.hypotheses[0]
        if tokens and tokens[0] == NLLB_CODES["zh"]:
            tokens = tokens[1:]
        return simplify_chinese(self.sp.decode(tokens).strip())


@dataclass(slots=True)
class PartialResult:
    revision: int
    start: float
    end: float
    text: str
    language: str | None


class LiveModelRuntime:
    """Persistent, single-worker model bundle for one local meeting."""

    def __init__(self, asr_model: str, translation_model: str, requested_device: str) -> None:
        self.asr_model_name = asr_model
        self.translation_model_name = translation_model
        self.requested_device = requested_device
        self.device = "cpu"
        self.compute_type = "int8"
        self.asr = None
        self.translator: LiveChineseTranslator | None = None
        self.detector: TrilingualDetector | None = None
        self.speakers: OnlineSpeakerClusterer | None = None
        self.ready = False
        self.status = "等待加载"

    def load(self, progress: Callable[[str], None] = lambda _message: None) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
        from faster_whisper import WhisperModel

        self.device, self.compute_type = choose_device(self.requested_device)
        self.status = "正在加载语音识别模型"
        progress(self.status)
        self.asr = WhisperModel(
            self.asr_model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        self.status = "正在加载语言识别"
        progress(self.status)
        self.detector = TrilingualDetector()
        self.translator = LiveChineseTranslator(
            self.translation_model_name, self.device, progress
        )
        self.status = "正在加载说话人模型"
        progress(self.status)
        self.speakers = OnlineSpeakerClusterer(self.device)
        self.status = "正在预热 GPU 推理"
        progress(self.status)
        self._warmup()
        self.ready = True
        self.status = "模型已就绪"
        progress(self.status)

    def _warmup(self) -> None:
        """Initialize CUDA kernels before the first participant starts speaking."""
        assert self.asr is not None and self.translator is not None
        silent = np.zeros(SAMPLE_RATE, dtype=np.float32)
        segments, _ = self.asr.transcribe(
            silent,
            beam_size=1,
            best_of=1,
            multilingual=True,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        list(segments)
        self.translator.translate("Guten Morgen", "de")

    @staticmethod
    def _float_audio(pcm: bytes) -> np.ndarray:
        return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    def _recognize(
        self, pcm: bytes, recent_text: str, hotwords: str | None, partial: bool
    ) -> tuple[list[object], str | None, float]:
        if not self.ready or self.asr is None:
            raise RuntimeError("模型尚未就绪")
        # A previous-language prompt can make Whisper translate the next speaker
        # into that language. Keep user-supplied terminology, but do not carry
        # transcript text across Chinese/English/German turns.
        prompt = hotwords or ""
        segments, info = self.asr.transcribe(
            self._float_audio(pcm),
            language=None,  # detect afresh for every stable audio segment
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
        return list(segments), whisper_language, whisper_confidence

    def transcribe_partial(
        self, event: SegmentEvent, recent_text: str = "", hotwords: str | None = None
    ) -> PartialResult | None:
        segments, whisper_language, whisper_confidence = self._recognize(
            event.pcm, recent_text, hotwords, partial=True
        )
        text = simplify_chinese(
            " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        )
        if not text:
            return None
        language = (
            self.detector.detect(
                text,
                whisper_language=whisper_language,
                whisper_confidence=whisper_confidence,
            ).code
            if self.detector
            else None
        )
        return PartialResult(event.revision, event.start, event.end, text, language)

    def transcribe_final(
        self,
        event: SegmentEvent,
        *,
        next_id: int,
        previous_language: str | None,
        recent_text: str = "",
        hotwords: str | None = None,
    ) -> list[Utterance]:
        assert self.detector is not None and self.translator is not None and self.speakers is not None
        raw_segments, whisper_language, whisper_confidence = self._recognize(
            event.pcm, recent_text, hotwords, partial=False
        )
        if not raw_segments:
            return []
        content_start = min(float(raw.start) for raw in raw_segments)
        content_end = max(float(raw.end) for raw in raw_segments)
        speaker_id = self.speakers.assign(event.pcm, max(0.0, content_end - content_start))
        utterances: list[Utterance] = []
        language = previous_language
        for raw in raw_segments:
            # Whisper may emit Traditional Chinese for a Chinese turn.  Do
            # this before clause splitting so the live UI, JSONL and exports
            # all use the same Simplified Chinese source text.
            text = simplify_chinese(raw.text.strip())
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
                translation_zh = self.translator.translate(clause, guess.code)
                utterances.append(
                    Utterance(
                        id=next_id + len(utterances),
                        start=round(cursor, 3),
                        end=round(end, 3),
                        speaker_id=speaker_id,
                        language=guess.code,
                        language_confidence=round(guess.confidence, 4),
                        text=clause,
                        translation_zh=translation_zh,
                    )
                )
                language = guess.code
                cursor = end
        return utterances


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
    # Forced cuts retain a short overlap. Compare the previous suffix against a
    # short next fragment to suppress near-duplicates such as "any rent" /
    # "penny rent" without hiding a full new sentence.
    if 6 <= len(right) <= 20 and len(left) >= len(right):
        suffix = left[-len(right) :]
        return SequenceMatcher(None, suffix, right).ratio() >= 0.65
    return False

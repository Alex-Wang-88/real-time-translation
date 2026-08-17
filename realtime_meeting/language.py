from __future__ import annotations

import re
from dataclasses import dataclass


SUPPORTED_LANGUAGE_CODES = frozenset({"zh", "en", "de"})
OFFICIAL_SPEECH_VARIANTS = (
    "anhui",
    "dongbei",
    "fujian",
    "gansu",
    "guizhou",
    "hebei",
    "henan",
    "hubei",
    "hunan",
    "jiangxi",
    "ningxia",
    "shandong",
    "shaanxi",
    "shanxi",
    "sichuan",
    "tianjin",
    "yunnan",
    "zhejiang",
    "cantonese_hong_kong",
    "cantonese_guangdong",
    "wu",
    "minnan",
)
CANONICAL_SPEECH_VARIANTS = ("mandarin", *OFFICIAL_SPEECH_VARIANTS, "cantonese_unknown")
SUPPORTED_SPEECH_VARIANTS = frozenset(CANONICAL_SPEECH_VARIANTS)

VARIANT_LABELS = {
    "mandarin": "普通话",
    "anhui": "安徽方言",
    "dongbei": "东北方言",
    "fujian": "福建方言",
    "gansu": "甘肃方言",
    "guizhou": "贵州方言",
    "hebei": "河北方言",
    "henan": "河南方言",
    "hubei": "湖北方言",
    "hunan": "湖南方言",
    "jiangxi": "江西方言",
    "ningxia": "宁夏方言",
    "shandong": "山东方言",
    "shaanxi": "陕西方言",
    "shanxi": "山西方言",
    "sichuan": "四川方言",
    "tianjin": "天津方言",
    "yunnan": "云南方言",
    "zhejiang": "浙江方言",
    "cantonese_hong_kong": "粤语（香港口音）",
    "cantonese_guangdong": "粤语（广东口音）",
    "cantonese_unknown": "粤语（口音未确认）",
    "wu": "吴语",
    "minnan": "闽南语",
}

_CJK = re.compile(r"[\u3400-\u9fff]")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


@dataclass(frozen=True, slots=True)
class LanguageGuess:
    """A product-level interpretation of one Qwen language result.

    Qwen does not expose a calibrated language probability in every wrapper,
    so ``confidence`` is an internal quality/stability score rather than a
    claim about model probability.  ``stable`` is set by the session after the
    two-observation confirmation rule.
    """

    code: str
    confidence: float
    speech_variant: str | None = None
    raw_qwen_label: str = ""
    stable: bool = False

    @property
    def language(self) -> str:
        return self.code


def contains_cjk(text: str | None) -> bool:
    return bool(_CJK.search(str(text or "")))


def contains_latin(text: str | None) -> bool:
    return bool(_LATIN_WORD.search(str(text or "")))


def is_mixed_source_text(text: str | None) -> bool:
    """Return whether a paragraph contains both Chinese and foreign words."""

    value = str(text or "")
    return contains_cjk(value) and contains_latin(value)


def normalize_language_code(value: object) -> str | None:
    raw = str(value or "").strip().casefold().replace("_", "-")
    if not raw:
        return None
    aliases = {
        "cmn": "zh",
        "mandarin": "zh",
        "putonghua": "zh",
        "chinese": "zh",
        "中文": "zh",
        "普通话": "zh",
        "english": "en",
        "英语": "en",
        "german": "de",
        "deutsch": "de",
        "德语": "de",
    }
    normalized = aliases.get(raw, raw.split("-", 1)[0])
    return normalized if normalized in SUPPORTED_LANGUAGE_CODES else None


def normalize_speech_variant(value: object) -> str | None:
    raw = str(value or "").strip().casefold().replace("_", "-")
    if not raw:
        return None
    aliases = {
        "mandarin": "mandarin",
        "standard mandarin": "mandarin",
        "standard chinese": "mandarin",
        "putonghua": "mandarin",
        "普通话": "mandarin",
        "国语": "mandarin",
        "cantonese (hong kong accent)": "cantonese_hong_kong",
        "cantonese (guangdong accent)": "cantonese_guangdong",
        "cantonese-hong-kong": "cantonese_hong_kong",
        "cantonese-guangdong": "cantonese_guangdong",
        "hong kong cantonese": "cantonese_hong_kong",
        "香港粤语": "cantonese_hong_kong",
        "guangdong cantonese": "cantonese_guangdong",
        "广东话": "cantonese_guangdong",
        "廣東話": "cantonese_guangdong",
        "粤语": "cantonese_unknown",
        "cantonese": "cantonese_unknown",
        "canton": "cantonese_unknown",
        "yue": "cantonese_unknown",
        "anhui": "anhui",
        "anhui dialect": "anhui",
        "安徽": "anhui",
        "安徽话": "anhui",
        "安徽方言": "anhui",
        "dongbei": "dongbei",
        "dongbei dialect": "dongbei",
        "northeastern chinese": "dongbei",
        "东北": "dongbei",
        "东北话": "dongbei",
        "东北方言": "dongbei",
        "fujian": "fujian",
        "fujian dialect": "fujian",
        "福建": "fujian",
        "福建话": "fujian",
        "福建方言": "fujian",
        "gansu": "gansu",
        "gansu dialect": "gansu",
        "甘肃": "gansu",
        "甘肃话": "gansu",
        "甘肃方言": "gansu",
        "guizhou": "guizhou",
        "guizhou dialect": "guizhou",
        "贵州": "guizhou",
        "贵州话": "guizhou",
        "贵州方言": "guizhou",
        "hebei": "hebei",
        "hebei dialect": "hebei",
        "河北": "hebei",
        "河北话": "hebei",
        "河北方言": "hebei",
        "henan": "henan",
        "henan dialect": "henan",
        "河南": "henan",
        "河南话": "henan",
        "河南方言": "henan",
        "hubei": "hubei",
        "hubei dialect": "hubei",
        "湖北": "hubei",
        "湖北话": "hubei",
        "湖北方言": "hubei",
        "hunan": "hunan",
        "hunan dialect": "hunan",
        "湖南": "hunan",
        "湖南话": "hunan",
        "湖南方言": "hunan",
        "jiangxi": "jiangxi",
        "jiangxi dialect": "jiangxi",
        "江西": "jiangxi",
        "江西话": "jiangxi",
        "江西方言": "jiangxi",
        "ningxia": "ningxia",
        "ningxia dialect": "ningxia",
        "宁夏": "ningxia",
        "宁夏话": "ningxia",
        "宁夏方言": "ningxia",
        "shandong": "shandong",
        "shandong dialect": "shandong",
        "山东": "shandong",
        "山东话": "shandong",
        "山东方言": "shandong",
        "shaanxi": "shaanxi",
        "shaanxi dialect": "shaanxi",
        "陕西": "shaanxi",
        "陕西话": "shaanxi",
        "陕西方言": "shaanxi",
        "shanxi": "shanxi",
        "shanxi dialect": "shanxi",
        "山西": "shanxi",
        "山西话": "shanxi",
        "山西方言": "shanxi",
        "sichuan": "sichuan",
        "sichuanese": "sichuan",
        "sichuan dialect": "sichuan",
        "四川话": "sichuan",
        "四川話": "sichuan",
        "四川方言": "sichuan",
        "tianjin": "tianjin",
        "tianjin dialect": "tianjin",
        "天津": "tianjin",
        "天津话": "tianjin",
        "天津方言": "tianjin",
        "yunnan": "yunnan",
        "yunnan dialect": "yunnan",
        "云南": "yunnan",
        "云南话": "yunnan",
        "云南方言": "yunnan",
        "zhejiang": "zhejiang",
        "zhejiang dialect": "zhejiang",
        "浙江": "zhejiang",
        "浙江话": "zhejiang",
        "浙江方言": "zhejiang",
        "hangzhou": "zhejiang",
        "hangzhouese": "zhejiang",
        "hangzhou dialect": "zhejiang",
        "杭州话": "zhejiang",
        "杭州方言": "zhejiang",
        "wu": "wu",
        "wuu": "wu",
        "wu language": "wu",
        "wu dialect": "wu",
        "shanghainese": "wu",
        "吴语": "wu",
        "吳語": "wu",
        "minnan": "minnan",
        "minnan language": "minnan",
        "minnan dialect": "minnan",
        "hokkien": "minnan",
        "闽南语": "minnan",
        "閩南語": "minnan",
    }
    if raw in aliases:
        return aliases[raw]
    if "hong kong" in raw and "cantonese" in raw:
        return "cantonese_hong_kong"
    if ("guangdong" in raw or "广东" in raw or "廣東" in raw) and "cantonese" in raw:
        return "cantonese_guangdong"
    for token, variant in (
        ("cantonese", "cantonese_unknown"),
        ("anhui", "anhui"),
        ("dongbei", "dongbei"),
        ("northeastern chinese", "dongbei"),
        ("fujian", "fujian"),
        ("gansu", "gansu"),
        ("guizhou", "guizhou"),
        ("hebei", "hebei"),
        ("henan", "henan"),
        ("hubei", "hubei"),
        ("hunan", "hunan"),
        ("jiangxi", "jiangxi"),
        ("ningxia", "ningxia"),
        ("shandong", "shandong"),
        ("shaanxi", "shaanxi"),
        ("shanxi", "shanxi"),
        ("sichuan", "sichuan"),
        ("tianjin", "tianjin"),
        ("yunnan", "yunnan"),
        ("zhejiang", "zhejiang"),
        ("hangzhou", "zhejiang"),
        ("minnan", "minnan"),
        ("hokkien", "minnan"),
        ("wuu", "wu"),
        ("wu language", "wu"),
        ("shanghai", "wu"),
        ("mandarin", "mandarin"),
    ):
        if token in raw:
            return variant
    if re.search(r"\bwu\b", raw) or "吴语" in raw or "吳語" in raw:
        return "wu"
    if "普通话" in raw:
        return "mandarin"
    return None


def normalize_qwen_label(value: object, text: str | None = None) -> LanguageGuess:
    """Normalize a raw Qwen label while keeping explicit model labels intact."""

    raw = str(value or "").strip()
    normalized = re.sub(r"\s+", " ", raw.casefold().replace("_", "-")).strip()
    if not normalized:
        return LanguageGuess("unknown", 0.0, raw_qwen_label=raw)

    variant = normalize_speech_variant(normalized)
    if variant:
        return LanguageGuess("zh", 0.95, variant, raw_qwen_label=raw)

    if normalized in {"english", "en", "英语", "英文"} or normalized.startswith("english"):
        return LanguageGuess("en", 0.92, raw_qwen_label=raw)
    if normalized in {"german", "de", "deutsch", "德语", "德文"} or normalized.startswith("german"):
        return LanguageGuess("de", 0.92, raw_qwen_label=raw)
    if normalized in {"chinese", "zh", "cmn", "中文", "汉语", "漢語"}:
        return LanguageGuess("zh", 0.78, None, raw_qwen_label=raw)

    code = normalize_language_code(normalized)
    if code:
        if code == "zh":
            return LanguageGuess("zh", 0.72, None, raw_qwen_label=raw)
        return LanguageGuess(code, 0.65, raw_qwen_label=raw)
    return LanguageGuess("unknown", 0.0, raw_qwen_label=raw)


def normalize_detected_language(value: object, text: str | None = None) -> tuple[str | None, str | None]:
    """Compatibility helper for callers that need the old tuple shape."""

    guess = normalize_qwen_label(value, text)
    return (guess.code if guess.code != "unknown" else None), guess.speech_variant


def language_key(guess: LanguageGuess | None) -> tuple[str, str]:
    if guess is None:
        return "unknown", "unknown"
    return guess.code or "unknown", guess.speech_variant or "unknown"

from typing import Dict, Optional

# Canonical ISO 639-1 to standard English name (for display/logging)
ISO_TO_NAME: Dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "tr": "Turkish",
    "ru": "Russian",
    "nl": "Dutch",
    "cs": "Czech",
    "ar": "Arabic",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    # Add more as needed
}

# NLLB requires 3-letter codes with script (e.g. eng_Latn)
# This is a simplified mapping for common languages.
# NLLB-200 supports 200 languages, we map the most common ones.
ISO_TO_NLLB: Dict[str, str] = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "pl": "pol_Latn",
    "tr": "tur_Latn",
    "ru": "rus_Cyrl",
    "nl": "nld_Latn",
    "cs": "ces_Latn",
    "ar": "arb_Arab",
    "zh": "zho_Hans", # Simplified Chinese by default
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "hi": "hin_Deva",
}

# XTTS v2 supports specific language codes (usually 2 letter)
# Supported languages: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, ko, hu, hi
ISO_TO_XTTS: Dict[str, str] = {
    "en": "en",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "pl": "pl",
    "tr": "tr",
    "ru": "ru",
    "nl": "nl",
    "cs": "cs",
    "ar": "ar",
    "zh": "zh-cn",
    "ja": "ja",
    "ko": "ko",
    "hi": "hi",
}

# Faster-Whisper uses standard codes, generally ISO 639-1 (2-letter)
# So we can pass the CLI code directly usually, but let's be explicit if needed.

def get_nllb_code(iso_code: str) -> str:
    """Returns the NLLB language code for a given ISO 639-1 code."""
    return ISO_TO_NLLB.get(iso_code, f"{iso_code}_Latn") # Fallback guess

def get_xtts_code(iso_code: str) -> str:
    """Returns the XTTS language code for a given ISO 639-1 code."""
    # XTTS is picky, if not in list it might fail.
    if iso_code not in ISO_TO_XTTS:
        raise ValueError(f"Language {iso_code} not supported by XTTS v2.")
    return ISO_TO_XTTS[iso_code]

def normalize_lang_code(code: str) -> str:
    """
    Normalizes input code to ISO 639-1 (lower case).
    """
    return code.lower().strip()

import logging
from typing import Optional
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, NllbTokenizer, M2M100Tokenizer

from dubber.utils.device import get_device_strategy
from dubber.utils.languages import get_nllb_code

logger = logging.getLogger(__name__)

class Translator:
    def __init__(self, model_name: str = "nllb-200-distilled-600M", device: str = "cpu", low_mem: bool = False):
        self.device = device # "mps" or "cpu"
        self.model_name = model_name
        self.low_mem = low_mem
        self._model = None
        self._tokenizer = None

        # Decide full model ID
        if "nllb" in model_name.lower():
            self.hf_model_id = f"facebook/{model_name}"
        elif "m2m100" in model_name.lower():
            # e.g., facebook/m2m100_418M
            self.hf_model_id = f"facebook/{model_name}"
        else:
            # Fallback or direct ID
            self.hf_model_id = model_name

    def load_model(self):
        if self._model is not None:
            return

        logger.info(f"Loading translation model '{self.hf_model_id}' on {self.device}...")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.hf_model_id)

            if self.device == "mps":
                self._model.to(torch.device("mps"))
            else:
                self._model.to(torch.device("cpu"))

            if self.low_mem:
                # We could quantize or just keep it as is.
                # Transformers has quantization support but requires bitsandbytes often (which has issues on Mac).
                # NLLB-distilled-600M is already small (~1.2GB or less).
                pass

        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            raise

    def translate_text(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translates text from source language to target language.
        src_lang/tgt_lang should be ISO codes (e.g. "en", "es").
        """
        if not text or not text.strip():
            return ""

        self.load_model()

        # Prepare languages
        # NLLB requires full codes (eng_Latn, etc.)
        # M2M100 requires simple codes usually? No, M2M also has specific codes but simpler.
        # We will assume NLLB logic primarily as it's default.

        is_nllb = "nllb" in self.model_name.lower()

        if is_nllb:
            src_code = get_nllb_code(src_lang)
            tgt_code = get_nllb_code(tgt_lang)
        else:
            # M2M100
            src_code = src_lang
            tgt_code = tgt_lang

        # Tokenize
        if is_nllb:
            inputs = self._tokenizer(text, return_tensors="pt", padding=True)
        else:
            # M2M100 needs forced_bos_token_id usually
            self._tokenizer.src_lang = src_code
            inputs = self._tokenizer(text, return_tensors="pt")

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            forced_bos_token_id = self._tokenizer.lang_code_to_id[tgt_code]
            generated_tokens = self._model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512
            )

        result = self._tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return result

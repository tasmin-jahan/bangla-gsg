import os
import json
from transformers import PreTrainedTokenizerFast
try:
    from bnunicodenormalizer import Normalizer
    _HAS_BNORM = True
except ImportError:
    _HAS_BNORM = False

class BanglaGSGTokenizer(PreTrainedTokenizerFast):
    """
    Custom Tokenizer for BanglaGSG.
    
    Automatically applies the `bnunicodenormalizer` to all inputs before tokenization.
    This guarantees that raw inference text matches the normalized pretraining corpus,
    preventing severe hallucinations.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _HAS_BNORM:
            self.bnorm = Normalizer()
        else:
            import warnings
            warnings.warn(
                "bnunicodenormalizer is not installed. Normalization will be skipped, "
                "which may cause the model to output gibberish. "
                "Please run `pip install bnunicodenormalizer`."
            )
            self.bnorm = None

    def _normalize_text(self, text: str) -> str:
        """Apply bnunicodenormalizer word-by-word."""
        if not self.bnorm or not isinstance(text, str):
            return text
            
        words = text.split()
        normalized_words = []
        for word in words:
            # bnorm returns a dict like {'normalized': '...'}
            res = self.bnorm(word)
            if res and res.get('normalized'):
                normalized_words.append(res['normalized'])
            else:
                normalized_words.append(word)
                
        # Rejoin with standard spaces
        return " ".join(normalized_words)

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        """Intercept raw string before base tokenizer handles it."""
        if not is_split_into_words:
            text = self._normalize_text(text)
        return super().prepare_for_tokenization(text, is_split_into_words=is_split_into_words, **kwargs)

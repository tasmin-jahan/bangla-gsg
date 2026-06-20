"""
BanglaGSG Tokenizer Module
==============================

Exports:
  - special_tokens:    All 146 special token definitions
  - train_tokenizer:   SentencePiece Unigram trainer
  - wrapper:           HuggingFace PreTrainedTokenizerFast integration
  - load_tokenizer:    Quick-load convenience function
"""

from src.tokenizer.special_tokens import (
    ALL_SPECIAL_TOKENS,
    SPECIAL_TOKEN_TO_ID,
    USER_DEFINED_SYMBOLS,
    VOCAB_SIZE,
    NUM_SPECIAL_TOKENS,
    get_lang_token,
    get_sentiment_tokens,
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
)
from src.tokenizer.wrapper import load_tokenizer

__all__ = [
    "ALL_SPECIAL_TOKENS",
    "SPECIAL_TOKEN_TO_ID",
    "USER_DEFINED_SYMBOLS",
    "VOCAB_SIZE",
    "NUM_SPECIAL_TOKENS",
    "get_lang_token",
    "get_sentiment_tokens",
    "load_tokenizer",
    "PAD_TOKEN",
    "UNK_TOKEN",
    "BOS_TOKEN",
    "EOS_TOKEN",
]

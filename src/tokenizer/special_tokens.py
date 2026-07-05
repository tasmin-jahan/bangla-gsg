"""
BanglaGSG Special Token Registry
===================================

Central, single-source-of-truth definition of all special tokens used
by the BanglaGSG tokenizer and model.  Every other module (tokenizer
trainer, HF wrapper, model, data pipeline) imports from here.

Token budget:  146 special  +  47,854 learned subwords  =  48,000 total vocab
Token IDs:     0 – 145  are reserved for special tokens
               146 – 47,999  are learned by SentencePiece
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ── Standard control tokens (IDs 0–3) ───────────────────────────────────────

PAD_TOKEN = "<pad>"          # ID 0  — padding, attention mask = 0
UNK_TOKEN = "<unk>"          # ID 1  — unknown (rare with byte_fallback)
BOS_TOKEN = "<s>"            # ID 2  — beginning of sequence / document
EOS_TOKEN = "</s>"           # ID 3  — end of sequence / document separator


# ── ChatML / instruction format (IDs 4–10) ──────────────────────────────────

CHAT_TOKENS = [
    "<|im_start|>",          # 4   start of a chat turn
    "<|im_end|>",            # 5   end of a chat turn
    "<|system|>",            # 6   system role marker
    "<|user|>",              # 7   user role marker
    "<|assistant|>",         # 8   assistant role marker
    "<|tool|>",              # 9   tool call role
    "<|tool_result|>",       # 10  tool call result
]


# ── Task control tokens (IDs 11–22) ─────────────────────────────────────────

TASK_TOKENS = [
    "<|task_sentiment|>",         # 11
    "<|task_ner|>",               # 12
    "<|task_qa|>",                # 13
    "<|task_summarize|>",         # 14
    "<|task_translate_bn_en|>",   # 15
    "<|task_translate_en_bn|>",   # 16
    "<|task_classify|>",          # 17
    "<|task_generate|>",          # 18
    "<|task_toxicity|>",          # 19
    "<|task_nli|>",               # 20
    "<|task_pos|>",               # 21
    "<|task_paraphrase|>",        # 22
]


# ── Language control tokens (IDs 23–28) ─────────────────────────────────────

LANG_TOKENS = [
    "<|lang_bn|>",           # 23  standard Bangla (BD majority)
    "<|lang_en|>",           # 24  English
    "<|lang_wbn|>",          # 25  West Bengali variant
    "<|lang_bnls|>",         # 26  Banglish (romanized Bangla)
    "<|lang_mix|>",          # 27  code-mixed Bangla+English
    "<|lang_code|>",         # 28  programming code
]


# ── Sentiment label tokens (IDs 29–33) ──────────────────────────────────────

SENTIMENT_TOKENS = [
    "<|positive|>",          # 29
    "<|negative|>",          # 30
    "<|neutral|>",           # 31
    "<|mixed|>",             # 32
    "<|offensive|>",         # 33
]


# ── Reasoning / chain-of-thought tokens (IDs 34–37) ────────────────────────

REASONING_TOKENS = [
    "<|think|>",             # 34
    "<|/think|>",            # 35
    "<|answer|>",            # 36
    "<|sep|>",               # 37
]


# ── Document structure tokens (IDs 38–45) ───────────────────────────────────

STRUCTURE_TOKENS = [
    "<|context|>",           # 38
    "<|/context|>",          # 39
    "<|question|>",          # 40
    "<|passage|>",           # 41
    "<|headline|>",          # 42
    "<|title|>",             # 43
    "<|speaker_a|>",         # 44
    "<|speaker_b|>",         # 45
]


# ── Reserved tokens (IDs 46–145) ────────────────────────────────────────────

NUM_RESERVED = 100
RESERVED_TOKENS = [f"<|reserved_{i}|>" for i in range(NUM_RESERVED)]


# ═════════════════════════════════════════════════════════════════════════════
#  Derived collections
# ═════════════════════════════════════════════════════════════════════════════

# Ordered list of ALL user-defined special tokens (excluding pad/unk/bos/eos
# which are handled by SentencePiece's --pad_id/--unk_id/--bos_id/--eos_id).
USER_DEFINED_SYMBOLS: List[str] = (
    CHAT_TOKENS
    + TASK_TOKENS
    + LANG_TOKENS
    + SENTIMENT_TOKENS
    + REASONING_TOKENS
    + STRUCTURE_TOKENS
    + RESERVED_TOKENS
)

# Full ordered list including standard control tokens (for HF wrapper).
ALL_SPECIAL_TOKENS: List[str] = [
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN,
] + USER_DEFINED_SYMBOLS

# Token → expected ID mapping (for validation).
SPECIAL_TOKEN_TO_ID: Dict[str, int] = {
    tok: idx for idx, tok in enumerate(ALL_SPECIAL_TOKENS)
}

# Total counts.
NUM_SPECIAL_TOKENS = len(ALL_SPECIAL_TOKENS)   # 146
NUM_STANDARD = 4   # pad, unk, bos, eos

assert NUM_SPECIAL_TOKENS == 146, (
    f"Expected 146 special tokens, got {NUM_SPECIAL_TOKENS}"
)


# ═════════════════════════════════════════════════════════════════════════════
#  Tokenizer configuration constants
# ═════════════════════════════════════════════════════════════════════════════

VOCAB_SIZE = 48_000                         # total vocabulary
LEARNED_SUBWORDS = VOCAB_SIZE - NUM_SPECIAL_TOKENS   # 47,854
CHARACTER_COVERAGE = 1.0
MODEL_TYPE = "unigram"                      # SentencePiece model type
BYTE_FALLBACK = True

# SentencePiece standard token IDs
SP_PAD_ID = 0
SP_UNK_ID = 1
SP_BOS_ID = 2
SP_EOS_ID = 3


# ═════════════════════════════════════════════════════════════════════════════
#  ChatML template (Jinja2 format for HuggingFace tokenizer)
# ═════════════════════════════════════════════════════════════════════════════

CHAT_TEMPLATE = """\
{% for message in messages %}\
<|im_start|><|{{ message['role'] }}|>
{{ message['content'] }}<|im_end|>
{% endfor %}\
{% if add_generation_prompt %}<|im_start|><|assistant|>
{% endif %}"""


# ═════════════════════════════════════════════════════════════════════════════
#  Convenience helpers
# ═════════════════════════════════════════════════════════════════════════════

def get_lang_token(language_region: str) -> str:
    """Map a language_region metadata tag to the appropriate language token."""
    lr = language_region.lower()
    if "code" in lr or lr == "python":
        return "<|lang_code|>"
    if "banglish" in lr or "bnls" in lr:
        return "<|lang_bnls|>"
    if "mix" in lr:
        return "<|lang_mix|>"
    if lr in ("en", "english"):
        return "<|lang_en|>"
    if "wb" in lr or "west" in lr:
        return "<|lang_wbn|>"
    # Default: standard Bangla (Bangladeshi)
    return "<|lang_bn|>"


def get_sentiment_tokens() -> Dict[str, str]:
    """Return a mapping of sentiment label names to token strings."""
    return {
        "positive":  "<|positive|>",
        "negative":  "<|negative|>",
        "neutral":   "<|neutral|>",
        "mixed":     "<|mixed|>",
        "offensive": "<|offensive|>",
    }


if __name__ == "__main__":
    print(f"Total special tokens: {NUM_SPECIAL_TOKENS}")
    print(f"Vocab size:           {VOCAB_SIZE}")
    print(f"Learned subwords:     {LEARNED_SUBWORDS}")
    print()
    print("Token ID assignments:")
    for tok, tid in SPECIAL_TOKEN_TO_ID.items():
        if tid < 46 or tid >= 140:  # show non-reserved for brevity
            print(f"  {tid:>3d}  {tok}")
        elif tid == 46:
            print(f"  {tid:>3d}  {tok}  ... (100 reserved tokens)")

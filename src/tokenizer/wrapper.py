"""
Wraps a trained SentencePiece model into a HuggingFace PreTrainedTokenizerFast
without requiring protobuf or LlamaTokenizer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ponytail: pick one import path; callers should always use the package path
from src.tokenizer.special_tokens import (
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN,
    USER_DEFINED_SYMBOLS, CHAT_TEMPLATE,
    SP_PAD_ID, SP_UNK_ID, SP_BOS_ID, SP_EOS_ID,
)


def create_hf_tokenizer(spm_model_path: str, output_dir: str | None = None):
    """Build a PreTrainedTokenizerFast from a SentencePiece .model file."""
    try:
        import sentencepiece as spm
        from tokenizers import Tokenizer, AddedToken
        from tokenizers.models import Unigram
        from tokenizers.pre_tokenizers import Metaspace
        from tokenizers.decoders import Metaspace as MetaspaceDecoder, ByteFallback, Sequence
        from transformers import PreTrainedTokenizerFast
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\nRun: pip install sentencepiece tokenizers transformers")

    spm_path = Path(spm_model_path)
    if not spm_path.exists():
        sys.exit(f"Model not found: {spm_model_path}")

    print(f"Loading SentencePiece model: {spm_path}")

    sp = spm.SentencePieceProcessor()
    sp.Load(str(spm_path))
    vocab = [(sp.IdToPiece(i), sp.GetScore(i)) for i in range(sp.GetPieceSize())]
    print(f"  Extracted {len(vocab):,} pieces")

    backend = Tokenizer(Unigram(vocab, unk_id=SP_UNK_ID, byte_fallback=True))
    backend.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="first", split=False)
    # ByteFallback() reassembles <0xXX> tokens back into raw bytes/UTF-8
    # characters BEFORE Metaspace handles the ▁ word-boundary marker.
    # Order matters: byte reassembly must happen first.
    backend.decoder = Sequence([
        ByteFallback(),
        MetaspaceDecoder(replacement="▁", prepend_scheme="first"),
    ])
    backend.add_special_tokens([
        AddedToken(tok, special=True, normalized=False)
        for tok in [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN] + USER_DEFINED_SYMBOLS
    ])

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        additional_special_tokens=USER_DEFINED_SYMBOLS,
    )
    tokenizer.chat_template = CHAT_TEMPLATE

    print(f"\n  Vocab size:     {tokenizer.vocab_size}")
    print(f"  BOS token:      {tokenizer.bos_token!r} (ID {tokenizer.bos_token_id})")
    print(f"  EOS token:      {tokenizer.eos_token!r} (ID {tokenizer.eos_token_id})")
    print(f"  PAD token:      {tokenizer.pad_token!r} (ID {tokenizer.pad_token_id})")
    print(f"  UNK token:      {tokenizer.unk_token!r} (ID {tokenizer.unk_token_id})")
    print(f"  Special tokens: {len(tokenizer.all_special_tokens)}")

    mismatches = [
        f"{name} expected {exp}, got {got}"
        for name, exp, got in [
            ("PAD", SP_PAD_ID, tokenizer.pad_token_id),
            ("UNK", SP_UNK_ID, tokenizer.unk_token_id),
            ("BOS", SP_BOS_ID, tokenizer.bos_token_id),
            ("EOS", SP_EOS_ID, tokenizer.eos_token_id),
        ]
        if exp != got
    ]
    if mismatches:
        print("\n  [WARN] Token ID mismatches:\n" + "\n".join(f"    - {m}" for m in mismatches))
    else:
        print("\n  [OK] All standard token IDs match.")

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    status = "[OK]" if im_start_id == 4 else "[WARN] expected 4, got"
    print(f"  {status} <|im_start|> -> ID {im_start_id}")

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(out_path))
        print(f"  [OK] HF tokenizer saved to: {out_path}")

    return tokenizer


def load_tokenizer(tokenizer_dir: str = "saved/tokenizer"):
    """Load a previously saved HuggingFace tokenizer."""
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)


def main():
    parser = argparse.ArgumentParser(description="Wrap a SentencePiece model as a HuggingFace tokenizer.")
    parser.add_argument("--spm-model", "-m", required=True)
    parser.add_argument("--output-dir", "-o", default="saved/tokenizer")
    parser.add_argument("--test", action="store_true",
                        help="Run sanity tests after saving (delegates to evaluate_tokenizer --sanity).")
    args = parser.parse_args()

    create_hf_tokenizer(args.spm_model, args.output_dir)

    if args.test:
        import subprocess
        subprocess.run([
            sys.executable, "-m", "scripts.util.evaluate_tokenizer",
            "--sanity", "--skip-references",
        ], check=True)


if __name__ == "__main__":
    main()
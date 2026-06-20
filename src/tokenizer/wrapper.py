"""
BanglaGSG HuggingFace Tokenizer Wrapper
==========================================

Wraps the trained SentencePiece model into a HuggingFace
PreTrainedTokenizerFast for seamless integration with the
transformers ecosystem.

Usage:
  python -m src.tokenizer.wrapper \
      --spm-model saved/tokenizer/banglagsg_tokenizer.model \
      --output-dir saved/tokenizer/hf

Reference: BanglaFM_Complete_Guide.md §2.7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.tokenizer.special_tokens import (
        PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN,
        USER_DEFINED_SYMBOLS, CHAT_TEMPLATE,
    )
except ImportError:
    from special_tokens import (
        PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN,
        USER_DEFINED_SYMBOLS, CHAT_TEMPLATE,
    )


def create_hf_tokenizer(
    spm_model_path: str,
    output_dir: str | None = None,
):
    """
    Load a SentencePiece .model and wrap it as a HuggingFace tokenizer.

    Parameters
    ----------
    spm_model_path : str
        Path to the trained .model file.
    output_dir : str | None
        If provided, save the HF tokenizer to this directory.

    Returns
    -------
    PreTrainedTokenizerFast
        The wrapped tokenizer ready for use with transformers.
    """
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        sys.exit(1)

    spm_path = Path(spm_model_path)
    if not spm_path.exists():
        print(f"ERROR: SentencePiece model not found: {spm_model_path}")
        sys.exit(1)

    print(f"Loading SentencePiece model: {spm_path}")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(spm_path),
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        additional_special_tokens=USER_DEFINED_SYMBOLS,
    )

    # Register the ChatML template
    tokenizer.chat_template = CHAT_TEMPLATE

    print(f"  Vocab size:     {tokenizer.vocab_size}")
    print(f"  BOS token:      {tokenizer.bos_token} (ID {tokenizer.bos_token_id})")
    print(f"  EOS token:      {tokenizer.eos_token} (ID {tokenizer.eos_token_id})")
    print(f"  PAD token:      {tokenizer.pad_token} (ID {tokenizer.pad_token_id})")
    print(f"  UNK token:      {tokenizer.unk_token} (ID {tokenizer.unk_token_id})")
    print(f"  Special tokens: {len(tokenizer.all_special_tokens)}")

    # Save if output dir specified
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(out_path))
        print(f"\n✅ HF tokenizer saved to: {out_path}")

    return tokenizer


def load_tokenizer(tokenizer_dir: str = "saved/tokenizer/hf"):
    """
    Load a previously saved HuggingFace tokenizer.

    Parameters
    ----------
    tokenizer_dir : str
        Directory containing the saved HF tokenizer files.

    Returns
    -------
    PreTrainedTokenizerFast
        The loaded tokenizer.
    """
    from transformers import PreTrainedTokenizerFast

    tok_path = Path(tokenizer_dir)
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer directory not found: {tokenizer_dir}. "
            f"Train a tokenizer first with: python -m src.tokenizer.train_tokenizer"
        )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_path))
    print(f"[Tokenizer] Loaded from {tok_path} (vocab={tokenizer.vocab_size})")
    return tokenizer


def _test_tokenizer(tokenizer) -> None:
    """Run basic encode/decode and chat template tests."""
    print(f"\n{'=' * 60}")
    print(f"  HuggingFace Tokenizer Tests")
    print(f"{'=' * 60}\n")

    # Encode/decode roundtrip
    tests = [
        "আমি বাংলাদেশের মানুষ।",
        "Hello, how are you?",
        "ami tomake bhalobashi",
    ]
    for text in tests:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"  Input:   {text}")
        print(f"  IDs:     {ids[:10]}...")
        print(f"  Decoded: {decoded}")
        print()

    # Chat template test
    messages = [
        {"role": "system", "content": "তুমি একটি সহায়ক বাংলা ভাষার মডেল।"},
        {"role": "user", "content": "আমাকে সাহায্য করো।"},
    ]
    try:
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(f"  Chat template output:")
        for line in chat_text.split("\n"):
            print(f"    {line}")
        print(f"\n  ✅ Chat template works correctly.")
    except Exception as e:
        print(f"  ⚠️  Chat template error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Wrap a SentencePiece model as a HuggingFace tokenizer.",
    )
    parser.add_argument("--spm-model", "-m", required=True,
                        help="Path to trained .model file.")
    parser.add_argument("--output-dir", "-o", default="saved/tokenizer/hf",
                        help="Directory to save HF tokenizer (default: saved/tokenizer/hf).")
    parser.add_argument("--test", action="store_true",
                        help="Run validation tests after wrapping.")

    args = parser.parse_args()

    tokenizer = create_hf_tokenizer(
        spm_model_path=args.spm_model,
        output_dir=args.output_dir,
    )

    if args.test:
        _test_tokenizer(tokenizer)


if __name__ == "__main__":
    main()

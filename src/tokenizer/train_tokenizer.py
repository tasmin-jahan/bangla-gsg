"""
BanglaGSG SentencePiece Tokenizer Trainer
============================================

Trains a SentencePiece Unigram tokenizer (48K vocab, byte fallback).

Usage:
  python -m src.tokenizer.train_tokenizer \
      --input saved/data/tokenizer/tokenizer_training_corpus.txt

Reference: BanglaFM_Complete_Guide.md §2.3–2.4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.tokenizer.special_tokens import (
        USER_DEFINED_SYMBOLS, VOCAB_SIZE, CHARACTER_COVERAGE,
        MODEL_TYPE, BYTE_FALLBACK, SP_PAD_ID, SP_UNK_ID, SP_BOS_ID, SP_EOS_ID,
    )
except ImportError:
    from special_tokens import (
        USER_DEFINED_SYMBOLS, VOCAB_SIZE, CHARACTER_COVERAGE,
        MODEL_TYPE, BYTE_FALLBACK, SP_PAD_ID, SP_UNK_ID, SP_BOS_ID, SP_EOS_ID,
    )


def train_tokenizer(
    input_file: str,
    output_dir: str,
    model_prefix: str = "banglagsg_tokenizer",
    vocab_size: int = VOCAB_SIZE,
    num_threads: int = 8,
    input_sentence_size: int = 50_000_000,
) -> Path:
    """Train a SentencePiece Unigram tokenizer."""
    try:
        import sentencepiece as spm
    except ImportError:
        print("ERROR: sentencepiece not installed. Run: pip install sentencepiece")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / model_prefix

    user_symbols = ",".join(USER_DEFINED_SYMBOLS)

    print(f"{'=' * 60}")
    print(f"  BanglaGSG SentencePiece Tokenizer Training")
    print(f"{'=' * 60}")
    print(f"  Input:      {input_file}")
    print(f"  Output:     {output_dir}/{model_prefix}")
    print(f"  Vocab:      {vocab_size:,}  |  Type: {MODEL_TYPE}")
    print(f"  Coverage:   {CHARACTER_COVERAGE}  |  Byte fallback: {BYTE_FALLBACK}")
    print(f"  Specials:   {len(USER_DEFINED_SYMBOLS)} user-defined symbols")
    print(f"{'=' * 60}\n")

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)

    file_size_gb = input_path.stat().st_size / (1024 ** 3)
    print(f"Input file size: {file_size_gb:.2f} GB")
    if file_size_gb < 0.001:
        print("WARNING: Input file is very small. Tokenizer quality may suffer.")

    print("\nTraining (may take 2–6 hours on a large corpus)...")

    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix=str(model_path),
        vocab_size=vocab_size,
        character_coverage=CHARACTER_COVERAGE,
        model_type=MODEL_TYPE,
        byte_fallback=BYTE_FALLBACK,
        shuffle_input_sentence=True,
        input_sentence_size=input_sentence_size,
        pad_id=SP_PAD_ID,
        unk_id=SP_UNK_ID,
        bos_id=SP_BOS_ID,
        eos_id=SP_EOS_ID,
        user_defined_symbols=user_symbols,
        num_threads=num_threads,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        train_extremely_large_corpus=file_size_gb > 5.0,
    )

    model_file = Path(f"{model_path}.model")
    vocab_file = Path(f"{model_path}.vocab")
    print(f"\n✅ Training complete!")
    print(f"   Model: {model_file}")
    print(f"   Vocab: {vocab_file}")

    _validate_tokenizer(str(model_file))
    return model_file


def _validate_tokenizer(model_path: str) -> None:
    """Validate with fertility checks and special token verification."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)

    print(f"\n{'=' * 60}")
    print(f"  Tokenizer Validation (vocab={sp.GetPieceSize()})")
    print(f"{'=' * 60}\n")

    test_cases = [
        ("Bangla formal",   "আমি বাংলাদেশের মানুষ। আমি বাংলায় কথা বলি।"),
        ("Bangla news",     "প্রধানমন্ত্রী আজ জাতীয় সংসদে ভাষণ দিয়েছেন।"),
        ("English",         "The quick brown fox jumps over the lazy dog."),
        ("Banglish",        "ami tomake bhalobashi, tumi kemon acho?"),
        ("Code-mixed",      "এই product টা really ভালো, must buy করো।"),
        ("Python",          "def hello(): print('Hello, World!')"),
    ]

    for label, text in test_cases:
        tokens = sp.EncodeAsPieces(text)
        words = len(text.split())
        fert = len(tokens) / max(words, 1)
        print(f"  [{label}]  {len(tokens)} tok / {words} words = {fert:.2f} fertility")
        print(f"    → {tokens[:12]}{'...' if len(tokens) > 12 else ''}\n")

    # Verify special tokens
    print("  Special token check:")
    for tok in ["<pad>", "<unk>", "<bos>", "<eos>", "<|im_start|>",
                "<|lang_bn|>", "<|positive|>", "<|reserved_0|>", "<|reserved_99|>"]:
        tid = sp.PieceToId(tok)
        ok = tid != sp.unk_id() or tok == "<unk>"
        print(f"    {'✅' if ok else '❌'} {tok:30s} → ID {tid}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece Unigram tokenizer for BanglaGSG.",
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to training corpus (one doc per line).")
    parser.add_argument("--output-dir", "-o", default="saved/tokenizer",
                        help="Output directory (default: saved/tokenizer).")
    parser.add_argument("--model-prefix", default="banglagsg_tokenizer",
                        help="Model file prefix (default: banglagsg_tokenizer).")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE,
                        help=f"Vocabulary size (default: {VOCAB_SIZE:,}).")
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--input-sentence-size", type=int, default=50_000_000)

    args = parser.parse_args()
    train_tokenizer(
        input_file=args.input,
        output_dir=args.output_dir,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        num_threads=args.num_threads,
        input_sentence_size=args.input_sentence_size,
    )


if __name__ == "__main__":
    main()

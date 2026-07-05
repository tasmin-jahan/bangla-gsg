"""
BanglaGSG SentencePiece Tokenizer Trainer
============================================

Trains a SentencePiece Unigram tokenizer (48K vocab, byte fallback).

Usage:
  # From a plain-text file (one doc per line):
  python -m src.tokenizer.train_tokenizer \
      --input saved/data/tokenizer/tokenizer_training_corpus.txt

  # Directly from a JSONL corpus (extracts 'text' field automatically):
  python -m src.tokenizer.train_tokenizer \
      --input saved/data/tokenizer_corpus/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
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


def _doc_to_text(text: str) -> str:
    """Return the document text as a single training entry.

    Joins paragraphs with spaces so the entire doc is ONE line in the
    temp file.  SentencePiece reads by newlines, so internal \n would
    inflate 2.83M docs into 32M+ entries and cause OOM.
    """
    lines = []
    for paragraph in text.splitlines():
        paragraph = " ".join(paragraph.split())  # normalise whitespace only
        if paragraph:
            lines.append(paragraph)
    return " ".join(lines)


def _jsonl_to_txt(jsonl_path: Path, txt_path: Path) -> int:
    """Stream JSONL, extract 'text' field, write one doc per line.

    Each document is written as a single entry (paragraphs joined by
    newlines).  This keeps entry count at ~2.83M instead of 82M,
    avoiding SentencePiece OOM (SP builds an internal lattice per entry).

    Returns the number of docs written.
    """
    from tqdm import tqdm

    print(f"\nExtracting text from JSONL -> {txt_path}")
    print("  (temp file -- will be deleted after training)")
    print("  (doc-level: one entry per document, no sentence splitting)")

    docs = 0
    with open(jsonl_path, "r") as fin, open(txt_path, "w") as fout:
        for line in tqdm(fin, desc="Extracting", unit="docs", unit_scale=True):
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = doc.get("text", "")
            if not text:
                continue
            doc_text = _doc_to_text(text)
            if doc_text:
                fout.write(doc_text + "\n")
                docs += 1

    size_gb = txt_path.stat().st_size / (1024 ** 3)
    print(f"  Extracted {docs:,} docs  ({size_gb:.2f} GB)")
    return docs


def train_tokenizer(
    input_file: str,
    output_dir: str,
    model_prefix: str = "banglagsg_tokenizer",
    vocab_size: int = VOCAB_SIZE,
    num_threads: int = 8,
    input_sentence_size: int = 3_000_000,
    max_sentence_length: int = 65536,
    jsonl: bool = False,
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

    # If JSONL, extract text to a temp .txt in the same directory
    tmp_txt_path = None
    if jsonl:
        tmp_txt_path = Path(input_path.parent / "_tokenizer_tmp.txt")
        _jsonl_to_txt(input_path, tmp_txt_path)
        actual_input = str(tmp_txt_path)
    else:
        actual_input = str(input_file)

    actual_path = Path(actual_input)
    training_size_gb = actual_path.stat().st_size / (1024 ** 3)
    print(f"Training input size: {training_size_gb:.2f} GB")
    if training_size_gb < 0.001:
        print("WARNING: Input file is very small. Tokenizer quality may suffer.")

    print("\nTraining (may take 2–6 hours on a large corpus)...")

    spm.SentencePieceTrainer.train(
        input=actual_input,
        model_prefix=str(model_path),
        vocab_size=vocab_size,
        character_coverage=CHARACTER_COVERAGE,
        model_type=MODEL_TYPE,
        byte_fallback=BYTE_FALLBACK,
        shuffle_input_sentence=True,
        input_sentence_size=input_sentence_size,
        max_sentence_length=max_sentence_length,
        pad_id=SP_PAD_ID,
        unk_id=SP_UNK_ID,
        bos_id=SP_BOS_ID,
        eos_id=SP_EOS_ID,
        user_defined_symbols=user_symbols,
        num_threads=num_threads,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        train_extremely_large_corpus=training_size_gb > 5.0,
    )

    # Clean up temp file
    if tmp_txt_path is not None and tmp_txt_path.exists():
        freed = tmp_txt_path.stat().st_size / (1024 ** 3)
        tmp_txt_path.unlink()
        print(f"\n[deleted] Temp file freed {freed:.2f} GB")

    model_file = Path(f"{model_path}.model")
    vocab_file = Path(f"{model_path}.vocab")
    print(f"\n[OK] Training complete!")
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
    for tok in ["<pad>", "<unk>", "<s>", "</s>", "<|im_start|>",
                "<|lang_bn|>", "<|positive|>", "<|reserved_0|>", "<|reserved_99|>"]:
        tid = sp.PieceToId(tok)
        ok = tid != sp.unk_id() or tok == "<unk>"
        print(f"    {'[OK]' if ok else '[FAIL]'} {tok:30s} -> ID {tid}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece Unigram tokenizer for BanglaGSG.",
    )
    parser.add_argument("--input", "-i", default="saved/data/tokenizer_corpus/corpus.jsonl",
                        help="Path to training corpus (.txt or .jsonl).")
    parser.add_argument("--jsonl", action="store_true",
                        help="Input is JSONL — extract 'text' field automatically.")
    parser.add_argument("--output-dir", "-o", default="saved/tokenizer/model",
                        help="Output directory (default: saved/tokenizer/model).")
    parser.add_argument("--model-prefix", default="banglagsg_tokenizer",
                        help="Model file prefix (default: banglagsg_tokenizer).")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE,
                        help=f"Vocabulary size (default: {VOCAB_SIZE:,}).")
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--input-sentence-size", type=int, default=3_000_000,
                        help="Max sentences loaded into RAM (default: 3M, tuned for 32GB).")
    parser.add_argument("--max-sentence-length", type=int, default=65536,
                        help="Max chars per sentence; longer lines skipped (default: 65536).")

    args = parser.parse_args()
    train_tokenizer(
        input_file=args.input,
        output_dir=args.output_dir,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        num_threads=args.num_threads,
        input_sentence_size=args.input_sentence_size,
        max_sentence_length=args.max_sentence_length,
        jsonl=(args.jsonl or str(args.input).endswith(".jsonl")),
    )


if __name__ == "__main__":
    main()

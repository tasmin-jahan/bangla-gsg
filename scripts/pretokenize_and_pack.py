#!/usr/bin/env python3
"""
BanglaGSG — Pretokenization & Sequence Packing
==================================================

Converts the full corpus to pretokenized uint16 binary .npy shards.
Sequence-packs documents into 2048-token sequences with <eos> separators.
Prepends language control tokens per document.

Input:  saved/data/cleaned/corpus_decontaminated.jsonl (+ banglish)
Output: saved/pretokenized/*.npy  (consumed by src/data/dataset.py)

Usage:
  python scripts/pretokenize_and_pack.py
  python scripts/pretokenize_and_pack.py --tokenizer-dir saved/tokenizer/hf

Reference: BanglaFM_Q1_Data_Plan.md Part 5
"""

from __future__ import annotations

import argparse
import json
import sys
import numpy as np
from pathlib import Path

DEFAULT_CORPUS = "saved/data/cleaned/corpus_decontaminated.jsonl"
DEFAULT_BANGLISH = "saved/data/banglish/synthetic_banglish.jsonl"
DEFAULT_TOKENIZER_DIR = "saved/tokenizer/hf"
DEFAULT_OUTPUT_DIR = "saved/pretokenized"
DEFAULT_SEQ_LEN = 2048
DEFAULT_CHUNK_SIZE = 100_000  # sequences per shard

# ── Language token mapping ───────────────────────────────────────────────────

LANG_TOKEN_MAP = {
    "EN":                    "<|lang_en|>",
    "ENGLISH":               "<|lang_en|>",
    "BD":                    "<|lang_bn|>",
    "BD_WB_MIX":             "<|lang_bn|>",
    "BD_BANGLISH":           "<|lang_bnls|>",
    "BD_BANGLISH_SYNTHETIC": "<|lang_bnls|>",
    "CODE":                  "<|lang_code|>",
    "PYTHON":                "<|lang_code|>",
}


def get_lang_token(language_region: str) -> str:
    """Map language_region metadata to a language control token."""
    lr = language_region.upper().strip()
    if lr in LANG_TOKEN_MAP:
        return LANG_TOKEN_MAP[lr]
    if "BANGLISH" in lr or "BNLS" in lr:
        return "<|lang_bnls|>"
    if "MIX" in lr:
        return "<|lang_mix|>"
    if "CODE" in lr:
        return "<|lang_code|>"
    if "EN" in lr:
        return "<|lang_en|>"
    return "<|lang_bn|>"


def pack_sequences(token_stream: list[int], seq_len: int) -> np.ndarray:
    """Pack a flat token stream into (N, seq_len) uint16 array."""
    n = (len(token_stream) // seq_len) * seq_len
    if n == 0:
        return np.array([], dtype=np.uint16).reshape(0, seq_len)
    return np.array(token_stream[:n], dtype=np.uint16).reshape(-1, seq_len)


def run_pretokenization(
    corpus_files: list[str],
    tokenizer_dir: str,
    output_dir: str,
    seq_len: int,
    chunk_size: int,
) -> None:
    """Pretokenize corpus and write packed .npy shards."""
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        sys.exit(1)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tok_path = Path(tokenizer_dir)
    if not tok_path.exists():
        print(f"ERROR: Tokenizer not found at {tokenizer_dir}")
        print("  Train one first: python -m src.tokenizer.train_tokenizer --input ...")
        sys.exit(1)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_path))
    eos_id = tokenizer.eos_token_id
    assert eos_id is not None, "Tokenizer must have an EOS token"

    print(f"{'=' * 60}")
    print(f"  Pretokenization & Sequence Packing")
    print(f"{'=' * 60}")
    print(f"  Tokenizer:  {tokenizer_dir} (vocab={tokenizer.vocab_size})")
    print(f"  Seq len:    {seq_len}")
    print(f"  Chunk size: {chunk_size:,} sequences/shard")
    print(f"  EOS ID:     {eos_id}")
    print(f"  Output:     {output_dir}")
    print()

    shard_idx = 0
    token_buffer: list[int] = []
    total_tokens = 0
    total_docs = 0

    for corpus_file in corpus_files:
        fpath = Path(corpus_file)
        if not fpath.exists():
            print(f"  WARNING: Skipping missing file: {corpus_file}")
            continue

        print(f"  Processing: {corpus_file}")

        with open(fpath, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    text = record.get("text", "")
                    if not text:
                        continue

                    # Prepend language token
                    lang_region = record.get("language_region", "")
                    lang_token = get_lang_token(lang_region)
                    full_text = f"{lang_token} {text}"

                    # Tokenize
                    tokens = tokenizer.encode(full_text, add_special_tokens=False)
                    tokens.append(eos_id)  # document separator

                    token_buffer.extend(tokens)
                    total_tokens += len(tokens)
                    total_docs += 1

                    # Write shard when buffer fills
                    if len(token_buffer) >= chunk_size * seq_len:
                        packed = pack_sequences(token_buffer, seq_len)
                        shard_path = out_path / f"shard_{shard_idx:05d}.npy"
                        np.save(shard_path, packed)
                        print(f"    Shard {shard_idx}: {packed.shape[0]:,} sequences → {shard_path.name}")
                        token_buffer = token_buffer[chunk_size * seq_len :]
                        shard_idx += 1

                    if total_docs % 100_000 == 0:
                        print(f"    Docs: {total_docs:,} | Tokens: {total_tokens:,}")

                except Exception:
                    continue

    # Write final partial shard
    if token_buffer:
        packed = pack_sequences(token_buffer, seq_len)
        if packed.shape[0] > 0:
            shard_path = out_path / f"shard_{shard_idx:05d}.npy"
            np.save(shard_path, packed)
            print(f"    Final shard {shard_idx}: {packed.shape[0]:,} sequences")
            shard_idx += 1

    total_seqs = sum(
        np.load(str(p), mmap_mode="r").shape[0]
        for p in sorted(out_path.glob("shard_*.npy"))
    )

    print(f"\n{'=' * 60}")
    print(f"  Pretokenization Complete")
    print(f"{'=' * 60}")
    print(f"  Documents:  {total_docs:,}")
    print(f"  Tokens:     {total_tokens:,}")
    print(f"  Sequences:  {total_seqs:,}  (× {seq_len} = {total_seqs * seq_len:,} tokens)")
    print(f"  Shards:     {shard_idx}")
    print(f"  Disk size:  ~{total_tokens * 2 / 1024**3:.1f} GB")
    print(f"  Output:     {out_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Pretokenize and sequence-pack corpus into .npy shards.",
    )
    parser.add_argument("--corpus", default=DEFAULT_CORPUS,
                        help=f"Main corpus JSONL (default: {DEFAULT_CORPUS}).")
    parser.add_argument("--banglish", default=DEFAULT_BANGLISH,
                        help=f"Banglish corpus JSONL (default: {DEFAULT_BANGLISH}).")
    parser.add_argument("--extra-corpus", nargs="*", default=[],
                        help="Additional corpus JSONL files.")
    parser.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER_DIR,
                        help=f"HF tokenizer directory (default: {DEFAULT_TOKENIZER_DIR}).")
    parser.add_argument("--output-dir", "-o", default=DEFAULT_OUTPUT_DIR,
                        help=f"Output shard directory (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                        help=f"Sequence length (default: {DEFAULT_SEQ_LEN}).")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Sequences per shard (default: {DEFAULT_CHUNK_SIZE:,}).")

    args = parser.parse_args()

    corpus_files = [args.corpus, args.banglish] + args.extra_corpus

    run_pretokenization(
        corpus_files=corpus_files,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()

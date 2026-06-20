#!/usr/bin/env python3
"""
BanglaGSG — Tokenizer Training Corpus Preparation
=====================================================

Samples from the full corpus proportionally by source type to create
a ~10 GB training corpus for SentencePiece tokenizer training.

Input:  saved/data/cleaned/corpus_decontaminated.jsonl  (+ banglish)
Output: saved/data/tokenizer/tokenizer_training_corpus.txt

Usage:
  python scripts/prepare_tokenizer_corpus.py
  python scripts/prepare_tokenizer_corpus.py --target-gb 5.0

Reference: BanglaFM_Q1_Data_Plan.md Part 4
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_CORPUS = "saved/data/cleaned/corpus_decontaminated.jsonl"
DEFAULT_BANGLISH = "saved/data/banglish/synthetic_banglish.jsonl"
DEFAULT_OUTPUT = "saved/data/tokenizer/tokenizer_training_corpus.txt"
DEFAULT_TARGET_GB = 10.0

# Target proportions per source type (in relative MB)
# These mirror the training data mix from the plan.
SOURCE_TARGETS_MB = {
    "formal_news":              2000,
    "web_mixed":                2000,
    "web_informal":             1000,
    "web_noisy":                500,
    "informal_blog":            500,
    "informal_commerce_review": 300,
    "formal_official":          200,
    "encyclopedic":             400,
    "formal_education":         2500,
    "code_mixed_commerce":      300,
    "synthetic_banglish":       500,
    "banglish_parallel":        200,
    "code_python":              200,
}


def run_preparation(
    corpus_files: list[str],
    output_file: str,
    target_gb: float,
    seed: int = 42,
) -> None:
    """Sample corpus proportionally and write tokenizer training file."""
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Scale targets to actual target size
    total_target_mb = sum(SOURCE_TARGETS_MB.values())
    actual_target_mb = target_gb * 1024
    scale = actual_target_mb / total_target_mb

    target_bytes = {
        k: int(v * scale * 1024 * 1024)
        for k, v in SOURCE_TARGETS_MB.items()
    }

    print(f"{'=' * 60}")
    print(f"  Tokenizer Corpus Preparation")
    print(f"{'=' * 60}")
    print(f"  Target size: {target_gb:.1f} GB")
    print(f"  Seed:        {seed}")
    print(f"  Inputs:      {len(corpus_files)} files")
    print()

    # Read and bucket by source type
    buckets: dict[str, list[str]] = {k: [] for k in SOURCE_TARGETS_MB}
    bucket_sizes: dict[str, int] = {k: 0 for k in SOURCE_TARGETS_MB}
    unknown_types: set[str] = set()

    for corpus_file in corpus_files:
        fpath = Path(corpus_file)
        if not fpath.exists():
            print(f"  WARNING: File not found, skipping: {corpus_file}")
            continue

        print(f"  Reading: {corpus_file}")
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    st = record.get("source_type", "unknown")
                    text = record.get("text", "")
                    if not text:
                        continue

                    if st not in buckets:
                        unknown_types.add(st)
                        continue

                    if bucket_sizes[st] >= target_bytes[st]:
                        continue

                    buckets[st].append(text)
                    bucket_sizes[st] += len(text.encode("utf-8"))
                except (json.JSONDecodeError, ValueError):
                    continue

    if unknown_types:
        print(f"\n  Unknown source types (ignored): {unknown_types}")

    # Report bucket fill
    print(f"\n  {'Source Type':35s} {'Collected':>10s} {'Target':>10s} {'Fill':>6s}")
    print(f"  {'-' * 65}")
    total_collected = 0
    for st in sorted(SOURCE_TARGETS_MB.keys()):
        collected_mb = bucket_sizes[st] / (1024 * 1024)
        target_mb = target_bytes[st] / (1024 * 1024)
        fill_pct = 100 * bucket_sizes[st] / max(target_bytes[st], 1)
        total_collected += bucket_sizes[st]
        print(f"  {st:35s} {collected_mb:>8.1f} MB {target_mb:>8.1f} MB {fill_pct:>5.1f}%")

    total_gb = total_collected / (1024 ** 3)
    print(f"\n  Total collected: {total_gb:.2f} GB")

    # Shuffle and write
    random.seed(seed)
    all_texts = []
    for texts in buckets.values():
        all_texts.extend(texts)
    random.shuffle(all_texts)

    print(f"\n  Writing {len(all_texts):,} documents to {output_file}...")

    with open(out_path, "w", encoding="utf-8") as f:
        for text in all_texts:
            # One document per line (SentencePiece expects this format)
            f.write(text.replace("\n", " ") + "\n")

    final_size_gb = out_path.stat().st_size / (1024 ** 3)
    print(f"  ✅ Written: {final_size_gb:.2f} GB, {len(all_texts):,} documents")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare proportionally sampled corpus for tokenizer training.",
    )
    parser.add_argument("--corpus", default=DEFAULT_CORPUS,
                        help=f"Main cleaned corpus (default: {DEFAULT_CORPUS}).")
    parser.add_argument("--banglish", default=DEFAULT_BANGLISH,
                        help=f"Banglish corpus (default: {DEFAULT_BANGLISH}).")
    parser.add_argument("--extra-corpus", nargs="*", default=[],
                        help="Additional corpus JSONL files to include.")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT,
                        help=f"Output text file (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--target-gb", type=float, default=DEFAULT_TARGET_GB,
                        help=f"Target corpus size in GB (default: {DEFAULT_TARGET_GB}).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42).")

    args = parser.parse_args()

    corpus_files = [args.corpus, args.banglish] + args.extra_corpus

    run_preparation(
        corpus_files=corpus_files,
        output_file=args.output,
        target_gb=args.target_gb,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

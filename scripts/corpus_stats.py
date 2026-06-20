#!/usr/bin/env python3
"""
BanglaGSG — Corpus Statistics
================================

Computes and prints corpus statistics for the paper's data section.
Outputs a formatted table of per-source document counts, word counts,
estimated tokens, and domain coverage.

Input:  saved/data/cleaned/corpus_decontaminated.jsonl (+ banglish)
Output: printed to stdout (redirect to file for paper use)

Usage:
  python scripts/corpus_stats.py
  python scripts/corpus_stats.py > saved/corpus_statistics.txt

Reference: BanglaFM_Q1_Data_Plan.md Part 6
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_CORPUS = "saved/data/cleaned/corpus_decontaminated.jsonl"
DEFAULT_BANGLISH = "saved/data/banglish/synthetic_banglish.jsonl"

# Approximate tokens-per-word ratio for different languages
TOKENS_PER_WORD = {
    "bn": 1.3,   # Bangla: more subword splitting
    "en": 1.2,   # English: typical BPE ratio
    "code": 1.5, # Code: lots of punctuation/operators
}


def compute_stats(corpus_files: list[str]) -> None:
    """Compute and print corpus statistics."""
    stats: dict[str, dict] = defaultdict(
        lambda: {"docs": 0, "words": 0, "chars": 0}
    )
    domain_counter: Counter = Counter()
    region_counter: Counter = Counter()

    for corpus_file in corpus_files:
        fpath = Path(corpus_file)
        if not fpath.exists():
            print(f"WARNING: File not found, skipping: {corpus_file}")
            continue

        with open(fpath, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    st = r.get("source_type", "unknown")
                    text = r.get("text", "")
                    word_count = len(text.split())

                    stats[st]["docs"] += 1
                    stats[st]["words"] += word_count
                    stats[st]["chars"] += len(text)

                    domain = r.get("domain", r.get("source", "unknown"))
                    domain_counter[domain] += 1

                    region = r.get("language_region", "unknown")
                    region_counter[region] += 1
                except (json.JSONDecodeError, ValueError):
                    continue

    # ── Print formatted statistics ───────────────────────────────────────
    print()
    print("=" * 90)
    print("  BANGLAGSG CORPUS STATISTICS")
    print("=" * 90)
    print()

    # Per-source table
    header = (f"  {'Source Type':35s} {'Docs':>10s} {'Words (M)':>12s} "
              f"{'Est. Tokens (M)':>18s} {'Avg Words/Doc':>15s}")
    print(header)
    print(f"  {'-' * 85}")

    total_docs = 0
    total_words = 0
    total_est_tokens = 0

    for st, s in sorted(stats.items(), key=lambda x: -x[1]["words"]):
        # Estimate token ratio based on source type
        if "code" in st.lower() or "python" in st.lower():
            ratio = TOKENS_PER_WORD["code"]
        elif "en" in st.lower() or "education" in st.lower():
            ratio = TOKENS_PER_WORD["en"]
        else:
            ratio = TOKENS_PER_WORD["bn"]

        est_tokens = s["words"] * ratio
        avg_words = s["words"] / max(s["docs"], 1)

        print(f"  {st:35s} {s['docs']:>10,} {s['words']/1e6:>12.1f} "
              f"{est_tokens/1e6:>18.1f} {avg_words:>15.0f}")

        total_docs += s["docs"]
        total_words += s["words"]
        total_est_tokens += est_tokens

    print(f"  {'-' * 85}")
    print(f"  {'TOTAL':35s} {total_docs:>10,} {total_words/1e6:>12.1f} "
          f"{total_est_tokens/1e6:>18.1f}")
    print()

    # Domain coverage
    print(f"  Unique domains/sources: {len(domain_counter)}")
    print()
    print(f"  Top 20 domains:")
    for domain, count in domain_counter.most_common(20):
        print(f"    {domain:50s} {count:>10,} docs")
    print()

    # Language region distribution
    print(f"  Language region distribution:")
    for region, count in region_counter.most_common():
        pct = 100 * count / max(total_docs, 1)
        print(f"    {region:30s} {count:>10,} docs  ({pct:5.1f}%)")
    print()

    # Disk estimates
    est_disk_gb = total_est_tokens * 1e6 * 2 / (1024 ** 3)  # uint16
    print(f"  Estimated pretokenized disk: ~{est_disk_gb:.1f} GB (uint16)")
    print()
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Compute corpus statistics for the BanglaGSG paper.",
    )
    parser.add_argument("--corpus", default=DEFAULT_CORPUS,
                        help=f"Main corpus JSONL (default: {DEFAULT_CORPUS}).")
    parser.add_argument("--banglish", default=DEFAULT_BANGLISH,
                        help=f"Banglish corpus JSONL (default: {DEFAULT_BANGLISH}).")
    parser.add_argument("--extra-corpus", nargs="*", default=[],
                        help="Additional corpus JSONL files.")

    args = parser.parse_args()
    corpus_files = [args.corpus, args.banglish] + args.extra_corpus
    compute_stats(corpus_files)


if __name__ == "__main__":
    main()

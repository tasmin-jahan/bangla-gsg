#!/usr/bin/env python3
"""
BanglaGSG — Evaluation Set Contamination Check
==================================================

Removes documents from the cleaned corpus that contain verbatim n-gram
overlaps with evaluation benchmarks (SentNoB, BLUB).

Run AFTER clean_pipeline.py, BEFORE pretokenization.

Usage:
  python scripts/contamination_check.py
  python scripts/contamination_check.py --ngram-size 13 --eval-dir saved/data/eval

Reference: BanglaFM_Q1_Data_Plan.md §2.2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_INPUT = "saved/data/cleaned/corpus_cleaned.jsonl"
DEFAULT_OUTPUT = "saved/data/cleaned/corpus_decontaminated.jsonl"
DEFAULT_EVAL_DIR = "saved/data/eval"
DEFAULT_NGRAM_SIZE = 13


def load_eval_ngrams(eval_dir: str, n: int) -> set[str]:
    """Load all n-grams from evaluation text files."""
    eval_path = Path(eval_dir)
    ngrams: set[str] = set()

    if not eval_path.exists():
        print(f"WARNING: Eval directory not found: {eval_dir}")
        print("  Create it with one .txt file per benchmark (one sentence per line).")
        print("  Expected files: sentnob_test.txt, blub_test.txt, etc.")
        return ngrams

    eval_files = list(eval_path.glob("*.txt"))
    if not eval_files:
        print(f"WARNING: No .txt files found in {eval_dir}")
        return ngrams

    for ef in eval_files:
        file_ngrams = 0
        with open(ef, encoding="utf-8") as f:
            for line in f:
                words = line.strip().split()
                for i in range(len(words) - n + 1):
                    ngrams.add(" ".join(words[i : i + n]))
                    file_ngrams += 1
        print(f"  {ef.name}: {file_ngrams:,} {n}-grams")

    print(f"  Total eval n-grams: {len(ngrams):,}")
    return ngrams


def is_contaminated(text: str, eval_ngrams: set[str], n: int) -> bool:
    """Check if text contains any verbatim n-gram from eval sets."""
    words = text.split()
    for i in range(len(words) - n + 1):
        candidate = " ".join(words[i : i + n])
        if candidate in eval_ngrams:
            return True
    return False


def run_contamination_check(
    input_file: str,
    output_file: str,
    eval_dir: str,
    ngram_size: int,
) -> None:
    """Remove contaminated documents from the corpus."""
    in_path = Path(input_file)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("  Run clean_pipeline.py first.")
        return

    print(f"\n{'=' * 60}")
    print(f"  Contamination Check ({ngram_size}-gram)")
    print(f"{'=' * 60}")
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_file}")
    print(f"  Eval:   {eval_dir}")
    print()

    eval_ngrams = load_eval_ngrams(eval_dir, ngram_size)

    if not eval_ngrams:
        print("\n  No eval n-grams loaded — copying input to output unchanged.")
        import shutil
        shutil.copy2(input_file, output_file)
        return

    kept = 0
    removed = 0

    with open(in_path, encoding="utf-8") as in_f, \
         open(out_path, "w", encoding="utf-8") as out_f:
        for line in in_f:
            try:
                record = json.loads(line)
                text = record.get("text", "")
                if is_contaminated(text, eval_ngrams, ngram_size):
                    removed += 1
                else:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1
            except (json.JSONDecodeError, ValueError):
                continue

            if (kept + removed) % 100_000 == 0:
                print(f"    Processed: {kept + removed:,}  "
                      f"(kept: {kept:,}, removed: {removed:,})")

    print(f"\n  ✅ Contamination check complete")
    print(f"     Kept:    {kept:,}")
    print(f"     Removed: {removed:,}")
    print(f"     Output:  {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove eval-set contaminated documents from the corpus.",
    )
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT,
                        help=f"Cleaned corpus JSONL (default: {DEFAULT_INPUT}).")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT,
                        help=f"Output file (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR,
                        help=f"Dir with eval .txt files (default: {DEFAULT_EVAL_DIR}).")
    parser.add_argument("--ngram-size", type=int, default=DEFAULT_NGRAM_SIZE,
                        help=f"N-gram size for overlap detection (default: {DEFAULT_NGRAM_SIZE}).")

    args = parser.parse_args()
    run_contamination_check(
        input_file=args.input,
        output_file=args.output,
        eval_dir=args.eval_dir,
        ngram_size=args.ngram_size,
    )


if __name__ == "__main__":
    main()

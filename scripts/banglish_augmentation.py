#!/usr/bin/env python3
"""
BanglaGSG — Synthetic Banglish Augmentation
==============================================

Generates synthetic Banglish (romanized Bangla) from native Bangla text
via aksharamukha transliteration.  This is the "Option 3" novelty claim:
first foundation model with explicit, large-scale Banglish pretraining.

Input:  saved/data/cleaned/corpus_decontaminated.jsonl  (or corpus_cleaned.jsonl)
Output: saved/data/banglish/synthetic_banglish.jsonl

Usage:
  python scripts/banglish_augmentation.py
  python scripts/banglish_augmentation.py --max-docs 100000

Reference: BanglaFM_Q1_Data_Plan.md Part 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_INPUT = "saved/data/cleaned/corpus_decontaminated.jsonl"
DEFAULT_OUTPUT = "saved/data/banglish/synthetic_banglish.jsonl"
DEFAULT_MAX_DOCS = 1_000_000
ELIGIBLE_REGIONS = {"BD", "BD_WB_mix", "bd", "bd_wb_mix"}
MIN_WORDS_INPUT = 20
MIN_WORDS_OUTPUT = 15

# ── Diacritic → ASCII mapping for natural Banglish appearance ────────────────

DIACRITIC_MAP = {
    "ā": "a", "ī": "i", "ū": "u",
    "ṭ": "t", "ḍ": "d", "ṇ": "n",
    "ś": "sh", "ṣ": "sh",
    "ṃ": "m", "ḥ": "h",
    "ṛ": "ri",
    "ñ": "n", "ṅ": "ng",
    "ṁ": "m",
}


def bangla_to_latin(text: str) -> str:
    """Transliterate Bangla Unicode → Latin (ISO 15919) via aksharamukha."""
    try:
        from aksharamukha import transliterate
        return transliterate.process("Bengali", "ISO", text)
    except ImportError:
        print("ERROR: aksharamukha not installed. Run: pip install aksharamukha")
        sys.exit(1)
    except Exception:
        return ""


def clean_romanized(text: str) -> str:
    """Post-process ISO 15919 diacritics to natural ASCII Banglish."""
    for src, tgt in DIACRITIC_MAP.items():
        text = text.replace(src, tgt)
    return text


def run_augmentation(
    input_file: str,
    output_file: str,
    max_docs: int,
) -> None:
    """Generate synthetic Banglish from Bangla documents."""
    in_path = Path(input_file)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"ERROR: Input corpus not found: {input_file}")
        print("  Run clean_pipeline.py and contamination_check.py first.")
        return

    print(f"{'=' * 60}")
    print(f"  Synthetic Banglish Augmentation")
    print(f"{'=' * 60}")
    print(f"  Input:    {input_file}")
    print(f"  Output:   {output_file}")
    print(f"  Max docs: {max_docs:,}")
    print()

    n_scanned = 0
    n_written = 0
    n_skipped_region = 0
    n_skipped_short = 0
    n_failed = 0

    with open(in_path, encoding="utf-8") as in_f, \
         open(out_path, "w", encoding="utf-8") as out_f:
        for line in in_f:
            if n_written >= max_docs:
                break

            try:
                record = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            n_scanned += 1

            # Only transliterate Bangla-region documents
            lang_region = record.get("language_region", "")
            if lang_region not in ELIGIBLE_REGIONS:
                n_skipped_region += 1
                continue

            text = record.get("text", "")
            if not text or len(text.split()) < MIN_WORDS_INPUT:
                n_skipped_short += 1
                continue

            # Transliterate
            roman = bangla_to_latin(text)
            roman = clean_romanized(roman)
            if len(roman.split()) < MIN_WORDS_OUTPUT:
                n_failed += 1
                continue

            banglish_record = {
                "source": record.get("source", record.get("domain", "scraped")),
                "source_type": "synthetic_banglish",
                "language_region": "BD_banglish_synthetic",
                "text": roman,
                "word_count": len(roman.split()),
                "original_doc_id": record.get("doc_id", ""),
            }
            out_f.write(json.dumps(banglish_record, ensure_ascii=False) + "\n")
            n_written += 1

            if n_written % 10_000 == 0:
                print(f"    Written: {n_written:,} / {n_scanned:,} scanned")

            if n_written % 100_000 == 0:
                out_f.flush()

    print(f"\n{'=' * 60}")
    print(f"  Augmentation Complete")
    print(f"{'=' * 60}")
    print(f"  Scanned:        {n_scanned:,}")
    print(f"  Written:        {n_written:,}")
    print(f"  Skipped region: {n_skipped_region:,}")
    print(f"  Skipped short:  {n_skipped_short:,}")
    print(f"  Failed translit:{n_failed:,}")
    print(f"  Output:         {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Banglish from Bangla corpus via transliteration.",
    )
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT,
                        help=f"Input corpus JSONL (default: {DEFAULT_INPUT}).")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT,
                        help=f"Output file (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS,
                        help=f"Max documents to transliterate (default: {DEFAULT_MAX_DOCS:,}).")

    args = parser.parse_args()
    run_augmentation(
        input_file=args.input,
        output_file=args.output,
        max_docs=args.max_docs,
    )


if __name__ == "__main__":
    main()

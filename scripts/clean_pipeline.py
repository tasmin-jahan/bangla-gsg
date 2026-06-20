#!/usr/bin/env python3
"""
BanglaGSG — Data Cleaning Pipeline
======================================

Full cleaning pipeline for scraped and downloaded Bangla corpus.
Stages: langid → unicode_norm → quality_filter → dedup → output

Input:  saved/data/raw/**/*.jsonl
Output: saved/data/cleaned/corpus_cleaned.jsonl

Usage:
  python scripts/clean_pipeline.py
  python scripts/clean_pipeline.py --raw-dir saved/data/raw/hf --langid-model saved/data/lid.176.bin

Reference: BanglaFM_Q1_Data_Plan.md §2.1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

# ── Config defaults ──────────────────────────────────────────────────────────

DEFAULT_RAW_DIR = "saved/data/raw"
DEFAULT_OUTPUT = "saved/data/cleaned/corpus_cleaned.jsonl"
DEFAULT_LANGID_MODEL = "saved/data/lid.176.bin"

LANGID_THRESHOLD = 0.80
MIN_WORDS = 30
MAX_PUNCT_RATIO = 0.30
MAX_ASCII_RATIO = 0.40
MINHASH_THRESHOLD = 0.80
MINHASH_NUM_PERM = 128
NGRAM_SIZE = 5


# ── Text processing ─────────────────────────────────────────────────────────


def load_normalizer():
    """Load bnunicodenormalizer (lazy import)."""
    try:
        from bnunicodenormalizer import Normalizer
        return Normalizer()
    except ImportError:
        print("WARNING: bnunicodenormalizer not installed. "
              "Run: pip install bnunicodenormalizer")
        print("         Falling back to NFC-only normalization.")
        return None


def normalize_text(text: str, bnorm=None) -> str:
    """Bangla-specific + NFC Unicode normalization."""
    if bnorm is not None:
        try:
            result = bnorm(text)
            text = result["normalized"] if isinstance(result, dict) else str(result)
        except Exception:
            pass
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_langid_model(model_path: str):
    """Load FastText language identification model."""
    try:
        import fasttext
        # Suppress FastText warnings about deprecated load_model
        fasttext.FastText.eprint = lambda x: None
        model = fasttext.load_model(model_path)
        print(f"  Loaded langid model: {model_path}")
        return model
    except ImportError:
        print("ERROR: fasttext not installed. Run: pip install fasttext")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot load langid model at {model_path}: {e}")
        print("  Download it: python scripts/download_datasets.py --download-langid")
        sys.exit(1)


def is_target_language(
    text: str,
    langid_model,
    target_lang: str = "bn",
    threshold: float = LANGID_THRESHOLD,
) -> bool:
    """Check if text is predominantly the target language."""
    clean = text.replace("\n", " ")[:500]
    labels, scores = langid_model.predict(clean, k=1)
    label = labels[0].replace("__label__", "")
    return label == target_lang and scores[0] >= threshold


def quality_filter(text: str) -> bool:
    """Return True if text passes quality checks."""
    words = text.split()
    if len(words) < MIN_WORDS:
        return False
    # Punctuation density
    punct_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if punct_count / max(len(text), 1) > MAX_PUNCT_RATIO:
        return False
    # ASCII ratio (reject English-dominant docs in Bangla pipeline)
    ascii_count = sum(1 for c in text if ord(c) < 128)
    if ascii_count / max(len(text), 1) > MAX_ASCII_RATIO:
        return False
    return True


def get_minhash(text: str, n: int = NGRAM_SIZE):
    """Compute MinHash from n-gram shingles."""
    from datasketch import MinHash
    m = MinHash(num_perm=MINHASH_NUM_PERM)
    words = text.split()
    for i in range(len(words) - n + 1):
        shingle = " ".join(words[i : i + n])
        m.update(shingle.encode("utf-8"))
    return m


# ── Pipeline ─────────────────────────────────────────────────────────────────


def run_pipeline(
    raw_dir: str,
    output_file: str,
    langid_model_path: str,
    skip_langid: bool = False,
    skip_dedup: bool = False,
    bangla_only: bool = True,
) -> None:
    """Run the full cleaning pipeline."""
    raw_path = Path(raw_dir)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all .jsonl files recursively
    raw_files = sorted(raw_path.rglob("*.jsonl"))
    if not raw_files:
        print(f"ERROR: No .jsonl files found in {raw_dir}")
        sys.exit(1)

    print(f"Found {len(raw_files)} raw JSONL files in {raw_dir}")

    # Load models
    bnorm = load_normalizer()

    langid_model = None
    if not skip_langid and bangla_only:
        langid_model = load_langid_model(langid_model_path)

    # Dedup
    lsh = None
    if not skip_dedup:
        try:
            from datasketch import MinHashLSH
            lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_NUM_PERM)
            print("  MinHash LSH deduplication: ENABLED")
        except ImportError:
            print("WARNING: datasketch not installed — skipping dedup.")
            print("         Run: pip install datasketch")

    stats = defaultdict(int)
    doc_id = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for raw_file in raw_files:
            print(f"\nProcessing: {raw_file.relative_to(raw_path)}")
            file_stats = defaultdict(int)

            with open(raw_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue

                    file_stats["total"] += 1
                    text = record.get("text", "")
                    if not text:
                        file_stats["empty"] += 1
                        continue

                    # Determine if this is a Bangla or English source
                    lang_region = record.get("language_region", "").upper()
                    is_english_source = lang_region in ("EN", "ENGLISH", "CODE")

                    # Stage 1: Language ID (skip for known English/code sources)
                    if langid_model and not is_english_source:
                        if not is_target_language(text, langid_model):
                            file_stats["langid_reject"] += 1
                            continue

                    # Stage 2: Unicode normalization
                    text = normalize_text(text, bnorm)
                    record["text"] = text

                    # Stage 3: Quality filter (skip for English — different thresholds)
                    if not is_english_source:
                        if not quality_filter(text):
                            file_stats["quality_reject"] += 1
                            continue
                    else:
                        # Basic length filter for English
                        if len(text.split()) < MIN_WORDS:
                            file_stats["quality_reject"] += 1
                            continue

                    # Stage 4: MinHash deduplication
                    if lsh is not None:
                        mh = get_minhash(text)
                        key = f"doc_{doc_id}"
                        try:
                            result = lsh.query(mh)
                            if result:
                                file_stats["dedup_reject"] += 1
                                continue
                            lsh.insert(key, mh)
                        except Exception:
                            pass

                    # Stage 5: Write clean record
                    record["doc_id"] = doc_id
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    doc_id += 1
                    file_stats["kept"] += 1

            # Per-file summary
            print(f"  {raw_file.name}: {dict(file_stats)}")
            for k, v in file_stats.items():
                stats[k] += v

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  CLEANING PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total processed:    {stats['total']:>10,}")
    print(f"  Empty:              {stats['empty']:>10,}")
    print(f"  LangID rejected:    {stats['langid_reject']:>10,}")
    print(f"  Quality rejected:   {stats['quality_reject']:>10,}")
    print(f"  Dedup rejected:     {stats['dedup_reject']:>10,}")
    print(f"  Kept:               {stats['kept']:>10,}  "
          f"({100 * stats['kept'] / max(stats['total'], 1):.1f}%)")
    print(f"  Output:             {out_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Clean and deduplicate raw Bangla corpus for BanglaGSG.",
    )
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR,
                        help=f"Directory containing raw JSONL files (default: {DEFAULT_RAW_DIR}).")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT,
                        help=f"Output cleaned JSONL file (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--langid-model", default=DEFAULT_LANGID_MODEL,
                        help=f"Path to FastText lid model (default: {DEFAULT_LANGID_MODEL}).")
    parser.add_argument("--skip-langid", action="store_true",
                        help="Skip language identification filtering.")
    parser.add_argument("--skip-dedup", action="store_true",
                        help="Skip MinHash deduplication.")

    args = parser.parse_args()
    run_pipeline(
        raw_dir=args.raw_dir,
        output_file=args.output,
        langid_model_path=args.langid_model,
        skip_langid=args.skip_langid,
        skip_dedup=args.skip_dedup,
    )


if __name__ == "__main__":
    main()

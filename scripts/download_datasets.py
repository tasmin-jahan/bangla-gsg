#!/usr/bin/env python3
"""
BanglaGSG — HuggingFace Dataset Downloader
=============================================

Downloads all datasets specified in the BanglaFM data plan via the
HuggingFace `datasets` library (streaming mode to avoid full download).
Each source is saved as a separate JSONL file in saved/data/raw/hf/.

Usage:
  # Download all sources
  python scripts/download_datasets.py

  # Download a single source
  python scripts/download_datasets.py --source wikipedia_bn

  # Limit samples per source (for testing)
  python scripts/download_datasets.py --max-samples 1000

  # Also download the FastText language-ID model
  python scripts/download_datasets.py --download-langid

Reference: BanglaFM_Q1_Data_Plan.md §2.3, BanglaFM_Complete_Guide.md §3.2–3.6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path
from collections import defaultdict

# ── Source registry ──────────────────────────────────────────────────────────

HF_SOURCES = {
    "culturax_bn": {
        "hf_path": "uonlp/CulturaX",
        "hf_name": "bn",
        "text_col": "text",
        "max_samples": 2_000_000,
        "source_type": "web_mixed",
        "language_region": "BD_WB_mix",
        "description": "CulturaX Bangla — mC4+OSCAR cleaned, deduplicated",
    },
    "oscar_bn": {
        "hf_path": "oscar-corpus/OSCAR-2301",
        "hf_name": "bn",
        "text_col": "text",
        "max_samples": 1_000_000,
        "source_type": "web_informal",
        "language_region": "BD_WB_mix",
        "description": "OSCAR 23.01 Bangla — CommonCrawl filtered",
    },
    "cc100_bn": {
        "hf_path": "statmt/cc100",
        "hf_name": "bn",
        "text_col": "text",
        "max_samples": 500_000,
        "source_type": "web_noisy",
        "language_region": "BD_WB_mix",
        "description": "CC-100 Bangla — noisy web (needs KenLM filter)",
    },
    "wikipedia_bn": {
        "hf_path": "wikimedia/wikipedia",
        "hf_name": "20231101.bn",
        "text_col": "text",
        "max_samples": None,
        "source_type": "encyclopedic",
        "language_region": "BD_WB_mix",
        "description": "Bangla Wikipedia — formal, clean",
    },
    "fineweb_edu": {
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "hf_name": "sample-10BT",
        "text_col": "text",
        "max_samples": 2_000_000,
        "source_type": "formal_education",
        "language_region": "EN",
        "description": "FineWeb-Edu English — education-filtered quality",
    },
    "banglishrev": {
        "hf_path": "BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset",
        "hf_name": None,
        "text_col": "review",
        "max_samples": None,
        "source_type": "code_mixed_commerce",
        "language_region": "BD_banglish",
        "description": "BanglishRev — 1.74M Bangla/English/Banglish reviews",
    },
    "banglatlit": {
        "hf_path": "sbnltk/BanglaTLit",
        "hf_name": None,
        "text_col": "roman",
        "max_samples": None,
        "source_type": "banglish_parallel",
        "language_region": "BD_banglish",
        "description": "BanglaTLit — 288K romanized↔Bangla pairs",
    },
    "bengali_transliteration": {
        "hf_path": "SKNahin/bengali-transliteration-data",
        "hf_name": None,
        "text_col": "rm",
        "max_samples": None,
        "source_type": "banglish_parallel",
        "language_region": "BD_banglish",
        "description": "Bengali transliteration — Banglish↔Bangla pairs",
    },
    "the_stack_python": {
        "hf_path": "bigcode/the-stack",
        "hf_name": "data/python",
        "text_col": "content",
        "max_samples": 500_000,
        "source_type": "code_python",
        "language_region": "code",
        "description": "The Stack Python — code reasoning scaffold",
    },
}

# ── Helpers ──────────────────────────────────────────────────────────────────


def normalize_basic(text: str) -> str:
    """Lightweight NFC normalization applied at download time."""
    return unicodedata.normalize("NFC", text).strip()


def download_langid_model(output_dir: Path) -> None:
    """Download the FastText language identification model."""
    import urllib.request

    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    dest = output_dir / "lid.176.bin"

    if dest.exists():
        print(f"  FastText langid model already exists: {dest}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading FastText langid model (~126 MB)...")
    print(f"  URL:  {url}")
    print(f"  Dest: {dest}")

    urllib.request.urlretrieve(url, str(dest))
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  ✅ Downloaded ({size_mb:.1f} MB)")


def download_source(
    source_key: str,
    config: dict,
    output_dir: Path,
    max_samples_override: int | None = None,
) -> int:
    """
    Download a single HuggingFace dataset source to JSONL.

    Returns the number of documents written.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` not installed. Run: pip install datasets")
        sys.exit(1)

    output_file = output_dir / f"{source_key}.jsonl"

    # Resume support: count existing records
    existing_count = 0
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            existing_count = sum(1 for _ in f)

    max_samples = max_samples_override or config.get("max_samples")
    if max_samples and existing_count >= max_samples:
        print(f"  ✅ Already have {existing_count:,} records (target: {max_samples:,})")
        return existing_count

    print(f"\n{'─' * 50}")
    print(f"  Source:  {source_key}")
    print(f"  HF:     {config['hf_path']}  ({config.get('hf_name', 'default')})")
    print(f"  Type:   {config['source_type']}")
    print(f"  Target: {max_samples:,}" if max_samples else "  Target: all")
    print(f"  Resume: {existing_count:,} existing records")
    print(f"{'─' * 50}")

    try:
        ds = load_dataset(
            config["hf_path"],
            config.get("hf_name"),
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  ❌ Failed to load dataset: {e}")
        return existing_count

    n_written = existing_count
    n_skipped = 0
    start_time = time.time()

    with open(output_file, "a", encoding="utf-8") as out_f:
        for i, item in enumerate(ds):
            # Skip already-downloaded records
            if i < existing_count:
                continue

            text = item.get(config["text_col"], "")
            if not text or len(text.split()) < 20:
                n_skipped += 1
                continue

            text = normalize_basic(text)

            record = {
                "source": config["hf_path"],
                "source_type": config["source_type"],
                "language_region": config["language_region"],
                "text": text,
                "word_count": len(text.split()),
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

            if max_samples and n_written >= max_samples:
                break

            if n_written % 10_000 == 0:
                elapsed = time.time() - start_time
                rate = (n_written - existing_count) / max(elapsed, 1)
                print(f"    {n_written:>10,} written  |  {rate:.0f} docs/sec")

            if n_written % 100_000 == 0:
                out_f.flush()

    elapsed = time.time() - start_time
    print(f"  ✅ Done: {n_written:,} total ({n_written - existing_count:,} new, "
          f"{n_skipped:,} skipped) in {elapsed:.0f}s")

    return n_written


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace datasets for BanglaGSG pretraining.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", "-s",
        choices=list(HF_SOURCES.keys()),
        default=None,
        help="Download a specific source only. If not set, downloads all.",
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=None,
        help="Override max samples per source (useful for testing).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="saved/data/raw/hf",
        help="Output directory for JSONL files (default: saved/data/raw/hf).",
    )
    parser.add_argument(
        "--download-langid",
        action="store_true",
        help="Also download the FastText lid.176.bin language-ID model.",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available sources and exit.",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("\nAvailable dataset sources:\n")
        for key, cfg in HF_SOURCES.items():
            samples = cfg.get("max_samples")
            limit = f"{samples:,}" if samples else "all"
            print(f"  {key:30s}  {cfg['source_type']:25s}  max={limit:>12s}")
            print(f"    {cfg['description']}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"  BanglaGSG Dataset Downloader")
    print(f"{'=' * 60}")
    print(f"  Output: {output_dir}")
    print()

    # Download FastText langid model if requested
    if args.download_langid:
        download_langid_model(output_dir.parent)

    # Select sources
    if args.source:
        sources = {args.source: HF_SOURCES[args.source]}
    else:
        sources = HF_SOURCES

    # Download
    stats = {}
    for key, cfg in sources.items():
        n = download_source(key, cfg, output_dir, args.max_samples)
        stats[key] = n

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Download Summary")
    print(f"{'=' * 60}")
    total = 0
    for key, count in stats.items():
        print(f"  {key:30s}  {count:>10,} documents")
        total += count
    print(f"  {'TOTAL':30s}  {total:>10,} documents")
    print()


if __name__ == "__main__":
    main()

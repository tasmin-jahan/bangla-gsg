#!/usr/bin/env python3
"""
BanglaGSG — Plan A Dataset Downloader
=========================================
Downloads only the 3 Plan A HF sources:
  - CulturaX bn  (Bangla, mixed BD/WB)
  - FineWeb-Edu  (English, sampled)
  - BanglishRev  (authentic Banglish + code-mixed)
Output: saved/data/raw/hf/*.jsonl

Usage:
  python scripts/download_sources.py                     # Download all
  python scripts/download_sources.py --source culturax   # Single source
  python scripts/download_sources.py --max-docs 10000    # Test mode
"""

import argparse
import json
import time
import unicodedata
from pathlib import Path

HF_SOURCES = {
    "culturax": {
        "hf_path": "uonlp/CulturaX",
        "hf_name": "bn",
        "text_col": "text",
        "max_docs": None,
        "source_type": "web_mixed",
        "language_region": "BD_WB_mix",
    },
    "fineweb": {
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "hf_name": "sample-10BT",
        "text_col": "text",
        "max_docs": 2_000_000,
        "source_type": "formal_education",
        "language_region": "EN",
    },
    "banglishrev": {
        "hf_path": "BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset",
        "hf_name": None,
        "text_col": "review",
        "max_docs": None,
        "source_type": "code_mixed_commerce",
        "language_region": "BD_banglish",
    },
}


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def download_source(key: str, cfg: dict, output_dir: Path, max_override: int | None = None) -> int:
    from datasets import load_dataset

    out_file = output_dir / f"{key}.jsonl"

    existing = 0
    if out_file.exists():
        with open(out_file) as f:
            existing = sum(1 for _ in f)

    max_docs = max_override or cfg.get("max_docs")
    if max_docs and existing >= max_docs:
        print(f"  [{key}] Already have {existing:,} docs, skipping")
        return existing

    print(f"\n  [{key}] Loading {cfg['hf_path']}/{cfg.get('hf_name', 'default')} ...")
    ds = load_dataset(cfg["hf_path"], cfg.get("hf_name"), split="train",
                      streaming=True, trust_remote_code=True)

    n_written = existing
    n_skipped = 0
    t0 = time.time()

    with open(out_file, "a", encoding="utf-8") as f:
        for i, item in enumerate(ds):
            if i < existing:
                continue

            text = item.get(cfg["text_col"], "")
            if not text or len(text.split()) < 20:
                n_skipped += 1
                continue

            text = normalize(text)
            record = {
                "source": cfg["hf_path"],
                "source_type": cfg["source_type"],
                "language_region": cfg["language_region"],
                "text": text,
                "word_count": len(text.split()),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

            if max_docs and n_written >= max_docs:
                break

            if n_written % 10_000 == 0:
                elapsed = time.time() - t0
                rate = (n_written - existing) / max(elapsed, 1)
                print(f"    [{key}] {n_written:,} written  |  {rate:.0f} docs/sec")

    elapsed = time.time() - t0
    print(f"  [{key}] Done: {n_written:,} docs ({n_written-existing:,} new, "
          f"{n_skipped:,} skipped) in {elapsed:.0f}s")
    return n_written


def main():
    parser = argparse.ArgumentParser(description="Download Plan A datasets")
    parser.add_argument("--source", choices=list(HF_SOURCES), default=None)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--output-dir", default="saved/data/raw/hf")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = {args.source: HF_SOURCES[args.source]} if args.source else HF_SOURCES

    for key, cfg in sources.items():
        download_source(key, cfg, output_dir, args.max_docs)


if __name__ == "__main__":
    main()

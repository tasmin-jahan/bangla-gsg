#!/usr/bin/env python3
"""
BanglaGSG — BD:WB Ratio Auditor
====================================
Samples documents from CulturaX bn and estimates the BD:WB ratio
by examining URL domains and content markers.

Usage:
  python scripts/audit_bd_wb.py --input saved/data/raw/hf/culturax.jsonl --sample 2000
  python scripts/audit_bd_wb.py --quick     # Just sample 500, print ratio
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

# Known BD domains
BD_DOMAINS = [
    "prothomalo.com", "bdnews24.com", "kalerkantho.com", "ittefaq.com.bd",
    "samakal.com", "jugantor.com", "banglanews24.com", "dailystar.net",
    "thedailystar.net", "dhakatribune.com", "banglatribune.com",
    "banglainsider.com", "en.prothom-alo.com", "nayadiganta.com",
    "bonikbarta.net", "dailyjanakantha.com", "amardesh24.com",
    "alokitobangladesh.com", "bhorerkagoj.com", "sarabangla.net",
    "bd-pratidin.com", "manobkantha.com", "dailyinqilab.com",
    "sangbadbd.com", "sharebiz.net", "bsmrau.edu.bd",
    ".bd",  # Any .bd domain
]

# Known WB/Indian domains
WB_DOMAINS = [
    "anandabazar.com", "bartamanpatrika.com", "sangbadpratidin.in",
    "znews24.in", "bangla.hindustantimes.com", "bengali.indianexpress.com",
    "bangla.zeenews.com", "bengali.news18.com", "bangla.aajtak.in",
    "tv9bangla.com", "kolkatatoday.com", "banglalive.com",
    "ebela.in", "bongodorshon.com", "bengali.abplive.com",
    "timesofindia.indiatimes.com", ".in",
]

# BD-specific content markers (words/phrases used predominantly in BD Bangla)
BD_MARKERS = [
    "বাংলাদেশ", "ঢাকা", "জাতীয় সংসদ", "বাংলাদেশী",
    "টাকা", "বাংলাদেশের", "ইত্তেফাক", "প্রথম আলো",
    "বিডিনিউজ", "সমকাল",
]

# WB-specific content markers
WB_MARKERS = [
    "পশ্চিমবঙ্গ", "কলকাতা", "মমতা", "নবান্ন",
    "পশ্চিমবঙ্গের", "তৃণমূল", "ভারতীয়",
]


def classify_domain(url: str) -> str | None:
    """Classify a URL as BD, WB, or None if unknown."""
    url = url.lower()
    for d in BD_DOMAINS:
        if d in url:
            return "BD"
    for d in WB_DOMAINS:
        if d in url:
            return "WB"
    return None


def classify_content(text: str) -> str | None:
    """Classify text by Bangla regional markers."""
    for m in BD_MARKERS:
        if m in text:
            return "BD"
    for m in WB_MARKERS:
        if m in text:
            return "WB"
    return None


def run_audit(input_file: str, sample_size: int = 1000, seed: int = 42):
    """Sample documents, classify BD/WB, print results."""
    in_path = Path(input_file)
    if not in_path.exists():
        print(f"ERROR: {input_file} not found")
        return

    print(f"\n{'='*60}")
    print(f"  BD:WB Ratio Audit")
    print(f"{'='*60}")
    print(f"  Input:  {input_file}")
    print(f"  Sample: {sample_size:,} docs")
    print()

    # Load all records
    records = []
    with open(in_path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                continue

    print(f"  Total available: {len(records):,} docs")
    sample = random.Random(seed).sample(records, min(sample_size, len(records)))

    classified = Counter()
    details = {"BD": [], "WB": [], "unclear": []}

    for r in sample:
        # Try URL first (if source field contains URL)
        source = r.get("source", "")
        region = classify_domain(source)

        # Try text content if URL didn't give a clear answer
        if not region:
            text = r.get("text", "")[:500]
            region = classify_content(text)

        if region:
            classified[region] += 1
            details[region].append(source)
        else:
            classified["unclear"] += 1
            details["unclear"].append(source)

    total_classified = classified.get("BD", 0) + classified.get("WB", 0)
    bd_pct = 100 * classified.get("BD", 0) / max(total_classified, 1)
    wb_pct = 100 * classified.get("WB", 0) / max(total_classified, 1)
    unclear_pct = 100 * classified.get("unclear", 0) / max(len(sample), 1)

    print(f"\n  Results:")
    print(f"    BD:      {classified['BD']:>5,}  ({bd_pct:.0f}% of classified)")
    print(f"    WB:      {classified['WB']:>5,}  ({wb_pct:.0f}% of classified)")
    print(f"    Unclear: {classified['unclear']:>5,}  ({unclear_pct:.0f}% of total)")
    print(f"    Total:   {len(sample):>5,}")
    print()

    if details["unclear"]:
        print(f"  Sample of unclear sources (first 10):")
        for s in details["unclear"][:10]:
            print(f"    {s[:100]}")
    print()

    # Estimate
    if total_classified > 0:
        est_bd_ratio = classified["BD"] / total_classified
        print(f"  Estimated BD:WB ratio in sample = {bd_pct:.0f}:{wb_pct:.0f}")
        print(f"  Estimated BD fraction: {est_bd_ratio:.1%}")
        print()
        print(f"  To achieve 80:20 BD:WB, keep all BD docs and subsample WB to:")
        wb_keep_ratio = (classified["BD"] * 0.2) / (classified["WB"] * 0.8) if classified["WB"] else 1
        print(f"    Keep {wb_keep_ratio:.1%} of WB docs")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BD:WB ratio auditor")
    parser.add_argument("--input", default="saved/data/raw/hf/culturax.jsonl")
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Quick audit with 500 samples")
    args = parser.parse_args()

    if args.quick:
        args.sample = 500

    run_audit(args.input, args.sample, args.seed)

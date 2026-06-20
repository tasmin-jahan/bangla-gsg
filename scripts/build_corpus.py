#!/usr/bin/env python3
"""
BanglaGSG — Final Corpus Builder (Plan A)
=============================================
Assembles the final training corpus from all sources:
  1. CulturaX bn (subsample WB docs to achieve target BD:WB ratio)
  2. Prothom Alo (scraped BD news)
  3. FineWeb-Edu (English)
  4. BanglishRev (authentic Banglish + code-mixed)
  5. Synthetic Banglish (transliterated from BD sources)
  6. Code-mixed (synthetic mixing of Bangla + Banglish + English)

Output: saved/data/final/banglagsg_corpus.jsonl
Usage:
  python scripts/build_corpus.py                                          # Full pipeline
  python scripts/build_corpus.py --target-bd-ratio 0.80 --dry-run         # Preview stats
  python scripts/build_corpus.py --skip-banglish --skip-codemixed         # Minimal build
"""

import argparse
import json
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────

RAW_DIR = Path("saved/data/raw")
FINAL_DIR = Path("saved/data/final")
BANGLISH_DIR = Path("saved/data/banglish")

# BD:WB ratio weights for each HF source
SOURCE_BD_WEIGHT = {
    "culturax": 0.35,     # estimate: 35% BD in CulturaX bn
    "fineweb": 0.0,       # English, no BD/WB
    "banglishrev": 1.0,   # Bangladesh-sourced reviews → BD
}

SOURCE_REGION = {
    "culturax": "BD_WB_mix",
    "fineweb": "EN",
    "banglishrev": "BD_banglish",
    "prothomalo": "BD",
}


def load_jsonl(path: Path) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def estimate_bd_ratio(records: list[dict], source_key: str) -> float:
    """Estimate how many docs are BD vs WB in a source."""
    # For now, use the SOURCE_BD_WEIGHT heuristic
    # In practice, run audit_bd_wb.py for real numbers
    return SOURCE_BD_WEIGHT.get(source_key, 0.5)


def pipeline(args):
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    output_path = FINAL_DIR / args.output

    print(f"{'='*60}")
    print(f"  BanglaGSG Corpus Builder (Plan A)")
    print(f"{'='*60}")
    print(f"  Output:  {output_path}")
    print(f"  BD:WB target: {args.target_bd_ratio:.0%}:{1-args.target_bd_ratio:.0%}")
    print()

    all_records = []
    stats = defaultdict(lambda: {"docs": 0, "words": 0})

    # ── 1. CulturaX bn (with WB subsampling) ─────────────────────────────
    culturax_path = RAW_DIR / "hf" / "culturax.jsonl"
    if culturax_path.exists():
        records = load_jsonl(culturax_path)
        n_total = len(records)
        est_bd = estimate_bd_ratio(records, "culturax")

        # Split: BD docs stay, WB docs get subsampled
        bd_docs = records[:int(n_total * est_bd)]
        wb_docs = records[int(n_total * est_bd):]

        # Subsampling: keep only enough WB to hit target BD:WB ratio
        # target_bd_ratio = BD / (BD + WB_kept)
        # WB_kept = BD * (1 - target_bd_ratio) / target_bd_ratio
        target_bd = args.target_bd_ratio
        n_wb_keep = int(len(bd_docs) * (1 - target_bd) / target_bd) if target_bd < 1 else 0
        n_wb_keep = min(n_wb_keep, len(wb_docs))
        wb_kept = random.sample(wb_docs, n_wb_keep) if n_wb_keep > 0 else []

        kept = bd_docs + wb_kept
        actual_bd_ratio = len(bd_docs) / max(len(kept), 1)

        for r in kept:
            r["_source_file"] = "culturax"
        all_records.extend(kept)

        stats["culturax"]["docs"] = len(kept)
        stats["culturax"]["words"] = sum(r.get("word_count", 0) for r in kept)
        print(f"  CulturaX:  {n_total:,} total → {len(kept):,} kept "
              f"(BD:WB ≈ {actual_bd_ratio:.0%}:{1-actual_bd_ratio:.0%})")
    else:
        print(f"  WARNING: {culturax_path} not found. Download first.")

    # ── 2. Prothom Alo (scraped) ─────────────────────────────────────────
    prothomalo_path = RAW_DIR / "prothomalo_raw.jsonl"
    if prothomalo_path.exists() and not args.skip_scraped:
        records = load_jsonl(prothomalo_path)
        for r in records:
            r["_source_file"] = "prothomalo"
        all_records.extend(records)
        stats["prothomalo"]["docs"] = len(records)
        stats["prothomalo"]["words"] = sum(r.get("word_count", 0) for r in records)
        print(f"  Prothom Alo: {len(records):,} docs")
    else:
        print(f"  Prothom Alo: not scraped yet (run scripts/parallel_prothomalo.py)")

    # ── 3. FineWeb-Edu (English) ─────────────────────────────────────────
    fineweb_path = RAW_DIR / "hf" / "fineweb.jsonl"
    if fineweb_path.exists():
        records = load_jsonl(fineweb_path)
        for r in records:
            r["_source_file"] = "fineweb"
        all_records.extend(records)
        stats["fineweb"]["docs"] = len(records)
        stats["fineweb"]["words"] = sum(r.get("word_count", 0) for r in records)
        print(f"  FineWeb-Edu: {len(records):,} docs")
    else:
        print(f"  WARNING: {fineweb_path} not found. Download first.")

    # ── 4. BanglishRev ───────────────────────────────────────────────────
    banglishrev_path = RAW_DIR / "hf" / "banglishrev.jsonl"
    if banglishrev_path.exists():
        records = load_jsonl(banglishrev_path)
        for r in records:
            r["_source_file"] = "banglishrev"
        all_records.extend(records)
        stats["banglishrev"]["docs"] = len(records)
        stats["banglishrev"]["words"] = sum(r.get("word_count", 0) for r in records)
        print(f"  BanglishRev: {len(records):,} docs")
    else:
        print(f"  WARNING: {banglishrev_path} not found.")

    # ── 5. Synthetic Banglish ────────────────────────────────────────────
    if not args.skip_banglish:
        syn_path = BANGLISH_DIR / "synthetic_banglish.jsonl"
        if syn_path.exists():
            records = load_jsonl(syn_path)
            for r in records:
                r["_source_file"] = "synthetic_banglish"
            all_records.extend(records)
            stats["synthetic_banglish"]["docs"] = len(records)
            stats["synthetic_banglish"]["words"] = sum(r.get("word_count", 0) for r in records)
            print(f"  Synthetic Banglish: {len(records):,} docs")
        else:
            print(f"  Synthetic Banglish: not generated yet (run scripts/banglish_augmentation.py)")

    # ── 6. Code-mixed synthetic ──────────────────────────────────────────
    if not args.skip_codemixed:
        # Generate code-mixed docs by interleaving Bangla + English + Banglish
        codemixed_records = generate_codemixed(all_records, args.codemixed_docs, args.seed)
        if codemixed_records:
            for r in codemixed_records:
                r["_source_file"] = "codemixed"
            all_records.extend(codemixed_records)
            stats["codemixed"]["docs"] = len(codemixed_records)
            stats["codemixed"]["words"] = sum(r.get("word_count", 0) for r in codemixed_records)
            print(f"  Code-mixed: {len(codemixed_records):,} docs")

    # ── Shuffle and write ────────────────────────────────────────────────
    random.shuffle(all_records)

    # Strip internal fields
    for r in all_records:
        r.pop("_source_file", None)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Summary ──────────────────────────────────────────────────────────
    total_words = sum(s["words"] for s in stats.values())
    total_docs = sum(s["docs"] for s in stats.values())
    est_tokens = total_words * 1.35  # rough estimate for mixed tokenizer

    print(f"\n{'='*60}")
    print(f"  BUILD COMPLETE")
    print(f"{'='*60}")
    for key, s in sorted(stats.items(), key=lambda x: -x[1]["docs"]):
        print(f"  {key:25s} {s['docs']:>10,} docs  {s['words']/1e6:>8.1f}M words")
    print(f"  {'─'*50}")
    print(f"  {'TOTAL':25s} {total_docs:>10,} docs  {total_words/1e6:>8.1f}M words")
    print(f"  Est. tokens (custom tok): ~{est_tokens/1e6:.0f}M ({est_tokens/1e9:.1f}B)")
    print(f"  Output: {output_path}")
    print()


def generate_codemixed(records: list[dict], n_target: int, seed: int) -> list[dict]:
    """Create synthetic code-mixed docs from existing records."""
    if n_target == 0 or len(records) < 10:
        return []

    # Separate by region
    bangla = [r for r in records if r.get("language_region", "").startswith("BD")]
    english = [r for r in records if r.get("language_region") == "EN"]
    banglish = [r for r in records if "banglish" in r.get("language_region", "").lower()]

    if not bangla or not english:
        return []

    rng = random.Random(seed)
    result = []

    for _ in range(min(n_target, len(bangla))):
        b = rng.choice(bangla)
        e = rng.choice(english) if english else b
        bl = rng.choice(banglish) if banglish else b

        b_text = b.get("text", "")
        e_text = e.get("text", "")
        bl_text = bl.get("text", "")

        if not b_text or not e_text:
            continue

        # Interleave: Bangla → English → Banglish
        b_words = b_text.split()
        e_words = e_text.split()
        bl_words = bl_text.split()

        # Take first half of Bangla, mix with English and Banglish
        mid = len(b_words) // 2
        mixed = " ".join(b_words[:mid]) + " " + " ".join(e_words[:30]) + " " + " ".join(bl_words[:20])

        result.append({
            "source": "synthetic_codemixed",
            "source_type": "code_mixed_synthetic",
            "language_region": "BD_codemixed",
            "text": mixed.strip(),
            "word_count": len(mixed.split()),
        })

    return result


def main():
    parser = argparse.ArgumentParser(description="Build Plan A corpus")
    parser.add_argument("--output", default="banglagsg_corpus.jsonl")
    parser.add_argument("--target-bd-ratio", type=float, default=0.80)
    parser.add_argument("--codemixed-docs", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing")
    parser.add_argument("--skip-scraped", action="store_true")
    parser.add_argument("--skip-banglish", action="store_true")
    parser.add_argument("--skip-codemixed", action="store_true")

    args = parser.parse_args()
    if args.dry_run:
        args.output = "/dev/null"

    pipeline(args)


if __name__ == "__main__":
    main()

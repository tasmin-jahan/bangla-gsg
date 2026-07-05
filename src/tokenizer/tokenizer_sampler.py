"""
Sample ~1B words from Bangla + English for tokenizer training.

Ratio: 85% Bangla / 15% English (matches target training composition).
Reads from cleaned JSONL files.
Shuffles output so tokenizer sees interleaved languages.

Output:
  saved/data/tokenizer_set/corpus.jsonl  — shuffled mix

Usage:
  python src/tokenizer/tokenizer_sampler.py
  python src/tokenizer/tokenizer_sampler.py --total-words 500_000_000
  python src/tokenizer/tokenizer_sampler.py --ratio 0.80
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

CLEANED_DIR = Path("saved/data/cleaned")
OUTPUT_DIR = Path("saved/data/tokenizer_set")
CORPUS_PATH = OUTPUT_DIR / "corpus.jsonl"

DEFAULT_TOTAL_WORDS = 1_000_000_000
DEFAULT_BN_RATIO = 0.85

BANGLA_SOURCES = [
    CLEANED_DIR / "bangla.jsonl",
]
ENGLISH_SOURCES = [
    CLEANED_DIR / "english.jsonl",
]


def _count_words(path: Path) -> int:
    """Fast word count — reads in chunks."""
    count = 0
    with open(path, "r") as f:
        for line in f:
            try:
                doc = json.loads(line)
                count += len(doc.get("text", "").split())
            except json.JSONDecodeError:
                continue
    return count


def _sample_source(
    sources: list[Path],
    word_budget: int,
    output_path: Path,
    desc: str,
) -> int:
    """Stream sources, sample docs until word budget is met. Returns words written."""
    # Count available words
    available = 0
    for src in sources:
        if src.exists():
            available += _count_words(src)

    # Adjust budget if not enough data
    actual_budget = min(word_budget, available)
    if actual_budget < word_budget:
        print(f"  [sampler] WARNING: {desc} only has ~{available:,} words, "
              f"using all of it (budget was {word_budget:,})")

    print(f"  [sampler] {desc}: sampling {actual_budget:,} words from {available:,} available")

    words_written = 0
    with open(output_path, "w") as fout:
        for src in sources:
            if not src.exists():
                continue
            with open(src, "r") as f:
                for line in f:
                    if words_written >= actual_budget:
                        break
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = doc.get("text", "").strip()
                    if not text:
                        continue
                    wc = len(text.split())
                    fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    words_written += wc
                if words_written >= actual_budget:
                    break

    return words_written


def _shuffle_merge(bangla_path: Path, english_path: Path, output_path: Path):
    """Random 2-way merge — streaming, memory-efficient."""
    # Count lines in each
    def count_lines(p):
        n = 0
        with open(p, "rb") as f:
            for _ in f:
                n += 1
        return n

    bn_count = count_lines(bangla_path)
    en_count = count_lines(english_path)
    total = bn_count + en_count

    print(f"  [sampler] Shuffle-merging {bn_count:,} Bangla + {en_count:,} English = {total:,} docs")

    # Create weighted random choices
    # Probability of picking Bangla = bn_count / total
    bn_prob = bn_count / total

    with open(bangla_path, "r") as fb, \
         open(english_path, "r") as fe, \
         open(output_path, "w") as fout:

        bn_line = fb.readline()
        en_line = fe.readline()

        bn_exhausted = not bn_line
        en_exhausted = not en_line

        while not bn_exhausted or not en_exhausted:
            if bn_exhausted:
                fout.write(en_line)
                en_line = fe.readline()
                en_exhausted = not en_line
            elif en_exhausted:
                fout.write(bn_line)
                bn_line = fb.readline()
                bn_exhausted = not bn_line
            elif random.random() < bn_prob:
                fout.write(bn_line)
                bn_line = fb.readline()
                bn_exhausted = not bn_line
            else:
                fout.write(en_line)
                en_line = fe.readline()
                en_exhausted = not en_line


def main():
    parser = argparse.ArgumentParser(description="Sample 1B words for tokenizer training.")
    parser.add_argument("--total-words", type=int, default=DEFAULT_TOTAL_WORDS,
                        help=f"Total word budget (default: {DEFAULT_TOTAL_WORDS:,}).")
    parser.add_argument("--ratio", type=float, default=DEFAULT_BN_RATIO,
                        help=f"Bangla ratio (default: {DEFAULT_BN_RATIO}).")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bn_words = int(args.total_words * args.ratio)
    en_words = args.total_words - bn_words

    print(f"── Tokenizer Corpus Sampler ──")
    print(f"   Total words : {args.total_words:,}")
    print(f"   Bangla      : {bn_words:,} ({args.ratio * 100:.0f}%)")
    print(f"   English     : {en_words:,} ({(1 - args.ratio) * 100:.0f}%)")
    print()

    # Step 1: Sample Bangla
    print("[1/3] Sampling Bangla...")
    bn_path = OUTPUT_DIR / "_bangla_sample.jsonl"
    bn_actual = _sample_source(BANGLA_SOURCES, bn_words, bn_path, "Bangla")
    print(f"       Wrote {bn_actual:,} words\n")

    # Step 2: Sample English
    print("[2/3] Sampling English...")
    en_path = OUTPUT_DIR / "_english_sample.jsonl"
    en_actual = _sample_source(ENGLISH_SOURCES, en_words, en_path, "English")
    print(f"       Wrote {en_actual:,} words\n")

    # Step 3: Shuffle-merge
    print("[3/3] Shuffling...")
    _shuffle_merge(bn_path, en_path, CORPUS_PATH)

    # Cleanup intermediates
    bn_path.unlink()
    en_path.unlink()

    total = bn_actual + en_actual
    actual_ratio = bn_actual / max(total, 1)

    size_gb = CORPUS_PATH.stat().st_size / (1024 ** 3)
    print(f"\nDone.")
    print(f"  Bangla words   : {bn_actual:>12,}")
    print(f"  English words  : {en_actual:>12,}")
    print(f"  Total words    : {total:>12,}")
    print(f"  Actual ratio   : {actual_ratio:.1%} BN / {1 - actual_ratio:.1%} EN")
    print(f"  Output         : {CORPUS_PATH}  ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()

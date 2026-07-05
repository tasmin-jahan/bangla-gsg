"""
Count tokens in JSONL, Parquet, or NPY files using a HuggingFace tokenizer.

Reports total tokens, tokens per doc stats, and overall token count.

Usage:
  python utils/count_tokens.py
    --tokenizer saved/tokenizer
    --dataset saved/data/tokenizer_corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np


def load_tokenizer(path: str):
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(path)


def count_tokens(path: Path, tokenizer):
    total_tokens = 0
    total_docs = 0
    token_counts = []

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = doc.get("text", "").strip()
                if not text:
                    continue

                tokens = tokenizer.encode(text, add_special_tokens=False)
                n = len(tokens)
                total_tokens += n
                total_docs += 1
                token_counts.append(n)

                if total_docs % 1_000_000 == 0:
                    print(f"  processed {total_docs:,} docs ({total_tokens:,} tokens)...", flush=True)

    elif suffix == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["text"]):
            texts = batch.column("text").to_pylist()
            for text in texts:
                if not text:
                    continue
                text = text.strip()
                if not text:
                    continue
                tokens = tokenizer.encode(text, add_special_tokens=False)
                n = len(tokens)
                total_tokens += n
                total_docs += 1
                token_counts.append(n)

                if total_docs % 1_000_000 == 0:
                    print(f"  processed {total_docs:,} docs ({total_tokens:,} tokens)...", flush=True)

    elif suffix == ".npy":
        arr = np.load(str(path), mmap_mode="r")
        # In .npy shards, each row is a sequence
        n_seqs = arr.shape[0]
        seq_len = arr.shape[1]
        
        # We can treat each sequence as a "doc"
        total_docs += n_seqs
        total_tokens += arr.size
        # For stats, every doc has exactly seq_len tokens
        token_counts.extend([seq_len] * n_seqs)

    else:
        print(f"Unsupported file type: {suffix}")
        return None

    if not token_counts:
        return None

    token_counts.sort()
    n = len(token_counts)
    avg = total_tokens / n
    median = token_counts[n // 2]

    return {
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "avg_tokens_per_doc": round(avg, 1),
        "median_tokens_per_doc": median,
        "min_tokens_per_doc": token_counts[0],
        "max_tokens_per_doc": token_counts[-1],
        "p10": token_counts[int(n * 0.10)],
        "p25": token_counts[int(n * 0.25)],
        "p75": token_counts[int(n * 0.75)],
        "p90": token_counts[int(n * 0.90)],
    }


def main():
    parser = argparse.ArgumentParser(description="Count tokens in JSONL, Parquet, or NPY files.")
    parser.add_argument("--tokenizer", required=True, help="Path to HF tokenizer.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL/Parquet/NPY file or directory.")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    print(f"Tokenizer loaded (vocab={tokenizer.vocab_size})")

    path = Path(args.dataset)
    if path.is_dir():
        files = []
        for ext in ["*.jsonl", "*.parquet", "*.npy"]:
            files.extend(sorted(path.rglob(ext)))
    elif path.is_file():
        files = [path]
    else:
        print(f"Dataset path not found: {path}")
        sys.exit(1)

    if not files:
        print(f"No supported files found in {path}")
        sys.exit(0)

    grand_total_tokens = 0
    grand_total_docs = 0

    for f in files:
        size_gb = f.stat().st_size / (1024 ** 3)
        print(f"\n{f.name} ({size_gb:.2f} GB)...")

        stats = count_tokens(f, tokenizer)
        if not stats:
            print("  No docs found.")
            continue
            
        grand_total_tokens += stats["total_tokens"]
        grand_total_docs += stats["total_docs"]

        print(f"  Total tokens:      {stats['total_tokens']:>15,}")
        print(f"  Total docs:        {stats['total_docs']:>15,}")
        print(f"  Avg tokens/doc:    {stats['avg_tokens_per_doc']:>15}")
        print(f"  Median tokens/doc: {stats['median_tokens_per_doc']:>15,}")
        print(f"  Min / Max:         {stats['min_tokens_per_doc']:,} / {stats['max_tokens_per_doc']:,}")
        print(f"  P10 / P90:         {stats['p10']:,} / {stats['p90']:,}")
        print(f"  ~{stats['total_tokens'] / 1e9:.2f}B tokens")

    print(f"\n{'='*50}")
    print(f"GRAND TOTALS")
    print(f"{'='*50}")
    print(f"  Total Files:  {len(files):,}")
    print(f"  Total Docs:   {grand_total_docs:,}")
    print(f"  Total Tokens: {grand_total_tokens:,} (~{grand_total_tokens / 1e9:.2f}B)")


if __name__ == "__main__":
    main()

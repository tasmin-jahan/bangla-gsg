"""
Count tokens in JSONL, Parquet, or NPY files.

Reports total tokens, tokens per doc/sequence stats, and overall token count.
Writes a YAML report to saved/reports/.

Usage:
  python utils/count_tokens.py saved/data/cleaned/bangla.jsonl
  python utils/count_tokens.py saved/data/pretokenized/bangla/train/ --format npy
  python utils/count_tokens.py saved/data/cleaned/ --format parquet

  python utils/count_tokens.py saved/data/train --format npy
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml
import pyarrow.parquet as pq


def load_tokenizer(path: str):
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(path)


def get_percentiles(counts: list[int], sample_size: int | None = None) -> dict:
    if not counts:
        return {}
    if sample_size and sample_size < len(counts):
        counts = random.sample(counts, sample_size)
    counts.sort()
    n = len(counts)
    avg = sum(counts) / n
    return {
        "avg_tokens": round(avg, 1),
        "median_tokens": counts[n // 2],
        "min_tokens": counts[0],
        "max_tokens": counts[-1],
        "p10": counts[int(n * 0.10)],
        "p25": counts[int(n * 0.25)],
        "p75": counts[int(n * 0.75)],
        "p90": counts[int(n * 0.90)],
    }


def count_tokens_jsonl(path: Path, tokenizer, sample_size: int | None = None) -> dict:
    total_tokens = 0
    total_docs = 0
    token_counts = []
    with open(path) as f:
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
    
    stats = {"total_tokens": total_tokens, "total_docs": total_docs}
    stats.update(get_percentiles(token_counts, sample_size))
    return stats


def count_tokens_parquet(path: Path, tokenizer, sample_size: int | None = None) -> dict:
    total_tokens = 0
    total_docs = 0
    token_counts = []
    table = pq.read_table(path, columns=["text"])
    for batch in table.to_batches():
        for text_item in batch["text"]:
            text = str(text_item.as_py()).strip()
            if not text:
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False)
            n = len(tokens)
            total_tokens += n
            total_docs += 1
            token_counts.append(n)
    
    stats = {"total_tokens": total_tokens, "total_docs": total_docs}
    stats.update(get_percentiles(token_counts, sample_size))
    return stats


def count_tokens_npy(path: Path) -> dict:
    arr = np.load(path, mmap_mode="r")
    total_tokens = int(arr.size)
    total_seqs = int(arr.shape[0])
    seq_len = int(arr.shape[1]) if len(arr.shape) > 1 else 1
    return {
        "total_tokens": total_tokens,
        "total_seqs": total_seqs,
        "seq_len": seq_len
    }


def save_yaml_report(report_data: dict, output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_dir / f"token_count_{name}_{timestamp}.yaml"
    with open(file_path, "w") as f:
        yaml.dump(report_data, f, default_flow_style=False, sort_keys=False)
    return file_path


def main():
    parser = argparse.ArgumentParser(description="Count tokens in datasets (JSONL, Parquet, NPY).")
    parser.add_argument("paths", nargs="+", help="Files or directories to process.")
    parser.add_argument("--tokenizer", default="saved/tokenizer/hf/", help="Path to HF tokenizer.")
    parser.add_argument("--sample", type=int, default=None, help="Sample N docs for stats.")
    parser.add_argument("--format", choices=["jsonl", "parquet", "npy"], default="jsonl", help="File format to search for if path is a directory (default: jsonl).")
    args = parser.parse_args()

    # We only need tokenizer for jsonl and parquet
    tokenizer = None
    if args.format in ["jsonl", "parquet"]:
        tokenizer = load_tokenizer(args.tokenizer)
        print(f"Tokenizer loaded (vocab={tokenizer.vocab_size})")

    report_dir = Path("saved/reports")
    overall_report = {
        "generated_at": datetime.now().isoformat(),
        "format": args.format,
        "files": {},
        "grand_total_tokens": 0,
    }
    
    if args.format in ["jsonl", "parquet"]:
        overall_report["grand_total_docs"] = 0
    else:
        overall_report["grand_total_seqs"] = 0

    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            files = sorted(path.glob(f"*.{args.format}"))
        elif path.is_file():
            files = [path]
        else:
            print(f"Skipping: {path}")
            continue

        for f in files:
            size_gb = f.stat().st_size / (1024 ** 3)
            print(f"\n{f.name} ({size_gb:.2f} GB)...")

            if args.format == "jsonl":
                stats = count_tokens_jsonl(f, tokenizer, args.sample)
            elif args.format == "parquet":
                stats = count_tokens_parquet(f, tokenizer, args.sample)
            elif args.format == "npy":
                stats = count_tokens_npy(f)

            if not stats or stats.get("total_tokens", 0) == 0:
                print("  No tokens found.")
                continue

            overall_report["files"][str(f)] = stats
            overall_report["grand_total_tokens"] += stats["total_tokens"]
            
            print(f"  Total tokens:      {stats['total_tokens']:>15,}")
            if args.format in ["jsonl", "parquet"]:
                overall_report["grand_total_docs"] += stats["total_docs"]
                print(f"  Total docs:        {stats['total_docs']:>15,}")
                print(f"  Avg tokens/doc:    {stats.get('avg_tokens'):>15}")
            elif args.format == "npy":
                overall_report["grand_total_seqs"] += stats["total_seqs"]
                print(f"  Total sequences:   {stats['total_seqs']:>15,}")
                print(f"  Sequence length:   {stats['seq_len']:>15,}")

    print(f"\n{'='*40}")
    print(f"Grand Total Tokens: {overall_report['grand_total_tokens']:,} (~{overall_report['grand_total_tokens']/1e9:.2f}B)")
    if "grand_total_docs" in overall_report:
        print(f"Grand Total Docs:   {overall_report['grand_total_docs']:,}")
    if "grand_total_seqs" in overall_report:
        print(f"Grand Total Seqs:   {overall_report['grand_total_seqs']:,}")

    if overall_report["files"]:
        p0 = Path(args.paths[0])
        if p0.is_dir() and p0.name == "train" and p0.parent.name:
            report_name = f"{p0.parent.name}_{p0.name}"
        else:
            report_name = p0.stem if p0.is_file() else p0.name
        
        saved_path = save_yaml_report(overall_report, report_dir, report_name)
        print(f"\nReport written to: {saved_path}")


if __name__ == "__main__":
    main()

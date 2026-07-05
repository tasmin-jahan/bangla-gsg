"""
Pretokenize & Pack — per source type.

Tokenizes every document, packs into 2048-token sequences, writes .npy shards.
Output directories:
  saved/data/pretokenized/bangla/train/
  saved/data/pretokenized/english/train/
  saved/data/pretokenized/nmt/train/
  saved/data/pretokenized/sangraha/train/

No language token injection — tokens are already in text from downloaders.
No eval split — everything goes to train.

Usage:
  python scripts/pretokenize_and_pack.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


DATA_DIR = Path("saved/data")
HF_TOKENIZER_DIR = Path("saved/tokenizer/hf")
PRETOKENIZED_DIR = Path("saved/data/pretokenized")

SEQ_LEN = 2048
BATCH_TOKENS = SEQ_LEN * 100_000  # 204.8M tokens per shard

# Source type → input directory + output directory
SOURCE_CONFIGS = {
    "bangla": {
        "input_dir": DATA_DIR / "bangla_corpus",
        "output": PRETOKENIZED_DIR / "bangla" / "train",
    },
    "english": {
        "input_dir": DATA_DIR / "fineweb_edu",
        "output": PRETOKENIZED_DIR / "english" / "train",
    },
    "nmt": {
        "input_dir": DATA_DIR / "nllb_nmt",
        "output": PRETOKENIZED_DIR / "nmt" / "train",
    },
    "sangraha": {
        "input_dir": DATA_DIR / "sangraha",
        "output": PRETOKENIZED_DIR / "sangraha" / "train",
    },
}


def load_tokenizer():
    from transformers import PreTrainedTokenizerFast

    for t_dir in [HF_TOKENIZER_DIR, Path("saved/tokenizer")]:
        if t_dir.exists():
            return PreTrainedTokenizerFast.from_pretrained(str(t_dir))

    print(f"[pretokenize] ERROR: HF tokenizer not found at {HF_TOKENIZER_DIR} or saved/tokenizer")
    sys.exit(1)


def save_shard(token_ids: list[int], shard_idx: int, output_dir: Path) -> int:
    usable = len(token_ids) - (len(token_ids) % SEQ_LEN)
    if usable == 0:
        return 0
    arr = np.array(token_ids[:usable], dtype=np.uint16).reshape(-1, SEQ_LEN)
    shard_path = output_dir / f"shard_{shard_idx:05d}.npy"
    tmp_path = shard_path.with_suffix(".npy.tmp")
    np.save(tmp_path, arr)
    tmp_path.replace(shard_path)
    return arr.shape[0]


def _count_rows(path: Path) -> int:
    return pq.read_metadata(path).num_rows


def _resolve_inputs(config: dict) -> list[Path]:
    """Return existing parquet files in the input directory."""
    input_dir = config["input_dir"]
    if not input_dir.exists():
        return []
    return sorted(input_dir.glob("*.parquet"))


def pretokenize_source(
    source_type: str,
    config: dict,
    tokenizer,
) -> tuple[int, int]:
    """Pretokenize one source type. Returns (tokens_written, docs_processed)."""
    output_dir = config["output"]
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = _resolve_inputs(config)
    if not inputs:
        print(f"[pretokenize] WARNING: No input files for {source_type}, skipping")
        return 0, 0

    # Count total rows
    total_rows = sum(_count_rows(p) for p in inputs)

    eos_id = tokenizer.eos_token_id
    buffer = []
    shard_idx = 0
    total_tokens = 0
    total_docs = 0

    print(f"[pretokenize] {source_type}: {len(inputs)} input file(s), {total_rows:,} rows")

    with tqdm(total=total_rows, desc=f"  {source_type}", unit="rows", unit_scale=True) as bar:
        for input_path in inputs:
            pf = pq.ParquetFile(input_path)
            for batch in pf.iter_batches(batch_size=10000, columns=["text"]):
                texts = batch.column("text").to_pylist()
                for text in texts:
                    if not text:
                        bar.update(1)
                        continue
                    text = text.strip()
                    if not text:
                        bar.update(1)
                        continue

                    # Tokenize — text already has special tokens from downloaders
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    tokens = tokens + [eos_id]

                    buffer.extend(tokens)
                    total_tokens += len(tokens)
                    total_docs += 1
                    bar.update(1)
                    bar.set_postfix(kept=total_docs, refresh=False)

                    # Flush when buffer is large enough
                    while len(buffer) >= BATCH_TOKENS:
                        chunk = buffer[:BATCH_TOKENS]
                        buffer = buffer[BATCH_TOKENS:]
                        save_shard(chunk, shard_idx, output_dir)
                        shard_idx += 1

    # Save remaining buffer
    if buffer:
        remainder = len(buffer) % SEQ_LEN
        if remainder:
            print(f"  [pretokenize] Truncating final buffer: discarding {remainder} tokens")
        save_shard(buffer, shard_idx, output_dir)
        shard_idx += 1

    return total_tokens, total_docs


def main():
    parser = argparse.ArgumentParser(description="Pretokenize and pack into .npy shards.")
    parser.add_argument("--source", choices=["bangla", "english", "nmt", "sangraha", "all"], default="all",
                        help="Which source type to pretokenize (default: all).")
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    print(f"[pretokenize] Tokenizer loaded (vocab={tokenizer.vocab_size})")

    sources = list(SOURCE_CONFIGS.keys()) if args.source == "all" else [args.source]

    grand_tokens = 0
    grand_docs = 0

    for source_type in sources:
        config = SOURCE_CONFIGS[source_type]
        tokens, docs = pretokenize_source(source_type, config, tokenizer)
        grand_tokens += tokens
        grand_docs += docs

    # Calculate stats
    total_shards = 0
    for source_type in sources:
        output_dir = SOURCE_CONFIGS[source_type]["output"]
        if output_dir.exists():
            shards = list(output_dir.glob("shard_*.npy"))
            total_shards += len(shards)

    tokens_per_step = 4 * 64 * SEQ_LEN
    approx_steps = grand_tokens // tokens_per_step

    print(f"\n{'=' * 50}")
    print(f"=== PRETOKENIZATION COMPLETE ===")
    print(f"  Documents:       {grand_docs:,}")
    print(f"  Tokens:          {grand_tokens:,}  ({grand_tokens / 1e9:.2f}B)")
    print(f"  Total shards:    {total_shards}")
    for source_type in sources:
        output_dir = SOURCE_CONFIGS[source_type]["output"]
        n = len(list(output_dir.glob("shard_*.npy"))) if output_dir.exists() else 0
        print(f"    {source_type:>10}: {n:>4} shards  →  {output_dir}")
    print(f"  Approx steps:    {approx_steps:,}")
    print(f"    (batch=4 × accum=64 × seq=2048 = {tokens_per_step:,} tokens/step)")
    print(f"\n  ACTION: set max_steps: {approx_steps} in configs/default_training.yaml")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()

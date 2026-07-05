import argparse
import json
import random
from pathlib import Path
import pyarrow.parquet as pq
from tqdm import tqdm

DATA_DIR = Path("saved/data")
BANGLA_DIR = DATA_DIR / "bangla_corpus"
ENGLISH_DIR = DATA_DIR / "fineweb_edu"
OUTPUT_DIR = DATA_DIR / "tokenizer_corpus"

def sample_parquet(input_dir, num_samples):
    paths = sorted(input_dir.glob("*.parquet"))
    if not paths:
        print(f"No parquet files found in {input_dir}")
        return []
    
    samples = []
    
    with tqdm(total=num_samples, desc=f"Sampling {input_dir.name}") as pbar:
        for p in paths:
            pf = pq.ParquetFile(p)
            for batch in pf.iter_batches(columns=["text"]):
                texts = batch.column("text").to_pylist()
                for text in texts:
                    if text and text.strip():
                        samples.append(text.strip())
                        pbar.update(1)
                        if len(samples) >= num_samples:
                            return samples
    return samples

def main():
    parser = argparse.ArgumentParser(description="Sample corpus for tokenizer training.")
    parser.add_argument("--bangla-samples", type=int, default=1_500_000)
    parser.add_argument("--english-samples", type=int, default=500_000)
    parser.add_argument("--output", type=str, default="saved/data/tokenizer_corpus/corpus.jsonl")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bangla_texts = sample_parquet(BANGLA_DIR, args.bangla_samples)
    english_texts = sample_parquet(ENGLISH_DIR, args.english_samples)

    all_texts = bangla_texts + english_texts
    random.shuffle(all_texts)

    out_path = Path(args.output)
    print(f"Writing output to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for text in tqdm(all_texts, desc="Writing JSONL"):
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved {len(all_texts)} samples to {out_path}")

if __name__ == "__main__":
    main()

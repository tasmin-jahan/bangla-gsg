"""
Bangla NIAH — Retrieval Evaluation.

Evaluates whether models can retrieve a "needle" (factoid) from a
"haystack" (Wikipedia context) at different context lengths and depths.

Context lengths: {256, 512, 1024, 1536, 2048} — hard cap at 2048.
Needle depths: {0.1, 0.3, 0.5, 0.7, 0.9}
20 samples per cell = 500 total per model.

Only gamba/gsg — BanglaBERT is excluded (not generative).

Usage:
    # Build dataset first (only needed once)
    python -m evaluation_suit.eval.05_long_context.build_niah

    # Run evaluation
    python -m evaluation_suit.eval.05_long_context.run --model gamba
    python -m evaluation_suit.eval.05_long_context.run --model gsg
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from evaluation_suit.eval.common.model_registry import load_model
from evaluation_suit.eval.common.seeding import set_seed
from evaluation_suit.eval.common.io_utils import read_jsonl, write_json, write_jsonl


def check_retrieval(generated: str, expected_answer: str) -> bool:
    """Check if the generated text contains the expected answer."""
    generated_lower = generated.lower().strip()
    expected_lower = expected_answer.lower().strip()
    return expected_lower in generated_lower


def run_niah_eval(
    model_key: str,
    max_new_tokens: int = 64,
    niah_data_dir: str = "evaluation_suit/results/05_long_context/niah_data",
    results_dir: str = "evaluation_suit/results/05_long_context",
) -> dict:
    """
    Run NIAH evaluation for a single model.

    Args:
        model_key: "gamba" or "gsg" (NOT banglabert).
    """
    if model_key == "banglabert":
        print("[05_niah] BanglaBERT excluded (not generative).")
        return {"model": model_key, "skipped": True}

    # Load NIAH samples
    samples_path = f"{niah_data_dir}/niah_samples.jsonl"
    samples = read_jsonl(samples_path)
    if not samples:
        print(f"[05_niah] No NIAH samples found at {samples_path}.")
        print("  Run build_niah.py first:")
        print("    python -m evaluation_suit.eval.05_long_context.build_niah")
        sys.exit(1)

    print(f"[05_niah] Loaded {len(samples)} NIAH samples for {model_key}")

    set_seed(42)
    loaded = load_model(model_key)
    device = loaded.device

    # Run inference
    results = []
    for sample in tqdm(samples, desc=f"[{model_key}] NIAH"):
        prompt = sample["prompt"]
        expected = sample["expected_answer"]

        inputs = loaded.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048,
        )
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                gen_model = loaded.model
                outputs = gen_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=loaded.tokenizer.eos_token_id,
                    pad_token_id=loaded.tokenizer.pad_token_id,
                )

        generated_ids = outputs[0][input_ids.shape[1]:]
        generated = loaded.tokenizer.decode(generated_ids, skip_special_tokens=True)

        correct = check_retrieval(generated, expected)

        results.append({
            "id": sample["id"],
            "context_length": sample["context_length"],
            "needle_depth": sample["needle_depth"],
            "expected_answer": expected,
            "generated": generated[:200],  # truncate for storage
            "correct": correct,
        })

    # Save raw results
    raw_path = f"{results_dir}/raw_{model_key}.jsonl"
    write_jsonl(raw_path, results)

    # Compute accuracy heatmap
    heatmap = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cell_key = f"{r['context_length']}_{r['needle_depth']}"
        heatmap[cell_key]["total"] += 1
        if r["correct"]:
            heatmap[cell_key]["correct"] += 1

    heatmap_data = {}
    for key, counts in heatmap.items():
        ctx_len, depth = key.split("_")
        acc = counts["correct"] / max(counts["total"], 1)
        if ctx_len not in heatmap_data:
            heatmap_data[ctx_len] = {}
        heatmap_data[ctx_len][depth] = {
            "accuracy": round(acc, 4),
            "correct": counts["correct"],
            "total": counts["total"],
        }

    # Overall accuracy
    total_correct = sum(1 for r in results if r["correct"])
    total = len(results)
    overall_acc = total_correct / max(total, 1)

    summary = {
        "model": model_key,
        "task": "05_long_context",
        "overall_accuracy": round(overall_acc, 4),
        "total_correct": total_correct,
        "total_samples": total,
        "heatmap": heatmap_data,
    }

    write_json(f"{results_dir}/summary_{model_key}.json", summary)
    print(f"\n[{model_key}] Overall NIAH accuracy: {overall_acc:.4f} ({total_correct}/{total})")
    print(f"  Results saved to {results_dir}/")

    del loaded
    torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Bangla NIAH Evaluation")
    parser.add_argument("--model", type=str, required=True, choices=["gamba", "gsg"])
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    run_niah_eval(model_key=args.model, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()

"""
XL-Sum Bangla Summarization — Generation & Scoring.

Prompt-based abstractive summarization on the XL-Sum Bengali test split.
Greedy decoding.

Metrics: ROUGE-L (required), BERTScore (attempt with try/except fallback).

Only gamba/gsg — BanglaBERT excluded (not generative).

Usage:
    python -m evaluation_suit.eval.06_summarization.generate --model gamba
    python -m evaluation_suit.eval.06_summarization.generate --model gsg
"""

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from evaluation_suit.eval.common.model_registry import load_model
from evaluation_suit.eval.common.seeding import set_seed
from evaluation_suit.eval.common.io_utils import append_result, write_json
from evaluation_suit.eval.common.metrics import compute_rouge, compute_bertscore


# ── Prompt Template ──────────────────────────────────────────────────────────

PROMPT_TEMPLATE = "নিম্নলিখিত লেখাটির সংক্ষিপ্তসার লিখুন:\n\n{article}\n\nসংক্ষিপ্তসার:"


def run_summarization_eval(
    model_key: str,
    max_new_tokens: int = 256,
    max_src_len: int = 1024,
    max_examples: int = None,
    results_dir: str = "evaluation_suit/results/06_summarization",
) -> dict:
    """
    Run summarization evaluation for a single model on XL-Sum Bengali.

    Args:
        model_key: "gamba" or "gsg" (NOT banglabert).
        max_new_tokens: Max tokens to generate for summary.
        max_src_len: Max source article tokens (truncated).
        max_examples: Limit examples for testing. None = full test set.
    """
    if model_key == "banglabert":
        print("[06_summarization] BanglaBERT excluded (not generative).")
        return {"model": model_key, "skipped": True}

    set_seed(42)

    # Load model
    loaded = load_model(model_key)
    device = loaded.device

    # Load XL-Sum Bengali test split
    print("[06_summarization] Loading XL-Sum Bengali...")
    ds = load_dataset("csebuetnlp/xlsum", "bengali", split="test")

    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    print(f"[06_summarization] Evaluating {len(ds)} examples for {model_key}...")

    references = []
    hypotheses = []

    for example in tqdm(ds, desc=f"[{model_key}] Summarization"):
        article = example["text"]
        reference = example["summary"]

        # Truncate article to fit within max_src_len
        article_tokens = loaded.tokenizer.encode(article, add_special_tokens=False)
        if len(article_tokens) > max_src_len:
            article_tokens = article_tokens[:max_src_len]
            article = loaded.tokenizer.decode(article_tokens, skip_special_tokens=True)

        prompt = PROMPT_TEMPLATE.format(article=article)

        inputs = loaded.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = loaded.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=loaded.tokenizer.eos_token_id,
                    pad_token_id=loaded.tokenizer.pad_token_id,
                )

        generated_ids = outputs[0][input_ids.shape[1]:]
        summary = loaded.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        references.append(reference)
        hypotheses.append(summary)

    # Score: ROUGE (required)
    print(f"\n[{model_key}] Computing ROUGE...")
    rouge_scores = compute_rouge(references, hypotheses)
    rouge_l = rouge_scores.get("rougeL", 0.0)

    print(f"  ROUGE-L: {rouge_l:.4f}")
    for k, v in rouge_scores.items():
        print(f"  {k}: {v:.4f}")

    # Score: BERTScore (optional, with fallback)
    print(f"[{model_key}] Computing BERTScore...")
    bert_scores = compute_bertscore(references, hypotheses, lang="bn")
    if bert_scores:
        print(f"  BERTScore F1: {bert_scores['f1']:.4f}")
    else:
        print("  BERTScore: FAILED (falling back to ROUGE-L only)")

    # Save results
    result = {
        "model": model_key,
        "task": "06_summarization",
        "seed": 42,
        "dataset": "xlsum_bengali",
        "n_examples": len(references),
        "rouge_l": round(rouge_l, 4),
        **{f"rouge_{k}": round(v, 4) for k, v in rouge_scores.items()},
    }
    if bert_scores:
        result["bertscore_f1"] = round(bert_scores["f1"], 4)
        result["bertscore_precision"] = round(bert_scores["precision"], 4)
        result["bertscore_recall"] = round(bert_scores["recall"], 4)

    append_result(f"{results_dir}/seeds.jsonl", result)

    # Save detailed scores + samples
    write_json(f"{results_dir}/scores_{model_key}.json", {
        **result,
        "sample_outputs": [
            {"reference": references[i], "hypothesis": hypotheses[i]}
            for i in range(min(10, len(references)))
        ],
    })

    print(f"  Results saved to {results_dir}/")

    del loaded
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="XL-Sum Bangla Summarization")
    parser.add_argument("--model", type=str, required=True, choices=["gamba", "gsg"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_src_len", type=int, default=1024)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    run_summarization_eval(
        model_key=args.model,
        max_new_tokens=args.max_new_tokens,
        max_src_len=args.max_src_len,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()

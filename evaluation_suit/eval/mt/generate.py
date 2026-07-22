"""
FLORES-200 Machine Translation — Generation & Scoring.

Prompt-based translation evaluation for bn→en and en→bn directions.
Greedy decoding (do_sample=False).

IMPORTANT: Run check_contamination.py FIRST. This script checks for a
contamination report and refuses to run if contamination is unresolved.

BanglaBERT is NOT applicable (not generative) — only gamba and gsg.

Usage:
    python -m evaluation_suit.eval.04_mt.generate --model gamba
    python -m evaluation_suit.eval.04_mt.generate --model gsg
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from evaluation_suit.eval.common.model_registry import load_model
from evaluation_suit.eval.common.seeding import set_seed
from evaluation_suit.eval.common.io_utils import append_result, read_json, write_json
from evaluation_suit.eval.common.metrics import compute_bleu, compute_chrf


# ── Prompt Templates ─────────────────────────────────────────────────────────

PROMPT_BN_TO_EN = "নিম্নলিখিত বাংলা বাক্যটি ইংরেজিতে অনুবাদ করুন:\n\nবাংলা: {source}\nইংরেজি:"

PROMPT_EN_TO_BN = "Translate the following English sentence to Bangla:\n\nEnglish: {source}\nBangla:"


def generate_translation(
    model,
    tokenizer,
    source_text: str,
    prompt_template: str,
    max_new_tokens: int = 128,
    device: torch.device = None,
) -> str:
    """Generate a translation using greedy decoding."""
    prompt = prompt_template.format(source=source_text)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device or model.device)

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda" if device else False):
            # Access the inner model that has .generate()
            gen_model = model
            if hasattr(model, "generate"):
                gen_model = model
            elif hasattr(model, "model") and hasattr(model.model, "generate"):
                gen_model = model.model

            outputs = gen_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

    # Decode only the generated tokens (not the prompt)
    generated_ids = outputs[0][input_ids.shape[1]:]
    translation = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return translation


def run_mt_eval(
    model_key: str,
    max_new_tokens: int = 128,
    max_examples: int = None,
    results_dir: str = "evaluation_suit/results/04_mt",
) -> dict:
    """
    Run MT evaluation for a single model on FLORES-200.

    Args:
        model_key: "gamba" or "gsg" (NOT banglabert).
        max_new_tokens: Max tokens to generate per translation.
        max_examples: Limit examples (for testing). None = all.
        results_dir: Where to save results.
    """
    if model_key == "banglabert":
        print("[04_mt] BanglaBERT is NOT applicable (not generative). Skipping.")
        return {"model": model_key, "skipped": True, "reason": "Not a generative model"}

    # Check contamination report
    report_path = f"{results_dir}/contamination_report.json"
    report = read_json(report_path)
    if report is None:
        print(
            "⚠ ERROR: No contamination report found. "
            "Run check_contamination.py FIRST.\n"
            "  python -m evaluation_suit.eval.04_mt.check_contamination"
        )
        sys.exit(1)

    if not report.get("proceed", False):
        print(
            f"⚠ MT eval is GATED by contamination check.\n"
            f"  Reason: {report.get('reason', 'Unknown')}\n"
            f"  Resolve contamination before running MT eval."
        )
        sys.exit(1)

    overlap_hashes = set()
    for sample in report.get("overlap_samples", []):
        from hashlib import sha256
        text = sample.get("flores_text", "")
        if text:
            overlap_hashes.add(sha256(text.strip().lower().encode("utf-8")).hexdigest())

    set_seed(42)

    # Load model
    loaded = load_model(model_key)
    device = loaded.device

    # Load FLORES
    print("[04_mt] Loading FLORES-200 devtest...")
    try:
        flores = load_dataset("facebook/flores", "ben_Beng-eng_Latn", split="devtest")
        bn_key = "sentence_ben_Beng"
        en_key = "sentence_eng_Latn"
    except Exception:
        try:
            flores = load_dataset("openlanguagedata/flores_plus", "ben_Beng-eng_Latn", split="devtest")
            bn_key = "sentence_ben_Beng"
            en_key = "sentence_eng_Latn"
        except Exception:
            # Try separate configs
            flores_bn = load_dataset("facebook/flores", "ben_Beng", split="devtest")
            flores_en = load_dataset("facebook/flores", "eng_Latn", split="devtest")
            # Create paired data
            flores_data = [
                {"sentence_ben_Beng": bn["sentence"], "sentence_eng_Latn": en["sentence"]}
                for bn, en in zip(flores_bn, flores_en)
            ]
            from datasets import Dataset
            flores = Dataset.from_list(flores_data)
            bn_key = "sentence_ben_Beng"
            en_key = "sentence_eng_Latn"

    if max_examples:
        flores = flores.select(range(min(max_examples, len(flores))))

    print(f"[04_mt] Evaluating {len(flores)} examples for {model_key}...")

    results = {}

    for direction, src_key, tgt_key, prompt_tmpl in [
        ("bn_to_en", bn_key, en_key, PROMPT_BN_TO_EN),
        ("en_to_bn", en_key, bn_key, PROMPT_EN_TO_BN),
    ]:
        print(f"\n[04_mt] Direction: {direction}")
        references = []
        hypotheses = []

        for ex in tqdm(flores, desc=direction):
            source = ex[src_key]
            reference = ex[tgt_key]

            translation = generate_translation(
                loaded.model, loaded.tokenizer,
                source, prompt_tmpl,
                max_new_tokens=max_new_tokens,
                device=device,
            )

            references.append(reference)
            hypotheses.append(translation)

        # Score
        bleu = compute_bleu(references, hypotheses)
        chrf = compute_chrf(references, hypotheses)

        print(f"  {direction}: BLEU={bleu:.2f}, chrF={chrf:.2f}")

        results[direction] = {
            "bleu": round(bleu, 2),
            "chrf": round(chrf, 2),
            "n_examples": len(references),
        }

        # Save individual results
        scores_path = f"{results_dir}/scores_{model_key}_{direction}.json"
        write_json(scores_path, {
            "model": model_key,
            "direction": direction,
            "bleu": round(bleu, 2),
            "chrf": round(chrf, 2),
            "n_examples": len(references),
            "sample_translations": [
                {"source": references[i], "hypothesis": hypotheses[i]}
                for i in range(min(5, len(references)))
            ],
        })

    # Combined result
    combined = {
        "model": model_key,
        "task": "04_mt",
        "seed": 42,
        **{f"{d}_{m}": results[d][m] for d in results for m in ["bleu", "chrf"]},
    }
    append_result(f"{results_dir}/seeds.jsonl", combined)
    print(f"\n  Results saved to {results_dir}/")

    del loaded
    torch.cuda.empty_cache()

    return combined


def main():
    parser = argparse.ArgumentParser(description="FLORES-200 MT Evaluation")
    parser.add_argument("--model", type=str, required=True, choices=["gamba", "gsg"])
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Limit examples for testing")
    args = parser.parse_args()

    run_mt_eval(
        model_key=args.model,
        max_new_tokens=args.max_new_tokens,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()

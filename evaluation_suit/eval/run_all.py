"""
Eval Suite Orchestrator — runs tasks 01→06 in order.

Resumable: checks existing results and skips completed (model, seed) pairs.
Writes results/manifest.json summarizing what ran and what was skipped.

Usage:
    # Run everything
    python -m evaluation_suit.eval.run_all

    # Run specific models only
    python -m evaluation_suit.eval.run_all --models gamba gsg

    # Run specific tasks only
    python -m evaluation_suit.eval.run_all --tasks 01 02 03

    # Dry run (show what would run without running)
    python -m evaluation_suit.eval.run_all --dry-run
"""

import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation_suit.eval.common.io_utils import write_json, read_json


SEEDS = [0, 1, 2]
ALL_MODELS = ["gamba", "gsg", "banglabert"]
GENERATIVE_MODELS = ["gamba", "gsg"]


def run_task_01(models, seeds, dry_run=False, save_checkpoints=False):
    """01_sentiment — SentNoB"""
    from evaluation_suit.eval.sentiment.run import train_and_evaluate

    results = []
    for model in models:
        for seed in seeds:
            desc = f"sentiment/{model}/seed={seed}"
            if dry_run:
                print(f"  [DRY RUN] Would run: {desc}")
                continue
            print(f"\n{'='*60}\n  Running: {desc}\n{'='*60}")
            try:
                r = train_and_evaluate(model_key=model, seed=seed, save_checkpoint=save_checkpoints)
                results.append({"task": "sentiment", "model": model, "seed": seed,
                                "status": "completed" if r else "skipped"})
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                traceback.print_exc()
                results.append({"task": "sentiment", "model": model, "seed": seed,
                                "status": "failed", "error": str(e)})
    return results


def run_task_02(models, seeds, dry_run=False, save_checkpoints=False):
    """02_ner — ANCHOLIK + WikiAnn"""
    from evaluation_suit.eval.ner.run import train_and_evaluate

    results = []
    for dataset in ["ancholik", "wikiann"]:
        for model in models:
            for seed in seeds:
                desc = f"ner/{model}/{dataset}/seed={seed}"
                if dry_run:
                    print(f"  [DRY RUN] Would run: {desc}")
                    continue
                print(f"\n{'='*60}\n  Running: {desc}\n{'='*60}")
                try:
                    r = train_and_evaluate(model_key=model, dataset_name=dataset, seed=seed, save_checkpoint=save_checkpoints)
                    results.append({"task": "ner", "model": model, "dataset": dataset,
                                    "seed": seed, "status": "completed" if r else "skipped"})
                except Exception as e:
                    print(f"  ✗ FAILED: {e}")
                    traceback.print_exc()
                    results.append({"task": "ner", "model": model, "dataset": dataset,
                                    "seed": seed, "status": "failed", "error": str(e)})
    return results


def run_task_03(models, seeds, dry_run=False, save_checkpoints=False):
    """03_nli — XNLI + BanglaParaphrase"""
    from evaluation_suit.eval.nli.run import train_and_evaluate

    results = []
    for dataset in ["xnli", "paraphrase"]:
        for model in models:
            for seed in seeds:
                desc = f"nli/{model}/{dataset}/seed={seed}"
                if dry_run:
                    print(f"  [DRY RUN] Would run: {desc}")
                    continue
                print(f"\n{'='*60}\n  Running: {desc}\n{'='*60}")
                try:
                    r = train_and_evaluate(model_key=model, dataset_name=dataset, seed=seed, save_checkpoint=save_checkpoints)
                    results.append({"task": "nli", "model": model, "dataset": dataset,
                                    "seed": seed, "status": "completed" if r else "skipped"})
                except Exception as e:
                    print(f"  ✗ FAILED: {e}")
                    traceback.print_exc()
                    results.append({"task": "nli", "model": model, "dataset": dataset,
                                    "seed": seed, "status": "failed", "error": str(e)})
    return results


def run_task_04(models, dry_run=False):
    """04_mt — FLORES-200 (contamination check + generation)"""
    from evaluation_suit.eval.mt.check_contamination import check_contamination

    results = []
    gen_models = [m for m in models if m in GENERATIVE_MODELS]

    if not gen_models:
        print("  [04_mt] No generative models selected. Skipping.")
        return [{"task": "04_mt", "status": "skipped", "reason": "No generative models"}]

    # Step 1: Contamination check
    if dry_run:
        print("  [DRY RUN] Would run: 04_mt/check_contamination")
        for m in gen_models:
            print(f"  [DRY RUN] Would run: 04_mt/{m}/generate")
        return []

    print(f"\n{'='*60}\n  Running: 04_mt/check_contamination\n{'='*60}")
    try:
        report = check_contamination()
        results.append({"task": "04_mt", "step": "contamination_check",
                        "status": "completed", "proceed": report.get("proceed", False)})
    except Exception as e:
        print(f"  ✗ Contamination check FAILED: {e}")
        traceback.print_exc()
        results.append({"task": "04_mt", "step": "contamination_check",
                        "status": "failed", "error": str(e)})
        return results

    if not report.get("proceed", False):
        print("  ⚠ MT eval GATED by contamination. Skipping generation.")
        results.append({"task": "04_mt", "step": "generate",
                        "status": "gated", "reason": report.get("reason", "")})
        return results

    # Step 2: Generation
    from evaluation_suit.eval.mt.generate import run_mt_eval

    for model in gen_models:
        desc = f"04_mt/{model}/generate"
        print(f"\n{'='*60}\n  Running: {desc}\n{'='*60}")
        try:
            r = run_mt_eval(model_key=model)
            results.append({"task": "04_mt", "model": model, "status": "completed"})
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            results.append({"task": "04_mt", "model": model,
                            "status": "failed", "error": str(e)})
    return results


def run_task_05(models, dry_run=False):
    """05_long_context — NIAH"""
    results = []
    gen_models = [m for m in models if m in GENERATIVE_MODELS]

    if not gen_models:
        return [{"task": "05_long_context", "status": "skipped", "reason": "No generative models"}]

    if dry_run:
        print("  [DRY RUN] Would run: 05_long_context/build_niah")
        for m in gen_models:
            print(f"  [DRY RUN] Would run: 05_long_context/{m}/run")
        return []

    # Build NIAH dataset (uses gamba tokenizer)
    from evaluation_suit.eval.long_context.build_niah import build_niah_dataset
    from transformers import AutoTokenizer

    niah_data = "evaluation_suit/results/05_long_context/niah_data/niah_samples.jsonl"
    if not Path(niah_data).exists():
        print(f"\n{'='*60}\n  Building NIAH dataset\n{'='*60}")
        tokenizer = AutoTokenizer.from_pretrained(
            "ahmed-farhanur-rashid/bangla-gamba", trust_remote_code=True,
        )
        build_niah_dataset(tokenizer)

    from evaluation_suit.eval.long_context.run import run_niah_eval

    for model in gen_models:
        desc = f"05_long_context/{model}"
        print(f"\n{'='*60}\n  Running: {desc}\n{'='*60}")
        try:
            r = run_niah_eval(model_key=model)
            results.append({"task": "05_long_context", "model": model, "status": "completed"})
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            results.append({"task": "05_long_context", "model": model,
                            "status": "failed", "error": str(e)})
    return results


def run_task_06(models, dry_run=False):
    """06_summarization — XL-Sum"""
    from evaluation_suit.eval.summarization.generate import run_summarization_eval

    results = []
    gen_models = [m for m in models if m in GENERATIVE_MODELS]

    if not gen_models:
        return [{"task": "06_summarization", "status": "skipped", "reason": "No generative models"}]

    for model in gen_models:
        desc = f"06_summarization/{model}"
        if dry_run:
            print(f"  [DRY RUN] Would run: {desc}")
            continue
        print(f"\n{'='*60}\n  Running: {desc}\n{'='*60}")
        try:
            r = run_summarization_eval(model_key=model)
            results.append({"task": "06_summarization", "model": model, "status": "completed"})
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            results.append({"task": "06_summarization", "model": model,
                            "status": "failed", "error": str(e)})
    return results


TASK_RUNNERS = {
    "01": ("sentiment", run_task_01),
    "02": ("ner", run_task_02),
    "03": ("nli", run_task_03),
    "04": ("mt", run_task_04),
    "05": ("long_context", run_task_05),
    "06": ("summarization", run_task_06),
}


def main():
    parser = argparse.ArgumentParser(description="Run Eval Suite (all tasks)")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS, help="Models to evaluate")
    parser.add_argument("--tasks", nargs="+", default=list(TASK_RUNNERS.keys()),
                        choices=list(TASK_RUNNERS.keys()), help="Tasks to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without running")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Save fine-tuned head weights to disk for HF upload")
    args = parser.parse_args()

    start_time = time.time()
    print(f"{'='*60}")
    print(f"  Bangla LM Eval Suite")
    print(f"  Models: {args.models}")
    print(f"  Tasks: {args.tasks}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Save Checkpoints: {args.save_checkpoints}")
    print(f"{'='*60}\n")

    all_results = []

    for task_key in sorted(args.tasks):
        task_name, runner = TASK_RUNNERS[task_key]
        print(f"\n{'#'*60}")
        print(f"  TASK: {task_name}")
        print(f"{'#'*60}")

        # Tasks 01-03 take models + seeds + save_checkpoints; 04-06 take models only
        if task_key in ("01", "02", "03"):
            results = runner(args.models, args.seeds, args.dry_run, args.save_checkpoints)
        else:
            results = runner(args.models, args.dry_run)

        all_results.extend(results)

    # Write manifest
    elapsed = time.time() - start_time
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "models": args.models,
        "tasks": args.tasks,
        "seeds": args.seeds,
        "dry_run": args.dry_run,
        "task_results": all_results,
        "summary": {
            "completed": sum(1 for r in all_results if r.get("status") == "completed"),
            "skipped": sum(1 for r in all_results if r.get("status") == "skipped"),
            "failed": sum(1 for r in all_results if r.get("status") == "failed"),
            "gated": sum(1 for r in all_results if r.get("status") == "gated"),
        },
    }

    manifest_path = "evaluation_suit/results/manifest.json"
    if not args.dry_run:
        write_json(manifest_path, manifest)
        print(f"\n{'='*60}")
        print(f"  Suite complete in {elapsed:.0f}s")
        print(f"  Manifest: {manifest_path}")
        print(f"  Completed: {manifest['summary']['completed']}")
        print(f"  Skipped: {manifest['summary']['skipped']}")
        print(f"  Failed: {manifest['summary']['failed']}")
        print(f"  Gated: {manifest['summary']['gated']}")
        print(f"{'='*60}")
    else:
        print(f"\n[DRY RUN] Would write manifest to {manifest_path}")


if __name__ == "__main__":
    main()

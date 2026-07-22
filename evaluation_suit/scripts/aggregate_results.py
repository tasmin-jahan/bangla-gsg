"""
Result Aggregation — produces summary tables from all task results.

Reads all results/0N_*/seeds.jsonl and scores.json files, produces:
1. Per-task markdown tables with mean ± std across seeds
2. Combined summary table: 6 tasks × 3 models (N/A for banglabert on 04/05/06)

This combined table is essentially the paper's main results table.

Usage:
    python -m evaluation_suit.scripts.aggregate_results
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation_suit.eval.common.io_utils import read_jsonl, read_json, write_json


RESULTS_DIR = Path("evaluation_suit/results")


def _mean_std(values):
    """Return formatted mean ± std string."""
    if not values:
        return "N/A"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    mean = np.mean(values)
    std = np.std(values)
    return f"{mean:.2f} ± {std:.2f}"


def aggregate_task_01():
    """Aggregate sentiment results."""
    records = read_jsonl(str(RESULTS_DIR / "01_sentiment" / "seeds.jsonl"))
    if not records:
        return None

    by_model = defaultdict(lambda: {"macro_f1": [], "accuracy": []})
    for r in records:
        model = r.get("model", "unknown")
        by_model[model]["macro_f1"].append(r.get("macro_f1", 0))
        by_model[model]["accuracy"].append(r.get("accuracy", 0))

    rows = []
    for model in ["gamba", "gsg", "banglabert"]:
        if model in by_model:
            rows.append({
                "model": model,
                "macro_f1": _mean_std(by_model[model]["macro_f1"]),
                "accuracy": _mean_std(by_model[model]["accuracy"]),
                "n_seeds": len(by_model[model]["macro_f1"]),
            })

    return {"task": "01_sentiment", "dataset": "SentNoB", "metric": "Macro-F1", "rows": rows}


def aggregate_task_02():
    """Aggregate NER results."""
    records = read_jsonl(str(RESULTS_DIR / "02_ner" / "seeds.jsonl"))
    if not records:
        return None

    by_model_ds = defaultdict(lambda: {"entity_f1": []})
    for r in records:
        key = f"{r.get('dataset', 'unknown')}"
        model_raw = r.get("model", "unknown")
        # model field is "model_dataset", split it
        if "_" in model_raw:
            model = model_raw.rsplit("_", 1)[0]
        else:
            model = model_raw
        by_model_ds[(model, key)]["entity_f1"].append(r.get("entity_f1", 0))

    rows = []
    for dataset in ["ancholik", "wikiann"]:
        for model in ["gamba", "gsg", "banglabert"]:
            if (model, dataset) in by_model_ds:
                rows.append({
                    "model": model,
                    "dataset": dataset,
                    "entity_f1": _mean_std(by_model_ds[(model, dataset)]["entity_f1"]),
                    "n_seeds": len(by_model_ds[(model, dataset)]["entity_f1"]),
                })

    return {"task": "02_ner", "metric": "Entity-F1", "rows": rows}


def aggregate_task_03():
    """Aggregate NLI results."""
    records = read_jsonl(str(RESULTS_DIR / "03_nli" / "seeds.jsonl"))
    if not records:
        return None

    by_model_ds = defaultdict(lambda: {"accuracy": [], "macro_f1": []})
    for r in records:
        key = r.get("dataset", "unknown")
        model_raw = r.get("model", "unknown")
        if "_" in model_raw:
            model = model_raw.rsplit("_", 1)[0]
        else:
            model = model_raw
        by_model_ds[(model, key)]["accuracy"].append(r.get("accuracy", 0))
        by_model_ds[(model, key)]["macro_f1"].append(r.get("macro_f1", 0))

    rows = []
    for dataset in ["xnli", "paraphrase"]:
        for model in ["gamba", "gsg", "banglabert"]:
            if (model, dataset) in by_model_ds:
                rows.append({
                    "model": model,
                    "dataset": dataset,
                    "accuracy": _mean_std(by_model_ds[(model, dataset)]["accuracy"]),
                    "macro_f1": _mean_std(by_model_ds[(model, dataset)]["macro_f1"]),
                    "n_seeds": len(by_model_ds[(model, dataset)]["accuracy"]),
                })

    return {"task": "03_nli", "metric": "Accuracy", "rows": rows}


def aggregate_task_04():
    """Aggregate MT results."""
    records = read_jsonl(str(RESULTS_DIR / "04_mt" / "seeds.jsonl"))
    if not records:
        return None

    rows = []
    for r in records:
        if r.get("skipped"):
            continue
        rows.append({
            "model": r.get("model", "unknown"),
            "bn_to_en_bleu": r.get("bn_to_en_bleu", "N/A"),
            "bn_to_en_chrf": r.get("bn_to_en_chrf", "N/A"),
            "en_to_bn_bleu": r.get("en_to_bn_bleu", "N/A"),
            "en_to_bn_chrf": r.get("en_to_bn_chrf", "N/A"),
        })

    return {"task": "04_mt", "metric": "BLEU / chrF", "rows": rows}


def aggregate_task_05():
    """Aggregate NIAH results."""
    rows = []
    for model in ["gamba", "gsg"]:
        summary = read_json(str(RESULTS_DIR / "05_long_context" / f"summary_{model}.json"))
        if summary:
            rows.append({
                "model": model,
                "overall_accuracy": summary.get("overall_accuracy", "N/A"),
                "total_correct": summary.get("total_correct", 0),
                "total_samples": summary.get("total_samples", 0),
            })

    return {"task": "05_long_context", "metric": "Accuracy", "rows": rows} if rows else None


def aggregate_task_06():
    """Aggregate summarization results."""
    records = read_jsonl(str(RESULTS_DIR / "06_summarization" / "seeds.jsonl"))
    if not records:
        return None

    rows = []
    for r in records:
        if r.get("skipped"):
            continue
        row = {
            "model": r.get("model", "unknown"),
            "rouge_l": r.get("rouge_l", "N/A"),
        }
        if "bertscore_f1" in r:
            row["bertscore_f1"] = r["bertscore_f1"]
        rows.append(row)

    return {"task": "06_summarization", "metric": "ROUGE-L", "rows": rows}


def generate_markdown_tables(all_results: list) -> str:
    """Generate markdown tables from aggregated results."""
    md = ["# Bangla LM Eval Suite — Results\n"]

    for result in all_results:
        if result is None:
            continue

        md.append(f"\n## {result['task']} (Primary: {result['metric']})\n")

        if not result.get("rows"):
            md.append("_No results available._\n")
            continue

        # Build table
        headers = list(result["rows"][0].keys())
        md.append("| " + " | ".join(headers) + " |")
        md.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in result["rows"]:
            md.append("| " + " | ".join(str(row.get(h, "N/A")) for h in headers) + " |")
        md.append("")

    # Combined summary table
    md.append("\n## Combined Summary\n")
    md.append("| Task | Dataset | BanglaGamba | BanglaGSG | BanglaBERT |")
    md.append("| --- | --- | --- | --- | --- |")

    # Build summary rows from individual results
    for result in all_results:
        if result is None:
            continue
        task = result["task"]
        metric = result["metric"]

        if task in ("02_ner", "03_nli"):
            # Multiple datasets
            datasets = sorted(set(r.get("dataset", "") for r in result.get("rows", [])))
            for ds in datasets:
                row_data = {"gamba": "N/A", "gsg": "N/A", "banglabert": "N/A"}
                for r in result.get("rows", []):
                    if r.get("dataset") == ds:
                        model = r.get("model", "")
                        primary_key = list(r.keys())[2]  # metric column
                        row_data[model] = str(r.get(primary_key, "N/A"))
                md.append(
                    f"| {task} ({metric}) | {ds} | "
                    f"{row_data['gamba']} | {row_data['gsg']} | {row_data['banglabert']} |"
                )
        else:
            row_data = {"gamba": "N/A", "gsg": "N/A", "banglabert": "N/A"}
            for r in result.get("rows", []):
                model = r.get("model", "")
                # Get primary metric value
                metric_keys = [k for k in r.keys() if k not in ("model", "dataset", "n_seeds")]
                if metric_keys:
                    row_data[model] = str(r.get(metric_keys[0], "N/A"))

            ds_name = result.get("dataset", "—")
            md.append(
                f"| {task} ({metric}) | {ds_name} | "
                f"{row_data['gamba']} | {row_data['gsg']} | {row_data['banglabert']} |"
            )

    return "\n".join(md)


def main():
    print("[Aggregation] Reading results from evaluation_suit/results/...\n")

    all_results = [
        aggregate_task_01(),
        aggregate_task_02(),
        aggregate_task_03(),
        aggregate_task_04(),
        aggregate_task_05(),
        aggregate_task_06(),
    ]

    # Save as JSON
    write_json(
        str(RESULTS_DIR / "aggregated_results.json"),
        [r for r in all_results if r is not None],
    )

    # Generate and save markdown
    md = generate_markdown_tables(all_results)
    md_path = RESULTS_DIR / "summary_tables.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")

    print(md)
    print(f"\n[Aggregation] Saved to:")
    print(f"  JSON: {RESULTS_DIR / 'aggregated_results.json'}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()

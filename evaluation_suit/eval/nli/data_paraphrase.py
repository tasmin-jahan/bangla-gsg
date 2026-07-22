"""
BanglaParaphrase dataset loader for NLI / sentence-pair evaluation.

Source: https://github.com/csebuetnlp/banglaparaphrase
Not on HF — cloned from GitHub and converted to a datasets-loadable format.

This is the "clean native-Bangla sentence-pair reasoning" counterpart to
XNLI's machine-translated artifacts.
"""

import os
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict


_GITHUB_URL = "https://github.com/csebuetnlp/banglaparaphrase.git"


def load_bangla_paraphrase(
    cache_dir: str = "evaluation_suit/data_cache/banglaparaphrase",
) -> DatasetDict:
    """
    Load BanglaParaphrase dataset from GitHub.

    Converts to a binary paraphrase detection task:
    - label 1: paraphrase pair
    - label 0: non-paraphrase pair (if available, or generated via negative sampling)

    Returns:
        DatasetDict with train, validation, test splits.
        Each example has 'premise' (str), 'hypothesis' (str), 'label' (int).
    """
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        print(f"[BanglaParaphrase] Cloning from {_GITHUB_URL}...")
        os.makedirs(cache_path.parent, exist_ok=True)
        ret = os.system(f"git clone {_GITHUB_URL} {cache_path}")
        if ret != 0:
            raise RuntimeError(
                f"Failed to clone BanglaParaphrase from {_GITHUB_URL}. "
                "Check your network connection or download manually."
            )

    # Discover data files
    data_files = {}
    search_dirs = [cache_path, cache_path / "data", cache_path / "dataset"]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for split_name in ["train", "test", "dev", "validation", "val"]:
            for ext in [".csv", ".tsv", ".jsonl", ".json", ".txt"]:
                cand = search_dir / f"{split_name}{ext}"
                if cand.exists():
                    canonical = "validation" if split_name in ("dev", "val") else split_name
                    data_files[canonical] = str(cand)
                    break

    if not data_files:
        # Search recursively
        all_files = (
            list(cache_path.rglob("*.csv"))
            + list(cache_path.rglob("*.tsv"))
            + list(cache_path.rglob("*.jsonl"))
        )
        all_files = [f for f in all_files if f.name.lower() not in (
            "readme.txt", "license.txt", "requirements.txt"
        )]
        if all_files:
            print(f"[BanglaParaphrase] Found: {[f.name for f in all_files]}")
            for f in all_files:
                name = f.stem.lower()
                if "train" in name:
                    data_files["train"] = str(f)
                elif "test" in name:
                    data_files["test"] = str(f)
                elif "dev" in name or "val" in name:
                    data_files["validation"] = str(f)
            if not data_files:
                data_files["train"] = str(all_files[0])

    if not data_files:
        raise FileNotFoundError(
            f"No data files found in {cache_path}. "
            "Check the repository structure."
        )

    print(f"[BanglaParaphrase] Found splits: {list(data_files.keys())}")

    datasets = {}
    for split, filepath in data_files.items():
        if filepath.endswith(".jsonl"):
            df = pd.read_json(filepath, lines=True)
        elif filepath.endswith(".json"):
            df = pd.read_json(filepath)
        elif filepath.endswith(".tsv"):
            df = pd.read_csv(filepath, sep="\t")
        else:
            df = pd.read_csv(filepath)

        # Find premise/hypothesis columns
        premise_col = None
        hypothesis_col = None
        label_col = None

        for col in df.columns:
            cl = col.lower().strip()
            if cl in ("premise", "sentence1", "text1", "source", "sent1"):
                premise_col = col
            elif cl in ("hypothesis", "sentence2", "text2", "target", "sent2", "paraphrase"):
                hypothesis_col = col
            elif cl in ("label", "is_paraphrase", "class", "gold_label"):
                label_col = col

        if premise_col is None:
            premise_col = df.columns[0]
            warnings.warn(f"[BanglaParaphrase] Guessing premise column: '{premise_col}'")
        if hypothesis_col is None:
            hypothesis_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            warnings.warn(f"[BanglaParaphrase] Guessing hypothesis column: '{hypothesis_col}'")

        records = {
            "premise": df[premise_col].astype(str).tolist(),
            "hypothesis": df[hypothesis_col].astype(str).tolist(),
        }

        if label_col is not None:
            labels = df[label_col].tolist()
            if isinstance(labels[0], str):
                label_map = {"1": 1, "0": 0, "true": 1, "false": 0,
                             "paraphrase": 1, "not_paraphrase": 0,
                             "yes": 1, "no": 0}
                labels = [label_map.get(str(l).lower(), 1) for l in labels]
            records["label"] = labels
        else:
            # All pairs are paraphrases (positive only dataset)
            records["label"] = [1] * len(records["premise"])
            warnings.warn(
                "[BanglaParaphrase] No label column found — treating all as positive pairs. "
                "Negative sampling may be needed for a balanced evaluation."
            )

        datasets[split] = Dataset.from_dict(records)
        print(f"[BanglaParaphrase] {split}: {len(datasets[split])} examples")

    ds = DatasetDict(datasets)

    if "test" not in ds and "train" in ds:
        split = ds["train"].train_test_split(test_size=0.2, seed=42)
        ds["train"] = split["train"]
        ds["test"] = split["test"]

    if "validation" not in ds and "train" in ds:
        split = ds["train"].train_test_split(test_size=0.1, seed=42)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]

    return ds


if __name__ == "__main__":
    ds = load_bangla_paraphrase()
    print(f"\nLoaded BanglaParaphrase: {ds}")
    for split in ds:
        print(f"  {split}: {len(ds[split])} examples")
        if len(ds[split]) > 0:
            ex = ds[split][0]
            print(f"    premise:    {ex['premise'][:80]}")
            print(f"    hypothesis: {ex['hypothesis'][:80]}")
            print(f"    label:      {ex['label']}")

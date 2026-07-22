"""
SentNoB dataset loader for sentiment analysis evaluation.

SentNoB is a 3-class (Positive, Negative, Neutral) Bangla sentiment
classification dataset from the BLP 2023 shared task.

Loading strategy (in order):
1. Try HF dataset hub mirrors.
2. Fall back to original GitHub: https://github.com/KhondokerIslam/SentNoB
3. Use local cached copy if available.

This module validates 3-class label distribution before returning.
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


# Known HF dataset IDs to try (some may be stale or renamed)
_HF_CANDIDATES = [
    "khondoker/SentNoB",
    "SentNoB",
    "KhondokerIslam/SentNoB",
    "sepidmnorozy/Bengali_sentiment",
]

_GITHUB_URL = "https://github.com/KhondokerIslam/SentNoB.git"
_LABEL_MAP = {"Positive": 0, "Negative": 1, "Neutral": 2}
_LABEL_NAMES = ["Positive", "Negative", "Neutral"]


def _try_load_from_hf() -> Optional[DatasetDict]:
    """Try loading SentNoB from known HF dataset IDs."""
    for candidate in _HF_CANDIDATES:
        try:
            ds = load_dataset(candidate)
            print(f"[SentNoB] Loaded from HF: {candidate}")
            # Normalize column names to 'text' and 'label'
            for split_name in list(ds.keys()):
                cols = ds[split_name].column_names
                if "Data" in cols:
                    ds[split_name] = ds[split_name].rename_column("Data", "text")
                if "Label" in cols:
                    ds[split_name] = ds[split_name].rename_column("Label", "label")
            return ds
        except Exception:
            continue
    return None


def _load_from_github(cache_dir: str = "evaluation_suit/data_cache/sentnob") -> DatasetDict:
    """
    Clone SentNoB from GitHub and convert to HF Dataset format.

    Expects TSV/CSV files with columns: text, label
    """
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        print(f"[SentNoB] Cloning from {_GITHUB_URL}...")
        os.makedirs(cache_path.parent, exist_ok=True)
        ret = os.system(f"git clone {_GITHUB_URL} {cache_path}")
        if ret != 0:
            raise RuntimeError(
                f"Failed to clone SentNoB from {_GITHUB_URL}. "
                "Check your network connection or download manually."
            )

    # Discover data files — SentNoB typically has train.csv / test.csv or similar
    data_files = {}
    for split_name in ["train", "test", "validation", "dev", "val"]:
        for ext in [".csv", ".tsv", ".txt"]:
            candidates = [
                cache_path / f"{split_name}{ext}",
                cache_path / "data" / f"{split_name}{ext}",
                cache_path / "dataset" / f"{split_name}{ext}",
            ]
            for cand in candidates:
                if cand.exists():
                    canonical = "validation" if split_name in ("dev", "val") else split_name
                    data_files[canonical] = str(cand)
                    break

    if not data_files:
        # Try to find any CSV/TSV files
        all_files = list(cache_path.rglob("*.csv")) + list(cache_path.rglob("*.tsv"))
        if all_files:
            warnings.warn(
                f"[SentNoB] No standard split files found. "
                f"Found: {[str(f) for f in all_files]}. "
                f"Loading the first file as 'train' and splitting."
            )
            data_files["train"] = str(all_files[0])
        else:
            raise FileNotFoundError(
                f"No data files found in {cache_path}. "
                "Please check the repository structure."
            )

    print(f"[SentNoB] Found splits: {list(data_files.keys())}")

    # Load and convert
    datasets = {}
    for split, filepath in data_files.items():
        sep = "\t" if filepath.endswith(".tsv") else ","
        try:
            df = pd.read_csv(filepath, sep=sep)
        except Exception:
            # Try with different encodings
            df = pd.read_csv(filepath, sep=sep, encoding="utf-8-sig")

        # Identify text and label columns
        text_col = None
        label_col = None
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ("text", "sentence", "review", "content"):
                text_col = col
            elif col_lower in ("label", "sentiment", "class", "category"):
                label_col = col

        if text_col is None:
            text_col = df.columns[0]
            warnings.warn(f"[SentNoB] Guessing text column: '{text_col}'")
        if label_col is None:
            label_col = df.columns[-1]
            warnings.warn(f"[SentNoB] Guessing label column: '{label_col}'")

        # Convert string labels to integers if needed
        labels = df[label_col].tolist()
        if isinstance(labels[0], str):
            label_set = sorted(set(labels))
            auto_map = {l: i for i, l in enumerate(label_set)}
            # Try to use our canonical map first
            final_map = {}
            for l in label_set:
                if l in _LABEL_MAP:
                    final_map[l] = _LABEL_MAP[l]
                elif l.capitalize() in _LABEL_MAP:
                    final_map[l] = _LABEL_MAP[l.capitalize()]
                else:
                    final_map[l] = auto_map[l]
            labels = [final_map[l] for l in labels]

        datasets[split] = Dataset.from_dict({
            "text": df[text_col].tolist(),
            "label": labels,
        })

    ds = DatasetDict(datasets)

    # If no test split, create one from train
    if "test" not in ds and "train" in ds:
        split = ds["train"].train_test_split(test_size=0.2, seed=42)
        ds["train"] = split["train"]
        ds["test"] = split["test"]

    # If no validation split, carve from train
    if "validation" not in ds and "train" in ds:
        split = ds["train"].train_test_split(test_size=0.1, seed=42)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]

    return ds


def load_sentnob(cache_dir: str = "evaluation_suit/data_cache/sentnob") -> DatasetDict:
    """
    Load SentNoB dataset, trying HF first, then GitHub fallback.

    Returns:
        DatasetDict with train, validation, test splits.
        Each example has 'text' (str) and 'label' (int: 0=Pos, 1=Neg, 2=Neutral).
    """
    # Strategy 1: Try HF
    ds = _try_load_from_hf()

    # Strategy 2: GitHub fallback
    if ds is None:
        print("[SentNoB] Not found on HF, falling back to GitHub clone...")
        ds = _load_from_github(cache_dir)

    # Ensure dataset has train, validation, test splits
    if "test" not in ds:
        # Split train into train (80%), val (10%), test (10%)
        split1 = ds["train"].train_test_split(test_size=0.2, seed=42)
        split2 = split1["test"].train_test_split(test_size=0.5, seed=42)
        ds = DatasetDict({
            "train": split1["train"],
            "validation": split2["train"],
            "test": split2["test"]
        })

    # Validate 3-class distribution
    for split_name in ds:
        labels = ds[split_name]["label"]
        unique_labels = set(labels)
        n_classes = len(unique_labels)
        if n_classes != 3:
            warnings.warn(
                f"[SentNoB] Expected 3 classes, found {n_classes} in '{split_name}' "
                f"split: {unique_labels}. This may indicate a loading error."
            )

        dist = {}
        for l in labels:
            dist[l] = dist.get(l, 0) + 1
        print(f"[SentNoB] {split_name}: {len(labels)} examples, distribution: {dist}")

    return ds


if __name__ == "__main__":
    ds = load_sentnob()
    print(f"\nLoaded SentNoB: {ds}")
    for split in ds:
        print(f"  {split}: {len(ds[split])} examples")
        print(f"    Sample: {ds[split][0]}")

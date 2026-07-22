"""
ANCHOLIK-NER dataset loader.

Source: https://github.com/AridHasan/ancholik-ner
Not on HF — cloned from GitHub and converted to a datasets-loadable format.

Yields examples as {tokens: [...], ner_tags: [...]}.
"""

import os
import warnings
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, ClassLabel, Sequence, Features


_GITHUB_URL = "https://github.com/AridHasan/ancholik-ner.git"


def _parse_conll_file(filepath: str) -> dict:
    """
    Parse a CoNLL-format NER file into tokens and tags.

    Expects one token per line with whitespace-separated columns,
    blank lines between sentences. Common formats:
    - token tag
    - token POS tag
    """
    all_tokens = []
    all_tags = []
    current_tokens = []
    current_tags = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                if current_tokens:
                    all_tokens.append(current_tokens)
                    all_tags.append(current_tags)
                    current_tokens = []
                    current_tags = []
                continue

            parts = line.split()
            if len(parts) >= 2:
                token = parts[0]
                tag = parts[-1]  # tag is typically the last column
                current_tokens.append(token)
                current_tags.append(tag)

    # Don't forget the last sentence
    if current_tokens:
        all_tokens.append(current_tokens)
        all_tags.append(current_tags)

    return {"tokens": all_tokens, "ner_tags": all_tags}


def _discover_tag_set(all_tags: list) -> list:
    """Discover unique NER tags and return sorted list."""
    tag_set = set()
    for seq in all_tags:
        for tag in seq:
            tag_set.add(tag)
    return sorted(tag_set)


def load_ancholik(
    cache_dir: str = "evaluation_suit/data_cache/ancholik_ner",
) -> DatasetDict:
    """
    Load ANCHOLIK-NER dataset from GitHub.

    Returns:
        DatasetDict with train/test (and validation if available).
        Each example has 'tokens' (list[str]) and 'ner_tags' (list[int]).
    """
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        print(f"[ANCHOLIK] Cloning from {_GITHUB_URL}...")
        os.makedirs(cache_path.parent, exist_ok=True)
        ret = os.system(f"git clone {_GITHUB_URL} {cache_path}")
        if ret != 0:
            raise RuntimeError(
                f"Failed to clone ANCHOLIK-NER from {_GITHUB_URL}. "
                "Check your network connection or download manually."
            )

    # Discover data files (CoNLL format)
    data_files = {}
    for split_name in ["train", "test", "dev", "validation", "val"]:
        for ext in [".txt", ".conll", ".tsv", ".bio", ""]:
            candidates = [
                cache_path / f"{split_name}{ext}",
                cache_path / "data" / f"{split_name}{ext}",
                cache_path / "dataset" / f"{split_name}{ext}",
            ]
            for cand in candidates:
                if cand.exists() and cand.is_file():
                    canonical = "validation" if split_name in ("dev", "val") else split_name
                    data_files[canonical] = str(cand)
                    break

    if not data_files:
        # Search recursively for likely NER data files
        all_files = list(cache_path.rglob("*.txt")) + list(cache_path.rglob("*.conll"))
        # Filter out READMEs, licenses, etc.
        all_files = [f for f in all_files if f.name.lower() not in (
            "readme.txt", "license.txt", "requirements.txt"
        )]
        if all_files:
            print(f"[ANCHOLIK] Found candidate files: {[f.name for f in all_files]}")
            # Heuristic: assign by name
            for f in all_files:
                name = f.stem.lower()
                if "train" in name:
                    data_files["train"] = str(f)
                elif "test" in name:
                    data_files["test"] = str(f)
                elif "dev" in name or "val" in name:
                    data_files["validation"] = str(f)
            # If still nothing matched, use all as train
            if not data_files:
                data_files["train"] = str(all_files[0])

    if not data_files:
        raise FileNotFoundError(
            f"No NER data files found in {cache_path}. "
            "Check the repository structure."
        )

    print(f"[ANCHOLIK] Found splits: {list(data_files.keys())}")

    # Parse all files to discover the global tag set
    parsed = {}
    all_tag_seqs = []
    for split, filepath in data_files.items():
        result = _parse_conll_file(filepath)
        parsed[split] = result
        all_tag_seqs.extend(result["ner_tags"])

    tag_names = _discover_tag_set(all_tag_seqs)
    tag_to_id = {tag: i for i, tag in enumerate(tag_names)}
    print(f"[ANCHOLIK] Tag set ({len(tag_names)}): {tag_names}")

    # Determine schema type (BIO, BILOU, etc.)
    has_b = any(t.startswith("B-") for t in tag_names)
    has_i = any(t.startswith("I-") for t in tag_names)
    has_l = any(t.startswith("L-") for t in tag_names)
    has_u = any(t.startswith("U-") for t in tag_names)
    if has_b and has_i and has_l and has_u:
        schema = "BILOU"
    elif has_b and has_i:
        schema = "BIO"
    else:
        schema = "UNKNOWN"
    print(f"[ANCHOLIK] Label schema: {schema}")

    # Build HF datasets
    features = Features({
        "tokens": Sequence(feature={"dtype": "string", "_type": "Value"}),
        "ner_tags": Sequence(feature=ClassLabel(names=tag_names)),
    })

    datasets = {}
    for split, result in parsed.items():
        # Convert string tags to integer IDs
        int_tags = [[tag_to_id[t] for t in seq] for seq in result["ner_tags"]]
        datasets[split] = Dataset.from_dict(
            {"tokens": result["tokens"], "ner_tags": int_tags},
        )
        print(f"[ANCHOLIK] {split}: {len(datasets[split])} sentences")

    ds = DatasetDict(datasets)

    # Create missing splits if needed
    if "test" not in ds and "train" in ds:
        split = ds["train"].train_test_split(test_size=0.2, seed=42)
        ds["train"] = split["train"]
        ds["test"] = split["test"]

    if "validation" not in ds and "train" in ds:
        split = ds["train"].train_test_split(test_size=0.1, seed=42)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]

    # Store tag metadata for later use
    ds._tag_names = tag_names
    ds._tag_to_id = tag_to_id
    ds._schema = schema

    return ds


if __name__ == "__main__":
    ds = load_ancholik()
    print(f"\nLoaded ANCHOLIK-NER: {ds}")
    for split in ds:
        print(f"  {split}: {len(ds[split])} sentences")
        if len(ds[split]) > 0:
            print(f"    Sample tokens: {ds[split][0]['tokens'][:5]}")
            print(f"    Sample tags:   {ds[split][0]['ner_tags'][:5]}")

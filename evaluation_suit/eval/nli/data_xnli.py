"""
XNLI-bn dataset loader for Natural Language Inference.

XNLI provides machine-translated NLI data for 15 languages including Bangla.
The Bangla translations are known to be noisy — this caveat must be noted
in the paper.
"""

import warnings

from datasets import DatasetDict, load_dataset


def load_xnli_bn() -> DatasetDict:
    """
    Load XNLI Bangla dataset from HF.

    Tries known config names (config naming has changed across
    dataset versions historically).

    Returns:
        DatasetDict with train, validation, test splits.
        Each example has 'premise' (str), 'hypothesis' (str), 'label' (int).
        Labels: 0=entailment, 1=neutral, 2=contradiction
    """
    # Config candidates — "bn" was historically valid but may have changed
    config_candidates = ["bn", "all_languages"]

    ds = None
    for config in config_candidates:
        try:
            ds = load_dataset("xnli", config)
            print(f"[XNLI] Loaded with config='{config}'")
            break
        except Exception as e:
            print(f"[XNLI] Config '{config}' failed: {e}")
            continue

    if ds is None:
        # Try loading without config and filtering by language
        try:
            ds = load_dataset("xnli", "all_languages")
            # Filter to Bangla
            ds = ds.filter(lambda x: x.get("language", "") == "bn")
            print("[XNLI] Loaded via 'all_languages' config, filtered to 'bn'")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load XNLI Bangla from HF. "
                f"Tried configs: {config_candidates}. "
                f"Last error: {e}. "
                f"Check the XNLI HF card for current Bangla config name."
            )

    # Verify we have the expected columns
    sample_split = list(ds.keys())[0]
    columns = ds[sample_split].column_names
    print(f"[XNLI] Columns: {columns}")

    # Standardize column names if needed
    for split in ds:
        if "premise" not in ds[split].column_names:
            # Try common alternatives
            for alt in ["sentence1", "text1"]:
                if alt in ds[split].column_names:
                    ds[split] = ds[split].rename_column(alt, "premise")
                    break
        if "hypothesis" not in ds[split].column_names:
            for alt in ["sentence2", "text2"]:
                if alt in ds[split].column_names:
                    ds[split] = ds[split].rename_column(alt, "hypothesis")
                    break

    # Print stats
    label_names = ["entailment", "neutral", "contradiction"]
    for split in ds:
        n = len(ds[split])
        labels = ds[split]["label"]
        dist = {}
        for l in labels:
            dist[label_names[l] if l < len(label_names) else str(l)] = dist.get(
                label_names[l] if l < len(label_names) else str(l), 0
            ) + 1
        print(f"[XNLI] {split}: {n} examples, distribution: {dist}")

    return ds


if __name__ == "__main__":
    ds = load_xnli_bn()
    print(f"\nLoaded XNLI-bn: {ds}")
    for split in ds:
        print(f"  {split}: {len(ds[split])} examples")
        if len(ds[split]) > 0:
            ex = ds[split][0]
            print(f"    premise:    {ex.get('premise', 'N/A')[:80]}")
            print(f"    hypothesis: {ex.get('hypothesis', 'N/A')[:80]}")
            print(f"    label:      {ex.get('label', 'N/A')}")

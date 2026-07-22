"""
WikiAnn-bn dataset loader for NER evaluation.

Uses the HF datasets library to load WikiAnn Bangla split.
This is the "comparable to published BanglaBERT NER numbers" baseline.
"""

import warnings

from datasets import DatasetDict, load_dataset


def load_wikiann_bn() -> DatasetDict:
    """
    Load WikiAnn Bangla NER dataset from HF.

    Tries known config names in order (config naming has shifted
    across datasets library versions).

    Returns:
        DatasetDict with train, validation, test splits.
        Each example has 'tokens' (list[str]) and 'ner_tags' (list[int]).
    """
    config_candidates = ["bn", "bn-BD"]

    ds = None
    for config in config_candidates:
        try:
            ds = load_dataset("wikiann", config)
            print(f"[WikiAnn] Loaded with config='{config}'")
            break
        except Exception as e:
            print(f"[WikiAnn] Config '{config}' failed: {e}")
            continue

    if ds is None:
        raise RuntimeError(
            "Failed to load WikiAnn Bangla from HF. "
            "Tried configs: {config_candidates}. "
            "Check 'datasets' library version and the WikiAnn HF card "
            "for the current Bangla split identifier."
        )

    # WikiAnn uses: 0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG, 5=B-LOC, 6=I-LOC
    tag_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    for split in ds:
        print(f"[WikiAnn] {split}: {len(ds[split])} sentences")

    # Store tag metadata
    ds._tag_names = tag_names
    ds._tag_to_id = {t: i for i, t in enumerate(tag_names)}
    ds._schema = "BIO"

    return ds


if __name__ == "__main__":
    ds = load_wikiann_bn()
    print(f"\nLoaded WikiAnn-bn: {ds}")
    for split in ds:
        print(f"  {split}: {len(ds[split])} sentences")
        if len(ds[split]) > 0:
            print(f"    Sample tokens: {ds[split][0]['tokens'][:5]}")
            print(f"    Sample tags:   {ds[split][0]['ner_tags'][:5]}")

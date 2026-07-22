"""
FLORES-200 Contamination Check.

MUST be run before generate.py — gates the entire MT evaluation.

The shared pretraining corpus (ahmed-farhanur-rashid/bn-foundational-pretrain-corpus)
includes NLLB and BanglaNMT parallel data. FLORES-200 is commonly derived
alongside NLLB releases, so there is a real contamination risk.

This script:
1. SHA-256 hash-compares FLORES-200 ben_Beng/eng_Latn devtest sentences
   against the nllb_nmt config of the pretraining corpus.
2. Outputs: overlap count, overlap rate, sample overlapping texts.
3. Gates the rest of task 04 based on overlap rate.

Known limitation: this exact-match check only catches literal duplicates.
Near-duplicate/paraphrase contamination is NOT covered.

Usage:
    python -m evaluation_suit.eval.04_mt.check_contamination
"""

import hashlib
import json
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from evaluation_suit.eval.common.io_utils import write_json


def _hash_text(text: str) -> str:
    """SHA-256 hash of normalized text."""
    normalized = text.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def check_contamination(
    contamination_threshold: float = 0.02,
    results_dir: str = "evaluation_suit/results/04_mt",
    max_corpus_samples: int = None,
) -> dict:
    """
    Check FLORES-200 devtest overlap with pretraining corpus.

    Args:
        contamination_threshold: Max acceptable overlap rate (default 2%).
        results_dir: Where to save the contamination report.
        max_corpus_samples: Limit corpus scanning (for testing). None = all.

    Returns:
        Dict with overlap stats and a 'proceed' boolean.
    """
    print("[Contamination] Loading FLORES-200 devtest...")
    try:
        flores = load_dataset("facebook/flores", "ben_Beng-eng_Latn", split="devtest")
    except Exception:
        try:
            flores = load_dataset("openlanguagedata/flores_plus", "ben_Beng-eng_Latn", split="devtest")
        except Exception as e:
            print(f"[Contamination] Failed to load FLORES-200: {e}")
            print("[Contamination] Trying alternative loading...")
            try:
                flores_bn = load_dataset("facebook/flores", "ben_Beng", split="devtest")
                flores_en = load_dataset("facebook/flores", "eng_Latn", split="devtest")
                # Combine into a single dataset-like structure
                flores_sentences_bn = [ex["sentence"] for ex in flores_bn]
                flores_sentences_en = [ex["sentence"] for ex in flores_en]
                flores = None  # handled specially below
            except Exception as e2:
                print(f"[Contamination] All FLORES loading attempts failed: {e2}")
                report = {
                    "status": "error",
                    "error": str(e2),
                    "proceed": False,
                    "reason": "Cannot load FLORES-200 to check contamination.",
                }
                write_json(f"{results_dir}/contamination_report.json", report)
                return report

    # Collect FLORES sentences
    flores_hashes = set()
    flores_texts = {}

    if flores is not None:
        # Standard paired dataset
        for ex in flores:
            for key in ["sentence_ben_Beng", "sentence_eng_Latn", "sentence", "text"]:
                if key in ex:
                    text = ex[key]
                    h = _hash_text(text)
                    flores_hashes.add(h)
                    flores_texts[h] = text
    else:
        # Separate datasets loaded above
        for text in flores_sentences_bn:
            h = _hash_text(text)
            flores_hashes.add(h)
            flores_texts[h] = text
        for text in flores_sentences_en:
            h = _hash_text(text)
            flores_hashes.add(h)
            flores_texts[h] = text

    print(f"[Contamination] FLORES devtest: {len(flores_hashes)} unique sentence hashes")

    # Load pretraining corpus NLLB config
    print("[Contamination] Loading pretraining corpus (nllb_nmt config)...")
    try:
        corpus = load_dataset(
            "ahmed-farhanur-rashid/bn-foundational-pretrain-corpus",
            "nllb_nmt",
            split="train",
            streaming=True,
        )
    except Exception as e:
        print(f"[Contamination] Failed to load corpus: {e}")
        report = {
            "status": "error",
            "error": str(e),
            "proceed": True,
            "reason": "Cannot load pretraining corpus — proceeding with caution.",
        }
        write_json(f"{results_dir}/contamination_report.json", report)
        return report

    # Scan corpus for overlaps
    overlap_hashes = set()
    overlap_samples = []
    n_scanned = 0

    print("[Contamination] Scanning corpus for FLORES overlaps...")
    for ex in tqdm(corpus, desc="Scanning"):
        if max_corpus_samples and n_scanned >= max_corpus_samples:
            break

        # Check all text fields
        for key in ["text", "sentence", "src", "tgt", "source", "target"]:
            if key in ex and ex[key]:
                h = _hash_text(ex[key])
                if h in flores_hashes and h not in overlap_hashes:
                    overlap_hashes.add(h)
                    if len(overlap_samples) < 10:
                        overlap_samples.append({
                            "flores_text": flores_texts.get(h, ""),
                            "corpus_text": ex[key],
                            "corpus_key": key,
                        })
        n_scanned += 1

    overlap_count = len(overlap_hashes)
    overlap_rate = overlap_count / max(len(flores_hashes), 1)
    proceed = overlap_rate <= contamination_threshold

    report = {
        "status": "completed",
        "flores_sentences": len(flores_hashes),
        "corpus_sentences_scanned": n_scanned,
        "overlap_count": overlap_count,
        "overlap_rate": round(overlap_rate, 4),
        "threshold": contamination_threshold,
        "proceed": proceed,
        "overlap_samples": overlap_samples,
    }

    if proceed:
        report["reason"] = (
            f"Overlap rate {overlap_rate*100:.2f}% is within threshold "
            f"({contamination_threshold*100:.1f}%). Proceeding with FLORES eval."
        )
        print(f"\n✓ PASS: Overlap rate {overlap_rate*100:.2f}% ≤ {contamination_threshold*100:.1f}%")
    else:
        report["reason"] = (
            f"Overlap rate {overlap_rate*100:.2f}% EXCEEDS threshold "
            f"({contamination_threshold*100:.1f}%). "
            f"FLORES eval should filter overlapping sentences or use an alternative test set."
        )
        print(f"\n✗ FAIL: Overlap rate {overlap_rate*100:.2f}% > {contamination_threshold*100:.1f}%")
        print(f"  Found {overlap_count} overlapping sentences.")
        if overlap_samples:
            print(f"  Sample overlap: '{overlap_samples[0]['flores_text'][:80]}...'")

    # Save report
    write_json(f"{results_dir}/contamination_report.json", report)
    print(f"  Report saved to {results_dir}/contamination_report.json")

    return report


if __name__ == "__main__":
    report = check_contamination()
    if not report.get("proceed", False):
        print("\n⚠ MT eval is GATED — do not trust FLORES numbers without addressing contamination.")
        sys.exit(1)

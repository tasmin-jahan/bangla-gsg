"""
Shared metric wrappers for the eval suite.

One function per metric so task scripts don't each reimplement scoring.
All functions accept plain Python lists and return float scores.
"""

from typing import List, Optional

import numpy as np
from sklearn.metrics import f1_score as sklearn_f1
from sklearn.metrics import accuracy_score as sklearn_accuracy


def macro_f1(y_true: List, y_pred: List) -> float:
    """Macro-averaged F1 score (sentiment, NLI)."""
    return float(sklearn_f1(y_true, y_pred, average="macro"))


def accuracy_score(y_true: List, y_pred: List) -> float:
    """Simple accuracy (NLI primary metric)."""
    return float(sklearn_accuracy(y_true, y_pred))


def entity_f1(y_true_tags: List[List[str]], y_pred_tags: List[List[str]]) -> float:
    """
    Entity-level F1 using seqeval (NER standard metric).

    Args:
        y_true_tags: List of tag sequences, e.g. [["B-PER", "I-PER", "O"], ...]
        y_pred_tags: Same shape as y_true_tags.

    Returns:
        Entity-level micro-averaged F1.
    """
    from seqeval.metrics import f1_score as seqeval_f1
    return float(seqeval_f1(y_true_tags, y_pred_tags))


def compute_bleu(references: List[str], hypotheses: List[str]) -> float:
    """
    Corpus-level BLEU using sacrebleu.

    Args:
        references: List of reference strings.
        hypotheses: List of hypothesis strings.

    Returns:
        BLEU score (0-100 scale, sacrebleu convention).
    """
    import sacrebleu
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return float(result.score)


def compute_chrf(references: List[str], hypotheses: List[str]) -> float:
    """Corpus-level chrF using sacrebleu."""
    import sacrebleu
    result = sacrebleu.corpus_chrf(hypotheses, [references])
    return float(result.score)


def compute_rouge(references: List[str], hypotheses: List[str]) -> dict:
    """
    ROUGE scores (1, 2, L) using the evaluate library.

    Returns:
        Dict with keys: rouge1, rouge2, rougeL, rougeLsum (each 0-1 scale).
    """
    import evaluate
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=hypotheses, references=references)
    return {k: float(v) for k, v in results.items()}


def compute_bertscore(
    references: List[str],
    hypotheses: List[str],
    lang: str = "bn",
) -> Optional[dict]:
    """
    BERTScore using the evaluate library.

    Returns dict with precision, recall, f1 (averaged across examples),
    or None if BERTScore fails to load.
    """
    try:
        import evaluate
        bertscore = evaluate.load("bertscore")
        results = bertscore.compute(
            predictions=hypotheses,
            references=references,
            lang=lang,
        )
        return {
            "precision": float(np.mean(results["precision"])),
            "recall": float(np.mean(results["recall"])),
            "f1": float(np.mean(results["f1"])),
        }
    except Exception as e:
        import warnings
        warnings.warn(
            f"BERTScore failed to compute: {e}. "
            "Falling back to ROUGE-L only."
        )
        return None

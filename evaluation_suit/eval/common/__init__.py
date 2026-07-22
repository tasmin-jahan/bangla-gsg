"""
Eval Suite — Common Utilities Package.

Exports: load_model, set_seed, io helpers, metric wrappers.
"""

from evaluation_suit.eval.common.model_registry import load_model, LoadedModel
from evaluation_suit.eval.common.seeding import set_seed
from evaluation_suit.eval.common.io_utils import write_jsonl, read_jsonl, append_result
from evaluation_suit.eval.common.metrics import (
    macro_f1,
    accuracy_score,
    entity_f1,
    compute_bleu,
    compute_chrf,
    compute_rouge,
    compute_bertscore,
)

__all__ = [
    "load_model",
    "LoadedModel",
    "set_seed",
    "write_jsonl",
    "read_jsonl",
    "append_result",
    "macro_f1",
    "accuracy_score",
    "entity_f1",
    "compute_bleu",
    "compute_chrf",
    "compute_rouge",
    "compute_bertscore",
]

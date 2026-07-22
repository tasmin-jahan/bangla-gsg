"""
SentNoB Sentiment Classification — Fine-tuning & Evaluation.

Fine-tunes a classification head on top of gamba/gsg/banglabert for
3-class sentiment analysis. Supports multi-seed runs for statistical
significance.

Usage:
    python -m evaluation_suit.eval.01_sentiment.run \
        --model gamba --seed 0 --epochs 5 --lr 2e-5 --batch_size 16

    # Run all 3 models × 3 seeds:
    for model in gamba gsg banglabert; do
        for seed in 0 1 2; do
            python -m evaluation_suit.eval.01_sentiment.run \
                --model $model --seed $seed
        done
    done
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from evaluation_suit.eval.common.model_registry import load_model
from evaluation_suit.eval.common.seeding import set_seed
from evaluation_suit.eval.common.io_utils import append_result, get_completed_runs
from evaluation_suit.eval.common.metrics import macro_f1, accuracy_score
from evaluation_suit.eval.sentiment.data import load_sentnob


# ── Classification Head ──────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Simple classification head over a pretrained LM.

    Pooling strategy:
    - causal_lm (gamba/gsg): last non-pad token's hidden state
    - masked_lm (banglabert): [CLS] token (first token)
    """

    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor,
                model_type: str) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, H) from the base model
            attention_mask: (B, T) with 1s for real tokens, 0s for padding
            model_type: "causal_lm" or "masked_lm"

        Returns:
            logits: (B, num_classes)
        """
        if model_type == "causal_lm":
            # Last non-pad token pooling
            # Find the index of the last real token for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
            seq_lengths = seq_lengths.clamp(min=0)
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            pooled = hidden_states[batch_idx, seq_lengths]  # (B, H)
        else:
            # [CLS] token pooling (first token)
            pooled = hidden_states[:, 0]  # (B, H)

        return self.classifier(self.dropout(pooled))


# ── Dataset Wrapper ──────────────────────────────────────────────────────────

class SentimentDataset(Dataset):
    """Wraps HF dataset for PyTorch DataLoader with tokenization."""

    def __init__(self, hf_dataset, tokenizer, max_len: int = 256, padding: str | bool = "max_length"):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.padding = padding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = item["label"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=self.padding,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ── Hidden State Extraction ──────────────────────────────────────────────────

def get_hidden_states(model, input_ids, model_type, model_key):
    """
    Extract hidden states from the base model.

    Handles the different model architectures:
    - BanglaGamba: model.model.model(input_ids, return_hidden=True)
    - BanglaGSG: similar pattern with trust_remote_code
    - BanglaBERT: model.bert(input_ids) or model(input_ids, output_hidden_states=True)
    """
    if model_type == "causal_lm":
        # For causal LMs (gamba/gsg), we need the hidden states before lm_head
        # The HF wrapper's inner model has return_hidden support
        inner = model
        if hasattr(model, "model"):
            inner = model.model

        # Try return_hidden=True (BanglaGamba native)
        if hasattr(inner, "forward") and "return_hidden" in str(inner.forward.__code__.co_varnames):
            hidden = inner(input_ids, return_hidden=True)
            return hidden

        # Fallback: use output_hidden_states
        outputs = model(input_ids, output_hidden_states=True)
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]

        # Last resort: hook into the model
        raise RuntimeError(
            f"Cannot extract hidden states from {model_key}. "
            "Model does not support return_hidden or output_hidden_states."
        )
    else:
        # Masked LM (BanglaBERT)
        outputs = model(input_ids, output_hidden_states=True)
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        raise RuntimeError("Cannot extract hidden states from masked LM.")


# ── Training Loop ────────────────────────────────────────────────────────────

def train_and_evaluate(
    model_key: str,
    seed: int,
    epochs: int = 5,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_seq_len: int = 256,
    results_dir: str = "evaluation_suit/results/01_sentiment",
) -> dict:
    """
    Fine-tune and evaluate a single model on SentNoB.

    Returns:
        Dict with model, seed, macro_f1, accuracy, and metadata.
    """
    set_seed(seed)

    # Check if already completed
    results_path = os.path.join(results_dir, "seeds.jsonl")
    completed = get_completed_runs(results_path)
    if (model_key, seed) in completed:
        print(f"[01_sentiment] {model_key}/seed={seed} already completed. Skipping.")
        return None

    # Load model
    loaded = load_model(model_key)
    device = loaded.device

    # Load dataset
    dataset = load_sentnob()

    # Determine hidden size
    config = loaded.model.config
    hidden_size = getattr(config, "d_model", None) or getattr(config, "hidden_size", 768)

    # Build classification head
    head = ClassificationHead(hidden_size=hidden_size, num_classes=3).to(device)

    # Handle batch size & padding for causal LMs (unpadded dense batches required)
    eff_batch_size = 1 if loaded.model_type == "causal_lm" else batch_size
    padding_strat = False if loaded.model_type == "causal_lm" else "max_length"

    # Build data loaders
    train_ds = SentimentDataset(dataset["train"], loaded.tokenizer, max_len=max_seq_len, padding=padding_strat)
    val_ds = SentimentDataset(
        dataset.get("validation", dataset.get("test")),
        loaded.tokenizer,
        max_len=max_seq_len,
        padding=padding_strat,
    )
    test_ds = SentimentDataset(dataset["test"], loaded.tokenizer, max_len=max_seq_len, padding=padding_strat)

    train_loader = DataLoader(train_ds, batch_size=eff_batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=eff_batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=eff_batch_size, shuffle=False, num_workers=2)

    # Optimizer — only train the classification head, freeze base model
    # (This is the standard approach for comparing base model representations)
    loaded.model.eval()
    for param in loaded.model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

    # Training
    best_val_f1 = 0.0
    best_head_state = None

    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"[{model_key}] Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    hidden = get_hidden_states(
                        loaded.model, input_ids, loaded.model_type, model_key
                    )

            logits = head(hidden.float(), attn_mask, loaded.model_type)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Validation
        head.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels = batch["label"]

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    hidden = get_hidden_states(
                        loaded.model, input_ids, loaded.model_type, model_key
                    )

                logits = head(hidden.float(), attn_mask, loaded.model_type)
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        val_f1 = macro_f1(all_labels, all_preds)
        val_acc = accuracy_score(all_labels, all_preds)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, val_macro_f1={val_f1:.4f}, val_acc={val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_head_state = {k: v.clone() for k, v in head.state_dict().items()}

    # Test evaluation with best head
    if best_head_state is not None:
        head.load_state_dict(best_head_state)

    head.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden = get_hidden_states(
                    loaded.model, input_ids, loaded.model_type, model_key
                )

            logits = head(hidden.float(), attn_mask, loaded.model_type)
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    test_f1 = macro_f1(all_labels, all_preds)
    test_acc = accuracy_score(all_labels, all_preds)

    print(f"\n[{model_key}] seed={seed} → test_macro_f1={test_f1:.4f}, test_acc={test_acc:.4f}")

    # Sanity check for BanglaBERT
    if model_key == "banglabert":
        if abs(test_f1 - 0.7289) > 0.05:
            print(
                f"  ⚠ WARNING: BanglaBERT macro-F1 ({test_f1:.4f}) is >5 points "
                f"from the reference 72.89%. Debug the fine-tuning setup before "
                f"trusting downstream comparisons."
            )

    # Save result
    result = {
        "model": model_key,
        "seed": seed,
        "task": "01_sentiment",
        "dataset": "sentnob",
        "macro_f1": round(test_f1, 4),
        "accuracy": round(test_acc, 4),
        "best_val_f1": round(best_val_f1, 4),
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
    }
    append_result(results_path, result)
    print(f"  Result appended to {results_path}")

    # Cleanup GPU memory
    del loaded, head
    torch.cuda.empty_cache()

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SentNoB Sentiment Classification")
    parser.add_argument("--model", type=str, required=True,
                        choices=["gamba", "gsg", "banglabert"],
                        help="Model key to evaluate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Max sequence length")
    args = parser.parse_args()

    result = train_and_evaluate(
        model_key=args.model,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    if result is not None:
        print(f"\nFinal: {result}")


if __name__ == "__main__":
    main()

"""
NLI Sentence-Pair Classification — Fine-tuning & Evaluation.

Fine-tunes a classification head on top of gamba/gsg/banglabert for
Natural Language Inference (XNLI-bn) and paraphrase detection
(BanglaParaphrase).

Sentence-pair encoding:
- For tokenizers with pair-encoding support: use text_pair argument
- Fallback for custom tokenizers (GSG/Gamba): concatenate with explicit
  separator "। " (Bangla danda + space)

Usage:
    python -m evaluation_suit.eval.03_nli.run \
        --model gamba --dataset xnli --seed 0

    python -m evaluation_suit.eval.03_nli.run \
        --model banglabert --dataset paraphrase --seed 0
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from evaluation_suit.eval.common.model_registry import load_model
from evaluation_suit.eval.common.seeding import set_seed
from evaluation_suit.eval.common.io_utils import append_result, get_completed_runs
from evaluation_suit.eval.common.metrics import accuracy_score, macro_f1


# ── Classification Head (reused from 01_sentiment) ───────────────────────────

class ClassificationHead(nn.Module):
    """Classification head with last-token (causal) or [CLS] (masked) pooling."""

    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, attention_mask, model_type):
        if model_type == "causal_lm":
            seq_lengths = attention_mask.sum(dim=1) - 1
            seq_lengths = seq_lengths.clamp(min=0)
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            pooled = hidden_states[batch_idx, seq_lengths]
        else:
            pooled = hidden_states[:, 0]
        return self.classifier(self.dropout(pooled))


# ── Dataset Wrapper ──────────────────────────────────────────────────────────

# Bangla separator for concatenating premise + hypothesis in causal LMs
_SEPARATOR = " । "


class NLIDataset(Dataset):
    """Wraps HF NLI dataset with sentence-pair tokenization."""

    def __init__(self, hf_dataset, tokenizer, max_len: int = 256, model_type: str = "causal_lm", padding: str | bool = "max_length"):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_type = model_type
        self.padding = padding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        premise = item["premise"]
        hypothesis = item["hypothesis"]
        label = item["label"]

        if self.model_type == "masked_lm":
            # BanglaBERT — use standard text_pair encoding with [SEP]
            encoding = self.tokenizer(
                premise,
                text_pair=hypothesis,
                truncation=True,
                max_length=self.max_len,
                padding=self.padding,
                return_tensors="pt",
            )
        else:
            # Causal LMs — concatenate with explicit separator
            # Custom tokenizers (GSG/Gamba) may not have pair-encoding support
            combined = premise + _SEPARATOR + hypothesis
            encoding = self.tokenizer(
                combined,
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

def get_hidden_states(model, input_ids, model_type):
    """Extract hidden states from the base model."""
    if model_type == "causal_lm":
        inner = model
        if hasattr(model, "model"):
            inner = model.model
        if hasattr(inner, "forward") and "return_hidden" in str(inner.forward.__code__.co_varnames):
            return inner(input_ids, return_hidden=True)
        outputs = model(input_ids, output_hidden_states=True)
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        raise RuntimeError("Cannot extract hidden states from causal LM.")
    else:
        outputs = model(input_ids, output_hidden_states=True)
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        raise RuntimeError("Cannot extract hidden states from masked LM.")


# ── Training Loop ────────────────────────────────────────────────────────────

def train_and_evaluate(
    model_key: str,
    dataset_name: str,
    seed: int,
    epochs: int = 5,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_seq_len: int = 256,
    results_dir: str = "evaluation_suit/results/nli",
    save_checkpoint: bool = False,
) -> dict:
    """Fine-tune and evaluate NLI on a single (model, dataset, seed) combo."""
    set_seed(seed)

    results_path = f"{results_dir}/seeds.jsonl"
    completed = get_completed_runs(results_path)
    run_key = f"{model_key}_{dataset_name}"
    if (run_key, seed) in completed:
        print(f"[03_nli] {run_key}/seed={seed} already completed. Skipping.")
        return None

    # Load model
    loaded = load_model(model_key)
    device = loaded.device

    # Load dataset
    if dataset_name == "xnli":
        from evaluation_suit.eval.nli.data_xnli import load_xnli_bn
        dataset = load_xnli_bn()
        num_classes = 3  # entailment, neutral, contradiction
    elif dataset_name == "paraphrase":
        from evaluation_suit.eval.nli.data_paraphrase import load_bangla_paraphrase
        dataset = load_bangla_paraphrase()
        num_classes = 2  # paraphrase, not_paraphrase
    else:
        raise ValueError(f"Unknown NLI dataset: {dataset_name}")

    # Hidden size
    config = loaded.model.config
    hidden_size = getattr(config, "d_model", None) or getattr(config, "hidden_size", 768)

    # Build head
    head = ClassificationHead(hidden_size, num_classes).to(device)

    # Handle batch size & padding for causal LMs (unpadded dense batches required)
    eff_batch_size = 1 if loaded.model_type == "causal_lm" else batch_size
    padding_strat = False if loaded.model_type == "causal_lm" else "max_length"

    # Data loaders
    train_ds = NLIDataset(dataset["train"], loaded.tokenizer, max_seq_len, loaded.model_type, padding=padding_strat)
    val_split = "validation" if "validation" in dataset else "test"
    val_ds = NLIDataset(dataset[val_split], loaded.tokenizer, max_seq_len, loaded.model_type, padding=padding_strat)
    test_ds = NLIDataset(dataset["test"], loaded.tokenizer, max_seq_len, loaded.model_type, padding=padding_strat)

    train_loader = DataLoader(train_ds, batch_size=eff_batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=eff_batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=eff_batch_size, shuffle=False, num_workers=2)

    # Freeze base model
    loaded.model.eval()
    for param in loaded.model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    best_val_acc = 0.0
    best_head_state = None

    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"[{model_key}/{dataset_name}] Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    hidden = get_hidden_states(loaded.model, input_ids, loaded.model_type)

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
                    hidden = get_hidden_states(loaded.model, input_ids, loaded.model_type)

                logits = head(hidden.float(), attn_mask, loaded.model_type)
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = macro_f1(all_labels, all_preds)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_head_state = {k: v.clone() for k, v in head.state_dict().items()}

    # Test with best head
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
                hidden = get_hidden_states(loaded.model, input_ids, loaded.model_type)

            logits = head(hidden.float(), attn_mask, loaded.model_type)
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = macro_f1(all_labels, all_preds)

    print(f"\n[{model_key}/{dataset_name}] seed={seed} → test_acc={test_acc:.4f}, test_f1={test_f1:.4f}")

    result = {
        "model": run_key,
        "seed": seed,
        "task": "03_nli",
        "dataset": dataset_name,
        "accuracy": round(test_acc, 4),
        "macro_f1": round(test_f1, 4),
        "best_val_acc": round(best_val_acc, 4),
        "num_classes": num_classes,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
    }
    append_result(results_path, result)
    print(f"  Result appended to {results_path}")

    # Save fine-tuned head checkpoint if requested
    if save_checkpoint and best_head_state is not None:
        ckpt_dir = Path(f"evaluation_suit/checkpoints/nli/{model_key}_{dataset_name}_seed{seed}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_head_state, ckpt_dir / "classifier_head.pt")
        from evaluation_suit.eval.common.io_utils import write_json
        write_json(ckpt_dir / "checkpoint_info.json", {
            "model": model_key,
            "task": "nli",
            "dataset": dataset_name,
            "seed": seed,
            "test_accuracy": round(test_acc, 4),
            "test_macro_f1": round(test_f1, 4),
            "num_classes": num_classes,
        })
        print(f"  ✓ Fine-tuned head checkpoint saved to {ckpt_dir}")

    del loaded, head
    torch.cuda.empty_cache()

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NLI Sentence-Pair Classification")
    parser.add_argument("--model", type=str, required=True,
                        choices=["gamba", "gsg", "banglabert"])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["xnli", "paraphrase"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=256)
    args = parser.parse_args()

    train_and_evaluate(
        model_key=args.model,
        dataset_name=args.dataset,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )


if __name__ == "__main__":
    main()

"""
NER Token Classification — Fine-tuning & Evaluation.

Fine-tunes a per-token classification head for Named Entity Recognition
on ANCHOLIK-NER and WikiAnn-bn datasets.

Key differences from sentence classification (01_sentiment):
- Per-token linear classifier over hidden_states (no pooling)
- Label alignment: word-level NER tags → subword tokens
  - First subword gets the word's label
  - Continuation subwords get -100 (ignored in loss)
- Metric: seqeval entity-level F1 (not token-level accuracy)

Usage:
    python -m evaluation_suit.eval.02_ner.run \
        --model gamba --dataset ancholik --seed 0

    python -m evaluation_suit.eval.02_ner.run \
        --model banglabert --dataset wikiann --seed 0
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from evaluation_suit.eval.common.model_registry import load_model
from evaluation_suit.eval.common.seeding import set_seed
from evaluation_suit.eval.common.io_utils import append_result, get_completed_runs
from evaluation_suit.eval.common.metrics import entity_f1


# ── Token Classification Head ────────────────────────────────────────────────

class TokenClassificationHead(nn.Module):
    """
    Per-token linear classifier over hidden states.

    Unlike sentence classification, there is NO pooling step — we classify
    every token position independently.
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, H)
        Returns:
            logits: (B, T, num_labels)
        """
        return self.classifier(self.dropout(hidden_states))


# ── Label Alignment ──────────────────────────────────────────────────────────

def align_labels_with_tokens(
    tokens: List[str],
    ner_tags: List[int],
    tokenizer,
    max_len: int = 256,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Align word-level NER tags with subword tokens.

    Standard NER label alignment:
    - First subword of each word gets the word's NER tag
    - Continuation subwords get -100 (ignored in loss/eval)
    - Special tokens (BOS, EOS, PAD) get -100
    - Padding positions get -100

    This is the most common source of silent NER bugs — test carefully.

    Returns:
        (input_ids, attention_mask, aligned_labels) — all length max_len
    """
    # Tokenize each word individually to track word boundaries
    word_ids = []
    all_input_ids = []

    # Add BOS if tokenizer uses it
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None:
        all_input_ids.append(bos_id)
        word_ids.append(None)  # special token

    for word_idx, word in enumerate(tokens):
        word_tokens = tokenizer.encode(word, add_special_tokens=False)
        if not word_tokens:
            continue
        for i, token_id in enumerate(word_tokens):
            all_input_ids.append(token_id)
            if i == 0:
                word_ids.append(word_idx)  # first subword → word index
            else:
                word_ids.append(-1)  # continuation subword

    # Truncate to max_len (leaving room for possible EOS)
    max_content = max_len - 1 if getattr(tokenizer, "eos_token_id", None) else max_len
    all_input_ids = all_input_ids[:max_content]
    word_ids = word_ids[:max_content]

    # Build aligned labels
    aligned_labels = []
    for wid in word_ids:
        if wid is None or wid == -1:
            aligned_labels.append(-100)
        else:
            if wid < len(ner_tags):
                aligned_labels.append(ner_tags[wid])
            else:
                aligned_labels.append(-100)

    # Pad to max_len
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    pad_len = max_len - len(all_input_ids)
    attention_mask = [1] * len(all_input_ids) + [0] * pad_len
    aligned_labels = aligned_labels + [-100] * pad_len
    all_input_ids = all_input_ids + [pad_id] * pad_len

    return all_input_ids, attention_mask, aligned_labels


# ── Dataset Wrapper ──────────────────────────────────────────────────────────

class NERDataset(Dataset):
    """Wraps HF NER dataset for PyTorch with subword label alignment."""

    def __init__(self, hf_dataset, tokenizer, max_len: int = 256):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        ner_tags = item["ner_tags"]

        input_ids, attention_mask, labels = align_labels_with_tokens(
            tokens, ner_tags, self.tokenizer, self.max_len
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ── Hidden State Extraction (reused pattern) ─────────────────────────────────

def get_hidden_states(model, input_ids, model_type):
    """Extract per-token hidden states from the base model."""
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


# ── Decode Predictions to Tag Strings ─────────────────────────────────────────

def decode_predictions(
    all_preds: List[List[int]],
    all_labels: List[List[int]],
    tag_names: List[str],
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Convert integer predictions/labels back to string tags for seqeval.
    Filters out -100 positions (subword continuations and padding).
    """
    pred_tags = []
    true_tags = []

    for preds, labels in zip(all_preds, all_labels):
        seq_preds = []
        seq_labels = []
        for p, l in zip(preds, labels):
            if l == -100:
                continue  # skip subword continuations and padding
            seq_preds.append(tag_names[p] if p < len(tag_names) else "O")
            seq_labels.append(tag_names[l] if l < len(tag_names) else "O")
        if seq_preds:  # non-empty sequence
            pred_tags.append(seq_preds)
            true_tags.append(seq_labels)

    return true_tags, pred_tags


# ── Training Loop ────────────────────────────────────────────────────────────

def train_and_evaluate(
    model_key: str,
    dataset_name: str,
    seed: int,
    epochs: int = 10,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_seq_len: int = 256,
    results_dir: str = "evaluation_suit/results/02_ner",
) -> dict:
    """Fine-tune and evaluate NER on a single (model, dataset, seed) combo."""
    set_seed(seed)

    results_path = f"{results_dir}/seeds.jsonl"
    completed = get_completed_runs(results_path)
    run_key = f"{model_key}_{dataset_name}"
    if (run_key, seed) in completed:
        print(f"[02_ner] {run_key}/seed={seed} already completed. Skipping.")
        return None

    # Load model
    loaded = load_model(model_key)
    device = loaded.device

    # Load dataset
    if dataset_name == "ancholik":
        from evaluation_suit.eval.ner.data_ancholik import load_ancholik
        dataset = load_ancholik()
    elif dataset_name == "wikiann":
        from evaluation_suit.eval.ner.data_wikiann import load_wikiann_bn
        dataset = load_wikiann_bn()
    else:
        raise ValueError(f"Unknown NER dataset: {dataset_name}")

    tag_names = dataset._tag_names
    num_labels = len(tag_names)

    # Hidden size
    config = loaded.model.config
    hidden_size = getattr(config, "d_model", None) or getattr(config, "hidden_size", 768)

    # Build head
    head = TokenClassificationHead(hidden_size, num_labels).to(device)

    # Data loaders
    train_ds = NERDataset(dataset["train"], loaded.tokenizer, max_len=max_seq_len)
    val_split = "validation" if "validation" in dataset else "test"
    val_ds = NERDataset(dataset[val_split], loaded.tokenizer, max_len=max_seq_len)
    test_ds = NERDataset(dataset["test"], loaded.tokenizer, max_len=max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Freeze base model
    loaded.model.eval()
    for param in loaded.model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    best_val_f1 = 0.0
    best_head_state = None

    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"[{model_key}/{dataset_name}] Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    hidden = get_hidden_states(loaded.model, input_ids, loaded.model_type)

            logits = head(hidden.float())  # (B, T, num_labels)
            loss = F.cross_entropy(
                logits.view(-1, num_labels),
                labels.view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Validation
        head.eval()
        all_preds_seq = []
        all_labels_seq = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"]

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    hidden = get_hidden_states(loaded.model, input_ids, loaded.model_type)

                logits = head(hidden.float())
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds_seq.extend(preds)
                all_labels_seq.extend(labels.tolist())

        true_tags, pred_tags = decode_predictions(all_preds_seq, all_labels_seq, tag_names)
        val_f1 = entity_f1(true_tags, pred_tags) if true_tags else 0.0
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, val_entity_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_head_state = {k: v.clone() for k, v in head.state_dict().items()}

    # Test with best head
    if best_head_state is not None:
        head.load_state_dict(best_head_state)

    head.eval()
    all_preds_seq = []
    all_labels_seq = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"]

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden = get_hidden_states(loaded.model, input_ids, loaded.model_type)

            logits = head(hidden.float())
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds_seq.extend(preds)
            all_labels_seq.extend(labels.tolist())

    true_tags, pred_tags = decode_predictions(all_preds_seq, all_labels_seq, tag_names)
    test_f1 = entity_f1(true_tags, pred_tags) if true_tags else 0.0

    print(f"\n[{model_key}/{dataset_name}] seed={seed} → test_entity_f1={test_f1:.4f}")

    result = {
        "model": run_key,
        "seed": seed,
        "task": "02_ner",
        "dataset": dataset_name,
        "entity_f1": round(test_f1, 4),
        "best_val_f1": round(best_val_f1, 4),
        "num_labels": num_labels,
        "tag_schema": getattr(dataset, "_schema", "unknown"),
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
    }
    append_result(results_path, result)
    print(f"  Result appended to {results_path}")

    del loaded, head
    torch.cuda.empty_cache()

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NER Token Classification")
    parser.add_argument("--model", type=str, required=True,
                        choices=["gamba", "gsg", "banglabert"])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["ancholik", "wikiann"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
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

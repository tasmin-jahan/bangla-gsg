"""
Checkpoint management for BanglaGSG.

Handles save/load of dual optimizer states (Muon + AdamW),
both schedulers, RNG state, and training progress.

Spec §7: Keep last N + best-by-validation-perplexity.
"""

import os
from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    muon_optimizer: torch.optim.Optimizer,
    adamw_optimizer: torch.optim.Optimizer,
    muon_scheduler,
    adamw_scheduler,
    step: int,
    tokens_seen: int,
    train_loss: float,
    val_perplexity: Optional[float] = None,
    config: Optional[dict] = None,
):
    """
    Save a complete training checkpoint with dual optimizer states.

    Args:
        path: File path to save the checkpoint.
        model: The model (state_dict will be saved).
        muon_optimizer: Muon optimizer.
        adamw_optimizer: AdamW optimizer.
        muon_scheduler: Muon LR scheduler.
        adamw_scheduler: AdamW LR scheduler.
        step: Current training step.
        tokens_seen: Total tokens processed.
        train_loss: Current training loss.
        val_perplexity: Validation perplexity (if computed).
        config: Model config dict (for reproducibility).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "tokens_seen": tokens_seen,
        "train_loss": train_loss,
        "val_perplexity": val_perplexity,
        "model_state_dict": model.state_dict(),
        "muon_optimizer_state_dict": muon_optimizer.state_dict(),
        "adamw_optimizer_state_dict": adamw_optimizer.state_dict(),
        "muon_sched_state_dict": muon_scheduler.state_dict(),
        "adamw_sched_state_dict": adamw_scheduler.state_dict(),
        "config": config,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }

    torch.save(checkpoint, path)
    print(f"[Checkpoint] Saved step {step} → {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    muon_optimizer: torch.optim.Optimizer,
    adamw_optimizer: torch.optim.Optimizer,
    muon_scheduler,
    adamw_scheduler,
    device: str = "cuda",
) -> dict:
    """
    Load a checkpoint and restore all state.

    Returns the checkpoint dict (for step, tokens_seen, etc.).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    muon_optimizer.load_state_dict(checkpoint["muon_optimizer_state_dict"])
    adamw_optimizer.load_state_dict(checkpoint["adamw_optimizer_state_dict"])
    muon_scheduler.load_state_dict(checkpoint["muon_sched_state_dict"])
    adamw_scheduler.load_state_dict(checkpoint["adamw_sched_state_dict"])

    # Restore RNG state
    torch.set_rng_state(checkpoint["rng_state"])
    if checkpoint.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])

    print(f"[Checkpoint] Resumed from step {checkpoint['step']} ← {path}")
    return checkpoint


def manage_checkpoints(
    checkpoint_dir: str,
    keep_last: int = 3,
    best_path: Optional[str] = None,
):
    """
    Keep only the last N checkpoints + best checkpoint.
    Deletes older ones to save disk space.
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return

    # Find all step checkpoints
    ckpts = sorted(
        ckpt_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )

    # Keep best and last N
    protected = set()
    if best_path and Path(best_path).exists():
        protected.add(Path(best_path).resolve())

    for ckpt in ckpts[-keep_last:]:
        protected.add(ckpt.resolve())

    for ckpt in ckpts:
        if ckpt.resolve() not in protected:
            ckpt.unlink()
            print(f"[Checkpoint] Deleted old checkpoint: {ckpt.name}")

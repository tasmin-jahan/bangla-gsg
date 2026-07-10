"""
Checkpoint management for BanglaGSG.

Handles save/load of dual optimizer states (Muon + AdamW),
both schedulers, RNG state, and training progress.

Features:
  - Atomic saves (write to .tmp then os.replace) — no corrupted files on power loss
  - Epoch + batches_consumed tracking for deterministic resume
  - Wall-clock persistence for accurate elapsed time across restarts
  - Model export (weights + config only, no optimizer bloat)
"""

import os
from pathlib import Path
from typing import Optional

import torch
import yaml
from tqdm import tqdm


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
    epoch: int = 0,
    batches_consumed_this_epoch: int = 0,
    data_seed: Optional[int] = None,
    wall_clock: float = 0.0,
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
        epoch: Current epoch index (0-based). Required to rebuild the
            correct epoch-seeded shuffle order on resume.
        batches_consumed_this_epoch: Micro-batches consumed so far in
            `epoch`, at the current shuffle order. Required for exact
            fast-forward on resume (no duplication, no skipped data).
        data_seed: The base_seed used to build the training dataloader's
            per-epoch sampler (base_seed + epoch). Saved for sanity
            verification on resume — if the loader is rebuilt with a
            different base seed, fast-forwarding would silently replay
            the wrong permutation.
        wall_clock: Cumulative wall-clock seconds of actual training
            across all sessions. Used to restore accurate elapsed time
            on resume.
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
        "epoch": epoch,
        "batches_consumed_this_epoch": batches_consumed_this_epoch,
        "data_seed": data_seed,
        "wall_clock": wall_clock,
    }

    # Atomic save: write to tmp then rename — prevents corrupted
    # checkpoints if power is lost mid-write.
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)

    tqdm.write(f"[Checkpoint] Saved step {step} (epoch {epoch}, "
               f"batch {batches_consumed_this_epoch} in epoch) → {path}")


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
    torch.set_rng_state(checkpoint["rng_state"].cpu())
    if checkpoint.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu())

    print(f"[Checkpoint] Resumed from step {checkpoint['step']} "
          f"(epoch {checkpoint.get('epoch', 0)}, "
          f"batch {checkpoint.get('batches_consumed_this_epoch', 0)} in epoch) ← {path}")
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


def export_model(
    model: torch.nn.Module,
    config: dict,
    model_dir: str,
    run_name: str = "default",
) -> str:
    """
    Export the final trained model (weights + config only, no optimizer state).

    This produces a stripped-down version suitable for HuggingFace upload:
      - model.pt: just the state_dict (no optimizer, scheduler, RNG state)
      - config.yaml: model architecture config for reconstruction

    Args:
        model: The trained model.
        config: Model config dict.
        model_dir: Base directory for model exports.
        run_name: Run identifier (subdirectory name).

    Returns:
        Path to the export directory.
    """
    export_dir = Path(model_dir) / run_name
    export_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights only
    model_path = export_dir / "model.pt"
    torch.save(model.state_dict(), str(model_path))

    # Save config for reconstruction
    config_path = export_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    tqdm.write(f"[Export] Model exported to {export_dir}/ "
               f"(model.pt + config.yaml, no optimizer state)")
    return str(export_dir)
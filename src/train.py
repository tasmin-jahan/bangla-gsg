"""
BanglaGSG Training Entry Point.

Usage:
    cd bangla-gsg/
    python src/train.py \
        --model configs/banglagsg_12l.yaml \
        --training configs/default_training.yaml \
        --optimizer configs/muon_adamw.yaml \
        --data configs/default_data.yaml

    # Resume from checkpoint:
    python src/train.py \
        --model configs/banglagsg_12l.yaml \
        --training configs/default_training.yaml \
        --optimizer configs/muon_adamw.yaml \
        --data configs/default_data.yaml \
        --resume
"""

import argparse
import sys
import os

# Add project root to path so 'from src.xxx import yyy' works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch

from src.model.config import BanglaGSGConfig
from src.model.model import BanglaGSGModel
from src.model.optim import build_optimizers, load_optimizer_config
from src.data.collator import build_dataloader
from src.training.trainer import Trainer, TrainerConfig
from src.training.scheduler import build_schedulers
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="BanglaGSG Training")
    parser.add_argument("--model", type=str, default="configs/banglagsg_12l.yaml",
                        help="Path to model config YAML")
    parser.add_argument("--training", type=str, default="configs/default_training.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--optimizer", type=str, default="configs/muon_adamw.yaml",
                        help="Path to optimizer config YAML")
    parser.add_argument("--data", type=str, default="configs/default_data.yaml",
                        help="Path to data config YAML")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--resume-path", type=str, default=None,
                        help="Resume from a specific checkpoint path")
    parser.add_argument("--seed", type=int, default=1552,
                        help="Random seed (default: 1552)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Set seed ──────────────────────────────────────────────────────────
    set_seed(args.seed)

    # ── Load configs ──────────────────────────────────────────────────────
    model_config = BanglaGSGConfig.from_yaml(args.model)
    print(model_config.summary())

    trainer_config = TrainerConfig.from_yaml(args.training)
    optimizer_config = load_optimizer_config(args.optimizer)

    with open(args.data, "r") as f:
        data_config = yaml.safe_load(f)

    # ── Derive run name from model config filename ────────────────────────
    if trainer_config.run_name == "default":
        config_stem = os.path.splitext(os.path.basename(args.model))[0]
        trainer_config.run_name = config_stem

    # ── Persist model config to run log directory ─────────────────────────
    from pathlib import Path
    run_log_dir = Path(trainer_config.log_dir) / trainer_config.run_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    model_config.to_yaml(str(run_log_dir / "model_config.yaml"))

    # ── Device ────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARNING] No CUDA device found. Training on CPU will be extremely slow.")

    # ── Build model ───────────────────────────────────────────────────────
    print(f"\n[Init] Building model...")
    model = BanglaGSGModel(model_config).to(device)
    param_counts = model.count_parameters()
    print(f"[Init] Total parameters: {param_counts['total']:,} ({param_counts['total']/1e6:.1f}M)")
    print(f"[Init]   Embedding: {param_counts['embedding']:,}")
    print(f"[Init]   GDN:       {param_counts['gdn']:,}")
    print(f"[Init]   SWA:       {param_counts['swa']:,}")
    print(f"[Init]   GQA:       {param_counts['gqa']:,}")
    print(f"[Init]   FFN:       {param_counts['ffn']:,}")

    # ── Enable gradient checkpointing ─────────────────────────────────────
    if trainer_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("[Init] Gradient checkpointing enabled")

    # ── torch.compile ─────────────────────────────────────────────────────
    if trainer_config.compile_model:
        try:
            model = torch.compile(model)
            print("[Init] torch.compile enabled (default mode)")
        except Exception as e:
            print(f"[Init] torch.compile failed: {e} — continuing without compilation")

    # ── Build optimizers ──────────────────────────────────────────────────
    print(f"\n[Init] Building optimizers...")
    muon_optimizer, adamw_optimizer = build_optimizers(model, optimizer_config)

    # ── Build dataloaders ─────────────────────────────────────────────────
    print(f"\n[Init] Building dataloaders...")
    train_npy_dir = data_config.get("train_npy_dir", "data/tokenized/train")
    eval_npy_dir = data_config.get("eval_npy_dir", None)
    batch_size = data_config.get("batch_size", 4)
    num_workers = data_config.get("num_workers", 2)
    max_shards = data_config.get("max_shards", 0)

    train_loader = build_dataloader(
        npy_dir=train_npy_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # Set to False to respect manual shard interleaving
        pin_memory=True,
        max_shards=max_shards,
    )

    eval_loader = None
    if eval_npy_dir:
        from pathlib import Path
        eval_path = Path(eval_npy_dir)
        try:
            subdirs = [d for d in eval_path.iterdir() if d.is_dir()]
            if subdirs:
                eval_loader = {}
                for subdir in subdirs:
                    if list(subdir.rglob("*.npy")):
                        eval_loader[subdir.name] = build_dataloader(
                            npy_dir=str(subdir),
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False,
                            pin_memory=True,
                        )
                if not eval_loader:
                    eval_loader = None
            else:
                eval_loader = build_dataloader(
                    npy_dir=eval_npy_dir,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False,
                    pin_memory=True,
                )
        except FileNotFoundError:
            print(f"[Init] Eval dir not found: {eval_npy_dir} — skipping eval")

    # ── Compute total steps and build schedulers ──────────────────────────
    if trainer_config.max_steps > 0:
        total_steps = trainer_config.max_steps
    else:
        total_steps = len(train_loader) // trainer_config.accumulation_steps
        total_steps = max(total_steps, 1)

    warmup_steps = int(total_steps * trainer_config.warmup_ratio)

    muon_scheduler, adamw_scheduler = build_schedulers(
        muon_optimizer, adamw_optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=trainer_config.min_lr_ratio,
    )
    print(f"[Init] Total steps: {total_steps:,} | Warmup: {warmup_steps:,}")

    # ── Build trainer ─────────────────────────────────────────────────────
    print(f"\n[Init] Building trainer (run: {trainer_config.run_name})...")
    trainer = Trainer(
        model=model,
        muon_optimizer=muon_optimizer,
        adamw_optimizer=adamw_optimizer,
        muon_scheduler=muon_scheduler,
        adamw_scheduler=adamw_scheduler,
        train_loader=train_loader,
        eval_loader=eval_loader,
        config=trainer_config,
        model_config=model_config,
        device=device,
    )

    # ── Resume if requested ───────────────────────────────────────────────
    if args.resume or args.resume_path:
        trainer.resume(path=args.resume_path)

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  BanglaGSG Training — {trainer_config.run_name}")
    print(f"  Model: {param_counts['total']/1e6:.1f}M params")
    print(f"  Device: {device}")
    print(f"  Steps: {trainer.total_steps:,}")
    print(f"{'='*70}\n")

    trainer.train()

    print("\n[Done] Training complete.")


if __name__ == "__main__":
    main()

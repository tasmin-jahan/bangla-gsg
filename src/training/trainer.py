"""
BanglaGSG Training Loop.

Implements the full training loop:
- BF16 autocast
- Gradient accumulation
- Z-loss: 1e-4 * logsumexp(logits).pow(2).mean()
- Global gradient clipping across both param groups
- Dual optimizer step order: muon → adamw → sched × 2 → zero_grad × 2
- Per-group gradient norm logging (first 300 steps)
- Gradient checkpointing
- torch.compile support
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import signal

from src.model.optim import build_param_groups
from src.training.checkpoint import save_checkpoint, load_checkpoint, manage_checkpoints
from src.utils.logging import MetricLogger


@dataclass
class TrainerConfig:
    """Training hyperparameters loaded from YAML."""
    # Schedule
    warmup_ratio: float = 0.015
    min_lr_ratio: float = 0.1

    # Gradient
    max_grad_norm: float = 1.0
    accumulation_steps: int = 64

    # Stability
    z_loss_weight: float = 1e-4

    # Memory
    gradient_checkpointing: bool = True
    compile_model: bool = True

    # Checkpointing
    checkpoint_dir: str = "saved/checkpoints"
    checkpoint_every: int = 2000
    keep_checkpoints: int = 3
    log_dir: str = "saved/logs"
    model_dir: str = "saved/model"

    # Run
    run_name: str = "default"
    max_steps: int = 0  # 0 = compute from data
    log_every: int = 10
    eval_every: int = 500
    eval_batches: int = 50 # How many batches to evaluate on mid-training
    grad_norm_monitor_steps: int = 300

    # Pad token
    pad_token_id: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "TrainerConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str) -> "TrainerConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})


class Trainer:
    """
    Training loop for BanglaGSG with dual Muon + AdamW optimizers.
    """

    def __init__(
        self,
        model: nn.Module,
        muon_optimizer: torch.optim.Optimizer,
        adamw_optimizer: torch.optim.Optimizer,
        muon_scheduler,
        adamw_scheduler,
        train_loader,
        eval_loader=None,
        config: TrainerConfig = None,
        model_config=None,
        device: str = "cuda",
        train_loader_fn=None,
    ):
        self.model = model
        self.muon_optimizer = muon_optimizer
        self.adamw_optimizer = adamw_optimizer
        self.muon_scheduler = muon_scheduler
        self.adamw_scheduler = adamw_scheduler
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config or TrainerConfig()
        self.model_config = model_config
        self.device = device
        self._interrupt_requested = False

        # train_loader_fn(epoch) -> DataLoader, used to rebuild the training
        # loader with the correct epoch-seeded shuffle order on resume and
        # at epoch boundaries. If not provided, the loader passed in above
        # is reused for every epoch (no per-epoch reshuffling / no exact
        # resume guarantee — only leave this None for eval-only use).
        self.train_loader_fn = train_loader_fn
        self.epoch = 0
        # Micro-batches (pre-accumulation) consumed so far in the current
        # epoch. Tracked at micro-batch granularity (not optimizer-step
        # granularity) so fast-forward-on-resume is exact even mid-accumulation.
        self.batches_consumed_this_epoch = 0
        # Base seed used by train_loader_fn to derive each epoch's shuffle
        # order (base_seed + epoch). Set by the caller (see src/train.py)
        # so resume() can sanity-check it against the checkpoint.
        self.data_seed = None

        # Speed optimizations (TF32)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Get param lists for gradient clipping
        self.muon_params, self.adamw_params = build_param_groups(model)

        # Compute total steps
        if self.config.max_steps > 0:
            self.total_steps = self.config.max_steps
        else:
            # Estimate from dataloader
            steps_per_epoch = len(train_loader) // self.config.accumulation_steps
            self.total_steps = max(steps_per_epoch, 1)

        self.warmup_steps = int(self.total_steps * self.config.warmup_ratio)

        # Logging
        self.logger = MetricLogger(self.config.log_dir, self.config.run_name)

        # State
        self.global_step = 0
        self.tokens_seen = 0
        self.best_val_ppl = float("inf")

    def _install_signal_handlers(self) -> None:
        def _handler(signum, frame):
            sig_name = signal.Signals(signum).name
            tqdm.write(f"\n[Signal] Received {sig_name} — finishing current step then saving emergency checkpoint...")
            self._interrupt_requested = True
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple:
        """
        Compute CE loss + z-loss.

        Args:
            logits: (B, T, V) raw logits.
            targets: (B, T) target token IDs.

        Returns:
            (total_loss, ce_loss_value, z_loss_value)
        """
        B, T, V = logits.shape

        ce_loss = F.cross_entropy(
            logits.view(-1, V),
            targets.view(-1),
            ignore_index=self.config.pad_token_id,
        )

        # Z-loss: penalty on logit magnitude
        z_loss = self.config.z_loss_weight * torch.logsumexp(logits, dim=-1).pow(2).mean()

        total_loss = ce_loss + z_loss

        return total_loss, ce_loss.item(), z_loss.item()

    def compute_grad_norms(self) -> tuple:
        """Compute per-group gradient norms for monitoring."""
        muon_norm = torch.nn.utils.clip_grad_norm_(self.muon_params, float("inf")).item()
        adamw_norm = torch.nn.utils.clip_grad_norm_(self.adamw_params, float("inf")).item()
        return muon_norm, adamw_norm

    def compute_component_grad_norms(self) -> dict:
        """Compute per-component gradient L2 norms for analysis."""
        gdn_sq = 0.0
        swa_sq = 0.0
        gqa_sq = 0.0

        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g_sq = p.grad.data.float().pow(2).sum().item()
            name_lower = name.lower()
            if "gdn" in name_lower:
                gdn_sq += g_sq
            elif "swa" in name_lower:
                swa_sq += g_sq
            elif "gqa" in name_lower:
                gqa_sq += g_sq

        return {
            "gdn_grad_norm": math.sqrt(gdn_sq),
            "swa_grad_norm": math.sqrt(swa_sq),
            "gqa_grad_norm": math.sqrt(gqa_sq),
        }

    @torch.no_grad()
    def evaluate(self) -> float:
        """Compute validation perplexity."""
        if self.eval_loader is None:
            return float("inf")

        self.model.eval()

        def eval_single_loader(loader):
            total_loss = 0.0
            n_batches = 0
            for i, batch in enumerate(loader):
                if self.config.eval_batches > 0 and i >= self.config.eval_batches:
                    break
                input_ids = batch["input_ids"].to(self.device)
                targets = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.model(input_ids)
                    loss, _, _ = self.compute_loss(logits, targets)

                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / max(n_batches, 1)
            return math.exp(min(avg_loss, 20))

        if isinstance(self.eval_loader, dict):
            ppls = {}
            for name, loader in self.eval_loader.items():
                ppls[name] = eval_single_loader(loader)
            self.model.train()
            avg = sum(ppls.values()) / len(ppls)
            return {"overall": avg, **ppls}
        else:
            ppl = eval_single_loader(self.eval_loader)
            self.model.train()
            return {"overall": ppl}

    def resume(self, path: Optional[str] = None):
        """Resume training from a checkpoint."""
        if path is None:
            # Find latest checkpoint
            ckpt_dir = self.config.checkpoint_dir
            ckpts = sorted(
                (p for p in __import__("pathlib").Path(ckpt_dir).glob("step_*.pt")),
                key=lambda p: int(p.stem.split("_")[1]),
            )
            if not ckpts:
                print("[Trainer] No checkpoints found, starting from scratch.")
                return
            path = str(ckpts[-1])

        ckpt = load_checkpoint(
            path, self.model,
            self.muon_optimizer, self.adamw_optimizer,
            self.muon_scheduler, self.adamw_scheduler,
            device=self.device,
        )
        self.global_step = ckpt["step"]
        self.tokens_seen = ckpt.get("tokens_seen", 0)
        self.best_val_ppl = ckpt.get("val_perplexity", float("inf"))
        self.epoch = ckpt.get("epoch", 0)
        self.batches_consumed_this_epoch = ckpt.get("batches_consumed_this_epoch", 0)

        ckpt_seed = ckpt.get("data_seed")
        if ckpt_seed is not None and self.data_seed is not None and ckpt_seed != self.data_seed:
            print(f"[Trainer] WARNING: checkpoint was trained with data_seed="
                  f"{ckpt_seed}, but this run is using data_seed={self.data_seed}. "
                  f"Fast-forward-on-resume will NOT replay the correct shuffle "
                  f"order — you will get duplicated/skipped data this epoch. "
                  f"Fix by passing --seed {ckpt_seed} (or matching data_seed) "
                  f"to this run.")

    def train(self):
        """Run the full training loop."""
        self.model.train()
        accum = self.config.accumulation_steps
        seq_len = self.model_config.seq_len if self.model_config else 2048

        # Zero grads initially
        self.muon_optimizer.zero_grad(set_to_none=True)
        self.adamw_optimizer.zero_grad(set_to_none=True)

        micro_step = 0
        running_loss = 0.0
        running_ce = 0.0
        running_z = 0.0
        step_start = time.time()

        # Build (or rebuild) the loader for the current epoch with its
        # deterministic, epoch-seeded shuffle order, then fast-forward
        # WITHIN that epoch only, by the exact number of micro-batches
        # already consumed. Because the shuffle order for (base_seed,
        # epoch) is reproducible, this lands exactly on the next unseen
        # batch: no duplicates, no skipped/unseen sequences.
        if self.train_loader_fn is not None:
            self.train_loader = self.train_loader_fn(self.epoch)

        data_iter = iter(self.train_loader)
        batches_to_skip = self.batches_consumed_this_epoch

        if batches_to_skip > 0:
            print(f"[Trainer] Resuming epoch {self.epoch}: fast-forwarding "
                  f"dataloader by {batches_to_skip} batches (exact replay of "
                  f"this epoch's shuffle order)...")
            for _ in range(batches_to_skip):
                next(data_iter)  # must not wrap here — wrapping would mean
                                  # we mis-tracked batches_consumed_this_epoch
            print("[Trainer] Dataloader fast-forward complete.")

        session_start = time.time()
        resume_step = self.global_step

        pbar = tqdm(
            total=self.total_steps,
            initial=self.global_step,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            bar_format=(
                "\033[36m{desc}\033[0m "
                "{percentage:5.1f}% "
                "|{bar}| "
                "{n_fmt}/{total_fmt} "
                "{postfix}"
            ),
            colour="cyan",
        )

        self._install_signal_handlers()

        while self.global_step < self.total_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                # Epoch boundary: advance to the next epoch's deterministic
                # shuffle order (base_seed + epoch), not a re-shuffle of the
                # same epoch. Reset the in-epoch counter so a resume that
                # lands right after this point fast-forwards correctly
                # against the *new* epoch's permutation.
                self.epoch += 1
                self.batches_consumed_this_epoch = 0
                if self.train_loader_fn is not None:
                    self.train_loader = self.train_loader_fn(self.epoch)
                tqdm.write(f"[Trainer] Epoch {self.epoch - 1} complete. "
                           f"Starting epoch {self.epoch} (new shuffle order).")
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            self.batches_consumed_this_epoch += 1

            input_ids = batch["input_ids"].to(self.device)
            targets = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()

            # Forward + backward
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(input_ids)
                loss, ce_val, z_val = self.compute_loss(logits, targets)
                loss = loss / accum  # scale for accumulation

            loss.backward()

            running_loss += loss.item() * accum
            running_ce += ce_val
            running_z += z_val
            micro_step += 1
            self.tokens_seen += input_ids.numel()

            # Optimizer step at accumulation boundary
            if micro_step % accum == 0:
                self.global_step += 1

                # Compute all grad norms on log steps, before clipping
                muon_gnorm, adamw_gnorm = None, None
                comp_norms = None
                if self.global_step % self.config.log_every == 0:
                    muon_gnorm, adamw_gnorm = self.compute_grad_norms()
                    comp_norms = self.compute_component_grad_norms()

                # Global gradient clipping across both groups
                torch.nn.utils.clip_grad_norm_(
                    list(self.muon_params) + list(self.adamw_params),
                    max_norm=self.config.max_grad_norm,
                )

                # Step order
                self.muon_optimizer.step()
                self.adamw_optimizer.step()
                self.muon_scheduler.step()
                self.adamw_scheduler.step()
                self.muon_optimizer.zero_grad(set_to_none=True)
                self.adamw_optimizer.zero_grad(set_to_none=True)

                # Terminal UI calculations (Every Step)
                avg_loss = running_loss / accum
                avg_ce = running_ce / accum
                avg_z = running_z / accum

                elapsed = time.time() - step_start
                tokens_per_sec = (accum * input_ids.numel()) / max(elapsed, 1e-6)
                ppl = math.exp(min(avg_ce, 20.0))
                peak_gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

                # ETA calculation
                session_elapsed = time.time() - session_start
                steps_this_session = self.global_step - resume_step
                if steps_this_session > 0:
                    sec_per_step = session_elapsed / steps_this_session
                    eta_sec = sec_per_step * (self.total_steps - self.global_step)
                else:
                    eta_sec = -1
                    
                def _fmt_time(s):
                    if s < 0: return "??:??:??"
                    h, m = divmod(s, 3600)
                    m, s = divmod(m, 60)
                    return f"{int(h)}h{int(m):02d}m{int(s):02d}s" if h > 0 else (f"{int(m)}m{int(s):02d}s" if m > 0 else f"{int(s)}s")

                eta_str = _fmt_time(eta_sec)
                elapsed_str = _fmt_time(self.logger._elapsed())

                pbar.set_postfix_str(
                    f"\033[90m[{elapsed_str} < \033[97m{eta_str}\033[90m]\033[0m "
                    f"loss=\033[93m{avg_loss:.3f}\033[0m "
                    f"ppl=\033[93m{ppl:.1f}\033[0m "
                    f"tok/s=\033[92m{tokens_per_sec:,.0f}\033[0m "
                    f"gpu=\033[95m{peak_gpu_mem:.0f}\033[0mMB"
                )

                # CSV Logging (Only every 'log_every' steps)
                if self.global_step % self.config.log_every == 0:
                    lr_muon = self.muon_scheduler.get_last_lr()[0]
                    lr_adamw = self.adamw_scheduler.get_last_lr()[0]
                    gpu_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                    epoch_frac = self.global_step / max(self.total_steps, 1)

                    self.logger.log({
                        "step": self.global_step,
                        "tokens_seen": self.tokens_seen,
                        "epoch_frac": round(epoch_frac, 5),
                        "loss": round(avg_loss, 5),
                        "perplexity": round(ppl, 3),
                        "z_loss": round(avg_z, 7),
                        "lr_muon": lr_muon,
                        "lr_adamw": lr_adamw,
                        "grad_norm_muon": muon_gnorm,
                        "grad_norm_adamw": adamw_gnorm,
                        "gdn_grad_norm": comp_norms["gdn_grad_norm"] if comp_norms else None,
                        "swa_grad_norm": comp_norms["swa_grad_norm"] if comp_norms else None,
                        "gqa_grad_norm": comp_norms["gqa_grad_norm"] if comp_norms else None,
                        "tokens_per_sec": round(tokens_per_sec, 1),
                        "gpu_mem_mb": round(gpu_mem, 1),
                        "peak_gpu_mem_mb": round(peak_gpu_mem, 1)
                    })

                pbar.update(1)
                running_loss = 0.0
                running_ce = 0.0
                running_z = 0.0
                step_start = time.time()

                # Evaluation
                if self.eval_loader and self.global_step % self.config.eval_every == 0:
                    eval_results = self.evaluate()
                    val_ppl = eval_results["overall"]
                    tqdm.write(f"  ✦ [Eval] overall_val_ppl={val_ppl:.2f}")

                    # Log to eval_metrics.csv
                    eval_log = {"step": self.global_step, "tokens_seen": self.tokens_seen}
                    eval_log.update({f"val_ppl_{k}": round(v, 3) for k, v in eval_results.items()})
                    self.logger.log_eval(eval_log)

                # Periodic checkpointing
                if self.global_step % self.config.checkpoint_every == 0:
                    ckpt_path = f"{self.config.checkpoint_dir}/step_{self.global_step:08d}.pt"
                    tqdm.write(f"  💾 Checkpoint step={self.global_step:,} → {ckpt_path}")
                    save_checkpoint(
                        ckpt_path, self.model,
                        self.muon_optimizer, self.adamw_optimizer,
                        self.muon_scheduler, self.adamw_scheduler,
                        self.global_step, self.tokens_seen,
                        avg_loss,
                        config=self.model_config.__dict__ if self.model_config else None,
                        epoch=self.epoch,
                        batches_consumed_this_epoch=self.batches_consumed_this_epoch,
                        data_seed=self.data_seed,
                    )
                    manage_checkpoints(
                        self.config.checkpoint_dir,
                        keep_last=self.config.keep_checkpoints,
                        best_path=f"{self.config.checkpoint_dir}/best.pt",
                    )

                if self._interrupt_requested:
                    tqdm.write(f"\n[Trainer] Interrupt acknowledged at step {self.global_step}. Saving emergency checkpoint...")
                    ckpt_path = f"{self.config.checkpoint_dir}/step_{self.global_step:08d}.pt"
                    save_checkpoint(
                        ckpt_path, self.model,
                        self.muon_optimizer, self.adamw_optimizer,
                        self.muon_scheduler, self.adamw_scheduler,
                        self.global_step, self.tokens_seen,
                        running_loss / max(accum, 1) if running_loss > 0 else 0.0,
                        config=self.model_config.__dict__ if self.model_config else None,
                        epoch=self.epoch,
                        batches_consumed_this_epoch=self.batches_consumed_this_epoch,
                        data_seed=self.data_seed,
                    )
                    manage_checkpoints(
                        self.config.checkpoint_dir,
                        keep_last=self.config.keep_checkpoints,
                        best_path=f"{self.config.checkpoint_dir}/best.pt",
                    )
                    pbar.close()
                    import sys
                    print("[Trainer] Emergency save complete. Exiting gracefully.")
                    sys.exit(0)

        import os
        os.makedirs(self.config.model_dir, exist_ok=True)
        final_path = f"{self.config.model_dir}/model.pt"
        torch.save(self.model.state_dict(), final_path)
        tqdm.write(f"  💾 Final model saved to {final_path}")

        pbar.close()
        print(f"[Trainer] Training complete. {self.global_step} steps, {self.tokens_seen:,} tokens.")
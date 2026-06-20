"""
Learning rate scheduler for BanglaGSG.

Linear warmup → cosine decay to min_lr_ratio of peak (spec §4.4).
Applied identically to both Muon and AdamW via LambdaLR.
"""

import math
from typing import Tuple

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_lr_multiplier(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """
    Compute LR multiplier for warmup + cosine decay schedule.

    Args:
        step: Current training step.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps.
        min_lr_ratio: Final LR as fraction of peak (default 0.1 = 10%).

    Returns:
        LR multiplier in [min_lr_ratio, 1.0].
    """
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def build_schedulers(
    muon_optimizer: torch.optim.Optimizer,
    adamw_optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> Tuple[LambdaLR, LambdaLR]:
    """
    Build LR schedulers for both optimizers.

    Both use the same schedule shape (warmup + cosine decay) but
    different peak LRs (set in the optimizer config).

    Returns:
        (muon_scheduler, adamw_scheduler)
    """
    lr_fn = lambda step: get_lr_multiplier(step, warmup_steps, total_steps, min_lr_ratio)

    muon_sched = LambdaLR(muon_optimizer, lr_lambda=lr_fn)
    adamw_sched = LambdaLR(adamw_optimizer, lr_lambda=lr_fn)

    return muon_sched, adamw_sched

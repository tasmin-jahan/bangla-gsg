"""
Hybrid Muon + AdamW optimizer factory for BanglaGSG.

Uses torch.optim.Muon (PyTorch 2.9+) for all large 2D dense matmul weights
and torch.optim.AdamW for everything else (embeddings, norms, 1D params).

Parameter routing:
  - Muon: SWA/GQA attention projections, FFN projections, GDN projections
  - AdamW: embeddings, norms, biases, 1D params, GDN internal states
"""

from typing import Dict, Any, Tuple, List

import yaml
import torch
import torch.nn as nn


# ── Muon parameter name substrings ────────────────────────────────────────
# All large 2D dense matmul weights that should use Muon
MUON_NAME_SUBSTRINGS = (
    # GDN projections (flash-linear-attention naming)
    "q_proj.weight", "k_proj.weight",
    "v_proj.weight", "o_proj.weight",
    "g_proj.weight",                          # GDN gate projection
    # SWA / GQA attention projections
    # (q_proj, k_proj, v_proj, o_proj already covered above)
    # FFN projections
    "gate_proj.weight", "up_proj.weight",
    "down_proj.weight",
)

# AdamW no-decay patterns
NO_DECAY_PATTERNS = ("bias", "norm", "ln", "embed")


def build_param_groups(model: nn.Module) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Split model parameters into Muon and AdamW groups.

    Muon group: all large 2D dense matmul weights (attention, FFN, GDN projections).
    AdamW group: everything else (embeddings, norms, 1D params, biases).

    Returns:
        (muon_params, adamw_params) — lists of parameter tensors.
    """
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(s in name for s in MUON_NAME_SUBSTRINGS) and p.ndim == 2:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    # Sanity check: embedding should NOT be in Muon group
    embed_in_muon = any(
        p.data_ptr() == model.embedding.weight.data_ptr() for p in muon_params
    )
    assert not embed_in_muon, (
        "BUG: embedding weight landed in Muon group! "
        "The name-substring filter should have excluded it."
    )

    n_muon = sum(p.numel() for p in muon_params)
    n_adamw = sum(p.numel() for p in adamw_params)
    print(f"[Optimizer] Muon params: {n_muon:,} ({n_muon/1e6:.1f}M)")
    print(f"[Optimizer] AdamW params: {n_adamw:,} ({n_adamw/1e6:.1f}M)")

    return muon_params, adamw_params


def build_optimizers(
    model: nn.Module,
    config: Dict[str, Any],
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """
    Build hybrid Muon + AdamW optimizers from config.

    Args:
        model: The model to optimize.
        config: Dict with 'muon' and 'adamw' sub-dicts.

    Returns:
        (muon_optimizer, adamw_optimizer)
    """
    muon_cfg = config.get("muon", {})
    adamw_cfg = config.get("adamw", {})

    muon_params, adamw_params = build_param_groups(model)

    # Muon optimizer (torch.optim.Muon, available in PyTorch 2.9+)
    muon_optimizer = torch.optim.Muon(
        muon_params,
        lr=muon_cfg.get("lr", 0.02),
        momentum=muon_cfg.get("momentum", 0.95),
        nesterov=muon_cfg.get("nesterov", True),
        ns_steps=muon_cfg.get("ns_steps", 5),
        weight_decay=muon_cfg.get("weight_decay", 0.01),
    )

    # AdamW optimizer — split into decay and no-decay groups
    decay_params = []
    no_decay_params = []

    # We need names for AdamW params to decide decay vs no-decay
    adamw_param_set = set(id(p) for p in adamw_params)
    for name, p in model.named_parameters():
        if id(p) not in adamw_param_set:
            continue
        if any(pat in name for pat in NO_DECAY_PATTERNS):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    adamw_param_groups = [
        {"params": decay_params, "weight_decay": adamw_cfg.get("weight_decay", 0.1)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    adamw_optimizer = torch.optim.AdamW(
        adamw_param_groups,
        lr=adamw_cfg.get("lr", 3e-4),
        betas=tuple(adamw_cfg.get("betas", [0.9, 0.95])),
        eps=adamw_cfg.get("eps", 1e-8),
        fused=adamw_cfg.get("fused", True),
    )

    return muon_optimizer, adamw_optimizer


def load_optimizer_config(path: str) -> Dict[str, Any]:
    """Load optimizer config from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

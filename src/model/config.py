"""
BanglaGSG Model Configuration.

Central dataclass for all model hyperparameters, loaded from YAML files.
Designed for a GDN:SWA:GQA 1:1:1 interleaved hybrid Bangla foundation model.

Layer pattern is specified directly in the YAML as a list:
    layer_types: [gdn, swa, gqa, gdn, swa, gqa, ...]

Components:
    GDN  — Gated DeltaNet  (linear-time recurrent, long-range memory)
    SWA  — Sliding Window Attention (local context, O(T·W))
    GQA  — Grouped Query Attention  (global context, full causal)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


# Default layer pattern: 12 layers, 1:1:1 interleaved
# G S A G S A G S A G S A
DEFAULT_LAYER_TYPES = [
    "gdn", "swa", "gqa",
    "gdn", "swa", "gqa",
    "gdn", "swa", "gqa",
    "gdn", "swa", "gqa",
]


@dataclass
class BanglaGSGConfig:
    """
    Complete model configuration for the BanglaGSG hybrid GDN/SWA/GQA LM.

    All fields are YAML-configurable. Use `BanglaGSGConfig.from_yaml(path)` to load.
    """

    # ── Core architecture ─────────────────────────────────────────────────
    d_model: int = 1024
    n_layers: int = 12
    n_heads: int = 16           # query heads (for SWA and GQA)
    n_kv_heads: int = 4         # KV heads for GQA/SWA (n_heads must be divisible by n_kv_heads)
    d_head: int = 64            # head dimension
    d_ff: int = 2560            # SwiGLU intermediate: floor(2/3 * 4 * 1024 / 256) * 256
    vocab_size: int = 48000
    seq_len: int = 2048
    dropout: float = 0.0
    bias: bool = False          # bias in linear layers

    # ── Layer pattern (explicit, no algorithm) ────────────────────────────
    # G S A G S A G S A G S A  (GDN first, GQA last in each triplet)
    layer_types: List[str] = field(default_factory=lambda: list(DEFAULT_LAYER_TYPES))

    # ── GDN (Gated DeltaNet) specific ────────────────────────────────────
    gdn_num_heads: int = 4          # number of GDN heads (key_dim = num_heads * head_dim)
    gdn_head_dim: int = 256         # GDN per-head key dimension
    gdn_expand_v: float = 1.0       # value expansion (head_v_dim = head_dim * expand_v)
    gdn_use_short_conv: bool = True # short causal conv before gating
    gdn_conv_size: int = 4          # short-conv kernel width

    # ── SWA (Sliding Window Attention) specific ──────────────────────────
    swa_window_size: int = 512      # one-sided window size (tokens to the left)

    # ── RoPE (for SWA and GQA layers) ────────────────────────────────────
    rope_base: float = 10000.0

    # ── Norm ──────────────────────────────────────────────────────────────
    rms_norm_eps: float = 1e-5

    # ── QK-Norm (stability with Muon optimizer, for SWA and GQA) ─────────
    qk_norm: bool = True

    # ── Weight tying ──────────────────────────────────────────────────────
    tie_embeddings: bool = True

    def __post_init__(self):
        """Validate config."""
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        assert self.d_model == self.n_heads * self.d_head, (
            f"d_model ({self.d_model}) must equal n_heads ({self.n_heads}) * d_head ({self.d_head})"
        )
        assert len(self.layer_types) == self.n_layers, (
            f"layer_types has {len(self.layer_types)} entries but n_layers={self.n_layers}"
        )
        valid_types = {"gdn", "swa", "gqa"}
        invalid = [lt for lt in self.layer_types if lt not in valid_types]
        assert not invalid, (
            f"layer_types must only contain {valid_types}, got invalid: {invalid}"
        )

    @classmethod
    def from_yaml(cls, path: str) -> "BanglaGSGConfig":
        """Load config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str) -> None:
        """Save config to a YAML file."""
        from dataclasses import asdict
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @property
    def n_params_estimate(self) -> int:
        """Rough parameter count estimate (for logging, not exact)."""
        embed = self.vocab_size * self.d_model
        ffn_per_layer = 3 * self.d_model * self.d_ff

        attn_per_layer = (
            self.d_model * self.d_model      # Q
            + self.d_model * (self.n_kv_heads * self.d_head)  # K
            + self.d_model * (self.n_kv_heads * self.d_head)  # V
            + self.d_model * self.d_model     # O
        )

        # GDN: rough estimate based on key_dim and value_dim
        gdn_key_dim = self.gdn_num_heads * self.gdn_head_dim
        gdn_value_dim = int(self.gdn_num_heads * self.gdn_head_dim * self.gdn_expand_v)
        gdn_per_layer = (
            self.d_model * gdn_key_dim * 2    # q_proj + k_proj
            + self.d_model * gdn_value_dim     # v_proj
            + gdn_value_dim * self.d_model     # o_proj
            + self.d_model * self.gdn_num_heads * 2  # a_proj + b_proj
            + self.d_model * (gdn_key_dim + gdn_value_dim)  # gate proj (approx)
        )

        total = embed  # embedding (tied, count once)
        for lt in self.layer_types:
            total += ffn_per_layer
            if lt in ("swa", "gqa"):
                total += attn_per_layer
            else:  # gdn
                total += gdn_per_layer

        return total

    def summary(self) -> str:
        """Return a human-readable summary of the config."""
        n_gdn = sum(1 for t in self.layer_types if t == "gdn")
        n_swa = sum(1 for t in self.layer_types if t == "swa")
        n_gqa = sum(1 for t in self.layer_types if t == "gqa")
        lines = [
            f"BanglaGSG Config Summary",
            f"  d_model={self.d_model}, n_layers={self.n_layers}",
            f"  n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}, d_head={self.d_head}",
            f"  d_ff={self.d_ff}, vocab={self.vocab_size}, seq_len={self.seq_len}",
            f"  Layers: {n_gdn} GDN + {n_swa} SWA + {n_gqa} GQA",
            f"  Pattern: {' '.join({'gdn':'G','swa':'S','gqa':'A'}[t] for t in self.layer_types)}",
            f"  GDN: heads={self.gdn_num_heads}, head_dim={self.gdn_head_dim}, expand_v={self.gdn_expand_v}, conv={self.gdn_conv_size}",
            f"  SWA: window_size={self.swa_window_size}",
            f"  QK-Norm: {self.qk_norm}",
            f"  Est. params: {self.n_params_estimate / 1e6:.1f}M",
        ]
        return "\n".join(lines)

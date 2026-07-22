from __future__ import annotations
from .configuration_banglagsg import BanglaGSGConfig
from dataclasses import dataclass, field
from fla.layers.gated_deltanet import GatedDeltaNet
try:
    import importlib
    flash_attn = importlib.import_module("flash_attn")
    flash_attn_func = flash_attn.flash_attn_func
except ImportError:
    flash_attn_func = None

from pathlib import Path
from torch import nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
from typing import List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

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

# Default layer pattern: 12 layers, 1:1:1 interleaved
# G S A G S A G S A G S A
DEFAULT_LAYER_TYPES = [
    "gdn",
    "swa",
    "gqa",
    "gdn",
    "swa",
    "gqa",
    "gdn",
    "swa",
    "gqa",
    "gdn",
    "swa",
    "gqa",
]


@dataclass
class RawConfig:
    """
    Complete model configuration for the BanglaGSG hybrid GDN/SWA/GQA LM.

    All fields are YAML-configurable. Use `RawConfig.from_yaml(path)` to load.
    """

    # ── Core architecture ─────────────────────────────────────────────────
    d_model: int = 1024
    n_layers: int = 12
    n_heads: int = 16  # query heads (for SWA and GQA)
    n_kv_heads: int = (
        4  # KV heads for GQA/SWA (n_heads must be divisible by n_kv_heads)
    )
    d_head: int = 64  # head dimension
    d_ff: int = 2560  # SwiGLU intermediate: floor(2/3 * 4 * 1024 / 256) * 256
    vocab_size: int = 48000
    seq_len: int = 2048
    dropout: float = 0.0
    bias: bool = False  # bias in linear layers

    # ── Layer pattern (explicit, no algorithm) ────────────────────────────
    # G S A G S A G S A G S A  (GDN first, GQA last in each triplet)
    layer_types: List[str] = field(default_factory=lambda: list(DEFAULT_LAYER_TYPES))

    # ── GDN (Gated DeltaNet) specific ────────────────────────────────────
    gdn_num_heads: int = 4  # number of GDN heads (key_dim = num_heads * head_dim)
    gdn_head_dim: int = 256  # GDN per-head key dimension
    gdn_expand_v: float = 1.0  # value expansion (head_v_dim = head_dim * expand_v)
    gdn_use_short_conv: bool = True  # short causal conv before gating
    gdn_conv_size: int = 4  # short-conv kernel width

    # ── SWA (Sliding Window Attention) specific ──────────────────────────
    swa_window_size: int = 512  # one-sided window size (tokens to the left)

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
        assert (
            self.n_heads % self.n_kv_heads == 0
        ), f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        assert (
            self.d_model == self.n_heads * self.d_head
        ), f"d_model ({self.d_model}) must equal n_heads ({self.n_heads}) * d_head ({self.d_head})"
        assert (
            len(self.layer_types) == self.n_layers
        ), f"layer_types has {len(self.layer_types)} entries but n_layers={self.n_layers}"
        valid_types = {"gdn", "swa", "gqa"}
        invalid = [lt for lt in self.layer_types if lt not in valid_types]
        assert (
            not invalid
        ), f"layer_types must only contain {valid_types}, got invalid: {invalid}"

    @classmethod
    def from_yaml(cls, path: str) -> "RawConfig":
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
            self.d_model * self.d_model  # Q
            + self.d_model * (self.n_kv_heads * self.d_head)  # K
            + self.d_model * (self.n_kv_heads * self.d_head)  # V
            + self.d_model * self.d_model  # O
        )

        # GDN: rough estimate based on key_dim and value_dim
        gdn_key_dim = self.gdn_num_heads * self.gdn_head_dim
        gdn_value_dim = int(self.gdn_num_heads * self.gdn_head_dim * self.gdn_expand_v)
        gdn_per_layer = (
            self.d_model * gdn_key_dim * 2  # q_proj + k_proj
            + self.d_model * gdn_value_dim  # v_proj
            + gdn_value_dim * self.d_model  # o_proj
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


"""
Token embeddings and RMSNorm for BanglaGSG.

RMSNorm: Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
TokenEmbedding: Embedding layer with weight-tying support for the LM head.

Embedding init uses std = 1/sqrt(d_model) per BanglaGSG spec §2 to keep
activation scale consistent with RMSNorm at d_model=1024.
"""


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for stability, cast back to input dtype
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f * rms).to(x.dtype) * self.weight


class PerHeadRMSNorm(nn.Module):
    """
    Per-head RMSNorm for QK-Norm (spec §6.1).

    Applied independently to each attention head's Q or K before RoPE.
    Learnable per-head scale parameter.
    """

    def __init__(self, d_head: int, n_heads: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Learnable scale: one scalar per head dimension, shared across heads
        self.weight = nn.Parameter(torch.ones(d_head))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, d_head)
        Returns:
            (B, T, H, d_head)
        """
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f * rms).to(x.dtype) * self.weight


class TokenEmbedding(nn.Module):
    """
    Token embedding layer with optional dropout.

    The embedding weight can be shared with the output LM head
    (weight tying) to save parameters — configured in BanglaGSGModel.

    Init std = 1/sqrt(d_model) per spec §2 (≈0.03125 for d_model=1024).
    """

    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Custom initialization: std = 1/sqrt(d_model)
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

    @property
    def weight(self) -> torch.Tensor:
        """Expose embedding weight for weight tying with LM head."""
        return self.embed.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, T) int64 tensor of token IDs.
        Returns:
            (B, T, d_model) float tensor of token embeddings.
        """
        return self.dropout(self.embed(token_ids))


"""
Rotary Position Embedding (RoPE) for BanglaGSG.

Standard RoPE (Su et al., 2022) with float32 angle computation.
Compatible with GQA — broadcasts over KV heads.

Applied ONLY to Q and K inside SWA and GQA attention layers.
NOT applied to GDN blocks (delta-rule recurrence encodes position implicitly).
"""


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotary half-swap: (x1, x2) → (-x2, x1) on the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Standard Rotary Position Embedding.

    Computes angles in float32 for numerical stability (standard practice
    in LLaMA, Mistral, Gemma), then casts cos/sin to the model dtype.

    Parameters
    ----------
    d_head : int
        Per-head dimension. Must be even.
    max_seq_len : int
        Maximum sequence length (for sanity checks only; not a hard limit).
    base : float
        RoPE frequency base. Default 10000.0.
    """

    def __init__(self, d_head: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        assert d_head % 2 == 0, f"d_head must be even, got {d_head}"
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(
        self,
        q: torch.Tensor,  # (B, T, H, d_head)
        k: torch.Tensor,  # (B, T, Hkv, d_head)
        positions: torch.Tensor,  # (B, T) int64
    ) -> tuple:
        """
        Apply RoPE rotations to queries and keys.

        Returns rotated (q, k) with the same shapes and dtype as inputs.
        """
        dtype = q.dtype
        device = q.device

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.d_head, 2, dtype=torch.float32, device=device) / self.d_head)
        )

        # Compute angles in float32 for precision
        pos_f = positions.float().unsqueeze(-1)  # (B, T, 1)
        freqs = pos_f * inv_freq  # (B, T, d_head//2)

        # Duplicate for the rotate_half trick
        emb = torch.cat([freqs, freqs], dim=-1)  # (B, T, d_head)

        # Cast to model dtype after computing cos/sin in float32
        cos = emb.cos().to(dtype=dtype).unsqueeze(2)  # (B, T, 1, d_head)
        sin = emb.sin().to(dtype=dtype).unsqueeze(2)  # (B, T, 1, d_head)

        # Apply rotation — broadcasts over head dimension
        q_rot = q * cos + _rotate_half(q) * sin  # (B, T, H,   d_head)
        k_rot = k * cos + _rotate_half(k) * sin  # (B, T, Hkv, d_head)

        return q_rot, k_rot


"""
SwiGLU Feed-Forward Network for BanglaGSG.

SwiGLU (Shazeer, 2020) replaces the standard GELU FFN with a gated
linear unit using SiLU activation:
    out = down_proj(SiLU(gate_proj(x)) * up_proj(x))

Spec §1.2: intermediate_size = floor(2/3 * 4 * d_model / 256) * 256 = 2560
"""


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Parameters
    ----------
    d_model : int
        Model dimension.
    d_ff : int
        Intermediate (hidden) dimension of the FFN.
    bias : bool
        Whether to use bias in linear layers. Default False.
    dropout : float
        Dropout after the activation. Default 0.0.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


"""
Grouped Query Attention (GQA) with QK-Norm for BanglaGSG.

Full causal attention — no window restriction. Provides global context
on every 3rd layer in the GDN:SWA:GQA 1:1:1 interleaved pattern.

Uses Flash Attention 2 (flash_attn_func) for hardware-efficient
attention. Includes per-head QK-Norm for stability with the Muon
optimizer.

QK-Norm: RMSNorm applied to Q and K projections per-head, BEFORE RoPE.
This bounds attention logit magnitudes regardless of upstream weight scale
drift — cheap insurance for a single-shot training run.
"""


class GQAttention(nn.Module):
    """
    Grouped Query Attention with QK-Norm, RoPE, and full causal masking.

    Uses Flash Attention 2 for efficient computation. No window
    restriction — attends to the full causal context.

    Parameters
    ----------
    config : RawConfig
        Model configuration.
    layer_idx : int
        Layer index (for potential per-layer modifications).
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_head
        self.layer_idx = layer_idx
        self.qk_norm = config.qk_norm

        self.q_proj = nn.Linear(
            config.d_model, config.n_heads * config.d_head, bias=config.bias
        )
        self.k_proj = nn.Linear(
            config.d_model, config.n_kv_heads * config.d_head, bias=config.bias
        )
        self.v_proj = nn.Linear(
            config.d_model, config.n_kv_heads * config.d_head, bias=config.bias
        )
        self.o_proj = nn.Linear(
            config.n_heads * config.d_head, config.d_model, bias=config.bias
        )

        # QK-Norm: per-head RMSNorm on Q and K before RoPE
        if self.qk_norm:
            self.q_norm = PerHeadRMSNorm(
                config.d_head, config.n_heads, eps=config.rms_norm_eps
            )
            self.k_norm = PerHeadRMSNorm(
                config.d_head, config.n_kv_heads, eps=config.rms_norm_eps
            )

    def forward(
        self,
        x: torch.Tensor,  # (B, T, d_model)
        positions: torch.Tensor,  # (B, T) int64
        rope: RotaryEmbedding,  # RoPE module
        past_key_value: tuple = None,
        use_cache: bool = False,
    ):
        B, T, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head)

        # QK-Norm: per-head RMSNorm BEFORE RoPE
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE to Q and K only
        q, k = rope(q, k, positions)

        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        if use_cache:
            new_past_key_value = (k, v)

        # Flash Attention 2 — full causal (no window restriction)
        if flash_attn_func is not None:
            # flash_attn_func expects (B, T, H, D) layout
            attn_output = flash_attn_func(
                q.to(torch.bfloat16),
                k,
                v,
                causal=True,
            )
        else:
            # PyTorch SDPA fallback
            q = q.transpose(1, 2)  # (B, H, T, d_head)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn_output = F.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )
            attn_output = attn_output.transpose(1, 2)  # (B, T, H, d_head)

        # (B, T, n_heads, d_head) -> (B, T, n_heads * d_head)
        attn_output = attn_output.contiguous().view(B, T, -1)

        out = self.o_proj(attn_output)

        if use_cache:
            return out, new_past_key_value
        return out


"""
Sliding Window Attention (SWA) with QK-Norm for BanglaGSG.

SWA is structurally identical to GQA but restricts each token's
attention to a local window of size `swa_window_size`. This provides
efficient O(T·W) local context modelling that complements:
  - GDN: long-range recurrent memory (linear time)
  - GQA: full quadratic global attention (on 1/3 of layers)

Uses Flash Attention 2's `window_size` argument for hardware-efficient
sliding window masking — zero overhead vs. full attention.

SWA layers use their own RoPE positions (shared base with GQA).
"""


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention with QK-Norm and RoPE.

    Identical projection structure to GQA, but attention is restricted
    to a local window of `window_size` tokens on each side.

    Parameters
    ----------
    config : RawConfig
        Model configuration.
    layer_idx : int
        Layer index (for potential per-layer modifications).
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_head
        self.layer_idx = layer_idx
        self.qk_norm = config.qk_norm
        self.window_size = config.swa_window_size

        self.q_proj = nn.Linear(
            config.d_model, config.n_heads * config.d_head, bias=config.bias
        )
        self.k_proj = nn.Linear(
            config.d_model, config.n_kv_heads * config.d_head, bias=config.bias
        )
        self.v_proj = nn.Linear(
            config.d_model, config.n_kv_heads * config.d_head, bias=config.bias
        )
        self.o_proj = nn.Linear(
            config.n_heads * config.d_head, config.d_model, bias=config.bias
        )

        # QK-Norm: per-head RMSNorm on Q and K before RoPE
        if self.qk_norm:
            self.q_norm = PerHeadRMSNorm(
                config.d_head, config.n_heads, eps=config.rms_norm_eps
            )
            self.k_norm = PerHeadRMSNorm(
                config.d_head, config.n_kv_heads, eps=config.rms_norm_eps
            )

    def forward(
        self,
        x: torch.Tensor,  # (B, T, d_model)
        positions: torch.Tensor,  # (B, T) int64
        rope: RotaryEmbedding,  # RoPE module
        past_key_value: tuple = None,
        use_cache: bool = False,
    ):
        B, T, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head)

        # QK-Norm: per-head RMSNorm BEFORE RoPE
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE to Q and K only
        q, k = rope(q, k, positions)

        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat(
                [past_k, k], dim=1
            )  # concatenate along the T (sequence) dimension
            v = torch.cat([past_v, v], dim=1)

        if use_cache:
            # Truncate to the most recent window_size tokens — anything older
            # is outside the attention window and irrelevant to future steps.
            if k.shape[1] > self.window_size:
                k_cache = k[:, -self.window_size :]
                v_cache = v[:, -self.window_size :]
            else:
                k_cache = k
                v_cache = v
            new_past_key_value = (k_cache, v_cache)

        # Flash Attention 2 with sliding window
        if flash_attn_func is not None:
            # flash_attn_func expects (B, T, H, D) layout
            attn_output = flash_attn_func(
                q.to(torch.bfloat16),
                k,
                v,
                causal=True,
                window_size=(
                    self.window_size,
                    0,
                ),  # (left_window, right_window=0 for causal)
            )
        else:
            # PyTorch SDPA fallback
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            T_q, T_k = q.size(2), k.size(2)
            
            if T_q == 1:
                # During generation, KV cache is already truncated to window_size
                attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                # During prefill or batched generation, create sliding window causal mask
                past_len = T_k - T_q
                i_idx = torch.arange(T_q, device=q.device).unsqueeze(1)
                j_idx = torch.arange(T_k, device=q.device).unsqueeze(0)
                
                # mask[i, j] is True if j <= past_len + i (causal) 
                # and j >= past_len + i - window_size + 1 (sliding window)
                mask = (j_idx <= i_idx + past_len) & (j_idx >= i_idx + past_len - self.window_size + 1)
                attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.view(1, 1, T_q, T_k))
            
            attn_output = attn_output.transpose(1, 2)

        # (B, T, n_heads, d_head) -> (B, T, n_heads * d_head)
        attn_output = attn_output.contiguous().view(B, T, -1)

        out = self.o_proj(attn_output)

        if use_cache:
            return out, new_past_key_value
        return out


"""
Gated DeltaNet (GDN) wrapper for BanglaGSG.

Thin wrapper around the GatedDeltaNet module from flash-linear-attention.
GDN is a linear-time recurrent layer that replaces SSM blocks (Mamba)
in the hybrid architecture. It encodes position implicitly via its
gated delta-rule recurrence — no RoPE is applied inside GDN blocks.

Reference: Yang et al., "Gated Delta Networks: Improving Mamba2 with
Delta Rule", 2025.
"""


class GDNBlock(nn.Module):
    """
    Wrapper around the GatedDeltaNet layer from flash-linear-attention.

    GatedDeltaNet API (fla-org/flash-linear-attention):
        hidden_size : int   — input/output dimension (our d_model)
        num_heads   : int   — number of attention heads
        head_dim    : int   — per-head key dimension
        expand_v    : float — value expansion factor (head_v_dim = head_dim * expand_v)
        use_short_conv : bool — short causal conv pre-mixing
        conv_size   : int   — conv kernel width

    Parameters
    ----------
    config : RawConfig
        Model configuration.
    layer_idx : int
        Layer index within the full model stack.
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.gdn = GatedDeltaNet(
            hidden_size=config.d_model,
            num_heads=config.gdn_num_heads,
            head_dim=config.gdn_head_dim,
            expand_v=config.gdn_expand_v,
            use_short_conv=config.gdn_use_short_conv,
            conv_size=config.gdn_conv_size,
            layer_idx=layer_idx,
        )

    def forward(
        self, x: torch.Tensor, past_key_values=None, use_cache: bool = False, **kwargs
    ):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        out, _, past_key_values = self.gdn(
            x,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if use_cache:
            return out, past_key_values
        return out


"""
BanglaGSG Hybrid GDN / SWA / GQA Language Model.

Builds a heterogeneous layer stack from RawConfig.layer_types:
  - GDN layers: RMSNorm → GatedDeltaNet → Residual → RMSNorm → SwiGLU → Residual
  - SWA layers: RMSNorm → SlidingWindowAttn (w/ QK-Norm) → Residual → RMSNorm → SwiGLU → Residual
  - GQA layers: RMSNorm → FullCausalGQA (w/ QK-Norm) → Residual → RMSNorm → SwiGLU → Residual

Architecture rationale:
  - GDN provides linear-time recurrent long-range memory (replaces SSM/Mamba)
  - SWA provides efficient local context modelling (O(T·W))
  - GQA provides full global attention for high-fidelity information routing

Stability features:
  - QK-Norm: per-head RMSNorm on Q/K before RoPE (SWA + GQA layers)
  - Residual init scaling: output projections scaled by 1/sqrt(2 * n_layers)
  - Embedding init: std = 1/sqrt(d_model)
"""


class BanglaGSGBlock(nn.Module):
    """
    Single transformer block — GDN, SWA, or GQA.
    All variants include a SwiGLU FFN sublayer (following Jamba/Zamba block shape).

    Each block = [Mixer (GDN OR SWA OR GQA)] + [SwiGLU FFN]
    Both pre-normed with RMSNorm, both with residual connections.
    """

    def __init__(self, config: RawConfig, layer_idx: int, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.layer_idx = layer_idx
        self.gradient_checkpointing = False

        # Pre-norm before mixer
        self.norm1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # Mixer: GDN, SWA, or GQA
        if layer_type == "gdn":
            self.mixer = GDNBlock(config, layer_idx)
        elif layer_type == "swa":
            self.mixer = SlidingWindowAttention(config, layer_idx)
        elif layer_type == "gqa":
            self.mixer = GQAttention(config, layer_idx)
        else:
            raise ValueError(f"Unknown layer_type: {layer_type}")

        # Pre-norm before FFN
        self.norm2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # FFN: SwiGLU
        self.ffn = SwiGLU(
            config.d_model, config.d_ff, bias=config.bias, dropout=config.dropout
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, T, d_model)
        positions: torch.Tensor = None,  # (B, T) int64 — needed for swa/gqa
        rope: RotaryEmbedding = None,  # RoPE module — needed for swa/gqa
        past_key_values=None,
        use_cache: bool = False,
    ):
        # ── Mixer ────────────────────────────────────────────────────────
        h = self.norm1(x)
        if self.layer_type in ("swa", "gqa"):
            if use_cache:
                h, past_key_values = self.mixer(
                    h,
                    positions=positions,
                    rope=rope,
                    past_key_value=past_key_values,
                    use_cache=True,
                )
            else:
                h = self.mixer(h, positions=positions, rope=rope)
        else:  # gdn
            if use_cache:
                h, past_key_values = self.mixer(
                    h, past_key_values=past_key_values, use_cache=True
                )
            else:
                h = self.mixer(h)
        x = x + h  # residual

        # ── FFN ──────────────────────────────────────────────────────────
        h = self.norm2(x)
        if self.gradient_checkpointing and self.training:
            h = grad_checkpoint(self.ffn, h, use_reentrant=False)
        else:
            h = self.ffn(h)
        x = x + h  # residual

        if use_cache:
            return x, past_key_values
        return x


class BanglaGSGModel(nn.Module):
    """
    BanglaGSG: Hybrid GDN / SWA / GQA Language Model.

    Architecture:
        Token Embedding → [BanglaGSGBlock × n_layers] → RMSNorm → LM Head

    The layer stack is heterogeneous: each layer is one of:
      - GDN  (Gated DeltaNet — linear-time recurrent)
      - SWA  (Sliding Window Attention — local context)
      - GQA  (Grouped Query Attention — global context)
    as specified by config.layer_types. All block types include a SwiGLU FFN sublayer.

    Weight tying: the LM head shares weights with the token embedding.

    Init: residual-branch output projections (out_proj, o_proj,
    down_proj) are scaled by 1/sqrt(2 * n_layers) to prevent activation
    variance growth with depth.
    """

    def __init__(self, config: RawConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        # Token embedding
        self.embedding = TokenEmbedding(
            config.vocab_size, config.d_model, dropout=config.dropout
        )

        # Positional encoding: standard RoPE (used by SWA and GQA layers)
        self.rope = RotaryEmbedding(
            d_head=config.d_head,
            max_seq_len=config.seq_len,
            base=config.rope_base,
        )

        # Heterogeneous layer stack
        self.layers = nn.ModuleList(
            [
                BanglaGSGBlock(config, layer_idx=i, layer_type=lt)
                for i, lt in enumerate(config.layer_types)
            ]
        )

        # Final norm
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # LM head (output projection)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights per spec:
        - General 2D weights: std=0.02 (except GDN internals which self-init)
        - Embedding: std = 1/sqrt(d_model) (handled by TokenEmbedding.__init__)
        - Residual output projections: scaled by 1/sqrt(2 * n_layers)
        """
        # Standard init for non-GDN 2D weights
        for name, p in self.named_parameters():
            if p.dim() > 1 and "gdn" not in name and "embed" not in name:
                nn.init.normal_(p, mean=0.0, std=0.02)

        # Residual-branch output scaling
        # Scale out_proj (GDN), o_proj (SWA/GQA), down_proj (FFN) by 1/sqrt(2*n_layers)
        scale = 1.0 / math.sqrt(2 * self.config.n_layers)
        for name, p in self.named_parameters():
            if name.endswith(("out_proj.weight", "o_proj.weight", "down_proj.weight")):
                p.data.mul_(scale)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for all blocks."""
        self.gradient_checkpointing = True
        for layer in self.layers:
            layer.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for all blocks."""
        self.gradient_checkpointing = False
        for layer in self.layers:
            layer.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, T) int64
        past_key_values=None,
        use_cache: bool = False,
        position_offset: int = 0,
    ):
        """
        Forward pass.

        Args:
            input_ids: (B, T) token IDs.

        Returns:
            logits: (B, T, vocab_size) raw logits (no softmax).
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.embedding(input_ids)  # (B, T, d_model)

        # Positions: offset + 0..T-1 for each sequence
        positions = (
            torch.arange(
                position_offset, position_offset + T, device=device, dtype=torch.long
            )
            .unsqueeze(0)
            .expand(B, -1)
        )

        gdn_cache = None
        swa_gqa_cache = None
        if use_cache:
            if past_key_values is None:
                from fla.models.utils import Cache

                past_key_values = {"gdn": Cache(), "swa_gqa": [None] * len(self.layers)}
            gdn_cache = past_key_values.get("gdn")
            if gdn_cache is None:
                from fla.models.utils import Cache

                gdn_cache = Cache()
            swa_gqa_cache = past_key_values.get("swa_gqa")
            if swa_gqa_cache is None:
                swa_gqa_cache = [None] * len(self.layers)

        # Pass through all layers
        for i, layer in enumerate(self.layers):
            if use_cache:
                if layer.layer_type in ("swa", "gqa"):
                    x, swa_gqa_cache[i] = layer(
                        x,
                        positions=positions,
                        rope=self.rope,
                        past_key_values=swa_gqa_cache[i],
                        use_cache=True,
                    )
                else:
                    x, gdn_cache = layer(
                        x,
                        positions=positions,
                        rope=self.rope,
                        past_key_values=gdn_cache,
                        use_cache=True,
                    )
            else:
                x = layer(x, positions=positions, rope=self.rope)

        # Final norm + LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)

        if use_cache:
            past_key_values = {"gdn": gdn_cache, "swa_gqa": swa_gqa_cache}
            return logits, past_key_values
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=50,
        eos_token_id=None,
        do_sample=False,
        temperature=1.0,
    ):
        """
        Single-sequence (batch_size=1) incremental generation using GDN/SWA/GQA caching.
        Greedy by default (do_sample=False). No batching support.
        """
        assert (
            input_ids.shape[0] == 1
        ), "generate() currently only supports batch_size=1"

        # Prefill: run the full prompt through once, building the initial cache
        logits, past_key_values = self.forward(
            input_ids, use_cache=True, position_offset=0
        )
        # last position's logits determine the first new token
        next_token_logits = logits[:, -1, :]

        generated = input_ids
        position = input_ids.shape[1]

        for _ in range(max_new_tokens):
            if do_sample:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            # Decode step: only the new token goes in, cache carries the rest
            logits, past_key_values = self.forward(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
                position_offset=position,
            )
            next_token_logits = logits[:, -1, :]
            position += 1

        return generated

    def count_parameters(self) -> dict:
        """Return parameter count breakdown."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        embed_params = sum(p.numel() for p in self.embedding.parameters())
        gdn_params = sum(
            p.numel() for name, p in self.named_parameters() if "gdn" in name.lower()
        )
        swa_params = sum(
            p.numel()
            for name, p in self.named_parameters()
            if any(k in name for k in ["swa", "sliding"])
            or (
                any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"])
                and "layers." in name
                and f".mixer." in name
                and self.config.layer_types[int(name.split("layers.")[1].split(".")[0])]
                == "swa"
            )
        )
        gqa_params = sum(
            p.numel()
            for name, p in self.named_parameters()
            if any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"])
            and "layers." in name
            and f".mixer." in name
            and self.config.layer_types[int(name.split("layers.")[1].split(".")[0])]
            == "gqa"
        )
        ffn_params = sum(
            p.numel()
            for name, p in self.named_parameters()
            if any(k in name for k in ["gate_proj", "up_proj", "down_proj"])
        )

        return {
            "total": total,
            "trainable": trainable,
            "embedding": embed_params,
            "gdn": gdn_params,
            "swa": swa_params,
            "gqa": gqa_params,
            "ffn": ffn_params,
        }


# HF wrapper: translates the HF-facing BanglaGSGConfig into the internal
# RawConfig dataclass and adapts BanglaGSGModel to the PreTrainedModel API.


class BanglaGSGForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = BanglaGSGConfig
    base_model_prefix = "model"
    _no_split_modules = [
        "BanglaGSGBlock",
        "GDNBlock",
        "SlidingWindowAttention",
        "GQAttention",
    ]

    def __init__(self, config: BanglaGSGConfig):
        super().__init__(config)

        # Translate HF config back to our raw dataclass config
        raw_config = RawConfig(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            d_head=config.d_head,
            d_ff=config.d_ff,
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            dropout=config.dropout,
            bias=config.bias,
            layer_types=config.layer_types,
            gdn_num_heads=config.gdn_num_heads,
            gdn_head_dim=config.gdn_head_dim,
            gdn_expand_v=config.gdn_expand_v,
            gdn_use_short_conv=config.gdn_use_short_conv,
            gdn_conv_size=config.gdn_conv_size,
            swa_window_size=config.swa_window_size,
            rope_base=config.rope_base,
            rms_norm_eps=config.rms_norm_eps,
            qk_norm=config.qk_norm,
            tie_embeddings=config.tie_embeddings,
        )

        # Initialize the raw model
        self.model = BanglaGSGModel(raw_config)

        # PreTrainedModel automatically calls post_init() here, which will
        # try to initialize weights. We let it pass because we will load
        # pre-trained weights immediately after.
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.LongTensor = None,
        **kwargs,
    ):
        """
        Forward pass for standard HF Causal LM tracking.
        """
        if attention_mask is not None:
            if not torch.all(attention_mask.bool()):
                raise NotImplementedError(
                    "BanglaGSG v1 does not support padded batches. All sequences in the batch "
                    "must be dense, unpadded, and of equal length. If you are evaluating "
                    "variable-length sequences, please process them one at a time (batch_size=1) "
                    "without padding."
                )

        logits = self.model(input_ids)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
        )
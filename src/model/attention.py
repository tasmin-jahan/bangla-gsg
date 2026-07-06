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

import torch
import torch.nn as nn

from flash_attn import flash_attn_func

from src.model.embeddings import PerHeadRMSNorm
from src.model.rope import RotaryEmbedding


class GQAttention(nn.Module):
    """
    Grouped Query Attention with QK-Norm, RoPE, and full causal masking.

    Uses Flash Attention 2 for efficient computation. No window
    restriction — attends to the full causal context.

    Parameters
    ----------
    config : BanglaGSGConfig
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

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=config.bias)
        self.o_proj = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=config.bias)

        # QK-Norm: per-head RMSNorm on Q and K before RoPE
        if self.qk_norm:
            self.q_norm = PerHeadRMSNorm(config.d_head, config.n_heads, eps=config.rms_norm_eps)
            self.k_norm = PerHeadRMSNorm(config.d_head, config.n_kv_heads, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,                       # (B, T, d_model)
        positions: torch.Tensor,                # (B, T) int64
        rope: RotaryEmbedding,                  # RoPE module
    ) -> torch.Tensor:
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

        # Flash Attention 2 — full causal (no window restriction)
        # flash_attn_func expects (B, T, H, D) layout — already correct
        attn_output = flash_attn_func(
            q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
            causal=True,
        )

        # (B, T, n_heads, d_head) -> (B, T, n_heads * d_head)
        attn_output = attn_output.contiguous().view(B, T, -1)

        return self.o_proj(attn_output)

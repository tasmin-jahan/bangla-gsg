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

import torch
import torch.nn as nn

from flash_attn import flash_attn_func

from .embeddings import PerHeadRMSNorm
from .rope import RotaryEmbedding


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention with QK-Norm and RoPE.

    Identical projection structure to GQA, but attention is restricted
    to a local window of `window_size` tokens on each side.

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
        self.window_size = config.swa_window_size

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
            k = torch.cat([past_k, k], dim=1)  # concatenate along the T (sequence) dimension
            v = torch.cat([past_v, v], dim=1)

        if use_cache:
            # Truncate to the most recent window_size tokens — anything older
            # is outside the attention window and irrelevant to future steps.
            if k.shape[1] > self.window_size:
                k_cache = k[:, -self.window_size:]
                v_cache = v[:, -self.window_size:]
            else:
                k_cache = k
                v_cache = v
            new_past_key_value = (k_cache, v_cache)

        # Flash Attention 2 with sliding window
        # flash_attn_func expects (B, T, H, D) layout — already correct
        attn_output = flash_attn_func(
            q.to(torch.bfloat16), k, v,
            causal=True,
            window_size=(self.window_size, 0),  # (left_window, right_window=0 for causal)
        )

        # (B, T, n_heads, d_head) -> (B, T, n_heads * d_head)
        attn_output = attn_output.contiguous().view(B, T, -1)

        out = self.o_proj(attn_output)

        if use_cache:
            return out, new_past_key_value
        return out

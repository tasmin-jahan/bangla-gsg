"""
Rotary Position Embedding (RoPE) for BanglaGSG.

Standard RoPE (Su et al., 2022) with float32 angle computation.
Compatible with GQA — broadcasts over KV heads.

Applied ONLY to Q and K inside SWA and GQA attention layers.
NOT applied to GDN blocks (delta-rule recurrence encodes position implicitly).
"""

import torch
import torch.nn as nn


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

        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
        )
        self.register_buffer("inv_freq", inv_freq)  # (d_head // 2,)

    def forward(
        self,
        q: torch.Tensor,          # (B, T, H, d_head)
        k: torch.Tensor,          # (B, T, Hkv, d_head)
        positions: torch.Tensor,  # (B, T) int64
    ) -> tuple:
        """
        Apply RoPE rotations to queries and keys.

        Returns rotated (q, k) with the same shapes and dtype as inputs.
        """
        dtype = q.dtype

        # Compute angles in float32 for precision
        pos_f = positions.float().unsqueeze(-1)             # (B, T, 1)
        freqs = pos_f * self.inv_freq.float()               # (B, T, d_head//2)

        # Duplicate for the rotate_half trick
        emb = torch.cat([freqs, freqs], dim=-1)             # (B, T, d_head)

        # Cast to model dtype after computing cos/sin in float32
        cos = emb.cos().to(dtype=dtype).unsqueeze(2)        # (B, T, 1, d_head)
        sin = emb.sin().to(dtype=dtype).unsqueeze(2)        # (B, T, 1, d_head)

        # Apply rotation — broadcasts over head dimension
        q_rot = q * cos + _rotate_half(q) * sin             # (B, T, H,   d_head)
        k_rot = k * cos + _rotate_half(k) * sin             # (B, T, Hkv, d_head)

        return q_rot, k_rot

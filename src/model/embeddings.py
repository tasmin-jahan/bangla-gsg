"""
Token embeddings and RMSNorm for BanglaGSG.

RMSNorm: Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
TokenEmbedding: Embedding layer with weight-tying support for the LM head.

Embedding init uses std = 1/sqrt(d_model) per BanglaFM spec §2 to keep
activation scale consistent with RMSNorm at d_model=1024.
"""

import math

import torch
import torch.nn as nn


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

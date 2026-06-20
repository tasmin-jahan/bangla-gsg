"""
SwiGLU Feed-Forward Network for BanglaGSG.

SwiGLU (Shazeer, 2020) replaces the standard GELU FFN with a gated
linear unit using SiLU activation:
    out = down_proj(SiLU(gate_proj(x)) * up_proj(x))

Spec §1.2: intermediate_size = floor(2/3 * 4 * d_model / 256) * 256 = 2560
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

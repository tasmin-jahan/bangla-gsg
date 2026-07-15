"""
Gated DeltaNet (GDN) wrapper for BanglaGSG.

Thin wrapper around the GatedDeltaNet module from flash-linear-attention.
GDN is a linear-time recurrent layer that replaces SSM blocks (Mamba)
in the hybrid architecture. It encodes position implicitly via its
gated delta-rule recurrence — no RoPE is applied inside GDN blocks.

Reference: Yang et al., "Gated Delta Networks: Improving Mamba2 with
Delta Rule", 2025.
"""

import torch
import torch.nn as nn

from fla.layers.gated_deltanet import GatedDeltaNet


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
    config : BanglaGSGConfig
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

    def forward(self, x: torch.Tensor, past_key_values=None, use_cache: bool = False, **kwargs):
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

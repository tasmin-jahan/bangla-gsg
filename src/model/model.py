"""
BanglaGSG Hybrid GDN / SWA / GQA Language Model.

Builds a heterogeneous layer stack from BanglaGSGConfig.layer_types:
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

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .config import BanglaGSGConfig
from .embeddings import RMSNorm, TokenEmbedding
from .attention import GQAttention
from .swa import SlidingWindowAttention
from .gdn import GDNBlock
from .ffn import SwiGLU
from .rope import RotaryEmbedding


class BanglaGSGBlock(nn.Module):
    """
    Single transformer block — GDN, SWA, or GQA.
    All variants include a SwiGLU FFN sublayer (following Jamba/Zamba block shape).

    Each block = [Mixer (GDN OR SWA OR GQA)] + [SwiGLU FFN]
    Both pre-normed with RMSNorm, both with residual connections.
    """

    def __init__(self, config: BanglaGSGConfig, layer_idx: int, layer_type: str):
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
        self.ffn = SwiGLU(config.d_model, config.d_ff, bias=config.bias, dropout=config.dropout)

    def forward(
        self,
        x: torch.Tensor,                       # (B, T, d_model)
        positions: torch.Tensor = None,         # (B, T) int64 — needed for swa/gqa
        rope: RotaryEmbedding = None,           # RoPE module — needed for swa/gqa
        past_key_values=None,
        use_cache: bool = False,
    ):
        # ── Mixer ────────────────────────────────────────────────────────
        h = self.norm1(x)
        if self.layer_type in ("swa", "gqa"):
            if use_cache:
                h, past_key_values = self.mixer(h, positions=positions, rope=rope, past_key_value=past_key_values, use_cache=True)
            else:
                h = self.mixer(h, positions=positions, rope=rope)
        else:  # gdn
            if use_cache:
                h, past_key_values = self.mixer(h, past_key_values=past_key_values, use_cache=True)
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

    def __init__(self, config: BanglaGSGConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        # Token embedding
        self.embedding = TokenEmbedding(config.vocab_size, config.d_model, dropout=config.dropout)

        # Positional encoding: standard RoPE (used by SWA and GQA layers)
        self.rope = RotaryEmbedding(
            d_head=config.d_head,
            max_seq_len=config.seq_len,
            base=config.rope_base,
        )

        # Heterogeneous layer stack
        self.layers = nn.ModuleList([
            BanglaGSGBlock(config, layer_idx=i, layer_type=lt)
            for i, lt in enumerate(config.layer_types)
        ])

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
        input_ids: torch.Tensor,                # (B, T) int64
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
        positions = torch.arange(position_offset, position_offset + T, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)

        gdn_cache = None
        swa_gqa_cache = None
        if use_cache:
            if past_key_values is None:
                from fla.models.utils import Cache
                past_key_values = {'gdn': Cache(), 'swa_gqa': [None] * len(self.layers)}
            gdn_cache = past_key_values.get('gdn')
            if gdn_cache is None:
                from fla.models.utils import Cache
                gdn_cache = Cache()
            swa_gqa_cache = past_key_values.get('swa_gqa')
            if swa_gqa_cache is None:
                swa_gqa_cache = [None] * len(self.layers)

        # Pass through all layers
        for i, layer in enumerate(self.layers):
            if use_cache:
                if layer.layer_type in ("swa", "gqa"):
                    x, swa_gqa_cache[i] = layer(
                        x, positions=positions, rope=self.rope, 
                        past_key_values=swa_gqa_cache[i], use_cache=True
                    )
                else:
                    x, gdn_cache = layer(
                        x, positions=positions, rope=self.rope, 
                        past_key_values=gdn_cache, use_cache=True
                    )
            else:
                x = layer(x, positions=positions, rope=self.rope)

        # Final norm + LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)

        if use_cache:
            past_key_values = {'gdn': gdn_cache, 'swa_gqa': swa_gqa_cache}
            return logits, past_key_values
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, eos_token_id=None, do_sample=False, temperature=1.0):
        """
        Single-sequence (batch_size=1) incremental generation using GDN/SWA/GQA caching.
        Greedy by default (do_sample=False). No batching support.
        """
        assert input_ids.shape[0] == 1, "generate() currently only supports batch_size=1"

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
                next_token, past_key_values=past_key_values, use_cache=True, position_offset=position
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
            p.numel() for name, p in self.named_parameters()
            if any(k in name for k in ["swa", "sliding"])
            or (any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"])
                and "layers." in name
                and f".mixer." in name
                and self.config.layer_types[int(name.split("layers.")[1].split(".")[0])] == "swa")
        )
        gqa_params = sum(
            p.numel() for name, p in self.named_parameters()
            if any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"])
            and "layers." in name
            and f".mixer." in name
            and self.config.layer_types[int(name.split("layers.")[1].split(".")[0])] == "gqa"
        )
        ffn_params = sum(
            p.numel() for name, p in self.named_parameters()
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

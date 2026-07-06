import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .configuration_banglagsg import BanglaGSGConfig
# Note: In the final HF repo, this will resolve to the 'model' directory in the root.
from .model.model import BanglaGSGModel
# We also need the raw config to pass into the model.
from .model.config import BanglaGSGConfig as RawConfig


class BanglaGSGForCausalLM(PreTrainedModel):
    config_class = BanglaGSGConfig
    _no_split_modules = ["BanglaGSGBlock", "GDNBlock", "SlidingWindowAttention", "GQAttention"]

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
        
        # PreTrainedModel automatically calls post_init() here, 
        # which will try to initialize weights. We let it pass because 
        # we will load pre-trained weights immediately after.
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.LongTensor = None,
        **kwargs
    ):
        """
        Forward pass for standard HF Causal LM tracking.
        Note: KV-caching (.generate()) is unsupported in v1.
        """
        # BanglaGSG currently does NOT support padded sequences natively due to GDN logic.
        # We must explicitly fail if a user passes an attention_mask containing padding (0s)
        # to prevent silent corruption of attention calculations.
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
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            
        return CausalLMOutput(
            loss=loss,
            logits=logits,
        )

from transformers import PretrainedConfig

class BanglaGSGConfig(PretrainedConfig):
    model_type = "banglagsg"

    def __init__(
        self,
        d_model=1024,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        d_head=64,
        d_ff=2560,
        vocab_size=48000,
        seq_len=2048,
        dropout=0.0,
        bias=False,
        layer_types=None,
        gdn_num_heads=4,
        gdn_head_dim=256,
        gdn_expand_v=1.0,
        gdn_use_short_conv=True,
        gdn_conv_size=4,
        swa_window_size=512,
        rope_base=10000.0,
        rms_norm_eps=1e-5,
        qk_norm=True,
        tie_embeddings=True,
        **kwargs,
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dropout = dropout
        self.bias = bias
        
        # Default layer pattern if none provided
        if layer_types is None:
            self.layer_types = [
                "gdn", "swa", "gqa",
                "gdn", "swa", "gqa",
                "gdn", "swa", "gqa",
                "gdn", "swa", "gqa",
            ]
        else:
            self.layer_types = layer_types
            
        self.gdn_num_heads = gdn_num_heads
        self.gdn_head_dim = gdn_head_dim
        self.gdn_expand_v = gdn_expand_v
        self.gdn_use_short_conv = gdn_use_short_conv
        self.gdn_conv_size = gdn_conv_size
        self.swa_window_size = swa_window_size
        self.rope_base = rope_base
        self.rms_norm_eps = rms_norm_eps
        self.qk_norm = qk_norm
        self.tie_embeddings = tie_embeddings
        
        # Super init handles kwargs drop-in, serialization, etc.
        super().__init__(**kwargs)

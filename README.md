# BanglaGSG

**GDN:SWA:GQA 1:1:1 Interleaved Hybrid Foundation Model for Bangla**

A heterogeneous-architecture language model combining three complementary mixer types in a repeating 1:1:1 interleaved pattern:

| Layer Type | Full Name | Mechanism | Complexity | Role |
|:---:|---|---|:---:|---|
| **G** | Gated DeltaNet (GDN) | Linear recurrent (delta-rule) | O(T) | Long-range memory |
| **S** | Sliding Window Attention (SWA) | Local attention (bounded window) | O(T·W) | Local context |
| **A** | Grouped Query Attention (GQA) | Full causal attention | O(T²) | Global information routing |

## Architecture

```
Layer  0 ─ GDN   ──┐
Layer  1 ─ SWA   ──│── Triplet 1
Layer  2 ─ GQA   ──┘
Layer  3 ─ GDN   ──┐
Layer  4 ─ SWA   ──│── Triplet 2
Layer  5 ─ GQA   ──┘
Layer  6 ─ GDN   ──┐
Layer  7 ─ SWA   ──│── Triplet 3
Layer  8 ─ GQA   ──┘
Layer  9 ─ GDN   ──┐
Layer 10 ─ SWA   ──│── Triplet 4
Layer 11 ─ GQA   ──┘
```

Each block follows the same Pre-Norm → Mixer → Residual → Pre-Norm → SwiGLU FFN → Residual structure, regardless of mixer type.

### Design Rationale

- **GDN** (Gated DeltaNet) replaces traditional SSM/Mamba blocks. It uses a gated delta-rule recurrence for O(T) long-range context compression. Position is encoded implicitly — no RoPE needed.
- **SWA** (Sliding Window Attention) provides efficient local context modelling within a fixed window (default 512 tokens). Uses Flash Attention 2 with the `window_size` parameter for zero-overhead sliding window masking.
- **GQA** (Grouped Query Attention) provides full quadratic causal attention on every 3rd layer for high-fidelity global information routing. Uses 4 KV heads with 16 query heads (4:1 GQA ratio).

This three-way decomposition ensures:
- **Efficiency**: 2/3 of layers are sub-quadratic (GDN + SWA)
- **Expressiveness**: Full global attention every 3 layers prevents information bottlenecks
- **Stability**: Complementary inductive biases reduce failure modes of any single mechanism

## Model Configuration (Default: 12L)

| Parameter | Value | Notes |
|---|:---:|---|
| `d_model` | 1024 | Hidden dimension |
| `n_layers` | 12 | 4 GDN + 4 SWA + 4 GQA |
| `n_heads` | 16 | Query heads (SWA/GQA) |
| `n_kv_heads` | 4 | KV heads (4:1 GQA ratio) |
| `d_head` | 64 | Per-head dimension |
| `d_ff` | 2560 | SwiGLU intermediate dim |
| `vocab_size` | 48000 | SentencePiece Unigram |
| `seq_len` | 2048 | Context length |
| **GDN** | | |
| `gdn_num_heads` | 4 | key_dim = 4 × 256 = 1024 |
| `gdn_head_dim` | 256 | Per-head key dimension |
| `gdn_expand_v` | 1.0 | Value expansion factor |
| `gdn_conv_size` | 4 | Short-conv kernel width |
| **SWA** | | |
| `swa_window_size` | 512 | One-sided window (tokens left) |
| **Total params** | ~186M | With weight tying |

## Project Structure

```
bangla-gsg/
├── configs/
│   ├── banglagsg_12l.yaml        # Model architecture config
│   ├── default_training.yaml     # Training hyperparameters
│   ├── default_data.yaml         # Data pipeline config
│   └── muon_adamw.yaml           # Optimizer config
├── src/
│   ├── model/
│   │   ├── config.py             # BanglaGSGConfig dataclass
│   │   ├── model.py              # BanglaGSGModel (full LM)
│   │   ├── gdn.py                # Gated DeltaNet wrapper
│   │   ├── swa.py                # Sliding Window Attention
│   │   ├── attention.py          # Grouped Query Attention (full)
│   │   ├── ffn.py                # SwiGLU feed-forward
│   │   ├── rope.py               # Rotary Position Embeddings
│   │   ├── embeddings.py         # Token embeddings + RMSNorm
│   │   └── optim.py              # Hybrid Muon + AdamW factory
│   ├── data/
│   │   ├── dataset.py            # Sharded NumPy dataset
│   │   └── collator.py           # DataLoader builder
│   ├── training/
│   │   ├── trainer.py            # Training loop + TrainerConfig
│   │   ├── scheduler.py          # Warmup + cosine decay LR
│   │   └── checkpoint.py         # Checkpoint save/load/manage
│   ├── tokenizer/                # SentencePiece tokenizer
│   ├── utils/
│   │   ├── logging.py            # CSV metric logger
│   │   └── seed.py               # Reproducibility seeds
│   └── train.py                  # Training entry point
├── scripts/                      # Data scraping / cleaning
├── tests/
│   └── test_smoke.py             # Architecture smoke test
├── docs/
│   └── literature_review.md
├── setup_venv.sh                 # Full environment setup
└── README.md
```

## Dependencies

The model requires three key CUDA libraries:

| Library | Purpose | Component |
|---|---|---|
| [`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) | GatedDeltaNet implementation | GDN layers |
| [`flash-attn`](https://github.com/Dao-AILab/flash-attention) ≥ 2.8.3 | Flash Attention 2 with `window_size` | SWA + GQA layers |
| [`causal-conv1d`](https://github.com/Dao-AILab/causal-conv1d) ≥ 1.4 | Short causal convolution | GDN pre-mixing |

### Full Setup

```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

This installs Python 3.12, CUDA 13.0 toolkit, PyTorch 2.12+cu130, and all CUDA extension libraries. See [`setup_venv.sh`](setup_venv.sh) for the full verified version matrix.

### Quick Verify

```bash
source .venv/bin/activate
python -c "
from fla.layers.gated_deltanet import GatedDeltaNet
from flash_attn import flash_attn_func
import inspect
sig = inspect.signature(flash_attn_func)
assert 'window_size' in sig.parameters
print('GDN: OK | SWA: OK | GQA: OK')
"
```

## Training

### Optimizer: Hybrid Muon + AdamW

| Group | Optimizer | Parameters | Purpose |
|---|---|---|---|
| Dense 2D weights | Muon | Attention projections, FFN, GDN projections | Fast convergence on matmul weights |
| Everything else | AdamW | Embeddings, norms, 1D params | Standard treatment for non-matmul params |

### Run Training

```bash
source .venv/bin/activate
python src/train.py
```

### Resume from Checkpoint

```bash
python src/train.py --resume
```

### Training Features

- **BF16 Autocast**: Mixed-precision training with bfloat16
- **Gradient Accumulation**: Effective batch size = `batch_size × accumulation_steps × seq_len` tokens
- **Z-Loss**: `1e-4 × logsumexp(logits)² .mean()` for logit stability
- **Gradient Clipping**: Global max_norm=1.0 across both optimizer groups
- **Gradient Checkpointing**: Trades compute for memory on FFN sublayers
- **torch.compile**: `reduce-overhead` mode for fused kernels
- **Cosine LR Decay**: Linear warmup (1.5%) → cosine decay to 10% of peak

## Testing

```bash
source .venv/bin/activate
python tests/test_smoke.py
```

The smoke test validates:
1. Config loads from YAML and validates layer types
2. Config rejects invalid layer types (e.g., `mamba`)
3. Model builds with correct GDN/SWA/GQA layer distribution
4. Forward pass produces correct output shape (no NaN/Inf)
5. Parameter counting across all component types
6. Gradient flow verified through all 12 layers
7. Gradient checkpointing enable/disable

## Stability Features

| Feature | Mechanism | Layers |
|---|---|---|
| QK-Norm | Per-head RMSNorm on Q/K before RoPE | SWA, GQA |
| Residual Scaling | Output projections × 1/√(2·n_layers) | All |
| Embedding Init | std = 1/√d_model | Embedding |
| Z-Loss | Penalizes logit magnitude growth | LM Head |
| Weight Tying | LM head shares embedding weights | Input ↔ Output |

## Data Pipeline

The data pipeline (in `scripts/`) handles:
1. **Corpus Collection**: Scraping Bangla text from multiple sources
2. **Cleaning**: Normalization, deduplication, language ID filtering
3. **Tokenization**: SentencePiece Unigram tokenizer training
4. **Packing**: Pretokenized `.npy` shards for efficient training

## Download The Model From [Here.](https://huggingface.co/tasmin-jahan/bangla-gsg)

## References

- Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule", 2025
- Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", 2023
- Beltagy et al., "Longformer: The Long-Document Transformer" (sliding window attention), 2020
- Shazeer, "GLU Variants Improve Transformer" (SwiGLU), 2020
- Jordan, "Muon: An Optimizer for Hidden Layers in Neural Networks", 2024

## License

MIT License. See [LICENSE](LICENSE).

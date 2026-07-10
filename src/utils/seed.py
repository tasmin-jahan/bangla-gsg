"""
Reproducibility seed utility for BanglaGSG.

Sets Python, NumPy, and PyTorch random seeds for reproducibility.

NOTE: We do NOT set torch.backends.cudnn.deterministic = True or
benchmark = False here. Those flags cripple training throughput
(~20-30% slower) and are unnecessary when using our epoch-seeded
RandomSampler for data ordering. The seed utility ensures model
weight initialization and data shuffling are reproducible — the
minor non-determinism in cuDNN convolution algorithm selection is
negligible and does not affect training outcomes in practice.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 1552):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Let cuDNN auto-tune for best performance (benchmark=True is default).
    # Deterministic mode is intentionally NOT set — it kills throughput
    # and our data ordering is already deterministic via epoch-seeded sampler.

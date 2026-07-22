"""
Reproducibility seed utility for the eval suite.

Sets Python, NumPy, and PyTorch random seeds for reproducibility.
Mirrors src/utils/seed.py from the main BanglaGamba project.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility across the eval suite."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

"""
Reproducibility seed utility for BanglaGSG.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility (spec §7)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

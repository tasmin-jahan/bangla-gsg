"""
Sharded NumPy dataset for BanglaGSG.

Loads pretokenized .npy shard files produced by the pretokenization pipeline.
Each shard is a 2D array of shape (N, seq_len) with dtype uint16.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional


class ShardedNpyDataset(Dataset):
    """
    Dataset backed by pretokenized .npy shards.

    Each shard file contains a 2D numpy array of shape (N_shard, seq_len)
    with dtype uint16. All shards are concatenated virtually via index
    mapping (no full load into RAM).

    Parameters
    ----------
    npy_dir : str
        Directory containing .npy shard files.
    max_shards : int
        Maximum number of shards to load (0 = all).
    """

    def __init__(self, npy_dir: str, max_shards: int = 0):
        self.npy_dir = Path(npy_dir)
        self.shard_paths = sorted(self.npy_dir.rglob("*.npy"))

        if max_shards > 0:
            self.shard_paths = self.shard_paths[:max_shards]

        if not self.shard_paths:
            raise FileNotFoundError(f"No .npy files found in {npy_dir}")

        # Memory-map all shards and build index
        self.shards: List[np.ndarray] = []
        self.shard_offsets: List[int] = []
        total = 0

        for path in self.shard_paths:
            mmap = np.load(str(path), mmap_mode="r")
            self.shards.append(mmap)
            self.shard_offsets.append(total)
            total += mmap.shape[0]

        self.total_sequences = total
        self.seq_len = self.shards[0].shape[1] if self.shards else 0

        print(f"[Dataset] Loaded {len(self.shards)} shards, "
              f"{self.total_sequences:,} sequences, seq_len={self.seq_len}")

    def __len__(self) -> int:
        return self.total_sequences

    def __getitem__(self, idx: int) -> dict:
        # Binary search for the correct shard
        shard_idx = 0
        for i, offset in enumerate(self.shard_offsets):
            if offset <= idx:
                shard_idx = i
            else:
                break

        local_idx = idx - self.shard_offsets[shard_idx]
        tokens = self.shards[shard_idx][local_idx].astype(np.int64)

        return {
            "input_ids": torch.from_numpy(tokens),
        }

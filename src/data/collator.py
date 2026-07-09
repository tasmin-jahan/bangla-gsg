"""
DataLoader builder for BanglaGSG.

Wraps ShardedNpyDataset with PyTorch DataLoader.

Reproducible shuffling
-----------------------
Training uses an epoch-seeded RandomSampler instead of DataLoader's
built-in shuffle=True. Each epoch gets its own permutation derived from
`base_seed + epoch`, so:

  - You still get full random shuffling, and a *different* shuffle every
    epoch (real benefit of shuffling is preserved).
  - The permutation for a given (base_seed, epoch) pair is 100%
    deterministic and reproducible across process restarts.
  - On resume, Trainer rebuilds the sampler for the correct epoch and
    fast-forwards by exactly the number of batches already consumed in
    that epoch. This lands on exactly the next unseen batch: no
    duplicated sequences, no skipped/unseen sequences, within an epoch.
"""

import torch
from torch.utils.data import DataLoader, RandomSampler

from src.data.dataset import ShardedNpyDataset
import signal


def ignore_sigint(worker_id):
    """Prevent DataLoader workers from dying instantly on Ctrl+C."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def build_dataset(npy_dir: str, max_shards: int = 0) -> ShardedNpyDataset:
    """Build (memory-map) the underlying dataset once, reused across epochs."""
    return ShardedNpyDataset(npy_dir, max_shards=max_shards)


def make_epoch_loader(
    dataset: ShardedNpyDataset,
    epoch: int,
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle: bool = True,
    pin_memory: bool = True,
    base_seed: int = 1234,
) -> DataLoader:
    """
    Build a DataLoader for a specific epoch with a deterministic,
    epoch-dependent shuffle order.

    Args:
        dataset: Pre-built ShardedNpyDataset (share across epochs; cheap to
            reuse, avoids re-mmap'ing shards every epoch).
        epoch: Epoch index (0-based). Determines the shuffle seed.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        shuffle: Whether to shuffle. If False, iterates in fixed dataset
            order every epoch (used for eval loaders).
        pin_memory: Whether to pin memory (for CUDA).
        base_seed: Base RNG seed for shuffling. Combined with `epoch` to
            get a distinct-but-reproducible permutation per epoch. Keep
            this fixed across all resumes of the same run.

    Returns:
        Configured DataLoader whose iteration order is exactly
        reproducible given (base_seed, epoch).
    """
    sampler = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(base_seed + epoch)
        sampler = RandomSampler(dataset, generator=generator)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # shuffling is handled by the seeded sampler above
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches for gradient accumulation consistency
        worker_init_fn=ignore_sigint,
    )


def build_dataloader(
    npy_dir: str,
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle: bool = True,
    pin_memory: bool = True,
    max_shards: int = 0,
    base_seed: int = 1234,
    epoch: int = 0,
) -> DataLoader:
    """
    Convenience one-shot builder (dataset + epoch-0 loader in one call).

    Used for eval loaders and any place that just needs *a* loader
    without epoch-to-epoch reproducibility bookkeeping (eval always
    passes shuffle=False so `epoch` is irrelevant there).

    For the *training* loader, prefer `build_dataset` once + calling
    `make_epoch_loader` per epoch, so the Trainer can rebuild the loader
    with the correct epoch seed on resume. See Trainer._build_train_loader.
    """
    dataset = build_dataset(npy_dir, max_shards=max_shards)
    return make_epoch_loader(
        dataset,
        epoch=epoch,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        base_seed=base_seed,
    )
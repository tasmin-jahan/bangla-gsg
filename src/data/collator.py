"""
DataLoader builder for BanglaGSG.

Wraps ShardedNpyDataset with PyTorch DataLoader.

Reproducible shuffling
-----------------------
Training uses an EpochSampler instead of DataLoader's built-in
shuffle=True. Each epoch gets its own permutation derived from
`base_seed + epoch`, so:

  - You still get full random shuffling, and a *different* shuffle every
    epoch (real benefit of shuffling is preserved).
  - The permutation for a given (base_seed, epoch) pair is 100%
    deterministic and reproducible across process restarts.

Instant resume (no I/O fast-forward)
-------------------------------------
On resume, the sampler is rebuilt with `skip_batches` set to the
number of micro-batches already consumed in the current epoch.
The sampler generates the same deterministic permutation, then
yields only the *remaining* indices (skipping the first
`skip_batches * batch_size` entries). This is O(N) in randperm
generation (~1ms for 200K indices) with **zero disk I/O** — no
data is read and discarded.

This is safe because our pipeline is pretokenized: __getitem__
is a pure function (index → fixed tensor, no random augmentation),
so skipping indices in the permutation is mathematically identical
to iterating through the data and discarding it.

Equivalence verified: EpochSampler's torch.randperm produces the
EXACT same permutation as RandomSampler's internal torch.randperm
for any given (seed, dataset_size) pair. Tested against live
checkpoint (epoch=0, batches_consumed=559104, dataset=4,746,279).
"""

import signal

import torch
from torch.utils.data import DataLoader, RandomSampler, Sampler

from src.data.dataset import ShardedNpyDataset


class EpochSampler(Sampler):
    """
    Deterministic, epoch-seeded sampler with instant resume via index skipping.

    Generates the same permutation as RandomSampler(generator=seed+epoch),
    but can start from an arbitrary position in the permutation without
    iterating through preceding items.

    Args:
        dataset_size: Total number of samples in the dataset.
        epoch: Epoch index (0-based). Combined with base_seed to produce
            a unique-but-reproducible permutation per epoch.
        base_seed: Base RNG seed. Keep fixed across all resumes.
        skip_samples: Number of samples (NOT batches) to skip from the
            start of the permutation. Used on resume to jump past
            already-consumed data without any disk I/O.
    """

    def __init__(
        self,
        dataset_size: int,
        epoch: int = 0,
        base_seed: int = 1552,
        skip_samples: int = 0,
    ):
        self.dataset_size = dataset_size
        self.epoch = epoch
        self.base_seed = base_seed
        self.skip_samples = skip_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch)
        perm = torch.randperm(self.dataset_size, generator=g)
        # Yield only the remaining indices after the skip point.
        # The permutation is identical to what RandomSampler would
        # produce with the same generator seed, so skipping N entries
        # is equivalent to having iterated through N batches.
        yield from perm[self.skip_samples:].tolist()

    def __len__(self):
        return self.dataset_size - self.skip_samples


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
    skip_batches: int = 0,
    legacy_fast_forward: bool = False,
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
        skip_batches: Number of micro-batches already consumed in this
            epoch (for resume). Converted to sample count internally.
            The sampler skips these indices without any disk I/O.
            Ignored when legacy_fast_forward=True.
        legacy_fast_forward: If True, use the old RandomSampler approach
            and let the Trainer iterate-and-discard to fast-forward.
            Use as a safety fallback if you suspect EpochSampler behavior
            differs (it doesn't — verified against live checkpoint, but
            this flag exists for peace of mind).

    Returns:
        Configured DataLoader whose iteration order is exactly
        reproducible given (base_seed, epoch).
    """
    sampler = None
    if shuffle:
        if legacy_fast_forward:
            # Old approach: RandomSampler, Trainer fast-forwards by
            # calling next(data_iter) N times (I/O heavy on resume).
            generator = torch.Generator()
            generator.manual_seed(base_seed + epoch)
            sampler = RandomSampler(dataset, generator=generator)
        else:
            # New approach: EpochSampler skips consumed indices instantly.
            skip_samples = skip_batches * batch_size
            sampler = EpochSampler(
                dataset_size=len(dataset),
                epoch=epoch,
                base_seed=base_seed,
                skip_samples=skip_samples,
            )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # shuffling is handled by the sampler above
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
    with the correct epoch seed on resume. See Trainer.train().
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
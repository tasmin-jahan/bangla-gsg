"""
DataLoader builder for BanglaGSG.

Wraps ShardedNpyDataset with PyTorch DataLoader.
"""

from torch.utils.data import DataLoader

from src.data.dataset import ShardedNpyDataset
import signal

def ignore_sigint(worker_id):
    """Prevent DataLoader workers from dying instantly on Ctrl+C."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def build_dataloader(
    npy_dir: str,
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle: bool = True,
    pin_memory: bool = True,
    max_shards: int = 0,
) -> DataLoader:
    """
    Build a DataLoader from pretokenized .npy shards.

    Args:
        npy_dir: Directory containing .npy shard files.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        shuffle: Whether to shuffle.
        pin_memory: Whether to pin memory (for CUDA).
        max_shards: Max shards to load (0 = all).

    Returns:
        Configured DataLoader.
    """
    dataset = ShardedNpyDataset(npy_dir, max_shards=max_shards)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches for gradient accumulation consistency
        worker_init_fn=ignore_sigint,
    )

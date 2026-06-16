from __future__ import annotations

from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


class SortedBlockBatchSampler(Sampler[list[int]]):
    """
    Batch sampler that improves HDF5 locality.

    The dataset receives positions inside the split array. For an HDF5 dataset,
    the physical sample index is usually dataset.indices[position].

    This sampler:
      1. sorts split positions by their physical HDF5 index;
      2. groups nearby samples into batches;
      3. shuffles the batch order every epoch.

    This preserves stochasticity at batch level while avoiding fully random
    sample-by-sample HDF5 access.
    """

    def __init__(
        self,
        split_indices: np.ndarray,
        batch_size: int,
        drop_last: bool = True,
        seed: int = 123,
        shuffle_batches: bool = True,
        shuffle_within_batch: bool = False,
    ):
        self.split_indices = np.asarray(split_indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.shuffle_batches = bool(shuffle_batches)
        self.shuffle_within_batch = bool(shuffle_within_batch)
        self.epoch = 0

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.split_indices.ndim != 1:
            raise ValueError(
                f"split_indices must be 1D, got shape {self.split_indices.shape}"
            )

        # Positions in the Dataset sorted by physical HDF5 sample index.
        self.sorted_positions = np.argsort(self.split_indices).astype(np.int64)

        n = len(self.sorted_positions)
        if self.drop_last:
            self.n_batches = n // self.batch_size
        else:
            self.n_batches = int(np.ceil(n / self.batch_size))

    def __len__(self) -> int:
        return self.n_batches

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)

        if self.shuffle_batches:
            batch_order = rng.permutation(self.n_batches)
        else:
            batch_order = np.arange(self.n_batches)

        for batch_id in batch_order:
            start = int(batch_id) * self.batch_size
            stop = start + self.batch_size

            batch_positions = self.sorted_positions[start:stop]

            if len(batch_positions) < self.batch_size and self.drop_last:
                continue

            if self.shuffle_within_batch:
                batch_positions = rng.permutation(batch_positions)

            yield batch_positions.tolist()
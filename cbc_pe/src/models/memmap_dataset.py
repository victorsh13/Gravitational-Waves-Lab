"""float16 memmap batch dataset (M13.3 I/O study).

Reads the per-split export written by scripts/m13/build_fp16_memmap.py:
    X_<split>.npy       float16, (N, C, T), rows in ascending HDF5 row-id order
    y_<split>.npy       float32, (N, D), unstandardized labels [chirp_mass, total_mass, chi_eff]
    row_ids_<split>.npy int64,   (N,), original HDF5 row IDs

Positions handled here are local positions 0..N-1 inside one split, not HDF5 row IDs.
X is returned as float16 (the float32 cast is done on the GPU); y is standardized with the
train-only y_mean/y_std exactly as HDF5BatchIterableDataset does (y - mean) / (std + 1e-8).

Usage:
    sampler = RandomBatchSampler(len(ds), batch_size=256, seed=123)
    loader = DataLoader(ds, sampler=sampler, batch_size=None,
                        num_workers=4, pin_memory=True)
    sampler.set_epoch(epoch)  # before iterating each epoch
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class FP16MemmapBatchDataset(Dataset):
    """Map-style dataset whose index is an array of local positions (one whole batch)."""

    def __init__(self, memmap_dir, split: str, y_mean: np.ndarray, y_std: np.ndarray):
        self.memmap_dir = Path(memmap_dir)
        self.split = str(split)
        self.x_path = self.memmap_dir / f"X_{self.split}.npy"
        self.y_path = self.memmap_dir / f"y_{self.split}.npy"
        self.row_ids_path = self.memmap_dir / f"row_ids_{self.split}.npy"
        self.y_mean = np.asarray(y_mean, dtype=np.float32)
        self.y_std = np.asarray(y_std, dtype=np.float32)

        X = np.load(self.x_path, mmap_mode="r")
        y = np.load(self.y_path, mmap_mode="r")
        if X.dtype != np.float16 or X.ndim != 3:
            raise ValueError(f"Expected float16 (N, C, T) X, got {X.dtype} {X.shape}")
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"X/y length mismatch: {X.shape[0]} vs {y.shape[0]}")
        if y.shape[1:] != self.y_mean.shape or y.shape[1:] != self.y_std.shape:
            raise ValueError(f"Label stats shape {self.y_mean.shape} does not match y {y.shape}")
        self.x_shape = tuple(X.shape)
        self.y_shape = tuple(y.shape)
        del X, y

        # Opened lazily in each worker; never pickled into worker processes.
        self._X = None
        self._y = None

    def __len__(self) -> int:
        return self.x_shape[0]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_X"] = None
        state["_y"] = None
        return state

    def _open(self):
        if self._X is None:
            self._X = np.load(self.x_path, mmap_mode="r")
            self._y = np.load(self.y_path)

    def __getitem__(self, batch_positions):
        self._open()
        pos = np.asarray(batch_positions, dtype=np.int64)
        # Sorted reads keep page-cache/readahead access monotone; then restore sampled order.
        order = np.argsort(pos, kind="stable")
        X_sorted = self._X[pos[order]]
        X = np.empty_like(X_sorted)
        X[order] = X_sorted
        y = (self._y[pos].astype(np.float32) - self.y_mean) / (self.y_std + 1e-8)
        return torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))


class RandomBatchSampler(Sampler):
    """Yields arrays of local positions; new permutation default_rng(seed + epoch) per epoch."""

    def __init__(self, n: int, batch_size: int, seed: int, drop_last: bool = True):
        self.n = int(n)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _order(self) -> np.ndarray:
        return np.random.default_rng(self.seed + self.epoch).permutation(self.n)

    def __len__(self) -> int:
        if self.drop_last:
            return self.n // self.batch_size
        return int(np.ceil(self.n / self.batch_size))

    def __iter__(self):
        order = self._order()
        for b in range(len(self)):
            yield order[b * self.batch_size:(b + 1) * self.batch_size]


class SequentialBatchSampler(RandomBatchSampler):
    """Validation variant: no shuffling, keeps the last partial batch by default."""

    def __init__(self, n: int, batch_size: int, drop_last: bool = False):
        super().__init__(n, batch_size, seed=0, drop_last=drop_last)

    def _order(self) -> np.ndarray:
        return np.arange(self.n, dtype=np.int64)

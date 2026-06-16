#!/usr/bin/env python
"""
Benchmark HDF5 I/O throughput for CBC PE datasets.

This script measures:
1. direct sequential HDF5 reads;
2. direct random HDF5 reads;
3. PyTorch DataLoader throughput for different num_workers.

It does not train a model.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class HDF5SplitDataset(Dataset):
    """Minimal HDF5 dataset for I/O benchmarking."""

    def __init__(
        self,
        hdf5_path: str | Path,
        split_indices: np.ndarray,
        x_key: str = "X",
        y_key: str = "y",
    ):
        self.hdf5_path = str(hdf5_path)
        self.indices = np.asarray(split_indices, dtype=np.int64)
        self.x_key = x_key
        self.y_key = y_key
        self._h5 = None

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        h5 = self._ensure_open()
        idx = int(self.indices[i])

        x = h5[self.x_key][idx]
        y = h5[self.y_key][idx]

        # Force materialization and conversion, similar to training.
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        y = torch.from_numpy(np.asarray(y, dtype=np.float32))

        return x, y
    

class SortedBlockBatchSampler(Sampler):
    """
    Batch sampler that groups physically nearby HDF5 indices.

    It sorts split indices by their HDF5 index, forms batches of contiguous
    entries in that sorted order, and shuffles the batch order each epoch.

    This improves HDF5 locality while keeping stochasticity at batch level.
    """

    def __init__(
        self,
        split_indices: np.ndarray,
        batch_size: int,
        drop_last: bool = True,
        seed: int = 123,
    ):
        self.split_indices = np.asarray(split_indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        sorted_positions = np.argsort(self.split_indices)
        self.sorted_positions = sorted_positions.astype(np.int64)

        n = len(self.sorted_positions)
        if self.drop_last:
            n_batches = n // self.batch_size
        else:
            n_batches = int(np.ceil(n / self.batch_size))

        self.n_batches = n_batches

    def __len__(self):
        return self.n_batches

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batch_order = rng.permutation(self.n_batches)

        for batch_id in batch_order:
            start = batch_id * self.batch_size
            stop = start + self.batch_size

            batch_positions = self.sorted_positions[start:stop]

            if len(batch_positions) < self.batch_size and self.drop_last:
                continue

            yield batch_positions.tolist()


def load_split_indices(split_path: Path, split_name: str) -> np.ndarray:
    with np.load(split_path) as data:
        print("Available split keys:", list(data.keys()))

        possible_keys = [
            split_name,
            f"idx_{split_name}",
            f"{split_name}_idx",
            f"indices_{split_name}",
            f"{split_name}_indices",
        ]

        for key in possible_keys:
            if key in data:
                return data[key]

        raise KeyError(
            f"Could not find split '{split_name}' in {split_path}. "
            f"Tried keys: {possible_keys}"
        )


def print_hdf5_info(hdf5_path: Path):
    print("=" * 80)
    print("HDF5 info")
    print("=" * 80)

    with h5py.File(hdf5_path, "r") as h5:
        print("File:", hdf5_path)
        print("Keys:", list(h5.keys()))

        for key in h5.keys():
            obj = h5[key]
            if hasattr(obj, "shape"):
                print(
                    f"{key}: shape={obj.shape}, dtype={obj.dtype}, "
                    f"chunks={obj.chunks}, compression={obj.compression}"
                )


def benchmark_direct_reads(
    hdf5_path: Path,
    indices: np.ndarray,
    mode: str,
    n_samples: int,
    x_key: str = "X",
    y_key: str = "y",
):
    assert mode in {"sequential", "random"}

    indices = np.asarray(indices, dtype=np.int64)

    if mode == "sequential":
        selected = np.sort(indices[:n_samples])
    else:
        rng = np.random.default_rng(123)
        selected = rng.choice(indices, size=min(n_samples, len(indices)), replace=False)

    n = len(selected)

    print("=" * 80)
    print(f"Direct HDF5 read benchmark: {mode}")
    print("=" * 80)
    print(f"n_samples: {n}")

    t0 = time.perf_counter()

    n_read = 0
    x_sum = 0.0
    y_sum = 0.0

    with h5py.File(hdf5_path, "r") as h5:
        x_ds = h5[x_key]
        y_ds = h5[y_key]

        for idx in selected:
            x = np.asarray(x_ds[int(idx)], dtype=np.float32)
            y = np.asarray(y_ds[int(idx)], dtype=np.float32)

            # Use values to avoid lazy/no-op benchmark.
            x_sum += float(x.mean())
            y_sum += float(y.mean())
            n_read += 1

    elapsed = time.perf_counter() - t0
    samples_per_s = n_read / elapsed

    print(f"elapsed_s:     {elapsed:.3f}")
    print(f"samples_read:  {n_read}")
    print(f"samples/s:     {samples_per_s:.2f}")
    print(f"checksum:      {x_sum:.6e}, {y_sum:.6e}")


def benchmark_contiguous_slices(
    hdf5_path: Path,
    n_samples: int,
    batch_size: int,
    x_key: str = "X",
    y_key: str = "y",
):
    print("=" * 80)
    print("Direct HDF5 contiguous slice benchmark")
    print("=" * 80)
    print(f"n_samples:  {n_samples}")
    print(f"batch_size: {batch_size}")

    t0 = time.perf_counter()

    n_read = 0
    x_sum = 0.0
    y_sum = 0.0

    with h5py.File(hdf5_path, "r") as h5:
        x_ds = h5[x_key]
        y_ds = h5[y_key]

        n_total = x_ds.shape[0]
        start = min(100_000, max(0, n_total - n_samples - 1))
        stop = start + n_samples

        for batch_start in range(start, stop, batch_size):
            batch_stop = min(batch_start + batch_size, stop)

            x = np.asarray(x_ds[batch_start:batch_stop], dtype=np.float32)
            y = np.asarray(y_ds[batch_start:batch_stop], dtype=np.float32)

            x_sum += float(x.mean())
            y_sum += float(y.mean())
            n_read += x.shape[0]

    elapsed = time.perf_counter() - t0
    samples_per_s = n_read / elapsed

    print(f"elapsed_s:     {elapsed:.3f}")
    print(f"samples_read:  {n_read}")
    print(f"samples/s:     {samples_per_s:.2f}")
    print(f"checksum:      {x_sum:.6e}, {y_sum:.6e}")

def benchmark_dataloader(
    hdf5_path: Path,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    n_batches: int,
    pin_memory: bool,
    x_key: str = "X",
    y_key: str = "y",
):
    print("=" * 80)
    print("DataLoader benchmark")
    print("=" * 80)
    print(f"batch_size:  {batch_size}")
    print(f"num_workers: {num_workers}")
    print(f"n_batches:   {n_batches}")
    print(f"pin_memory:  {pin_memory}")

    dataset = HDF5SplitDataset(
        hdf5_path=hdf5_path,
        split_indices=indices,
        x_key=x_key,
        y_key=y_key,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    t0 = time.perf_counter()

    n_seen = 0
    checksum = 0.0

    for batch_i, (x, y) in enumerate(loader):
        checksum += float(x.mean()) + float(y.mean())
        n_seen += x.shape[0]

        if batch_i + 1 >= n_batches:
            break

    elapsed = time.perf_counter() - t0
    batches_per_s = n_batches / elapsed
    samples_per_s = n_seen / elapsed

    print(f"elapsed_s:    {elapsed:.3f}")
    print(f"samples_seen: {n_seen}")
    print(f"batches/s:    {batches_per_s:.3f}")
    print(f"samples/s:    {samples_per_s:.2f}")
    print(f"checksum:     {checksum:.6e}")


def benchmark_block_dataloader(
    hdf5_path: Path,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    n_batches: int,
    pin_memory: bool,
    x_key: str = "X",
    y_key: str = "y",
):
    print("=" * 80)
    print("BlockBatchSampler DataLoader benchmark")
    print("=" * 80)
    print(f"batch_size:  {batch_size}")
    print(f"num_workers: {num_workers}")
    print(f"n_batches:   {n_batches}")
    print(f"pin_memory:  {pin_memory}")

    dataset = HDF5SplitDataset(
        hdf5_path=hdf5_path,
        split_indices=indices,
        x_key=x_key,
        y_key=y_key,
    )

    batch_sampler = SortedBlockBatchSampler(
        split_indices=indices,
        batch_size=batch_size,
        drop_last=True,
        seed=123,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    t0 = time.perf_counter()

    n_seen = 0
    checksum = 0.0

    for batch_i, (x, y) in enumerate(loader):
        checksum += float(x.mean()) + float(y.mean())
        n_seen += x.shape[0]

        if batch_i + 1 >= n_batches:
            break

    elapsed = time.perf_counter() - t0
    batches_per_s = n_batches / elapsed
    samples_per_s = n_seen / elapsed

    print(f"elapsed_s:    {elapsed:.3f}")
    print(f"samples_seen: {n_seen}")
    print(f"batches/s:    {batches_per_s:.3f}")
    print(f"samples/s:    {samples_per_s:.2f}")
    print(f"checksum:     {checksum:.6e}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hdf5-path", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--split-name", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-direct-samples", type=int, default=2048)
    parser.add_argument("--n-batches", type=int, default=200)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--x-key", type=str, default="X")
    parser.add_argument("--y-key", type=str, default="y")

    args = parser.parse_args()

    print_hdf5_info(args.hdf5_path)

    indices = load_split_indices(args.split_path, args.split_name)

    print("=" * 80)
    print("Split info")
    print("=" * 80)
    print("split_name:", args.split_name)
    print("n_indices:", len(indices))
    print("first indices:", indices[:10])
    print("min/max:", int(indices.min()), int(indices.max()))

    benchmark_direct_reads(
        hdf5_path=args.hdf5_path,
        indices=indices,
        mode="sequential",
        n_samples=args.n_direct_samples,
        x_key=args.x_key,
        y_key=args.y_key,
    )

    benchmark_direct_reads(
        hdf5_path=args.hdf5_path,
        indices=indices,
        mode="random",
        n_samples=args.n_direct_samples,
        x_key=args.x_key,
        y_key=args.y_key,
    )

    benchmark_contiguous_slices(
        hdf5_path=args.hdf5_path,
        n_samples=args.n_direct_samples,
        batch_size=args.batch_size,
        x_key=args.x_key,
        y_key=args.y_key,
    )

    for num_workers in args.workers:
        benchmark_dataloader(
            hdf5_path=args.hdf5_path,
            indices=indices,
            batch_size=args.batch_size,
            num_workers=num_workers,
            n_batches=args.n_batches,
            pin_memory=args.pin_memory,
            x_key=args.x_key,
            y_key=args.y_key,
        )

    for num_workers in args.workers:
        benchmark_block_dataloader(
            hdf5_path=args.hdf5_path,
            indices=indices,
            batch_size=args.batch_size,
            num_workers=num_workers,
            n_batches=args.n_batches,
            pin_memory=args.pin_memory,
            x_key=args.x_key,
            y_key=args.y_key,
        )


if __name__ == "__main__":
    main()
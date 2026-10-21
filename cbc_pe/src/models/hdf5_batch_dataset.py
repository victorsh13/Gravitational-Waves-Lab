from __future__ import annotations

from typing import Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


def normalize_batch_per_sample_per_detector_zscore(
    X: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Normalize a batch per sample and per detector/channel.

    Expected shape:
        X: (B, C, T)

    For each sample b and detector c:
        X[b, c, :] <- (X[b, c, :] - mean) / std
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X shape (B, C, T), got {X.shape}")

    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True)

    return ((X - mean) / (std + eps)).astype(np.float32)


def apply_input_normalization_to_batch(
    X: np.ndarray,
    input_normalization: dict | None,
) -> np.ndarray:
    """
    Apply configured input normalization to a batch.
    """
    if input_normalization is None:
        return X

    if not input_normalization.get("enabled", False):
        return X

    mode = input_normalization.get("mode", "none")
    eps = float(input_normalization.get("eps", 1e-6))

    if mode == "per_sample_per_detector_zscore":
        return normalize_batch_per_sample_per_detector_zscore(X, eps=eps)

    raise ValueError(f"Unknown input_normalization mode: {mode}")



class HDF5BatchIterableDataset(IterableDataset):
    """
    IterableDataset that reads full HDF5 batches instead of single samples.

    It is optimized for HDF5 files where X is chunked along the sample axis,
    for example chunks=(64, C, T), and batch_size=64.

    Strategy:
      1. Sort split indices by physical HDF5 index.
      2. Build batches of nearby physical indices.
      3. Shuffle batch order each epoch.
      4. Each worker opens its own HDF5 file handle.
      5. For each batch, read the minimal contiguous slice [min_idx:max_idx+1],
         then select the requested indices inside that slice.

    The dataset yields:
      X_batch: torch.float32, shape (B, C, T)
      y_batch: torch.float32, shape (B, D)
      Optional weights: torch.float32, shape (B,) or (B, 3), when a row lookup is supplied.
    """

    def __init__(
        self,
        h5_path,
        indices: np.ndarray,
        y_mean: np.ndarray | None = None,
        y_std: np.ndarray | None = None,
        input_normalization: dict | None = None,
        batch_size: int = 64,
        drop_last: bool = True,
        seed: int = 123,
        shuffle_batches: bool = True,
        shuffle_within_batch: bool = False,
        max_slice_overread: float = 4.0,
        x_key: str = "X",
        y_key: str = "y",
        sample_weights_by_row: np.ndarray | None = None,
        balanced_sampling=None,
    ):
        super().__init__()

        self.h5_path = str(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.shuffle_batches = bool(shuffle_batches)
        self.shuffle_within_batch = bool(shuffle_within_batch)
        self.max_slice_overread = float(max_slice_overread)
        self.x_key = x_key
        self.y_key = y_key

        self.y_mean = None if y_mean is None else np.asarray(y_mean, dtype=np.float32)
        self.y_std = None if y_std is None else np.asarray(y_std, dtype=np.float32)
        self.input_normalization = input_normalization

        if self.y_mean is not None and self.y_std is None:
            raise ValueError("If y_mean is provided, y_std must also be provided.")

        if self.y_std is not None and self.y_mean is None:
            raise ValueError("If y_std is provided, y_mean must also be provided.")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.max_slice_overread < 1.0:
            raise ValueError(
                f"max_slice_overread must be >= 1.0, got {self.max_slice_overread}"
            )

        if self.indices.ndim != 1:
            raise ValueError(f"indices must be 1D, got shape {self.indices.shape}")

        with h5py.File(self.h5_path, "r") as f:
            if self.x_key not in f:
                raise KeyError(f"HDF5 file does not contain dataset '{self.x_key}'.")
            if self.y_key not in f:
                raise KeyError(f"HDF5 file does not contain dataset '{self.y_key}'.")

            self.x_shape = f[self.x_key].shape
            self.y_shape = f[self.y_key].shape

        if self.x_shape[0] != self.y_shape[0]:
            raise ValueError(
                f"X and y have different number of samples: "
                f"{self.x_shape[0]} vs {self.y_shape[0]}"
            )

        if np.any(self.indices < 0) or np.any(self.indices >= self.x_shape[0]):
            raise ValueError("Some indices are outside the valid dataset range.")

        self.sample_weights_by_row = None
        if sample_weights_by_row is not None:
            weights = np.asarray(sample_weights_by_row, dtype=np.float32)
            if (weights.shape != (self.x_shape[0],)
                    and not (self.y_shape[1:] == (3,) and weights.shape == (self.x_shape[0], 3))):
                raise ValueError("Weights require shape (N_dataset,) or (N_dataset, 3), indexed by row ID")
            selected = weights[self.indices]
            if not np.isfinite(selected).all() or (selected <= 0).any():
                raise ValueError("Split weights must be finite and positive")
            self.sample_weights_by_row = weights

        # Physical HDF5 indices sorted increasingly.
        self.sorted_indices = np.sort(self.indices).astype(np.int64)

        self.balanced_sampling = balanced_sampling
        self.sampling_epoch_provenance = None
        self._schedule_epoch = None
        if balanced_sampling is not None:
            if sample_weights_by_row is not None:
                raise ValueError("Balanced sampling cannot be combined with sample weighting")
            if not np.array_equal(self.sorted_indices, balanced_sampling.train_ids):
                raise ValueError("Sampler IDs must exactly match the dataset train membership")
            # Updated only by the parent between exhausted epoch iterators. Shared
            # storage also reaches persistent workers without stale Python state.
            self._sampling_schedule = torch.empty(
                len(self.sorted_indices), dtype=torch.int64).share_memory_()

        n = len(self.sorted_indices)
        if self.drop_last:
            self.n_batches = n // self.batch_size
        else:
            self.n_batches = int(np.ceil(n / self.batch_size))

        self.epoch = 0
        if self.balanced_sampling is not None:
            self.set_epoch(0)

    def __len__(self) -> int:
        return self.n_batches

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        if self.balanced_sampling is not None and self._schedule_epoch != epoch:
            from src.models.sample_weighting import array_hash
            schedule = self.balanced_sampling.schedule(epoch)
            self._sampling_schedule.numpy()[:] = schedule
            consumed = (self.n_batches * self.batch_size if self.drop_last else len(schedule))
            self.sampling_epoch_provenance = {
                "epoch_zero_based": int(epoch), "nominal_draws": len(schedule),
                "consumed_draws": consumed, "dropped_draws": len(schedule) - consumed,
                "optimizer_steps": self.n_batches,
                "schedule_sha256": array_hash(schedule),
                "consumed_schedule_sha256": array_hash(schedule[:consumed]),
            }
            self._schedule_epoch = int(epoch)

    def _batch_ids_for_worker(self) -> np.ndarray:
        batch_ids = np.arange(self.n_batches, dtype=np.int64)

        if self.shuffle_batches and self.balanced_sampling is None:
            rng = np.random.default_rng(self.seed + self.epoch)
            batch_ids = rng.permutation(batch_ids)

        worker_info = get_worker_info()

        if worker_info is None:
            return batch_ids

        # Split batches across workers.
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

        return batch_ids[worker_id::num_workers]

    def _read_batch(self, h5: h5py.File, batch_id: int):
        if self.balanced_sampling is not None:
            return self._read_sampled_batch(h5, batch_id)
        start_pos = int(batch_id) * self.batch_size
        stop_pos = start_pos + self.batch_size

        batch_indices = self.sorted_indices[start_pos:stop_pos]

        if len(batch_indices) < self.batch_size and self.drop_last:
            return None

        if self.shuffle_within_batch:
            rng = np.random.default_rng(self.seed + self.epoch + int(batch_id))
            batch_indices = rng.permutation(batch_indices)

        # h5py fancy indexing requires increasing indices.
        # To preserve optional shuffled order, read sorted then restore order.
        order = np.argsort(batch_indices)
        sorted_batch_indices = batch_indices[order]

        slice_start = int(sorted_batch_indices[0])
        slice_stop = int(sorted_batch_indices[-1]) + 1
        span = slice_stop - slice_start
        overread = span / max(1, len(sorted_batch_indices))

        if overread <= self.max_slice_overread:
            # Dense batch: read one contiguous slice, then select local positions.
            X_block = h5[self.x_key][slice_start:slice_stop]
            y_block = h5[self.y_key][slice_start:slice_stop]

            local_indices = sorted_batch_indices - slice_start

            X_sorted = np.asarray(X_block[local_indices], dtype=np.float32)
            y_sorted = np.asarray(y_block[local_indices], dtype=np.float32)

        else:
            # Sparse batch: avoid reading a huge contiguous span.
            # h5py requires fancy indices to be sorted increasingly.
            X_sorted = np.asarray(h5[self.x_key][sorted_batch_indices], dtype=np.float32)
            y_sorted = np.asarray(h5[self.y_key][sorted_batch_indices], dtype=np.float32)

        if self.shuffle_within_batch:
            inverse_order = np.argsort(order)
            X = X_sorted[inverse_order]
            y = y_sorted[inverse_order]
        else:
            X = X_sorted
            y = y_sorted

        X = apply_input_normalization_to_batch(
            X,
            self.input_normalization,
        )

        if self.y_mean is not None and self.y_std is not None:
            y = (y - self.y_mean) / (self.y_std + 1e-8)

        if self.sample_weights_by_row is not None:
            # batch_indices already matches X/y after restoring shuffled order.
            weights = self.sample_weights_by_row[batch_indices]
            return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(weights)
        return torch.from_numpy(X), torch.from_numpy(y)

    def _read_sampled_batch(self, h5: h5py.File, batch_id: int):
        start = int(batch_id) * self.batch_size
        rows = self._sampling_schedule.numpy()[start:start + self.batch_size]
        if len(rows) == 0 or (self.drop_last and len(rows) < self.batch_size):
            return None
        unique, inverse = np.unique(rows, return_inverse=True)
        # Same greedy bounded spans as benchmark_sampled_hdf5_reads.py (B).
        # Retain one dataset handle across slices; never materialize whole chunks
        # just to fetch scattered rows. Non-G1 fixtures remain bounded as well.
        x_dataset = h5[self.x_key]
        chunk_rows = min(64, x_dataset.chunks[0]) if x_dataset.chunks else 64
        overread_limit = min(4.0, self.max_slice_overread)
        X_unique = np.empty((len(unique), *x_dataset.shape[1:]), dtype=x_dataset.dtype)
        start = 0
        while start < len(unique):
            end = start + 1
            while end < len(unique):
                if unique[end] // chunk_rows != unique[start] // chunk_rows:
                    break
                if unique[end] - unique[start] + 1 > overread_limit * (end - start + 1):
                    break
                end += 1
            lo, hi = int(unique[start]), int(unique[end - 1]) + 1
            block = x_dataset[lo:hi]
            X_unique[start:end] = block[unique[start:end] - lo]
            del block
            start = end
        # Inverse restores the exact sampled order and multiplicity.
        # Labels are small: read only selected train IDs even in the slice path.
        y_unique = h5[self.y_key][unique]
        X = np.asarray(X_unique[inverse], dtype=np.float32)
        y = np.asarray(y_unique[inverse], dtype=np.float32)
        X = apply_input_normalization_to_batch(X, self.input_normalization)
        if self.y_mean is not None and self.y_std is not None:
            y = (y - self.y_mean) / (self.y_std + 1e-8)
        return torch.from_numpy(X), torch.from_numpy(y)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        # Important: each worker opens its own HDF5 file handle.
        with h5py.File(self.h5_path, "r") as h5:
            for batch_id in self._batch_ids_for_worker():
                out = self._read_batch(h5, int(batch_id))
                if out is not None:
                    yield out

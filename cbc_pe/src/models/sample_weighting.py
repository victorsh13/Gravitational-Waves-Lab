"""Optional train-only label-space weights; M12.0 reference, not latent geometry."""
from __future__ import annotations

import hashlib

import h5py
import numpy as np
import torch


def array_hash(values):
    values = np.ascontiguousarray(values)
    return hashlib.sha256(
        str((values.dtype.str, values.shape)).encode() + values.tobytes()
    ).hexdigest()


def mahalanobis_sample_weights(train_y, alpha=0.5):
    """Fit on physical train labels in sorted original-row order, once per run.

    Return mean-one weights, train empirical percentiles in [0,1], and small
    reference metadata. Ties use the right empirical CDF, exactly as M12.0.
    """
    y = np.asarray(train_y, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 3 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("Expected finite train labels with shape (N, 3)")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    mu, sigma = y.mean(axis=0), y.std(axis=0)
    if (sigma <= 0).any():
        raise ValueError("Degenerate train label scale")
    cov = np.cov(y, rowvar=False, ddof=1)
    scaled_cov = cov / np.outer(sigma, sigma)
    rank = int(np.linalg.matrix_rank(scaled_cov))
    precision = (np.linalg.inv(scaled_cov) if rank == 3 else
                 np.linalg.pinv(scaled_cov, rcond=1e-12))
    z = (y - mu) / sigma
    squared = np.einsum("ni,ij,nj->n", z, precision, z)
    if not np.isfinite(squared).all() or (squared < -1e-10).any():
        raise ValueError("Invalid squared Mahalanobis distance")
    distances = np.sqrt(np.maximum(squared, 0))
    reference = np.sort(distances)
    q = np.searchsorted(reference, distances, side="right") / len(y)
    raw = 1 + alpha * q
    raw_mean = float(raw.mean())
    weights = (raw / raw_mean).astype(np.float32)
    metadata = dict(
        method="train_label_mahalanobis", alpha=float(alpha),
        normalization="train_mean_one", reduction="mean(weight * mean_label_squared_error)",
        n_train=len(y), mean=mu.tolist(), std_ddof0=sigma.tolist(),
        covariance_ddof1=cov.tolist(), scaled_covariance=scaled_cov.tolist(),
        precision=precision.tolist(), rank=rank,
        condition_number=float(np.linalg.cond(scaled_cov)),
        pinv_rcond=1e-12 if rank < 3 else None,
        percentile_mapping="searchsorted(sorted train distances, d, side=right) / n_train",
        reference_distance_sha256=array_hash(reference), raw_weight_mean=raw_mean,
        normalized_weight_sha256=array_hash(weights),
        validation_objective="unweighted standardized MSELoss",
    )
    return weights, q, metadata


def marginal_label_weights(train_y, alpha=0.5):
    """Independent train-only marginal CDFs, in the frozen three-label order.

    Input rows must be sorted by original HDF5 ID, as for the global reference.
    Fit in float64; return float32 weights normalized per label over all train.
    """
    y = np.asarray(train_y, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 3 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("Expected finite train labels with shape (N, 3)")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    mu, sigma = y.mean(axis=0), y.std(axis=0)
    if (sigma <= 0).any():
        raise ValueError("Degenerate train label scale")
    labels = ["chirp_mass", "total_mass", "chi_eff"]
    q = np.empty_like(y)
    reference_hashes = {}
    for j, label in enumerate(labels):
        distance = np.abs((y[:, j] - mu[j]) / sigma[j])
        reference = np.sort(distance)
        q[:, j] = np.searchsorted(reference, distance, side="right") / len(y)
        reference_hashes[label] = array_hash(reference)
    raw = 1 + alpha * q
    raw_mean = raw.mean(axis=0)
    weights = (raw / raw_mean).astype(np.float32)
    metadata = dict(
        method="train_label_marginal", alpha=float(alpha),
        normalization="train_mean_one", normalization_axis="independent per label",
        reduction="mean(weight * squared_error) over batch and labels",
        label_names=labels, n_train=len(y), mean=mu.tolist(), std_ddof0=sigma.tolist(),
        percentile_mapping="searchsorted(sorted train absolute standardized distances, d, side=right) / n_train",
        reference_distance_sha256=reference_hashes, raw_weight_mean=raw_mean.tolist(),
        normalized_weight_sha256=array_hash(weights),
        normalized_weight_sha256_per_label={label: array_hash(weights[:, j])
                                          for j, label in enumerate(labels)},
        validation_objective="unweighted standardized MSELoss",
    )
    return weights, q, metadata


def load_train_sample_weights(h5_path, train_idx, alpha=0.5,
                              method="train_label_mahalanobis"):
    """Read only train labels; return a lookup indexed by original HDF5 row ID.

    Nontraining rows remain NaN so accidental cross-split use fails validation.
    No RNG, signal reads, scaler changes, or minibatch reference fitting.
    """
    calculators = {"train_label_mahalanobis": mahalanobis_sample_weights,
                   "train_label_marginal": marginal_label_weights}
    if method not in calculators:
        raise ValueError(f"Unknown train weighting method: {method}")
    ids = np.asarray(train_idx)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Expected integer train row IDs")
    ids = np.sort(ids.astype(np.int64))
    with h5py.File(h5_path, "r") as h:
        membership = h["split"].asstr()[:]
        if not np.array_equal(ids, np.flatnonzero(membership == "train")):
            raise ValueError("Weights must use the exact frozen HDF5 train membership")
        if h.attrs.get("labels") != "chirp_mass,total_mass,chi_eff":
            raise ValueError("Unexpected physical label order")
        y = np.empty((len(ids), 3), dtype=np.float64)
        # Bounded, sorted fancy reads; no val/cal/test labels enter this fit.
        for start in range(0, len(ids), 4096):
            y[start:start + 4096] = h["y"][ids[start:start + 4096]]
        n_rows = len(membership)
    weights, q, metadata = calculators[method](y, alpha)
    # Global mode retains a vector; marginal mode supplies one coefficient per task.
    by_row = np.full((n_rows, *weights.shape[1:]), np.nan, dtype=np.float32)
    by_row[ids] = weights
    metadata["train_indices_sha256"] = array_hash(ids)
    return by_row, q, metadata


def sample_weighted_mse(prediction, target, weights):
    """Weights already have full-train mean one; never renormalize per batch."""
    if (prediction.shape != target.shape or prediction.ndim != 2
            or prediction.numel() == 0):
        raise ValueError("Expected matching (batch, labels) prediction/target shapes")
    global_mode = weights.shape == (prediction.shape[0],)
    marginal_mode = prediction.shape[1] == 3 and weights.shape == prediction.shape
    if not (global_mode or marginal_mode):
        raise ValueError("Expected (B,) global weights or (B, 3) marginal weights")
    if not all(torch.isfinite(value).all().item() for value in (prediction, target, weights)):
        raise ValueError("Prediction, target and weights must be finite")
    if marginal_mode:
        return torch.mean(weights * (prediction - target) ** 2)
    # Preserve the historical M12.1 reduction order exactly.
    per_sample = torch.mean((prediction - target) ** 2, dim=1)
    return torch.mean(weights * per_sample)

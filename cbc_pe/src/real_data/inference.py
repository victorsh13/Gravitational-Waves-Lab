from __future__ import annotations

"""
Utilities for CNN inference on real gravitational-wave inputs.

This module contains model-inference helpers only. Input preprocessing,
GWOSC access and conformal calibration are handled elsewhere.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.evaluate import (
    extract_predictions_and_embeddings,
    inverse_standardize,
)


def predict_real_with_embeddings(
    model: torch.nn.Module,
    X_real: np.ndarray,
    device: torch.device | str,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    *,
    batch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict standardized labels, physical labels and embeddings.

    Parameters
    ----------
    model : torch.nn.Module
        Regression model supporting ``return_embedding=True``.

    X_real : np.ndarray, shape (N, C, T)
        Fully preprocessed model inputs. Any required input normalization
        must already have been applied.

    device : torch.device or str
        Device used for inference.

    y_mean : np.ndarray, shape (D,)
        Train-set label means.

    y_std : np.ndarray, shape (D,)
        Train-set label standard deviations.

    batch_size : int
        Inference batch size.

    Returns
    -------
    pred_std : np.ndarray, shape (N, D)
        Predictions in standardized label space.

    pred_phys : np.ndarray, shape (N, D)
        Predictions transformed back to physical label space.

    emb : np.ndarray, shape (N, E)
        Model embeddings.
    """
    X_real = np.asarray(
        X_real,
        dtype=np.float32,
    )

    y_mean = np.asarray(
        y_mean,
        dtype=np.float32,
    )

    y_std = np.asarray(
        y_std,
        dtype=np.float32,
    )

    if X_real.ndim != 3:
        raise ValueError(
            "X_real must have shape (N, C, T)."
        )

    if X_real.shape[0] < 1:
        raise ValueError(
            "X_real must contain at least one sample."
        )

    if not np.all(np.isfinite(X_real)):
        raise ValueError(
            "X_real must contain only finite values."
        )

    if y_mean.ndim != 1:
        raise ValueError(
            "y_mean must be a 1D array."
        )

    if y_std.ndim != 1:
        raise ValueError(
            "y_std must be a 1D array."
        )

    if y_mean.shape != y_std.shape:
        raise ValueError(
            "y_mean and y_std must have the same shape."
        )

    if len(y_mean) < 1:
        raise ValueError(
            "y_mean and y_std must contain at least one label."
        )

    if not np.all(np.isfinite(y_mean)):
        raise ValueError(
            "y_mean must contain only finite values."
        )

    if not np.all(np.isfinite(y_std)):
        raise ValueError(
            "y_std must contain only finite values."
        )

    if np.any(y_std <= 0):
        raise ValueError(
            "All y_std values must be positive."
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be a positive integer."
        )

    X_tensor = torch.from_numpy(X_real)

    # extract_predictions_and_embeddings expects a loader yielding
    # (X, y). These dummy targets are structural only and are never
    # used to evaluate real-event inference.
    y_dummy = torch.zeros(
        (X_tensor.shape[0], len(y_mean)),
        dtype=torch.float32,
    )

    loader = DataLoader(
        TensorDataset(
            X_tensor,
            y_dummy,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    pred_std, emb, _ = (
        extract_predictions_and_embeddings(
            model=model,
            loader=loader,
            device=device,
        )
    )

    if pred_std.ndim != 2:
        raise RuntimeError(
            "Model predictions must be a 2D array."
        )

    if pred_std.shape[1] != len(y_mean):
        raise RuntimeError(
            "Model output dimension does not match "
            "the provided label statistics."
        )

    pred_phys = inverse_standardize(
        pred_std,
        y_mean[None, :],
        y_std[None, :],
    )

    return (
        pred_std,
        pred_phys,
        emb,
    )
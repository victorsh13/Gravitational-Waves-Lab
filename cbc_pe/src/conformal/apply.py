import numpy as np


def apply_indices(
    values: np.ndarray,
    bin_indices: np.ndarray,
    intervals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply calibrated conformal offsets to predictions according to bin indices.

    Parameters
    ----------
    values : np.ndarray, shape (n_samples, n_labels)
        Model predictions or point estimates.

    bin_indices : np.ndarray, shape (n_samples, n_labels)
        Bin index assigned to each sample/label.

    intervals : np.ndarray, shape (n_labels, n_bins, 2)
        Calibrated offsets per label and bin.

        intervals[label_idx, bin_idx, 0] is the lower offset.
        intervals[label_idx, bin_idx, 1] is the upper offset.

    Returns
    -------
    lower : np.ndarray, shape (n_samples, n_labels)
        Lower interval bounds.

    upper : np.ndarray, shape (n_samples, n_labels)
        Upper interval bounds.
    """
    values = np.asarray(values, dtype=float)
    bin_indices = np.asarray(bin_indices)
    intervals = np.asarray(intervals, dtype=float)

    _validate_apply_inputs(
        values=values,
        bin_indices=bin_indices,
        intervals=intervals,
    )

    bin_indices = _ensure_integer_bin_indices(bin_indices)

    n_samples, n_labels = values.shape

    lower = np.empty((n_samples, n_labels), dtype=float)
    upper = np.empty((n_samples, n_labels), dtype=float)

    for label_idx in range(n_labels):
        lower_offsets = intervals[label_idx, bin_indices[:, label_idx], 0]
        upper_offsets = intervals[label_idx, bin_indices[:, label_idx], 1]

        lower[:, label_idx] = values[:, label_idx] + lower_offsets
        upper[:, label_idx] = values[:, label_idx] + upper_offsets

    if np.any(lower > upper):
        raise ValueError(
            "Some constructed intervals have lower bound greater than upper bound. "
            "Check the calibrated offsets in intervals."
        )

    return lower, upper


def _validate_apply_inputs(
    values: np.ndarray,
    bin_indices: np.ndarray,
    intervals: np.ndarray,
) -> None:
    if values.ndim != 2:
        raise ValueError("values must be a 2D array.")

    if bin_indices.ndim != 2:
        raise ValueError("bin_indices must be a 2D array.")

    if intervals.ndim != 3:
        raise ValueError("intervals must be a 3D array.")

    if values.shape != bin_indices.shape:
        raise ValueError("values and bin_indices must have the same shape.")

    if intervals.shape[0] != values.shape[1]:
        raise ValueError(
            "intervals must have one first-dimension entry per label."
        )

    if intervals.shape[2] != 2:
        raise ValueError("The last dimension of intervals must have size 2.")

    if not np.all(np.isfinite(values)):
        raise ValueError("values must contain only finite values.")

    if not np.all(np.isfinite(intervals)):
        raise ValueError("intervals must contain only finite values.")

    n_bins = intervals.shape[1]

    if np.any(bin_indices < 0) or np.any(bin_indices >= n_bins):
        raise ValueError("bin_indices must be in the range [0, n_bins).")


def _ensure_integer_bin_indices(bin_indices: np.ndarray) -> np.ndarray:
    if np.issubdtype(bin_indices.dtype, np.integer):
        return bin_indices

    bin_indices_as_int = bin_indices.astype(int)

    if np.all(bin_indices == bin_indices_as_int):
        return bin_indices_as_int

    raise ValueError("bin_indices must contain integer values.")
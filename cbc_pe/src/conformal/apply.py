import numpy as np

def apply_indices(values, bin_indices, intervals):
    """
    Apply the bin indices to the values to get the confidence intervals.

    Parameters
    ----------
    values : numpy.ndarray
        The values of the model. With shape (n_samples, n_labels).
    bin_indices : numpy.ndarray
        The bin indices for each value. With shape (n_samples, n_labels).
    intervals : numpy.ndarray
        The intervals for each bin and label. With shape (n_labels, n_bins, 2).

    Returns
    -------
    lower : numpy.ndarray
        The lower bound of the confidence intervals.
    upper : numpy.ndarray
        The upper bound of the confidence intervals.
    """
    values = np.asarray(values, dtype=float)
    bin_indices = np.asarray(bin_indices)
    intervals = np.asarray(intervals, dtype=float)

    if values.ndim != 2: # shape (n_samples, n_labels)
        raise ValueError("The values must be a 2D array.")
        
    if bin_indices.ndim != 2:  # shape (n_samples, n_labels)
        raise ValueError("The bin indices must be a 2D array.")

    if values.shape != bin_indices.shape:
        raise ValueError("The values and bin indices must have the same shape.")

    if intervals.ndim !=3:
        raise ValueError("The intervals must be a 3D array.")
    
    if intervals.shape[0] != values.shape[1]:
        raise ValueError("Intervals must have one first-dimension entry per label.")
    
    if intervals.shape[2] != 2:
        raise ValueError("The last dimension of intervals must have size 2.")
    
    if np.any(bin_indices < 0) or np.any(bin_indices >= intervals.shape[1]):
        raise ValueError("Bin indices must be in the range [0, n_bins).")

    n_labels = intervals.shape[0]
    value_with_lower = np.empty_like(values, dtype=float) # (n_samples, n_labels)
    value_with_upper = np.empty_like(values, dtype=float)

    for n in range(n_labels):
        # Compute the offset for the lower bound
        lower_offsets = intervals[n, bin_indices[:, n], 0]
        value_with_lower[:, n] = values[:, n] + lower_offsets

        # Compute the offset for the upper bound
        upper_offsets = intervals[n, bin_indices[:, n], 1]
        value_with_upper[:, n] = values[:, n] + upper_offsets

    return value_with_lower, value_with_upper
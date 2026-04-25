import numpy as np

class CoverageEvaluator:
    def __init__(
            self,
            confidence_level: float = 0.90,):
        
        self.confidence_level = confidence_level

    def evaluate_intervals(self, y, upper_bound, lower_bound, bin_indices):
        """
        Evaluate the coverage and width of intervals.

        Parameters
        ----------
        y : numpy.ndarray, shape (n_samples, n_labels)
            The true labels.
        upper_bound : numpy.ndarray, shape (n_samples, n_labels)
            The upper_bound of the intervals.
        lower_bound : numpy.ndarray, shape (n_samples, n_labels)
            The lower_bound of the intervals.
        bin_indices : numpy.ndarray, shape (n_samples, n_labels), optional
            The bin indices for each sample and label.

        Returns
        -------
        metrics : dict
            A dictionary containing the following metrics:
            "global_coverage": numpy.ndarray, shape (n_labels,), the proportion of y that fall in the intervals, averaged over all samples for each label.
            "miscoverage": numpy.ndarray, shape (n_labels,), 1 - global_coverage.
            "global_mean_width": numpy.ndarray, shape (n_labels,), the average interval width for each label, averaged over all samples.
            "global_median_width": numpy.ndarray, shape (n_labels,), the median interval width for each label, over all samples.
            "global_coverage_gap": numpy.ndarray, shape (n_labels,), the absolute gap between the global coverage and the confidence level for each label.
            "coverage_per_bin": numpy.ndarray, shape (n_bins, n_labels), the proportion of y that fall in the intervals for each bin, averaged over all samples for each label.
            "mean_width_per_bin": numpy.ndarray, shape (n_bins, n_labels), the average interval width for each bin, averaged over all samples for each label.
            "median_width_per_bin": numpy.ndarray, shape (n_bins, n_labels), the median interval width for each bin, over all samples for each label.
            "counts_per_bin": numpy.ndarray, shape (n_bins, n_labels), the number of samples in each bin for each label.
            "min_coverage_per_label": numpy.ndarray, shape (n_labels,), the minimum coverage across bins for each label.
            "max_undercoverage_gap": numpy.ndarray, shape (n_labels, ), the maximum gap between the coverage and the confidence level across bins for each label.
        """
        if y.ndim != 2:
            raise ValueError("'y' must be a 2D array")
        if upper_bound.ndim != 2:
            raise ValueError("upper_bound must be a 2D array")
        if lower_bound.ndim != 2:
            raise ValueError("lower_bound must be a 2D array")
        if y.shape != upper_bound.shape or y.shape != lower_bound.shape:
            raise ValueError("'y', upper_bound and lower_bound must have the same shape")
        if bin_indices is None:
            raise ValueError("bin_indices must be provided to compute the metrics per bin")
        if bin_indices is not None:
            if bin_indices.ndim != 2:
                raise ValueError("bin_indices must be a 2D array")
            if bin_indices.shape != y.shape:
                raise ValueError("bin_indices must be a 2D array with the same shape as 'y', upper_bound and lower_bound.")

        n_labels = y.shape[1]
        n_bins = np.max(bin_indices) + 1 if bin_indices is not None else 1

        # Ensure that lower_bound <= upper_bound, in special for cases with asymmetric intervals
        if np.any(lower_bound > upper_bound):
            lower_bound, upper_bound = np.minimum(lower_bound, upper_bound), np.maximum(lower_bound, upper_bound)


        # Calculate the GLOBAL coverage, mean and median width for each label
        global_coverage = np.mean((lower_bound <= y) & (y <= upper_bound), axis=0)
        global_mean_width = np.mean(upper_bound - lower_bound, axis=0)
        global_median_width = np.median(upper_bound - lower_bound, axis=0)
        global_coverage_gap = np.abs(global_coverage - self.confidence_level)

        # Calculate the coverage, mean and median width for each bin and label if bin_indices is not None
        if n_bins > 0:
            coverage_per_bin = np.empty((n_bins, n_labels))
            mean_width_per_bin = np.empty((n_bins, n_labels))
            median_width_per_bin = np.empty((n_bins, n_labels))
            counts_per_bin = np.empty((n_bins, n_labels), dtype=int)
            min_coverage_per_label = np.empty(n_labels)
            max_undercoverage_gap = np.empty(n_labels)

            for bin_idx in range(n_bins):
                for label_idx in range(n_labels):
                    mask = bin_indices[:, label_idx] == bin_idx
                    count = np.sum(mask)
                    counts_per_bin[bin_idx, label_idx] = count

                    if count == 0:
                        coverage_per_bin[bin_idx, label_idx] = np.nan
                        mean_width_per_bin[bin_idx, label_idx] = np.nan
                        median_width_per_bin[bin_idx, label_idx] = np.nan
                        counts_per_bin[bin_idx, label_idx] = 0
                        continue

                    covered = (lower_bound[mask, label_idx] <= y[mask, label_idx]) & (y[mask, label_idx] <= upper_bound[mask, label_idx])
                    widths = upper_bound[mask, label_idx] - lower_bound[mask, label_idx]

                    coverage_per_bin[bin_idx, label_idx] = np.mean(covered)
                    mean_width_per_bin[bin_idx, label_idx] = np.mean(widths)
                    median_width_per_bin[bin_idx, label_idx] = np.median(widths)
            
            min_coverage_per_label = np.nanmin(coverage_per_bin, axis=0)
            max_undercoverage_gap = np.nanmax(np.maximum(0, self.confidence_level - coverage_per_bin), axis=0)
        

        return {
            "global_coverage": global_coverage,
            "miscoverage": 1 - global_coverage,
            "global_mean_width": global_mean_width,
            "global_median_width": global_median_width,
            "global_coverage_gap": global_coverage_gap,
            "coverage_per_bin": coverage_per_bin,
            "mean_width_per_bin": mean_width_per_bin,
            "median_width_per_bin": median_width_per_bin,
            "counts_per_bin": counts_per_bin,
            "min_coverage_per_label": min_coverage_per_label,
            "max_undercoverage_gap": max_undercoverage_gap
        }
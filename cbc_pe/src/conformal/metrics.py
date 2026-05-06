import numpy as np
from scipy.stats import binomtest


class CoverageEvaluator:
    """
    Evaluate empirical coverage and interval efficiency for Mondrian conformal
    regression intervals.

    The evaluator computes:
    - global coverage per label;
    - interval widths per label;
    - bin-wise coverage and width;
    - bin counts and covered counts;
    - nominal normal tolerance bands;
    - one-sided binomial undercoverage p-values;
    - lower/upper miss rates.
    """

    def __init__(
        self,
        confidence_level: float = 0.90,
        tolerance_sigmas: tuple[int, ...] = (1, 2, 3),
    ) -> None:
        if not (0.0 < confidence_level < 1.0):
            raise ValueError("confidence_level must be between 0 and 1.")

        if len(tolerance_sigmas) == 0:
            raise ValueError("tolerance_sigmas must contain at least one sigma level.")

        if any(k <= 0 for k in tolerance_sigmas):
            raise ValueError("All tolerance sigma levels must be positive.")

        self.confidence_level = confidence_level
        self.tolerance_sigmas = tolerance_sigmas

    def evaluate_intervals(
        self,
        y: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        bin_indices: np.ndarray,
        n_bins: int | None = None,
    ) -> dict:
        """
        Evaluate conformal intervals.

        Parameters
        ----------
        y : np.ndarray, shape (n_samples, n_labels)
            True labels.
        lower_bound : np.ndarray, shape (n_samples, n_labels)
            Lower interval bounds.
        upper_bound : np.ndarray, shape (n_samples, n_labels)
            Upper interval bounds.
        bin_indices : np.ndarray, shape (n_samples, n_labels)
            Bin index assigned to each sample/label.
        n_bins : int | None
            Total number of bins. If None, inferred from bin_indices.

        Returns
        -------
        dict
            Dictionary containing global and bin-wise coverage/width metrics.
        """
        y = np.asarray(y, dtype=float)
        lower_bound = np.asarray(lower_bound, dtype=float)
        upper_bound = np.asarray(upper_bound, dtype=float)
        bin_indices = np.asarray(bin_indices)

        self._validate_inputs(
            y=y,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            bin_indices=bin_indices,
        )

        bin_indices = self._ensure_integer_bin_indices(bin_indices)

        if n_bins is None:
            n_bins = int(np.max(bin_indices)) + 1

        if n_bins < 1:
            raise ValueError("n_bins must be at least 1.")

        if np.any(bin_indices < 0) or np.any(bin_indices >= n_bins):
            raise ValueError("bin_indices contain values outside [0, n_bins).")

        # Defensive handling. In principle lower_bound should already be <= upper_bound.
        if np.any(lower_bound > upper_bound):
            lower_bound, upper_bound = (
                np.minimum(lower_bound, upper_bound),
                np.maximum(lower_bound, upper_bound),
            )

        n_samples, n_labels = y.shape

        covered = (lower_bound <= y) & (y <= upper_bound)
        lower_miss = y < lower_bound
        upper_miss = y > upper_bound
        widths = upper_bound - lower_bound

        # -------------------------
        # Global metrics per label
        # -------------------------
        covered_count_global = np.sum(covered, axis=0).astype(int)
        n_samples_per_label = np.full(n_labels, n_samples, dtype=int)

        global_coverage = covered_count_global / n_samples_per_label
        miscoverage = 1.0 - global_coverage

        global_mean_width = np.mean(widths, axis=0)
        global_median_width = np.median(widths, axis=0)
        global_coverage_gap = np.abs(global_coverage - self.confidence_level)

        global_lower_miss_rate = np.mean(lower_miss, axis=0)
        global_upper_miss_rate = np.mean(upper_miss, axis=0)
        global_tail_miss_imbalance = np.abs(
            global_lower_miss_rate - global_upper_miss_rate
        )

        # -------------------------
        # Bin-wise metrics
        # -------------------------
        coverage_per_bin = np.full((n_bins, n_labels), np.nan, dtype=float)
        mean_width_per_bin = np.full((n_bins, n_labels), np.nan, dtype=float)
        median_width_per_bin = np.full((n_bins, n_labels), np.nan, dtype=float)

        lower_miss_rate_per_bin = np.full((n_bins, n_labels), np.nan, dtype=float)
        upper_miss_rate_per_bin = np.full((n_bins, n_labels), np.nan, dtype=float)

        counts_per_bin = np.zeros((n_bins, n_labels), dtype=int)
        covered_count_per_bin = np.zeros((n_bins, n_labels), dtype=int)

        for bin_idx in range(n_bins):
            for label_idx in range(n_labels):
                mask = bin_indices[:, label_idx] == bin_idx
                count = int(np.sum(mask))
                counts_per_bin[bin_idx, label_idx] = count

                if count == 0:
                    continue

                covered_bin = covered[mask, label_idx]
                widths_bin = widths[mask, label_idx]
                lower_miss_bin = lower_miss[mask, label_idx]
                upper_miss_bin = upper_miss[mask, label_idx]

                covered_count = int(np.sum(covered_bin))
                covered_count_per_bin[bin_idx, label_idx] = covered_count

                coverage_per_bin[bin_idx, label_idx] = covered_count / count
                mean_width_per_bin[bin_idx, label_idx] = np.mean(widths_bin)
                median_width_per_bin[bin_idx, label_idx] = np.median(widths_bin)

                lower_miss_rate_per_bin[bin_idx, label_idx] = np.mean(lower_miss_bin)
                upper_miss_rate_per_bin[bin_idx, label_idx] = np.mean(upper_miss_bin)

        min_coverage_per_label = np.nanmin(coverage_per_bin, axis=0)
        max_undercoverage_gap = np.nanmax(
            np.maximum(0.0, self.confidence_level - coverage_per_bin),
            axis=0,
        )

        # Nominal normal approximation bands.
        global_tolerance_normal = self.normal_tolerance_bands_nominal(
            n=n_samples_per_label,
            sigmas=self.tolerance_sigmas,
        )

        bin_tolerance_normal = self.normal_tolerance_bands_nominal(
            n=counts_per_bin,
            sigmas=self.tolerance_sigmas,
        )

        # One-sided binomial p-values for undercoverage.
        global_undercoverage_pvalue = self.undercoverage_pvalues(
            covered_count=covered_count_global,
            total_count=n_samples_per_label,
        )

        bin_undercoverage_pvalue = self.undercoverage_pvalues(
            covered_count=covered_count_per_bin,
            total_count=counts_per_bin,
        )

        return {
            # Global coverage
            "global_coverage": global_coverage,
            "miscoverage": miscoverage,
            "global_coverage_gap": global_coverage_gap,
            "covered_count_global": covered_count_global,
            "n_samples_per_label": n_samples_per_label,
            "global_undercoverage_pvalue": global_undercoverage_pvalue,

            # Global width
            "global_mean_width": global_mean_width,
            "global_median_width": global_median_width,

            # Global lower/upper miss rates
            "global_lower_miss_rate": global_lower_miss_rate,
            "global_upper_miss_rate": global_upper_miss_rate,
            "global_tail_miss_imbalance": global_tail_miss_imbalance,

            # Bin-wise coverage
            "coverage_per_bin": coverage_per_bin,
            "counts_per_bin": counts_per_bin,
            "covered_count_per_bin": covered_count_per_bin,
            "min_coverage_per_label": min_coverage_per_label,
            "max_undercoverage_gap": max_undercoverage_gap,
            "bin_undercoverage_pvalue": bin_undercoverage_pvalue,

            # Bin-wise widths
            "mean_width_per_bin": mean_width_per_bin,
            "median_width_per_bin": median_width_per_bin,

            # Bin-wise lower/upper miss rates
            "lower_miss_rate_per_bin": lower_miss_rate_per_bin,
            "upper_miss_rate_per_bin": upper_miss_rate_per_bin,

            # Tolerance bands
            "global_tolerance_normal": global_tolerance_normal,
            "bin_tolerance_normal": bin_tolerance_normal,
        }

    def normal_tolerance_bands_nominal(
        self,
        n: int | np.ndarray,
        sigmas: tuple[int, ...] = (1, 2, 3),
    ) -> dict:
        """
        Normal approximation bands around the nominal confidence level.

        This answers:
        If the true coverage were self.confidence_level, what empirical coverage
        fluctuations would be expected for sample size n?

        Important:
        These are not error bars around the measured coverage. They are nominal
        tolerance bands centered at the target confidence level.
        """
        n = np.asarray(n, dtype=float)

        if np.any(n < 0):
            raise ValueError("n must be non-negative.")

        p = self.confidence_level

        sigma = np.full_like(n, np.nan, dtype=float)
        valid = n > 0
        sigma[valid] = np.sqrt(p * (1.0 - p) / n[valid])

        bands = {}
        for k in sigmas:
            width = k * sigma
            bands[f"{k}sigma_width"] = width
            bands[f"{k}sigma_low"] = np.maximum(0.0, p - width)
            bands[f"{k}sigma_high"] = np.minimum(1.0, p + width)

        return bands

    def undercoverage_pvalues(
        self,
        covered_count: int | np.ndarray,
        total_count: int | np.ndarray,
    ) -> np.ndarray:
        """
        One-sided binomial p-values for undercoverage.

        Null hypothesis:
            true coverage = self.confidence_level

        Alternative:
            true coverage < self.confidence_level

        Small p-values indicate that the observed coverage is unusually low
        under the nominal target coverage.
        """
        covered_count = np.asarray(covered_count)
        total_count = np.asarray(total_count)

        if covered_count.shape != total_count.shape:
            raise ValueError("covered_count and total_count must have the same shape.")

        if np.any(total_count < 0):
            raise ValueError("total_count must be non-negative.")

        if np.any(covered_count < 0):
            raise ValueError("covered_count must be non-negative.")

        if np.any(covered_count > total_count):
            raise ValueError("covered_count cannot exceed total_count.")

        pvalues = np.full(total_count.shape, np.nan, dtype=float)

        for idx in np.ndindex(total_count.shape):
            n = int(total_count[idx])
            k = int(covered_count[idx])

            if n == 0:
                continue

            pvalues[idx] = binomtest(
                k=k,
                n=n,
                p=self.confidence_level,
                alternative="less",
            ).pvalue

        return pvalues

    @staticmethod
    def _validate_inputs(
        y: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        bin_indices: np.ndarray,
    ) -> None:
        if y.ndim != 2:
            raise ValueError("'y' must be a 2D array.")
        if lower_bound.ndim != 2:
            raise ValueError("lower_bound must be a 2D array.")
        if upper_bound.ndim != 2:
            raise ValueError("upper_bound must be a 2D array.")
        if bin_indices.ndim != 2:
            raise ValueError("bin_indices must be a 2D array.")

        if y.shape != lower_bound.shape or y.shape != upper_bound.shape:
            raise ValueError("'y', lower_bound and upper_bound must have the same shape.")

        if bin_indices.shape != y.shape:
            raise ValueError("bin_indices must have the same shape as y.")

        if not np.all(np.isfinite(y)):
            raise ValueError("'y' must contain only finite values.")
        if not np.all(np.isfinite(lower_bound)):
            raise ValueError("lower_bound must contain only finite values.")
        if not np.all(np.isfinite(upper_bound)):
            raise ValueError("upper_bound must contain only finite values.")

    @staticmethod
    def _ensure_integer_bin_indices(bin_indices: np.ndarray) -> np.ndarray:
        if np.issubdtype(bin_indices.dtype, np.integer):
            return bin_indices

        bin_indices_as_int = bin_indices.astype(int)

        if np.all(bin_indices == bin_indices_as_int):
            return bin_indices_as_int

        raise ValueError("bin_indices must contain integer values.")
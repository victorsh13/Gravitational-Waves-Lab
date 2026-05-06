import warnings

import numpy as np


class ConformalIntervalCalibrator:
    """
    Compute conformal calibration offsets for each label/bin pair.

    The input is a grouped residual array with shape:

        (n_labels, n_bins)

    where each cell contains a 1D array of calibration residuals:

        residual = y_cal - pred_cal

    The output intervals_ has shape:

        (n_labels, n_bins, 2)

    where:

        intervals_[label_idx, bin_idx, 0] = lower_offset
        intervals_[label_idx, bin_idx, 1] = upper_offset
    """

    def __init__(
        self,
        confidence_level: float = 0.90,
        interval_mode: str = "symmetric",
        min_samples_per_bin: int = 10,
        warn_on_extreme_quantiles: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        confidence_level : float
            Target coverage level, e.g. 0.90.

        interval_mode : {"symmetric", "asymmetric"}
            "symmetric":
                Uses absolute residuals and returns [-q, q].

            "asymmetric":
                Uses signed residuals and returns [q_low, q_high],
                with alpha/2 assigned to each tail.

        min_samples_per_bin : int
            Minimum number of residuals required in each label/bin pair.

        warn_on_extreme_quantiles : bool
            If True, warn when asymmetric calibration uses the minimum or maximum
            residual. This usually indicates a small or noisy bin.
        """
        valid_modes = {"symmetric", "asymmetric"}

        if not (0.0 < confidence_level < 1.0):
            raise ValueError("confidence_level must be between 0 and 1.")

        if interval_mode not in valid_modes:
            raise ValueError(f"interval_mode must be one of {sorted(valid_modes)}.")

        if min_samples_per_bin < 1:
            raise ValueError("min_samples_per_bin must be at least 1.")

        self.confidence_level = confidence_level
        self.interval_mode = interval_mode
        self.min_samples_per_bin = min_samples_per_bin
        self.warn_on_extreme_quantiles = warn_on_extreme_quantiles

        self.intervals_: np.ndarray | None = None
        self.bin_counts_: np.ndarray | None = None
        self.quantile_indices_: np.ndarray | None = None
        self.interval_widths_: np.ndarray | None = None
        self.is_fitted_ = False

    def fit(self, grouped_residuals: np.ndarray) -> "ConformalIntervalCalibrator":
        """
        Fit calibration offsets from residuals grouped by label and bin.

        Parameters
        ----------
        grouped_residuals : np.ndarray, shape (n_labels, n_bins), dtype=object
            grouped_residuals[label_idx, bin_idx] must contain a 1D array-like
            of residuals.

        Returns
        -------
        self
        """
        if grouped_residuals is None:
            raise ValueError("grouped_residuals must not be None.")

        grouped_residuals = np.asarray(grouped_residuals, dtype=object)

        if grouped_residuals.ndim != 2:
            raise ValueError(
                "grouped_residuals must be a 2D object array with shape "
                "(n_labels, n_bins)."
            )

        n_labels, n_bins = grouped_residuals.shape

        if n_labels == 0:
            raise ValueError("grouped_residuals must contain at least one label.")
        if n_bins == 0:
            raise ValueError("grouped_residuals must contain at least one bin.")

        intervals = np.empty((n_labels, n_bins, 2), dtype=float)
        bin_counts = np.empty((n_labels, n_bins), dtype=int)
        quantile_indices = np.empty((n_labels, n_bins, 2), dtype=int)

        for label_idx in range(n_labels):
            for bin_idx in range(n_bins):
                residuals_per_bin = np.asarray(
                    grouped_residuals[label_idx, bin_idx],
                    dtype=float,
                )

                self._validate_residuals_per_bin(
                    residuals_per_bin=residuals_per_bin,
                    label_idx=label_idx,
                    bin_idx=bin_idx,
                )

                m = residuals_per_bin.size
                bin_counts[label_idx, bin_idx] = m

                if m < self.min_samples_per_bin:
                    raise ValueError(
                        f"Bin {bin_idx} for label {label_idx} has {m} samples, "
                        f"but at least {self.min_samples_per_bin} are required."
                    )

                if self.interval_mode == "symmetric":
                    interval, indices = self._compute_symmetric_interval(
                        residuals_per_bin
                    )

                elif self.interval_mode == "asymmetric":
                    interval, indices = self._compute_asymmetric_interval(
                        residuals_per_bin=residuals_per_bin,
                        label_idx=label_idx,
                        bin_idx=bin_idx,
                    )

                else:
                    raise RuntimeError(
                        f"Unsupported interval_mode={self.interval_mode!r}. "
                        "This should have been validated in __init__."
                    )

                intervals[label_idx, bin_idx] = interval
                quantile_indices[label_idx, bin_idx] = indices

        self.intervals_ = intervals
        self.bin_counts_ = bin_counts
        self.quantile_indices_ = quantile_indices
        self.interval_widths_ = intervals[:, :, 1] - intervals[:, :, 0]
        self.is_fitted_ = True

        return self

    def _compute_symmetric_interval(
        self,
        residuals_per_bin: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute symmetric conformal offset [-q, q].

        q is selected as the conformal order statistic:

            k = ceil((m + 1) * confidence_level) - 1

        using zero-based indexing.
        """
        abs_residuals = np.sort(np.abs(residuals_per_bin))
        m = abs_residuals.size

        k = int(np.ceil((m + 1) * self.confidence_level)) - 1
        k = int(np.clip(k, 0, m - 1))

        q = abs_residuals[k]

        interval = np.array([-q, q], dtype=float)
        indices = np.array([k, k], dtype=int)

        return interval, indices

    def _compute_asymmetric_interval(
        self,
        residuals_per_bin: np.ndarray,
        label_idx: int,
        bin_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute equal-tailed asymmetric conformal offsets [q_low, q_high].

        Residuals are signed:

            residual = y - pred

        q_low controls the lower tail.
        q_high controls the upper tail.

        The construction is conservative and uses order statistics rather than
        interpolated percentiles.
        """
        sorted_residuals = np.sort(residuals_per_bin)
        m = sorted_residuals.size
        alpha = 1.0 - self.confidence_level

        # Lower tail target: alpha / 2
        # Upper tail target: alpha / 2
        #
        # Zero-based indices.
        k_low = int(np.floor((m + 1) * (alpha / 2.0)))
        k_high = int(np.ceil((m + 1) * (1.0 - alpha / 2.0))) - 1

        k_low = int(np.clip(k_low, 0, m - 1))
        k_high = int(np.clip(k_high, 0, m - 1))

        q_low = sorted_residuals[k_low]
        q_high = sorted_residuals[k_high]

        if q_low > q_high:
            raise ValueError(
                f"Invalid asymmetric interval for label {label_idx}, bin {bin_idx}: "
                f"q_low={q_low} is greater than q_high={q_high}."
            )

        if self.warn_on_extreme_quantiles and (k_low == 0 or k_high == m - 1):
            warnings.warn(
                "Asymmetric calibration is using an extreme residual "
                f"for label {label_idx}, bin {bin_idx}. "
                f"m={m}, k_low={k_low}, k_high={k_high}. "
                "This usually means the bin has limited samples for equal-tailed "
                "calibration and the interval may be conservative.",
                RuntimeWarning,
                stacklevel=2,
            )

        interval = np.array([q_low, q_high], dtype=float)
        indices = np.array([k_low, k_high], dtype=int)

        return interval, indices

    @staticmethod
    def _validate_residuals_per_bin(
        residuals_per_bin: np.ndarray,
        label_idx: int,
        bin_idx: int,
    ) -> None:
        """
        Validate residuals from one label/bin pair.
        """
        if residuals_per_bin.ndim != 1:
            raise ValueError(
                f"Residuals in bin {bin_idx} for label {label_idx} must be 1D."
            )

        if not np.all(np.isfinite(residuals_per_bin)):
            raise ValueError(
                f"Residuals in bin {bin_idx} for label {label_idx} "
                "must contain only finite values."
            )
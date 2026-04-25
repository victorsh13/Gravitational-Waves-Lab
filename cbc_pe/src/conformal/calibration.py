import numpy as np


class ConformalIntervalCalibrator:
    def __init__(
        self,
        confidence_level: float = 0.90,
        interval_mode: str = "symmetric",
        min_samples_per_bin: int = 10,
    ) -> None:
        """
        Compute conformal calibration offsets for each label/bin pair.

        Parameters
        ----------
        confidence_level : float
            Target coverage level (1 - alpha), must be in (0, 1).
        interval_mode : str
            "symmetric"  -> calibrate on absolute residuals and return [-q, q]
            "asymmetric" -> calibrate lower/upper offsets separately from signed residuals
        min_samples_per_bin : int
            Minimum number of residuals required in each bin.
        """
        valid_modes = {"symmetric", "asymmetric"}

        if not (0 < confidence_level < 1):
            raise ValueError("confidence_level must be between 0 and 1.")

        if interval_mode not in valid_modes:
            raise ValueError(f"interval_mode must be one of {sorted(valid_modes)}.")

        if min_samples_per_bin < 1:
            raise ValueError("min_samples_per_bin must be at least 1.")

        self.confidence_level = confidence_level
        self.interval_mode = interval_mode
        self.min_samples_per_bin = min_samples_per_bin

        self.intervals_ = None
        self.bin_counts_ = None
        self.is_fitted_ = False

    def fit(self, grouped_residuals):
        """
        Fit calibration offsets from residuals already grouped by label and bin.

        Parameters
        ----------
        grouped_residuals : np.ndarray of shape (n_labels, n_bins), dtype=object
            Each cell must contain a 1D array-like of residuals for one label/bin pair.

        Returns
        -------
        self
        """
        if grouped_residuals is None:
            raise ValueError("grouped_residuals must not be None.")

        grouped_residuals = np.asarray(grouped_residuals, dtype=object)

        if grouped_residuals.ndim != 2:
            raise ValueError(
                "grouped_residuals must be a 2D object array with shape (n_labels, n_bins)."
            )

        n_labels, n_bins = grouped_residuals.shape
        intervals = np.empty((n_labels, n_bins, 2), dtype=float)
        bin_counts = np.empty((n_labels, n_bins), dtype=int)

        for label_index in range(n_labels):
            for bin_index in range(n_bins):
                residuals_per_bin = np.asarray(
                    grouped_residuals[label_index, bin_index], dtype=float
                )

                if residuals_per_bin.ndim != 1:
                    raise ValueError(
                        f"Residuals in bin {bin_index} for label {label_index} must be 1D."
                    )

                if not np.all(np.isfinite(residuals_per_bin)):
                    raise ValueError(
                        f"Residuals in bin {bin_index} for label {label_index} "
                        "must contain only finite values."
                    )

                m = residuals_per_bin.size
                bin_counts[label_index, bin_index] = m

                if m < self.min_samples_per_bin:
                    raise ValueError(
                        f"Bin {bin_index} for label {label_index} has {m} samples, "
                        f"but at least {self.min_samples_per_bin} are required."
                    )

                intervals[label_index, bin_index] = self._compute_bin_interval(
                    residuals_per_bin
                )

        self.intervals_ = intervals
        self.bin_counts_ = bin_counts
        self.is_fitted_ = True
        return self

    def _compute_bin_interval(self, residuals_per_bin: np.ndarray) -> np.ndarray:
        """
        Compute [lower_offset, upper_offset] for a single bin.

        symmetric:
            Uses absolute residuals and returns [-q, q], where q is the
            conformal order statistic for the requested confidence level.

        asymmetric:
            Uses signed residuals and returns equal-tailed offsets
            [q_low, q_high], allocating alpha/2 to each tail.
        """
        rpb = np.sort(residuals_per_bin)
        m = rpb.size
        alpha = 1.0 - self.confidence_level

        if self.interval_mode == "symmetric":
            abs_rpb = np.sort(np.abs(residuals_per_bin))

            k = int(np.ceil((m + 1) * self.confidence_level)) - 1
            k = np.clip(k, 0, m - 1)

            q = abs_rpb[k]
            bin_interval = np.array([-q, q], dtype=float)

        elif self.interval_mode == "asymmetric":
            # Equal-tailed interval from signed residuals:
            # lower tail mass ~ alpha/2, upper tail mass ~ alpha/2
            k_low = int(np.floor((m + 1) * (alpha / 2.0)))
            k_high = int(np.ceil((m + 1) * (1.0 - alpha / 2.0))) - 1

            k_low = np.clip(k_low, 0, m - 1)
            k_high = np.clip(k_high, 0, m - 1)

            q_low = rpb[k_low]
            q_high = rpb[k_high]

            if q_low > q_high:
                raise ValueError(
                    f"Invalid interval: q_low={q_low} is greater than q_high={q_high}."
                )

            bin_interval = np.array([q_low, q_high], dtype=float)

        else:
            raise RuntimeError(
                f"Unsupported interval_mode={self.interval_mode!r}. "
                "This should have been validated in __init__."
            )

        return bin_interval
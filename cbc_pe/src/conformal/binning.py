from __future__ import annotations
import warnings

import numpy as np


class QuantileBinner:
    """
    Quantile-based binner for Mondrian conformal regression.

    This class computes bin edges from calibration binning scores and then
    assigns calibration/test samples to bins.

    Binning is performed independently for each label.
    """

    def __init__(
        self,
        n_bins: int,
        rng: np.random.Generator | None = None,
        apply_jitter: bool = False,
        jitter_variation: float = 1e-10,
        warn_on_degenerate_edges: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        n_bins : int
            Number of bins. Must be greater than 1.

        rng : np.random.Generator | None
            Random number generator used for jitter. If None, a default generator
            is created.

        apply_jitter : bool
            If True, add tiny uniform noise to binning scores while computing
            the bin edges. This helps break ties at quantile edges.

        jitter_variation : float
            Half-width of the uniform jitter interval:
            U(-jitter_variation, +jitter_variation).

        warn_on_degenerate_edges : bool
            If True, warn when repeated quantile edges are detected.
        """
        if n_bins < 2:
            raise ValueError("n_bins must be greater than 1.")

        if jitter_variation <= 0:
            raise ValueError("jitter_variation must be greater than 0.")

        if rng is None:
            rng = np.random.default_rng()

        self.n_bins = n_bins
        self.rng = rng
        self.apply_jitter = apply_jitter
        self.jitter_variation = jitter_variation
        self.warn_on_degenerate_edges = warn_on_degenerate_edges

        self.bin_edges_per_label_: np.ndarray | None = None

    def set_bin_edges(self, binning_scores: np.ndarray) -> "QuantileBinner":
        """
        Compute quantile bin edges from calibration binning scores.

        Parameters
        ----------
        binning_scores : np.ndarray, shape (n_samples, n_labels)
            Scores used to define Mondrian bins.

            Examples:
            - prediction taxonomy: CNN predictions
            - difficulty taxonomy: difficulty scores

        Returns
        -------
        self
            The fitted binner.
        """
        binning_scores = self._validate_binning_scores(
            binning_scores,
            context="set_bin_edges",
        )

        n_samples, _ = binning_scores.shape

        if n_samples < self.n_bins:
            raise ValueError(
                "The number of samples is smaller than n_bins. "
                "This can lead to empty bins. Decrease n_bins."
            )

        scores_for_edges = np.array(binning_scores, copy=True, dtype=float)

        if self.apply_jitter:
            jitter = self.rng.uniform(
                low=-self.jitter_variation,
                high=self.jitter_variation,
                size=scores_for_edges.shape,
            )
            scores_for_edges = scores_for_edges + jitter

        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)

        # Shape: (n_bins + 1, n_labels)
        self.bin_edges_per_label_ = np.quantile(
            scores_for_edges,
            quantiles,
            axis=0,
        )

        if self.warn_on_degenerate_edges:
            self._warn_if_degenerate_edges()

        return self

    def get_bin_indices(self, binning_scores: np.ndarray) -> np.ndarray:
        """
        Assign samples to the already-computed bin edges.

        Parameters
        ----------
        binning_scores : np.ndarray, shape (n_samples, n_labels)
            Scores to assign to bins.

        Returns
        -------
        bin_indices : np.ndarray, shape (n_samples, n_labels)
            Integer bin indices in [0, n_bins).
        """
        if self.bin_edges_per_label_ is None:
            raise ValueError("Bin edges have not been computed. Call set_bin_edges() first.")

        binning_scores = self._validate_binning_scores(
            binning_scores,
            context="get_bin_indices",
        )

        if binning_scores.shape[1] != self.bin_edges_per_label_.shape[1]:
            raise ValueError(
                "binning_scores have a different number of labels than "
                "the computed bin edges."
            )

        n_samples, n_labels = binning_scores.shape
        bin_indices = np.empty((n_samples, n_labels), dtype=int)

        for label_idx in range(n_labels):
            # Remove the first and last edges because they are the global min/max.
            # np.digitize uses only internal thresholds to assign bins.
            internal_edges = self.bin_edges_per_label_[1:-1, label_idx]

            bin_indices[:, label_idx] = np.digitize(
                binning_scores[:, label_idx],
                internal_edges,
                right=False,
            )

        return bin_indices

    def bin_edges_and_indices(self, binning_scores: np.ndarray) -> np.ndarray:
        """
        Compute bin edges from the provided scores and return bin indices
        for those same scores.

        This is used for the calibration set.

        Parameters
        ----------
        binning_scores : np.ndarray, shape (n_samples, n_labels)
            Calibration binning scores.

        Returns
        -------
        bin_indices : np.ndarray, shape (n_samples, n_labels)
            Bin indices for the calibration samples.
        """
        self.set_bin_edges(binning_scores)
        return self.get_bin_indices(binning_scores)

    @staticmethod
    def _validate_binning_scores(
        binning_scores: np.ndarray,
        context: str,
    ) -> np.ndarray:
        """
        Validate binning scores and convert them to float arrays.
        """
        binning_scores = np.asarray(binning_scores, dtype=float)

        if binning_scores.ndim != 2:
            raise ValueError(f"binning_scores must be a 2D array during {context}.")

        if binning_scores.shape[1] == 0:
            raise ValueError("binning_scores must have at least one label.")

        if not np.all(np.isfinite(binning_scores)):
            raise ValueError("binning_scores must contain only finite values.")

        return binning_scores

    def _warn_if_degenerate_edges(self) -> None:
        """
        Warn if repeated quantile edges are present.

        Repeated edges can happen when many binning scores are identical or nearly
        identical. This can lead to empty bins or very uneven bins.
        """
        if self.bin_edges_per_label_ is None:
            return

        n_labels = self.bin_edges_per_label_.shape[1]

        for label_idx in range(n_labels):
            unique_edges = np.unique(self.bin_edges_per_label_[:, label_idx])

            if unique_edges.size < self.n_bins + 1:
                warnings.warn(
                    "Degenerate quantile edges detected for "
                    f"label {label_idx}: only {unique_edges.size} unique edges "
                    f"for {self.n_bins + 1} requested edges. "
                    "This can create empty bins. Consider enabling jitter, "
                    "reducing n_bins, or using a different taxonomy.",
                    RuntimeWarning,
                    stacklevel=2,
                )


class BinGrouper:
    """
    Group calibration residuals by label and Mondrian bin.
    """

    def group_by_bin(
        self,
        residuals: np.ndarray,
        bin_indices: np.ndarray,
        n_bins: int,
    ) -> np.ndarray:
        """
        Assign calibration residuals to bins.

        Parameters
        ----------
        residuals : np.ndarray, shape (n_samples, n_labels)
            Calibration residuals, usually y_cal - pred_cal.

        bin_indices : np.ndarray, shape (n_samples, n_labels)
            Bin index for each sample and label.

        n_bins : int
            Total number of bins.

        Returns
        -------
        grouped_residuals : np.ndarray, shape (n_labels, n_bins), dtype=object
            grouped_residuals[label_idx, bin_idx] contains a 1D array with the
            residuals assigned to that label/bin pair.
        """
        residuals = np.asarray(residuals, dtype=float)
        bin_indices = np.asarray(bin_indices)

        if n_bins < 1:
            raise ValueError("n_bins must be at least 1.")

        if residuals.ndim != 2:
            raise ValueError("residuals must be a 2D array.")

        if bin_indices.ndim != 2:
            raise ValueError("bin_indices must be a 2D array.")

        if residuals.shape != bin_indices.shape:
            raise ValueError("residuals and bin_indices must have the same shape.")

        if residuals.shape[1] == 0:
            raise ValueError("residuals must have at least one label.")

        if not np.all(np.isfinite(residuals)):
            raise ValueError("residuals must contain only finite values.")

        bin_indices = self._ensure_integer_bin_indices(bin_indices)

        if np.any(bin_indices < 0) or np.any(bin_indices >= n_bins):
            raise ValueError("bin_indices contain values outside [0, n_bins).")

        n_labels = residuals.shape[1]

        grouped_residuals = np.empty((n_labels, n_bins), dtype=object)

        for label_idx in range(n_labels):
            for bin_idx in range(n_bins):
                mask = bin_indices[:, label_idx] == bin_idx
                grouped_residuals[label_idx, bin_idx] = residuals[mask, label_idx]

        return grouped_residuals

    @staticmethod
    def _ensure_integer_bin_indices(bin_indices: np.ndarray) -> np.ndarray:
        """
        Ensure bin indices are integer-valued.

        Allows arrays like [0.0, 1.0, 2.0], but rejects non-integer values
        like [0.0, 1.5, 2.0].
        """
        if np.issubdtype(bin_indices.dtype, np.integer):
            return bin_indices

        bin_indices_as_int = bin_indices.astype(int)

        if np.all(bin_indices == bin_indices_as_int):
            return bin_indices_as_int

        raise ValueError("bin_indices must contain integer values.")
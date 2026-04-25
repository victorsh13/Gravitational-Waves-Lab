import numpy as np

class QuantileBinner:
    def __init__(
        self,
        n_bins: int,
        rng: np.random.Generator | None = None,
        apply_jitter: bool = False,
        jitter_variation: float = 1e-10,
    ) -> None:
        """
        Initialize a QuantileBinner object.

        Parameters
        ----------
        n_bins : int
            The number of bins to partition the binning_scores into. Must be greater than 1. Default is 6.
        rng : np.random.Generator | None
            The random number generator to use. If None, use the default random number generator.
        apply_jitter : bool
            If True, apply a small variation on the binning_scores. This can help to break ties at the bin edges, avoiding degenerated quantiles.
        jitter_variation : float
            The amount of variation to apply to the bin edges. Must be greater than 0.
        """
        if n_bins < 2:
            raise ValueError("The number of bins (n_bins) must be greater than 1.")
        
        if jitter_variation <= 0:
            raise ValueError("The jitter variation must be greater than 0.")

        if rng is None:
            rng = np.random.default_rng()

        self.n_bins = n_bins
        self.rng = rng
        self.apply_jitter = apply_jitter
        self.jitter_variation = jitter_variation  # Small uniform perturbation applied to binning_scores during fit to break ties at quantile edges.
        self.bin_edges_per_label_ = None


    def set_bin_edges(self, binning_scores):
        """
        Compute the edges of the bins for a given set of binning_scores.

        Parameters
        ----------
        binning_scores : numpy.ndarray
            The scores used for binning.

        Returns
        -------
        edges : numpy.ndarray
            The edges of the bins for each label. With shape (n_bins + 1, n_labels).
        """
        binning_scores = np.asarray(binning_scores)

        if binning_scores.ndim != 2:
            raise ValueError("The binning_scores must be a 2D array.")
        
        if not np.all(np.isfinite(binning_scores)):
            raise ValueError("binning_scores must contain only finite binning_scores.")
        
        if binning_scores.shape[0] < self.n_bins:
            raise ValueError("The number of samples (n_samples) is smaller than the number of bins (n_bins). This would lead to empty bins. Consider decreasing the number of bins.")
           
        fit_binning_scores = np.copy(binning_scores)

        if self.apply_jitter: 
            for i in range(fit_binning_scores.shape[1]):
                fit_binning_scores[:,i] = fit_binning_scores[:,i] + self.rng.uniform(-self.jitter_variation, self.jitter_variation, size=len(fit_binning_scores[:,i]))
        
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
        self.bin_edges_per_label_ = np.quantile(fit_binning_scores, quantiles, axis=0) # Array of shape (n_bins + 1, n_labels)
        

        return self
    
    def get_bin_indices(self, binning_scores):
        """
        Compute the bin index for a given set of binning_scores.

        Parameters
        ----------
        binning_scores : numpy.ndarray
            The binning_scores of the model.

        Returns
        -------
        bin_index : numpy.ndarray
            The bin index for each prediction. With shape (n_samples, n_labels).
        """
        binning_scores = np.asarray(binning_scores)

        if not np.all(np.isfinite(binning_scores)):
            raise ValueError("binning_scores must contain only finite values.")
        if binning_scores.ndim != 2:
            raise ValueError("The binning_scores must be a 2D array.")
        if self.bin_edges_per_label_ is None:
            raise ValueError("The bin edges must be computed first. Call fit() method first.")
        if binning_scores.shape[1] != self.bin_edges_per_label_.shape[1]:
            raise ValueError("binning_scores have a different number of labels than the fitted bin edges.")
        
        
        bin_indices = np.empty_like(binning_scores, dtype=int)
    
        for i in range(binning_scores.shape[1]):
            internal_edges = self.bin_edges_per_label_[1:-1, i]
            bin_indices[:, i] = np.digitize(binning_scores[:,i], internal_edges, right=False) # This returns the bin index as 0, 1... n_bins-1
            
            # This could be used laterto asign the binning_scores to the right bin
            #preds_in_right_bins = [binning_scores[:,i][indices == j] for j in range(1, len(self.bin_edges_per_label_[i]))]
            #bin_index_array.append(preds_in_right_bins)

        return bin_indices

    def bin_edges_and_indices(self, binning_scores):
        """
        Compute the bin edges from the binning_scores and then compute the bin index for the same values.

        Parameters
        ----------
        binning_scores : numpy.ndarray
            The binning_scores of the model.

        Returns
        -------
        bin_indices : numpy.ndarray
            The bin index for each prediction. With shape (n_samples, n_labels).

        """
        self.set_bin_edges(binning_scores) # Compute the bin edges from the binning_scores

        return self.get_bin_indices(binning_scores) # Compute the bin index for the same binning_scores
            


## CLASS FOR ASSIGNING binning_scores TO BINS

class BinGrouper:

    def __init__(
            self):
        pass
        
            
    def group_by_bin(self, residuals: np.ndarray, bin_indices: np.ndarray) -> np.ndarray:
        """
        Assign residuals to bins.

        Parameters
        ----------
        residuals : numpy.ndarray
            The residual values to be grouped in the right bins. With shape (n_samples, n_labels).
        bin_indices : numpy.ndarray
            The bin index for each binning_score. With shape (n_samples, n_labels).

        Returns
        -------
        grouped_residuals : numpy.ndarray
            The residuals grouped by bins. With a nested structure of arrays. (Outer: labels, Inner: bins, Content: residuals)
        """
        residuals = np.asarray(residuals)
        bin_indices = np.asarray(bin_indices)
        n_bins = np.max(bin_indices) + 1
        
        if not np.all(np.isfinite(residuals)):
            raise ValueError("residuals must contain only finite values.")
        
        if residuals.ndim != 2 or bin_indices.ndim != 2:
            raise ValueError("residuals and bin_indices must be 2D arrays.")
        
        if residuals.shape != bin_indices.shape:
            raise ValueError("The residuals and bin indices must have the same shape.")
        
        if residuals.shape[1] == 0:
            raise ValueError("The residuals must have at least one label.")
        
        if np.any(bin_indices < 0) or np.any(bin_indices >= n_bins):
            raise ValueError("bin_indices contain values outside the valid range.")
        
        grouped_residuals = np.empty((residuals.shape[1], n_bins), dtype=object)

        for label_idx in range(residuals.shape[1]):
            for bin_idx in range(n_bins):
                grouped_residuals[label_idx, bin_idx] = residuals[:, label_idx][bin_indices[:, label_idx] == bin_idx]

        return grouped_residuals
    


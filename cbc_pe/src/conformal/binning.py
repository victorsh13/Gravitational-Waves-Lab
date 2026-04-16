import numpy as np
import warnings

class PredictionBinner:
    def __init__(
        self,
        n_bins: int = 6,
        rng: np.random.Generator | None = None,
        apply_jitter: bool = False,
        jitter_variation: float = 1e-10,
    ) -> None:
        """
        Initialize a PredictionBinner object.

        Parameters
        ----------
        n_bins : int
            The number of bins to partition the predictions into. Must be greater than 1. Default is 6.
        rng : np.random.Generator | None
            The random number generator to use. If None, use the default random number generator.
        apply_jitter : bool
            If True, apply a small variation on the predictions. This can help to reduce the overfitting of the bin edges, avoiding ties at edges.
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
        self.jitter_variation = jitter_variation  # Small uniform perturbation applied to predictions during fit to break ties at quantile edges.
        self.bin_edges = None


    def fit(self, predictions):
        """
        Compute the edges of the bins for a given set of predictions.

        Parameters
        ----------
        predictions : numpy.ndarray
            The predictions of the model.

        Returns
        -------
        edges : numpy.ndarray
            The edges of the bins for each label. With shape (n_bins + 1, n_labels).
        """
        predictions = np.asarray(predictions)

        if predictions.ndim != 2:
            raise ValueError("The predictions must be a 2D array.")
        
        if not np.all(np.isfinite(predictions)):
            raise ValueError("Predictions must contain only finite values.")
        
        if predictions.shape[0] < self.n_bins:
            warnings.warn("The number of samples (n_samples) is smaller than the number of bins (n_bins). This would lead to empty bins. Consider decreasing the number of bins.")
           
        fit_predictions = np.copy(predictions)

        if self.apply_jitter: 
            for i in range(fit_predictions.shape[1]):
                fit_predictions[:,i] = fit_predictions[:,i] + self.rng.uniform(-self.jitter_variation, self.jitter_variation, size=len(fit_predictions[:,i]))
        
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
        self.bin_edges = np.quantile(fit_predictions, quantiles, axis=0) # Array of shape (n_bins + 1, n_labels)
        

        return self
    
    def transform(self, predictions):
        """
        Compute the bin index for a given set of predictions.

        Parameters
        ----------
        predictions : numpy.ndarray
            The predictions of the model.

        Returns
        -------
        bin_index : numpy.ndarray
            The bin index for each prediction. With shape (n_samples, n_labels).
        """
        predictions = np.asarray(predictions)

        if not np.all(np.isfinite(predictions)):
            raise ValueError("Predictions must contain only finite values.")
        if predictions.ndim != 2:
            raise ValueError("The predictions must be a 2D array.")
        if self.bin_edges is None:
            raise ValueError("The bin edges must be computed first. Call fit() method first.")
        if predictions.shape[1] != self.bin_edges.shape[1]:
            raise ValueError("Predictions have a different number of labels than the fitted bin edges.")
        
        
        bin_indices = np.empty_like(predictions, dtype=int)
    
        for i in range(predictions.shape[1]):
            internal_edges = self.bin_edges[1:-1, i]
            bin_indices[:, i] = np.digitize(predictions[:,i], internal_edges, right=False) # This returns the bin index as 0, 1... n_bins-1
            
            # This could be used laterto asign the predictions to the right bin
            #preds_in_right_bins = [predictions[:,i][indices == j] for j in range(1, len(self.bin_edges[i]))]
            #bin_index_array.append(preds_in_right_bins)

        return bin_indices

    def fit_transform(self, predictions):
        """
        Compute the bin edges from the predictions and then compute the bin index for the same predictions.

        Parameters
        ----------
        predictions : numpy.ndarray
            The predictions of the model.

        Returns
        -------
        bin_indices : numpy.ndarray
            The bin index for each prediction. With shape (n_samples, n_labels).

        """
        self.fit(predictions) # Compute the bin edges from the predictions

        return self.transform(predictions) # Compute the bin index for the same predictions
            


## CLASS FOR ASSIGNING PREDICTIONS TO BINS

class BinGrouper:
    def __init__(
            self,
            n_bins: int = 6, 
            ) -> None:
        """
        Initialize a BinAssigner object.

        Parameters
        ----------
        n_bins : int
            The number of bins to partition the values into. Must be greater than 1. Default is 6.
        """
        if n_bins < 2:
            raise ValueError("The number of bins (n_bins) must be greater than 1.")

        self.n_bins = n_bins

    def group_by_bin(self, values: np.ndarray, bin_indices: np.ndarray):
        """
        Assign predictions to bins.

        Parameters
        ----------
        values : numpy.ndarray
            The values to be grouped in the right bins. With shape (n_samples, n_labels).
        bin_indices : numpy.ndarray
            The bin index for each value. With shape (n_samples, n_labels).

        Returns
        -------
        grouped_values : numpy.ndarray
            The values grouped by bins. With a nested structure of arrays. (Outer: labels, Inner: bins, Content: values)
        """
        values = np.asarray(values)
        bin_indices = np.asarray(bin_indices)
        
        if not np.all(np.isfinite(values)):
            raise ValueError("Values must contain only finite values.")
        
        if values.ndim != 2 or bin_indices.ndim != 2:
            raise ValueError("values and bin_indices must be 2D arrays.")
        
        if values.shape != bin_indices.shape:
            raise ValueError("The values and bin indices must have the same shape.")
        
        if values.shape[1] == 0:
            raise ValueError("The values must have at least one label.")
        
        if np.any(bin_indices < 0) or np.any(bin_indices >= self.n_bins):
            raise ValueError("bin_indices contain values outside the valid range.")
        
        grouped_values = []

        for i in range(values.shape[1]):
            grouped_values_label = [values[:,i][bin_indices[:, i] == j] for j in range(self.n_bins)]
            grouped_values.append(grouped_values_label)

        return np.array(grouped_values, dtype=object)
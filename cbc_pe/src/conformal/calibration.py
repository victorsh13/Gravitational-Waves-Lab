
import numpy as np
import warnings

class ConformalIntervalCalibrator:
    def __init__(
            self,
            confidence_level: float = 0.90,
            interval_mode: str = "symmetric", 
            ) -> None:
        """
        Initialize a ConformalIntervalCalibrator object.
        The function fit() computes the error intervals for each bin.

        Parameters
        ----------
        confidence_level : float
            The confidence level for the intervals (1-alpha). Default is 0.90.
        interval_mode : str
            The mode for the intervals. Can be "symmetric" or "asymmetric". Default is "symmetric".
        """
        implemented_interval_modes = ["symmetric"] # TO DO:"asymmetric"

        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError("The confidence level must be between 0 and 1.")

        if interval_mode not in implemented_interval_modes:
            raise ValueError(f"The interval mode must be in {implemented_interval_modes}.")

        self.confidence_level = confidence_level
        self.interval_mode = interval_mode
        self.intervals_ = None

    def fit(self, grouped_residuals):

        grouped_residuals = np.asarray(grouped_residuals, dtype=object)

        if grouped_residuals.ndim != 2:
            raise ValueError("The grouped values must be a 2D array. The first dimension corresponds to labels, and the second dimension corresponds to bins. In each cell, there should be an array of values that belong to that label and bin.")
        
        n_labels, n_bins = grouped_residuals.shape[0], grouped_residuals.shape[1]
        intervals = np.empty((n_labels, n_bins, 2), dtype=float)

        for label_index in range(n_labels):
            for bin_index in range(n_bins):
                residuals_per_bin = np.abs(np.asarray(grouped_residuals[label_index, bin_index], dtype=float))
                if len(residuals_per_bin) == 0:
                    intervals[label_index, bin_index] = np.array([np.nan, np.nan])
                    warnings.warn(f"No values in bin {bin_index} for label {label_index}. Setting interval to NaN.")
                    continue
                
                # In conformal prediction is common to use the explicit order statistic, you
                # can control the 'k' index. Reduces ambiguity when interpolating. 
                rpb = np.sort(residuals_per_bin)
                m = len(rpb)
                k = int(np.ceil((m + 1) * self.confidence_level)) - 1
                k = min(max(k, 0), m - 1)
                q = rpb[k]

                intervals[label_index, bin_index] = [-q, q]
                
        self.intervals_ = intervals

        return self

    
  

    
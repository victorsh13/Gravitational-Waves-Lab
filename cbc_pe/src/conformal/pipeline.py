"""
Mondrian Regression Pipeline:

1. Binning: 
    QuantileBinner (creates bins indices for the values) -> BinGrouper (groups values/residuals into bins) 
2. Calibration:
    ConformalIntervalCalibrator (computes the error intervals for each bin. The lower and upper bounds of the intervals are also computed.)
3. Prediction:
    CoverageEvaluator (computes the metrics (coverage, width, etc.) for the intervals)
"""

import numpy as np
from dataclasses import dataclass
from .difficulty import DifficultyEstimator
from .binning import QuantileBinner, BinGrouper
from .calibration import ConformalIntervalCalibrator
from .metrics import CoverageEvaluator
from .apply import apply_indices

@dataclass(frozen=True)
class MondrianResults:
    bin_indices_cal: np.ndarray
    bin_indices_test: np.ndarray
    intervals: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    metrics: dict
    grouped_residuals_cal: np.ndarray
    binner: QuantileBinner
    calibrator: ConformalIntervalCalibrator
    taxonomy_mode: str
    interval_mode: str
    binning_scores_cal: np.ndarray | None = None
    binning_scores_test: np.ndarray | None = None



def run_mondrian_regression(
        pred_cal: np.ndarray,
        pred_test: np.ndarray,
        y_cal: np.ndarray,
        y_test: np.ndarray,
        n_bins: int,
        cal_embedding: np.ndarray | None = None,
        target_embedding: np.ndarray | None = None,
        n_neighbors: int | None = None,
        confidence_level: float = 0.90,
        apply_jitter: bool = False,
        jitter_variation: float = 1e-10,
        interval_mode: str = "symmetric",
        taxonomy_mode: str = "prediction",
        ) -> MondrianResults:
    """
    Run the Mondrian Regression algorithm.

    Parameters
    ----------
    pred_cal : np.ndarray
        The predictions for the calibration set. Must be a 2D array of shape: (n_calibration_samples, n_labels).
    pred_test : np.ndarray
        The predictions for the test set. Must be a 2D array of shape: (n_test_samples, n_labels).
    y_cal : numpy.ndarray
        The true values for the calibration set.
    y_test : numpy.ndarray
        The true values for the test set.
    n_bins : int
        The number of bins to partition the values into. Must be greater than 1. Default is 6.
    cal_embedding : np.ndarray, optional
        The embedding for the calibration set. If not provided and taxonomy_mode is "difficulty", a ValueError will be raised. 
    target_embedding : np.ndarray, optional
        The embedding for the target set. If not provided, and taxonomy_mode is "difficulty", a ValueError will be raised.
    n_neighbors : int, optional
        The number of neighbors to consider for the difficulty estimation. If not provided and taxonomy_mode is "difficulty", a ValueError will be raised.
    confidence_level : float
        The confidence level for the intervals (1-alpha). Default is 0.90.
    apply_jitter : bool
        If True, apply a small variation on the values. This can help to break ties at the bin edges, avoiding degenerated quantiles.
    jitter_variation : float
        The amount of variation to apply to the bin edges. Must be greater than 0.
    interval_mode : str
        The mode for the intervals. Can be "symmetric" or "asymmetric".
    taxonomy_mode : str
        The taxonomy for the intervals. Can be "prediction" or "difficulty". This is used to determine the binning scores and the metrics to compute. Default is "prediction".

    Returns
    -------
    MondrianResults
        A MondrianResults object containing the bin indices for the calibration and test sets, the intervals, the lower and upper bounds, the metrics, and the grouped residuals.
    """
    ## -------------------- INPUT VALIDATION ------------------- ##
    if pred_cal.ndim != 2:
        raise ValueError("pred_cal must be a 2D array of shape: (n_calibration_samples, n_labels).")
    if pred_test.ndim != 2:
        raise ValueError("pred_test must be a 2D array of shape: (n_test_samples, n_labels).")
    if y_cal.ndim != 2:
        raise ValueError("y_cal must be a 2D array of shape: (n_calibration_samples, n_labels).")
    if y_test.ndim != 2:
        raise ValueError("y_test must be a 2D array of shape: (n_test_samples, n_labels).")
    if pred_cal.shape != y_cal.shape:
        raise ValueError("pred_cal and y_cal must have the same shape.")
    if pred_test.shape != y_test.shape:
        raise ValueError("pred_test and y_test must have the same shape.")
    if pred_cal.shape[1] != pred_test.shape[1]:
        raise ValueError("pred_cal and pred_test must have the same number of labels.")
    if y_cal.shape[1] != y_test.shape[1]:
        raise ValueError("y_cal and y_test must have the same number of labels.")
    if n_bins < 2:
        raise ValueError("n_bins must be greater than 1.")
    if jitter_variation <= 0:
        raise ValueError("jitter_variation must be greater than 0.")
    if taxonomy_mode not in {"prediction", "difficulty"}:
        raise ValueError("taxonomy_mode must be either 'prediction' or 'difficulty'.")
    if interval_mode not in {"symmetric", "asymmetric"}:
        raise ValueError("interval_mode must be either 'symmetric' or 'asymmetric'.")
    
    
    # -------------------- ---------- BINNING SCORE COMPUTATION -----------  ###################
  
    # Compute the residuals and the scores.
    residuals_cal = y_cal - pred_cal # The residuals are the difference between the true values and the predictions. This is used to compute the intervals for the calibration set.
    binning_scores_cal, binning_scores_test = compute_binning_scores(
                                            taxonomy_mode=taxonomy_mode,
                                            pred_cal=pred_cal,
                                            pred_test=pred_test,
                                            cal_embedding=cal_embedding,
                                            target_embedding=target_embedding,
                                            cal_residuals=residuals_cal,
                                            n_neighbors=n_neighbors

    )
    # The binning scores are the values that we will use to compute the bin indices. In this case, we are using the predictions as the binning scores, but we could also use the difficulty scores if we want to use a different taxonomy.

    ################### ---------- BINNING -----------  ###################
    #  1. Compute the bin indices for the calibration set
    binner = QuantileBinner(
                        n_bins=n_bins, 
                        apply_jitter=apply_jitter, 
                        jitter_variation=jitter_variation
    )

    bin_indices_cal = binner.bin_edges_and_indices(binning_scores_cal) # Gives the bin indices for the calibration set (we also need the bin edges to compute the bin indices for the test set)
    bin_indices_test = binner.get_bin_indices(binning_scores_test) # Gives the bin indices for the test set
    
    #  2.Group the residuals into bins given the bin indices 
    grouper = BinGrouper() 
    grouped_residuals_cal = grouper.group_by_bin(
                                    residuals=residuals_cal, 
                                    bin_indices=bin_indices_cal,
                                    n_bins=n_bins
    )


    ################### ---------- CALIBRATION ----------- ###################
    #  3. Compute the intervals
    calibrator = ConformalIntervalCalibrator(
                        confidence_level=confidence_level, 
                        interval_mode=interval_mode
    )
    calibrator.fit(grouped_residuals_cal) # Compute the error intervals for each bin.
    intervals = calibrator.intervals_


    ###################  --------- TEST INTERVAL CONSTRUCTION ---------  ###################
    # Now once the calibration is done, we can compute the intervals for the test set  
    lower, upper = apply_indices(
                values=pred_test, 
                bin_indices=bin_indices_test, 
                intervals=intervals
    )

    ################# --------- EVALUATION (METRICS) ---------  #################
    # Compute the metrics for the test set
    evaluator = CoverageEvaluator(confidence_level=confidence_level)
    metrics = evaluator.evaluate_intervals(
        y=y_test,
        lower_bound=lower,
        upper_bound=upper,
        bin_indices=bin_indices_test,
        n_bins=n_bins,
    )
    
   
    
    return MondrianResults(
                bin_indices_cal=bin_indices_cal,
                bin_indices_test=bin_indices_test,
                intervals=intervals,
                lower=lower,
                upper=upper,
                metrics=metrics,
                grouped_residuals_cal=grouped_residuals_cal,
                binner=binner,
                calibrator=calibrator,
                taxonomy_mode=taxonomy_mode,
                interval_mode=interval_mode,
                binning_scores_cal=binning_scores_cal,
                binning_scores_test=binning_scores_test,
    )


def compute_binning_scores(
            taxonomy_mode: str,
            pred_cal: np.ndarray | None = None,
            pred_test: np.ndarray | None = None,
            cal_embedding: np.ndarray | None = None, 
            target_embedding: np.ndarray | None = None,
            cal_residuals: np.ndarray | None = None,
            n_neighbors: int | None = None,
            ) -> tuple[np.ndarray, np.ndarray]:
    
    if taxonomy_mode == "prediction":

        if pred_cal is None or pred_test is None:
            raise ValueError("For prediction taxonomy, pred_cal and pred_test must be provided.")
        if pred_cal.ndim != 2:
            raise ValueError("pred_cal must be a 2D array of shape: (n_calibration_samples, n_labels).")
        if pred_test.ndim != 2:
            raise ValueError("pred_test must be a 2D array of shape: (n_test_samples, n_labels).")
        
        return pred_cal, pred_test
    
    elif taxonomy_mode == "difficulty":

        if cal_embedding is None or cal_residuals is None or target_embedding is None:
            raise ValueError("For difficulty taxonomy, cal_embedding, cal_residuals, and target_embedding must be provided.")
        if cal_embedding.ndim != 2:
            raise ValueError("cal_embedding must be a 2D array of shape: (n_calibration_samples, embedding_dim).")
        if target_embedding.ndim != 2:
            raise ValueError("target_embedding must be a 2D array of shape: (n_test_samples, embedding_dim).")
        if cal_residuals.ndim != 2:
            raise ValueError("cal_residuals must be a 2D array of shape: (n_calibration_samples, n_labels).")
        if n_neighbors is None or n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer.")

        difficulty_model = DifficultyEstimator(n_neighbors=n_neighbors )
        difficulty_model.calibrate_estimator(cal_embedding=cal_embedding, cal_residuals=cal_residuals)
        scores_cal =np.array(difficulty_model.compute_calibration_difficulty())
        scores_test = np.array(difficulty_model.compute_target_difficulty(target_embedding=target_embedding))
        
        return scores_cal, scores_test
    
    
  
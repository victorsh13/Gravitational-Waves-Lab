from __future__ import annotations
"""
Mondrian conformal regression pipeline.

Steps
-----
1. Compute calibration residuals.
2. Compute Mondrian binning scores:
   - prediction-based taxonomy, or
   - difficulty-based taxonomy.
3. Fit quantile bin edges on calibration scores.
4. Assign calibration and test samples to bins.
5. Group calibration residuals by label/bin.
6. Fit conformal offsets per label/bin.
7. Apply calibrated offsets to test predictions.
8. Evaluate coverage and interval efficiency.
"""

from dataclasses import dataclass

import numpy as np

from .apply import apply_indices
from .binning import BinGrouper, QuantileBinner
from .calibration import ConformalIntervalCalibrator
from .difficulty import DifficultyEstimator
from .taxonomy import compute_binning_scores
from .metrics import CoverageEvaluator


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
    binning_scores_cal: np.ndarray
    binning_scores_test: np.ndarray
    difficulty_model: DifficultyEstimator | None = None


@dataclass(frozen=True)
class FittedMondrianRegressor:
    """
    Fitted Mondrian conformal system.

    This object contains only calibration-derived state and can therefore
    be reused to construct intervals for arbitrary target predictions
    without requiring target labels.
    """

    binner: QuantileBinner
    calibrator: ConformalIntervalCalibrator

    taxonomy_mode: str
    interval_mode: str
    confidence_level: float
    n_bins: int

    bin_indices_cal: np.ndarray
    grouped_residuals_cal: np.ndarray
    binning_scores_cal: np.ndarray

    difficulty_model: DifficultyEstimator | None = None


@dataclass(frozen=True)
class MondrianPrediction:
    """
    Mondrian interval prediction for an arbitrary target set.

    No ground-truth labels are stored or required.
    """

    lower: np.ndarray
    upper: np.ndarray
    bin_indices: np.ndarray
    binning_scores: np.ndarray


def fit_mondrian(
    pred_cal: np.ndarray,
    y_cal: np.ndarray,
    n_bins: int,
    cal_embedding: np.ndarray | None = None,
    n_neighbors: int | None = None,
    confidence_level: float = 0.90,
    apply_jitter: bool = False,
    jitter_variation: float = 1e-10,
    interval_mode: str = "symmetric",
    taxonomy_mode: str = "prediction",
    min_samples_per_bin: int = 10,
) -> FittedMondrianRegressor:
    """
    Fit a Mondrian conformal system using calibration data only.

    Parameters
    ----------
    pred_cal : np.ndarray, shape (n_calibration_samples, n_labels)
        Calibration predictions.

    y_cal : np.ndarray, shape (n_calibration_samples, n_labels)
        True calibration labels.

    n_bins : int
        Number of Mondrian bins per label.

    cal_embedding : np.ndarray | None
        Calibration embeddings. Required for taxonomy_mode="difficulty".

    n_neighbors : int | None
        Number of nearest neighbours used for difficulty estimation.

    confidence_level : float
        Target conformal coverage level.

    apply_jitter : bool
        Whether to apply tiny jitter when fitting quantile bin edges.

    jitter_variation : float
        Uniform jitter half-width.

    interval_mode : {"symmetric", "asymmetric"}
        Conformal interval calibration mode.

    taxonomy_mode : {"prediction", "difficulty"}
        Mondrian taxonomy.

    min_samples_per_bin : int
        Minimum number of calibration residuals required per label/bin.

    Returns
    -------
    FittedMondrianRegressor
        Calibration-derived Mondrian system ready to be applied to
        arbitrary target predictions.
    """
    pred_cal = np.asarray(pred_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=float)

    if pred_cal.ndim != 2:
        raise ValueError("pred_cal must be a 2D array.")

    if y_cal.ndim != 2:
        raise ValueError("y_cal must be a 2D array.")

    if pred_cal.shape != y_cal.shape:
        raise ValueError("pred_cal and y_cal must have the same shape.")

    if not np.all(np.isfinite(pred_cal)):
        raise ValueError("pred_cal must contain only finite values.")

    if not np.all(np.isfinite(y_cal)):
        raise ValueError("y_cal must contain only finite values.")

    if pred_cal.shape[0] < n_bins:
        raise ValueError(
            "The calibration set must contain at least n_bins samples."
        )

    if n_bins < 2:
        raise ValueError("n_bins must be greater than 1.")

    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    if jitter_variation <= 0:
        raise ValueError(
            "jitter_variation must be greater than 0."
        )

    if interval_mode not in {"symmetric", "asymmetric"}:
        raise ValueError(
            "interval_mode must be either 'symmetric' or 'asymmetric'."
        )

    if taxonomy_mode not in {"prediction", "difficulty"}:
        raise ValueError(
            "taxonomy_mode must be either 'prediction' or 'difficulty'."
        )

    if min_samples_per_bin < 1:
        raise ValueError(
            "min_samples_per_bin must be at least 1."
        )

    residuals_cal = y_cal - pred_cal

    difficulty_model = None

    if taxonomy_mode == "prediction":
        binning_scores_cal = pred_cal

    else:
        if cal_embedding is None:
            raise ValueError(
                "For difficulty taxonomy, cal_embedding must be provided."
            )

        if n_neighbors is None or n_neighbors < 1:
            raise ValueError(
                "For difficulty taxonomy, n_neighbors must be "
                "a positive integer."
            )

        cal_embedding = np.asarray(
            cal_embedding,
            dtype=float,
        )

        difficulty_model = DifficultyEstimator(
            n_neighbors=n_neighbors
        )

        difficulty_model.calibrate_estimator(
            cal_embedding=cal_embedding,
            cal_residuals=residuals_cal,
        )

        binning_scores_cal = (
            difficulty_model.compute_calibration_difficulty()
        )

    binner = QuantileBinner(
        n_bins=n_bins,
        apply_jitter=apply_jitter,
        jitter_variation=jitter_variation,
    )

    bin_indices_cal = binner.bin_edges_and_indices(
        binning_scores_cal
    )

    grouper = BinGrouper()

    grouped_residuals_cal = grouper.group_by_bin(
        residuals=residuals_cal,
        bin_indices=bin_indices_cal,
        n_bins=n_bins,
    )

    calibrator = ConformalIntervalCalibrator(
        confidence_level=confidence_level,
        interval_mode=interval_mode,
        min_samples_per_bin=min_samples_per_bin,
    )

    calibrator.fit(grouped_residuals_cal)

    return FittedMondrianRegressor(
        binner=binner,
        calibrator=calibrator,
        taxonomy_mode=taxonomy_mode,
        interval_mode=interval_mode,
        confidence_level=confidence_level,
        n_bins=n_bins,
        bin_indices_cal=bin_indices_cal,
        grouped_residuals_cal=grouped_residuals_cal,
        binning_scores_cal=binning_scores_cal,
        difficulty_model=difficulty_model,
    )


def apply_mondrian(
    fitted: FittedMondrianRegressor,
    pred_target: np.ndarray,
    target_embedding: np.ndarray | None = None,
) -> MondrianPrediction:
    """
    Apply a fitted Mondrian conformal system to arbitrary target predictions.

    Ground-truth target labels are neither required nor used.
    """
    pred_target = np.asarray(
        pred_target,
        dtype=float,
    )

    if pred_target.ndim != 2:
        raise ValueError(
            "pred_target must be a 2D array."
        )

    if pred_target.shape[1] != fitted.binning_scores_cal.shape[1]:
        raise ValueError(
            "pred_target must have the same number of labels "
            "as the calibration predictions."
        )

    if not np.all(np.isfinite(pred_target)):
        raise ValueError(
            "pred_target must contain only finite values."
        )

    if fitted.taxonomy_mode == "prediction":
        binning_scores_target = pred_target

    elif fitted.taxonomy_mode == "difficulty":
        if fitted.difficulty_model is None:
            raise ValueError(
                "The fitted difficulty taxonomy does not contain "
                "a calibrated DifficultyEstimator."
            )

        if target_embedding is None:
            raise ValueError(
                "target_embedding is required for difficulty taxonomy."
            )

        target_embedding = np.asarray(
            target_embedding,
            dtype=float,
        )

        binning_scores_target = (
            fitted.difficulty_model.compute_target_difficulty(
                target_embedding=target_embedding,
            )
        )

    else:
        raise RuntimeError(
            f"Unsupported taxonomy_mode={fitted.taxonomy_mode!r}."
        )

    bin_indices_target = fitted.binner.get_bin_indices(
        binning_scores_target
    )

    lower, upper = apply_indices(
        values=pred_target,
        bin_indices=bin_indices_target,
        intervals=fitted.calibrator.intervals_,
    )

    return MondrianPrediction(
        lower=lower,
        upper=upper,
        bin_indices=bin_indices_target,
        binning_scores=binning_scores_target,
    )


def evaluate_mondrian(
    fitted: FittedMondrianRegressor,
    prediction: MondrianPrediction,
    y_true: np.ndarray,
    tolerance_sigmas: tuple[int, ...] = (1, 2, 3),
) -> dict:
    """
    Evaluate previously constructed Mondrian intervals against known truth.

    This function is intentionally separate from ``apply_mondrian`` so that
    interval construction can be used for real events where ground-truth
    parameters are unavailable.
    """
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    evaluator = CoverageEvaluator(
        confidence_level=fitted.confidence_level,
        tolerance_sigmas=tolerance_sigmas,
    )

    return evaluator.evaluate_intervals(
        y=y_true,
        lower_bound=prediction.lower,
        upper_bound=prediction.upper,
        bin_indices=prediction.bin_indices,
        n_bins=fitted.n_bins,
    )


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
    min_samples_per_bin: int = 10,
    tolerance_sigmas: tuple[int, ...] = (1, 2, 3),
) -> MondrianResults:
    """
    Run Mondrian conformal regression.

    Parameters
    ----------
    pred_cal : np.ndarray, shape (n_calibration_samples, n_labels)
        Model predictions for the calibration set.
    pred_test : np.ndarray, shape (n_test_samples, n_labels)
        Model predictions for the test set.
    y_cal : np.ndarray, shape (n_calibration_samples, n_labels)
        True calibration labels.
    y_test : np.ndarray, shape (n_test_samples, n_labels)
        True test labels.
    n_bins : int
        Number of Mondrian bins per label.
    cal_embedding : np.ndarray | None
        Calibration embeddings. Required for taxonomy_mode="difficulty".
    target_embedding : np.ndarray | None
        Target/test embeddings. Required for taxonomy_mode="difficulty".
    n_neighbors : int | None
        Number of nearest neighbors for difficulty estimation.
    confidence_level : float
        Target coverage level.
    apply_jitter : bool
        Whether to add tiny jitter when computing quantile bin edges.
    jitter_variation : float
        Uniform jitter scale.
    interval_mode : {"symmetric", "asymmetric"}
        Interval calibration mode. Default is "symmetric".
    taxonomy_mode : {"prediction", "difficulty"}
        Mondrian taxonomy mode. Default is "prediction".
    min_samples_per_bin : int
        Minimum calibration residual count required per label/bin.
    tolerance_sigmas : tuple[int, ...]
        Sigma levels for nominal normal tolerance bands.

    Returns
    -------
    MondrianResults
        Full pipeline output.
    """
    pred_cal = np.asarray(pred_cal, dtype=float)
    pred_test = np.asarray(pred_test, dtype=float)
    y_cal = np.asarray(y_cal, dtype=float)
    y_test = np.asarray(y_test, dtype=float)

    _validate_mondrian_inputs(
        pred_cal=pred_cal,
        pred_test=pred_test,
        y_cal=y_cal,
        y_test=y_test,
        n_bins=n_bins,
        confidence_level=confidence_level,
        jitter_variation=jitter_variation,
        interval_mode=interval_mode,
        taxonomy_mode=taxonomy_mode,
        min_samples_per_bin=min_samples_per_bin,
    )

    residuals_cal = y_cal - pred_cal

    (
        binning_scores_cal,
        binning_scores_test,
        difficulty_model,
    ) = compute_binning_scores(
        taxonomy_mode=taxonomy_mode,
        pred_cal=pred_cal,
        pred_test=pred_test,
        cal_embedding=cal_embedding,
        target_embedding=target_embedding,
        cal_residuals=residuals_cal,
        n_neighbors=n_neighbors,
    )

    binner = QuantileBinner(
        n_bins=n_bins,
        apply_jitter=apply_jitter,
        jitter_variation=jitter_variation,
    )

    bin_indices_cal = binner.bin_edges_and_indices(binning_scores_cal)
    bin_indices_test = binner.get_bin_indices(binning_scores_test)

    grouper = BinGrouper()
    grouped_residuals_cal = grouper.group_by_bin(
        residuals=residuals_cal,
        bin_indices=bin_indices_cal,
        n_bins=n_bins,
    )

    calibrator = ConformalIntervalCalibrator(
        confidence_level=confidence_level,
        interval_mode=interval_mode,
        min_samples_per_bin=min_samples_per_bin,
    )
    calibrator.fit(grouped_residuals_cal)
    intervals = calibrator.intervals_

    lower, upper = apply_indices(
        values=pred_test,
        bin_indices=bin_indices_test,
        intervals=intervals,
    )

    evaluator = CoverageEvaluator(
        confidence_level=confidence_level,
        tolerance_sigmas=tolerance_sigmas,
    )
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
        difficulty_model=difficulty_model,
    )


def _validate_mondrian_inputs(
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
    y_cal: np.ndarray,
    y_test: np.ndarray,
    n_bins: int,
    confidence_level: float,
    jitter_variation: float,
    interval_mode: str,
    taxonomy_mode: str,
    min_samples_per_bin: int,
) -> None:
    if pred_cal.ndim != 2:
        raise ValueError("pred_cal must be a 2D array.")
    if pred_test.ndim != 2:
        raise ValueError("pred_test must be a 2D array.")
    if y_cal.ndim != 2:
        raise ValueError("y_cal must be a 2D array.")
    if y_test.ndim != 2:
        raise ValueError("y_test must be a 2D array.")

    if pred_cal.shape != y_cal.shape:
        raise ValueError("pred_cal and y_cal must have the same shape.")
    if pred_test.shape != y_test.shape:
        raise ValueError("pred_test and y_test must have the same shape.")

    if pred_cal.shape[1] != pred_test.shape[1]:
        raise ValueError("pred_cal and pred_test must have the same number of labels.")
    if y_cal.shape[1] != y_test.shape[1]:
        raise ValueError("y_cal and y_test must have the same number of labels.")

    if pred_cal.shape[0] < n_bins:
        raise ValueError("The calibration set must contain at least n_bins samples.")

    if not np.all(np.isfinite(pred_cal)):
        raise ValueError("pred_cal must contain only finite values.")
    if not np.all(np.isfinite(pred_test)):
        raise ValueError("pred_test must contain only finite values.")
    if not np.all(np.isfinite(y_cal)):
        raise ValueError("y_cal must contain only finite values.")
    if not np.all(np.isfinite(y_test)):
        raise ValueError("y_test must contain only finite values.")

    if n_bins < 2:
        raise ValueError("n_bins must be greater than 1.")

    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be between 0 and 1.")

    if jitter_variation <= 0:
        raise ValueError("jitter_variation must be greater than 0.")

    if interval_mode not in {"symmetric", "asymmetric"}:
        raise ValueError("interval_mode must be either 'symmetric' or 'asymmetric'.")

    if taxonomy_mode not in {"prediction", "difficulty"}:
        raise ValueError("taxonomy_mode must be either 'prediction' or 'difficulty'.")

    if min_samples_per_bin < 1:
        raise ValueError("min_samples_per_bin must be at least 1.")
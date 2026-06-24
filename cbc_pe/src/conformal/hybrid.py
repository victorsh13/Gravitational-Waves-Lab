from __future__ import annotations

"""
Hybrid Mondrian conformal regression.

The hybrid taxonomy combines two operational variables available at inference:

    prediction bin × difficulty bin

Prediction and difficulty edges are fitted independently on the calibration
set using quantile bins. The two integer indices are then combined into one
joint Mondrian group:

    joint_bin = prediction_bin * n_difficulty_bins + difficulty_bin

No test labels or test residuals are used to define the taxonomy.
"""

from dataclasses import dataclass

import numpy as np

from .apply import apply_indices
from .binning import BinGrouper, QuantileBinner
from .calibration import ConformalIntervalCalibrator
from .difficulty import DifficultyEstimator
from .metrics import CoverageEvaluator


class HybridQuantileBinner:
    """
    Fit independent quantile bins for prediction and difficulty, then combine
    them into joint Mondrian groups.
    """

    def __init__(
        self,
        n_prediction_bins: int,
        n_difficulty_bins: int,
        apply_jitter: bool = False,
        jitter_variation: float = 1e-10,
        rng: np.random.Generator | None = None,
    ) -> None:
        if n_prediction_bins < 2:
            raise ValueError("n_prediction_bins must be greater than 1.")
        if n_difficulty_bins < 2:
            raise ValueError("n_difficulty_bins must be greater than 1.")

        if rng is None:
            rng = np.random.default_rng()

        self.n_prediction_bins = int(n_prediction_bins)
        self.n_difficulty_bins = int(n_difficulty_bins)
        self.n_joint_bins = (
            self.n_prediction_bins * self.n_difficulty_bins
        )

        # Use separate RNG streams so jitter, when enabled, is reproducible
        # without reusing exactly the same random sequence.
        seed_pred = int(rng.integers(0, np.iinfo(np.uint32).max))
        seed_diff = int(rng.integers(0, np.iinfo(np.uint32).max))

        self.prediction_binner = QuantileBinner(
            n_bins=self.n_prediction_bins,
            rng=np.random.default_rng(seed_pred),
            apply_jitter=apply_jitter,
            jitter_variation=jitter_variation,
        )

        self.difficulty_binner = QuantileBinner(
            n_bins=self.n_difficulty_bins,
            rng=np.random.default_rng(seed_diff),
            apply_jitter=apply_jitter,
            jitter_variation=jitter_variation,
        )

        self.is_fitted_ = False

    def fit(
        self,
        prediction_scores_cal: np.ndarray,
        difficulty_scores_cal: np.ndarray,
    ) -> "HybridQuantileBinner":
        prediction_scores_cal, difficulty_scores_cal = (
            self._validate_pair(
                prediction_scores_cal,
                difficulty_scores_cal,
                context="fit",
            )
        )

        self.prediction_binner.set_bin_edges(prediction_scores_cal)
        self.difficulty_binner.set_bin_edges(difficulty_scores_cal)
        self.is_fitted_ = True
        return self

    def get_component_indices(
        self,
        prediction_scores: np.ndarray,
        difficulty_scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted_:
            raise ValueError("HybridQuantileBinner must be fitted first.")

        prediction_scores, difficulty_scores = self._validate_pair(
            prediction_scores,
            difficulty_scores,
            context="get_component_indices",
        )

        prediction_indices = self.prediction_binner.get_bin_indices(
            prediction_scores
        )
        difficulty_indices = self.difficulty_binner.get_bin_indices(
            difficulty_scores
        )

        return prediction_indices, difficulty_indices

    def get_joint_indices(
        self,
        prediction_scores: np.ndarray,
        difficulty_scores: np.ndarray,
    ) -> np.ndarray:
        prediction_indices, difficulty_indices = self.get_component_indices(
            prediction_scores,
            difficulty_scores,
        )

        joint_indices = (
            prediction_indices * self.n_difficulty_bins
            + difficulty_indices
        )

        return joint_indices.astype(int, copy=False)

    def fit_transform(
        self,
        prediction_scores_cal: np.ndarray,
        difficulty_scores_cal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.fit(
            prediction_scores_cal=prediction_scores_cal,
            difficulty_scores_cal=difficulty_scores_cal,
        )

        prediction_indices, difficulty_indices = self.get_component_indices(
            prediction_scores_cal,
            difficulty_scores_cal,
        )

        joint_indices = (
            prediction_indices * self.n_difficulty_bins
            + difficulty_indices
        ).astype(int, copy=False)

        return joint_indices, prediction_indices, difficulty_indices

    @staticmethod
    def _validate_pair(
        prediction_scores: np.ndarray,
        difficulty_scores: np.ndarray,
        context: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        prediction_scores = np.asarray(prediction_scores, dtype=float)
        difficulty_scores = np.asarray(difficulty_scores, dtype=float)

        if prediction_scores.ndim != 2:
            raise ValueError(
                f"prediction_scores must be 2D during {context}."
            )
        if difficulty_scores.ndim != 2:
            raise ValueError(
                f"difficulty_scores must be 2D during {context}."
            )
        if prediction_scores.shape != difficulty_scores.shape:
            raise ValueError(
                "prediction_scores and difficulty_scores must have "
                f"the same shape during {context}."
            )
        if not np.all(np.isfinite(prediction_scores)):
            raise ValueError("prediction_scores must contain finite values.")
        if not np.all(np.isfinite(difficulty_scores)):
            raise ValueError("difficulty_scores must contain finite values.")

        return prediction_scores, difficulty_scores


@dataclass(frozen=True)
class HybridMondrianResults:
    bin_indices_cal: np.ndarray
    bin_indices_test: np.ndarray
    prediction_bin_indices_cal: np.ndarray
    prediction_bin_indices_test: np.ndarray
    difficulty_bin_indices_cal: np.ndarray
    difficulty_bin_indices_test: np.ndarray
    intervals: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    metrics: dict
    grouped_residuals_cal: np.ndarray
    hybrid_binner: HybridQuantileBinner
    calibrator: ConformalIntervalCalibrator
    difficulty_model: DifficultyEstimator
    interval_mode: str
    n_prediction_bins: int
    n_difficulty_bins: int
    n_joint_bins: int
    prediction_scores_cal: np.ndarray
    prediction_scores_test: np.ndarray
    difficulty_scores_cal: np.ndarray
    difficulty_scores_test: np.ndarray


def run_hybrid_mondrian_regression(
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
    y_cal: np.ndarray,
    y_test: np.ndarray,
    cal_embedding: np.ndarray,
    target_embedding: np.ndarray,
    n_prediction_bins: int,
    n_difficulty_bins: int,
    n_neighbors: int = 5,
    confidence_level: float = 0.90,
    apply_jitter: bool = False,
    jitter_variation: float = 1e-10,
    interval_mode: str = "asymmetric",
    min_samples_per_bin: int = 10,
    tolerance_sigmas: tuple[int, ...] = (1, 2, 3),
    random_seed: int = 123,
    standardize_embeddings: bool = False,
) -> HybridMondrianResults:
    """
    Run a hybrid prediction × difficulty Mondrian conformal pipeline.

    All bin edges are fitted only on calibration data. Test labels are used
    only for final evaluation.
    """
    pred_cal = np.asarray(pred_cal, dtype=float)
    pred_test = np.asarray(pred_test, dtype=float)
    y_cal = np.asarray(y_cal, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    cal_embedding = np.asarray(cal_embedding, dtype=float)
    target_embedding = np.asarray(target_embedding, dtype=float)

    _validate_hybrid_inputs(
        pred_cal=pred_cal,
        pred_test=pred_test,
        y_cal=y_cal,
        y_test=y_test,
        cal_embedding=cal_embedding,
        target_embedding=target_embedding,
        n_prediction_bins=n_prediction_bins,
        n_difficulty_bins=n_difficulty_bins,
        n_neighbors=n_neighbors,
        confidence_level=confidence_level,
        interval_mode=interval_mode,
        min_samples_per_bin=min_samples_per_bin,
    )

    residuals_cal = y_cal - pred_cal

    difficulty_model = DifficultyEstimator(
        n_neighbors=n_neighbors,
        standardize_embeddings=standardize_embeddings,
    )
    difficulty_model.calibrate_estimator(
        cal_embedding=cal_embedding,
        cal_residuals=residuals_cal,
    )

    difficulty_scores_cal = (
        difficulty_model.compute_calibration_difficulty()
    )
    difficulty_scores_test = difficulty_model.compute_target_difficulty(
        target_embedding=target_embedding
    )

    prediction_scores_cal = pred_cal
    prediction_scores_test = pred_test

    hybrid_binner = HybridQuantileBinner(
        n_prediction_bins=n_prediction_bins,
        n_difficulty_bins=n_difficulty_bins,
        apply_jitter=apply_jitter,
        jitter_variation=jitter_variation,
        rng=np.random.default_rng(random_seed),
    )

    (
        bin_indices_cal,
        prediction_bin_indices_cal,
        difficulty_bin_indices_cal,
    ) = hybrid_binner.fit_transform(
        prediction_scores_cal=prediction_scores_cal,
        difficulty_scores_cal=difficulty_scores_cal,
    )

    (
        prediction_bin_indices_test,
        difficulty_bin_indices_test,
    ) = hybrid_binner.get_component_indices(
        prediction_scores=prediction_scores_test,
        difficulty_scores=difficulty_scores_test,
    )

    bin_indices_test = (
        prediction_bin_indices_test * n_difficulty_bins
        + difficulty_bin_indices_test
    ).astype(int, copy=False)

    n_joint_bins = n_prediction_bins * n_difficulty_bins

    grouper = BinGrouper()
    grouped_residuals_cal = grouper.group_by_bin(
        residuals=residuals_cal,
        bin_indices=bin_indices_cal,
        n_bins=n_joint_bins,
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
        n_bins=n_joint_bins,
    )

    return HybridMondrianResults(
        bin_indices_cal=bin_indices_cal,
        bin_indices_test=bin_indices_test,
        prediction_bin_indices_cal=prediction_bin_indices_cal,
        prediction_bin_indices_test=prediction_bin_indices_test,
        difficulty_bin_indices_cal=difficulty_bin_indices_cal,
        difficulty_bin_indices_test=difficulty_bin_indices_test,
        intervals=intervals,
        lower=lower,
        upper=upper,
        metrics=metrics,
        grouped_residuals_cal=grouped_residuals_cal,
        hybrid_binner=hybrid_binner,
        calibrator=calibrator,
        difficulty_model=difficulty_model,
        interval_mode=interval_mode,
        n_prediction_bins=n_prediction_bins,
        n_difficulty_bins=n_difficulty_bins,
        n_joint_bins=n_joint_bins,
        prediction_scores_cal=prediction_scores_cal,
        prediction_scores_test=prediction_scores_test,
        difficulty_scores_cal=difficulty_scores_cal,
        difficulty_scores_test=difficulty_scores_test,
    )


def _validate_hybrid_inputs(
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
    y_cal: np.ndarray,
    y_test: np.ndarray,
    cal_embedding: np.ndarray,
    target_embedding: np.ndarray,
    n_prediction_bins: int,
    n_difficulty_bins: int,
    n_neighbors: int,
    confidence_level: float,
    interval_mode: str,
    min_samples_per_bin: int,
) -> None:
    arrays_2d = {
        "pred_cal": pred_cal,
        "pred_test": pred_test,
        "y_cal": y_cal,
        "y_test": y_test,
        "cal_embedding": cal_embedding,
        "target_embedding": target_embedding,
    }

    for name, array in arrays_2d.items():
        if array.ndim != 2:
            raise ValueError(f"{name} must be a 2D array.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain finite values.")

    if pred_cal.shape != y_cal.shape:
        raise ValueError("pred_cal and y_cal must have the same shape.")
    if pred_test.shape != y_test.shape:
        raise ValueError("pred_test and y_test must have the same shape.")
    if pred_cal.shape[1] != pred_test.shape[1]:
        raise ValueError(
            "Calibration and test predictions must have the same labels."
        )
    if cal_embedding.shape[0] != pred_cal.shape[0]:
        raise ValueError(
            "cal_embedding and pred_cal must have the same sample count."
        )
    if target_embedding.shape[0] != pred_test.shape[0]:
        raise ValueError(
            "target_embedding and pred_test must have the same sample count."
        )
    if cal_embedding.shape[1] != target_embedding.shape[1]:
        raise ValueError(
            "Calibration and target embeddings must have the same dimension."
        )

    if n_prediction_bins < 2:
        raise ValueError("n_prediction_bins must be greater than 1.")
    if n_difficulty_bins < 2:
        raise ValueError("n_difficulty_bins must be greater than 1.")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be between 0 and 1.")
    if interval_mode not in {"symmetric", "asymmetric"}:
        raise ValueError(
            "interval_mode must be 'symmetric' or 'asymmetric'."
        )
    if min_samples_per_bin < 1:
        raise ValueError("min_samples_per_bin must be at least 1.")

    n_joint_bins = n_prediction_bins * n_difficulty_bins
    if pred_cal.shape[0] < n_joint_bins:
        raise ValueError(
            "Calibration sample count must be at least the number of "
            "joint hybrid bins."
        )

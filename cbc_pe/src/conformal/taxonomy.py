import numpy as np

from .difficulty import DifficultyEstimator


def compute_binning_scores(
    taxonomy_mode: str,
    pred_cal: np.ndarray | None = None,
    pred_test: np.ndarray | None = None,
    cal_embedding: np.ndarray | None = None,
    target_embedding: np.ndarray | None = None,
    cal_residuals: np.ndarray | None = None,
    n_neighbors: int | None = None,
) -> tuple[np.ndarray, np.ndarray, DifficultyEstimator | None]:
    """
    Compute Mondrian binning scores for calibration and test sets.

    Parameters
    ----------
    taxonomy_mode : {"prediction", "difficulty"}
        Strategy used to define Mondrian bins.

    pred_cal : np.ndarray, shape (n_calibration_samples, n_labels)
        Calibration predictions. Required for taxonomy_mode="prediction".

    pred_test : np.ndarray, shape (n_test_samples, n_labels)
        Test predictions. Required for taxonomy_mode="prediction".

    cal_embedding : np.ndarray, shape (n_calibration_samples, embedding_dim)
        Calibration embeddings. Required for taxonomy_mode="difficulty".

    target_embedding : np.ndarray, shape (n_test_samples, embedding_dim)
        Test/target embeddings. Required for taxonomy_mode="difficulty".

    cal_residuals : np.ndarray, shape (n_calibration_samples, n_labels)
        Calibration residuals y_cal - pred_cal. Required for taxonomy_mode="difficulty".

    n_neighbors : int | None
        Number of nearest neighbors used by the difficulty estimator.

    Returns
    -------
    scores_cal : np.ndarray, shape (n_calibration_samples, n_labels)
        Calibration binning scores.

    scores_test : np.ndarray, shape (n_test_samples, n_labels)
        Test binning scores.

    difficulty_model : DifficultyEstimator | None
        Fitted difficulty estimator when taxonomy_mode="difficulty";
        otherwise None.
    """
    if taxonomy_mode == "prediction":
        return _prediction_binning_scores(
            pred_cal=pred_cal,
            pred_test=pred_test,
        )

    if taxonomy_mode == "difficulty":
        return _difficulty_binning_scores(
            cal_embedding=cal_embedding,
            target_embedding=target_embedding,
            cal_residuals=cal_residuals,
            n_neighbors=n_neighbors,
        )

    raise ValueError("taxonomy_mode must be either 'prediction' or 'difficulty'.")


def _prediction_binning_scores(
    pred_cal: np.ndarray | None,
    pred_test: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, None]:
    if pred_cal is None or pred_test is None:
        raise ValueError(
            "For prediction taxonomy, pred_cal and pred_test must be provided."
        )

    pred_cal = np.asarray(pred_cal, dtype=float)
    pred_test = np.asarray(pred_test, dtype=float)

    if pred_cal.ndim != 2:
        raise ValueError("pred_cal must be a 2D array.")
    if pred_test.ndim != 2:
        raise ValueError("pred_test must be a 2D array.")

    if pred_cal.shape[1] != pred_test.shape[1]:
        raise ValueError(
            "pred_cal and pred_test must have the same number of labels."
        )

    if not np.all(np.isfinite(pred_cal)):
        raise ValueError("pred_cal must contain only finite values.")
    if not np.all(np.isfinite(pred_test)):
        raise ValueError("pred_test must contain only finite values.")

    return pred_cal, pred_test, None


def _difficulty_binning_scores(
    cal_embedding: np.ndarray | None,
    target_embedding: np.ndarray | None,
    cal_residuals: np.ndarray | None,
    n_neighbors: int | None,
) -> tuple[np.ndarray, np.ndarray, DifficultyEstimator]:
    if cal_embedding is None or target_embedding is None or cal_residuals is None:
        raise ValueError(
            "For difficulty taxonomy, cal_embedding, target_embedding, "
            "and cal_residuals must be provided."
        )

    if n_neighbors is None or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer.")

    cal_embedding = np.asarray(cal_embedding, dtype=float)
    target_embedding = np.asarray(target_embedding, dtype=float)
    cal_residuals = np.asarray(cal_residuals, dtype=float)

    if cal_embedding.ndim != 2:
        raise ValueError("cal_embedding must be a 2D array.")
    if target_embedding.ndim != 2:
        raise ValueError("target_embedding must be a 2D array.")
    if cal_residuals.ndim != 2:
        raise ValueError("cal_residuals must be a 2D array.")

    if cal_embedding.shape[0] != cal_residuals.shape[0]:
        raise ValueError(
            "cal_embedding and cal_residuals must have the same number of samples."
        )

    if cal_embedding.shape[1] != target_embedding.shape[1]:
        raise ValueError(
            "cal_embedding and target_embedding must have the same embedding dimension."
        )

    if not np.all(np.isfinite(cal_embedding)):
        raise ValueError("cal_embedding must contain only finite values.")
    if not np.all(np.isfinite(target_embedding)):
        raise ValueError("target_embedding must contain only finite values.")
    if not np.all(np.isfinite(cal_residuals)):
        raise ValueError("cal_residuals must contain only finite values.")

    difficulty_model = DifficultyEstimator(n_neighbors=n_neighbors)
    difficulty_model.calibrate_estimator(
        cal_embedding=cal_embedding,
        cal_residuals=cal_residuals,
    )

    scores_cal = difficulty_model.compute_calibration_difficulty()
    scores_test = difficulty_model.compute_target_difficulty(
        target_embedding=target_embedding
    )

    return scores_cal, scores_test, difficulty_model
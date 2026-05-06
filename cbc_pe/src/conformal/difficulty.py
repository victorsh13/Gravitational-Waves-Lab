import numpy as np
from sklearn.neighbors import NearestNeighbors


class DifficultyEstimator:
    """
    Estimate local prediction difficulty from calibration residuals in embedding space.

    For each target sample, the difficulty score is computed as the local average
    of absolute calibration residuals among nearest calibration neighbors.

    The score is computed separately for each label.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        distance_weighted: bool = True,
        distance_eps: float = 1e-8,
        metric: str = "euclidean",
        standardize_embeddings: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        n_neighbors : int
            Number of calibration neighbors used to estimate difficulty.

        distance_weighted : bool
            If True, use inverse-distance weighted residual average.
            If False, use simple mean over neighbors.

        distance_eps : float
            Small constant added to distances when using inverse-distance weights.

        metric : str
            Distance metric passed to sklearn.neighbors.NearestNeighbors.

        standardize_embeddings : bool
            If True, standardize embedding dimensions using calibration mean/std
            before fitting the nearest-neighbor model.
        """
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be greater than or equal to 1.")

        if distance_eps <= 0:
            raise ValueError("distance_eps must be greater than 0.")

        self.n_neighbors = n_neighbors
        self.distance_weighted = distance_weighted
        self.distance_eps = distance_eps
        self.metric = metric
        self.standardize_embeddings = standardize_embeddings

        self.calibration_embeddings_ = None
        self.calibration_abs_residuals_ = None
        self.nn_model_ = None
        self.is_fitted_ = False

        self.embedding_mean_ = None
        self.embedding_std_ = None

        self.calibration_neighbor_indices_ = None
        self.calibration_neighbor_distances_ = None
        self.target_neighbor_indices_ = None
        self.target_neighbor_distances_ = None

    def calibrate_estimator(
        self,
        cal_embedding: np.ndarray,
        cal_residuals: np.ndarray,
    ) -> "DifficultyEstimator":
        """
        Store calibration embeddings/residuals and fit the nearest-neighbor index.

        Parameters
        ----------
        cal_embedding : np.ndarray, shape (n_calibration_samples, embedding_dim)
            Calibration embeddings.

        cal_residuals : np.ndarray, shape (n_calibration_samples, n_labels)
            Calibration residuals y_cal - pred_cal.

        Returns
        -------
        self
        """
        cal_embedding = np.asarray(cal_embedding, dtype=float)
        cal_residuals = np.asarray(cal_residuals, dtype=float)

        self._validate_calibration_inputs(
            cal_embedding=cal_embedding,
            cal_residuals=cal_residuals,
        )

        n_calibration_samples = cal_embedding.shape[0]

        if n_calibration_samples < self.n_neighbors + 1:
            raise ValueError(
                "The number of calibration samples must be at least "
                "n_neighbors + 1 so that calibration difficulty can exclude "
                "the sample itself."
            )

        if self.standardize_embeddings:
            self.embedding_mean_ = np.mean(cal_embedding, axis=0)
            self.embedding_std_ = np.std(cal_embedding, axis=0)

            # Avoid division by zero for constant embedding dimensions.
            self.embedding_std_ = np.where(
                self.embedding_std_ == 0.0,
                1.0,
                self.embedding_std_,
            )

            cal_embedding_for_nn = self._transform_embedding(cal_embedding)
        else:
            cal_embedding_for_nn = cal_embedding

        self.calibration_embeddings_ = cal_embedding_for_nn
        self.calibration_abs_residuals_ = np.abs(cal_residuals)

        self.nn_model_ = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
        )
        self.nn_model_.fit(self.calibration_embeddings_)

        self.is_fitted_ = True

        return self

    def compute_calibration_difficulty(self) -> np.ndarray:
        """
        Compute difficulty scores for the calibration set itself.

        Each calibration sample uses its nearest calibration neighbors,
        excluding itself.
        """
        self._check_is_fitted()

        difficulty_scores, distances, indices = self._compute_neighbor_based_difficulty(
            target_embedding=self.calibration_embeddings_,
            exclude_self=True,
        )

        self.calibration_neighbor_distances_ = distances
        self.calibration_neighbor_indices_ = indices

        return difficulty_scores

    def compute_target_difficulty(self, target_embedding: np.ndarray) -> np.ndarray:
        """
        Compute difficulty scores for a target/test set.

        Parameters
        ----------
        target_embedding : np.ndarray, shape (n_target_samples, embedding_dim)
            Target embeddings.

        Returns
        -------
        difficulty_scores : np.ndarray, shape (n_target_samples, n_labels)
            Difficulty scores per sample and label.
        """
        self._check_is_fitted()

        target_embedding = np.asarray(target_embedding, dtype=float)

        self._validate_target_embedding(target_embedding)

        if self.standardize_embeddings:
            target_embedding_for_nn = self._transform_embedding(target_embedding)
        else:
            target_embedding_for_nn = target_embedding

        difficulty_scores, distances, indices = self._compute_neighbor_based_difficulty(
            target_embedding=target_embedding_for_nn,
            exclude_self=False,
        )

        self.target_neighbor_distances_ = distances
        self.target_neighbor_indices_ = indices

        return difficulty_scores

    def _compute_neighbor_based_difficulty(
        self,
        target_embedding: np.ndarray,
        exclude_self: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute neighbor-based difficulty scores.

        Returns
        -------
        difficulty_scores : np.ndarray, shape (n_samples, n_labels)
        distances : np.ndarray, shape (n_samples, n_neighbors)
        indices : np.ndarray, shape (n_samples, n_neighbors)
        """
        n_neighbors_query = self.n_neighbors + 1 if exclude_self else self.n_neighbors

        distances, indices = self.nn_model_.kneighbors(
            target_embedding,
            n_neighbors=n_neighbors_query,
        )

        if exclude_self:
            distances = distances[:, 1:]
            indices = indices[:, 1:]

        neighbor_abs_residuals = self.calibration_abs_residuals_[indices]
        # shape: (n_samples, n_neighbors, n_labels)

        if not self.distance_weighted:
            difficulty_scores = np.mean(neighbor_abs_residuals, axis=1)
        else:
            weights = 1.0 / (distances + self.distance_eps)
            weights = weights / np.sum(weights, axis=1, keepdims=True)

            difficulty_scores = np.sum(
                weights[:, :, None] * neighbor_abs_residuals,
                axis=1,
            )

        if not np.all(np.isfinite(difficulty_scores)):
            raise ValueError("Computed difficulty scores contain non-finite values.")

        return difficulty_scores, distances, indices

    def _transform_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Standardize embeddings using calibration statistics.
        """
        if self.embedding_mean_ is None or self.embedding_std_ is None:
            raise ValueError("Embedding standardization statistics are not available.")

        return (embedding - self.embedding_mean_) / self.embedding_std_

    @staticmethod
    def _validate_calibration_inputs(
        cal_embedding: np.ndarray,
        cal_residuals: np.ndarray,
    ) -> None:
        if cal_embedding.ndim != 2:
            raise ValueError(
                "cal_embedding must be a 2D array of shape "
                "(n_calibration_samples, embedding_dim)."
            )

        if cal_residuals.ndim != 2:
            raise ValueError(
                "cal_residuals must be a 2D array of shape "
                "(n_calibration_samples, n_labels)."
            )

        if cal_embedding.shape[0] != cal_residuals.shape[0]:
            raise ValueError(
                "cal_embedding and cal_residuals must have the same number of samples."
            )

        if cal_embedding.shape[1] == 0:
            raise ValueError("cal_embedding must have at least one embedding dimension.")

        if cal_residuals.shape[1] == 0:
            raise ValueError("cal_residuals must have at least one label.")

        if not np.all(np.isfinite(cal_embedding)):
            raise ValueError("cal_embedding must contain only finite values.")

        if not np.all(np.isfinite(cal_residuals)):
            raise ValueError("cal_residuals must contain only finite values.")

    def _validate_target_embedding(self, target_embedding: np.ndarray) -> None:
        if target_embedding.ndim != 2:
            raise ValueError(
                "target_embedding must be a 2D array of shape "
                "(n_target_samples, embedding_dim)."
            )

        if target_embedding.shape[1] != self.calibration_embeddings_.shape[1]:
            raise ValueError(
                "The embedding dimension of target_embedding must match the "
                "embedding dimension of the calibration embeddings."
            )

        if not np.all(np.isfinite(target_embedding)):
            raise ValueError("target_embedding must contain only finite values.")

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise ValueError(
                "The difficulty estimator must be calibrated before computing "
                "difficulty scores."
            )
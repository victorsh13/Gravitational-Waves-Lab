import numpy as np
from sklearn.neighbors import NearestNeighbors


class DifficultyEstimator:
    def __init__(self, n_neighbors: int = 5) -> None:
        """
        Estimate difficulty scores from local calibration residuals in embedding space.

        The difficulty score of a sample is defined as the mean absolute residual
        of its nearest calibration neighbors, computed separately for each label.

        Parameters
        ----------
        n_neighbors : int, default=5
            Number of neighbors to use for difficulty estimation.
            Must be at least 1.
        """
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be greater than or equal to 1.")

        self.n_neighbors = n_neighbors
        
        self.calibration_embeddings_ = None
        self.calibration_abs_residuals_ = None
        self.nn_model_ = None
        self.is_fitted_ = False



    def calibrate_estimator(self, cal_embedding, cal_residuals):
        """
        Store calibration embeddings and absolute residuals, and fit the NN index.

        Parameters
        ----------
        cal_embedding : array-like of shape (n_calibration_samples, embedding_dim)
            Embeddings of the calibration samples.
        cal_residuals : array-like of shape (n_calibration_samples, n_labels)
            Residuals of the calibration samples.

        Returns
        -------
        self
        """
        cal_embedding = np.asarray(cal_embedding, dtype=float)
        cal_residuals = np.asarray(cal_residuals, dtype=float)

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
        if not np.isfinite(cal_embedding).all():
            raise ValueError("cal_embedding must contain only finite values.")
        if not np.isfinite(cal_residuals).all():
            raise ValueError("cal_residuals must contain only finite values.")

        n_calibration_samples = cal_embedding.shape[0]
        if n_calibration_samples <= self.n_neighbors:
            raise ValueError(
                "The number of calibration samples must be greater than n_neighbors "
                "so that calibration difficulty can exclude the sample itself."
            )

        self.calibration_embeddings_ = cal_embedding
        self.calibration_abs_residuals_ = np.abs(cal_residuals)

        # Fit with the full calibration reference set.
        self.nn_model_ = NearestNeighbors(n_neighbors=self.n_neighbors, metric="euclidean")
        self.nn_model_.fit(self.calibration_embeddings_)

        self.is_fitted_ = True
        return self



    def compute_calibration_difficulty(self):
        """
        Compute difficulty scores for the calibration set itself.

        Each calibration sample uses its k nearest neighbors in the calibration
        reference set, excluding the sample itself.

        Returns
        -------
        difficulty_scores : ndarray of shape (n_calibration_samples, n_labels)
            Difficulty scores for the calibration samples.
        """
        self._check_is_fitted()

        difficulty_scores = self._compute_neighbor_based_difficulty(
                                            target_embedding=self.calibration_embeddings_,
                                            exclude_self=True,
                                            distance_weighted=True,    
        )

        return difficulty_scores        
          



    def compute_target_difficulty(self, target_embedding):
        """
        Compute difficulty scores for a new target set.

        Parameters
        ----------
        target_embedding : array-like of shape (n_target_samples, embedding_dim)
            Embeddings of the target samples.

        Returns
        -------
        difficulty_scores : ndarray of shape (n_target_samples, n_labels)
            Difficulty scores for the target samples.
        """
        self._check_is_fitted()

        target_embedding = np.asarray(target_embedding, dtype=float)

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
        if not np.isfinite(target_embedding).all():
            raise ValueError("target_embedding must contain only finite values.")


        difficulty_scores = self._compute_neighbor_based_difficulty(
                                        target_embedding=target_embedding,
                                        exclude_self=False,
                                        distance_weighted=True,
        )  

        return difficulty_scores




    def _compute_neighbor_based_difficulty(
        self,
        target_embedding,
        exclude_self: bool,
        distance_weighted: bool,
    ):
        """
        Compute neighbor-based difficulty scores.

        Returns
        -------
        difficulty_scores : ndarray of shape (n_samples, n_labels)
            Weighted or unweighted mean absolute residual of nearest calibration
            neighbors, computed separately for each label.
        """
        n_neighbors_query = self.n_neighbors + 1 if exclude_self else self.n_neighbors

        distances, indices = self.nn_model_.kneighbors(
            target_embedding,
            n_neighbors=n_neighbors_query,
        )

        # For calibration samples, the closest neighbor is usually the point itself.
        # Remove it from both indices and distances.
        if exclude_self:
            distances = distances[:, 1:]
            indices = indices[:, 1:]

        # Shape: (n_samples, n_neighbors, n_labels)
        neighbor_abs_residuals = self.calibration_abs_residuals_[indices]

        if not distance_weighted:
            difficulty_scores = np.mean(neighbor_abs_residuals, axis=1)
        else:
            eps = 1e-8

            # Shape: (n_samples, n_neighbors)
            weights = 1.0 / (distances + eps)

            # Normalize weights per sample
            weights = weights / np.sum(weights, axis=1, keepdims=True)

            # Broadcast weights over labels:
            # (n_samples, n_neighbors, 1) * (n_samples, n_neighbors, n_labels)
            difficulty_scores = np.sum(
                weights[:, :, None] * neighbor_abs_residuals,
                axis=1,
            )

        return difficulty_scores



    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise ValueError(
                "The difficulty estimator must be calibrated before computing difficulty scores."
            )
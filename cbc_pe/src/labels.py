import numpy as np
from .parameters import CBCParameters

class LabelTransformer:
    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        """
        Initialize a LabelTransformer object.

        Parameters
        ----------
        mean : np.ndarray | None
            The mean of the labels. If None, the mean is not standardized.
        std : np.ndarray | None
            The standard deviation of the labels. If None, the standard deviation is not standardized.

        Raises
        ------
        ValueError
            If mean and std are not None and have different shapes or if std has zeros.

        Notes
        -----
        The mean and standard deviation are used to standardize the labels. The standardization is applied before the transformation.

        """
        if mean is not None and std is not None:
            if mean.shape != std.shape:
                raise ValueError("mean and std must have the same shape.")
            if np.any(std == 0):
                raise ValueError("std must not have zeros.")

        self.mean = None if mean is None else np.asarray(mean, dtype=float)
        self.std = None if std is None else np.asarray(std, dtype=float)

        if self.mean is not None and self.std is not None:
            if self.mean.shape != (3,) or self.std.shape != (3,):
                raise ValueError("mean and std must have shape (3,)")
            if np.any(self.std == 0):
                raise ValueError("std must not have zeros.")

    def transform(self, parameters: CBCParameters, standardize: bool = False) -> np.ndarray:
        """
        Transform the parameters into a set of labels.

        Parameters
        ----------
        parameters : CBCParameters
            The parameters of the binary compact object.
        standardize : bool
            Whether to standardize the labels. If True, the labels will be standardized to have a mean of 0 and a standard deviation of 1.

        Returns
        -------
        labels : np.ndarray
            The labels. The shape is (3,).
        """
        labels = np.array([
            parameters.chirp_mass,
            parameters.total_mass,
            parameters.chi_eff,
            ],
            dtype=float,
            )

        if standardize:
            if self.mean is None or self.std is None:
                raise ValueError("mean and std are required for standardization.")
            labels = (labels - self.mean) / self.std

        return labels
    
    def inverse_transform(self, labels: np.ndarray, standardize: bool = False) -> np.ndarray:
        """
        Reverse the standardization of the labels. Recovering the physical values of the labels.

        Parameters
        ----------
        labels : np.ndarray
            The labels. The shape is (3,).
        standardize : bool
            Whether to standardize the labels. If True, the labels will be standardized to have a mean of 0 and a standard deviation of 1.
        
        Returns
        -------
        physical_labels : np.ndarray
            The physical values of the labels. The shape is (3,).
        """

        if standardize:
            if self.mean is None or self.std is None:
                raise ValueError("mean and std are required to recover the physical values.")
            labels = labels * self.std + self.mean

        physical_labels = np.array([
            labels[0],
            labels[1],
            labels[2],
            ],
            dtype=float,
            )

        return physical_labels

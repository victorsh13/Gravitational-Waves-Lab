import numpy as np
from .parameters import CBCParameters

class LabelTransformer:
    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = None if mean is None else np.asarray(mean, dtype=float)
        self.std = None if std is None else np.asarray(std, dtype=float)

        if mean is not None and std is not None:
            if self.mean.shape != (3,) or self.std.shape != (3,):
                raise ValueError("mean and std must have shape (3,)")
            if np.any(self.std == 0):
                raise ValueError("std must not have zeros.")


    def transform(self, parameters: CBCParameters, standardize: bool = False) -> np.ndarray:
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

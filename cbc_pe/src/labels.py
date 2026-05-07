import numpy as np

from .parameters import CBCParameters


class LabelTransformer:
    """
    Transform CBCParameters into regression labels.

    The physical label vector is:

        [chirp_mass, total_mass, chi_eff]

    Optionally, labels can be standardized using mean/std computed from the
    training set only.
    """

    label_names = ("chirp_mass", "total_mass", "chi_eff")

    def __init__(
        self,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
        self.mean = None if mean is None else np.asarray(mean, dtype=float)
        self.std = None if std is None else np.asarray(std, dtype=float)

        if (self.mean is None) != (self.std is None):
            raise ValueError("mean and std must either both be provided or both be None.")

        if self.mean is not None:
            if self.mean.shape != (3,) or self.std.shape != (3,):
                raise ValueError("mean and std must have shape (3,).")

            if not np.all(np.isfinite(self.mean)):
                raise ValueError("mean must contain only finite values.")

            if not np.all(np.isfinite(self.std)):
                raise ValueError("std must contain only finite values.")

            if np.any(self.std <= 0):
                raise ValueError("std values must be positive.")

    def transform(
        self,
        parameters: CBCParameters,
        standardize: bool = False,
    ) -> np.ndarray:
        labels = self.to_physical_labels(parameters)

        if standardize:
            if self.mean is None or self.std is None:
                raise ValueError("mean and std are required for standardization.")

            labels = (labels - self.mean) / self.std

        return labels

    def inverse_transform(
        self,
        labels: np.ndarray,
        standardize: bool = False,
    ) -> np.ndarray:
        labels = np.asarray(labels, dtype=float)

        if labels.shape != (3,):
            raise ValueError("labels must have shape (3,).")

        if standardize:
            if self.mean is None or self.std is None:
                raise ValueError(
                    "mean and std are required to recover physical labels."
                )

            labels = labels * self.std + self.mean

        return labels.astype(float)

    @classmethod
    def to_physical_labels(cls, parameters: CBCParameters) -> np.ndarray:
        return np.array(
            [
                parameters.chirp_mass,
                parameters.total_mass,
                parameters.chi_eff,
            ],
            dtype=float,
        )

    def metadata(self) -> dict:
        return {
            "label_names": list(self.label_names),
            "standardization_available": self.mean is not None and self.std is not None,
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
        }
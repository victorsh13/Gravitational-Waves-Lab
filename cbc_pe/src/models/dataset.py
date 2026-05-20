from __future__ import annotations
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

class ArrayRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

        if self.X.ndim != 3:
            raise ValueError(f"Expected X with shape (N, C, T), got {self.X.shape}")

        if self.y.ndim != 2:
            raise ValueError(f"Expected y with shape (N, D), got {self.y.shape}")

        if len(self.X) != len(self.y):
            raise ValueError(f"X and y have different lengths: {len(self.X)} vs {len(self.y)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    


class HDF5RegressionDataset(Dataset):
    """
    PyTorch Dataset for lazy loading from an HDF5 file.

    Expected HDF5 structure:
        /X : shape (N, C, L), float32
        /y : shape (N, D), float32

    This class does not load the full dataset into RAM.
    Each __getitem__ reads only one sample.
    """

    def __init__(
        self,
        h5_path: str | Path,
        indices: np.ndarray,
        y_mean: np.ndarray | None = None,
        y_std: np.ndarray | None = None,
    ):
        self.h5_path = str(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)

        self.y_mean = None if y_mean is None else np.asarray(y_mean, dtype=np.float32)
        self.y_std = None if y_std is None else np.asarray(y_std, dtype=np.float32)

        if self.y_mean is not None and self.y_std is None:
            raise ValueError("If y_mean is provided, y_std must also be provided.")

        if self.y_std is not None and self.y_mean is None:
            raise ValueError("If y_std is provided, y_mean must also be provided.")

        self._file = None

        with h5py.File(self.h5_path, "r") as f:
            if "X" not in f:
                raise KeyError("HDF5 file does not contain dataset 'X'.")
            if "y" not in f:
                raise KeyError("HDF5 file does not contain dataset 'y'.")

            self.x_shape = f["X"].shape
            self.y_shape = f["y"].shape

        if self.x_shape[0] != self.y_shape[0]:
            raise ValueError(
                f"X and y have different number of samples: "
                f"{self.x_shape[0]} vs {self.y_shape[0]}"
            )

        if np.any(self.indices < 0) or np.any(self.indices >= self.x_shape[0]):
            raise ValueError("Some indices are outside the valid dataset range.")

    def _get_file(self):
        # Open lazily. This is important for DataLoader workers.
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        f = self._get_file()

        real_idx = int(self.indices[idx])

        X = f["X"][real_idx].astype(np.float32)
        y = f["y"][real_idx].astype(np.float32)

        if self.y_mean is not None and self.y_std is not None:
            y = (y - self.y_mean) / (self.y_std + 1e-8)

        return torch.from_numpy(X), torch.from_numpy(y)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

from __future__ import annotations
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
    

    import h5py
import torch
from torch.utils.data import Dataset


class HDF5RegressionDataset(Dataset):
    def __init__(self, h5_path, indices, y_mean=None, y_std=None):
        self.h5_path = str(h5_path)
        self.indices = indices.astype("int64")
        self.y_mean = y_mean
        self.y_std = y_std
        self._file = None

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        f = self._get_file()

        real_idx = self.indices[idx]

        X = f["X"][real_idx]
        y = f["y"][real_idx]

        if self.y_mean is not None and self.y_std is not None:
            y = (y - self.y_mean) / self.y_std

        X = torch.from_numpy(X.astype("float32"))
        y = torch.from_numpy(y.astype("float32"))

        return X, y
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
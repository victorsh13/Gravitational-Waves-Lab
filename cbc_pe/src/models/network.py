import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset



import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class ArrayRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TinyCNN(nn.Module):
    def __init__(self, n_detectors=3, n_outputs=3):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=n_detectors,
            out_channels=16,
            kernel_size=16,
            stride=2,
            padding=8,
        )

        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(16, n_outputs)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.head(x)
        return x

class ConvBlock(nn.Module):
    
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=16, 
        stride=2, 
    ):
        
        super(ConvBlock, self).__init__() # Initialize parent class nn.Module, otherwise PyTorch will have issues registering the parameters.

        padding = kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()


    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


class SimpleCNN(nn.Module):
    
    def __init__(
            self, 
            in_channels=3, 
            out_channels=3,
            embedding_dim=64,
            ):
        
        super().__init__()

        # 1D-Conv Block
        self.block1 = ConvBlock( 
            in_channels=in_channels,
            out_channels=16,
            kernel_size=16,
            stride=2,
        )
        
        # 1D-Conv Block
        self.block2 = ConvBlock(
            in_channels=16,
            out_channels=32,
            kernel_size=16,
            stride=2,
        )

        # Global Poolingg
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.embedding_layer = nn.Sequential(
            nn.Linear(32, embedding_dim),
            nn.ReLU(),
        )

        # Linear layer
        self.head = nn.Linear(embedding_dim, out_channels)


    # ENCODER
    def encode(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        return x

    # EMBEDDING
    def embed(self, x):
        features = self.encode(x)
        embedding = self.embedding_layer(features)
        return embedding

    # HEAD
    def forward(self, x, return_embedding=False):
        embedding = self.embed(x)
        y_pred = self.head(embedding)

        if return_embedding:
            return y_pred, embedding

        return y_pred

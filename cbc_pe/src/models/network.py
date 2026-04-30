import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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

class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=42,
        kernel_size=16,
        pool_size=2,
        dropout=0.1,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout1d(p=dropout)
        self.pool = nn.AvgPool1d(kernel_size=pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x
    


class SimpleCNN(nn.Module):
    
    def __init__(
            self, 
            n_detectors=3,
            n_outputs=3,
            embedding_dim=64,
            ):
        
        super().__init__()

        # 1D-Conv Block
        self.block1 = ConvBlock( 
            in_channels=n_detectors,
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

        self.block3 = ConvBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=16,
            stride=2,
        )

        self.block4 = ConvBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=16,
            stride=2,
        )

        # Global Poolingg
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.embedding_layer = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
        )

        # Linear layer
        self.head = nn.Linear(embedding_dim, n_outputs)


    # ENCODER
    def encode(self, x):    # (channels, time) = (3, 16384)
        x = self.block1(x)  # (16, 8192)
        x = self.block2(x)  # (32, 4096)
        x = self.block3(x)  # (64, 2048)
        x = self.block4(x)  # (128, 1024)

        x = self.pool(x)    # (128, 1)
        x = x.squeeze(-1)   # (128,)

        return x

    # EMBEDDING
    def embed(self, x):
        features = self.encode(x)  # (128,)
        embedding = self.embedding_layer(features)  # (128, 64)
        return embedding

    # HEAD
    def forward(self, x, return_embedding=False):
        embedding = self.embed(x)
        y_pred = self.head(embedding)

        if return_embedding:
            return y_pred, embedding

        return y_pred

class GaiaCNN_NoResidual(nn.Module):
    def __init__(
        self,
        n_detectors=3,
        n_outputs=3,
        embedding_dim=64,
        n_filters=42,
    ):
        super().__init__()

        self.block1 = BasicBlock(n_detectors, n_filters, kernel_size=16)
        self.block2 = BasicBlock(n_filters, n_filters, kernel_size=32)
        self.block3 = BasicBlock(n_filters, n_filters, kernel_size=64)
        self.block4 = BasicBlock(n_filters, n_filters, kernel_size=128)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.embedding_layer = nn.Sequential(
            nn.Linear(n_filters, embedding_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.2),
            nn.Linear(embedding_dim, 32),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.head = nn.Linear(32, n_outputs)

    def encode(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.pool(x)
        x = x.squeeze(-1)

        return x

    def embed(self, x):
        features = self.encode(x)
        embedding = self.embedding_layer(features)
        return embedding

    def forward(self, x, return_embedding=False):
        embedding = self.embed(x)
        pred = self.head(embedding)

        if return_embedding:
            return pred, embedding

        return pred
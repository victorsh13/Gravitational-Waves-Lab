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

        # Global Poolingg
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.embedding_layer = nn.Sequential(
            nn.Linear(32, embedding_dim),
            nn.ReLU(),
        )

        # Linear layer
        self.head = nn.Linear(embedding_dim, n_outputs)


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

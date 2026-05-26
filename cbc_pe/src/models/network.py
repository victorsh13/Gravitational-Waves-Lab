from __future__ import annotations
import torch.nn as nn


class ConvBlock(nn.Module):
    
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=16, 
        stride=2, 
        dropout=0.05,
        num_groups=8,
    ):
        
        super().__init__() # Initialize parent class nn.Module, otherwise PyTorch will have issues registering the parameters.

        padding = kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        #self.batch_norm = nn.BatchNorm1d(out_channels)

        self.group_norm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=out_channels,
        )

        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

    
 
class SimpleCNN_Baseline(nn.Module):
    
    def __init__(
            self, 
            n_detectors=3,
            n_outputs=3,
            embedding_dim=32,
            dropout_conv=0.05,
            dropout_dense=0.1,
            ):
        
        super().__init__()

        # 1D-Conv Block
        self.block1 = ConvBlock( 
            in_channels=n_detectors,
            out_channels=16,
            kernel_size=16,
            stride=2,
            dropout=dropout_conv,
        )
        
        # 1D-Conv Block
        self.block2 = ConvBlock(
            in_channels=16,
            out_channels=32,
            kernel_size=16,
            stride=2,
            dropout=dropout_conv,
        )

        self.block3 = ConvBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=16,
            stride=2,
            dropout=dropout_conv,
        )

        self.block4 = ConvBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=16,
            stride=2,
            dropout=dropout_conv,
        )

        # Global Poolingg
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.embedding_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout_dense),
            nn.Linear(64, embedding_dim),
            nn.LeakyReLU(negative_slope=0.1),
        )

        # Linear layer
        self.head = nn.Linear(embedding_dim, n_outputs)


    # ENCODER
    def encode(self, x):    # (batch, channels, time) = (batch, 3, 16384)
        x = self.block1(x)  # (b, 16, ~8192)
        x = self.block2(x)  # (b, 32, ~4096)
        x = self.block3(x)  # (b, 64, ~2048)
        x = self.block4(x)  # (b, 128, ~1024)

        x = self.pool(x)    # (b, 128, 1)
        x = x.squeeze(-1)   # (b, 128)

        return x

    # EMBEDDING
    def embed(self, x):
        features = self.encode(x)  # (b, 128)
        embedding = self.embedding_layer(features)  # (b, emb_dim)
        return embedding

    # HEAD
    def forward(self, x, return_embedding=False):
        embedding = self.embed(x)
        y_pred = self.head(embedding) #(b, 3)

        if return_embedding:
            return y_pred, embedding

        return y_pred


class SimpleCNN_Pool(nn.Module): # Here we used a deeper embedding layer
    
    def __init__(
            self, 
            n_detectors=3,
            n_outputs=3,
            embedding_dim=32,
            dropout_conv=0.05,
            dropout_dense=0.1,
            pool_size=4,
            ):
        
        super().__init__()

        # 1D-Conv Block
        self.block1 = ConvBlock( 
            in_channels=n_detectors,
            out_channels=16,
            kernel_size=16,
            stride=2,
            dropout=dropout_conv,
        )
        
        # 1D-Conv Block
        self.block2 = ConvBlock(
            in_channels=16,
            out_channels=32,
            kernel_size=16,
            stride=2,
            dropout=dropout_conv,
        )

        self.block3 = ConvBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=16,
            stride=2,
            dropout=dropout_conv,
        )

        self.block4 = ConvBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=16,
            stride=2,
            dropout=dropout_conv,
        )

        # Global Poolingg
        self.pool = nn.AdaptiveAvgPool1d(pool_size)

        self.embedding_layer = nn.Sequential(
            nn.Linear(128 * pool_size, 128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout_dense),

            nn.Linear(128, embedding_dim),
        )

        # Linear layer
        self.head = nn.Linear(embedding_dim, n_outputs)


    # ENCODER
    def encode(self, x):    # (batch, channels, time) = (batch, 3, 16384)
        x = self.block1(x)  # (b, 16, ~8192)
        x = self.block2(x)  # (b, 32, ~4096)
        x = self.block3(x)  # (b, 64, ~2048)
        x = self.block4(x)  # (b, 128, ~1024)

        x = self.pool(x)    # (b, 128, 8)
        x = x.flatten(start_dim=1)  # (b, 128*8) = (b, 1024)

        return x

    # EMBEDDING
    def embed(self, x):
        features = self.encode(x)  # (b, 128)
        embedding = self.embedding_layer(features)  # (b, emb_dim)
        return embedding

    # HEAD
    def forward(self, x, return_embedding=False):
        embedding = self.embed(x)
        y_pred = self.head(embedding) #(b, 3)

        if return_embedding:
            return y_pred, embedding

        return y_pred
    

class SimpleCNN_PoolDeep(nn.Module): # Here we used a deeper embedding layer
    
    def __init__(
            self, 
            n_detectors=3,
            n_outputs=3,
            embedding_dim=32,
            dropout_conv=0.05,
            dropout_dense=0.1,
            pool_size=4,
            ):
        
        super().__init__()

        # 1D-Conv Block
        self.block1 = ConvBlock( 
            in_channels=n_detectors,
            out_channels=16,
            kernel_size=16,
            stride=2,
            dropout=dropout_conv,
        )
        
        # 1D-Conv Block
        self.block2 = ConvBlock(
            in_channels=16,
            out_channels=32,
            kernel_size=16,
            stride=2,
            dropout=dropout_conv,
        )

        self.block3 = ConvBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=16,
            stride=2,
            dropout=dropout_conv,
        )

        self.block4 = ConvBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=16,
            stride=2,
            dropout=dropout_conv,
        )

        # Adaptive average pooling
        self.pool = nn.AdaptiveAvgPool1d(pool_size)

        self.embedding_layer = nn.Sequential(
            nn.Linear(128 * pool_size, 512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout_dense),

            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout_dense),

            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout_dense),

            nn.Linear(128, embedding_dim),
        )

        # Linear layer
        self.head = nn.Linear(embedding_dim, n_outputs)


    # ENCODER
    def encode(self, x):    # (batch, channels, time) = (batch, 3, 16384)
        x = self.block1(x)  # (b, 16, ~8192)
        x = self.block2(x)  # (b, 32, ~4096)
        x = self.block3(x)  # (b, 64, ~2048)
        x = self.block4(x)  # (b, 128, ~1024)

        x = self.pool(x)    # (b, 128, 4)
        x = x.flatten(start_dim=1)  # (b, 128*4) = (b, 512)

        return x

    # EMBEDDING
    def embed(self, x):
        features = self.encode(x)  # (b, 128 * pool_size)
        embedding = self.embedding_layer(features)  # (b, emb_dim)
        return embedding

    # HEAD
    def forward(self, x, return_embedding=False):
        embedding = self.embed(x)
        y_pred = self.head(embedding) #(b, 3)

        if return_embedding:
            return y_pred, embedding

        return y_pred
    

class WideCNN_Pool(nn.Module):
    """
    Wider 1D CNN for CBC parameter regression.

    M06 architecture:
        encoder channels: 32 -> 64 -> 128 -> 256
        adaptive pooling: pool_size
        dense head: 256 * pool_size -> 256 -> 128 -> n_outputs

    Intended experiment:
        Test whether the bottleneck is the convolutional feature extractor.
    """

    def __init__(
        self,
        n_detectors=3,
        n_outputs=3,
        embedding_dim=128,
        dropout_conv=0.05,
        dropout_dense=0.1,
        pool_size=4,
    ):
        super().__init__()

        self.pool_size = pool_size

        self.block1 = ConvBlock(
            in_channels=n_detectors,
            out_channels=32,
            kernel_size=16,
            stride=2,
            dropout=dropout_conv,
        )

        self.block2 = ConvBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=16,
            stride=2,
            dropout=dropout_conv,
        )

        self.block3 = ConvBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=16,
            stride=2,
            dropout=dropout_conv,
        )

        self.block4 = ConvBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=16,
            stride=2,
            dropout=dropout_conv,
        )

        self.pool = nn.AdaptiveAvgPool1d(pool_size)

        self.embedding_layer = nn.Sequential(
            nn.Linear(256 * pool_size, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout_dense),

            nn.Linear(256, embedding_dim),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.head = nn.Linear(embedding_dim, n_outputs)

    def encode(self, x):
        # x: (batch, n_detectors, time)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.pool(x)  # (batch, 256, pool_size)
        x = x.flatten(start_dim=1)  # (batch, 256 * pool_size)

        return x

    def embed(self, x):
        features = self.encode(x)
        embedding = self.embedding_layer(features)
        return embedding

    def forward(self, x, return_embedding=False):
        embedding = self.embed(x)
        y_pred = self.head(embedding)

        if return_embedding:
            return y_pred, embedding

        return y_pred


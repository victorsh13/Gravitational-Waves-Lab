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
        self.dropout = nn.Dropout1d(p=dropout)


    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

    
 
class SimpleCNN(nn.Module):
    
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

class SimpleCNN_v2(nn.Module):
    
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
            nn.Linear(128, 128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout_dense),
            nn.Linear(128, embedding_dim),
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


class SimpleCNN_Pool(nn.Module):
    
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
        self.pool = nn.AdaptiveAvgPool1d(8)

        self.embedding_layer = nn.Sequential(
            nn.Linear(128 * 8, 512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout_dense),

            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout_dense),
            
            nn.Linear(256, embedding_dim),
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
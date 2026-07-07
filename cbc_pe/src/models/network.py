from __future__ import annotations
import torch
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

    
class ResidualDilatedBlock(nn.Module):
    """
    Residual 1D convolutional block with dilated convolutions.

    The input and output shapes are identical:

        (batch, channels, time) -> (batch, channels, time)

    The residual path preserves the original representation, while the
    dilated convolutional path adds wider temporal context.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.05,
        num_groups: int = 8,
    ):
        super().__init__()

        if channels <= 0:
            raise ValueError(
                f"channels must be positive, got {channels}"
            )

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be a positive odd integer."
            )

        if dilation <= 0:
            raise ValueError(
                f"dilation must be positive, got {dilation}"
            )

        if channels % num_groups != 0:
            raise ValueError(
                f"channels={channels} must be divisible by "
                f"num_groups={num_groups}."
            )

        # For an odd kernel and stride=1, this preserves temporal length.
        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )

        self.norm1 = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=channels,
        )

        self.activation1 = nn.LeakyReLU(
            negative_slope=0.01
        )

        self.dropout = nn.Dropout(p=dropout)

        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )

        self.norm2 = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=channels,
        )

        self.output_activation = nn.LeakyReLU(
            negative_slope=0.01
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation1(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = residual + out
        out = self.output_activation(out)

        return out    
 

class TemporalAttentionPool1d(nn.Module):
    """
    Content-based temporal attention pooling.

    Input
    -----
    x : torch.Tensor
        Shape (batch, channels, time).

    Output
    ------
    pooled : torch.Tensor
        Shape (batch, channels, 1).

    Notes
    -----
    The output shape deliberately matches AdaptiveAvgPool1d(1), so it can
    replace the original M08 pooling layer without changing its forward pass.
    """

    def __init__(
        self,
        channels: int,
        attention_hidden: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive")

        if attention_hidden <= 0:
            raise ValueError("attention_hidden must be positive")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.channels = channels
        self.attention_hidden = attention_hidden

        self.score_network = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=attention_hidden,
                kernel_size=1,
                bias=True,
            ),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=attention_hidden,
                out_channels=1,
                kernel_size=1,
                bias=False,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        if x.ndim != 3:
            raise ValueError(
                "TemporalAttentionPool1d expects shape "
                f"(batch, channels, time), received {tuple(x.shape)}"
            )

        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, "
                f"received {x.shape[1]}"
            )

        # (B, 1, T)
        attention_logits = self.score_network(x)

        # Normalization is performed along the temporal dimension.
        attention_weights = torch.softmax(
            attention_logits,
            dim=-1,
        )

        # Weighted temporal sum: (B, C, 1)
        pooled = torch.sum(
            x * attention_weights,
            dim=-1,
            keepdim=True,
        )

        if return_attention:
            return pooled, attention_weights

        return pooled


class MultiSlotTemporalAttentionPool1d(nn.Module):
    """
    Content-based temporal attention pooling with multiple learned slots.

    Input
    -----
    x : Tensor
        Shape (batch, channels, time).

    Output
    ------
    pooled : Tensor
        Shape (batch, channels, n_slots).

    attention_weights : Tensor, optional
        Shape (batch, n_slots, time).
    """

    def __init__(
        self,
        channels: int,
        n_slots: int = 4,
        attention_hidden: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive")

        if n_slots <= 0:
            raise ValueError("n_slots must be positive")

        if attention_hidden <= 0:
            raise ValueError("attention_hidden must be positive")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.channels = channels
        self.n_slots = n_slots

        self.score_network = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=attention_hidden,
                kernel_size=1,
            ),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=attention_hidden,
                out_channels=n_slots,
                kernel_size=1,
                bias=False,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        if x.ndim != 3:
            raise ValueError(
                f"Expected (batch, channels, time), got {tuple(x.shape)}"
            )

        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {x.shape[1]}"
            )

        # Shape: (B, K, T)
        logits = self.score_network(x)

        # Each slot defines a probability distribution over time.
        weights = torch.softmax(logits, dim=-1)

        # x:       (B, C, T)
        # weights: (B, K, T)
        # pooled:  (B, C, K)
        pooled = torch.einsum(
            "bct,bkt->bck",
            x,
            weights,
        )

        if return_attention:
            return pooled, weights

        return pooled




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
    

class SimpleCNN_MultiHead(nn.Module):
    def __init__(
        self,
        n_detectors: int = 3,
        n_outputs: int = 3,
        embedding_dim: int = 64,
        head_hidden_dim: int = 32,
        dropout_conv: float = 0.05,
        dropout_dense: float = 0.1,
        dropout_head: float = 0.0,
    ):
        super().__init__()

        if n_outputs != 3:
            raise ValueError(
                "SimpleCNN_MultiHead currently expects exactly 3 outputs: "
                "chirp_mass, total_mass and chi_eff."
            )

        # Shared convolutional encoder: identical to the baseline.
        self.block1 = ConvBlock(
            in_channels=n_detectors,
            out_channels=16,
            kernel_size=16,
            stride=2,
            dropout=dropout_conv,
        )

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

        self.pool = nn.AdaptiveAvgPool1d(1)

        # Shared embedding: identical to the bs256 baseline.
        self.embedding_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout_dense),
            nn.Linear(64, embedding_dim),
            nn.LeakyReLU(negative_slope=0.1),
        )

        # Independent nonlinear task-specific heads.
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, head_hidden_dim),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Dropout(p=dropout_head),
                nn.Linear(head_hidden_dim, 1),
            )
            for _ in range(n_outputs)
        ])

    def encode(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape: (batch, n_detectors, time)

        Returns
        -------
        torch.Tensor
            Shared encoder features with shape (batch, 128).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.pool(x)
        x = x.squeeze(-1)

        return x

    def embed(self, x):
        """
        Returns the shared embedding used by all task-specific heads.
        """
        features = self.encode(x)
        embedding = self.embedding_layer(features)
        return embedding

    def forward(self, x, return_embedding: bool = False):
        embedding = self.embed(x)

        outputs = [
            head(embedding)
            for head in self.heads
        ]

        y_pred = torch.cat(outputs, dim=1)

        if return_embedding:
            return y_pred, embedding

        return y_pred   # Output order must match y:
                        # [chirp_mass, total_mass, chi_eff]


class SimpleCNN_ResidualDilated(nn.Module):
    """
    Residual-dilated CNN for CBC parameter regression.

    Architecture:
        - Three strided ConvBlocks for initial feature extraction and
          temporal downsampling.
        - A sequence of residual dilated blocks operating at 64 channels.
        - A 1x1 projection from 64 to 128 channels.
        - Global average pooling.
        - Shared embedding.
        - Linear regression head.

    Intended experiment:
        Test whether increasing temporal context before global pooling
        improves parameter estimation, particularly chi_eff.
    """

    def __init__(
        self,
        n_detectors: int = 3,
        n_outputs: int = 3,
        embedding_dim: int = 64,
        residual_channels: int = 64,
        dilations=(1, 2, 4),
        residual_kernel_size: int = 7,
        dropout_conv: float = 0.05,
        dropout_dense: float = 0.1,
        num_groups: int = 8,
    ):
        super().__init__()

        if residual_channels != 64:
            raise ValueError(
                "The current M08 design expects residual_channels=64 "
                "because block3 outputs 64 channels."
            )

        if len(dilations) == 0:
            raise ValueError("dilations must contain at least one value.")

        if any(int(d) <= 0 for d in dilations):
            raise ValueError(
                f"All dilations must be positive, got {dilations}."
            )

        self.dilations = tuple(int(d) for d in dilations)

        # Initial shared encoder.
        self.block1 = ConvBlock(
            in_channels=n_detectors,
            out_channels=16,
            kernel_size=16,
            stride=2,
            dropout=dropout_conv,
        )

        self.block2 = ConvBlock(
            in_channels=16,
            out_channels=32,
            kernel_size=16,
            stride=2,
            dropout=dropout_conv,
        )

        self.block3 = ConvBlock(
            in_channels=32,
            out_channels=residual_channels,
            kernel_size=16,
            stride=2,
            dropout=dropout_conv,
        )

        # Wider-context temporal processing.
        self.residual_blocks = nn.Sequential(
            *[
                ResidualDilatedBlock(
                    channels=residual_channels,
                    kernel_size=residual_kernel_size,
                    dilation=dilation,
                    dropout=dropout_conv,
                    num_groups=num_groups,
                )
                for dilation in self.dilations
            ]
        )

        # Channel projection without modifying temporal resolution.
        self.projection = nn.Sequential(
            nn.Conv1d(
                in_channels=residual_channels,
                out_channels=128,
                kernel_size=1,
                stride=1,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=128,
            ),
            nn.LeakyReLU(negative_slope=0.01),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        # Same embedding structure as M00.
        self.embedding_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout_dense),
            nn.Linear(64, embedding_dim),
            nn.LeakyReLU(negative_slope=0.1),
        )

        # Same shared head as M00.
        self.head = nn.Linear(
            embedding_dim,
            n_outputs,
        )

    def encode(self, x):
        # Input: (batch, 3, 16384)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Shape remains (batch, 64, time) through all residual blocks.
        x = self.residual_blocks(x)

        # (batch, 64, time) -> (batch, 128, time)
        x = self.projection(x)

        # (batch, 128, time) -> (batch, 128, 1)
        x = self.pool(x)

        # (batch, 128, 1) -> (batch, 128)
        x = x.squeeze(-1)

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


class SimpleCNN_ResidualDilatedMultiAttention(
    SimpleCNN_ResidualDilated
):
    """
    M09: M08 residual-dilated encoder with multi-slot temporal
    attention pooling.

    Architecture
    ------------
    Encoder output:
        (B, 128, T)

    Multi-slot attention:
        (B, 128, T) -> (B, 128, K)

    Shared slot projection:
        (B, 128, K) -> (B, slot_dim, K)

    Flatten:
        (B, slot_dim * K)

    By default:
        K = 4
        slot_dim = 32
        slot_dim * K = 128

    Therefore, the inherited M08 embedding layer remains unchanged.
    """

    def __init__(
        self,
        n_detectors: int = 3,
        n_outputs: int = 3,
        embedding_dim: int = 64,
        residual_channels: int = 64,
        dilations=(1, 2, 4),
        residual_kernel_size: int = 7,
        dropout_conv: float = 0.05,
        dropout_dense: float = 0.1,
        num_groups: int = 8,
        attention_slots: int = 4,
        attention_hidden: int = 64,
        attention_dropout: float = 0.0,
        slot_dim: int = 32,
    ):
        super().__init__(
            n_detectors=n_detectors,
            n_outputs=n_outputs,
            embedding_dim=embedding_dim,
            residual_channels=residual_channels,
            dilations=dilations,
            residual_kernel_size=residual_kernel_size,
            dropout_conv=dropout_conv,
            dropout_dense=dropout_dense,
            num_groups=num_groups,
        )

        if attention_slots <= 0:
            raise ValueError(
                f"attention_slots must be positive, got {attention_slots}"
            )

        if slot_dim <= 0:
            raise ValueError(
                f"slot_dim must be positive, got {slot_dim}"
            )

        temporal_channels = 128
        flattened_dim = attention_slots * slot_dim

        # Keep the input dimension to the inherited M08 embedding fixed.
        if flattened_dim != temporal_channels:
            raise ValueError(
                "For a controlled comparison with M08, "
                "attention_slots * slot_dim must equal 128. "
                f"Received {attention_slots} * {slot_dim} "
                f"= {flattened_dim}."
            )

        self.attention_slots = attention_slots
        self.slot_dim = slot_dim
        self.temporal_channels = temporal_channels

        self.pool = MultiSlotTemporalAttentionPool1d(
            channels=temporal_channels,
            n_slots=attention_slots,
            attention_hidden=attention_hidden,
            dropout=attention_dropout,
        )

        # This Conv1d is applied to the K-slot representation.
        # It mixes the original 128 feature channels independently
        # at each attention slot.
        self.slot_projection = nn.Sequential(
            nn.Conv1d(
                in_channels=temporal_channels,
                out_channels=slot_dim,
                kernel_size=1,
                stride=1,
                bias=True,
            ),
            nn.GroupNorm(
                num_groups=min(num_groups, slot_dim),
                num_channels=slot_dim,
            ),
            nn.LeakyReLU(negative_slope=0.01),
        )

    def encode(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        # Initial M08 encoder.
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Residual dilated processing: d = [1, 2, 4].
        x = self.residual_blocks(x)

        # (B, 64, T) -> (B, 128, T)
        x = self.projection(x)

        # (B, 128, T) -> (B, 128, K)
        if return_attention:
            x, attention_weights = self.pool(
                x,
                return_attention=True,
            )
        else:
            x = self.pool(x)

        # (B, 128, K) -> (B, slot_dim, K)
        x = self.slot_projection(x)

        # With K=4 and slot_dim=32:
        # (B, 32, 4) -> (B, 128)
        x = torch.flatten(x, start_dim=1)

        if return_attention:
            return x, attention_weights

        return x

    def get_attention_weights(self, x: torch.Tensor):
        """
        Return the K temporal attention maps without running the
        embedding and regression head.

        Returns
        -------
        features : torch.Tensor
            Shape (B, 128).

        attention_weights : torch.Tensor
            Shape (B, K, T_encoder).
        """
        return self.encode(
            x,
            return_attention=True,
        )



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


"""Dimension-aware transformer attention mechanisms.

Provides multi-head attention and transformer encoder components that track
physical dimensions through Query-Key-Value operations. Attention scores are
made dimensionless via characteristic scaling before softmax.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.dimensions import Dimension, DIMENSIONLESS
from ..core.units import Unit
from ..errors import DimensionError
from .dimtensor import DimTensor
from .layers import DimLayer, DimLinear
from .normalization import DimLayerNorm


class DimMultiheadAttention(DimLayer):
    """Multi-head attention with physical dimension tracking.

    This layer implements scaled dot-product attention with unit safety.
    Query, Key, and Value projections all accept inputs with the same physical
    dimension. Attention scores are made dimensionless by dividing by a
    characteristic scale computed from the keys, enabling proper use of softmax.

    The characteristic scale is computed as: scale² = mean(K²) + epsilon
    This ensures attention scores (Q @ K^T) / scale² are dimensionless.

    Args:
        embed_dim: Embedding dimension (must be divisible by num_heads).
        num_heads: Number of attention heads.
        input_dim: Physical dimension of input (Q, K share this dimension).
        output_dim: Physical dimension of output (V projection target).
                   If not specified, defaults to input_dim.
        dropout: Dropout probability for attention weights.
        bias: If True, adds bias to input projections.
        validate_input: If True, raises DimensionError on dimension mismatch.

    Examples:
        >>> from dimtensor.torch import DimTensor, DimMultiheadAttention
        >>> from dimtensor import units, Dimension
        >>>
        >>> # Attention over particle positions
        >>> attn = DimMultiheadAttention(
        ...     embed_dim=64,
        ...     num_heads=4,
        ...     input_dim=Dimension(L=1),  # meters
        ...     output_dim=Dimension(L=1)  # meters
        ... )
        >>>
        >>> # Input: (batch, seq_len, embed_dim)
        >>> x = DimTensor(torch.randn(32, 100, 64), units.m)
        >>> output = attn(x)  # Shape: (32, 100, 64), dimension: meters
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        input_dim: Dimension = DIMENSIONLESS,
        output_dim: Dimension | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        validate_input: bool = True,
    ) -> None:
        # Default output_dim to input_dim if not specified
        if output_dim is None:
            output_dim = input_dim

        super().__init__(input_dim, output_dim, validate_input)

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout

        # Q and K projections: input_dim -> input_dim (for self-attention)
        self.q_proj = DimLinear(
            embed_dim, embed_dim, input_dim, input_dim, bias=bias, validate_input=False
        )
        self.k_proj = DimLinear(
            embed_dim, embed_dim, input_dim, input_dim, bias=bias, validate_input=False
        )

        # V projection: input_dim -> output_dim (allows dimension transformation)
        self.v_proj = DimLinear(
            embed_dim, embed_dim, input_dim, output_dim, bias=bias, validate_input=False
        )

        # Output projection: output_dim -> output_dim
        self.out_proj = DimLinear(
            embed_dim, embed_dim, output_dim, output_dim, bias=bias, validate_input=False
        )

        self.dropout = nn.Dropout(dropout)

        # Epsilon for numerical stability in scale computation
        self.scale_epsilon = 1e-8

        # Store input unit for internal conversions
        self._input_unit = Unit(str(input_dim), input_dim, 1.0)

    def _forward_impl(self, x: Tensor) -> Tensor:
        """Implement multi-head attention computation.

        Args:
            x: Input tensor (batch, seq_len, embed_dim).

        Returns:
            Output tensor (batch, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = x.shape

        # Convert raw tensor to DimTensor for projection layers
        x_dim = DimTensor._from_tensor_and_unit(x, self._input_unit)

        # Compute Q, K, V projections
        Q = self.q_proj(x_dim)  # (batch, seq, embed_dim) with input_dim
        K = self.k_proj(x_dim)  # (batch, seq, embed_dim) with input_dim
        V = self.v_proj(x_dim)  # (batch, seq, embed_dim) with output_dim

        # Reshape for multi-head attention: (batch, seq, heads, head_dim)
        Q_data = Q.data.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K_data = K.data.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V_data = V.data.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch, heads, seq, head_dim)
        Q_data = Q_data.transpose(1, 2)
        K_data = K_data.transpose(1, 2)
        V_data = V_data.transpose(1, 2)

        # Compute characteristic scale from K: scale² = mean(K²) + epsilon
        # K has units [input_dim], so K² has units [input_dim²]
        K_squared = K_data * K_data  # Element-wise, units: [input_dim²]
        K_squared_mean = K_squared.mean(dim=-1, keepdim=True).mean(
            dim=-2, keepdim=True
        )
        scale_squared = K_squared_mean + self.scale_epsilon  # Units: [input_dim²]

        # Attention scores: Q @ K^T
        # Q: (batch, heads, seq, head_dim) with units [input_dim]
        # K^T: (batch, heads, head_dim, seq) with units [input_dim]
        # Result: (batch, heads, seq, seq) with units [input_dim²]
        scores = torch.matmul(Q_data, K_data.transpose(-2, -1))

        # Make scores dimensionless: scores / scale²
        # scores has units [input_dim²], scale_squared has units [input_dim²]
        # Result is dimensionless!
        scores_dimensionless = scores / scale_squared.sqrt()

        # Apply softmax over keys (last dimension)
        # Softmax requires dimensionless input
        attn_weights = F.softmax(scores_dimensionless, dim=-1)  # Dimensionless
        attn_weights = self.dropout(attn_weights)

        # Weighted sum: attn_weights @ V
        # attn_weights: (batch, heads, seq, seq) dimensionless
        # V: (batch, heads, seq, head_dim) with units [output_dim]
        # Result: (batch, heads, seq, head_dim) with units [output_dim]
        attn_output = torch.matmul(attn_weights, V_data)

        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Merge heads: (batch, seq, heads * head_dim) = (batch, seq, embed_dim)
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        # Apply output projection
        attn_output_dim = DimTensor._from_tensor_and_unit(attn_output, self._output_unit)
        output = self.out_proj(attn_output_dim)

        return output.data

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"dropout={self.dropout_p}, "
            f"input_dim={self.input_dim}, output_dim={self.output_dim}"
        )


class DimTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with dimension tracking.

    A standard transformer encoder layer consisting of:
    - Multi-head self-attention with residual connection
    - Feedforward network (2-layer MLP) with residual connection
    - Layer normalization after each sub-layer

    Note: Residual connections require input_dim = output_dim for both
    attention and feedforward blocks. This is enforced in the constructor.

    Args:
        d_model: Model dimension (embed_dim).
        nhead: Number of attention heads.
        dim_feedforward: Dimension of feedforward network (default: 2048).
        dimension: Physical dimension (same for input and output due to residuals).
        dropout: Dropout probability (default: 0.1).
        activation: Activation function in feedforward network (default: "relu").

    Examples:
        >>> from dimtensor.torch import DimTensor, DimTransformerEncoderLayer
        >>> from dimtensor import units, Dimension
        >>>
        >>> # Encoder layer for particle trajectories
        >>> layer = DimTransformerEncoderLayer(
        ...     d_model=128,
        ...     nhead=8,
        ...     dim_feedforward=512,
        ...     dimension=Dimension(L=1),  # meters
        ... )
        >>>
        >>> x = DimTensor(torch.randn(32, 100, 128), units.m)
        >>> output = layer(x)  # Shape: (32, 100, 128), dimension: meters
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dimension: Dimension = DIMENSIONLESS,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.dimension = dimension

        # Self-attention (input_dim = output_dim for residual connection)
        self.self_attn = DimMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            input_dim=dimension,
            output_dim=dimension,
            dropout=dropout,
        )

        # Feedforward network (preserves dimension for residual)
        self.linear1 = DimLinear(
            d_model, dim_feedforward, dimension, dimension, validate_input=False
        )
        self.linear2 = DimLinear(
            dim_feedforward, d_model, dimension, dimension, validate_input=False
        )

        # Layer normalization
        self.norm1 = DimLayerNorm(d_model, dimension)
        self.norm2 = DimLayerNorm(d_model, dimension)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: DimTensor | Tensor) -> DimTensor:
        """Forward pass with residual connections.

        Args:
            x: Input (batch, seq, d_model) with self.dimension.

        Returns:
            Output (batch, seq, d_model) with self.dimension.

        Raises:
            DimensionError: If input dimension doesn't match expected dimension.
        """
        # Convert to DimTensor if needed
        if isinstance(x, Tensor):
            unit = Unit(str(self.dimension), self.dimension, 1.0)
            x = DimTensor._from_tensor_and_unit(x, unit)

        # Validate dimension
        if x.dimension != self.dimension:
            raise DimensionError(
                f"Expected input dimension {self.dimension}, got {x.dimension}"
            )

        # Self-attention block with residual
        attn_output = self.self_attn(x)
        # Dropout on raw tensor, then convert back to DimTensor for addition
        attn_output_dropped = DimTensor._from_tensor_and_unit(
            self.dropout1(attn_output.data), attn_output.unit
        )
        x = x + attn_output_dropped  # Residual connection (same dimensions!)
        x = self.norm1(x)

        # Feedforward block with residual
        ff_output = self.linear1(x)
        # Apply activation on raw tensor, then wrap back
        ff_output = DimTensor._from_tensor_and_unit(
            self.activation(ff_output.data), ff_output.unit
        )
        ff_output = self.linear2(ff_output)
        ff_output_dropped = DimTensor._from_tensor_and_unit(
            self.dropout2(ff_output.data), ff_output.unit
        )
        x = x + ff_output_dropped  # Residual connection (same dimensions!)
        x = self.norm2(x)

        return x


class DimTransformerEncoder(nn.Module):
    """Stack of transformer encoder layers with dimension tracking.

    Args:
        encoder_layer: A single DimTransformerEncoderLayer instance to be replicated.
        num_layers: Number of encoder layers in the stack.
        norm: Optional final normalization layer.

    Examples:
        >>> from dimtensor.torch import DimTensor, DimTransformerEncoderLayer, DimTransformerEncoder
        >>> from dimtensor import units, Dimension
        >>>
        >>> # Create encoder for particle system
        >>> layer = DimTransformerEncoderLayer(
        ...     d_model=128,
        ...     nhead=8,
        ...     dimension=Dimension(L=1),
        ... )
        >>> encoder = DimTransformerEncoder(layer, num_layers=6)
        >>>
        >>> x = DimTensor(torch.randn(32, 100, 128), units.m)
        >>> output = encoder(x)  # Shape: (32, 100, 128), dimension: meters
    """

    def __init__(
        self,
        encoder_layer: DimTransformerEncoderLayer,
        num_layers: int,
        norm: DimLayerNorm | None = None,
    ) -> None:
        super().__init__()

        # Create copies of the encoder layer
        self.layers = nn.ModuleList(
            [
                DimTransformerEncoderLayer(
                    d_model=encoder_layer.self_attn.embed_dim,
                    nhead=encoder_layer.self_attn.num_heads,
                    dim_feedforward=encoder_layer.linear1.out_features,
                    dimension=encoder_layer.dimension,
                    dropout=encoder_layer.self_attn.dropout_p,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = norm
        self.num_layers = num_layers
        self.dimension = encoder_layer.dimension

    def forward(self, x: DimTensor | Tensor) -> DimTensor:
        """Forward pass through all encoder layers.

        Args:
            x: Input (batch, seq, d_model) with encoder dimension.

        Returns:
            Output (batch, seq, d_model) with encoder dimension.
        """
        # Convert to DimTensor if needed
        if isinstance(x, Tensor):
            unit = Unit(str(self.dimension), self.dimension, 1.0)
            x = DimTensor._from_tensor_and_unit(x, unit)

        # Pass through each encoder layer
        for layer in self.layers:
            x = layer(x)

        # Optional final normalization
        if self.norm is not None:
            x = self.norm(x)

        return x

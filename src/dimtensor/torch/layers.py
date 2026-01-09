"""Dimension-aware neural network layers.

Provides PyTorch layers that track physical dimensions through forward passes,
enabling dimensional consistency checking in physics-informed neural networks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ..core.dimensions import Dimension, DIMENSIONLESS
from ..core.units import Unit, dimensionless
from ..errors import DimensionError
from .dimtensor import DimTensor


class DimLayer(nn.Module, ABC):
    """Base class for dimension-aware neural network layers.

    DimLayer tracks physical dimensions through forward passes, ensuring
    that outputs have the expected physical dimension.

    Subclasses must implement:
    - _forward_impl(x: Tensor) -> Tensor

    Args:
        input_dim: Expected dimension of input tensors.
        output_dim: Dimension of output tensors.
        validate_input: If True, raise DimensionError on dimension mismatch.
                       If False, only track output dimension.
    """

    def __init__(
        self,
        input_dim: Dimension = DIMENSIONLESS,
        output_dim: Dimension = DIMENSIONLESS,
        validate_input: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.validate_input = validate_input
        # Create output unit from dimension (symbol derived from dimension)
        self._output_unit = Unit(str(output_dim), output_dim, 1.0)

    @abstractmethod
    def _forward_impl(self, x: Tensor) -> Tensor:
        """Implement the actual forward computation.

        Args:
            x: Raw tensor (dimension already validated).

        Returns:
            Result tensor.
        """
        pass

    def forward(self, x: DimTensor | Tensor) -> DimTensor:
        """Forward pass with dimension tracking.

        Args:
            x: Input tensor. If DimTensor, dimension is validated.
               If raw Tensor, no validation is performed.

        Returns:
            DimTensor with the layer's output dimension.

        Raises:
            DimensionError: If input dimension doesn't match expected
                           (and validate_input is True).
        """
        # Extract raw tensor and validate dimension
        if isinstance(x, DimTensor):
            if self.validate_input and x.dimension != self.input_dim:
                raise DimensionError(
                    f"Layer expects input dimension {self.input_dim}, "
                    f"got {x.dimension}"
                )
            tensor = x.data
        else:
            tensor = x

        # Compute
        result = self._forward_impl(tensor)

        # Return with output dimension
        return DimTensor._from_tensor_and_unit(result, self._output_unit)

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"


class DimLinear(DimLayer):
    """Linear transformation with physical dimension tracking.

    This layer performs y = xW^T + b where:
    - x has physical dimension input_dim
    - W has dimension (output_dim / input_dim) (implicit)
    - y has physical dimension output_dim

    The weight matrix implicitly carries the transformation between
    physical dimensions, similar to how a matrix converting position
    to velocity would have units of [1/T].

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        input_dim: Physical dimension of input.
        output_dim: Physical dimension of output.
        bias: If True, adds a learnable bias.
        device: Device to place parameters on.
        dtype: Data type for parameters.

    Examples:
        >>> from dimtensor.torch import DimTensor, DimLinear
        >>> from dimtensor import units, Dimension
        >>>
        >>> # Layer that converts position [m] to velocity [m/s]
        >>> layer = DimLinear(
        ...     in_features=3,
        ...     out_features=3,
        ...     input_dim=Dimension(L=1),      # meters
        ...     output_dim=Dimension(L=1, T=-1) # m/s
        ... )
        >>>
        >>> x = DimTensor(torch.randn(10, 3), units.m)
        >>> v = layer(x)  # Output has dimension m/s
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        input_dim: Dimension = DIMENSIONLESS,
        output_dim: Dimension = DIMENSIONLESS,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        validate_input: bool = True,
    ) -> None:
        super().__init__(input_dim, output_dim, validate_input)

        self.in_features = in_features
        self.out_features = out_features

        # Create underlying linear layer
        self.linear = nn.Linear(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )

    def _forward_impl(self, x: Tensor) -> Tensor:
        """Apply linear transformation."""
        result: Tensor = self.linear(x)
        return result

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"bias={self.linear.bias is not None}"
        )


class DimConv1d(DimLayer):
    """1D convolution with physical dimension tracking.

    Applies a 1D convolution over an input signal with physical dimensions.
    The output has the specified output_dim regardless of input dimension.

    Args:
        in_channels: Number of channels in the input.
        out_channels: Number of channels in the output.
        kernel_size: Size of the convolving kernel.
        input_dim: Physical dimension of input.
        output_dim: Physical dimension of output.
        stride: Stride of the convolution.
        padding: Padding added to both sides.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections.
        bias: If True, adds a learnable bias.
        padding_mode: Padding mode ('zeros', 'reflect', 'replicate', 'circular').

    Examples:
        >>> from dimtensor.torch import DimTensor, DimConv1d
        >>> from dimtensor import units, Dimension
        >>>
        >>> # Convolve a signal with physical units
        >>> conv = DimConv1d(
        ...     in_channels=1, out_channels=4, kernel_size=3,
        ...     input_dim=Dimension(L=1),  # meters
        ...     output_dim=Dimension(L=1)  # meters
        ... )
        >>>
        >>> x = DimTensor(torch.randn(32, 1, 100), units.m)
        >>> y = conv(x)  # Shape: (32, 4, 98), dimension: m
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        input_dim: Dimension = DIMENSIONLESS,
        output_dim: Dimension = DIMENSIONLESS,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        validate_input: bool = True,
    ) -> None:
        super().__init__(input_dim, output_dim, validate_input)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def _forward_impl(self, x: Tensor) -> Tensor:
        """Apply 1D convolution."""
        result: Tensor = self.conv(x)
        return result

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"input_dim={self.input_dim}, output_dim={self.output_dim}"
        )


class DimConv2d(DimLayer):
    """2D convolution with physical dimension tracking.

    Applies a 2D convolution over an input image with physical dimensions.
    Useful for physics simulations on 2D grids or image-based physics.

    Args:
        in_channels: Number of channels in the input.
        out_channels: Number of channels in the output.
        kernel_size: Size of the convolving kernel.
        input_dim: Physical dimension of input.
        output_dim: Physical dimension of output.
        stride: Stride of the convolution.
        padding: Padding added to all sides.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections.
        bias: If True, adds a learnable bias.
        padding_mode: Padding mode.

    Examples:
        >>> from dimtensor.torch import DimTensor, DimConv2d
        >>> from dimtensor import units, Dimension
        >>>
        >>> # Process a temperature field
        >>> conv = DimConv2d(
        ...     in_channels=1, out_channels=8, kernel_size=3,
        ...     input_dim=Dimension(Theta=1),  # temperature
        ...     output_dim=Dimension(Theta=1)  # temperature
        ... )
        >>>
        >>> T = DimTensor(torch.randn(16, 1, 64, 64), units.K)
        >>> T_out = conv(T)  # Shape: (16, 8, 62, 62), dimension: K
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        input_dim: Dimension = DIMENSIONLESS,
        output_dim: Dimension = DIMENSIONLESS,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        validate_input: bool = True,
    ) -> None:
        super().__init__(input_dim, output_dim, validate_input)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def _forward_impl(self, x: Tensor) -> Tensor:
        """Apply 2D convolution."""
        result: Tensor = self.conv(x)
        return result

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"input_dim={self.input_dim}, output_dim={self.output_dim}"
        )


class DimSequential(nn.Module):
    """Sequential container for dimension-aware layers.

    A sequential container that chains DimLayers together, automatically
    validating that dimensions are compatible between layers.

    Args:
        *layers: DimLayer instances to chain together.

    Raises:
        ValueError: If layer output/input dimensions don't match.

    Examples:
        >>> from dimtensor.torch import DimLinear, DimSequential
        >>> from dimtensor import Dimension
        >>>
        >>> # Build a network that converts position -> velocity -> acceleration
        >>> L = Dimension(L=1)
        >>> V = Dimension(L=1, T=-1)
        >>> A = Dimension(L=1, T=-2)
        >>>
        >>> model = DimSequential(
        ...     DimLinear(3, 16, input_dim=L, output_dim=V),
        ...     DimLinear(16, 3, input_dim=V, output_dim=A),
        ... )
    """

    def __init__(self, *layers: DimLayer) -> None:
        super().__init__()

        # Validate dimension chain
        for i in range(len(layers) - 1):
            if layers[i].output_dim != layers[i + 1].input_dim:
                raise ValueError(
                    f"Dimension mismatch: layer {i} outputs {layers[i].output_dim} "
                    f"but layer {i+1} expects {layers[i + 1].input_dim}"
                )

        self.layers = nn.ModuleList(layers)

        # Store chain dimensions
        self.input_dim = layers[0].input_dim if layers else DIMENSIONLESS
        self.output_dim = layers[-1].output_dim if layers else DIMENSIONLESS

    def forward(self, x: DimTensor | Tensor) -> DimTensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x  # type: ignore[return-value]

    def __len__(self) -> int:
        """Number of layers."""
        return len(self.layers)

    def __getitem__(self, idx: int) -> DimLayer:
        """Get layer by index."""
        layer: DimLayer = self.layers[idx]
        return layer


class DimGraphConv(DimLayer):
    """Graph convolution layer with dimension tracking.

    Implements a simple graph convolution that preserves physical units through
    message passing operations. This is useful for physics-based graph neural networks
    where nodes represent physical entities (atoms, particles, etc.) with units.

    The layer computes:
        h_i' = W_self · h_i + W_neigh · AGG_{j∈N(i)}(h_j)

    Where:
    - h_i: node feature vector for node i [input_dim]
    - h_i': updated node feature vector [output_dim]
    - N(i): neighbors of node i (from edge_index)
    - W_self, W_neigh: learnable weight matrices
    - AGG: aggregation function (mean or sum)

    Args:
        in_features: Number of input features per node.
        out_features: Number of output features per node.
        input_dim: Physical dimension of node features.
        output_dim: Physical dimension of output features.
        aggr: Aggregation method ('mean' or 'sum'). Default: 'mean'.
        bias: If True, add learnable bias. Default: True.
        normalize: If True, normalize aggregated messages by node degree. Default: True.
        device: Device to place parameters on.
        dtype: Data type for parameters.
        validate_input: If True, validate input dimensions.

    Examples:
        >>> from dimtensor.torch import DimTensor, DimGraphConv
        >>> from dimtensor import units, Dimension
        >>> import torch
        >>>
        >>> # Molecular dynamics: atoms with positions [m] -> forces [N]
        >>> layer = DimGraphConv(
        ...     in_features=3,
        ...     out_features=3,
        ...     input_dim=Dimension(L=1),           # meters
        ...     output_dim=Dimension(L=1, M=1, T=-2) # Newtons (kg·m/s²)
        ... )
        >>>
        >>> # 4 atoms with 3D positions
        >>> positions = DimTensor(torch.randn(4, 3), units.m)
        >>>
        >>> # Connectivity: atom 0→1, 1→2, 2→3, 3→0 (ring)
        >>> edge_index = torch.tensor([[0, 1, 2, 3],
        ...                            [1, 2, 3, 0]])
        >>>
        >>> forces = layer(positions, edge_index)
        >>> print(forces.dimension)  # L·M·T⁻² (force)
        >>> print(forces.shape)      # (4, 3)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        input_dim: Dimension = DIMENSIONLESS,
        output_dim: Dimension = DIMENSIONLESS,
        aggr: str = "mean",
        bias: bool = True,
        normalize: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        validate_input: bool = True,
    ) -> None:
        super().__init__(input_dim, output_dim, validate_input)

        if aggr not in ("mean", "sum"):
            raise ValueError(f"aggr must be 'mean' or 'sum', got {aggr!r}")

        self.in_features = in_features
        self.out_features = out_features
        self.aggr = aggr
        self.normalize = normalize

        # Linear transformations for self and neighbor features
        self.lin_self = nn.Linear(
            in_features, out_features, bias=False, device=device, dtype=dtype
        )
        self.lin_neigh = nn.Linear(
            in_features, out_features, bias=False, device=device, dtype=dtype
        )

        # Optional bias
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

    def _forward_impl(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Compute graph convolution.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge connectivity [2, num_edges]
                       edge_index[0]: source nodes
                       edge_index[1]: target nodes

        Returns:
            Updated node features [num_nodes, out_features]
        """
        num_nodes = x.size(0)

        # Self transformation
        out_self = self.lin_self(x)

        # Neighbor aggregation
        if edge_index.size(1) > 0:  # Check if there are any edges
            row, col = edge_index[0], edge_index[1]

            # Get neighbor features and transform
            neighbor_feats = x[row]  # Features of source nodes
            neighbor_msg = self.lin_neigh(neighbor_feats)

            # Aggregate messages to target nodes
            out_neigh = torch.zeros_like(out_self)
            out_neigh.index_add_(0, col, neighbor_msg)

            # Normalize by degree if requested
            if self.normalize and self.aggr == "mean":
                # Compute degree (number of incoming edges per node)
                deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
                deg.index_add_(0, col, torch.ones(col.size(0), device=x.device, dtype=x.dtype))
                deg = deg.clamp(min=1.0)  # Avoid division by zero for isolated nodes
                out_neigh = out_neigh / deg.unsqueeze(-1)
        else:
            # No edges - only self transformation
            out_neigh = torch.zeros_like(out_self)

        # Combine self and neighbor contributions
        out = out_self + out_neigh

        # Add bias
        if self.bias is not None:
            out = out + self.bias

        return out

    def forward(
        self,
        x: DimTensor | Tensor,
        edge_index: Tensor,
    ) -> DimTensor:
        """Forward pass with dimension tracking.

        Args:
            x: Node features with units [num_nodes, in_features].
               Can be DimTensor (validated) or raw Tensor (no validation).
            edge_index: Edge connectivity (unitless indices) [2, num_edges].
                       edge_index[0] = source nodes, edge_index[1] = target nodes.

        Returns:
            Updated node features with output_dim [num_nodes, out_features].

        Raises:
            DimensionError: If input dimension doesn't match expected
                           (and validate_input is True).
        """
        # Validate input dimension and extract raw tensor
        if isinstance(x, DimTensor):
            if self.validate_input and x.dimension != self.input_dim:
                raise DimensionError(
                    f"Layer expects input dimension {self.input_dim}, "
                    f"got {x.dimension}"
                )
            tensor = x.data
        else:
            tensor = x

        # Compute graph convolution
        result = self._forward_impl(tensor, edge_index)

        # Return with output dimension
        return DimTensor._from_tensor_and_unit(result, self._output_unit)

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"aggr={self.aggr!r}, normalize={self.normalize}, "
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"bias={self.bias is not None}"
        )

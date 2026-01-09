"""PyTorch integration for dimtensor.

Provides DimTensor, a torch.Tensor wrapper with physical unit tracking.
Supports autograd, GPU acceleration, and neural network compatibility.

Also provides dimension-aware neural network layers, loss functions,
normalization layers, scalers for physics-informed machine learning,
and CUDA profiling utilities.

Example:
    >>> import torch
    >>> from dimtensor.torch import DimTensor, DimLinear
    >>> from dimtensor import units, Dimension
    >>>
    >>> # Unit-aware tensors
    >>> velocity = DimTensor(torch.randn(32, 3), units.m / units.s)
    >>> velocity.requires_grad_(True)
    >>> energy = 0.5 * mass * velocity**2
    >>> energy.sum().backward()
    >>>
    >>> # Dimension-aware layers
    >>> layer = DimLinear(3, 16, input_dim=Dimension(L=1), output_dim=Dimension(L=1, T=-1))
"""

from .dimtensor import DimTensor

# Benchmarking utilities
from . import benchmarks

# Dimension-aware layers
from .layers import (
    DimConv1d,
    DimConv2d,
    DimGraphConv,
    DimLayer,
    DimLinear,
    DimSequential,
)

# Dimension-aware attention and transformers
from .attention import (
    DimMultiheadAttention,
    DimTransformerEncoder,
    DimTransformerEncoderLayer,
)

# Dimension-aware loss functions
from .losses import (
    CompositeLoss,
    DimHuberLoss,
    DimL1Loss,
    DimMSELoss,
    PhysicsLoss,
)

# Dimension-aware normalization
from .normalization import (
    DimBatchNorm1d,
    DimBatchNorm2d,
    DimInstanceNorm1d,
    DimInstanceNorm2d,
    DimLayerNorm,
)

# Scalers for non-dimensionalization
from .scaler import DimScaler, MultiScaler

# Physics priors for physics-informed machine learning
from .priors import (
    ConservationPrior,
    DimensionalConsistencyPrior,
    EnergyConservationPrior,
    MomentumConservationPrior,
    PhysicalBoundsPrior,
    PhysicsPrior,
    SymmetryPrior,
)

__all__ = [
    # Core tensor
    "DimTensor",
    # Layers
    "DimLayer",
    "DimLinear",
    "DimConv1d",
    "DimConv2d",
    "DimGraphConv",
    "DimSequential",
    # Attention
    "DimMultiheadAttention",
    "DimTransformerEncoderLayer",
    "DimTransformerEncoder",
    # Losses
    "DimMSELoss",
    "DimL1Loss",
    "DimHuberLoss",
    "PhysicsLoss",
    "CompositeLoss",
    # Normalization
    "DimBatchNorm1d",
    "DimBatchNorm2d",
    "DimLayerNorm",
    "DimInstanceNorm1d",
    "DimInstanceNorm2d",
    # Scalers
    "DimScaler",
    "MultiScaler",
    # Physics Priors
    "PhysicsPrior",
    "ConservationPrior",
    "EnergyConservationPrior",
    "MomentumConservationPrior",
    "SymmetryPrior",
    "DimensionalConsistencyPrior",
    "PhysicalBoundsPrior",
    # Benchmarks (as module)
    "benchmarks",
]

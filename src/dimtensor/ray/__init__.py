"""Ray integration for dimtensor.

Provides unit-aware distributed computing with Ray, including:
- Serialization for DimArray/DimTensor in Ray object store
- @dim_remote decorator for unit-aware remote functions
- DimDataset for Ray Data pipelines with unit tracking
- DimTrainContext for distributed training with physical units

Example:
    >>> import ray
    >>> from dimtensor import DimArray, units
    >>> from dimtensor.ray import dim_remote, dim_get
    >>>
    >>> ray.init()
    >>>
    >>> @dim_remote(num_cpus=2)
    ... def compute_energy(mass: DimArray, velocity: DimArray) -> DimArray:
    ...     return 0.5 * mass * velocity**2
    >>>
    >>> mass = DimArray([1.0, 2.0], units.kg)
    >>> velocity = DimArray([10.0, 20.0], units.m / units.s)
    >>> ref = compute_energy.remote(mass, velocity)
    >>> energy = dim_get(ref)
    >>> print(energy)  # [50.0, 400.0] J

Note:
    Ray must be installed separately: pip install dimtensor[ray]
"""

from __future__ import annotations

# Lazy imports to avoid requiring ray at import time
def __getattr__(name: str):
    """Lazy import Ray integration components."""
    if name == "serialize_dimarray":
        from .serialization import serialize_dimarray
        return serialize_dimarray
    elif name == "deserialize_dimarray":
        from .serialization import deserialize_dimarray
        return deserialize_dimarray
    elif name == "serialize_dimtensor":
        from .serialization import serialize_dimtensor
        return serialize_dimtensor
    elif name == "deserialize_dimtensor":
        from .serialization import deserialize_dimtensor
        return deserialize_dimtensor
    elif name == "register_serializers":
        from .serialization import register_serializers
        return register_serializers
    elif name == "dim_remote":
        from .remote import dim_remote
        return dim_remote
    elif name == "dim_get":
        from .remote import dim_get
        return dim_get
    elif name == "dim_put":
        from .remote import dim_put
        return dim_put
    elif name == "DimDataset":
        from .data import DimDataset
        return DimDataset
    elif name == "DimTrainContext":
        from .train import DimTrainContext
        return DimTrainContext
    elif name == "create_dim_trainer":
        from .train import create_dim_trainer
        return create_dim_trainer
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Serialization
    "serialize_dimarray",
    "deserialize_dimarray",
    "serialize_dimtensor",
    "deserialize_dimtensor",
    "register_serializers",
    # Remote functions
    "dim_remote",
    "dim_get",
    "dim_put",
    # Data pipelines
    "DimDataset",
    # Training
    "DimTrainContext",
    "create_dim_trainer",
]

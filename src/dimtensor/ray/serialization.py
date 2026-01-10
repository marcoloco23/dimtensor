"""Serialization for DimArray/DimTensor in Ray object store.

Provides efficient serialization that preserves unit information when
transferring dimensional data between Ray workers.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, TYPE_CHECKING

import numpy as np

from ..core.dimarray import DimArray
from ..core.dimensions import Dimension
from ..core.units import Unit

if TYPE_CHECKING:
    import torch


# Marker keys for identifying serialized types
DIMARRAY_MARKER = "__dimarray__"
DIMTENSOR_MARKER = "__dimtensor__"


def serialize_dimarray(arr: DimArray) -> dict[str, Any]:
    """Serialize DimArray for Ray object store.

    The numpy array data is stored directly (Ray handles numpy efficiently),
    while unit metadata is stored as simple types.

    Args:
        arr: DimArray to serialize.

    Returns:
        Dictionary representation suitable for Ray serialization.
    """
    result: dict[str, Any] = {
        DIMARRAY_MARKER: True,
        "data": arr._data,  # numpy array - Ray handles efficiently
        "unit": {
            "symbol": arr.unit.symbol,
            "dimension": {
                "length": float(arr.dimension.length),
                "mass": float(arr.dimension.mass),
                "time": float(arr.dimension.time),
                "current": float(arr.dimension.current),
                "temperature": float(arr.dimension.temperature),
                "amount": float(arr.dimension.amount),
                "luminosity": float(arr.dimension.luminosity),
            },
            "scale": arr.unit.scale,
        },
    }

    if arr._uncertainty is not None:
        result["uncertainty"] = arr._uncertainty

    return result


def deserialize_dimarray(data: dict[str, Any]) -> DimArray:
    """Deserialize DimArray from Ray object store.

    Args:
        data: Dictionary representation from serialize_dimarray.

    Returns:
        Reconstructed DimArray.

    Raises:
        ValueError: If data is not a valid serialized DimArray.
    """
    if not data.get(DIMARRAY_MARKER):
        raise ValueError("Not a serialized DimArray")

    # Reconstruct dimension
    dim_data = data["unit"]["dimension"]
    dimension = Dimension(
        length=Fraction(dim_data["length"]).limit_denominator(1000),
        mass=Fraction(dim_data["mass"]).limit_denominator(1000),
        time=Fraction(dim_data["time"]).limit_denominator(1000),
        current=Fraction(dim_data["current"]).limit_denominator(1000),
        temperature=Fraction(dim_data["temperature"]).limit_denominator(1000),
        amount=Fraction(dim_data["amount"]).limit_denominator(1000),
        luminosity=Fraction(dim_data["luminosity"]).limit_denominator(1000),
    )

    # Reconstruct unit
    unit = Unit(
        symbol=data["unit"]["symbol"],
        dimension=dimension,
        scale=data["unit"]["scale"],
    )

    # Get data and uncertainty
    arr_data = np.asarray(data["data"])
    uncertainty = data.get("uncertainty")
    if uncertainty is not None:
        uncertainty = np.asarray(uncertainty)

    return DimArray._from_data_and_unit(arr_data, unit, uncertainty)


def serialize_dimtensor(tensor: "DimTensor") -> dict[str, Any]:
    """Serialize DimTensor for Ray object store.

    GPU tensors are moved to CPU before serialization.

    Args:
        tensor: DimTensor to serialize.

    Returns:
        Dictionary representation suitable for Ray serialization.
    """
    from ..torch.dimtensor import DimTensor

    # Move to CPU and convert to numpy for efficient serialization
    cpu_data = tensor._data.detach().cpu().numpy()

    return {
        DIMTENSOR_MARKER: True,
        "data": cpu_data,
        "unit": {
            "symbol": tensor.unit.symbol,
            "dimension": {
                "length": float(tensor.dimension.length),
                "mass": float(tensor.dimension.mass),
                "time": float(tensor.dimension.time),
                "current": float(tensor.dimension.current),
                "temperature": float(tensor.dimension.temperature),
                "amount": float(tensor.dimension.amount),
                "luminosity": float(tensor.dimension.luminosity),
            },
            "scale": tensor.unit.scale,
        },
        "dtype": str(tensor.dtype).split(".")[-1],  # e.g., "float32"
        "requires_grad": tensor.requires_grad,
    }


def deserialize_dimtensor(
    data: dict[str, Any],
    device: str | None = None,
) -> "DimTensor":
    """Deserialize DimTensor from Ray object store.

    Args:
        data: Dictionary representation from serialize_dimtensor.
        device: Target device ('cpu', 'cuda', 'mps'). If None, uses 'cpu'.

    Returns:
        Reconstructed DimTensor.

    Raises:
        ValueError: If data is not a valid serialized DimTensor.
    """
    import torch
    from ..torch.dimtensor import DimTensor

    if not data.get(DIMTENSOR_MARKER):
        raise ValueError("Not a serialized DimTensor")

    # Reconstruct dimension
    dim_data = data["unit"]["dimension"]
    dimension = Dimension(
        length=Fraction(dim_data["length"]).limit_denominator(1000),
        mass=Fraction(dim_data["mass"]).limit_denominator(1000),
        time=Fraction(dim_data["time"]).limit_denominator(1000),
        current=Fraction(dim_data["current"]).limit_denominator(1000),
        temperature=Fraction(dim_data["temperature"]).limit_denominator(1000),
        amount=Fraction(dim_data["amount"]).limit_denominator(1000),
        luminosity=Fraction(dim_data["luminosity"]).limit_denominator(1000),
    )

    # Reconstruct unit
    unit = Unit(
        symbol=data["unit"]["symbol"],
        dimension=dimension,
        scale=data["unit"]["scale"],
    )

    # Convert numpy back to tensor
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    dtype = dtype_map.get(data.get("dtype", "float32"), torch.float32)

    tensor = torch.tensor(data["data"], dtype=dtype)

    if device is not None:
        tensor = tensor.to(device=device)

    if data.get("requires_grad", False):
        tensor = tensor.requires_grad_(True)

    return DimTensor._from_tensor_and_unit(tensor, unit)


def is_serialized_dimarray(data: Any) -> bool:
    """Check if data is a serialized DimArray.

    Args:
        data: Object to check.

    Returns:
        True if data is a serialized DimArray.
    """
    return isinstance(data, dict) and data.get(DIMARRAY_MARKER, False)


def is_serialized_dimtensor(data: Any) -> bool:
    """Check if data is a serialized DimTensor.

    Args:
        data: Object to check.

    Returns:
        True if data is a serialized DimTensor.
    """
    return isinstance(data, dict) and data.get(DIMTENSOR_MARKER, False)


def is_serialized_dim(data: Any) -> bool:
    """Check if data is a serialized DimArray or DimTensor.

    Args:
        data: Object to check.

    Returns:
        True if data is a serialized dimensional type.
    """
    return is_serialized_dimarray(data) or is_serialized_dimtensor(data)


def register_serializers() -> None:
    """Register dimtensor serializers with Ray.

    Call this once at startup to enable automatic serialization
    of DimArray and DimTensor in Ray remote functions.

    Note:
        This modifies global Ray state. For more explicit control,
        use dim_remote and dim_get instead.

    Example:
        >>> import ray
        >>> from dimtensor.ray import register_serializers
        >>> ray.init()
        >>> register_serializers()
        >>> # Now DimArray/DimTensor can be passed directly to @ray.remote functions
    """
    import ray

    # Register DimArray serializer
    ray.util.register_serializer(
        DimArray,
        serializer=serialize_dimarray,
        deserializer=deserialize_dimarray,
    )

    # Try to register DimTensor serializer if torch is available
    try:
        from ..torch.dimtensor import DimTensor

        ray.util.register_serializer(
            DimTensor,
            serializer=serialize_dimtensor,
            deserializer=deserialize_dimtensor,
        )
    except ImportError:
        pass  # torch not available

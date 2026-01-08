"""JSON serialization for DimArray.

Simple JSON format that preserves unit information.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..core.dimarray import DimArray
from ..core.units import Unit, dimensionless


def to_dict(arr: DimArray) -> dict[str, Any]:
    """Convert DimArray to a dictionary suitable for JSON.

    Args:
        arr: DimArray to serialize.

    Returns:
        Dictionary with data, unit, and optional uncertainty.
    """
    result: dict[str, Any] = {
        "data": arr._data.tolist(),
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
        "dtype": str(arr.dtype),
    }

    if arr.has_uncertainty and arr._uncertainty is not None:
        result["uncertainty"] = arr._uncertainty.tolist()

    return result


def from_dict(data: dict[str, Any]) -> DimArray:
    """Create DimArray from a dictionary.

    Args:
        data: Dictionary with data, unit, and optional uncertainty.

    Returns:
        DimArray.
    """
    from ..core.dimensions import Dimension
    from fractions import Fraction

    arr_data = np.array(data["data"])

    # Reconstruct dimension
    dim_data = data["unit"]["dimension"]
    dimension = Dimension(
        length=Fraction(dim_data["length"]).limit_denominator(),
        mass=Fraction(dim_data["mass"]).limit_denominator(),
        time=Fraction(dim_data["time"]).limit_denominator(),
        current=Fraction(dim_data["current"]).limit_denominator(),
        temperature=Fraction(dim_data["temperature"]).limit_denominator(),
        amount=Fraction(dim_data["amount"]).limit_denominator(),
        luminosity=Fraction(dim_data["luminosity"]).limit_denominator(),
    )

    # Reconstruct unit
    unit = Unit(
        symbol=data["unit"]["symbol"],
        dimension=dimension,
        scale=data["unit"]["scale"],
    )

    # Handle uncertainty
    uncertainty = None
    if "uncertainty" in data:
        uncertainty = np.array(data["uncertainty"])

    return DimArray(arr_data, unit, uncertainty=uncertainty)


def to_json(arr: DimArray, indent: int | None = 2) -> str:
    """Serialize DimArray to JSON string.

    Args:
        arr: DimArray to serialize.
        indent: Indentation for pretty printing. None for compact.

    Returns:
        JSON string.
    """
    return json.dumps(to_dict(arr), indent=indent)


def from_json(json_str: str) -> DimArray:
    """Deserialize DimArray from JSON string.

    Args:
        json_str: JSON string.

    Returns:
        DimArray.
    """
    data = json.loads(json_str)
    return from_dict(data)


def save_json(arr: DimArray, path: str | Path, indent: int | None = 2) -> None:
    """Save DimArray to JSON file.

    Args:
        arr: DimArray to save.
        path: File path.
        indent: Indentation for pretty printing.
    """
    path = Path(path)
    with open(path, "w") as f:
        json.dump(to_dict(arr), f, indent=indent)


def load_json(path: str | Path) -> DimArray:
    """Load DimArray from JSON file.

    Args:
        path: File path.

    Returns:
        DimArray.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return from_dict(data)

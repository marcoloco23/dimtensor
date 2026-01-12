"""HTTP-friendly serialization for serverless environments.

Provides efficient base64 encoding for DimArray objects suitable for
JSON-based HTTP APIs (AWS Lambda, Google Cloud Functions).
"""

from __future__ import annotations

from typing import Any
import json
import base64
import numpy as np
from ..core.dimarray import DimArray
from ..core.dimensions import Dimension
from ..core.units import Unit
from fractions import Fraction


def serialize_for_http(arr: DimArray) -> dict[str, Any]:
    """Serialize DimArray for HTTP response (JSON-compatible).

    Uses base64 encoding for efficient binary transport while remaining
    JSON-compatible for serverless HTTP APIs.

    Args:
        arr: DimArray to serialize.

    Returns:
        Dictionary with:
        - data: base64-encoded numpy array
        - unit: unit symbol
        - dimension: dimension tuple (7 floats)
        - shape: array shape
        - dtype: numpy dtype as string
        - uncertainty: base64-encoded uncertainty (if present)
        - scale: unit scale factor

    Example:
        >>> from dimtensor import DimArray, units
        >>> v = DimArray([1.0, 2.0], units.m / units.s)
        >>> data = serialize_for_http(v)
        >>> # Can be JSON-serialized for HTTP response
        >>> import json
        >>> json.dumps(data)
    """
    # Encode data as base64 bytes
    data_bytes = arr._data.tobytes()
    data_b64 = base64.b64encode(data_bytes).decode("utf-8")

    # Encode uncertainty if present
    uncertainty_b64 = None
    if arr._uncertainty is not None:
        unc_bytes = arr._uncertainty.tobytes()
        uncertainty_b64 = base64.b64encode(unc_bytes).decode("utf-8")

    return {
        "data": data_b64,
        "unit": arr.unit.symbol,
        "dimension": [
            float(arr.dimension.length),
            float(arr.dimension.mass),
            float(arr.dimension.time),
            float(arr.dimension.current),
            float(arr.dimension.temperature),
            float(arr.dimension.amount),
            float(arr.dimension.luminosity),
        ],
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "scale": arr.unit.scale,
        "uncertainty": uncertainty_b64,
    }


def deserialize_from_http(data: dict[str, Any]) -> DimArray:
    """Deserialize DimArray from HTTP request data.

    Args:
        data: Dictionary with serialized DimArray data
              (output from serialize_for_http).

    Returns:
        DimArray reconstructed from HTTP data.

    Example:
        >>> data = {
        ...     "data": "AACAPwAAAEA=",  # base64 [1.0, 2.0]
        ...     "unit": "m/s",
        ...     "dimension": [1, 0, -1, 0, 0, 0, 0],
        ...     "shape": [2],
        ...     "dtype": "float64",
        ...     "scale": 1.0,
        ...     "uncertainty": None
        ... }
        >>> arr = deserialize_from_http(data)
    """
    # Decode numpy array from base64
    data_bytes = base64.b64decode(data["data"])
    arr = np.frombuffer(data_bytes, dtype=np.dtype(data["dtype"])).reshape(
        data["shape"]
    )

    # Reconstruct dimension
    dim_tuple = data["dimension"]
    dimension = Dimension(
        length=Fraction(dim_tuple[0]).limit_denominator(),
        mass=Fraction(dim_tuple[1]).limit_denominator(),
        time=Fraction(dim_tuple[2]).limit_denominator(),
        current=Fraction(dim_tuple[3]).limit_denominator(),
        temperature=Fraction(dim_tuple[4]).limit_denominator(),
        amount=Fraction(dim_tuple[5]).limit_denominator(),
        luminosity=Fraction(dim_tuple[6]).limit_denominator(),
    )

    # Reconstruct unit
    unit = Unit(symbol=data["unit"], dimension=dimension, scale=data["scale"])

    # Decode uncertainty if present
    uncertainty = None
    if data.get("uncertainty"):
        unc_bytes = base64.b64decode(data["uncertainty"])
        uncertainty = np.frombuffer(unc_bytes, dtype=np.dtype(data["dtype"])).reshape(
            data["shape"]
        )

    return DimArray._from_data_and_unit(arr, unit, uncertainty)


def simple_response(
    arr: DimArray, status_code: int = 200, version: str = "5.0.0"
) -> dict[str, Any]:
    """Create Lambda/Cloud Functions response dictionary.

    Wraps serialized DimArray in standard HTTP response format.

    Args:
        arr: DimArray to return in response.
        status_code: HTTP status code (default 200).
        version: dimtensor version for X-DimTensor-Version header.

    Returns:
        Dictionary with statusCode, body (JSON string), and headers.

    Example:
        >>> from dimtensor import DimArray, units
        >>> result = DimArray([42.0], units.J)
        >>> response = simple_response(result)
        >>> # Returns format suitable for Lambda/Cloud Functions
    """
    return {
        "statusCode": status_code,
        "body": json.dumps(serialize_for_http(arr)),
        "headers": {
            "Content-Type": "application/json",
            "X-DimTensor-Version": version,
        },
    }


def error_response(
    error_type: str, message: str, status_code: int = 500, traceback: str | None = None
) -> dict[str, Any]:
    """Create error response dictionary.

    Args:
        error_type: Type of error (e.g., "DimensionError").
        message: Error message.
        status_code: HTTP status code (default 500).
        traceback: Optional traceback string.

    Returns:
        Dictionary with statusCode, error body, and headers.
    """
    body: dict[str, Any] = {"error": error_type, "message": message}
    if traceback:
        body["traceback"] = traceback

    return {
        "statusCode": status_code,
        "body": json.dumps(body),
        "headers": {"Content-Type": "application/json"},
    }

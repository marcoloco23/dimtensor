"""Rust backend integration for dimtensor.

This module provides a unified interface to the optional Rust backend.
If the Rust backend is not available, it provides Python fallbacks for
all operations.

Usage:
    from dimtensor._rust import HAS_RUST_BACKEND, add_arrays

    if HAS_RUST_BACKEND:
        # Using Rust-accelerated operations
        result = add_arrays(a, b, dim_a, dim_b)
    else:
        # Using pure Python
        result = add_arrays(a, b, dim_a, dim_b)  # Same API, Python impl
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .core.dimensions import Dimension

# Try to import the Rust backend
_rust_add_arrays: Any = None
_rust_sub_arrays: Any = None
_rust_mul_arrays: Any = None
_rust_div_arrays: Any = None
RustDimension: Any = None

try:
    import dimtensor_core as _core  # noqa: F401

    RustDimension = _core.RustDimension
    _rust_add_arrays = _core.add_arrays
    _rust_sub_arrays = _core.sub_arrays
    _rust_mul_arrays = _core.mul_arrays
    _rust_div_arrays = _core.div_arrays
    HAS_RUST_BACKEND = True

except ImportError:
    HAS_RUST_BACKEND = False

    # Python fallback implementations

    def dimensions_compatible(dim_a: Dimension, dim_b: Dimension) -> bool:
        """Check if two dimensions are compatible (Python fallback)."""
        return dim_a == dim_b

    def multiply_dimensions(dim_a: Dimension, dim_b: Dimension) -> Dimension:
        """Multiply two dimensions (Python fallback)."""
        return dim_a * dim_b

    def divide_dimensions(dim_a: Dimension, dim_b: Dimension) -> Dimension:
        """Divide two dimensions (Python fallback)."""
        return dim_a / dim_b

    def power_dimension(dim: Dimension, power: int) -> Dimension:
        """Raise dimension to a power (Python fallback)."""
        return dim ** power


def add_arrays(
    a: NDArray[Any],
    b: NDArray[Any],
    dim_a: Dimension,
    dim_b: Dimension,
) -> NDArray[Any]:
    """Add two arrays with dimension checking.

    Args:
        a: First array
        b: Second array
        dim_a: Dimension of first array
        dim_b: Dimension of second array

    Returns:
        Result array

    Raises:
        ValueError: If dimensions are not compatible
    """
    if HAS_RUST_BACKEND:
        rust_dim_a = _to_rust_dimension(dim_a)
        rust_dim_b = _to_rust_dimension(dim_b)
        result: NDArray[Any] = _rust_add_arrays(a, b, rust_dim_a, rust_dim_b)
        return result
    else:
        if dim_a != dim_b:
            raise ValueError(
                f"Cannot add arrays with incompatible dimensions: {dim_a} and {dim_b}"
            )
        result = a + b
        return result


def sub_arrays(
    a: NDArray[Any],
    b: NDArray[Any],
    dim_a: Dimension,
    dim_b: Dimension,
) -> NDArray[Any]:
    """Subtract two arrays with dimension checking.

    Args:
        a: First array
        b: Second array
        dim_a: Dimension of first array
        dim_b: Dimension of second array

    Returns:
        Result array

    Raises:
        ValueError: If dimensions are not compatible
    """
    if HAS_RUST_BACKEND:
        rust_dim_a = _to_rust_dimension(dim_a)
        rust_dim_b = _to_rust_dimension(dim_b)
        result: NDArray[Any] = _rust_sub_arrays(a, b, rust_dim_a, rust_dim_b)
        return result
    else:
        if dim_a != dim_b:
            raise ValueError(
                f"Cannot subtract arrays with incompatible dimensions: {dim_a} and {dim_b}"
            )
        result = a - b
        return result


def mul_arrays(
    a: NDArray[Any],
    b: NDArray[Any],
    dim_a: Dimension,
    dim_b: Dimension,
) -> tuple[NDArray[Any], Dimension]:
    """Multiply two arrays, combining their dimensions.

    Args:
        a: First array
        b: Second array
        dim_a: Dimension of first array
        dim_b: Dimension of second array

    Returns:
        Tuple of (result array, result dimension)
    """
    if HAS_RUST_BACKEND:
        rust_dim_a = _to_rust_dimension(dim_a)
        rust_dim_b = _to_rust_dimension(dim_b)
        rust_result, rust_dim = _rust_mul_arrays(a, b, rust_dim_a, rust_dim_b)
        return rust_result, _from_rust_dimension(rust_dim)
    result_arr: NDArray[Any] = a * b
    result_dim: Dimension = dim_a * dim_b
    return result_arr, result_dim


def div_arrays(
    a: NDArray[Any],
    b: NDArray[Any],
    dim_a: Dimension,
    dim_b: Dimension,
) -> tuple[NDArray[Any], Dimension]:
    """Divide two arrays, combining their dimensions.

    Args:
        a: First array
        b: Second array
        dim_a: Dimension of first array
        dim_b: Dimension of second array

    Returns:
        Tuple of (result array, result dimension)
    """
    if HAS_RUST_BACKEND:
        rust_dim_a = _to_rust_dimension(dim_a)
        rust_dim_b = _to_rust_dimension(dim_b)
        rust_result, rust_dim = _rust_div_arrays(a, b, rust_dim_a, rust_dim_b)
        return rust_result, _from_rust_dimension(rust_dim)
    result_arr: NDArray[Any] = a / b
    result_dim: Dimension = dim_a / dim_b
    return result_arr, result_dim


def _to_rust_dimension(dim: Dimension) -> Any:
    """Convert a Python Dimension to a RustDimension.

    Args:
        dim: Python Dimension object

    Returns:
        RustDimension object
    """
    if not HAS_RUST_BACKEND:
        raise RuntimeError("Rust backend not available")

    from fractions import Fraction

    def frac_to_tuple(f: Fraction) -> tuple[int, int]:
        return (f.numerator, f.denominator)

    return RustDimension(
        length=int(dim.length),
        mass=int(dim.mass),
        time=int(dim.time),
        current=int(dim.current),
        temperature=int(dim.temperature),
        amount=int(dim.amount),
        luminosity=int(dim.luminosity),
    )


def _from_rust_dimension(rust_dim: Any) -> Dimension:
    """Convert a RustDimension to a Python Dimension.

    Args:
        rust_dim: RustDimension object

    Returns:
        Python Dimension object
    """
    from fractions import Fraction

    from .core.dimensions import Dimension

    def tuple_to_frac(t: tuple[int, int]) -> Fraction:
        return Fraction(t[0], t[1])

    return Dimension(
        length=tuple_to_frac(rust_dim.length),
        mass=tuple_to_frac(rust_dim.mass),
        time=tuple_to_frac(rust_dim.time),
        current=tuple_to_frac(rust_dim.current),
        temperature=tuple_to_frac(rust_dim.temperature),
        amount=tuple_to_frac(rust_dim.amount),
        luminosity=tuple_to_frac(rust_dim.luminosity),
    )


__all__ = [
    "HAS_RUST_BACKEND",
    "add_arrays",
    "sub_arrays",
    "mul_arrays",
    "div_arrays",
    "dimensions_compatible",
    "multiply_dimensions",
    "divide_dimensions",
    "power_dimension",
]

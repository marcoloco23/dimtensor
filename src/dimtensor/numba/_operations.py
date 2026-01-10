"""JIT-compiled dimension operations for Numba.

These functions operate on encoded dimensions (int64 arrays of length 14)
and can be used inside Numba nopython mode for efficient dimension algebra.
"""

from __future__ import annotations

import numpy as np
from numba import njit, types
from numpy.typing import NDArray

from ._encoding import ENCODING_SIZE, NUM_DIMENSIONS


@njit(inline="always")
def gcd(a: int, b: int) -> int:
    """Greatest common divisor using Euclidean algorithm.

    Args:
        a: First integer (can be negative).
        b: Second integer (must be positive for correct operation).

    Returns:
        GCD of |a| and |b|.
    """
    a = abs(a)
    while b:
        a, b = b, a % b
    return a


@njit(inline="always")
def simplify_fraction(num: int, den: int) -> tuple[int, int]:
    """Simplify a fraction by dividing by GCD.

    Args:
        num: Numerator.
        den: Denominator (must be non-zero).

    Returns:
        Simplified (numerator, denominator) tuple with positive denominator.
    """
    if den < 0:
        num = -num
        den = -den
    if num == 0:
        return (0, 1)
    g = gcd(num, den)
    return (num // g, den // g)


@njit
def dims_equal(dim_a: NDArray[np.int64], dim_b: NDArray[np.int64]) -> bool:
    """Check if two encoded dimensions are equal.

    Dimensions are equal if all exponents are equal (after simplification).

    Args:
        dim_a: First encoded dimension (int64[14]).
        dim_b: Second encoded dimension (int64[14]).

    Returns:
        True if dimensions are equal.
    """
    for i in range(NUM_DIMENSIONS):
        idx = 2 * i
        # Get simplified fractions for comparison
        a_num, a_den = simplify_fraction(dim_a[idx], dim_a[idx + 1])
        b_num, b_den = simplify_fraction(dim_b[idx], dim_b[idx + 1])

        # Compare cross-multiplied to avoid floating point
        if a_num * b_den != b_num * a_den:
            return False
    return True


@njit
def dims_multiply(dim_a: NDArray[np.int64], dim_b: NDArray[np.int64]) -> NDArray[np.int64]:
    """Multiply two encoded dimensions (add exponents).

    For multiplication: (a/b) + (c/d) = (ad + bc) / (bd)

    Args:
        dim_a: First encoded dimension (int64[14]).
        dim_b: Second encoded dimension (int64[14]).

    Returns:
        Encoded dimension representing dim_a * dim_b.
    """
    result = np.empty(ENCODING_SIZE, dtype=np.int64)

    for i in range(NUM_DIMENSIONS):
        idx = 2 * i
        a_num, a_den = dim_a[idx], dim_a[idx + 1]
        b_num, b_den = dim_b[idx], dim_b[idx + 1]

        # Add fractions: (a_num/a_den) + (b_num/b_den)
        new_num = a_num * b_den + b_num * a_den
        new_den = a_den * b_den

        # Simplify
        new_num, new_den = simplify_fraction(new_num, new_den)
        result[idx] = new_num
        result[idx + 1] = new_den

    return result


@njit
def dims_divide(dim_a: NDArray[np.int64], dim_b: NDArray[np.int64]) -> NDArray[np.int64]:
    """Divide two encoded dimensions (subtract exponents).

    For division: (a/b) - (c/d) = (ad - bc) / (bd)

    Args:
        dim_a: Numerator encoded dimension (int64[14]).
        dim_b: Denominator encoded dimension (int64[14]).

    Returns:
        Encoded dimension representing dim_a / dim_b.
    """
    result = np.empty(ENCODING_SIZE, dtype=np.int64)

    for i in range(NUM_DIMENSIONS):
        idx = 2 * i
        a_num, a_den = dim_a[idx], dim_a[idx + 1]
        b_num, b_den = dim_b[idx], dim_b[idx + 1]

        # Subtract fractions: (a_num/a_den) - (b_num/b_den)
        new_num = a_num * b_den - b_num * a_den
        new_den = a_den * b_den

        # Simplify
        new_num, new_den = simplify_fraction(new_num, new_den)
        result[idx] = new_num
        result[idx + 1] = new_den

    return result


@njit
def dims_power(dim: NDArray[np.int64], power_num: int, power_den: int) -> NDArray[np.int64]:
    """Raise an encoded dimension to a rational power.

    For power: (a/b) * (p/q) = (a*p) / (b*q)

    Args:
        dim: Encoded dimension (int64[14]).
        power_num: Numerator of the power.
        power_den: Denominator of the power (must be non-zero).

    Returns:
        Encoded dimension representing dim ** (power_num/power_den).
    """
    result = np.empty(ENCODING_SIZE, dtype=np.int64)

    for i in range(NUM_DIMENSIONS):
        idx = 2 * i
        d_num, d_den = dim[idx], dim[idx + 1]

        # Multiply: (d_num/d_den) * (power_num/power_den)
        new_num = d_num * power_num
        new_den = d_den * power_den

        # Simplify
        new_num, new_den = simplify_fraction(new_num, new_den)
        result[idx] = new_num
        result[idx + 1] = new_den

    return result


@njit
def dims_is_dimensionless(dim: NDArray[np.int64]) -> bool:
    """Check if an encoded dimension is dimensionless.

    Args:
        dim: Encoded dimension (int64[14]).

    Returns:
        True if all numerators are zero.
    """
    for i in range(NUM_DIMENSIONS):
        if dim[2 * i] != 0:
            return False
    return True


@njit
def create_dimensionless() -> NDArray[np.int64]:
    """Create an encoded dimensionless quantity.

    Returns:
        Encoded dimension with all exponents zero (denominators = 1).
    """
    result = np.zeros(ENCODING_SIZE, dtype=np.int64)
    for i in range(NUM_DIMENSIONS):
        result[2 * i + 1] = 1  # Set denominators to 1
    return result


__all__ = [
    "gcd",
    "simplify_fraction",
    "dims_equal",
    "dims_multiply",
    "dims_divide",
    "dims_power",
    "dims_is_dimensionless",
    "create_dimensionless",
]

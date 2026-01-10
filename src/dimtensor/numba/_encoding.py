"""Dimension encoding/decoding for Numba compatibility.

Dimensions are encoded as a tuple of 14 int64 values:
  - 7 numerator/denominator pairs for the 7 SI base dimensions
  - Order: (L_num, L_den, M_num, M_den, T_num, T_den, I_num, I_den,
            Theta_num, Theta_den, N_num, N_den, J_num, J_den)

This encoding allows dimension operations to be performed in Numba's
nopython mode using pure integer arithmetic.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..core.dimensions import Dimension


# Number of SI base dimensions
NUM_DIMENSIONS = 7

# Encoding size: 2 integers (num, den) per dimension
ENCODING_SIZE = NUM_DIMENSIONS * 2  # = 14


def encode_dimension(dim: Dimension) -> NDArray[np.int64]:
    """Encode a Dimension as a numpy array of 14 int64 values.

    The encoding stores numerator/denominator pairs for each of the
    7 SI base dimensions, allowing dimension algebra in Numba nopython mode.

    Args:
        dim: Dimension object to encode.

    Returns:
        numpy array of shape (14,) with dtype int64 containing the
        encoded dimension exponents.

    Example:
        >>> from dimtensor.core.dimensions import Dimension
        >>> dim = Dimension(length=1, time=-2)  # acceleration
        >>> encoded = encode_dimension(dim)
        >>> encoded  # [1, 1, 0, 1, -2, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    """
    result = np.zeros(ENCODING_SIZE, dtype=np.int64)
    for i, exp in enumerate(dim._exponents):
        result[2 * i] = exp.numerator
        result[2 * i + 1] = exp.denominator
    return result


def decode_dimension(encoded: NDArray[np.int64]) -> Dimension:
    """Decode a numpy array of 14 int64 values back to a Dimension.

    Args:
        encoded: numpy array of shape (14,) with dtype int64 containing
            the encoded dimension exponents.

    Returns:
        Dimension object reconstructed from the encoding.

    Example:
        >>> encoded = np.array([1, 1, 0, 1, -2, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        >>> dim = decode_dimension(encoded)
        >>> dim  # Dimension(length=1, time=-2)
    """
    from ..core.dimensions import Dimension

    exponents = []
    for i in range(NUM_DIMENSIONS):
        num = int(encoded[2 * i])
        den = int(encoded[2 * i + 1])
        # Handle zero denominator (shouldn't happen with valid input)
        if den == 0:
            den = 1
        exponents.append(Fraction(num, den))

    return Dimension._from_exponents(tuple(exponents))


def encode_dimensionless() -> NDArray[np.int64]:
    """Return the encoding for a dimensionless quantity.

    Returns:
        numpy array representing dimensionless (all zeros for numerators,
        ones for denominators).
    """
    result = np.zeros(ENCODING_SIZE, dtype=np.int64)
    # Set all denominators to 1
    for i in range(NUM_DIMENSIONS):
        result[2 * i + 1] = 1
    return result


def is_encoded_dimensionless(encoded: NDArray[np.int64]) -> bool:
    """Check if an encoded dimension represents dimensionless.

    Args:
        encoded: numpy array of shape (14,) with encoded dimension.

    Returns:
        True if all numerators are zero (dimensionless).
    """
    for i in range(NUM_DIMENSIONS):
        if encoded[2 * i] != 0:
            return False
    return True


__all__ = [
    "ENCODING_SIZE",
    "NUM_DIMENSIONS",
    "encode_dimension",
    "decode_dimension",
    "encode_dimensionless",
    "is_encoded_dimensionless",
]

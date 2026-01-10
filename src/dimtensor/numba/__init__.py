"""Numba integration for dimtensor.

Provides Numba JIT compilation support for unit-aware operations.
Dimension checking occurs before JIT-compiled code runs, enabling
efficient parallel loops with `prange` while maintaining dimensional safety.

Example:
    >>> from dimtensor import DimArray, units
    >>> from dimtensor.numba import dim_jit
    >>>
    >>> @dim_jit
    ... def kinetic_energy(mass, velocity):
    ...     return 0.5 * mass * velocity**2
    >>>
    >>> m = DimArray([1.0, 2.0], units.kg)
    >>> v = DimArray([3.0, 4.0], units.m / units.s)
    >>> energy = kinetic_energy(m, v)  # JIT-compiled with dimension checking

Note:
    Numba is an optional dependency. Install with:
        pip install dimtensor[numba]
    or:
        pip install numba
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Check if numba is available
try:
    import numba  # noqa: F401
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    from ._decorators import dim_jit, extract_arrays, wrap_result
    from ._encoding import decode_dimension, encode_dimension
    from ._operations import (
        dims_equal,
        dims_multiply,
        dims_divide,
        dims_power,
        gcd,
    )
    # Re-export prange for convenience
    from numba import prange
else:
    # Provide stub that raises helpful error
    def _not_available(*args, **kwargs):
        raise ImportError(
            "Numba is not installed. Install it with:\n"
            "  pip install dimtensor[numba]\n"
            "or:\n"
            "  pip install numba"
        )

    dim_jit = _not_available
    extract_arrays = _not_available
    wrap_result = _not_available
    encode_dimension = _not_available
    decode_dimension = _not_available
    dims_equal = _not_available
    dims_multiply = _not_available
    dims_divide = _not_available
    dims_power = _not_available
    gcd = _not_available
    prange = _not_available


__all__ = [
    "HAS_NUMBA",
    "dim_jit",
    "extract_arrays",
    "wrap_result",
    "encode_dimension",
    "decode_dimension",
    "dims_equal",
    "dims_multiply",
    "dims_divide",
    "dims_power",
    "gcd",
    "prange",
]

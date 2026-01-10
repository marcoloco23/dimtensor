"""Decorators for Numba JIT compilation with dimension awareness.

The @dim_jit decorator wraps functions to:
1. Extract raw arrays and dimensions from DimArray inputs
2. Validate dimensions before JIT compilation
3. Run the JIT-compiled function on raw arrays
4. Wrap results back into DimArray with computed dimensions
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

from numba import njit, prange  # noqa: F401

from ..core.dimarray import DimArray
from ..core.units import Unit, dimensionless
from ..errors import DimensionError
from ._encoding import decode_dimension, encode_dimension

F = TypeVar("F", bound=Callable[..., Any])


def extract_arrays(
    *args: Any,
) -> tuple[tuple[NDArray[Any], ...], tuple[NDArray[np.int64] | None, ...], tuple[float, ...]]:
    """Extract raw arrays, encoded dimensions, and scales from DimArray inputs.

    Args:
        *args: Mix of DimArray and other values.

    Returns:
        Tuple of:
        - raw_arrays: tuple of numpy arrays (raw data from DimArrays, or original values)
        - encoded_dims: tuple of encoded dimensions (or None for non-DimArray values)
        - scales: tuple of scale factors (or 1.0 for non-DimArray values)
    """
    raw_arrays = []
    encoded_dims = []
    scales = []

    for arg in args:
        if isinstance(arg, DimArray):
            raw_arrays.append(arg._data)
            encoded_dims.append(encode_dimension(arg.dimension))
            scales.append(arg.unit.scale)
        else:
            # Keep raw values as-is
            if isinstance(arg, (int, float)):
                raw_arrays.append(np.asarray(arg))
            else:
                raw_arrays.append(arg)
            encoded_dims.append(None)
            scales.append(1.0)

    return tuple(raw_arrays), tuple(encoded_dims), tuple(scales)


def wrap_result(
    result: NDArray[Any],
    encoded_dim: NDArray[np.int64],
    scale: float = 1.0,
    symbol: str | None = None,
) -> DimArray:
    """Wrap a raw array result into a DimArray.

    Args:
        result: Raw numpy array from JIT-compiled function.
        encoded_dim: Encoded dimension for the result.
        scale: Scale factor for the unit (default 1.0 for SI base units).
        symbol: Optional symbol for the unit.

    Returns:
        DimArray with the result and computed dimension.
    """
    dim = decode_dimension(encoded_dim)
    if symbol is None:
        symbol = str(dim) if not dim.is_dimensionless else "1"
    unit = Unit(symbol, dim, scale)
    return DimArray._from_data_and_unit(result, unit)


@overload
def dim_jit(fn: F) -> F: ...


@overload
def dim_jit(
    fn: None = None,
    *,
    parallel: bool = False,
    cache: bool = True,
    fastmath: bool = False,
) -> Callable[[F], F]: ...


def dim_jit(
    fn: F | None = None,
    *,
    parallel: bool = False,
    cache: bool = True,
    fastmath: bool = False,
) -> F | Callable[[F], F]:
    """JIT compile a function with dimension-aware inputs.

    This decorator provides Numba JIT compilation for functions that operate
    on DimArray inputs. Dimension checking occurs before the JIT-compiled
    code runs, allowing for efficient parallel loops while maintaining
    dimensional safety.

    The decorated function should be written to work with raw numpy arrays.
    The decorator handles:
    1. Extracting raw arrays from DimArray inputs
    2. Validating dimension compatibility
    3. Running the JIT-compiled function
    4. Wrapping results back into DimArray

    Args:
        fn: Function to decorate (optional, allows use without parentheses).
        parallel: Enable automatic parallelization with prange (default: False).
        cache: Cache compiled function to disk (default: True).
        fastmath: Enable fast math optimizations (default: False).

    Returns:
        Decorated function that accepts DimArray inputs and returns DimArray.

    Example:
        >>> @dim_jit
        ... def kinetic_energy(mass, velocity):
        ...     return 0.5 * mass * velocity**2
        >>>
        >>> @dim_jit(parallel=True)
        ... def parallel_sum(arr):
        ...     total = 0.0
        ...     for i in prange(arr.shape[0]):
        ...         total += arr[i]
        ...     return total

    Note:
        The decorated function receives raw numpy arrays, not DimArray objects.
        Dimension checking and unit handling is done by the wrapper.
    """

    def decorator(func: F) -> F:
        # Get function signature for argument inspection
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        # Create the JIT-compiled inner function
        jit_func = njit(parallel=parallel, cache=cache, fastmath=fastmath)(func)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Bind arguments to parameter names
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Extract arrays, dimensions, and scales
            raw_args = []
            dim_arrays_info: list[tuple[int, NDArray[np.int64], float]] = []

            for i, (name, value) in enumerate(bound.arguments.items()):
                if isinstance(value, DimArray):
                    raw_args.append(value._data)
                    dim_arrays_info.append((i, encode_dimension(value.dimension), value.unit.scale))
                elif isinstance(value, (int, float)):
                    raw_args.append(np.asarray(value, dtype=np.float64))
                else:
                    raw_args.append(value)

            # Run the JIT-compiled function
            result = jit_func(*raw_args)

            # If no DimArray inputs, return raw result
            if not dim_arrays_info:
                return result

            # Compute output dimension based on first DimArray input
            # This is a simple heuristic - for more complex dimension inference,
            # users should use explicit dimension tracking
            first_dim_idx, first_encoded_dim, first_scale = dim_arrays_info[0]

            # Return result wrapped in DimArray
            # For scalar results, ensure they're arrays
            if np.isscalar(result):
                result = np.array([result])

            # Default: assume output has same dimension as first DimArray input
            # This works for many common operations but may need refinement
            # for operations that change dimensions (like multiplication)
            return wrap_result(result, first_encoded_dim, first_scale)

        return wrapper  # type: ignore

    if fn is not None:
        return decorator(fn)
    return decorator


__all__ = [
    "dim_jit",
    "extract_arrays",
    "wrap_result",
]

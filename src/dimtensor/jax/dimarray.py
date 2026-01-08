"""JAX-compatible DimArray with pytree registration.

DimArray wraps JAX arrays with unit tracking that is preserved through
JIT compilation, vmap, and grad transformations.
"""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from ..core.dimensions import Dimension
from ..core.units import Unit, dimensionless
from ..errors import DimensionError, UnitConversionError


class DimArray:
    """A JAX array with attached physical units.

    DimArray wraps a JAX array and tracks its physical dimensions through
    all arithmetic operations. Operations between incompatible dimensions
    raise DimensionError immediately.

    Registered as a JAX pytree node, enabling use with jax.jit, jax.vmap,
    and jax.grad.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dimtensor.jax import DimArray
        >>> from dimtensor import units
        >>>
        >>> @jax.jit
        ... def kinetic_energy(mass, velocity):
        ...     return 0.5 * mass * velocity**2
        >>>
        >>> m = DimArray(jnp.array([1.0]), units.kg)
        >>> v = DimArray(jnp.array([10.0]), units.m / units.s)
        >>> E = kinetic_energy(m, v)
        >>> print(E)  # 50.0 J
    """

    __slots__ = ("_data", "_unit")

    def __init__(
        self,
        data: Array | Sequence[float] | float,
        unit: Unit | None = None,
    ) -> None:
        """Create a JAX DimArray.

        Args:
            data: Array data (jax.Array, list, or scalar).
            unit: Physical unit. If None, assumes dimensionless.
        """
        if isinstance(data, DimArray):
            arr = data._data
            unit = unit if unit is not None else data._unit
        elif isinstance(data, Array):
            arr = data
        else:
            arr = jnp.array(data)

        self._data: Array = arr
        self._unit: Unit = unit if unit is not None else dimensionless

    @classmethod
    def _from_data_and_unit(cls, data: Array, unit: Unit) -> DimArray:
        """Internal constructor that doesn't copy data."""
        result = object.__new__(cls)
        result._data = data
        result._unit = unit
        return result

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def data(self) -> Array:
        """The underlying JAX array."""
        return self._data

    @property
    def unit(self) -> Unit:
        """The physical unit of this array."""
        return self._unit

    @property
    def dimension(self) -> Dimension:
        """The physical dimension of this array."""
        return self._unit.dimension

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying array."""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._data.ndim

    @property
    def size(self) -> int:
        """Total number of elements."""
        return self._data.size

    @property
    def dtype(self) -> Any:
        """Data type of the underlying array."""
        return self._data.dtype

    @property
    def is_dimensionless(self) -> bool:
        """Check if this array is dimensionless."""
        return self._unit.dimension.is_dimensionless

    # =========================================================================
    # Unit conversion
    # =========================================================================

    def to(self, unit: Unit) -> DimArray:
        """Convert to a different unit with the same dimension.

        Args:
            unit: Target unit (must have same dimension).

        Returns:
            New DimArray with converted values.

        Raises:
            UnitConversionError: If dimensions don't match.
        """
        if not self._unit.is_compatible(unit):
            raise UnitConversionError.incompatible(self._unit.symbol, unit.symbol)

        factor = self._unit.conversion_factor(unit)
        new_data = self._data * factor
        return DimArray._from_data_and_unit(new_data, unit)

    def magnitude(self) -> Array:
        """Return the numerical magnitude (stripping units).

        Use with caution - this loses dimensional safety.
        """
        return self._data

    # =========================================================================
    # Arithmetic operations
    # =========================================================================

    def __add__(self, other: DimArray | Array | float) -> DimArray:
        """Add arrays (must have same dimension)."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "add"
                )
            other_converted = other.to(self._unit)
            new_data = self._data + other_converted._data
            return DimArray._from_data_and_unit(new_data, self._unit)
        else:
            if not self.is_dimensionless:
                raise DimensionError(
                    f"Cannot add dimensionless number to quantity with dimension {self.dimension}"
                )
            other_arr = other if isinstance(other, Array) else jnp.array(other)
            new_data = self._data + other_arr
            return DimArray._from_data_and_unit(new_data, self._unit)

    def __radd__(self, other: Array | float) -> DimArray:
        """Right add."""
        return self.__add__(other)

    def __sub__(self, other: DimArray | Array | float) -> DimArray:
        """Subtract arrays (must have same dimension)."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "subtract"
                )
            other_converted = other.to(self._unit)
            new_data = self._data - other_converted._data
            return DimArray._from_data_and_unit(new_data, self._unit)
        else:
            if not self.is_dimensionless:
                raise DimensionError(
                    f"Cannot subtract dimensionless number from quantity with dimension {self.dimension}"
                )
            other_arr = other if isinstance(other, Array) else jnp.array(other)
            new_data = self._data - other_arr
            return DimArray._from_data_and_unit(new_data, self._unit)

    def __rsub__(self, other: Array | float) -> DimArray:
        """Right subtract."""
        if not self.is_dimensionless:
            raise DimensionError(
                f"Cannot subtract quantity with dimension {self.dimension} from dimensionless number"
            )
        other_arr = other if isinstance(other, Array) else jnp.array(other)
        new_data = other_arr - self._data
        return DimArray._from_data_and_unit(new_data, self._unit)

    def __mul__(self, other: DimArray | Array | float) -> DimArray:
        """Multiply arrays (dimensions multiply)."""
        if isinstance(other, DimArray):
            new_unit = self._unit * other._unit
            new_data = self._data * other._data
            return DimArray._from_data_and_unit(new_data, new_unit)
        else:
            other_arr = other if isinstance(other, Array) else jnp.array(other)
            new_data = self._data * other_arr
            return DimArray._from_data_and_unit(new_data, self._unit)

    def __rmul__(self, other: Array | float) -> DimArray:
        """Right multiply."""
        return self.__mul__(other)

    def __truediv__(self, other: DimArray | Array | float) -> DimArray:
        """Divide arrays (dimensions divide)."""
        if isinstance(other, DimArray):
            new_unit = self._unit / other._unit
            new_data = self._data / other._data
            return DimArray._from_data_and_unit(new_data, new_unit)
        else:
            other_arr = other if isinstance(other, Array) else jnp.array(other)
            new_data = self._data / other_arr
            return DimArray._from_data_and_unit(new_data, self._unit)

    def __rtruediv__(self, other: Array | float) -> DimArray:
        """Right divide."""
        new_unit = Unit(
            f"1/{self._unit.symbol}",
            self._unit.dimension ** -1,
            1.0 / self._unit.scale,
        )
        other_arr = other if isinstance(other, Array) else jnp.array(other)
        new_data = other_arr / self._data
        return DimArray._from_data_and_unit(new_data, new_unit)

    def __pow__(self, power: int | float) -> DimArray:
        """Raise to a power."""
        new_unit = self._unit ** power
        new_data = self._data ** power
        return DimArray._from_data_and_unit(new_data, new_unit)

    def __neg__(self) -> DimArray:
        """Negate values."""
        return DimArray._from_data_and_unit(-self._data, self._unit)

    def __pos__(self) -> DimArray:
        """Unary positive."""
        return DimArray._from_data_and_unit(+self._data, self._unit)

    def __abs__(self) -> DimArray:
        """Absolute value."""
        return DimArray._from_data_and_unit(jnp.abs(self._data), self._unit)

    def sqrt(self) -> DimArray:
        """Square root (dimension exponents halve)."""
        return self ** 0.5

    # =========================================================================
    # Comparison operations
    # =========================================================================

    def __eq__(self, other: object) -> Array | bool:  # type: ignore[override]
        """Element-wise equality."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                return False
            other_converted = other.to(self._unit)
            return self._data == other_converted._data
        elif self.is_dimensionless:
            other_arr = other if isinstance(other, Array) else jnp.array(other)
            return self._data == other_arr
        return False

    def __lt__(self, other: DimArray | Array | float) -> Array:
        """Element-wise less than."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to(self._unit)
            return self._data < other_converted._data
        elif self.is_dimensionless:
            other_arr = other if isinstance(other, Array) else jnp.array(other)
            return self._data < other_arr
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    def __le__(self, other: DimArray | Array | float) -> Array:
        """Element-wise less than or equal."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to(self._unit)
            return self._data <= other_converted._data
        elif self.is_dimensionless:
            other_arr = other if isinstance(other, Array) else jnp.array(other)
            return self._data <= other_arr
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    def __gt__(self, other: DimArray | Array | float) -> Array:
        """Element-wise greater than."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to(self._unit)
            return self._data > other_converted._data
        elif self.is_dimensionless:
            other_arr = other if isinstance(other, Array) else jnp.array(other)
            return self._data > other_arr
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    def __ge__(self, other: DimArray | Array | float) -> Array:
        """Element-wise greater than or equal."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to(self._unit)
            return self._data >= other_converted._data
        elif self.is_dimensionless:
            other_arr = other if isinstance(other, Array) else jnp.array(other)
            return self._data >= other_arr
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    # =========================================================================
    # Indexing
    # =========================================================================

    def __getitem__(self, key: Any) -> DimArray:
        """Index into the array, preserving units."""
        result = self._data[key]
        return DimArray._from_data_and_unit(result, self._unit)

    def __len__(self) -> int:
        """Length of first dimension."""
        return len(self._data)

    # =========================================================================
    # Reduction operations
    # =========================================================================

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> DimArray:
        """Sum of array elements."""
        result = jnp.sum(self._data, axis=axis, keepdims=keepdims)
        return DimArray._from_data_and_unit(result, self._unit)

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> DimArray:
        """Mean of array elements."""
        result = jnp.mean(self._data, axis=axis, keepdims=keepdims)
        return DimArray._from_data_and_unit(result, self._unit)

    def std(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> DimArray:
        """Standard deviation of array elements."""
        result = jnp.std(self._data, axis=axis, keepdims=keepdims)
        return DimArray._from_data_and_unit(result, self._unit)

    def var(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> DimArray:
        """Variance of array elements (squared units)."""
        result = jnp.var(self._data, axis=axis, keepdims=keepdims)
        new_unit = self._unit ** 2
        return DimArray._from_data_and_unit(result, new_unit)

    def min(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> DimArray:
        """Minimum value."""
        result = jnp.min(self._data, axis=axis, keepdims=keepdims)
        return DimArray._from_data_and_unit(result, self._unit)

    def max(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> DimArray:
        """Maximum value."""
        result = jnp.max(self._data, axis=axis, keepdims=keepdims)
        return DimArray._from_data_and_unit(result, self._unit)

    # =========================================================================
    # Reshaping operations
    # =========================================================================

    def reshape(self, shape: tuple[int, ...] | list[int]) -> DimArray:
        """Reshape array preserving units."""
        new_data = self._data.reshape(shape)
        return DimArray._from_data_and_unit(new_data, self._unit)

    def transpose(self, axes: tuple[int, ...] | list[int] | None = None) -> DimArray:
        """Transpose array."""
        new_data = jnp.transpose(self._data, axes)
        return DimArray._from_data_and_unit(new_data, self._unit)

    def flatten(self) -> DimArray:
        """Flatten to 1D array."""
        new_data = self._data.flatten()
        return DimArray._from_data_and_unit(new_data, self._unit)

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> DimArray:
        """Remove size-1 dimensions."""
        new_data = jnp.squeeze(self._data, axis=axis)
        return DimArray._from_data_and_unit(new_data, self._unit)

    def expand_dims(self, axis: int | tuple[int, ...]) -> DimArray:
        """Add size-1 dimension."""
        new_data = jnp.expand_dims(self._data, axis=axis)
        return DimArray._from_data_and_unit(new_data, self._unit)

    # =========================================================================
    # Linear algebra
    # =========================================================================

    def dot(self, other: DimArray) -> DimArray:
        """Dot product (dimensions multiply)."""
        new_unit = self._unit * other._unit
        new_data = jnp.dot(self._data, other._data)
        return DimArray._from_data_and_unit(new_data, new_unit)

    def matmul(self, other: DimArray) -> DimArray:
        """Matrix multiplication (dimensions multiply)."""
        new_unit = self._unit * other._unit
        new_data = jnp.matmul(self._data, other._data)
        return DimArray._from_data_and_unit(new_data, new_unit)

    def __matmul__(self, other: DimArray) -> DimArray:
        """Matrix multiplication operator @."""
        return self.matmul(other)

    def norm(
        self,
        ord: int | float | str | None = None,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> DimArray:
        """Vector or matrix norm (preserves units)."""
        result = jnp.linalg.norm(self._data, ord=ord, axis=axis, keepdims=keepdims)
        return DimArray._from_data_and_unit(result, self._unit)

    # =========================================================================
    # String representations
    # =========================================================================

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"DimArray({self._data!r}, unit={self._unit.symbol!r})"

    def __str__(self) -> str:
        """Human-readable string."""
        if self.is_dimensionless:
            return str(self._data)
        simplified = self._unit.simplified()
        return f"{self._data} {simplified.symbol}"

    # =========================================================================
    # Conversion
    # =========================================================================

    def numpy(self) -> Any:
        """Convert to numpy array (loses unit information)."""
        import numpy as np
        return np.asarray(self._data)


# =============================================================================
# JAX Pytree Registration
# =============================================================================

def _dimarray_flatten(arr: DimArray) -> tuple[tuple[Array], tuple[Unit]]:
    """Flatten DimArray for JAX pytree."""
    # Children are the traceable arrays, aux_data is the static metadata
    return (arr._data,), (arr._unit,)


def _dimarray_unflatten(aux_data: tuple[Unit], children: tuple[Array]) -> DimArray:
    """Unflatten DimArray from JAX pytree."""
    (data,) = children
    (unit,) = aux_data
    return DimArray._from_data_and_unit(data, unit)


def register_pytree() -> None:
    """Register DimArray as a JAX pytree node.

    This is called automatically when importing dimtensor.jax.
    """
    jax.tree_util.register_pytree_node(
        DimArray,
        _dimarray_flatten,
        _dimarray_unflatten,
    )

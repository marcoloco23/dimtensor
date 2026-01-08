"""DimArray: numpy arrays with dimensional awareness.

DimArray wraps numpy arrays and tracks physical units through all operations,
catching dimensional errors at operation time rather than after hours of computation.
"""

from __future__ import annotations

from typing import Any, overload

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from ..errors import DimensionError, UnitConversionError
from .dimensions import Dimension
from .units import Unit, dimensionless


class DimArray:
    """A numpy array with attached physical units.

    DimArray wraps a numpy array and tracks its physical dimensions through
    all arithmetic operations. Operations between incompatible dimensions
    raise DimensionError immediately, catching physics errors early.

    Examples:
        >>> from dimtensor import DimArray, units
        >>> v = DimArray([1.0, 2.0, 3.0], units.m / units.s)
        >>> t = DimArray([0.5, 1.0, 1.5], units.s)
        >>> d = v * t  # distance in meters
        >>> print(d)
        [0.5 2.0 4.5] m

        >>> a = DimArray([9.8], units.m / units.s**2)
        >>> v + a  # raises DimensionError
    """

    __slots__ = ("_data", "_unit")

    def __init__(
        self,
        data: ArrayLike,
        unit: Unit | None = None,
        dtype: DTypeLike = None,
        copy: bool = False,
    ) -> None:
        """Create a DimArray.

        Args:
            data: Array-like data (list, tuple, numpy array, or scalar).
            unit: Physical unit of the data. If None, assumes dimensionless.
            dtype: Numpy dtype for the underlying array.
            copy: If True, always copy the data.
        """
        # Convert to numpy array
        if isinstance(data, DimArray):
            arr = np.array(data._data, dtype=dtype, copy=copy)
            unit = unit if unit is not None else data._unit
        else:
            arr = np.array(data, dtype=dtype, copy=copy)

        self._data: NDArray[Any] = arr
        self._unit: Unit = unit if unit is not None else dimensionless

    @classmethod
    def _from_data_and_unit(cls, data: NDArray[Any], unit: Unit) -> DimArray:
        """Internal constructor that doesn't copy data."""
        result = object.__new__(cls)
        result._data = data
        result._unit = unit
        return result

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def data(self) -> NDArray[Any]:
        """The underlying numpy array (read-only view)."""
        result = self._data.view()
        result.flags.writeable = False
        return result

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
    def dtype(self) -> np.dtype[Any]:
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
            New DimArray with converted values and new unit.

        Raises:
            UnitConversionError: If dimensions don't match.
        """
        if not self._unit.is_compatible(unit):
            raise UnitConversionError.incompatible(self._unit.symbol, unit.symbol)

        factor = self._unit.conversion_factor(unit)
        new_data = self._data * factor
        return DimArray._from_data_and_unit(new_data, unit)

    def to_base_units(self) -> DimArray:
        """Convert to SI base units.

        Returns a DimArray with scale factor 1.0 (pure SI units).
        """
        # Create a unit with the same dimension but scale 1.0
        base_unit = Unit(str(self._unit.dimension), self._unit.dimension, 1.0)
        new_data = self._data * self._unit.scale
        return DimArray._from_data_and_unit(new_data, base_unit)

    def magnitude(self) -> NDArray[Any]:
        """Return the numerical magnitude (stripping units).

        Use with caution - this loses dimensional safety.
        """
        return self._data.copy()

    # =========================================================================
    # Arithmetic operations
    # =========================================================================

    def __add__(self, other: DimArray | ArrayLike) -> DimArray:
        """Add two DimArrays (must have same dimension)."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "add"
                )
            # Convert other to same unit for consistent result
            other_converted = other.to(self._unit)
            new_data = self._data + other_converted._data
            return DimArray._from_data_and_unit(new_data, self._unit)
        else:
            # Adding a raw number - only valid if dimensionless
            if not self.is_dimensionless:
                raise DimensionError(
                    f"Cannot add dimensionless number to quantity with dimension {self.dimension}"
                )
            new_data = self._data + np.asarray(other)
            return DimArray._from_data_and_unit(new_data, self._unit)

    def __radd__(self, other: ArrayLike) -> DimArray:
        """Right add (for scalar + DimArray)."""
        return self.__add__(other)

    def __sub__(self, other: DimArray | ArrayLike) -> DimArray:
        """Subtract two DimArrays (must have same dimension)."""
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
            new_data = self._data - np.asarray(other)
            return DimArray._from_data_and_unit(new_data, self._unit)

    def __rsub__(self, other: ArrayLike) -> DimArray:
        """Right subtract (for scalar - DimArray)."""
        if not self.is_dimensionless:
            raise DimensionError(
                f"Cannot subtract quantity with dimension {self.dimension} from dimensionless number"
            )
        new_data = np.asarray(other) - self._data
        return DimArray._from_data_and_unit(new_data, self._unit)

    def __mul__(self, other: DimArray | ArrayLike) -> DimArray:
        """Multiply DimArrays (dimensions multiply)."""
        if isinstance(other, DimArray):
            new_unit = self._unit * other._unit
            new_data = self._data * other._data
            return DimArray._from_data_and_unit(new_data, new_unit)
        else:
            # Scalar multiplication preserves units
            new_data = self._data * np.asarray(other)
            return DimArray._from_data_and_unit(new_data, self._unit)

    def __rmul__(self, other: ArrayLike) -> DimArray:
        """Right multiply (for scalar * DimArray)."""
        return self.__mul__(other)

    def __truediv__(self, other: DimArray | ArrayLike) -> DimArray:
        """Divide DimArrays (dimensions divide)."""
        if isinstance(other, DimArray):
            new_unit = self._unit / other._unit
            new_data = self._data / other._data
            return DimArray._from_data_and_unit(new_data, new_unit)
        else:
            # Scalar division preserves units
            new_data = self._data / np.asarray(other)
            return DimArray._from_data_and_unit(new_data, self._unit)

    def __rtruediv__(self, other: ArrayLike) -> DimArray:
        """Right divide (for scalar / DimArray)."""
        new_unit = Unit(
            f"1/{self._unit.symbol}",
            self._unit.dimension ** -1,
            1.0 / self._unit.scale,
        )
        new_data = np.asarray(other) / self._data
        return DimArray._from_data_and_unit(new_data, new_unit)

    def __pow__(self, power: int | float) -> DimArray:
        """Raise DimArray to a power."""
        new_unit = self._unit ** power
        new_data = self._data ** power
        return DimArray._from_data_and_unit(new_data, new_unit)

    def __neg__(self) -> DimArray:
        """Negate values."""
        return DimArray._from_data_and_unit(-self._data, self._unit)

    def __pos__(self) -> DimArray:
        """Unary positive (returns copy)."""
        return DimArray._from_data_and_unit(+self._data, self._unit)

    def __abs__(self) -> DimArray:
        """Absolute value."""
        return DimArray._from_data_and_unit(np.abs(self._data), self._unit)

    # =========================================================================
    # Comparison operations
    # =========================================================================

    def __eq__(self, other: object) -> NDArray[np.bool_] | bool:  # type: ignore[override]
        """Element-wise equality (requires same dimension)."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                return False
            other_converted = other.to(self._unit)
            return self._data == other_converted._data  # type: ignore[return-value]
        elif self.is_dimensionless:
            return self._data == np.asarray(other)  # type: ignore[return-value]
        return False

    def __ne__(self, other: object) -> NDArray[np.bool_] | bool:  # type: ignore[override]
        """Element-wise inequality."""
        result = self.__eq__(other)
        if isinstance(result, np.ndarray):
            return ~result
        return not result

    def __lt__(self, other: DimArray | ArrayLike) -> NDArray[np.bool_]:
        """Element-wise less than."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to(self._unit)
            return self._data < other_converted._data  # type: ignore[return-value]
        elif self.is_dimensionless:
            return self._data < np.asarray(other)  # type: ignore[return-value]
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    def __le__(self, other: DimArray | ArrayLike) -> NDArray[np.bool_]:
        """Element-wise less than or equal."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to(self._unit)
            return self._data <= other_converted._data  # type: ignore[return-value]
        elif self.is_dimensionless:
            return self._data <= np.asarray(other)  # type: ignore[return-value]
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    def __gt__(self, other: DimArray | ArrayLike) -> NDArray[np.bool_]:
        """Element-wise greater than."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to(self._unit)
            return self._data > other_converted._data  # type: ignore[return-value]
        elif self.is_dimensionless:
            return self._data > np.asarray(other)  # type: ignore[return-value]
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    def __ge__(self, other: DimArray | ArrayLike) -> NDArray[np.bool_]:
        """Element-wise greater than or equal."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to(self._unit)
            return self._data >= other_converted._data  # type: ignore[return-value]
        elif self.is_dimensionless:
            return self._data >= np.asarray(other)  # type: ignore[return-value]
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    # =========================================================================
    # Indexing
    # =========================================================================

    @overload
    def __getitem__(self, key: int) -> DimArray: ...

    @overload
    def __getitem__(self, key: slice | NDArray[Any] | tuple[Any, ...]) -> DimArray: ...

    def __getitem__(
        self, key: int | slice | NDArray[Any] | tuple[Any, ...]
    ) -> DimArray:
        """Index into the array, preserving units."""
        result = self._data[key]
        # Always return DimArray to maintain dimensional safety
        if np.isscalar(result):
            result = np.array([result])
        return DimArray._from_data_and_unit(result, self._unit)

    def __len__(self) -> int:
        """Length of first dimension."""
        return len(self._data)

    def __iter__(self):
        """Iterate over first dimension."""
        for i in range(len(self)):
            yield self[i]

    # =========================================================================
    # Reduction operations
    # =========================================================================

    def sum(self, axis: int | None = None, keepdims: bool = False) -> DimArray:
        """Sum of array elements."""
        result = self._data.sum(axis=axis, keepdims=keepdims)
        if np.isscalar(result):
            result = np.array([result])
        return DimArray._from_data_and_unit(result, self._unit)

    def mean(self, axis: int | None = None, keepdims: bool = False) -> DimArray:
        """Mean of array elements."""
        result = self._data.mean(axis=axis, keepdims=keepdims)
        if np.isscalar(result):
            result = np.array([result])
        return DimArray._from_data_and_unit(result, self._unit)

    def std(self, axis: int | None = None, keepdims: bool = False) -> DimArray:
        """Standard deviation of array elements."""
        result = self._data.std(axis=axis, keepdims=keepdims)
        if np.isscalar(result):
            result = np.array([result])
        return DimArray._from_data_and_unit(result, self._unit)

    def min(self, axis: int | None = None, keepdims: bool = False) -> DimArray:
        """Minimum value."""
        result = self._data.min(axis=axis, keepdims=keepdims)
        if np.isscalar(result):
            result = np.array([result])
        return DimArray._from_data_and_unit(result, self._unit)

    def max(self, axis: int | None = None, keepdims: bool = False) -> DimArray:
        """Maximum value."""
        result = self._data.max(axis=axis, keepdims=keepdims)
        if np.isscalar(result):
            result = np.array([result])
        return DimArray._from_data_and_unit(result, self._unit)

    # =========================================================================
    # String representations
    # =========================================================================

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"DimArray({self._data!r}, unit={self._unit.symbol!r})"

    def __str__(self) -> str:
        """Human-readable string."""
        # Format the array nicely
        if self._data.ndim == 0:
            value_str = str(self._data.item())
        elif self._data.size <= 10:
            value_str = np.array2string(
                self._data, separator=", ", precision=4, suppress_small=True
            )
        else:
            value_str = np.array2string(
                self._data,
                separator=", ",
                precision=4,
                suppress_small=True,
                threshold=6,
            )

        if self.is_dimensionless:
            return value_str
        return f"{value_str} {self._unit.symbol}"

    # =========================================================================
    # Numpy array protocol
    # =========================================================================

    def __array__(self, dtype: DTypeLike = None) -> NDArray[Any]:
        """Convert to numpy array (loses unit information)."""
        if dtype is None:
            return self._data.copy()
        return self._data.astype(dtype)

    @property
    def __array_priority__(self) -> float:
        """Ensure DimArray operations take precedence."""
        return 10.0

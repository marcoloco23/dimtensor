"""DaskDimArray: Dask arrays with dimensional awareness.

DaskDimArray wraps dask arrays and tracks physical units through all operations,
enabling lazy evaluation and parallel computation with dimensional safety.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import dask
import dask.array as da
import dask.threaded
import numpy as np
from dask.base import DaskMethodsMixin, tokenize
from dask.highlevelgraph import HighLevelGraph

from ..core.dimensions import Dimension
from ..core.units import Unit, dimensionless
from ..errors import DimensionError, UnitConversionError


class DaskDimArray(DaskMethodsMixin):
    """A Dask array with attached physical units.

    DaskDimArray wraps a dask.array.Array and tracks its physical dimensions
    through all arithmetic operations. Operations between incompatible
    dimensions raise DimensionError immediately (before computation).

    Supports lazy evaluation, chunked computation, and parallel execution
    while maintaining dimensional safety.

    Examples:
        >>> import numpy as np
        >>> from dimtensor.dask import DaskDimArray
        >>> from dimtensor import units
        >>>
        >>> # Create from numpy array with chunks
        >>> data = np.random.randn(1000, 1000)
        >>> velocity = DaskDimArray(data, unit=units.m/units.s, chunks=(100, 100))
        >>>
        >>> # Lazy operations
        >>> speed = (velocity**2).sum(axis=1).sqrt()
        >>>
        >>> # Compute result
        >>> result = speed.compute()  # Returns numpy DimArray
    """

    __slots__ = ("_data", "_unit")

    def __init__(
        self,
        data: da.Array | np.ndarray | Sequence[Any] | float,
        unit: Unit | None = None,
        chunks: str | int | tuple[int, ...] | dict[int, int] = "auto",
    ) -> None:
        """Create a DaskDimArray.

        Args:
            data: Array data (dask.array.Array, numpy array, list, or scalar).
            unit: Physical unit. If None, assumes dimensionless.
            chunks: Chunk specification for dask (ignored if data is already
                a dask array). Can be "auto", an integer, tuple, or dict.
        """
        if isinstance(data, DaskDimArray):
            arr = data._data
            unit = unit if unit is not None else data._unit
        elif isinstance(data, da.Array):
            arr = data
        elif isinstance(data, np.ndarray):
            arr = da.from_array(data, chunks=chunks)
        else:
            # Convert to numpy first, then to dask
            np_arr = np.asarray(data)
            arr = da.from_array(np_arr, chunks=chunks)

        self._data: da.Array = arr
        self._unit: Unit = unit if unit is not None else dimensionless

    @classmethod
    def _from_data_and_unit(cls, data: da.Array, unit: Unit) -> DaskDimArray:
        """Internal constructor that doesn't copy data."""
        result = object.__new__(cls)
        result._data = data
        result._unit = unit
        return result

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def data(self) -> da.Array:
        """The underlying dask array."""
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
    def dtype(self) -> np.dtype[Any]:
        """Data type of the underlying array."""
        return self._data.dtype

    @property
    def chunks(self) -> tuple[tuple[int, ...], ...]:
        """Chunk sizes for each dimension."""
        return self._data.chunks

    @property
    def chunksize(self) -> tuple[int, ...]:
        """Uniform chunk size (one value per dimension)."""
        return self._data.chunksize

    @property
    def is_dimensionless(self) -> bool:
        """Check if this array is dimensionless."""
        return self._unit.dimension.is_dimensionless

    # =========================================================================
    # Dask Collection Protocol
    # =========================================================================

    def __dask_graph__(self) -> HighLevelGraph:
        """Return the task graph."""
        return self._data.__dask_graph__()

    def __dask_keys__(self) -> list[Any]:
        """Return the output keys."""
        return self._data.__dask_keys__()

    __dask_scheduler__ = staticmethod(dask.threaded.get)

    def __dask_postcompute__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
        """Return finalization function and arguments."""
        # Import here to avoid circular imports
        from ..core.dimarray import DimArray as NumpyDimArray

        def finalize(result: np.ndarray, unit: Unit) -> NumpyDimArray:
            return NumpyDimArray._from_data_and_unit(result, unit)

        return finalize, (self._unit,)

    def __dask_postpersist__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
        """Return rebuild function and arguments."""
        def rebuild(data: da.Array, unit: Unit) -> DaskDimArray:
            return DaskDimArray._from_data_and_unit(data, unit)

        return rebuild, (self._unit,)

    def __dask_tokenize__(self) -> tuple[Any, ...]:
        """Return unique hash input for caching."""
        return (
            type(self).__name__,
            tokenize(self._data),
            self._unit.symbol,
            str(self._unit.dimension),
        )

    # =========================================================================
    # Compute/Persist Methods
    # =========================================================================

    def compute(self, **kwargs: Any) -> Any:
        """Compute and return a numpy DimArray.

        Args:
            **kwargs: Passed to dask.compute().

        Returns:
            A numpy DimArray with the computed result.
        """
        from ..core.dimarray import DimArray as NumpyDimArray

        result = self._data.compute(**kwargs)
        return NumpyDimArray._from_data_and_unit(result, self._unit)

    def persist(self, **kwargs: Any) -> DaskDimArray:
        """Persist to distributed memory.

        Args:
            **kwargs: Passed to dask.persist().

        Returns:
            DaskDimArray with data persisted in memory.
        """
        persisted_data = self._data.persist(**kwargs)
        return DaskDimArray._from_data_and_unit(persisted_data, self._unit)

    def visualize(self, **kwargs: Any) -> Any:
        """Visualize the task graph.

        Args:
            **kwargs: Passed to dask.visualize().

        Returns:
            Graph visualization (format depends on kwargs).
        """
        return self._data.visualize(**kwargs)

    # =========================================================================
    # Array Creation Class Methods
    # =========================================================================

    @classmethod
    def from_array(
        cls,
        data: np.ndarray | Sequence[Any],
        unit: Unit,
        chunks: str | int | tuple[int, ...] = "auto",
    ) -> DaskDimArray:
        """Create DaskDimArray from array-like data.

        Args:
            data: Array-like data.
            unit: Physical unit.
            chunks: Chunk specification.

        Returns:
            New DaskDimArray.
        """
        return cls(data, unit=unit, chunks=chunks)

    @classmethod
    def from_delayed(
        cls,
        value: Any,
        shape: tuple[int, ...],
        dtype: np.dtype[Any] | type,
        unit: Unit,
    ) -> DaskDimArray:
        """Create DaskDimArray from a dask.delayed object.

        Args:
            value: A dask.delayed object that will produce the array.
            shape: Shape of the resulting array.
            dtype: Data type of the array.
            unit: Physical unit.

        Returns:
            New DaskDimArray with lazy evaluation.
        """
        arr = da.from_delayed(value, shape=shape, dtype=dtype)
        return cls._from_data_and_unit(arr, unit)

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, ...] | int,
        unit: Unit,
        chunks: str | int | tuple[int, ...] = "auto",
        dtype: type = float,
    ) -> DaskDimArray:
        """Create array of zeros.

        Args:
            shape: Array shape.
            unit: Physical unit.
            chunks: Chunk specification.
            dtype: Data type.

        Returns:
            DaskDimArray filled with zeros.
        """
        arr = da.zeros(shape, dtype=dtype, chunks=chunks)
        return cls._from_data_and_unit(arr, unit)

    @classmethod
    def ones(
        cls,
        shape: tuple[int, ...] | int,
        unit: Unit,
        chunks: str | int | tuple[int, ...] = "auto",
        dtype: type = float,
    ) -> DaskDimArray:
        """Create array of ones.

        Args:
            shape: Array shape.
            unit: Physical unit.
            chunks: Chunk specification.
            dtype: Data type.

        Returns:
            DaskDimArray filled with ones.
        """
        arr = da.ones(shape, dtype=dtype, chunks=chunks)
        return cls._from_data_and_unit(arr, unit)

    @classmethod
    def arange(
        cls,
        start: float,
        stop: float | None = None,
        step: float = 1,
        unit: Unit | None = None,
        chunks: str | int | tuple[int, ...] = "auto",
    ) -> DaskDimArray:
        """Create evenly spaced values.

        Args:
            start: Start value (or stop if stop is None).
            stop: Stop value.
            step: Step size.
            unit: Physical unit.
            chunks: Chunk specification.

        Returns:
            DaskDimArray with evenly spaced values.
        """
        arr = da.arange(start, stop, step, chunks=chunks)
        return cls._from_data_and_unit(arr, unit or dimensionless)

    # =========================================================================
    # Unit Conversion
    # =========================================================================

    def to(self, unit: Unit) -> DaskDimArray:
        """Convert to a different unit with the same dimension.

        Args:
            unit: Target unit (must have same dimension).

        Returns:
            New DaskDimArray with converted values.

        Raises:
            UnitConversionError: If dimensions don't match.
        """
        if not self._unit.is_compatible(unit):
            raise UnitConversionError.incompatible(self._unit.symbol, unit.symbol)

        factor = self._unit.conversion_factor(unit)
        new_data = self._data * factor
        return DaskDimArray._from_data_and_unit(new_data, unit)

    def to_base_units(self) -> DaskDimArray:
        """Convert to SI base units.

        Returns:
            DaskDimArray with scale factor 1.0 (pure SI units).
        """
        base_unit = Unit(str(self._unit.dimension), self._unit.dimension, 1.0)
        new_data = self._data * self._unit.scale
        return DaskDimArray._from_data_and_unit(new_data, base_unit)

    def magnitude(self) -> da.Array:
        """Return the numerical magnitude (stripping units).

        Use with caution - this loses dimensional safety.
        """
        return self._data

    # =========================================================================
    # Arithmetic Operations
    # =========================================================================

    def __add__(self, other: DaskDimArray | da.Array | np.ndarray | float) -> DaskDimArray:
        """Add arrays (must have same dimension)."""
        if isinstance(other, DaskDimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "add"
                )
            other_converted = other.to(self._unit)
            new_data = self._data + other_converted._data
            return DaskDimArray._from_data_and_unit(new_data, self._unit)
        else:
            if not self.is_dimensionless:
                raise DimensionError(
                    f"Cannot add dimensionless number to quantity with dimension {self.dimension}"
                )
            other_arr = da.asarray(other) if not isinstance(other, da.Array) else other
            new_data = self._data + other_arr
            return DaskDimArray._from_data_and_unit(new_data, self._unit)

    def __radd__(self, other: da.Array | np.ndarray | float) -> DaskDimArray:
        """Right add."""
        return self.__add__(other)

    def __sub__(self, other: DaskDimArray | da.Array | np.ndarray | float) -> DaskDimArray:
        """Subtract arrays (must have same dimension)."""
        if isinstance(other, DaskDimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "subtract"
                )
            other_converted = other.to(self._unit)
            new_data = self._data - other_converted._data
            return DaskDimArray._from_data_and_unit(new_data, self._unit)
        else:
            if not self.is_dimensionless:
                raise DimensionError(
                    f"Cannot subtract dimensionless number from quantity with dimension {self.dimension}"
                )
            other_arr = da.asarray(other) if not isinstance(other, da.Array) else other
            new_data = self._data - other_arr
            return DaskDimArray._from_data_and_unit(new_data, self._unit)

    def __rsub__(self, other: da.Array | np.ndarray | float) -> DaskDimArray:
        """Right subtract."""
        if not self.is_dimensionless:
            raise DimensionError(
                f"Cannot subtract quantity with dimension {self.dimension} from dimensionless number"
            )
        other_arr = da.asarray(other) if not isinstance(other, da.Array) else other
        new_data = other_arr - self._data
        return DaskDimArray._from_data_and_unit(new_data, self._unit)

    def __mul__(self, other: DaskDimArray | da.Array | np.ndarray | float) -> DaskDimArray:
        """Multiply arrays (dimensions multiply)."""
        if isinstance(other, DaskDimArray):
            new_unit = self._unit * other._unit
            new_data = self._data * other._data
            return DaskDimArray._from_data_and_unit(new_data, new_unit)
        else:
            other_arr = da.asarray(other) if not isinstance(other, da.Array) else other
            new_data = self._data * other_arr
            return DaskDimArray._from_data_and_unit(new_data, self._unit)

    def __rmul__(self, other: da.Array | np.ndarray | float) -> DaskDimArray:
        """Right multiply."""
        return self.__mul__(other)

    def __truediv__(self, other: DaskDimArray | da.Array | np.ndarray | float) -> DaskDimArray:
        """Divide arrays (dimensions divide)."""
        if isinstance(other, DaskDimArray):
            new_unit = self._unit / other._unit
            new_data = self._data / other._data
            return DaskDimArray._from_data_and_unit(new_data, new_unit)
        else:
            other_arr = da.asarray(other) if not isinstance(other, da.Array) else other
            new_data = self._data / other_arr
            return DaskDimArray._from_data_and_unit(new_data, self._unit)

    def __rtruediv__(self, other: da.Array | np.ndarray | float) -> DaskDimArray:
        """Right divide."""
        new_unit = Unit(
            f"1/{self._unit.symbol}",
            self._unit.dimension ** -1,
            1.0 / self._unit.scale,
        )
        other_arr = da.asarray(other) if not isinstance(other, da.Array) else other
        new_data = other_arr / self._data
        return DaskDimArray._from_data_and_unit(new_data, new_unit)

    def __pow__(self, power: int | float) -> DaskDimArray:
        """Raise to a power."""
        new_unit = self._unit ** power
        new_data = self._data ** power
        return DaskDimArray._from_data_and_unit(new_data, new_unit)

    def __neg__(self) -> DaskDimArray:
        """Negate values."""
        return DaskDimArray._from_data_and_unit(-self._data, self._unit)

    def __pos__(self) -> DaskDimArray:
        """Unary positive."""
        return DaskDimArray._from_data_and_unit(+self._data, self._unit)

    def __abs__(self) -> DaskDimArray:
        """Absolute value."""
        return DaskDimArray._from_data_and_unit(da.abs(self._data), self._unit)

    def sqrt(self) -> DaskDimArray:
        """Square root (dimension exponents halve)."""
        return self ** 0.5

    # =========================================================================
    # Comparison Operations
    # =========================================================================

    def __eq__(self, other: object) -> da.Array | bool:  # type: ignore[override]
        """Element-wise equality."""
        if isinstance(other, DaskDimArray):
            if self.dimension != other.dimension:
                return False
            other_converted = other.to(self._unit)
            return self._data == other_converted._data
        elif self.is_dimensionless:
            other_arr = da.asarray(other) if not isinstance(other, da.Array) else other
            return self._data == other_arr
        return False

    def __ne__(self, other: object) -> da.Array | bool:  # type: ignore[override]
        """Element-wise inequality."""
        result = self.__eq__(other)
        if isinstance(result, da.Array):
            return ~result
        return not result

    def __lt__(self, other: DaskDimArray | da.Array | np.ndarray | float) -> da.Array:
        """Element-wise less than."""
        if isinstance(other, DaskDimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to(self._unit)
            return self._data < other_converted._data
        elif self.is_dimensionless:
            other_arr = da.asarray(other) if not isinstance(other, da.Array) else other
            return self._data < other_arr
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    def __le__(self, other: DaskDimArray | da.Array | np.ndarray | float) -> da.Array:
        """Element-wise less than or equal."""
        if isinstance(other, DaskDimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to(self._unit)
            return self._data <= other_converted._data
        elif self.is_dimensionless:
            other_arr = da.asarray(other) if not isinstance(other, da.Array) else other
            return self._data <= other_arr
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    def __gt__(self, other: DaskDimArray | da.Array | np.ndarray | float) -> da.Array:
        """Element-wise greater than."""
        if isinstance(other, DaskDimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to(self._unit)
            return self._data > other_converted._data
        elif self.is_dimensionless:
            other_arr = da.asarray(other) if not isinstance(other, da.Array) else other
            return self._data > other_arr
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    def __ge__(self, other: DaskDimArray | da.Array | np.ndarray | float) -> da.Array:
        """Element-wise greater than or equal."""
        if isinstance(other, DaskDimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to(self._unit)
            return self._data >= other_converted._data
        elif self.is_dimensionless:
            other_arr = da.asarray(other) if not isinstance(other, da.Array) else other
            return self._data >= other_arr
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    # =========================================================================
    # Indexing
    # =========================================================================

    def __getitem__(self, key: Any) -> DaskDimArray:
        """Index into the array, preserving units."""
        result = self._data[key]
        return DaskDimArray._from_data_and_unit(result, self._unit)

    def __len__(self) -> int:
        """Length of first dimension."""
        return len(self._data)

    # =========================================================================
    # Reduction Operations
    # =========================================================================

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> DaskDimArray:
        """Sum of array elements (lazy).

        Args:
            axis: Axis or axes along which to sum.
            keepdims: Keep reduced dimensions as size 1.

        Returns:
            DaskDimArray with sum result.
        """
        result = self._data.sum(axis=axis, keepdims=keepdims)
        return DaskDimArray._from_data_and_unit(result, self._unit)

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> DaskDimArray:
        """Mean of array elements (lazy).

        Args:
            axis: Axis or axes along which to compute mean.
            keepdims: Keep reduced dimensions as size 1.

        Returns:
            DaskDimArray with mean result.
        """
        result = self._data.mean(axis=axis, keepdims=keepdims)
        return DaskDimArray._from_data_and_unit(result, self._unit)

    def std(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> DaskDimArray:
        """Standard deviation of array elements (lazy).

        Args:
            axis: Axis or axes along which to compute std.
            keepdims: Keep reduced dimensions as size 1.

        Returns:
            DaskDimArray with std result (same unit).
        """
        result = self._data.std(axis=axis, keepdims=keepdims)
        return DaskDimArray._from_data_and_unit(result, self._unit)

    def var(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> DaskDimArray:
        """Variance of array elements (lazy).

        Note: Variance has units squared (e.g., m -> m^2).

        Args:
            axis: Axis or axes along which to compute variance.
            keepdims: Keep reduced dimensions as size 1.

        Returns:
            DaskDimArray with variance result (squared unit).
        """
        result = self._data.var(axis=axis, keepdims=keepdims)
        new_unit = self._unit ** 2
        return DaskDimArray._from_data_and_unit(result, new_unit)

    def min(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> DaskDimArray:
        """Minimum value (lazy).

        Args:
            axis: Axis or axes along which to find minimum.
            keepdims: Keep reduced dimensions as size 1.

        Returns:
            DaskDimArray with minimum.
        """
        result = self._data.min(axis=axis, keepdims=keepdims)
        return DaskDimArray._from_data_and_unit(result, self._unit)

    def max(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> DaskDimArray:
        """Maximum value (lazy).

        Args:
            axis: Axis or axes along which to find maximum.
            keepdims: Keep reduced dimensions as size 1.

        Returns:
            DaskDimArray with maximum.
        """
        result = self._data.max(axis=axis, keepdims=keepdims)
        return DaskDimArray._from_data_and_unit(result, self._unit)

    # =========================================================================
    # Reshaping Operations
    # =========================================================================

    def reshape(self, shape: tuple[int, ...] | list[int]) -> DaskDimArray:
        """Reshape array preserving units (lazy).

        Args:
            shape: New shape. One dimension may be -1.

        Returns:
            Reshaped DaskDimArray.
        """
        new_data = self._data.reshape(shape)
        return DaskDimArray._from_data_and_unit(new_data, self._unit)

    def transpose(self, axes: tuple[int, ...] | list[int] | None = None) -> DaskDimArray:
        """Transpose array (lazy).

        Args:
            axes: Permutation of dimensions. If None, reverses dimensions.

        Returns:
            Transposed DaskDimArray.
        """
        new_data = da.transpose(self._data, axes)
        return DaskDimArray._from_data_and_unit(new_data, self._unit)

    def flatten(self) -> DaskDimArray:
        """Flatten to 1D array (lazy).

        Returns:
            Flattened DaskDimArray.
        """
        new_data = self._data.flatten()
        return DaskDimArray._from_data_and_unit(new_data, self._unit)

    def rechunk(
        self,
        chunks: str | int | tuple[int, ...] | dict[int, int] = "auto",
    ) -> DaskDimArray:
        """Change the chunking of the array.

        Args:
            chunks: New chunk specification.

        Returns:
            DaskDimArray with new chunking.
        """
        new_data = self._data.rechunk(chunks)
        return DaskDimArray._from_data_and_unit(new_data, self._unit)

    # =========================================================================
    # Linear Algebra
    # =========================================================================

    def dot(self, other: DaskDimArray) -> DaskDimArray:
        """Dot product (dimensions multiply).

        Args:
            other: Another DaskDimArray.

        Returns:
            Dot product with multiplied dimensions.
        """
        new_unit = self._unit * other._unit
        new_data = da.dot(self._data, other._data)
        return DaskDimArray._from_data_and_unit(new_data, new_unit)

    def matmul(self, other: DaskDimArray) -> DaskDimArray:
        """Matrix multiplication (dimensions multiply).

        Args:
            other: Another DaskDimArray.

        Returns:
            Matrix product with multiplied dimensions.
        """
        new_unit = self._unit * other._unit
        new_data = da.matmul(self._data, other._data)
        return DaskDimArray._from_data_and_unit(new_data, new_unit)

    def __matmul__(self, other: DaskDimArray) -> DaskDimArray:
        """Matrix multiplication operator @."""
        return self.matmul(other)

    def norm(
        self,
        ord: int | float | str | None = None,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> DaskDimArray:
        """Vector or matrix norm (preserves units).

        Args:
            ord: Order of the norm.
            axis: Axis or axes along which to compute norm.
            keepdims: Keep reduced dimensions as size 1.

        Returns:
            DaskDimArray with norm result.
        """
        result = da.linalg.norm(self._data, ord=ord, axis=axis, keepdims=keepdims)
        return DaskDimArray._from_data_and_unit(result, self._unit)

    # =========================================================================
    # Conversion Methods
    # =========================================================================

    def to_dimarray(self) -> Any:
        """Compute and return a numpy DimArray.

        Returns:
            NumPy DimArray with computed values.
        """
        return self.compute()

    @classmethod
    def from_dimarray(
        cls,
        dimarray: Any,
        chunks: str | int | tuple[int, ...] = "auto",
    ) -> DaskDimArray:
        """Create DaskDimArray from a numpy DimArray.

        Args:
            dimarray: A numpy DimArray.
            chunks: Chunk specification.

        Returns:
            New DaskDimArray.
        """
        return cls(dimarray.data, unit=dimarray.unit, chunks=chunks)

    # =========================================================================
    # String Representations
    # =========================================================================

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"DaskDimArray(shape={self.shape}, "
            f"chunks={self.chunksize}, "
            f"dtype={self.dtype}, "
            f"unit={self._unit.symbol!r})"
        )

    def __str__(self) -> str:
        """Human-readable string."""
        simplified = self._unit.simplified()
        return f"DaskDimArray({self.shape}, chunks={self.chunksize}) {simplified.symbol}"

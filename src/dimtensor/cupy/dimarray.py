"""CuPy DimArray: GPU arrays with dimensional awareness and uncertainty.

This module provides a CuPy-based DimArray that wraps CuPy GPU arrays with
physical unit tracking, mirroring the NumPy DimArray API while supporting
GPU memory management.
"""

from __future__ import annotations

from typing import Any, Iterator, overload

import cupy as cp
from cupy import ndarray as CuPyArray

from ..core.dimensions import Dimension
from ..core.units import Unit, dimensionless
from ..errors import DimensionError, UnitConversionError


class DimArray:
    """A CuPy GPU array with attached physical units.

    DimArray wraps a CuPy array and tracks its physical dimensions through
    all arithmetic operations. Operations between incompatible dimensions
    raise DimensionError immediately, catching physics errors early.

    Supports uncertainty propagation for scientific computing workflows.

    Examples:
        >>> import cupy as cp
        >>> from dimtensor.cupy import DimArray
        >>> from dimtensor import units
        >>>
        >>> # Create GPU arrays with units
        >>> v = DimArray(cp.array([1.0, 2.0, 3.0]), units.m / units.s)
        >>> t = DimArray(cp.array([0.5, 1.0, 1.5]), units.s)
        >>> d = v * t  # distance in meters, computed on GPU
        >>> print(d)
        [0.5 2.0 4.5] m
        >>>
        >>> # With uncertainty
        >>> mass = DimArray(cp.array([1.0]), units.kg, uncertainty=cp.array([0.01]))
        >>> accel = DimArray(cp.array([9.8]), units.m / units.s**2, uncertainty=cp.array([0.1]))
        >>> force = mass * accel  # Uncertainty propagates
        >>>
        >>> # Transfer to CPU
        >>> d_numpy = d.numpy()
    """

    __slots__ = ("_data", "_unit", "_uncertainty")

    def __init__(
        self,
        data: CuPyArray | Any,
        unit: Unit | None = None,
        dtype: Any = None,
        copy: bool = False,
        uncertainty: CuPyArray | Any | None = None,
    ) -> None:
        """Create a CuPy DimArray.

        Args:
            data: Array data (CuPy array, NumPy array, list, or scalar).
            unit: Physical unit. If None, assumes dimensionless.
            dtype: CuPy dtype for the underlying array.
            copy: If True, always copy the data.
            uncertainty: Optional absolute uncertainty (same shape as data).
        """
        # Handle DimArray input
        if isinstance(data, DimArray):
            if copy:
                arr = cp.array(data._data, dtype=dtype, copy=True)
            else:
                arr = cp.asarray(data._data, dtype=dtype)
            unit = unit if unit is not None else data._unit
            # Inherit uncertainty if not explicitly provided
            if uncertainty is None and data._uncertainty is not None:
                uncertainty = data._uncertainty
        else:
            # Convert to CuPy array (handles NumPy arrays, lists, scalars)
            if copy:
                arr = cp.array(data, dtype=dtype, copy=True)
            else:
                arr = cp.asarray(data, dtype=dtype)

        self._data: CuPyArray = arr
        self._unit: Unit = unit if unit is not None else dimensionless

        # Handle uncertainty
        if uncertainty is not None:
            if copy:
                unc_arr = cp.array(uncertainty, dtype=dtype, copy=True)
            else:
                unc_arr = cp.asarray(uncertainty, dtype=dtype)
            if unc_arr.shape != arr.shape:
                raise ValueError(
                    f"Uncertainty shape {unc_arr.shape} must match data shape {arr.shape}"
                )
            self._uncertainty: CuPyArray | None = unc_arr
        else:
            self._uncertainty = None

    @classmethod
    def _from_data_and_unit(
        cls,
        data: CuPyArray,
        unit: Unit,
        uncertainty: CuPyArray | None = None,
    ) -> DimArray:
        """Internal constructor that doesn't copy data."""
        result = object.__new__(cls)
        result._data = data
        result._unit = unit
        result._uncertainty = uncertainty
        return result

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def data(self) -> CuPyArray:
        """The underlying CuPy array (read-only view)."""
        # CuPy arrays don't have a simple read-only view like NumPy
        # Return the array directly; users should not modify it
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

    @property
    def uncertainty(self) -> CuPyArray | None:
        """The absolute uncertainty of this array, or None if not tracked."""
        return self._uncertainty

    @property
    def has_uncertainty(self) -> bool:
        """Check if this array has uncertainty information."""
        return self._uncertainty is not None

    @property
    def relative_uncertainty(self) -> CuPyArray | None:
        """The relative uncertainty (sigma/|value|), or None if not tracked.

        Returns infinity where data values are zero.
        """
        if self._uncertainty is None:
            return None
        rel_unc = cp.abs(self._uncertainty / self._data)
        # Handle division by zero: where data is 0, relative uncertainty is inf
        rel_unc = cp.where(self._data == 0, cp.inf, rel_unc)
        return rel_unc

    @property
    def device(self) -> int:
        """The CUDA device ID where this array resides."""
        return self._data.device.id

    # =========================================================================
    # Unit conversion
    # =========================================================================

    def to(self, unit: Unit) -> DimArray:
        """Convert to a different unit with the same dimension.

        Args:
            unit: Target unit (must have same dimension).

        Returns:
            New DimArray with converted values and new unit.
            Uncertainty is scaled by the same conversion factor.

        Raises:
            UnitConversionError: If dimensions don't match.
        """
        if not self._unit.is_compatible(unit):
            raise UnitConversionError.incompatible(self._unit.symbol, unit.symbol)

        factor = self._unit.conversion_factor(unit)
        new_data = self._data * factor
        new_uncertainty = None
        if self._uncertainty is not None:
            new_uncertainty = self._uncertainty * factor
        return DimArray._from_data_and_unit(new_data, unit, new_uncertainty)

    def to_base_units(self) -> DimArray:
        """Convert to SI base units.

        Returns a DimArray with scale factor 1.0 (pure SI units).
        Uncertainty is scaled accordingly.
        """
        # Create a unit with the same dimension but scale 1.0
        base_unit = Unit(str(self._unit.dimension), self._unit.dimension, 1.0)
        new_data = self._data * self._unit.scale
        new_uncertainty = None
        if self._uncertainty is not None:
            new_uncertainty = self._uncertainty * self._unit.scale
        return DimArray._from_data_and_unit(new_data, base_unit, new_uncertainty)

    def magnitude(self) -> CuPyArray:
        """Return the numerical magnitude (stripping units).

        Use with caution - this loses dimensional safety.
        """
        return self._data.copy()

    # =========================================================================
    # Uncertainty propagation helpers
    # =========================================================================

    @staticmethod
    def _propagate_add_sub(
        unc_a: CuPyArray | None,
        unc_b: CuPyArray | None,
    ) -> CuPyArray | None:
        """Propagate uncertainty for addition/subtraction.

        sigma_z = sqrt(sigma_x^2 + sigma_y^2)
        """
        if unc_a is None and unc_b is None:
            return None
        if unc_a is None:
            assert unc_b is not None  # for mypy
            return unc_b.copy()
        if unc_b is None:
            return unc_a.copy()
        return cp.sqrt(unc_a**2 + unc_b**2)

    @staticmethod
    def _propagate_mul_div(
        val_a: CuPyArray,
        unc_a: CuPyArray | None,
        val_b: CuPyArray,
        unc_b: CuPyArray | None,
        result: CuPyArray,
    ) -> CuPyArray | None:
        """Propagate uncertainty for multiplication/division.

        sigma_z/|z| = sqrt((sigma_x/x)^2 + (sigma_y/y)^2)
        sigma_z = |z| * sqrt((sigma_x/x)^2 + (sigma_y/y)^2)
        """
        if unc_a is None and unc_b is None:
            return None

        rel_a_sq = (unc_a / val_a) ** 2 if unc_a is not None else 0
        rel_b_sq = (unc_b / val_b) ** 2 if unc_b is not None else 0
        rel_combined = cp.sqrt(rel_a_sq + rel_b_sq)
        unc_result = cp.abs(result) * rel_combined
        # Handle NaN from 0/0 cases
        unc_result = cp.nan_to_num(unc_result, nan=0.0, posinf=cp.inf, neginf=cp.inf)

        return unc_result

    @staticmethod
    def _propagate_power(
        val: CuPyArray,
        unc: CuPyArray | None,
        power: float,
        result: CuPyArray,
    ) -> CuPyArray | None:
        """Propagate uncertainty for power operation.

        sigma_z/|z| = |n| * sigma_x/|x|
        sigma_z = |z| * |n| * sigma_x/|x|
        """
        if unc is None:
            return None

        rel_unc = unc / cp.abs(val)
        unc_result = cp.abs(result) * abs(power) * rel_unc
        unc_result = cp.nan_to_num(unc_result, nan=0.0, posinf=cp.inf, neginf=cp.inf)

        return unc_result

    # =========================================================================
    # Arithmetic operations
    # =========================================================================

    def __add__(self, other: DimArray | Any) -> DimArray:
        """Add two DimArrays (must have same dimension)."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "add"
                )
            # Convert other to same unit for consistent result
            other_converted = other.to(self._unit)
            new_data = self._data + other_converted._data
            new_uncertainty = self._propagate_add_sub(
                self._uncertainty, other_converted._uncertainty
            )
            return DimArray._from_data_and_unit(new_data, self._unit, new_uncertainty)
        else:
            # Adding a raw number - only valid if dimensionless
            if not self.is_dimensionless:
                raise DimensionError(
                    f"Cannot add dimensionless number to quantity with dimension {self.dimension}"
                )
            new_data = self._data + cp.asarray(other)
            # Scalar has no uncertainty, so uncertainty unchanged
            return DimArray._from_data_and_unit(new_data, self._unit, self._uncertainty)

    def __radd__(self, other: Any) -> DimArray:
        """Right add (for scalar + DimArray)."""
        return self.__add__(other)

    def __sub__(self, other: DimArray | Any) -> DimArray:
        """Subtract two DimArrays (must have same dimension)."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "subtract"
                )
            other_converted = other.to(self._unit)
            new_data = self._data - other_converted._data
            new_uncertainty = self._propagate_add_sub(
                self._uncertainty, other_converted._uncertainty
            )
            return DimArray._from_data_and_unit(new_data, self._unit, new_uncertainty)
        else:
            if not self.is_dimensionless:
                raise DimensionError(
                    f"Cannot subtract dimensionless number from quantity with dimension {self.dimension}"
                )
            new_data = self._data - cp.asarray(other)
            # Scalar has no uncertainty, so uncertainty unchanged
            return DimArray._from_data_and_unit(new_data, self._unit, self._uncertainty)

    def __rsub__(self, other: Any) -> DimArray:
        """Right subtract (for scalar - DimArray)."""
        if not self.is_dimensionless:
            raise DimensionError(
                f"Cannot subtract quantity with dimension {self.dimension} from dimensionless number"
            )
        new_data = cp.asarray(other) - self._data
        # Scalar has no uncertainty, uncertainty unchanged in magnitude
        return DimArray._from_data_and_unit(new_data, self._unit, self._uncertainty)

    def __mul__(self, other: DimArray | Any) -> DimArray:
        """Multiply DimArrays (dimensions multiply)."""
        # Handle Constant type (late import to avoid circular dependency)
        from ..constants._base import Constant

        if isinstance(other, Constant):
            return self * other.to_dimarray()
        if isinstance(other, DimArray):
            new_unit = self._unit * other._unit
            new_data = self._data * other._data
            new_uncertainty = self._propagate_mul_div(
                self._data,
                self._uncertainty,
                other._data,
                other._uncertainty,
                new_data,
            )
            return DimArray._from_data_and_unit(new_data, new_unit, new_uncertainty)
        else:
            # Scalar multiplication: sigma_z = |scalar| * sigma_x
            scalar = cp.asarray(other)
            new_data = self._data * scalar
            new_uncertainty = None
            if self._uncertainty is not None:
                new_uncertainty = cp.abs(scalar) * self._uncertainty
            return DimArray._from_data_and_unit(new_data, self._unit, new_uncertainty)

    def __rmul__(self, other: Any) -> DimArray:
        """Right multiply (for scalar * DimArray)."""
        return self.__mul__(other)

    def __truediv__(self, other: DimArray | Any) -> DimArray:
        """Divide DimArrays (dimensions divide)."""
        # Handle Constant type (late import to avoid circular dependency)
        from ..constants._base import Constant

        if isinstance(other, Constant):
            return self / other.to_dimarray()
        if isinstance(other, DimArray):
            new_unit = self._unit / other._unit
            new_data = self._data / other._data
            new_uncertainty = self._propagate_mul_div(
                self._data,
                self._uncertainty,
                other._data,
                other._uncertainty,
                new_data,
            )
            return DimArray._from_data_and_unit(new_data, new_unit, new_uncertainty)
        else:
            # Scalar division: sigma_z = sigma_x / |scalar|
            scalar = cp.asarray(other)
            new_data = self._data / scalar
            new_uncertainty = None
            if self._uncertainty is not None:
                new_uncertainty = self._uncertainty / cp.abs(scalar)
            return DimArray._from_data_and_unit(new_data, self._unit, new_uncertainty)

    def __rtruediv__(self, other: Any) -> DimArray:
        """Right divide (for scalar / DimArray)."""
        new_unit = Unit(
            f"1/{self._unit.symbol}",
            self._unit.dimension ** -1,
            1.0 / self._unit.scale,
        )
        scalar = cp.asarray(other)
        new_data = scalar / self._data
        new_uncertainty = None
        if self._uncertainty is not None:
            # sigma_z/|z| = sigma_x/|x| for scalar/x
            rel_unc = self._uncertainty / cp.abs(self._data)
            new_uncertainty = cp.abs(new_data) * rel_unc
            new_uncertainty = cp.nan_to_num(new_uncertainty, nan=0.0)
        return DimArray._from_data_and_unit(new_data, new_unit, new_uncertainty)

    def __pow__(self, power: int | float) -> DimArray:
        """Raise DimArray to a power."""
        new_unit = self._unit ** power
        new_data = self._data ** power
        new_uncertainty = self._propagate_power(
            self._data, self._uncertainty, power, new_data
        )
        return DimArray._from_data_and_unit(new_data, new_unit, new_uncertainty)

    def __neg__(self) -> DimArray:
        """Negate values."""
        return DimArray._from_data_and_unit(-self._data, self._unit, self._uncertainty)

    def __pos__(self) -> DimArray:
        """Unary positive (returns copy)."""
        unc_copy = self._uncertainty.copy() if self._uncertainty is not None else None
        return DimArray._from_data_and_unit(+self._data, self._unit, unc_copy)

    def __abs__(self) -> DimArray:
        """Absolute value."""
        return DimArray._from_data_and_unit(
            cp.abs(self._data), self._unit, self._uncertainty
        )

    def sqrt(self) -> DimArray:
        """Square root (dimension exponents halve)."""
        return self ** 0.5

    # =========================================================================
    # Comparison operations
    # =========================================================================

    def __eq__(self, other: object) -> CuPyArray | bool:  # type: ignore[override]
        """Element-wise equality (requires same dimension)."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                return False
            other_converted = other.to(self._unit)
            result: CuPyArray = self._data == other_converted._data
            return result
        elif self.is_dimensionless:
            result = self._data == cp.asarray(other)
            return result
        return False

    def __ne__(self, other: object) -> CuPyArray | bool:  # type: ignore[override]
        """Element-wise inequality."""
        result = self.__eq__(other)
        if isinstance(result, cp.ndarray):
            return ~result
        return not result

    def __lt__(self, other: DimArray | Any) -> CuPyArray:
        """Element-wise less than."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to(self._unit)
            return self._data < other_converted._data
        elif self.is_dimensionless:
            return self._data < cp.asarray(other)
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    def __le__(self, other: DimArray | Any) -> CuPyArray:
        """Element-wise less than or equal."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to(self._unit)
            return self._data <= other_converted._data
        elif self.is_dimensionless:
            return self._data <= cp.asarray(other)
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    def __gt__(self, other: DimArray | Any) -> CuPyArray:
        """Element-wise greater than."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to(self._unit)
            return self._data > other_converted._data
        elif self.is_dimensionless:
            return self._data > cp.asarray(other)
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    def __ge__(self, other: DimArray | Any) -> CuPyArray:
        """Element-wise greater than or equal."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to(self._unit)
            return self._data >= other_converted._data
        elif self.is_dimensionless:
            return self._data >= cp.asarray(other)
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    # =========================================================================
    # Indexing
    # =========================================================================

    @overload
    def __getitem__(self, key: int) -> DimArray: ...

    @overload
    def __getitem__(self, key: slice | CuPyArray | tuple[Any, ...]) -> DimArray: ...

    def __getitem__(
        self, key: int | slice | CuPyArray | tuple[Any, ...]
    ) -> DimArray:
        """Index into the array, preserving units and uncertainty."""
        result = self._data[key]
        unc_result = None
        if self._uncertainty is not None:
            unc_result = self._uncertainty[key]

        # Always return DimArray to maintain dimensional safety
        if result.ndim == 0:
            result = result.reshape((1,))
            if unc_result is not None:
                unc_result = unc_result.reshape((1,))

        return DimArray._from_data_and_unit(result, self._unit, unc_result)

    def __len__(self) -> int:
        """Length of first dimension."""
        return len(self._data)

    def __iter__(self) -> Iterator[DimArray]:
        """Iterate over first dimension."""
        for i in range(len(self)):
            yield self[i]

    # =========================================================================
    # Reduction operations
    # =========================================================================

    def sum(self, axis: int | None = None, keepdims: bool = False) -> DimArray:
        """Sum of array elements.

        Uncertainty propagation: sigma_sum = sqrt(sum(sigma_i^2))
        """
        result = self._data.sum(axis=axis, keepdims=keepdims)
        if result.ndim == 0:
            result = result.reshape((1,))

        new_uncertainty = None
        if self._uncertainty is not None:
            # For sum: sigma = sqrt(sum(sigma_i^2))
            unc_squared_sum = (self._uncertainty**2).sum(axis=axis, keepdims=keepdims)
            new_uncertainty = cp.sqrt(unc_squared_sum)
            if new_uncertainty.ndim == 0:
                new_uncertainty = new_uncertainty.reshape((1,))

        return DimArray._from_data_and_unit(result, self._unit, new_uncertainty)

    def mean(self, axis: int | None = None, keepdims: bool = False) -> DimArray:
        """Mean of array elements.

        Uncertainty propagation: sigma_mean = sqrt(sum(sigma_i^2)) / N
        """
        result = self._data.mean(axis=axis, keepdims=keepdims)
        if result.ndim == 0:
            result = result.reshape((1,))

        new_uncertainty = None
        if self._uncertainty is not None:
            # Count elements along axis
            if axis is None:
                n = self._data.size
            else:
                n = self._data.shape[axis]

            unc_squared_sum = (self._uncertainty**2).sum(axis=axis, keepdims=keepdims)
            new_uncertainty = cp.sqrt(unc_squared_sum) / n
            if new_uncertainty.ndim == 0:
                new_uncertainty = new_uncertainty.reshape((1,))

        return DimArray._from_data_and_unit(result, self._unit, new_uncertainty)

    def std(self, axis: int | None = None, keepdims: bool = False) -> DimArray:
        """Standard deviation of array elements.

        Note: Uncertainty propagation through std is complex and not implemented.
        The result will have no uncertainty information.
        """
        result = self._data.std(axis=axis, keepdims=keepdims)
        if result.ndim == 0:
            result = result.reshape((1,))
        return DimArray._from_data_and_unit(result, self._unit, None)

    def var(self, axis: int | None = None, keepdims: bool = False) -> DimArray:
        """Variance of array elements.

        Note: Variance has units squared (e.g., m -> m^2).
        Uncertainty propagation through variance is not implemented.

        Args:
            axis: Axis along which to compute variance.
            keepdims: If True, reduced dimensions are kept with size 1.

        Returns:
            Variance with squared unit.
        """
        result = self._data.var(axis=axis, keepdims=keepdims)
        if result.ndim == 0:
            result = result.reshape((1,))
        # Variance squares the unit: m -> m^2
        new_unit = self._unit**2
        return DimArray._from_data_and_unit(result, new_unit, None)

    def min(self, axis: int | None = None, keepdims: bool = False) -> DimArray:
        """Minimum value.

        Uncertainty: takes uncertainty of the minimum element.
        """
        result = self._data.min(axis=axis, keepdims=keepdims)
        if result.ndim == 0:
            result = result.reshape((1,))

        new_uncertainty = None
        if self._uncertainty is not None:
            # Get index of minimum and extract corresponding uncertainty
            if axis is None:
                idx = self._data.argmin()
                new_uncertainty = self._uncertainty.flatten()[idx].reshape((1,))
            else:
                indices = cp.expand_dims(self._data.argmin(axis=axis), axis=axis)
                new_uncertainty = cp.take_along_axis(self._uncertainty, indices, axis=axis)
                if not keepdims:
                    new_uncertainty = cp.squeeze(new_uncertainty, axis=axis)
                if new_uncertainty.ndim == 0:
                    new_uncertainty = new_uncertainty.reshape((1,))

        return DimArray._from_data_and_unit(result, self._unit, new_uncertainty)

    def max(self, axis: int | None = None, keepdims: bool = False) -> DimArray:
        """Maximum value.

        Uncertainty: takes uncertainty of the maximum element.
        """
        result = self._data.max(axis=axis, keepdims=keepdims)
        if result.ndim == 0:
            result = result.reshape((1,))

        new_uncertainty = None
        if self._uncertainty is not None:
            # Get index of maximum and extract corresponding uncertainty
            if axis is None:
                idx = self._data.argmax()
                new_uncertainty = self._uncertainty.flatten()[idx].reshape((1,))
            else:
                indices = cp.expand_dims(self._data.argmax(axis=axis), axis=axis)
                new_uncertainty = cp.take_along_axis(self._uncertainty, indices, axis=axis)
                if not keepdims:
                    new_uncertainty = cp.squeeze(new_uncertainty, axis=axis)
                if new_uncertainty.ndim == 0:
                    new_uncertainty = new_uncertainty.reshape((1,))

        return DimArray._from_data_and_unit(result, self._unit, new_uncertainty)

    # =========================================================================
    # Searching operations
    # =========================================================================

    def argmin(self, axis: int | None = None) -> Any:
        """Return indices of minimum values.

        Args:
            axis: Axis along which to find minimum. If None, returns
                index into flattened array.

        Returns:
            Index or array of indices (dimensionless integers).
        """
        return self._data.argmin(axis=axis)

    def argmax(self, axis: int | None = None) -> Any:
        """Return indices of maximum values.

        Args:
            axis: Axis along which to find maximum. If None, returns
                index into flattened array.

        Returns:
            Index or array of indices (dimensionless integers).
        """
        return self._data.argmax(axis=axis)

    # =========================================================================
    # Reshaping operations
    # =========================================================================

    def reshape(self, shape: tuple[int, ...] | list[int]) -> DimArray:
        """Return a reshaped array with the same unit.

        Args:
            shape: New shape. One dimension may be -1, which is inferred.

        Returns:
            Reshaped DimArray with same unit and reshaped uncertainty.
        """
        new_data = self._data.reshape(shape)
        new_uncertainty = None
        if self._uncertainty is not None:
            new_uncertainty = self._uncertainty.reshape(shape)
        return DimArray._from_data_and_unit(new_data, self._unit, new_uncertainty)

    def transpose(self, axes: tuple[int, ...] | list[int] | None = None) -> DimArray:
        """Permute the dimensions of the array.

        Args:
            axes: Permutation of dimensions. If None, reverses dimensions.

        Returns:
            Transposed DimArray with same unit.
        """
        new_data = self._data.transpose(axes)
        new_uncertainty = None
        if self._uncertainty is not None:
            new_uncertainty = self._uncertainty.transpose(axes)
        return DimArray._from_data_and_unit(new_data, self._unit, new_uncertainty)

    def flatten(self) -> DimArray:
        """Return a 1D flattened copy of the array.

        Returns:
            Flattened DimArray with same unit.
        """
        new_data = self._data.flatten()
        new_uncertainty = None
        if self._uncertainty is not None:
            new_uncertainty = self._uncertainty.flatten()
        return DimArray._from_data_and_unit(new_data, self._unit, new_uncertainty)

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> DimArray:
        """Remove size-1 dimensions.

        Args:
            axis: Axis or axes to squeeze. If None, all size-1 dimensions are removed.

        Returns:
            Squeezed DimArray with same unit.
        """
        new_data = cp.squeeze(self._data, axis=axis)
        new_uncertainty = None
        if self._uncertainty is not None:
            new_uncertainty = cp.squeeze(self._uncertainty, axis=axis)
        return DimArray._from_data_and_unit(new_data, self._unit, new_uncertainty)

    def expand_dims(self, axis: int | tuple[int, ...]) -> DimArray:
        """Add size-1 dimension.

        Args:
            axis: Axis position(s) for the new dimension(s).

        Returns:
            DimArray with expanded dimensions.
        """
        new_data = cp.expand_dims(self._data, axis=axis)
        new_uncertainty = None
        if self._uncertainty is not None:
            new_uncertainty = cp.expand_dims(self._uncertainty, axis=axis)
        return DimArray._from_data_and_unit(new_data, self._unit, new_uncertainty)

    # =========================================================================
    # Linear algebra
    # =========================================================================

    def dot(self, other: DimArray) -> DimArray:
        """Dot product (dimensions multiply).

        Args:
            other: DimArray to compute dot product with.

        Returns:
            Dot product with combined units.
        """
        new_unit = self._unit * other._unit
        new_data = cp.dot(self._data, other._data)
        if new_data.ndim == 0:
            new_data = new_data.reshape((1,))
        # Uncertainty propagation for dot product is complex; not implemented
        return DimArray._from_data_and_unit(new_data, new_unit, None)

    def matmul(self, other: DimArray) -> DimArray:
        """Matrix multiplication (dimensions multiply).

        Args:
            other: DimArray for matrix multiplication.

        Returns:
            Matrix product with combined units.
        """
        new_unit = self._unit * other._unit
        new_data = cp.matmul(self._data, other._data)
        # Uncertainty propagation for matmul is complex; not implemented
        return DimArray._from_data_and_unit(new_data, new_unit, None)

    def __matmul__(self, other: DimArray) -> DimArray:
        """Matrix multiplication operator @."""
        return self.matmul(other)

    def norm(
        self,
        ord: int | float | str | None = None,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> DimArray:
        """Vector or matrix norm (preserves units).

        Args:
            ord: Order of the norm (see numpy.linalg.norm).
            axis: Axis or axes along which to compute norm.
            keepdims: If True, reduced dimensions are kept with size 1.

        Returns:
            Norm with same unit as input.
        """
        result = cp.linalg.norm(self._data, ord=ord, axis=axis, keepdims=keepdims)
        if result.ndim == 0:
            result = result.reshape((1,))
        return DimArray._from_data_and_unit(result, self._unit, None)

    # =========================================================================
    # String representations
    # =========================================================================

    def __repr__(self) -> str:
        """Detailed representation."""
        if self._uncertainty is not None:
            return (
                f"DimArray({self._data!r}, unit={self._unit.symbol!r}, "
                f"uncertainty={self._uncertainty!r})"
            )
        return f"DimArray({self._data!r}, unit={self._unit.symbol!r})"

    def __str__(self) -> str:
        """Human-readable string."""
        # Get CPU version for display
        data_cpu = cp.asnumpy(self._data)

        if self.is_dimensionless:
            return str(data_cpu)

        simplified = self._unit.simplified()
        return f"{data_cpu} {simplified.symbol}"

    def __format__(self, format_spec: str) -> str:
        """Support format strings like f'{distance:.2f}'."""
        data_cpu = cp.asnumpy(self._data)

        if data_cpu.size == 1:
            value_str = format(data_cpu.item(), format_spec)
            if self._uncertainty is not None:
                unc_cpu = cp.asnumpy(self._uncertainty)
                unc_str = format(unc_cpu.item(), format_spec)
                value_str = f"{value_str} +/- {unc_str}"
        else:
            if format_spec:
                formatted = [format(x, format_spec) for x in data_cpu.flat]
                value_str = "[" + ", ".join(formatted) + "]"
            else:
                value_str = str(data_cpu)
            if self._uncertainty is not None:
                value_str += " +/- [...]"

        if self.is_dimensionless:
            return value_str
        simplified = self._unit.simplified()
        return f"{value_str} {simplified.symbol}"

    # =========================================================================
    # NumPy/CuPy interoperability
    # =========================================================================

    def __array__(self, dtype: Any = None) -> Any:
        """Convert to numpy array (loses unit information).

        This enables np.asarray(dim_array) to work.
        """
        import numpy as np
        data_cpu = cp.asnumpy(self._data)
        if dtype is None:
            return data_cpu.copy()
        return data_cpu.astype(dtype)

    @property
    def __array_priority__(self) -> float:
        """Ensure DimArray operations take precedence."""
        return 10.0

    def numpy(self) -> Any:
        """Convert to numpy array (explicit CPU transfer).

        Returns:
            NumPy ndarray with data transferred from GPU.

        Note:
            This involves a GPU-to-CPU data transfer. For large arrays,
            this may be slow.
        """
        import numpy as np
        return np.asarray(cp.asnumpy(self._data))

    def cupy(self) -> CuPyArray:
        """Return the underlying CuPy array.

        Returns:
            The underlying CuPy ndarray.
        """
        return self._data

    def get(self) -> Any:
        """Transfer data to CPU (CuPy convention).

        This is equivalent to numpy() and follows CuPy's convention
        for CPU transfer.

        Returns:
            NumPy ndarray with data transferred from GPU.
        """
        return self.numpy()

    def set(self, data: Any) -> None:
        """Update data from a NumPy array (in-place).

        Args:
            data: NumPy array or array-like to copy to GPU.

        Note:
            This modifies the DimArray in-place. Shape must match.
        """
        new_data = cp.asarray(data)
        if new_data.shape != self._data.shape:
            raise ValueError(
                f"Shape mismatch: expected {self._data.shape}, got {new_data.shape}"
            )
        self._data[:] = new_data

    # =========================================================================
    # GPU memory management
    # =========================================================================

    def to_device(self, device_id: int) -> DimArray:
        """Transfer array to a different GPU device.

        Args:
            device_id: Target CUDA device ID.

        Returns:
            New DimArray on the specified device.
        """
        with cp.cuda.Device(device_id):
            new_data = cp.asarray(self._data)
            new_uncertainty = None
            if self._uncertainty is not None:
                new_uncertainty = cp.asarray(self._uncertainty)
        return DimArray._from_data_and_unit(new_data, self._unit, new_uncertainty)

    def cpu(self) -> Any:
        """Transfer to CPU and return as NumPy DimArray.

        Returns:
            A NumPy-based DimArray from the core module.
        """
        from ..core.dimarray import DimArray as NumpyDimArray
        data_cpu = cp.asnumpy(self._data)
        uncertainty_cpu = None
        if self._uncertainty is not None:
            uncertainty_cpu = cp.asnumpy(self._uncertainty)
        return NumpyDimArray._from_data_and_unit(data_cpu, self._unit, uncertainty_cpu)

    def gpu(self) -> DimArray:
        """Ensure array is on GPU (no-op for CuPy DimArray).

        Returns:
            Self (already on GPU).
        """
        return self

"""DimArray: numpy arrays with dimensional awareness.

DimArray wraps numpy arrays and tracks physical units through all operations,
catching dimensional errors at operation time rather than after hours of computation.
"""

from __future__ import annotations

from typing import Any, Iterator, overload

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from ..config import display as _display
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

    __slots__ = ("_data", "_unit", "_uncertainty")

    def __init__(
        self,
        data: ArrayLike,
        unit: Unit | None = None,
        dtype: DTypeLike = None,
        copy: bool = False,
        uncertainty: ArrayLike | None = None,
    ) -> None:
        """Create a DimArray.

        Args:
            data: Array-like data (list, tuple, numpy array, or scalar).
            unit: Physical unit of the data. If None, assumes dimensionless.
            dtype: Numpy dtype for the underlying array.
            copy: If True, always copy the data.
            uncertainty: Optional absolute uncertainty (same shape as data).
        """
        # Convert to numpy array
        if isinstance(data, DimArray):
            arr = np.array(data._data, dtype=dtype, copy=copy)
            unit = unit if unit is not None else data._unit
            # Inherit uncertainty if not explicitly provided
            if uncertainty is None and data._uncertainty is not None:
                uncertainty = data._uncertainty
        else:
            arr = np.array(data, dtype=dtype, copy=copy)

        self._data: NDArray[Any] = arr
        self._unit: Unit = unit if unit is not None else dimensionless

        # Handle uncertainty
        if uncertainty is not None:
            unc_arr = np.array(uncertainty, dtype=dtype, copy=copy)
            if unc_arr.shape != arr.shape:
                raise ValueError(
                    f"Uncertainty shape {unc_arr.shape} must match data shape {arr.shape}"
                )
            self._uncertainty: NDArray[Any] | None = unc_arr
        else:
            self._uncertainty = None

    @classmethod
    def _from_data_and_unit(
        cls,
        data: NDArray[Any],
        unit: Unit,
        uncertainty: NDArray[Any] | None = None,
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

    @property
    def uncertainty(self) -> NDArray[Any] | None:
        """The absolute uncertainty of this array (read-only view), or None if not tracked."""
        if self._uncertainty is None:
            return None
        result = self._uncertainty.view()
        result.flags.writeable = False
        return result

    @property
    def has_uncertainty(self) -> bool:
        """Check if this array has uncertainty information."""
        return self._uncertainty is not None

    @property
    def relative_uncertainty(self) -> NDArray[Any] | None:
        """The relative uncertainty (sigma/|value|), or None if uncertainty not tracked.

        Returns infinity where data values are zero.
        """
        if self._uncertainty is None:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_unc = np.abs(self._uncertainty / self._data)
            # Handle division by zero: where data is 0, relative uncertainty is inf
            rel_unc = np.where(self._data == 0, np.inf, rel_unc)
        result = rel_unc.view()
        result.flags.writeable = False
        return result

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(
        self, constraints: list[Any] | None = None
    ) -> DimArray:
        """Validate array values against constraints.

        Args:
            constraints: List of Constraint objects to check. If None,
                just returns self (no-op).

        Returns:
            Self if validation passes.

        Raises:
            ConstraintError: If any constraint is violated.

        Example:
            >>> from dimtensor.validation import Positive, Bounded
            >>> mass = DimArray([1.0, 2.0], units.kg)
            >>> mass.validate([Positive()])  # OK
            >>> prob = DimArray([0.5, 1.5], units.dimensionless)
            >>> prob.validate([Bounded(0, 1)])  # Raises ConstraintError
        """
        if constraints is None:
            return self
        for constraint in constraints:
            constraint.validate(self._data)
        return self

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

    def magnitude(self) -> NDArray[Any]:
        """Return the numerical magnitude (stripping units).

        Use with caution - this loses dimensional safety.
        """
        return self._data.copy()

    # =========================================================================
    # Uncertainty propagation helpers
    # =========================================================================

    @staticmethod
    def _propagate_add_sub(
        unc_a: NDArray[Any] | None,
        unc_b: NDArray[Any] | None,
    ) -> NDArray[Any] | None:
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
        return np.sqrt(unc_a**2 + unc_b**2)

    @staticmethod
    def _propagate_mul_div(
        val_a: NDArray[Any],
        unc_a: NDArray[Any] | None,
        val_b: NDArray[Any],
        unc_b: NDArray[Any] | None,
        result: NDArray[Any],
    ) -> NDArray[Any] | None:
        """Propagate uncertainty for multiplication/division.

        sigma_z/|z| = sqrt((sigma_x/x)^2 + (sigma_y/y)^2)
        sigma_z = |z| * sqrt((sigma_x/x)^2 + (sigma_y/y)^2)
        """
        if unc_a is None and unc_b is None:
            return None

        with np.errstate(divide="ignore", invalid="ignore"):
            rel_a_sq = (unc_a / val_a) ** 2 if unc_a is not None else 0
            rel_b_sq = (unc_b / val_b) ** 2 if unc_b is not None else 0
            rel_combined = np.sqrt(rel_a_sq + rel_b_sq)
            unc_result = np.abs(result) * rel_combined
            # Handle NaN from 0/0 cases
            unc_result = np.asarray(np.nan_to_num(unc_result, nan=0.0, posinf=np.inf, neginf=np.inf))

        return unc_result

    @staticmethod
    def _propagate_power(
        val: NDArray[Any],
        unc: NDArray[Any] | None,
        power: float,
        result: NDArray[Any],
    ) -> NDArray[Any] | None:
        """Propagate uncertainty for power operation.

        sigma_z/|z| = |n| * sigma_x/|x|
        sigma_z = |z| * |n| * sigma_x/|x|
        """
        if unc is None:
            return None

        with np.errstate(divide="ignore", invalid="ignore"):
            rel_unc = unc / np.abs(val)
            unc_result = np.abs(result) * np.abs(power) * rel_unc
            unc_result = np.asarray(np.nan_to_num(unc_result, nan=0.0, posinf=np.inf, neginf=np.inf))

        return unc_result

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
            new_data = self._data + np.asarray(other)
            # Scalar has no uncertainty, so uncertainty unchanged
            return DimArray._from_data_and_unit(new_data, self._unit, self._uncertainty)

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
            new_uncertainty = self._propagate_add_sub(
                self._uncertainty, other_converted._uncertainty
            )
            return DimArray._from_data_and_unit(new_data, self._unit, new_uncertainty)
        else:
            if not self.is_dimensionless:
                raise DimensionError(
                    f"Cannot subtract dimensionless number from quantity with dimension {self.dimension}"
                )
            new_data = self._data - np.asarray(other)
            # Scalar has no uncertainty, so uncertainty unchanged
            return DimArray._from_data_and_unit(new_data, self._unit, self._uncertainty)

    def __rsub__(self, other: ArrayLike) -> DimArray:
        """Right subtract (for scalar - DimArray)."""
        if not self.is_dimensionless:
            raise DimensionError(
                f"Cannot subtract quantity with dimension {self.dimension} from dimensionless number"
            )
        new_data = np.asarray(other) - self._data
        # Scalar has no uncertainty, uncertainty unchanged in magnitude
        return DimArray._from_data_and_unit(new_data, self._unit, self._uncertainty)

    def __mul__(self, other: DimArray | ArrayLike) -> DimArray:
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
            scalar = np.asarray(other)
            new_data = self._data * scalar
            new_uncertainty = None
            if self._uncertainty is not None:
                new_uncertainty = np.abs(scalar) * self._uncertainty
            return DimArray._from_data_and_unit(new_data, self._unit, new_uncertainty)

    def __rmul__(self, other: ArrayLike) -> DimArray:
        """Right multiply (for scalar * DimArray)."""
        return self.__mul__(other)

    def __truediv__(self, other: DimArray | ArrayLike) -> DimArray:
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
            scalar = np.asarray(other)
            new_data = self._data / scalar
            new_uncertainty = None
            if self._uncertainty is not None:
                new_uncertainty = self._uncertainty / np.abs(scalar)
            return DimArray._from_data_and_unit(new_data, self._unit, new_uncertainty)

    def __rtruediv__(self, other: ArrayLike) -> DimArray:
        """Right divide (for scalar / DimArray)."""
        new_unit = Unit(
            f"1/{self._unit.symbol}",
            self._unit.dimension ** -1,
            1.0 / self._unit.scale,
        )
        scalar = np.asarray(other)
        new_data = scalar / self._data
        new_uncertainty = None
        if self._uncertainty is not None:
            # sigma_z/|z| = sigma_x/|x| for scalar/x
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_unc = self._uncertainty / np.abs(self._data)
                new_uncertainty = np.abs(new_data) * rel_unc
                new_uncertainty = np.nan_to_num(new_uncertainty, nan=0.0)
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
            np.abs(self._data), self._unit, self._uncertainty
        )

    def sqrt(self) -> DimArray:
        """Square root (dimension exponents halve)."""
        return self ** 0.5

    # =========================================================================
    # Comparison operations
    # =========================================================================

    def __eq__(self, other: object) -> NDArray[np.bool_] | bool:  # type: ignore[override]
        """Element-wise equality (requires same dimension)."""
        if isinstance(other, DimArray):
            if self.dimension != other.dimension:
                return False
            other_converted = other.to(self._unit)
            result: NDArray[np.bool_] = self._data == other_converted._data
            return result
        elif self.is_dimensionless:
            result = self._data == np.asarray(other)
            return result
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
            return self._data < other_converted._data
        elif self.is_dimensionless:
            return self._data < np.asarray(other)
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
            return self._data <= other_converted._data
        elif self.is_dimensionless:
            return self._data <= np.asarray(other)
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
            return self._data > other_converted._data
        elif self.is_dimensionless:
            return self._data > np.asarray(other)
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
            return self._data >= other_converted._data
        elif self.is_dimensionless:
            return self._data >= np.asarray(other)
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
        """Index into the array, preserving units and uncertainty."""
        result = self._data[key]
        unc_result = None
        if self._uncertainty is not None:
            unc_result = self._uncertainty[key]

        # Always return DimArray to maintain dimensional safety
        if np.isscalar(result):
            result = np.array([result])
            if unc_result is not None:
                unc_result = np.array([unc_result])

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
        if np.isscalar(result):
            result = np.array([result])

        new_uncertainty = None
        if self._uncertainty is not None:
            # For sum: sigma = sqrt(sum(sigma_i^2))
            unc_squared_sum = (self._uncertainty**2).sum(axis=axis, keepdims=keepdims)
            new_uncertainty = np.sqrt(unc_squared_sum)
            if np.isscalar(new_uncertainty):
                new_uncertainty = np.array([new_uncertainty])

        return DimArray._from_data_and_unit(result, self._unit, new_uncertainty)

    def mean(self, axis: int | None = None, keepdims: bool = False) -> DimArray:
        """Mean of array elements.

        Uncertainty propagation: sigma_mean = sqrt(sum(sigma_i^2)) / N
        """
        result = self._data.mean(axis=axis, keepdims=keepdims)
        if np.isscalar(result):
            result = np.array([result])

        new_uncertainty = None
        if self._uncertainty is not None:
            # Count elements along axis
            if axis is None:
                n = self._data.size
            else:
                n = self._data.shape[axis]

            unc_squared_sum = (self._uncertainty**2).sum(axis=axis, keepdims=keepdims)
            new_uncertainty = np.sqrt(unc_squared_sum) / n
            if np.isscalar(new_uncertainty):
                new_uncertainty = np.array([new_uncertainty])

        return DimArray._from_data_and_unit(result, self._unit, new_uncertainty)

    def std(self, axis: int | None = None, keepdims: bool = False) -> DimArray:
        """Standard deviation of array elements.

        Note: Uncertainty propagation through std is complex and not implemented.
        The result will have no uncertainty information.
        """
        result = self._data.std(axis=axis, keepdims=keepdims)
        if np.isscalar(result):
            result = np.array([result])
        return DimArray._from_data_and_unit(result, self._unit, None)

    def min(self, axis: int | None = None, keepdims: bool = False) -> DimArray:
        """Minimum value.

        Uncertainty: takes uncertainty of the minimum element.
        """
        result = self._data.min(axis=axis, keepdims=keepdims)
        if np.isscalar(result):
            result = np.array([result])

        new_uncertainty = None
        if self._uncertainty is not None:
            # Get index of minimum and extract corresponding uncertainty
            if axis is None:
                idx = self._data.argmin()
                new_uncertainty = np.array([self._uncertainty.flat[idx]])
            else:
                indices = np.expand_dims(self._data.argmin(axis=axis), axis=axis)
                new_uncertainty = np.take_along_axis(self._uncertainty, indices, axis=axis)
                if not keepdims:
                    new_uncertainty = np.squeeze(new_uncertainty, axis=axis)
                if np.isscalar(new_uncertainty):
                    new_uncertainty = np.array([new_uncertainty])

        return DimArray._from_data_and_unit(result, self._unit, new_uncertainty)

    def max(self, axis: int | None = None, keepdims: bool = False) -> DimArray:
        """Maximum value.

        Uncertainty: takes uncertainty of the maximum element.
        """
        result = self._data.max(axis=axis, keepdims=keepdims)
        if np.isscalar(result):
            result = np.array([result])

        new_uncertainty = None
        if self._uncertainty is not None:
            # Get index of maximum and extract corresponding uncertainty
            if axis is None:
                idx = self._data.argmax()
                new_uncertainty = np.array([self._uncertainty.flat[idx]])
            else:
                indices = np.expand_dims(self._data.argmax(axis=axis), axis=axis)
                new_uncertainty = np.take_along_axis(self._uncertainty, indices, axis=axis)
                if not keepdims:
                    new_uncertainty = np.squeeze(new_uncertainty, axis=axis)
                if np.isscalar(new_uncertainty):
                    new_uncertainty = np.array([new_uncertainty])

        return DimArray._from_data_and_unit(result, self._unit, new_uncertainty)

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
        if np.isscalar(result):
            result = np.array([result])
        # Variance squares the unit: m -> m^2
        new_unit = self._unit**2
        return DimArray._from_data_and_unit(result, new_unit, None)

    # =========================================================================
    # Searching operations
    # =========================================================================

    def argmin(self, axis: int | None = None) -> np.intp | NDArray[np.intp]:
        """Return indices of minimum values.

        Args:
            axis: Axis along which to find minimum. If None, returns
                index into flattened array.

        Returns:
            Index or array of indices (dimensionless integers).
        """
        return self._data.argmin(axis=axis)

    def argmax(self, axis: int | None = None) -> np.intp | NDArray[np.intp]:
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
        # Format the array nicely using global display options
        if self._data.ndim == 0:
            value_str = f"{self._data.item():.{_display.precision}g}"
        elif self._data.size <= _display.threshold:
            value_str = np.array2string(
                self._data,
                separator=", ",
                precision=_display.precision,
                suppress_small=_display.suppress_small,
                max_line_width=_display.linewidth,
            )
        else:
            value_str = np.array2string(
                self._data,
                separator=", ",
                precision=_display.precision,
                suppress_small=_display.suppress_small,
                threshold=_display.threshold,
                edgeitems=_display.edgeitems,
                max_line_width=_display.linewidth,
            )

        # Add uncertainty if present
        if self._uncertainty is not None:
            if self._data.size == 1:
                unc_str = f" ± {self._uncertainty.item():.{_display.precision}g}"
            else:
                unc_str = " ± [...]"  # Abbreviated for arrays
            value_str = value_str + unc_str

        if self.is_dimensionless:
            return value_str
        # Use simplified unit symbol for display
        simplified = self._unit.simplified()
        return f"{value_str} {simplified.symbol}"

    def __format__(self, format_spec: str) -> str:
        """Support format strings like f'{distance:.2f}'.

        The format spec is applied to the numerical value(s).
        """
        if self._data.size == 1:
            # Single value: format directly
            value_str = format(self._data.item(), format_spec)
            if self._uncertainty is not None:
                unc_str = format(self._uncertainty.item(), format_spec)
                value_str = f"{value_str} ± {unc_str}"
        else:
            # Multiple values: format each element
            if format_spec:
                formatted = [format(x, format_spec) for x in self._data.flat]
                value_str = "[" + ", ".join(formatted) + "]"
            else:
                value_str = str(self._data)
            if self._uncertainty is not None:
                value_str += " ± [...]"

        if self.is_dimensionless:
            return value_str
        simplified = self._unit.simplified()
        return f"{value_str} {simplified.symbol}"

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

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        """Handle numpy ufuncs like np.sin, np.sqrt, np.add, etc.

        This enables expressions like:
            np.sin(angle)  # requires dimensionless
            np.sqrt(area)  # dimension exponents halve
            np.add(a, b)   # requires same dimension
        """
        # Only handle __call__ method for now
        if method != "__call__":
            return NotImplemented

        # Ufuncs that require dimensionless input and return dimensionless output
        _DIMENSIONLESS_UFUNCS = {
            np.sin, np.cos, np.tan,
            np.arcsin, np.arccos, np.arctan,
            np.sinh, np.cosh, np.tanh,
            np.arcsinh, np.arccosh, np.arctanh,
            np.exp, np.exp2, np.expm1,
            np.log, np.log2, np.log10, np.log1p,
        }

        # Ufuncs that preserve the unit (unary)
        _PRESERVE_UNIT_UFUNCS = {
            np.negative, np.positive, np.absolute, np.fabs,
            np.floor, np.ceil, np.trunc, np.rint,
        }

        # Handle dimensionless-requiring ufuncs
        if ufunc in _DIMENSIONLESS_UFUNCS:
            for inp in inputs:
                if isinstance(inp, DimArray) and not inp.is_dimensionless:
                    raise DimensionError(
                        f"{ufunc.__name__}() requires dimensionless input, "
                        f"got dimension {inp.dimension}"
                    )
            # Extract raw data
            raw_inputs = [
                inp._data if isinstance(inp, DimArray) else inp
                for inp in inputs
            ]
            result = ufunc(*raw_inputs, **kwargs)
            # Drop uncertainty for dimensionless ufuncs (propagation is complex)
            return DimArray._from_data_and_unit(result, dimensionless, None)

        # Handle sqrt specially - halves dimension exponents
        # Delegates to sqrt() method which handles uncertainty via __pow__
        if ufunc is np.sqrt:
            if len(inputs) != 1 or not isinstance(inputs[0], DimArray):
                return NotImplemented
            return inputs[0].sqrt()

        # Handle square - doubles dimension exponents
        # Delegates to __pow__ which handles uncertainty
        if ufunc is np.square:
            if len(inputs) != 1 or not isinstance(inputs[0], DimArray):
                return NotImplemented
            return inputs[0] ** 2

        # Handle unit-preserving ufuncs
        if ufunc in _PRESERVE_UNIT_UFUNCS:
            if len(inputs) != 1 or not isinstance(inputs[0], DimArray):
                return NotImplemented
            raw_result = ufunc(inputs[0]._data, **kwargs)
            # Preserve uncertainty for unit-preserving operations
            return DimArray._from_data_and_unit(
                raw_result, inputs[0]._unit, inputs[0]._uncertainty
            )

        # Handle binary arithmetic ufuncs
        if ufunc is np.add:
            return inputs[0] + inputs[1]
        if ufunc is np.subtract:
            return inputs[0] - inputs[1]
        if ufunc is np.multiply:
            return inputs[0] * inputs[1]
        if ufunc is np.divide or ufunc is np.true_divide:
            return inputs[0] / inputs[1]
        if ufunc is np.power:
            # Power requires dimensionless exponent
            base, exp = inputs
            if isinstance(exp, DimArray):
                if not exp.is_dimensionless:
                    raise DimensionError(
                        f"Exponent must be dimensionless, got {exp.dimension}"
                    )
                exp = exp._data
            if isinstance(base, DimArray):
                # For array exponents, all must be same value for dimension calc
                if hasattr(exp, "__len__") and len(set(exp.flat)) > 1:
                    raise DimensionError(
                        "Cannot raise to array of different powers"
                    )
                power = float(exp.flat[0]) if hasattr(exp, "flat") else float(exp)
                return base ** power
            return NotImplemented

        # For unhandled ufuncs, return NotImplemented
        return NotImplemented

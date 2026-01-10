"""DimTensor: TensorFlow tensors with dimensional awareness.

DimTensor wraps tf.Tensor and tracks physical units through all operations,
catching dimensional errors while preserving gradient functionality.
Works in both eager and graph (@tf.function) execution modes.
"""

from __future__ import annotations

import builtins
from typing import Any, Sequence, Union

import tensorflow as tf

from ..core.dimensions import Dimension
from ..core.units import Unit, dimensionless
from ..errors import DimensionError, UnitConversionError


class DimTensor:
    """A TensorFlow tensor with attached physical units.

    DimTensor wraps a tf.Tensor and tracks its physical dimensions through
    all arithmetic operations. Operations between incompatible dimensions
    raise DimensionError immediately.

    Works with both eager execution (TF2 default) and graph mode (@tf.function).
    Units are Python metadata stored outside the computation graph.

    Examples:
        >>> import tensorflow as tf
        >>> from dimtensor.tensorflow import DimTensor
        >>> from dimtensor import units
        >>>
        >>> v = DimTensor(tf.constant([1.0, 2.0, 3.0]), units.m / units.s)
        >>> t = DimTensor(tf.constant([0.5, 1.0, 1.5]), units.s)
        >>> d = v * t  # distance in meters
        >>> print(d)
        DimTensor([0.5, 2.0, 4.5], unit='m')
        >>>
        >>> @tf.function
        ... def compute_energy(mass, velocity):
        ...     return 0.5 * mass * velocity**2
        >>>
        >>> E = compute_energy(m, v)  # Works in graph mode
    """

    __slots__ = ("_data", "_unit")

    def __init__(
        self,
        data: Union[tf.Tensor, Sequence[float], float, "DimTensor"],
        unit: Unit | None = None,
        dtype: tf.DType | None = None,
    ) -> None:
        """Create a DimTensor.

        Args:
            data: Tensor data (tf.Tensor, list, or scalar).
            unit: Physical unit. If None, assumes dimensionless.
            dtype: TensorFlow dtype (float32, float64, etc.).
        """
        if isinstance(data, DimTensor):
            tensor = tf.identity(data._data)
            unit = unit if unit is not None else data._unit
        elif isinstance(data, tf.Tensor):
            tensor = tf.identity(data)
        else:
            tensor = tf.constant(data)

        if dtype is not None:
            tensor = tf.cast(tensor, dtype)

        self._data: tf.Tensor = tensor
        self._unit: Unit = unit if unit is not None else dimensionless

    @classmethod
    def _from_tensor_and_unit(cls, tensor: tf.Tensor, unit: Unit) -> DimTensor:
        """Internal constructor that doesn't copy tensor."""
        result = object.__new__(cls)
        result._data = tensor
        result._unit = unit
        return result

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def data(self) -> tf.Tensor:
        """The underlying tf.Tensor."""
        return self._data

    @property
    def unit(self) -> Unit:
        """The physical unit of this tensor."""
        return self._unit

    @property
    def dimension(self) -> Dimension:
        """The physical dimension of this tensor."""
        return self._unit.dimension

    @property
    def shape(self) -> tf.TensorShape:
        """Shape of the underlying tensor."""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._data.shape)

    @property
    def size(self) -> int:
        """Total number of elements."""
        return int(tf.size(self._data))

    @property
    def dtype(self) -> tf.DType:
        """Data type of the underlying tensor."""
        return self._data.dtype

    @property
    def device(self) -> str:
        """Device the tensor is on."""
        return self._data.device

    @property
    def is_dimensionless(self) -> bool:
        """Check if this tensor is dimensionless."""
        return self._unit.dimension.is_dimensionless

    # =========================================================================
    # Device and dtype operations
    # =========================================================================

    def to(self, dtype: tf.DType | None = None) -> DimTensor:
        """Cast tensor to a different dtype.

        For unit conversion, use to_unit() instead.

        Args:
            dtype: Target dtype (tf.float32, tf.float64, etc.).

        Returns:
            New DimTensor with target dtype.
        """
        if dtype is not None:
            new_tensor = tf.cast(self._data, dtype)
        else:
            new_tensor = self._data
        return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def float(self) -> DimTensor:
        """Cast to float32."""
        return DimTensor._from_tensor_and_unit(
            tf.cast(self._data, tf.float32), self._unit
        )

    def double(self) -> DimTensor:
        """Cast to float64."""
        return DimTensor._from_tensor_and_unit(
            tf.cast(self._data, tf.float64), self._unit
        )

    # =========================================================================
    # Unit conversion
    # =========================================================================

    def to_unit(self, unit: Unit) -> DimTensor:
        """Convert to a different unit with the same dimension.

        Args:
            unit: Target unit (must have same dimension).

        Returns:
            New DimTensor with converted values.

        Raises:
            UnitConversionError: If dimensions don't match.
        """
        if not self._unit.is_compatible(unit):
            raise UnitConversionError.incompatible(self._unit.symbol, unit.symbol)

        factor = self._unit.conversion_factor(unit)
        new_tensor = self._data * factor
        return DimTensor._from_tensor_and_unit(new_tensor, unit)

    def magnitude(self) -> tf.Tensor:
        """Return the numerical magnitude (stripping units).

        Use with caution - this loses dimensional safety.
        """
        return tf.identity(self._data)

    # =========================================================================
    # Arithmetic operations
    # =========================================================================

    def __add__(self, other: DimTensor | tf.Tensor | builtins.float) -> DimTensor:
        """Add tensors (must have same dimension)."""
        if isinstance(other, DimTensor):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "add"
                )
            other_converted = other.to_unit(self._unit)
            new_tensor = self._data + other_converted._data
            return DimTensor._from_tensor_and_unit(new_tensor, self._unit)
        else:
            if not self.is_dimensionless:
                raise DimensionError(
                    f"Cannot add dimensionless number to quantity with dimension {self.dimension}"
                )
            other_tensor = other if isinstance(other, tf.Tensor) else tf.constant(other, dtype=self._data.dtype)
            new_tensor = self._data + other_tensor
            return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def __radd__(self, other: tf.Tensor | builtins.float) -> DimTensor:
        """Right add."""
        return self.__add__(other)

    def __sub__(self, other: DimTensor | tf.Tensor | builtins.float) -> DimTensor:
        """Subtract tensors (must have same dimension)."""
        if isinstance(other, DimTensor):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "subtract"
                )
            other_converted = other.to_unit(self._unit)
            new_tensor = self._data - other_converted._data
            return DimTensor._from_tensor_and_unit(new_tensor, self._unit)
        else:
            if not self.is_dimensionless:
                raise DimensionError(
                    f"Cannot subtract dimensionless number from quantity with dimension {self.dimension}"
                )
            other_tensor = other if isinstance(other, tf.Tensor) else tf.constant(other, dtype=self._data.dtype)
            new_tensor = self._data - other_tensor
            return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def __rsub__(self, other: tf.Tensor | builtins.float) -> DimTensor:
        """Right subtract."""
        if not self.is_dimensionless:
            raise DimensionError(
                f"Cannot subtract quantity with dimension {self.dimension} from dimensionless number"
            )
        other_tensor = other if isinstance(other, tf.Tensor) else tf.constant(other, dtype=self._data.dtype)
        new_tensor = other_tensor - self._data
        return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def __mul__(self, other: DimTensor | tf.Tensor | builtins.float) -> DimTensor:
        """Multiply tensors (dimensions multiply)."""
        if isinstance(other, DimTensor):
            new_unit = self._unit * other._unit
            new_tensor = self._data * other._data
            return DimTensor._from_tensor_and_unit(new_tensor, new_unit)
        else:
            other_tensor = other if isinstance(other, tf.Tensor) else tf.constant(other, dtype=self._data.dtype)
            new_tensor = self._data * other_tensor
            return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def __rmul__(self, other: tf.Tensor | builtins.float) -> DimTensor:
        """Right multiply."""
        return self.__mul__(other)

    def __truediv__(self, other: DimTensor | tf.Tensor | builtins.float) -> DimTensor:
        """Divide tensors (dimensions divide)."""
        if isinstance(other, DimTensor):
            new_unit = self._unit / other._unit
            new_tensor = self._data / other._data
            return DimTensor._from_tensor_and_unit(new_tensor, new_unit)
        else:
            other_tensor = other if isinstance(other, tf.Tensor) else tf.constant(other, dtype=self._data.dtype)
            new_tensor = self._data / other_tensor
            return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def __rtruediv__(self, other: tf.Tensor | builtins.float) -> DimTensor:
        """Right divide."""
        new_unit = Unit(
            f"1/{self._unit.symbol}",
            self._unit.dimension ** -1,
            1.0 / self._unit.scale,
        )
        other_tensor = other if isinstance(other, tf.Tensor) else tf.constant(other, dtype=self._data.dtype)
        new_tensor = other_tensor / self._data
        return DimTensor._from_tensor_and_unit(new_tensor, new_unit)

    def __pow__(self, power: int | builtins.float) -> DimTensor:
        """Raise to a power."""
        new_unit = self._unit ** power
        new_tensor = tf.pow(self._data, power)
        return DimTensor._from_tensor_and_unit(new_tensor, new_unit)

    def __neg__(self) -> DimTensor:
        """Negate values."""
        return DimTensor._from_tensor_and_unit(-self._data, self._unit)

    def __pos__(self) -> DimTensor:
        """Unary positive."""
        return DimTensor._from_tensor_and_unit(tf.identity(self._data), self._unit)

    def __abs__(self) -> DimTensor:
        """Absolute value."""
        return DimTensor._from_tensor_and_unit(tf.abs(self._data), self._unit)

    def sqrt(self) -> DimTensor:
        """Square root (dimension exponents halve)."""
        return self ** 0.5

    # =========================================================================
    # Comparison operations
    # =========================================================================

    def __eq__(self, other: object) -> tf.Tensor | bool:  # type: ignore[override]
        """Element-wise equality."""
        if isinstance(other, DimTensor):
            if self.dimension != other.dimension:
                return False
            other_converted = other.to_unit(self._unit)
            return tf.equal(self._data, other_converted._data)
        elif self.is_dimensionless:
            other_tensor = other if isinstance(other, tf.Tensor) else tf.constant(other, dtype=self._data.dtype)
            return tf.equal(self._data, other_tensor)
        return False

    def __lt__(self, other: DimTensor | tf.Tensor | builtins.float) -> tf.Tensor:
        """Element-wise less than."""
        if isinstance(other, DimTensor):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to_unit(self._unit)
            return tf.less(self._data, other_converted._data)
        elif self.is_dimensionless:
            other_tensor = other if isinstance(other, tf.Tensor) else tf.constant(other, dtype=self._data.dtype)
            return tf.less(self._data, other_tensor)
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    def __le__(self, other: DimTensor | tf.Tensor | builtins.float) -> tf.Tensor:
        """Element-wise less than or equal."""
        if isinstance(other, DimTensor):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to_unit(self._unit)
            return tf.less_equal(self._data, other_converted._data)
        elif self.is_dimensionless:
            other_tensor = other if isinstance(other, tf.Tensor) else tf.constant(other, dtype=self._data.dtype)
            return tf.less_equal(self._data, other_tensor)
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    def __gt__(self, other: DimTensor | tf.Tensor | builtins.float) -> tf.Tensor:
        """Element-wise greater than."""
        if isinstance(other, DimTensor):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to_unit(self._unit)
            return tf.greater(self._data, other_converted._data)
        elif self.is_dimensionless:
            other_tensor = other if isinstance(other, tf.Tensor) else tf.constant(other, dtype=self._data.dtype)
            return tf.greater(self._data, other_tensor)
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    def __ge__(self, other: DimTensor | tf.Tensor | builtins.float) -> tf.Tensor:
        """Element-wise greater than or equal."""
        if isinstance(other, DimTensor):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(
                    self.dimension, other.dimension, "compare"
                )
            other_converted = other.to_unit(self._unit)
            return tf.greater_equal(self._data, other_converted._data)
        elif self.is_dimensionless:
            other_tensor = other if isinstance(other, tf.Tensor) else tf.constant(other, dtype=self._data.dtype)
            return tf.greater_equal(self._data, other_tensor)
        raise DimensionError(
            f"Cannot compare quantity with dimension {self.dimension} to dimensionless number"
        )

    # =========================================================================
    # Indexing
    # =========================================================================

    def __getitem__(self, key: Any) -> DimTensor:
        """Index into the tensor, preserving units."""
        result = self._data[key]
        return DimTensor._from_tensor_and_unit(result, self._unit)

    def __len__(self) -> int:
        """Length of first dimension."""
        return int(self._data.shape[0])

    # =========================================================================
    # Reduction operations
    # =========================================================================

    def sum(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> DimTensor:
        """Sum of tensor elements."""
        result = tf.reduce_sum(self._data, axis=axis, keepdims=keepdims)
        return DimTensor._from_tensor_and_unit(result, self._unit)

    def mean(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> DimTensor:
        """Mean of tensor elements."""
        result = tf.reduce_mean(self._data, axis=axis, keepdims=keepdims)
        return DimTensor._from_tensor_and_unit(result, self._unit)

    def std(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> DimTensor:
        """Standard deviation of tensor elements."""
        result = tf.math.reduce_std(self._data, axis=axis, keepdims=keepdims)
        return DimTensor._from_tensor_and_unit(result, self._unit)

    def var(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> DimTensor:
        """Variance of tensor elements (squared units)."""
        result = tf.math.reduce_variance(self._data, axis=axis, keepdims=keepdims)
        new_unit = self._unit ** 2
        return DimTensor._from_tensor_and_unit(result, new_unit)

    def min(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> DimTensor:
        """Minimum value."""
        result = tf.reduce_min(self._data, axis=axis, keepdims=keepdims)
        return DimTensor._from_tensor_and_unit(result, self._unit)

    def max(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> DimTensor:
        """Maximum value."""
        result = tf.reduce_max(self._data, axis=axis, keepdims=keepdims)
        return DimTensor._from_tensor_and_unit(result, self._unit)

    def norm(
        self,
        ord: str | int | float = "euclidean",
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> DimTensor:
        """Vector or matrix norm (preserves units for Euclidean norm)."""
        result = tf.norm(self._data, ord=ord, axis=axis, keepdims=keepdims)
        return DimTensor._from_tensor_and_unit(result, self._unit)

    # =========================================================================
    # Reshaping operations
    # =========================================================================

    def reshape(self, shape: Sequence[int]) -> DimTensor:
        """Reshape tensor preserving units."""
        new_tensor = tf.reshape(self._data, shape)
        return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def transpose(self, perm: Sequence[int] | None = None) -> DimTensor:
        """Transpose tensor."""
        new_tensor = tf.transpose(self._data, perm=perm)
        return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def flatten(self) -> DimTensor:
        """Flatten to 1D tensor."""
        new_tensor = tf.reshape(self._data, [-1])
        return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def squeeze(self, axis: int | Sequence[int] | None = None) -> DimTensor:
        """Remove size-1 dimensions."""
        new_tensor = tf.squeeze(self._data, axis=axis)
        return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    def expand_dims(self, axis: int) -> DimTensor:
        """Add size-1 dimension."""
        new_tensor = tf.expand_dims(self._data, axis=axis)
        return DimTensor._from_tensor_and_unit(new_tensor, self._unit)

    # =========================================================================
    # Linear algebra
    # =========================================================================

    def matmul(self, other: DimTensor) -> DimTensor:
        """Matrix multiplication (dimensions multiply)."""
        new_unit = self._unit * other._unit
        new_tensor = tf.matmul(self._data, other._data)
        return DimTensor._from_tensor_and_unit(new_tensor, new_unit)

    def __matmul__(self, other: DimTensor) -> DimTensor:
        """Matrix multiplication operator @."""
        return self.matmul(other)

    def dot(self, other: DimTensor) -> DimTensor:
        """Dot product (dimensions multiply)."""
        new_unit = self._unit * other._unit
        new_tensor = tf.tensordot(self._data, other._data, axes=1)
        return DimTensor._from_tensor_and_unit(new_tensor, new_unit)

    # =========================================================================
    # String representations
    # =========================================================================

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"DimTensor({self._data.numpy()!r}, unit={self._unit.symbol!r})"

    def __str__(self) -> str:
        """Human-readable string."""
        if self.is_dimensionless:
            return str(self._data.numpy())
        simplified = self._unit.simplified()
        return f"{self._data.numpy()} {simplified.symbol}"

    # =========================================================================
    # Conversion
    # =========================================================================

    def numpy(self) -> Any:
        """Convert to numpy array (loses unit information)."""
        return self._data.numpy()

    def item(self) -> builtins.float:
        """Get single-element tensor as Python scalar."""
        return float(self._data.numpy())

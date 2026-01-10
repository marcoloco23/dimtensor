"""DimVariable: TensorFlow variables with dimensional awareness.

DimVariable wraps tf.Variable with unit tracking, providing mutable
tensors with physical dimensions for trainable parameters.
"""

from __future__ import annotations

import builtins
from typing import Any, Sequence, Union

import tensorflow as tf

from ..core.dimensions import Dimension
from ..core.units import Unit, dimensionless
from ..errors import DimensionError, UnitConversionError
from .dimtensor import DimTensor


class DimVariable:
    """A TensorFlow variable with attached physical units.

    DimVariable wraps a tf.Variable and tracks its physical dimensions.
    Supports in-place assignment operations and gradient computation
    through GradientTape.

    Unlike DimTensor, DimVariable is mutable and can be updated in-place
    using assign(), assign_add(), and assign_sub() methods.

    Examples:
        >>> import tensorflow as tf
        >>> from dimtensor.tensorflow import DimVariable
        >>> from dimtensor import units
        >>>
        >>> # Create trainable variables
        >>> position = DimVariable([0.0, 0.0, 0.0], units.m, name='position')
        >>> velocity = DimVariable([1.0, 2.0, 3.0], units.m / units.s)
        >>>
        >>> # In-place updates
        >>> position.assign([1.0, 2.0, 3.0])
        >>> position.assign_add([0.1, 0.1, 0.1])
        >>>
        >>> # Gradient computation
        >>> with tf.GradientTape() as tape:
        ...     energy = 0.5 * mass * velocity**2
        ...     loss = energy.sum()
        >>> grads = tape.gradient(loss.data, [velocity.data])
    """

    __slots__ = ("_variable", "_unit")

    def __init__(
        self,
        initial_value: Union[tf.Tensor, Sequence[float], float, DimTensor],
        unit: Unit | None = None,
        dtype: tf.DType | None = None,
        name: str | None = None,
        trainable: bool = True,
    ) -> None:
        """Create a DimVariable.

        Args:
            initial_value: Initial value (tf.Tensor, list, scalar, or DimTensor).
            unit: Physical unit. If None, assumes dimensionless.
            dtype: TensorFlow dtype (float32, float64, etc.).
            name: Variable name for debugging and checkpointing.
            trainable: Whether variable is trainable (default True).
        """
        if isinstance(initial_value, DimVariable):
            tensor = initial_value._variable.value()
            unit = unit if unit is not None else initial_value._unit
        elif isinstance(initial_value, DimTensor):
            tensor = initial_value.data
            unit = unit if unit is not None else initial_value.unit
        elif isinstance(initial_value, tf.Tensor):
            tensor = initial_value
        else:
            tensor = tf.constant(initial_value)

        if dtype is not None:
            tensor = tf.cast(tensor, dtype)

        self._variable: tf.Variable = tf.Variable(
            tensor, name=name, trainable=trainable
        )
        self._unit: Unit = unit if unit is not None else dimensionless

    @classmethod
    def _from_variable_and_unit(cls, variable: tf.Variable, unit: Unit) -> DimVariable:
        """Internal constructor that wraps existing variable."""
        result = object.__new__(cls)
        result._variable = variable
        result._unit = unit
        return result

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def data(self) -> tf.Tensor:
        """The underlying tf.Tensor value."""
        return self._variable.value()

    @property
    def variable(self) -> tf.Variable:
        """The underlying tf.Variable."""
        return self._variable

    @property
    def unit(self) -> Unit:
        """The physical unit of this variable."""
        return self._unit

    @property
    def dimension(self) -> Dimension:
        """The physical dimension of this variable."""
        return self._unit.dimension

    @property
    def shape(self) -> tf.TensorShape:
        """Shape of the underlying variable."""
        return self._variable.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._variable.shape)

    @property
    def size(self) -> int:
        """Total number of elements."""
        return int(tf.size(self._variable))

    @property
    def dtype(self) -> tf.DType:
        """Data type of the underlying variable."""
        return self._variable.dtype

    @property
    def device(self) -> str:
        """Device the variable is on."""
        return self._variable.device

    @property
    def name(self) -> str:
        """Name of the variable."""
        return self._variable.name

    @property
    def trainable(self) -> bool:
        """Whether the variable is trainable."""
        return self._variable.trainable

    @property
    def is_dimensionless(self) -> bool:
        """Check if this variable is dimensionless."""
        return self._unit.dimension.is_dimensionless

    # =========================================================================
    # Assignment operations (mutating)
    # =========================================================================

    def assign(
        self,
        value: Union[tf.Tensor, Sequence[float], float, DimTensor, "DimVariable"],
    ) -> DimVariable:
        """Assign a new value to this variable.

        Args:
            value: New value (must be compatible with shape).
                   If DimTensor/DimVariable, dimensions must match.

        Returns:
            self, for chaining.

        Raises:
            DimensionError: If value has incompatible dimension.
        """
        if isinstance(value, (DimTensor, DimVariable)):
            if value.dimension != self.dimension:
                raise DimensionError.incompatible(
                    self.dimension, value.dimension, "assign"
                )
            # Convert to same unit
            if isinstance(value, DimTensor):
                converted = value.to_unit(self._unit)
                tensor = converted.data
            else:
                converted_tensor = value.to_unit(self._unit)
                tensor = converted_tensor.data
        elif isinstance(value, tf.Tensor):
            tensor = value
        else:
            tensor = tf.constant(value, dtype=self._variable.dtype)

        self._variable.assign(tensor)
        return self

    def assign_add(
        self,
        value: Union[tf.Tensor, Sequence[float], float, DimTensor, "DimVariable"],
    ) -> DimVariable:
        """Add value to this variable in-place.

        Args:
            value: Value to add (must have compatible dimension).

        Returns:
            self, for chaining.

        Raises:
            DimensionError: If value has incompatible dimension.
        """
        if isinstance(value, (DimTensor, DimVariable)):
            if value.dimension != self.dimension:
                raise DimensionError.incompatible(
                    self.dimension, value.dimension, "add"
                )
            if isinstance(value, DimTensor):
                converted = value.to_unit(self._unit)
                tensor = converted.data
            else:
                converted_tensor = value.to_unit(self._unit)
                tensor = converted_tensor.data
        elif isinstance(value, tf.Tensor):
            tensor = value
        else:
            tensor = tf.constant(value, dtype=self._variable.dtype)

        self._variable.assign_add(tensor)
        return self

    def assign_sub(
        self,
        value: Union[tf.Tensor, Sequence[float], float, DimTensor, "DimVariable"],
    ) -> DimVariable:
        """Subtract value from this variable in-place.

        Args:
            value: Value to subtract (must have compatible dimension).

        Returns:
            self, for chaining.

        Raises:
            DimensionError: If value has incompatible dimension.
        """
        if isinstance(value, (DimTensor, DimVariable)):
            if value.dimension != self.dimension:
                raise DimensionError.incompatible(
                    self.dimension, value.dimension, "subtract"
                )
            if isinstance(value, DimTensor):
                converted = value.to_unit(self._unit)
                tensor = converted.data
            else:
                converted_tensor = value.to_unit(self._unit)
                tensor = converted_tensor.data
        elif isinstance(value, tf.Tensor):
            tensor = value
        else:
            tensor = tf.constant(value, dtype=self._variable.dtype)

        self._variable.assign_sub(tensor)
        return self

    # =========================================================================
    # Unit conversion
    # =========================================================================

    def to_unit(self, unit: Unit) -> DimTensor:
        """Convert to a different unit with the same dimension.

        Note: Returns a DimTensor, not a DimVariable, since this creates
        a new value rather than mutating in-place.

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
        new_tensor = self._variable.value() * factor
        return DimTensor._from_tensor_and_unit(new_tensor, unit)

    def magnitude(self) -> tf.Tensor:
        """Return the numerical magnitude (stripping units).

        Use with caution - this loses dimensional safety.
        """
        return tf.identity(self._variable.value())

    # =========================================================================
    # Conversion to DimTensor
    # =========================================================================

    def to_tensor(self) -> DimTensor:
        """Convert to a DimTensor (read-only snapshot).

        Returns:
            DimTensor with current value and unit.
        """
        return DimTensor._from_tensor_and_unit(
            tf.identity(self._variable.value()), self._unit
        )

    # =========================================================================
    # Arithmetic operations (return DimTensor)
    # =========================================================================

    def __add__(self, other: DimVariable | DimTensor | tf.Tensor | builtins.float) -> DimTensor:
        """Add (returns DimTensor)."""
        return self.to_tensor() + other

    def __radd__(self, other: tf.Tensor | builtins.float) -> DimTensor:
        """Right add."""
        return self.to_tensor() + other

    def __sub__(self, other: DimVariable | DimTensor | tf.Tensor | builtins.float) -> DimTensor:
        """Subtract (returns DimTensor)."""
        return self.to_tensor() - other

    def __rsub__(self, other: tf.Tensor | builtins.float) -> DimTensor:
        """Right subtract."""
        tensor = self.to_tensor()
        if not tensor.is_dimensionless:
            raise DimensionError(
                f"Cannot subtract quantity with dimension {tensor.dimension} from dimensionless number"
            )
        other_tensor = other if isinstance(other, tf.Tensor) else tf.constant(other, dtype=self._variable.dtype)
        new_tensor = other_tensor - tensor.data
        return DimTensor._from_tensor_and_unit(new_tensor, tensor.unit)

    def __mul__(self, other: DimVariable | DimTensor | tf.Tensor | builtins.float) -> DimTensor:
        """Multiply (returns DimTensor)."""
        return self.to_tensor() * other

    def __rmul__(self, other: tf.Tensor | builtins.float) -> DimTensor:
        """Right multiply."""
        return self.to_tensor() * other

    def __truediv__(self, other: DimVariable | DimTensor | tf.Tensor | builtins.float) -> DimTensor:
        """Divide (returns DimTensor)."""
        return self.to_tensor() / other

    def __rtruediv__(self, other: tf.Tensor | builtins.float) -> DimTensor:
        """Right divide."""
        return other / self.to_tensor()

    def __pow__(self, power: int | builtins.float) -> DimTensor:
        """Raise to a power (returns DimTensor)."""
        return self.to_tensor() ** power

    def __neg__(self) -> DimTensor:
        """Negate values."""
        return -self.to_tensor()

    def __pos__(self) -> DimTensor:
        """Unary positive."""
        return +self.to_tensor()

    def __abs__(self) -> DimTensor:
        """Absolute value."""
        return abs(self.to_tensor())

    def sqrt(self) -> DimTensor:
        """Square root."""
        return self.to_tensor().sqrt()

    # =========================================================================
    # Comparison operations
    # =========================================================================

    def __eq__(self, other: object) -> tf.Tensor | bool:  # type: ignore[override]
        """Element-wise equality."""
        return self.to_tensor() == other

    def __lt__(self, other: DimVariable | DimTensor | tf.Tensor | builtins.float) -> tf.Tensor:
        """Element-wise less than."""
        return self.to_tensor() < other

    def __le__(self, other: DimVariable | DimTensor | tf.Tensor | builtins.float) -> tf.Tensor:
        """Element-wise less than or equal."""
        return self.to_tensor() <= other

    def __gt__(self, other: DimVariable | DimTensor | tf.Tensor | builtins.float) -> tf.Tensor:
        """Element-wise greater than."""
        return self.to_tensor() > other

    def __ge__(self, other: DimVariable | DimTensor | tf.Tensor | builtins.float) -> tf.Tensor:
        """Element-wise greater than or equal."""
        return self.to_tensor() >= other

    # =========================================================================
    # Indexing
    # =========================================================================

    def __getitem__(self, key: Any) -> DimTensor:
        """Index into the variable, returning DimTensor."""
        result = self._variable.value()[key]
        return DimTensor._from_tensor_and_unit(result, self._unit)

    def __len__(self) -> int:
        """Length of first dimension."""
        return int(self._variable.shape[0])

    # =========================================================================
    # Reduction operations (return DimTensor)
    # =========================================================================

    def sum(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> DimTensor:
        """Sum of elements."""
        return self.to_tensor().sum(axis=axis, keepdims=keepdims)

    def mean(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> DimTensor:
        """Mean of elements."""
        return self.to_tensor().mean(axis=axis, keepdims=keepdims)

    def std(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> DimTensor:
        """Standard deviation."""
        return self.to_tensor().std(axis=axis, keepdims=keepdims)

    def var(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> DimTensor:
        """Variance (squared units)."""
        return self.to_tensor().var(axis=axis, keepdims=keepdims)

    def min(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> DimTensor:
        """Minimum value."""
        return self.to_tensor().min(axis=axis, keepdims=keepdims)

    def max(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> DimTensor:
        """Maximum value."""
        return self.to_tensor().max(axis=axis, keepdims=keepdims)

    # =========================================================================
    # Reshaping operations (return DimTensor)
    # =========================================================================

    def reshape(self, shape: Sequence[int]) -> DimTensor:
        """Reshape (returns DimTensor)."""
        return self.to_tensor().reshape(shape)

    def transpose(self, perm: Sequence[int] | None = None) -> DimTensor:
        """Transpose (returns DimTensor)."""
        return self.to_tensor().transpose(perm)

    def flatten(self) -> DimTensor:
        """Flatten (returns DimTensor)."""
        return self.to_tensor().flatten()

    def squeeze(self, axis: int | Sequence[int] | None = None) -> DimTensor:
        """Remove size-1 dimensions (returns DimTensor)."""
        return self.to_tensor().squeeze(axis)

    def expand_dims(self, axis: int) -> DimTensor:
        """Add size-1 dimension (returns DimTensor)."""
        return self.to_tensor().expand_dims(axis)

    # =========================================================================
    # Linear algebra (return DimTensor)
    # =========================================================================

    def matmul(self, other: DimVariable | DimTensor) -> DimTensor:
        """Matrix multiplication."""
        if isinstance(other, DimVariable):
            other = other.to_tensor()
        return self.to_tensor().matmul(other)

    def __matmul__(self, other: DimVariable | DimTensor) -> DimTensor:
        """Matrix multiplication operator @."""
        return self.matmul(other)

    def dot(self, other: DimVariable | DimTensor) -> DimTensor:
        """Dot product."""
        if isinstance(other, DimVariable):
            other = other.to_tensor()
        return self.to_tensor().dot(other)

    # =========================================================================
    # String representations
    # =========================================================================

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"DimVariable({self._variable.numpy()!r}, "
            f"unit={self._unit.symbol!r}, "
            f"name={self.name!r})"
        )

    def __str__(self) -> str:
        """Human-readable string."""
        if self.is_dimensionless:
            return str(self._variable.numpy())
        simplified = self._unit.simplified()
        return f"{self._variable.numpy()} {simplified.symbol}"

    # =========================================================================
    # Conversion
    # =========================================================================

    def numpy(self) -> Any:
        """Convert to numpy array (loses unit information)."""
        return self._variable.numpy()

    def item(self) -> builtins.float:
        """Get single-element variable as Python scalar."""
        return float(self._variable.numpy())

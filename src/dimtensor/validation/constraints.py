"""Value constraints for DimArray.

Constraints enforce conditions on array values, catching physics errors
like negative mass or probability outside [0, 1].

Usage:
    >>> from dimtensor import DimArray, units
    >>> from dimtensor.validation import Positive, Bounded
    >>>
    >>> # Create with constraints
    >>> mass = DimArray([1.0, 2.0], units.kg, constraints=[Positive()])
    >>> prob = DimArray([0.5], units.dimensionless, constraints=[Bounded(0, 1)])
    >>>
    >>> # Validate manually
    >>> mass.validate()  # OK
    >>> mass = DimArray([-1.0], units.kg, constraints=[Positive()])  # Raises!
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..errors import ConstraintError

if TYPE_CHECKING:
    pass


class Constraint(ABC):
    """Base class for value constraints.

    Subclasses must implement:
    - `check(data)`: Return boolean array of valid values
    - `name`: Human-readable constraint name
    - `description`: Description of what values are valid
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this constraint."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of valid values for this constraint."""
        ...

    @abstractmethod
    def check(self, data: NDArray[Any]) -> NDArray[np.bool_]:
        """Check which values satisfy the constraint.

        Args:
            data: Array of values to check.

        Returns:
            Boolean array where True means the value is valid.
        """
        ...

    def is_satisfied(self, data: NDArray[Any]) -> bool:
        """Check if all values satisfy the constraint.

        Args:
            data: Array of values to check.

        Returns:
            True if all values satisfy the constraint.
        """
        return bool(np.all(self.check(data)))

    def validate(self, data: NDArray[Any]) -> None:
        """Validate all values, raising ConstraintError if any fail.

        Args:
            data: Array of values to check.

        Raises:
            ConstraintError: If any values violate the constraint.
        """
        valid = self.check(data)
        if not np.all(valid):
            invalid_indices = np.where(~valid)[0].tolist()
            invalid_values = data.flat[invalid_indices[:5]]  # Show first 5
            raise ConstraintError(
                f"Constraint '{self.name}' violated: {self.description}. "
                f"Invalid values at indices {invalid_indices[:10]}{'...' if len(invalid_indices) > 10 else ''}: "
                f"{invalid_values.tolist()}",
                constraint=self.name,
                indices=invalid_indices,
            )

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}()"


class Positive(Constraint):
    """Constraint requiring all values > 0.

    Use for quantities that must be strictly positive:
    - Mass
    - Temperature in Kelvin
    - Time intervals
    - Distance/length

    Note: NaN and -inf fail this constraint. +inf passes.
    """

    @property
    def name(self) -> str:
        return "Positive"

    @property
    def description(self) -> str:
        return "values must be > 0"

    def check(self, data: NDArray[Any]) -> NDArray[np.bool_]:
        """Check that all values are strictly positive."""
        return data > 0


class NonNegative(Constraint):
    """Constraint requiring all values >= 0.

    Use for quantities that can be zero but not negative:
    - Counts
    - Absolute values
    - Magnitudes

    Note: NaN and -inf fail this constraint. +inf and 0 pass.
    """

    @property
    def name(self) -> str:
        return "NonNegative"

    @property
    def description(self) -> str:
        return "values must be >= 0"

    def check(self, data: NDArray[Any]) -> NDArray[np.bool_]:
        """Check that all values are non-negative."""
        return data >= 0


class NonZero(Constraint):
    """Constraint requiring all values != 0.

    Use for quantities that cannot be zero:
    - Divisors
    - Non-trivial quantities

    Note: NaN fails this constraint (NaN != 0 is NaN, which is falsy).
    """

    @property
    def name(self) -> str:
        return "NonZero"

    @property
    def description(self) -> str:
        return "values must be != 0"

    def check(self, data: NDArray[Any]) -> NDArray[np.bool_]:
        """Check that no values are zero."""
        # Use logical not of (data == 0) to handle NaN correctly
        result: NDArray[np.bool_] = ~np.isclose(data, 0, atol=0, rtol=0) & ~np.isnan(data)
        return result


class Bounded(Constraint):
    """Constraint requiring all values in [min, max].

    Use for quantities with physical bounds:
    - Probability: Bounded(0, 1)
    - Efficiency: Bounded(0, 1)
    - Angle in degrees: Bounded(0, 360)

    Note: Bounds are inclusive. NaN and inf fail unless explicitly included.

    Args:
        min_val: Minimum allowed value (inclusive). Use -np.inf for no lower bound.
        max_val: Maximum allowed value (inclusive). Use np.inf for no upper bound.
    """

    def __init__(self, min_val: float, max_val: float) -> None:
        """Create a bounded constraint.

        Args:
            min_val: Minimum allowed value (inclusive).
            max_val: Maximum allowed value (inclusive).

        Raises:
            ValueError: If min_val > max_val.
        """
        if min_val > max_val:
            raise ValueError(f"min_val ({min_val}) must be <= max_val ({max_val})")
        self.min_val = min_val
        self.max_val = max_val

    @property
    def name(self) -> str:
        return "Bounded"

    @property
    def description(self) -> str:
        return f"values must be in [{self.min_val}, {self.max_val}]"

    def check(self, data: NDArray[Any]) -> NDArray[np.bool_]:
        """Check that all values are within bounds."""
        return (data >= self.min_val) & (data <= self.max_val)

    def __repr__(self) -> str:
        """String representation."""
        return f"Bounded({self.min_val}, {self.max_val})"


class Finite(Constraint):
    """Constraint requiring all values to be finite (no inf or NaN).

    Use when infinite values are invalid:
    - Physical measurements
    - Most numerical computations
    """

    @property
    def name(self) -> str:
        return "Finite"

    @property
    def description(self) -> str:
        return "values must be finite (no inf or NaN)"

    def check(self, data: NDArray[Any]) -> NDArray[np.bool_]:
        """Check that all values are finite."""
        result: NDArray[np.bool_] = np.isfinite(data)
        return result


class NotNaN(Constraint):
    """Constraint requiring no NaN values.

    Use when NaN is invalid but inf is acceptable:
    - Computed results where NaN indicates an error
    """

    @property
    def name(self) -> str:
        return "NotNaN"

    @property
    def description(self) -> str:
        return "values must not be NaN"

    def check(self, data: NDArray[Any]) -> NDArray[np.bool_]:
        """Check that no values are NaN."""
        result: NDArray[np.bool_] = ~np.isnan(data)
        return result


def validate_all(data: NDArray[Any], constraints: list[Constraint]) -> None:
    """Validate data against multiple constraints.

    Args:
        data: Array of values to check.
        constraints: List of constraints to check.

    Raises:
        ConstraintError: If any constraint is violated.
    """
    for constraint in constraints:
        constraint.validate(data)


def is_all_satisfied(data: NDArray[Any], constraints: list[Constraint]) -> bool:
    """Check if data satisfies all constraints.

    Args:
        data: Array of values to check.
        constraints: List of constraints to check.

    Returns:
        True if all constraints are satisfied.
    """
    return all(c.is_satisfied(data) for c in constraints)

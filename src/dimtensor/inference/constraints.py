"""Constraint system for dimensional inference.

Defines constraints between variables based on equation operations.
Constraints enforce dimensional consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core.dimensions import Dimension


@dataclass
class Constraint:
    """Base class for dimensional constraints."""

    def check(self, dimensions: dict[str, Dimension]) -> tuple[bool, Optional[str]]:
        """Check if constraint is satisfied.

        Args:
            dimensions: Mapping of variable names to dimensions.

        Returns:
            Tuple of (is_satisfied, error_message).
            error_message is None if satisfied.
        """
        raise NotImplementedError

    def propagate(
        self, known: dict[str, Dimension]
    ) -> dict[str, Dimension]:
        """Attempt to infer unknown dimensions from known ones.

        Args:
            known: Mapping of variable names to known dimensions.

        Returns:
            Mapping of newly inferred dimensions.
        """
        raise NotImplementedError


@dataclass
class EqualityConstraint(Constraint):
    """Constraint that two expressions must have the same dimension.

    Used for: a + b, a - b, a = b
    Requires: dim(var1) = dim(var2)
    """

    var1: str
    var2: str
    operation: str  # '+', '-', '='

    def check(self, dimensions: dict[str, Dimension]) -> tuple[bool, Optional[str]]:
        """Check if both variables have the same dimension."""
        if self.var1 not in dimensions or self.var2 not in dimensions:
            return True, None  # Can't check yet

        dim1 = dimensions[self.var1]
        dim2 = dimensions[self.var2]

        if dim1 == dim2:
            return True, None
        else:
            op_name = {
                '+': 'Addition',
                '-': 'Subtraction',
                '=': 'Equality'
            }.get(self.operation, 'Operation')
            return False, (
                f"{op_name} requires matching dimensions: "
                f"{self.var1} has {dim1}, but {self.var2} has {dim2}"
            )

    def propagate(
        self, known: dict[str, Dimension]
    ) -> dict[str, Dimension]:
        """Propagate dimension from one variable to the other."""
        inferred = {}

        if self.var1 in known and self.var2 not in known:
            inferred[self.var2] = known[self.var1]
        elif self.var2 in known and self.var1 not in known:
            inferred[self.var1] = known[self.var2]

        return inferred


@dataclass
class MultiplicationConstraint(Constraint):
    """Constraint for multiplication: result = left * right.

    Requires: dim(result) = dim(left) * dim(right)
    """

    result: str
    left: str
    right: str

    def check(self, dimensions: dict[str, Dimension]) -> tuple[bool, Optional[str]]:
        """Check if result dimension equals left * right."""
        if (self.result not in dimensions or
            self.left not in dimensions or
            self.right not in dimensions):
            return True, None  # Can't check yet

        expected = dimensions[self.left] * dimensions[self.right]
        actual = dimensions[self.result]

        if expected == actual:
            return True, None
        else:
            return False, (
                f"Multiplication dimension mismatch: "
                f"{self.result} = {self.left} * {self.right} should have "
                f"dimension {expected}, but has {actual}"
            )

    def propagate(
        self, known: dict[str, Dimension]
    ) -> dict[str, Dimension]:
        """Infer unknown dimension from the other two."""
        inferred = {}

        # If we know left and right, we can infer result
        if (self.left in known and self.right in known and
            self.result not in known):
            inferred[self.result] = known[self.left] * known[self.right]

        # If we know result and left, we can infer right
        elif (self.result in known and self.left in known and
              self.right not in known):
            inferred[self.right] = known[self.result] / known[self.left]

        # If we know result and right, we can infer left
        elif (self.result in known and self.right in known and
              self.left not in known):
            inferred[self.left] = known[self.result] / known[self.right]

        return inferred


@dataclass
class DivisionConstraint(Constraint):
    """Constraint for division: result = left / right.

    Requires: dim(result) = dim(left) / dim(right)
    """

    result: str
    left: str
    right: str

    def check(self, dimensions: dict[str, Dimension]) -> tuple[bool, Optional[str]]:
        """Check if result dimension equals left / right."""
        if (self.result not in dimensions or
            self.left not in dimensions or
            self.right not in dimensions):
            return True, None  # Can't check yet

        expected = dimensions[self.left] / dimensions[self.right]
        actual = dimensions[self.result]

        if expected == actual:
            return True, None
        else:
            return False, (
                f"Division dimension mismatch: "
                f"{self.result} = {self.left} / {self.right} should have "
                f"dimension {expected}, but has {actual}"
            )

    def propagate(
        self, known: dict[str, Dimension]
    ) -> dict[str, Dimension]:
        """Infer unknown dimension from the other two."""
        inferred = {}

        # If we know left and right, we can infer result
        if (self.left in known and self.right in known and
            self.result not in known):
            inferred[self.result] = known[self.left] / known[self.right]

        # If we know result and left, we can infer right
        elif (self.result in known and self.left in known and
              self.right not in known):
            inferred[self.right] = known[self.left] / known[self.result]

        # If we know result and right, we can infer left
        elif (self.result in known and self.right in known and
              self.left not in known):
            inferred[self.left] = known[self.result] * known[self.right]

        return inferred


@dataclass
class PowerConstraint(Constraint):
    """Constraint for power: result = base ** exponent.

    Requires:
    - dim(exponent) must be dimensionless
    - dim(result) = dim(base) ** exponent_value
    """

    result: str
    base: str
    exponent: str
    exponent_value: Optional[float] = None  # If known to be a constant

    def check(self, dimensions: dict[str, Dimension]) -> tuple[bool, Optional[str]]:
        """Check power constraint."""
        # First check: exponent must be dimensionless
        if self.exponent in dimensions:
            if not dimensions[self.exponent].is_dimensionless:
                return False, (
                    f"Exponent {self.exponent} must be dimensionless, "
                    f"but has dimension {dimensions[self.exponent]}"
                )

        # Second check: result = base ** exponent_value
        if (self.result in dimensions and self.base in dimensions and
            self.exponent_value is not None):
            expected = dimensions[self.base] ** self.exponent_value
            actual = dimensions[self.result]

            if expected == actual:
                return True, None
            else:
                return False, (
                    f"Power dimension mismatch: "
                    f"{self.result} = {self.base} ** {self.exponent_value} "
                    f"should have dimension {expected}, but has {actual}"
                )

        return True, None  # Can't fully check yet

    def propagate(
        self, known: dict[str, Dimension]
    ) -> dict[str, Dimension]:
        """Infer unknown dimension."""
        inferred = {}

        # Exponent should be dimensionless
        if self.exponent not in known:
            from ..core.dimensions import DIMENSIONLESS
            inferred[self.exponent] = DIMENSIONLESS

        # If we know base and have exponent_value, infer result
        if (self.base in known and self.exponent_value is not None and
            self.result not in known):
            inferred[self.result] = known[self.base] ** self.exponent_value

        # If we know result and have exponent_value, infer base
        elif (self.result in known and self.exponent_value is not None and
              self.base not in known and self.exponent_value != 0):
            inferred[self.base] = known[self.result] ** (1.0 / self.exponent_value)

        return inferred


@dataclass
class DimensionlessConstraint(Constraint):
    """Constraint that a variable must be dimensionless.

    Used for: arguments to sin(), cos(), exp(), etc.
    """

    var: str
    reason: str  # e.g., "argument to sin()"

    def check(self, dimensions: dict[str, Dimension]) -> tuple[bool, Optional[str]]:
        """Check if variable is dimensionless."""
        if self.var not in dimensions:
            return True, None  # Can't check yet

        if dimensions[self.var].is_dimensionless:
            return True, None
        else:
            return False, (
                f"{self.var} must be dimensionless ({self.reason}), "
                f"but has dimension {dimensions[self.var]}"
            )

    def propagate(
        self, known: dict[str, Dimension]
    ) -> dict[str, Dimension]:
        """Set variable to dimensionless if not already known."""
        if self.var not in known:
            from ..core.dimensions import DIMENSIONLESS
            return {self.var: DIMENSIONLESS}
        return {}

"""Custom exceptions for dimtensor."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core.dimensions import Dimension


class DimTensorError(Exception):
    """Base exception for all dimtensor errors."""

    pass


class DimensionError(DimTensorError):
    """Raised when dimensions are incompatible for an operation."""

    def __init__(
        self,
        message: str,
        left: Dimension | None = None,
        right: Dimension | None = None,
        operation: str | None = None,
    ) -> None:
        self.left = left
        self.right = right
        self.operation = operation
        super().__init__(message)

    @classmethod
    def incompatible(
        cls, left: Dimension, right: Dimension, operation: str
    ) -> DimensionError:
        """Create an error for incompatible dimensions."""
        return cls(
            f"Cannot {operation} quantities with dimensions {left} and {right}",
            left=left,
            right=right,
            operation=operation,
        )


class UnitConversionError(DimTensorError):
    """Raised when unit conversion is not possible."""

    def __init__(
        self,
        message: str,
        from_unit: str | None = None,
        to_unit: str | None = None,
    ) -> None:
        self.from_unit = from_unit
        self.to_unit = to_unit
        super().__init__(message)

    @classmethod
    def incompatible(cls, from_unit: str, to_unit: str) -> UnitConversionError:
        """Create an error for incompatible unit conversion."""
        return cls(
            f"Cannot convert from {from_unit} to {to_unit}: incompatible dimensions",
            from_unit=from_unit,
            to_unit=to_unit,
        )


class ConstraintError(DimTensorError):
    """Raised when a value constraint is violated."""

    def __init__(
        self,
        message: str,
        constraint: str | None = None,
        indices: list[int] | None = None,
    ) -> None:
        """Create a constraint error.

        Args:
            message: Error description.
            constraint: Name of the violated constraint.
            indices: Indices of values that violated the constraint.
        """
        self.constraint = constraint
        self.indices = indices
        super().__init__(message)

"""Validation and constraints for dimtensor.

This module provides value constraints for DimArray, allowing you to
enforce physical validity of values.

Usage:
    >>> from dimtensor import DimArray, units
    >>> from dimtensor.validation import Positive, Bounded
    >>>
    >>> # Mass must be positive
    >>> mass = DimArray([1.0, 2.0], units.kg, constraints=[Positive()])
    >>>
    >>> # Probability must be in [0, 1]
    >>> prob = DimArray([0.5], units.dimensionless, constraints=[Bounded(0, 1)])
"""

from __future__ import annotations

from .constraints import (
    Constraint,
    Positive,
    NonNegative,
    NonZero,
    Bounded,
    Finite,
    NotNaN,
    validate_all,
    is_all_satisfied,
)
from .conservation import ConservationTracker

__all__ = [
    # Base class
    "Constraint",
    # Constraint types
    "Positive",
    "NonNegative",
    "NonZero",
    "Bounded",
    "Finite",
    "NotNaN",
    # Utility functions
    "validate_all",
    "is_all_satisfied",
    # Conservation tracking
    "ConservationTracker",
]

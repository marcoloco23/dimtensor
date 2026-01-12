"""Research tools for paper reproduction and validation.

This module provides tools for reproducing physics papers with unit-aware
computations, comparing published vs computed results, and generating
reproducibility reports.

Example:
    >>> from dimtensor.research import Paper, ReproductionResult, compare_values
    >>> from dimtensor import DimArray, units
    >>>
    >>> # Create paper metadata
    >>> paper = Paper(
    ...     title="Measurement of electron charge",
    ...     authors=["R. A. Millikan"],
    ...     doi="10.1103/PhysRev.2.109",
    ...     year=1913,
    ...     published_values={
    ...         "electron_charge": DimArray(1.592e-19, units.C),
    ...     }
    ... )
    >>>
    >>> # Compute value
    >>> computed = DimArray(1.602e-19, units.C)
    >>>
    >>> # Compare
    >>> result = compare_values(
    ...     paper.published_values["electron_charge"],
    ...     computed,
    ...     rtol=0.01
    ... )
    >>> print(f"Match: {result.matches}")
"""

from .paper import Paper
from .reproduction import ReproductionResult, ReproductionStatus
from .comparison import ComparisonResult, compare_values, compare_all
from .reporting import generate_report, to_notebook

__all__ = [
    "Paper",
    "ReproductionResult",
    "ReproductionStatus",
    "ComparisonResult",
    "compare_values",
    "compare_all",
    "generate_report",
    "to_notebook",
]

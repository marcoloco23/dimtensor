"""Dimensional inference for dimtensor.

This module provides tools for inferring physical dimensions from:
- Variable names (heuristics)
- Equation patterns
- Code analysis

Usage:
    >>> from dimtensor.inference import infer_dimension
    >>>
    >>> # Infer from variable name
    >>> dim, confidence = infer_dimension("velocity")
    >>> print(dim)  # L·T⁻¹ (m/s)
    >>> print(confidence)  # 0.9
    >>>
    >>> # Get all matching patterns
    >>> from dimtensor.inference import get_matching_patterns
    >>> patterns = get_matching_patterns("initial_velocity")
    >>> for name, dim, conf in patterns:
    ...     print(f"{name}: {dim} ({conf:.0%})")
"""

from __future__ import annotations

from .heuristics import (
    infer_dimension,
    get_matching_patterns,
    InferenceResult,
    VARIABLE_PATTERNS,
    SUFFIX_PATTERNS,
    PREFIX_PATTERNS,
)
from .equations import (
    Equation,
    EQUATION_DATABASE,
    DOMAINS,
    get_equations_by_domain,
    get_equations_by_tag,
    find_equations_with_variable,
    suggest_dimension_from_equations,
)

__all__ = [
    # Heuristics
    "infer_dimension",
    "get_matching_patterns",
    "InferenceResult",
    "VARIABLE_PATTERNS",
    "SUFFIX_PATTERNS",
    "PREFIX_PATTERNS",
    # Equations
    "Equation",
    "EQUATION_DATABASE",
    "DOMAINS",
    "get_equations_by_domain",
    "get_equations_by_tag",
    "find_equations_with_variable",
    "suggest_dimension_from_equations",
]

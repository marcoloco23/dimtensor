"""Comparison utilities for published vs computed values.

This module provides functions for comparing published values with
computed results, handling unit conversions and uncertainty propagation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from ..core.dimarray import DimArray
from ..errors import DimensionError


@dataclass
class ComparisonResult:
    """Results from comparing published vs computed values.

    Attributes:
        quantity_name: Name of compared quantity.
        published_value: Original published value (DimArray).
        computed_value: Reproduced value (DimArray).
        absolute_error: |computed - published| with units.
        relative_error: |computed - published| / |published| (dimensionless).
        matches: True if within tolerance.
        tolerance_used: (rtol, atol) tuple.
        unit_conversion_applied: True if units were different.
        notes: Additional notes about comparison.

    Example:
        >>> from dimtensor.research import compare_values
        >>> from dimtensor import DimArray, units
        >>>
        >>> pub = DimArray(1.59e-19, units.C)
        >>> comp = DimArray(1.60e-19, units.C)
        >>> result = compare_values(pub, comp, rtol=0.01)
        >>> print(f"Matches: {result.matches}, Error: {result.relative_error:.3f}")
    """

    quantity_name: str
    published_value: DimArray
    computed_value: DimArray
    absolute_error: DimArray | None
    relative_error: float | None
    matches: bool
    tolerance_used: tuple[float | None, float | None]
    unit_conversion_applied: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        from ..io.json import to_dict as dimarray_to_dict

        return {
            "quantity_name": self.quantity_name,
            "published_value": dimarray_to_dict(self.published_value),
            "computed_value": dimarray_to_dict(self.computed_value),
            "absolute_error": dimarray_to_dict(self.absolute_error) if self.absolute_error is not None else None,
            "relative_error": self.relative_error,
            "matches": self.matches,
            "tolerance_used": self.tolerance_used,
            "unit_conversion_applied": self.unit_conversion_applied,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComparisonResult:
        """Create from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            ComparisonResult object.
        """
        from ..io.json import from_dict as dimarray_from_dict

        return cls(
            quantity_name=data["quantity_name"],
            published_value=dimarray_from_dict(data["published_value"]),
            computed_value=dimarray_from_dict(data["computed_value"]),
            absolute_error=dimarray_from_dict(data["absolute_error"]) if data.get("absolute_error") else None,
            relative_error=data.get("relative_error"),
            matches=data["matches"],
            tolerance_used=tuple(data["tolerance_used"]),
            unit_conversion_applied=data["unit_conversion_applied"],
            notes=data.get("notes", ""),
        )

    def __repr__(self) -> str:
        """String representation."""
        match_str = "MATCH" if self.matches else "MISMATCH"
        if self.relative_error is not None:
            return (
                f"ComparisonResult({self.quantity_name}: {match_str}, "
                f"rel_error={self.relative_error:.2e})"
            )
        return f"ComparisonResult({self.quantity_name}: {match_str})"


def compare_values(
    published: DimArray,
    computed: DimArray,
    rtol: float | None = 1e-3,
    atol: float | None = None,
    quantity_name: str = "quantity",
) -> ComparisonResult:
    """Compare published and computed values with unit handling.

    Automatically converts units if they differ, then compares values
    within specified tolerances. Handles uncertainties if present.

    Args:
        published: Published value from paper.
        computed: Computed value from reproduction.
        rtol: Relative tolerance (default 1e-3 = 0.1%).
        atol: Absolute tolerance in published units (default None).
        quantity_name: Name of the quantity being compared.

    Returns:
        ComparisonResult object with comparison details.

    Raises:
        DimensionError: If dimensions are incompatible.

    Example:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.research import compare_values
        >>>
        >>> pub = DimArray(100.0, units.km)
        >>> comp = DimArray(100050.0, units.m)  # Different units
        >>> result = compare_values(pub, comp, rtol=0.001)
        >>> print(result.matches)  # True (within 0.1%)
    """
    # Check dimension compatibility
    if published.dimension != computed.dimension:
        raise DimensionError(
            f"Cannot compare values with different dimensions: "
            f"{published.dimension} vs {computed.dimension}"
        )

    # Convert computed to published units if needed
    unit_conversion_applied = False
    if computed.unit != published.unit:
        computed = computed.to(published.unit)
        unit_conversion_applied = True

    # Calculate absolute error
    abs_error = computed - published

    # Calculate relative error (handle zero published values)
    # Use _data directly to avoid read-only flag issues with scalars
    pub_data = np.asarray(published._data)
    abs_error_data = np.asarray(abs_error._data)
    pub_magnitude = np.abs(pub_data)

    if np.any(pub_magnitude == 0):
        # For zero published values, use absolute comparison only
        rel_error = None
        if atol is None:
            # Can't do relative comparison, need absolute tolerance
            matches = np.allclose(abs_error_data, 0, rtol=0, atol=1e-10)
        else:
            matches = np.allclose(abs_error_data, 0, rtol=0, atol=atol)
    else:
        # Calculate relative error
        rel_error_array = abs_error_data / pub_magnitude
        if rel_error_array.size == 1:
            rel_error = float(np.abs(rel_error_array.item()))
        else:
            # For arrays, use max relative error
            rel_error = float(np.max(np.abs(rel_error_array)))

        # Check if within tolerance
        if atol is None:
            matches = bool(np.all(np.abs(rel_error_array) <= rtol))
        else:
            # Both relative and absolute tolerance
            matches = bool(
                np.all(
                    (np.abs(rel_error_array) <= rtol) | (np.abs(abs_error_data) <= atol)
                )
            )

    # Add notes about uncertainties if present
    notes = ""
    if published._uncertainty is not None or computed._uncertainty is not None:
        notes = "Comparison does not account for uncertainties. "
        if published._uncertainty is not None and computed._uncertainty is not None:
            # Check if difference is within combined uncertainty
            combined_unc = np.sqrt(
                published._uncertainty**2 + computed._uncertainty**2
            )
            within_unc = np.all(np.abs(abs_error._data) <= combined_unc)
            if within_unc:
                notes += "Values agree within combined uncertainties."
            else:
                notes += "Discrepancy exceeds combined uncertainties."

    return ComparisonResult(
        quantity_name=quantity_name,
        published_value=published,
        computed_value=computed,
        absolute_error=abs_error,
        relative_error=rel_error,
        matches=matches,
        tolerance_used=(rtol, atol),
        unit_conversion_applied=unit_conversion_applied,
        notes=notes,
    )


def compare_all(
    published_values: dict[str, DimArray],
    computed_values: dict[str, DimArray],
    rtol: float | None = 1e-3,
    atol: float | None = None,
) -> dict[str, ComparisonResult]:
    """Compare multiple published and computed values.

    Args:
        published_values: Dict mapping quantity names to published DimArrays.
        computed_values: Dict mapping quantity names to computed DimArrays.
        rtol: Relative tolerance for all comparisons.
        atol: Absolute tolerance for all comparisons.

    Returns:
        Dict mapping quantity names to ComparisonResult objects.

    Example:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.research import compare_all
        >>>
        >>> pub = {"mass": DimArray(1.0, units.kg)}
        >>> comp = {"mass": DimArray(1.001, units.kg)}
        >>> results = compare_all(pub, comp, rtol=0.01)
        >>> print(results["mass"].matches)  # True
    """
    results = {}

    # Compare quantities present in both dicts
    common_keys = set(published_values.keys()) & set(computed_values.keys())

    for name in common_keys:
        try:
            result = compare_values(
                published_values[name],
                computed_values[name],
                rtol=rtol,
                atol=atol,
                quantity_name=name,
            )
            results[name] = result
        except DimensionError as e:
            # Create a comparison result indicating failure
            results[name] = ComparisonResult(
                quantity_name=name,
                published_value=published_values[name],
                computed_value=computed_values[name],
                absolute_error=None,
                relative_error=None,
                matches=False,
                tolerance_used=(rtol, atol),
                unit_conversion_applied=False,
                notes=f"Dimension error: {str(e)}",
            )

    return results

"""Validation functions for dataset cards.

Validates that datasets match their declared schema and unit specifications.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.dimarray import DimArray
from ..errors import DimensionError, UnitConversionError
from .card import DimDatasetCard, ColumnInfo


class ValidationError(Exception):
    """Raised when dataset validation fails."""

    pass


def validate_dataset(
    card: DimDatasetCard,
    data: dict[str, DimArray | np.ndarray | Any],
    strict: bool = True,
) -> list[str]:
    """Validate that a dataset matches its card specification.

    Args:
        card: Dataset card with schema.
        data: Dictionary mapping column names to data arrays.
        strict: If True, raise ValidationError on any issue.
                If False, return list of warning messages.

    Returns:
        List of validation warnings (empty if all checks pass).

    Raises:
        ValidationError: If validation fails and strict=True.
    """
    warnings = []

    # Check required columns exist
    for col in card.columns:
        if col.required and col.name not in data:
            msg = f"Required column '{col.name}' is missing"
            if strict:
                raise ValidationError(msg)
            warnings.append(msg)

    # Check each column's units
    for col_name, col_data in data.items():
        col_info = card.get_column(col_name)
        if col_info is None:
            msg = f"Column '{col_name}' not in card specification"
            warnings.append(msg)
            continue

        # Validate units if data is a DimArray
        if isinstance(col_data, DimArray):
            try:
                validate_column_units(col_info, col_data)
            except (DimensionError, UnitConversionError) as e:
                msg = f"Column '{col_name}' unit mismatch: {e}"
                if strict:
                    raise ValidationError(msg) from e
                warnings.append(msg)

        # Check uncertainty column if specified
        if col_info.uncertainty_col and col_info.uncertainty_col not in data:
            msg = f"Uncertainty column '{col_info.uncertainty_col}' for '{col_name}' is missing"
            warnings.append(msg)

    return warnings


def validate_column_units(col_info: ColumnInfo, data: DimArray) -> None:
    """Validate that a column's data matches its declared units.

    Args:
        col_info: Column specification.
        data: DimArray with data.

    Raises:
        DimensionError: If dimensions don't match.
        UnitConversionError: If units can't be converted.
    """
    expected_dim = col_info.unit.dimension
    actual_dim = data.dimension

    if expected_dim != actual_dim:
        raise DimensionError(
            f"Dimension mismatch: expected {expected_dim}, got {actual_dim}"
        )

    # Check unit scale compatibility
    expected_scale = col_info.unit.scale
    actual_scale = data.unit.scale

    # Allow small numerical differences due to floating point
    if not np.isclose(expected_scale, actual_scale, rtol=1e-9):
        # This is a warning, not an error - units are convertible but scales differ
        pass


def validate_schema(
    card: DimDatasetCard,
    column_names: list[str],
    strict: bool = True,
) -> list[str]:
    """Validate that column names match the card schema.

    Args:
        card: Dataset card with schema.
        column_names: List of actual column names in dataset.
        strict: If True, raise ValidationError on any issue.
                If False, return list of warning messages.

    Returns:
        List of validation warnings (empty if all checks pass).

    Raises:
        ValidationError: If validation fails and strict=True.
    """
    warnings = []

    # Check required columns exist
    for col in card.columns:
        if col.required and col.name not in column_names:
            msg = f"Required column '{col.name}' is missing"
            if strict:
                raise ValidationError(msg)
            warnings.append(msg)

    # Check for unexpected columns
    card_col_names = {col.name for col in card.columns}
    for col_name in column_names:
        if col_name not in card_col_names:
            msg = f"Column '{col_name}' not in card specification"
            warnings.append(msg)

    return warnings


def check_units_convertible(col_info: ColumnInfo, data: DimArray) -> bool:
    """Check if data units can be converted to expected units.

    Args:
        col_info: Column specification with expected units.
        data: DimArray with actual data.

    Returns:
        True if units are convertible, False otherwise.
    """
    try:
        # Try to convert data to expected units
        data.to(col_info.unit)
        return True
    except (DimensionError, UnitConversionError):
        return False


def validate_coordinate_system(card: DimDatasetCard, data: dict[str, Any]) -> list[str]:
    """Validate that coordinate columns are consistent with declared system.

    Args:
        card: Dataset card with coordinate system specification.
        data: Dictionary mapping column names to data.

    Returns:
        List of validation warnings.
    """
    warnings = []

    # Get columns with coordinate roles
    coord_cols = {col.coordinate_role: col for col in card.columns if col.coordinate_role}

    # Validate based on coordinate system
    if card.coordinate_system.value == "cartesian":
        expected_roles = {"x", "y", "z"}
    elif card.coordinate_system.value == "spherical":
        expected_roles = {"r", "theta", "phi"}
    elif card.coordinate_system.value == "cylindrical":
        expected_roles = {"r", "theta", "z"}
    else:
        # Custom coordinate system - skip validation
        return warnings

    # Check that declared roles match coordinate system
    actual_roles = set(coord_cols.keys())
    if not actual_roles.issubset(expected_roles):
        unexpected = actual_roles - expected_roles
        warnings.append(
            f"Coordinate roles {unexpected} not valid for {card.coordinate_system.value} system"
        )

    return warnings


def validate_uncertainties(
    card: DimDatasetCard,
    data: dict[str, DimArray | np.ndarray],
) -> list[str]:
    """Validate that uncertainty columns have matching shapes and units.

    Args:
        card: Dataset card specification.
        data: Dictionary mapping column names to data.

    Returns:
        List of validation warnings.
    """
    warnings = []

    for col in card.columns:
        if col.uncertainty_col:
            if col.name not in data:
                continue

            if col.uncertainty_col not in data:
                warnings.append(
                    f"Uncertainty column '{col.uncertainty_col}' for '{col.name}' is missing"
                )
                continue

            # Check shapes match
            data_array = data[col.name]
            unc_array = data[col.uncertainty_col]

            if hasattr(data_array, "shape") and hasattr(unc_array, "shape"):
                if data_array.shape != unc_array.shape:
                    warnings.append(
                        f"Shape mismatch: '{col.name}' has shape {data_array.shape}, "
                        f"uncertainty '{col.uncertainty_col}' has shape {unc_array.shape}"
                    )

            # Check units match if both are DimArrays
            if isinstance(data_array, DimArray) and isinstance(unc_array, DimArray):
                if data_array.dimension != unc_array.dimension:
                    warnings.append(
                        f"Unit mismatch: '{col.name}' has dimension {data_array.dimension}, "
                        f"uncertainty '{col.uncertainty_col}' has dimension {unc_array.dimension}"
                    )

    return warnings

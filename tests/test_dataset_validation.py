"""Tests for dataset validation."""

import numpy as np
import pytest

from dimtensor import DimArray, units
from dimtensor.datasets import (
    DimDatasetCard,
    ValidationError,
    validate_dataset,
    validate_column_units,
    validate_schema,
    check_units_convertible,
    validate_coordinate_system,
    validate_uncertainties,
    CoordinateSystem,
)


@pytest.fixture
def simple_card():
    """Create a simple dataset card for testing."""
    card = DimDatasetCard(name="test_data")
    card.add_column("position", units.m, "Position in meters", required=True)
    card.add_column("velocity", units.m / units.s, "Velocity", required=True)
    card.add_column("mass", units.kg, "Mass", required=False)
    return card


def test_validate_dataset_success(simple_card):
    """Test successful dataset validation."""
    data = {
        "position": DimArray([1.0, 2.0, 3.0], units.m),
        "velocity": DimArray([0.5, 1.0, 1.5], units.m / units.s),
        "mass": DimArray([10.0, 10.0, 10.0], units.kg),
    }

    # Should not raise any errors
    warnings = validate_dataset(simple_card, data, strict=True)
    assert len(warnings) == 0


def test_validate_dataset_missing_required_column_strict(simple_card):
    """Test validation fails when required column is missing (strict mode)."""
    data = {
        "position": DimArray([1.0, 2.0], units.m),
        # Missing 'velocity' (required)
    }

    with pytest.raises(ValidationError, match="Required column 'velocity' is missing"):
        validate_dataset(simple_card, data, strict=True)


def test_validate_dataset_missing_required_column_non_strict(simple_card):
    """Test validation warns when required column is missing (non-strict mode)."""
    data = {
        "position": DimArray([1.0, 2.0], units.m),
        # Missing 'velocity' (required)
    }

    warnings = validate_dataset(simple_card, data, strict=False)
    assert len(warnings) > 0
    assert any("velocity" in w for w in warnings)


def test_validate_dataset_missing_optional_column(simple_card):
    """Test validation succeeds when optional column is missing."""
    data = {
        "position": DimArray([1.0, 2.0], units.m),
        "velocity": DimArray([0.5, 1.0], units.m / units.s),
        # 'mass' is optional, so missing is OK
    }

    warnings = validate_dataset(simple_card, data, strict=True)
    assert len(warnings) == 0


def test_validate_dataset_unexpected_column(simple_card):
    """Test validation warns about unexpected columns."""
    data = {
        "position": DimArray([1.0, 2.0], units.m),
        "velocity": DimArray([0.5, 1.0], units.m / units.s),
        "acceleration": DimArray([0.1, 0.2], units.m / units.s**2),  # Not in card
    }

    warnings = validate_dataset(simple_card, data, strict=False)
    assert any("acceleration" in w and "not in card" in w for w in warnings)


def test_validate_dataset_wrong_units_strict(simple_card):
    """Test validation fails when units don't match (strict mode)."""
    data = {
        "position": DimArray([1.0, 2.0], units.km),  # Wrong unit (should be m)
        "velocity": DimArray([0.5, 1.0], units.m / units.s),
    }

    # Should raise due to unit scale mismatch
    # (dimension is same but scale is different)
    # Actually this should pass since km and m have same dimension
    # Let's use wrong dimension instead
    data = {
        "position": DimArray([1.0, 2.0], units.s),  # Wrong dimension!
        "velocity": DimArray([0.5, 1.0], units.m / units.s),
    }

    with pytest.raises(ValidationError, match="unit mismatch"):
        validate_dataset(simple_card, data, strict=True)


def test_validate_column_units_success():
    """Test successful column unit validation."""
    from dimtensor.datasets.card import ColumnInfo

    col_info = ColumnInfo(name="force", unit=units.N)
    data = DimArray([10.0, 20.0], units.N)

    # Should not raise
    validate_column_units(col_info, data)


def test_validate_column_units_dimension_mismatch():
    """Test column unit validation fails on dimension mismatch."""
    from dimtensor.datasets.card import ColumnInfo
    from dimtensor.errors import DimensionError

    col_info = ColumnInfo(name="force", unit=units.N)
    data = DimArray([10.0, 20.0], units.m)  # Wrong dimension

    with pytest.raises(DimensionError):
        validate_column_units(col_info, data)


def test_validate_schema_success(simple_card):
    """Test successful schema validation."""
    column_names = ["position", "velocity", "mass"]

    warnings = validate_schema(simple_card, column_names, strict=True)
    assert len(warnings) == 0


def test_validate_schema_missing_required(simple_card):
    """Test schema validation fails when required column missing."""
    column_names = ["position"]  # Missing 'velocity'

    with pytest.raises(ValidationError, match="Required column 'velocity'"):
        validate_schema(simple_card, column_names, strict=True)


def test_validate_schema_extra_column(simple_card):
    """Test schema validation warns about extra columns."""
    column_names = ["position", "velocity", "mass", "extra"]

    warnings = validate_schema(simple_card, column_names, strict=False)
    assert any("extra" in w for w in warnings)


def test_check_units_convertible_success():
    """Test checking if units are convertible."""
    from dimtensor.datasets.card import ColumnInfo

    col_info = ColumnInfo(name="distance", unit=units.m)
    data = DimArray([1000.0, 2000.0], units.mm)  # Different scale, same dimension

    # mm can be converted to m
    assert check_units_convertible(col_info, data) is True


def test_check_units_convertible_failure():
    """Test checking units that are not convertible."""
    from dimtensor.datasets.card import ColumnInfo

    col_info = ColumnInfo(name="distance", unit=units.m)
    data = DimArray([10.0, 20.0], units.s)  # Wrong dimension

    assert check_units_convertible(col_info, data) is False


def test_validate_coordinate_system_cartesian():
    """Test Cartesian coordinate system validation."""
    card = DimDatasetCard(
        name="cartesian_data",
        coordinate_system=CoordinateSystem.CARTESIAN,
    )
    card.add_column("x", units.m, coordinate_role="x")
    card.add_column("y", units.m, coordinate_role="y")
    card.add_column("z", units.m, coordinate_role="z")

    data = {
        "x": DimArray([1.0], units.m),
        "y": DimArray([2.0], units.m),
        "z": DimArray([3.0], units.m),
    }

    warnings = validate_coordinate_system(card, data)
    assert len(warnings) == 0


def test_validate_coordinate_system_invalid_role():
    """Test coordinate system validation with invalid role."""
    card = DimDatasetCard(
        name="bad_coords",
        coordinate_system=CoordinateSystem.CARTESIAN,
    )
    card.add_column("x", units.m, coordinate_role="x")
    card.add_column("r", units.m, coordinate_role="r")  # Invalid for Cartesian

    data = {
        "x": DimArray([1.0], units.m),
        "r": DimArray([2.0], units.m),
    }

    warnings = validate_coordinate_system(card, data)
    assert len(warnings) > 0
    assert any("not valid for cartesian" in w.lower() for w in warnings)


def test_validate_coordinate_system_spherical():
    """Test spherical coordinate system validation."""
    card = DimDatasetCard(
        name="spherical_data",
        coordinate_system=CoordinateSystem.SPHERICAL,
    )
    card.add_column("r", units.m, coordinate_role="r")
    card.add_column("theta", units.dimensionless, coordinate_role="theta")
    card.add_column("phi", units.dimensionless, coordinate_role="phi")

    data = {
        "r": DimArray([1.0], units.m),
        "theta": DimArray([0.5], units.dimensionless),
        "phi": DimArray([1.0], units.dimensionless),
    }

    warnings = validate_coordinate_system(card, data)
    assert len(warnings) == 0


def test_validate_coordinate_system_custom():
    """Test custom coordinate system (no validation)."""
    card = DimDatasetCard(
        name="custom_data",
        coordinate_system=CoordinateSystem.CUSTOM,
    )
    card.add_column("weird_coord", units.m, coordinate_role="custom1")

    data = {"weird_coord": DimArray([1.0], units.m)}

    # Custom systems should not produce warnings
    warnings = validate_coordinate_system(card, data)
    assert len(warnings) == 0


def test_validate_uncertainties_success():
    """Test successful uncertainty validation."""
    card = DimDatasetCard(name="measured")
    card.add_column("value", units.m, uncertainty_col="value_unc")
    card.add_column("value_unc", units.m)

    data = {
        "value": DimArray([1.0, 2.0, 3.0], units.m),
        "value_unc": DimArray([0.1, 0.1, 0.1], units.m),
    }

    warnings = validate_uncertainties(card, data)
    assert len(warnings) == 0


def test_validate_uncertainties_missing_column():
    """Test uncertainty validation warns when uncertainty column missing."""
    card = DimDatasetCard(name="measured")
    card.add_column("value", units.m, uncertainty_col="value_unc")

    data = {
        "value": DimArray([1.0, 2.0], units.m),
        # Missing value_unc
    }

    warnings = validate_uncertainties(card, data)
    assert len(warnings) > 0
    assert any("value_unc" in w and "missing" in w for w in warnings)


def test_validate_uncertainties_shape_mismatch():
    """Test uncertainty validation warns on shape mismatch."""
    card = DimDatasetCard(name="measured")
    card.add_column("value", units.m, uncertainty_col="value_unc")
    card.add_column("value_unc", units.m)

    data = {
        "value": DimArray([1.0, 2.0, 3.0], units.m),
        "value_unc": DimArray([0.1, 0.1], units.m),  # Wrong shape
    }

    warnings = validate_uncertainties(card, data)
    assert len(warnings) > 0
    assert any("Shape mismatch" in w for w in warnings)


def test_validate_uncertainties_unit_mismatch():
    """Test uncertainty validation warns on unit mismatch."""
    card = DimDatasetCard(name="measured")
    card.add_column("value", units.m, uncertainty_col="value_unc")
    card.add_column("value_unc", units.m)

    data = {
        "value": DimArray([1.0, 2.0], units.m),
        "value_unc": DimArray([0.1, 0.1], units.s),  # Wrong units
    }

    warnings = validate_uncertainties(card, data)
    assert len(warnings) > 0
    assert any("Unit mismatch" in w for w in warnings)


def test_validate_dataset_with_uncertainties():
    """Test full dataset validation including uncertainties."""
    card = DimDatasetCard(name="experiment")
    card.add_column("temperature", units.K, uncertainty_col="temp_unc")
    card.add_column("temp_unc", units.K, required=True)

    data = {
        "temperature": DimArray([300.0, 301.0], units.K),
        "temp_unc": DimArray([0.1, 0.15], units.K),
    }

    warnings = validate_dataset(card, data, strict=True)
    assert len(warnings) == 0


def test_validate_dataset_mixed_types():
    """Test validation with mixed DimArray and ndarray types."""
    card = DimDatasetCard(name="mixed")
    card.add_column("with_units", units.m)
    card.add_column("without_units", units.dimensionless)

    data = {
        "with_units": DimArray([1.0, 2.0], units.m),
        "without_units": np.array([3.0, 4.0]),  # Plain ndarray
    }

    # Should succeed - plain arrays don't get unit validation
    warnings = validate_dataset(card, data, strict=False)
    # May have warnings about column not being DimArray, but shouldn't error


def test_validation_error_message_helpful():
    """Test that validation errors have helpful messages."""
    card = DimDatasetCard(name="test")
    card.add_column("required_col", units.m, required=True)

    data = {}

    try:
        validate_dataset(card, data, strict=True)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        # Check error message is informative
        assert "required_col" in str(e)
        assert "missing" in str(e).lower()

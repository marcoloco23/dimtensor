"""Tests for dataset cards."""

import json
import tempfile
from pathlib import Path

import pytest

from dimtensor import units
from dimtensor.datasets import (
    DimDatasetCard,
    ColumnInfo,
    CoordinateSystem,
    save_dataset_card,
    load_dataset_card,
)


def test_column_info_creation():
    """Test creating ColumnInfo."""
    col = ColumnInfo(
        name="position",
        unit=units.m,
        description="Position in meters",
        uncertainty_col="position_unc",
        coordinate_role="x",
        required=True,
    )

    assert col.name == "position"
    assert col.unit == units.m
    assert col.description == "Position in meters"
    assert col.uncertainty_col == "position_unc"
    assert col.coordinate_role == "x"
    assert col.required is True


def test_column_info_to_dict():
    """Test ColumnInfo serialization."""
    col = ColumnInfo(
        name="velocity",
        unit=units.m / units.s,
        description="Velocity",
    )

    data = col.to_dict()

    assert data["name"] == "velocity"
    assert "unit" in data
    assert data["unit"]["symbol"] == "m/s"
    assert "dimension" in data["unit"]


def test_column_info_from_dict():
    """Test ColumnInfo deserialization."""
    col = ColumnInfo(
        name="force",
        unit=units.N,
        description="Force in Newtons",
    )

    data = col.to_dict()
    loaded = ColumnInfo.from_dict(data)

    assert loaded.name == col.name
    assert loaded.unit.dimension == col.unit.dimension
    assert loaded.description == col.description


def test_dataset_card_creation():
    """Test creating DimDatasetCard."""
    card = DimDatasetCard(
        name="test_dataset",
        description="A test dataset",
        domain="mechanics",
        coordinate_system=CoordinateSystem.CARTESIAN,
        version="1.0",
        license="MIT",
    )

    assert card.name == "test_dataset"
    assert card.domain == "mechanics"
    assert card.coordinate_system == CoordinateSystem.CARTESIAN
    assert card.version == "1.0"


def test_dataset_card_add_column():
    """Test adding columns to dataset card."""
    card = DimDatasetCard(name="test")

    card.add_column("x", units.m, "X coordinate", coordinate_role="x")
    card.add_column("y", units.m, "Y coordinate", coordinate_role="y")
    card.add_column("v", units.m / units.s, "Velocity")

    assert len(card.columns) == 3
    assert card.get_column("x").coordinate_role == "x"
    assert card.get_column("v").unit.symbol == "m/s"


def test_dataset_card_get_column():
    """Test getting column by name."""
    card = DimDatasetCard(name="test")
    card.add_column("mass", units.kg, "Mass")

    col = card.get_column("mass")
    assert col is not None
    assert col.name == "mass"

    missing = card.get_column("nonexistent")
    assert missing is None


def test_dataset_card_to_dict():
    """Test DimDatasetCard serialization."""
    card = DimDatasetCard(
        name="physics_data",
        description="Physics dataset",
        domain="thermodynamics",
    )
    card.add_column("temperature", units.K, "Temperature in Kelvin")
    card.add_column("pressure", units.Pa, "Pressure in Pascals")

    data = card.to_dict()

    assert data["name"] == "physics_data"
    assert data["domain"] == "thermodynamics"
    assert len(data["columns"]) == 2
    assert data["dimtensor_card_version"] == "1.0"


def test_dataset_card_from_dict():
    """Test DimDatasetCard deserialization."""
    card = DimDatasetCard(name="test_data", domain="mechanics")
    card.add_column("position", units.m, "Position")
    card.add_column("velocity", units.m / units.s, "Velocity")

    data = card.to_dict()
    loaded = DimDatasetCard.from_dict(data)

    assert loaded.name == card.name
    assert loaded.domain == card.domain
    assert len(loaded.columns) == len(card.columns)
    assert loaded.get_column("position") is not None


def test_dataset_card_to_markdown():
    """Test generating markdown from dataset card."""
    card = DimDatasetCard(
        name="Pendulum Data",
        description="Simple pendulum measurements",
        domain="mechanics",
        version="1.0",
        license="CC-BY-4.0",
    )
    card.add_column("time", units.s, "Time", coordinate_role="t")
    card.add_column("angle", units.dimensionless, "Angle in radians")
    card.add_column("angle_unc", units.dimensionless, "Angle uncertainty")

    markdown = card.to_markdown()

    assert "# Pendulum Data" in markdown
    assert "Version:** 1.0" in markdown
    assert "Domain:** mechanics" in markdown
    assert "| time |" in markdown
    assert "| angle |" in markdown


def test_save_load_dataset_card_json(tmp_path):
    """Test saving and loading dataset card as JSON."""
    card = DimDatasetCard(
        name="test_dataset",
        description="Test dataset",
        domain="fluid_dynamics",
    )
    card.add_column("x", units.m, "X position")
    card.add_column("u", units.m / units.s, "X velocity")

    json_path = tmp_path / "card.json"
    save_dataset_card(card, json_path)

    assert json_path.exists()

    loaded = load_dataset_card(json_path)
    assert loaded.name == card.name
    assert loaded.domain == card.domain
    assert len(loaded.columns) == 2


def test_save_dataset_card_markdown(tmp_path):
    """Test saving dataset card as markdown."""
    card = DimDatasetCard(
        name="Test",
        description="A test",
    )
    card.add_column("value", units.m, "A value")

    md_path = tmp_path / "card.md"
    save_dataset_card(card, md_path)

    assert md_path.exists()

    with open(md_path) as f:
        content = f.read()
        assert "# Test" in content


def test_load_markdown_raises_error(tmp_path):
    """Test that loading from markdown raises error."""
    md_path = tmp_path / "card.md"
    md_path.write_text("# Test")

    with pytest.raises(ValueError, match="Cannot load.*markdown"):
        load_dataset_card(md_path)


def test_coordinate_system_enum():
    """Test CoordinateSystem enum."""
    assert CoordinateSystem.CARTESIAN.value == "cartesian"
    assert CoordinateSystem.SPHERICAL.value == "spherical"
    assert CoordinateSystem.CYLINDRICAL.value == "cylindrical"
    assert CoordinateSystem.CUSTOM.value == "custom"


def test_dataset_card_with_uncertainties():
    """Test dataset card with uncertainty columns."""
    card = DimDatasetCard(name="measured_data")
    card.add_column(
        "temperature",
        units.K,
        "Temperature measurement",
        uncertainty_col="temp_uncertainty",
    )
    card.add_column("temp_uncertainty", units.K, "Temperature uncertainty")

    data = card.to_dict()
    loaded = DimDatasetCard.from_dict(data)

    temp_col = loaded.get_column("temperature")
    assert temp_col.uncertainty_col == "temp_uncertainty"


def test_dataset_card_with_tags():
    """Test dataset card with tags."""
    card = DimDatasetCard(
        name="experiment_data",
        tags=["experimental", "high-precision", "2024"],
    )

    assert len(card.tags) == 3
    assert "experimental" in card.tags

    data = card.to_dict()
    loaded = DimDatasetCard.from_dict(data)
    assert loaded.tags == card.tags


def test_dataset_card_roundtrip():
    """Test complete roundtrip serialization."""
    card = DimDatasetCard(
        name="physics_experiment",
        description="Comprehensive physics data",
        domain="electromagnetism",
        coordinate_system=CoordinateSystem.SPHERICAL,
        version="2.1",
        source="https://example.com/data",
        license="MIT",
        citation="Author et al. (2024)",
        tags=["experiment", "EM"],
    )

    card.add_column("r", units.m, "Radial distance", coordinate_role="r")
    card.add_column("theta", units.dimensionless, "Polar angle", coordinate_role="theta")
    card.add_column("E", units.V / units.m, "Electric field")

    # Serialize and deserialize
    data = card.to_dict()
    loaded = DimDatasetCard.from_dict(data)

    # Check all fields
    assert loaded.name == card.name
    assert loaded.description == card.description
    assert loaded.domain == card.domain
    assert loaded.coordinate_system == card.coordinate_system
    assert loaded.version == card.version
    assert loaded.source == card.source
    assert loaded.license == card.license
    assert loaded.citation == card.citation
    assert loaded.tags == card.tags
    assert len(loaded.columns) == len(card.columns)

    # Check columns
    for orig_col, loaded_col in zip(card.columns, loaded.columns):
        assert loaded_col.name == orig_col.name
        assert loaded_col.unit.dimension == orig_col.unit.dimension
        assert loaded_col.description == orig_col.description
        assert loaded_col.coordinate_role == orig_col.coordinate_role

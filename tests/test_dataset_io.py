"""Tests for dataset I/O with cards."""

import numpy as np
import pytest

from dimtensor import DimArray, units
from dimtensor.datasets import DimDatasetCard, CoordinateSystem
from dimtensor.io import (
    save_csv_with_card,
    load_csv_with_card,
    save_hdf5_with_card,
    load_hdf5_with_card,
)

# Conditional imports for optional dependencies
try:
    from dimtensor.io import save_netcdf_with_card, load_netcdf_with_card

    HAS_NETCDF = True
except ImportError:
    HAS_NETCDF = False

try:
    from dimtensor.io import save_parquet_with_card, load_parquet_with_card

    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    time = DimArray(np.linspace(0, 10, 100), units.s)
    position = DimArray(np.sin(np.linspace(0, 10, 100)), units.m)
    velocity = DimArray(np.cos(np.linspace(0, 10, 100)), units.m / units.s)

    data = {
        "time": time,
        "position": position,
        "velocity": velocity,
    }

    card = DimDatasetCard(
        name="oscillation_data",
        description="Simple harmonic oscillator",
        domain="mechanics",
        version="1.0",
    )
    card.add_column("time", units.s, "Time coordinate", coordinate_role="t")
    card.add_column("position", units.m, "Position")
    card.add_column("velocity", units.m / units.s, "Velocity")

    return data, card


def test_csv_with_card_roundtrip(tmp_path, sample_dataset):
    """Test CSV save/load with dataset card."""
    data, card = sample_dataset
    csv_path = tmp_path / "data.csv"

    # Save
    save_csv_with_card(data, csv_path, card)

    # Check files exist
    assert csv_path.exists()
    assert (tmp_path / "data.json").exists()

    # Load
    loaded_data, loaded_card = load_csv_with_card(csv_path)

    # Check card
    assert loaded_card.name == card.name
    assert loaded_card.domain == card.domain
    assert len(loaded_card.columns) == 3

    # Check data
    assert set(loaded_data.keys()) == {"time", "position", "velocity"}
    assert isinstance(loaded_data["time"], DimArray)
    np.testing.assert_allclose(loaded_data["time"]._data, data["time"]._data)


def test_csv_with_card_auto_create(tmp_path, sample_dataset):
    """Test CSV save with auto-generated card."""
    data, _ = sample_dataset
    csv_path = tmp_path / "data.csv"

    # Save without card (should auto-create)
    save_csv_with_card(data, csv_path)

    # Load
    loaded_data, loaded_card = load_csv_with_card(csv_path)

    # Check auto-generated card
    assert loaded_card.name == "data"
    assert len(loaded_card.columns) == 3


def test_csv_missing_sidecar_raises_error(tmp_path, sample_dataset):
    """Test that loading CSV without sidecar raises error."""
    from dimtensor.io.csv import save_csv

    data, _ = sample_dataset
    csv_path = tmp_path / "data.csv"

    # Save plain CSV (no sidecar)
    save_csv({k: v._data for k, v in data.items()}, csv_path)

    # Should raise error when loading with card
    with pytest.raises(FileNotFoundError, match="JSON sidecar not found"):
        load_csv_with_card(csv_path)


def test_hdf5_with_card_roundtrip(tmp_path, sample_dataset):
    """Test HDF5 save/load with dataset card."""
    data, card = sample_dataset
    h5_path = tmp_path / "data.h5"

    # Save
    save_hdf5_with_card(data, h5_path, card)
    assert h5_path.exists()

    # Load
    loaded_data, loaded_card = load_hdf5_with_card(h5_path)

    # Check card
    assert loaded_card.name == card.name
    assert loaded_card.domain == card.domain
    assert len(loaded_card.columns) == 3

    # Check data
    assert set(loaded_data.keys()) == {"time", "position", "velocity"}
    np.testing.assert_allclose(loaded_data["position"]._data, data["position"]._data)


def test_hdf5_missing_card_raises_error(tmp_path, sample_dataset):
    """Test that loading HDF5 without card raises error."""
    from dimtensor.io import save_hdf5, load_hdf5

    data, _ = sample_dataset
    h5_path = tmp_path / "data.h5"

    # Save single array (no card)
    save_hdf5(data["position"], h5_path)

    # Should raise error when loading with card
    with pytest.raises(KeyError, match="No dataset card found"):
        load_hdf5_with_card(h5_path)


@pytest.mark.skipif(not HAS_NETCDF, reason="netCDF4 not installed")
def test_netcdf_with_card_roundtrip(tmp_path, sample_dataset):
    """Test NetCDF save/load with dataset card."""
    data, card = sample_dataset
    nc_path = tmp_path / "data.nc"

    # Save
    save_netcdf_with_card(data, nc_path, card)
    assert nc_path.exists()

    # Load
    loaded_data, loaded_card = load_netcdf_with_card(nc_path)

    # Check card
    assert loaded_card.name == card.name
    assert loaded_card.domain == card.domain

    # Check data
    assert "velocity" in loaded_data
    np.testing.assert_allclose(loaded_data["velocity"]._data, data["velocity"]._data)


@pytest.mark.skipif(not HAS_NETCDF, reason="netCDF4 not installed")
def test_netcdf_missing_card_raises_error(tmp_path, sample_dataset):
    """Test that loading NetCDF without card raises error."""
    from dimtensor.io import save_netcdf

    data, _ = sample_dataset
    nc_path = tmp_path / "data.nc"

    # Save single array (no card)
    save_netcdf(data["position"], nc_path)

    # Should raise error when loading with card
    with pytest.raises(AttributeError, match="No dataset card found"):
        load_netcdf_with_card(nc_path)


@pytest.mark.skipif(not HAS_PARQUET, reason="pyarrow not installed")
def test_parquet_with_card_roundtrip(tmp_path, sample_dataset):
    """Test Parquet save/load with dataset card."""
    data, card = sample_dataset
    pq_path = tmp_path / "data.parquet"

    # Save
    save_parquet_with_card(data, pq_path, card)
    assert pq_path.exists()

    # Load
    loaded_data, loaded_card = load_parquet_with_card(pq_path)

    # Check card
    assert loaded_card.name == card.name
    assert loaded_card.description == card.description

    # Check data
    assert "time" in loaded_data
    np.testing.assert_allclose(loaded_data["time"]._data, data["time"]._data)


@pytest.mark.skipif(not HAS_PARQUET, reason="pyarrow not installed")
def test_parquet_missing_card_raises_error(tmp_path, sample_dataset):
    """Test that loading Parquet without card raises error."""
    from dimtensor.io import save_parquet

    data, _ = sample_dataset
    pq_path = tmp_path / "data.parquet"

    # Save single array (no card)
    save_parquet(data["position"], pq_path)

    # Should raise error when loading with card
    with pytest.raises(KeyError, match="No dataset card found"):
        load_parquet_with_card(pq_path)


def test_csv_with_uncertainties(tmp_path):
    """Test CSV with uncertainty columns."""
    data = {
        "temperature": DimArray([300.0, 301.0, 302.0], units.K),
        "temp_unc": DimArray([0.1, 0.15, 0.12], units.K),
    }

    card = DimDatasetCard(name="temp_measurements")
    card.add_column(
        "temperature",
        units.K,
        "Temperature",
        uncertainty_col="temp_unc",
    )
    card.add_column("temp_unc", units.K, "Temperature uncertainty")

    csv_path = tmp_path / "temps.csv"
    save_csv_with_card(data, csv_path, card)

    loaded_data, loaded_card = load_csv_with_card(csv_path)

    # Check uncertainty column reference
    temp_col = loaded_card.get_column("temperature")
    assert temp_col.uncertainty_col == "temp_unc"

    # Check data
    np.testing.assert_allclose(loaded_data["temp_unc"]._data, data["temp_unc"]._data)


def test_hdf5_with_compression(tmp_path, sample_dataset):
    """Test HDF5 with different compression options."""
    data, card = sample_dataset
    h5_path = tmp_path / "data_compressed.h5"

    # Save with gzip compression
    save_hdf5_with_card(data, h5_path, card, compression="gzip")

    # Load and verify
    loaded_data, loaded_card = load_hdf5_with_card(h5_path)
    assert loaded_card.name == card.name
    np.testing.assert_allclose(loaded_data["time"]._data, data["time"]._data)


def test_csv_incompatible_shapes_raises_error(tmp_path):
    """Test that CSV save with incompatible shapes raises error."""
    data = {
        "short": DimArray([1.0, 2.0], units.m),
        "long": DimArray([1.0, 2.0, 3.0], units.m),
    }

    card = DimDatasetCard(name="bad_data")
    card.add_column("short", units.m)
    card.add_column("long", units.m)

    csv_path = tmp_path / "bad.csv"

    with pytest.raises(ValueError, match="same length"):
        save_csv_with_card(data, csv_path, card)


def test_dataset_with_multiple_coordinate_roles(tmp_path):
    """Test dataset with 3D coordinates."""
    data = {
        "x": DimArray([1.0, 2.0, 3.0], units.m),
        "y": DimArray([4.0, 5.0, 6.0], units.m),
        "z": DimArray([7.0, 8.0, 9.0], units.m),
        "value": DimArray([10.0, 11.0, 12.0], units.kg),
    }

    card = DimDatasetCard(
        name="3d_field",
        coordinate_system=CoordinateSystem.CARTESIAN,
    )
    card.add_column("x", units.m, "X coordinate", coordinate_role="x")
    card.add_column("y", units.m, "Y coordinate", coordinate_role="y")
    card.add_column("z", units.m, "Z coordinate", coordinate_role="z")
    card.add_column("value", units.kg, "Mass field value")

    csv_path = tmp_path / "field.csv"
    save_csv_with_card(data, csv_path, card)

    loaded_data, loaded_card = load_csv_with_card(csv_path)

    # Check coordinate roles preserved
    assert loaded_card.get_column("x").coordinate_role == "x"
    assert loaded_card.get_column("y").coordinate_role == "y"
    assert loaded_card.get_column("z").coordinate_role == "z"

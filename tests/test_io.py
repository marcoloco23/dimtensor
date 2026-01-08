"""Tests for IO/serialization module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dimtensor import DimArray, units
from dimtensor.io import to_json, from_json, save_json, load_json
from dimtensor.io.json import to_dict, from_dict


class TestJSONSerialization:
    """Tests for JSON serialization."""

    def test_to_dict_basic(self):
        """Convert simple DimArray to dict."""
        arr = DimArray([1.0, 2.0, 3.0], units.m)
        d = to_dict(arr)
        assert d["data"] == [1.0, 2.0, 3.0]
        assert d["unit"]["symbol"] == "m"
        assert d["unit"]["dimension"]["length"] == 1.0

    def test_to_dict_with_uncertainty(self):
        """Convert DimArray with uncertainty to dict."""
        arr = DimArray([10.0], units.m, uncertainty=[0.1])
        d = to_dict(arr)
        assert "uncertainty" in d
        assert d["uncertainty"] == [0.1]

    def test_from_dict_basic(self):
        """Create DimArray from dict."""
        d = {
            "data": [1.0, 2.0, 3.0],
            "unit": {
                "symbol": "m",
                "dimension": {"length": 1.0, "mass": 0.0, "time": 0.0, "current": 0.0, "temperature": 0.0, "amount": 0.0, "luminosity": 0.0},
                "scale": 1.0,
            },
            "dtype": "float64",
        }
        arr = from_dict(d)
        assert np.allclose(arr.data, [1.0, 2.0, 3.0])
        assert arr.unit.symbol == "m"

    def test_to_json_basic(self):
        """Serialize to JSON string."""
        arr = DimArray([1.0, 2.0], units.m)
        json_str = to_json(arr)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["data"] == [1.0, 2.0]

    def test_from_json_basic(self):
        """Deserialize from JSON string."""
        json_str = '{"data": [1.0, 2.0], "unit": {"symbol": "m", "dimension": {"length": 1.0, "mass": 0.0, "time": 0.0, "current": 0.0, "temperature": 0.0, "amount": 0.0, "luminosity": 0.0}, "scale": 1.0}, "dtype": "float64"}'
        arr = from_json(json_str)
        assert np.allclose(arr.data, [1.0, 2.0])

    def test_roundtrip(self):
        """Test JSON roundtrip preserves data and units."""
        original = DimArray([1.0, 2.0, 3.0], units.m / units.s)
        json_str = to_json(original)
        restored = from_json(json_str)

        assert np.allclose(original.data, restored.data)
        assert original.dimension == restored.dimension
        assert original.unit.scale == pytest.approx(restored.unit.scale)

    def test_roundtrip_with_uncertainty(self):
        """Test JSON roundtrip preserves uncertainty."""
        original = DimArray([10.0, 20.0], units.kg, uncertainty=[0.5, 1.0])
        json_str = to_json(original)
        restored = from_json(json_str)

        assert np.allclose(original.data, restored.data)
        assert np.allclose(original.uncertainty, restored.uncertainty)

    def test_save_load_file(self):
        """Test saving and loading to file."""
        arr = DimArray([1.0, 2.0, 3.0], units.m)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_json(arr, path)
            loaded = load_json(path)
            assert np.allclose(arr.data, loaded.data)
            assert arr.unit.symbol == loaded.unit.symbol
        finally:
            Path(path).unlink(missing_ok=True)

    def test_multidimensional_array(self):
        """Test JSON with multidimensional array."""
        arr = DimArray([[1.0, 2.0], [3.0, 4.0]], units.m)
        json_str = to_json(arr)
        restored = from_json(json_str)
        assert restored.shape == (2, 2)
        assert np.allclose(arr.data, restored.data)

    def test_derived_unit(self):
        """Test JSON with derived unit."""
        arr = DimArray([9.8], units.m / units.s**2)
        json_str = to_json(arr)
        restored = from_json(json_str)
        assert restored.dimension == arr.dimension


class TestPandasIntegration:
    """Tests for pandas integration."""

    @pytest.fixture
    def skip_if_no_pandas(self):
        """Skip test if pandas not available."""
        pytest.importorskip("pandas")

    def test_to_series(self, skip_if_no_pandas):
        """Convert DimArray to pandas Series."""
        from dimtensor.io import to_series

        arr = DimArray([1.0, 2.0, 3.0], units.m)
        series = to_series(arr, name="distance")

        assert series.name == "distance"
        assert np.allclose(series.values, [1.0, 2.0, 3.0])
        assert series.attrs["unit_symbol"] == "m"

    def test_from_series(self, skip_if_no_pandas):
        """Create DimArray from pandas Series."""
        import pandas as pd
        from dimtensor.io import to_series, from_series

        arr = DimArray([1.0, 2.0, 3.0], units.m)
        series = to_series(arr)
        restored = from_series(series)

        assert np.allclose(arr.data, restored.data)
        assert arr.dimension == restored.dimension

    def test_to_dataframe(self, skip_if_no_pandas):
        """Convert multiple DimArrays to DataFrame."""
        from dimtensor.io import to_dataframe

        distance = DimArray([1.0, 2.0, 3.0], units.m)
        time = DimArray([0.1, 0.2, 0.3], units.s)

        df = to_dataframe({"distance": distance, "time": time})

        assert list(df.columns) == ["distance", "time"]
        assert "dimtensor_units" in df.attrs

    def test_from_dataframe(self, skip_if_no_pandas):
        """Create DimArrays from DataFrame."""
        from dimtensor.io import to_dataframe, from_dataframe

        distance = DimArray([1.0, 2.0, 3.0], units.m)
        time = DimArray([0.1, 0.2, 0.3], units.s)

        df = to_dataframe({"distance": distance, "time": time})
        arrays = from_dataframe(df)

        assert "distance" in arrays
        assert "time" in arrays
        assert np.allclose(arrays["distance"].data, distance.data)
        assert arrays["distance"].dimension == distance.dimension

    def test_series_with_uncertainty(self, skip_if_no_pandas):
        """Series roundtrip preserves uncertainty."""
        from dimtensor.io import to_series, from_series

        arr = DimArray([10.0, 20.0], units.m, uncertainty=[0.5, 1.0])
        series = to_series(arr)
        restored = from_series(series)

        assert np.allclose(restored.uncertainty, arr.uncertainty)


class TestHDF5Serialization:
    """Tests for HDF5 serialization."""

    @pytest.fixture
    def skip_if_no_h5py(self):
        """Skip test if h5py not available."""
        pytest.importorskip("h5py")

    def test_save_load_basic(self, skip_if_no_h5py):
        """Save and load DimArray to HDF5."""
        from dimtensor.io import save_hdf5, load_hdf5

        arr = DimArray([1.0, 2.0, 3.0], units.m)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name

        try:
            save_hdf5(arr, path)
            loaded = load_hdf5(path)

            assert np.allclose(arr.data, loaded.data)
            assert arr.dimension == loaded.dimension
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_load_with_uncertainty(self, skip_if_no_h5py):
        """HDF5 roundtrip preserves uncertainty."""
        from dimtensor.io import save_hdf5, load_hdf5

        arr = DimArray([10.0, 20.0], units.kg, uncertainty=[0.5, 1.0])

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name

        try:
            save_hdf5(arr, path)
            loaded = load_hdf5(path)

            assert np.allclose(loaded.uncertainty, arr.uncertainty)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_load_multiple(self, skip_if_no_h5py):
        """Save and load multiple arrays."""
        from dimtensor.io.hdf5 import save_multiple_hdf5, load_multiple_hdf5

        distance = DimArray([1.0, 2.0, 3.0], units.m)
        time = DimArray([0.1, 0.2, 0.3], units.s)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name

        try:
            save_multiple_hdf5({"distance": distance, "time": time}, path)
            loaded = load_multiple_hdf5(path)

            assert "distance" in loaded
            assert "time" in loaded
            assert np.allclose(loaded["distance"].data, distance.data)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_compression(self, skip_if_no_h5py):
        """Test different compression options."""
        from dimtensor.io import save_hdf5, load_hdf5

        arr = DimArray(np.random.randn(1000), units.m)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name

        try:
            # Test with gzip compression
            save_hdf5(arr, path, compression="gzip")
            loaded = load_hdf5(path)
            assert np.allclose(arr.data, loaded.data)
        finally:
            Path(path).unlink(missing_ok=True)

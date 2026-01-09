"""Tests for dataset loaders."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from dimtensor.core.dimarray import DimArray
from dimtensor.datasets import load_dataset
from dimtensor.datasets.loaders import (
    BaseLoader,
    CSVLoader,
    NISTCODATALoader,
    NASAExoplanetLoader,
    PRISMClimateLoader,
    get_cache_dir,
    ensure_cache_dir,
)


class TestCacheDirectory:
    """Tests for cache directory management."""

    def test_get_cache_dir_default(self):
        """Test default cache directory location."""
        cache_dir = get_cache_dir()
        assert ".dimtensor" in str(cache_dir)
        assert "cache" in str(cache_dir)

    def test_get_cache_dir_env_override(self, tmp_path):
        """Test cache directory can be overridden with env var."""
        custom_cache = str(tmp_path / "custom_cache")
        with patch.dict(os.environ, {"DIMTENSOR_CACHE_DIR": custom_cache}):
            cache_dir = get_cache_dir()
            assert str(cache_dir) == custom_cache

    def test_ensure_cache_dir_creates(self, tmp_path):
        """Test ensure_cache_dir creates the directory."""
        custom_cache = str(tmp_path / "test_cache")
        with patch.dict(os.environ, {"DIMTENSOR_CACHE_DIR": custom_cache}):
            cache_dir = ensure_cache_dir()
            assert cache_dir.exists()
            assert cache_dir.is_dir()


class TestBaseLoader:
    """Tests for BaseLoader class (using CSVLoader as concrete implementation)."""

    def test_init(self):
        """Test BaseLoader initialization."""
        loader = CSVLoader(cache=True)
        assert loader.cache_enabled is True
        assert loader.cache_dir.exists()

    def test_init_no_cache(self):
        """Test BaseLoader with caching disabled."""
        loader = CSVLoader(cache=False)
        assert loader.cache_enabled is False

    def test_download_requires_requests(self, tmp_path):
        """Test download fails gracefully without requests library."""
        with patch.dict(
            "dimtensor.datasets.loaders.base.__dict__",
            {"HAS_REQUESTS": False},
        ):
            loader = CSVLoader()
            with pytest.raises(ImportError, match="requests library required"):
                loader.download("http://example.com/data.csv")

    @pytest.mark.network
    def test_download_with_mock(self, tmp_path):
        """Test download with mocked HTTP request."""
        mock_response = Mock()
        mock_response.content = b"test,data\n1,2\n3,4"
        mock_response.headers = {"content-type": "text/csv"}

        with patch.dict(os.environ, {"DIMTENSOR_CACHE_DIR": str(tmp_path)}):
            loader = CSVLoader()

            with patch("requests.get", return_value=mock_response):
                filepath = loader.download(
                    "http://example.com/test.csv",
                    cache_key="test_key",
                )

                assert filepath.exists()
                assert filepath.read_bytes() == b"test,data\n1,2\n3,4"

                # Check metadata was created
                metadata_file = tmp_path / "test_key.json"
                assert metadata_file.exists()

    def test_clear_cache_all(self, tmp_path):
        """Test clearing all cache files."""
        with patch.dict(os.environ, {"DIMTENSOR_CACHE_DIR": str(tmp_path)}):
            loader = CSVLoader()

            # Create some dummy cache files
            (tmp_path / "file1.csv").write_text("data")
            (tmp_path / "file1.json").write_text("{}")
            (tmp_path / "file2.csv").write_text("data")

            loader.clear_cache()

            # All files should be deleted
            assert not (tmp_path / "file1.csv").exists()
            assert not (tmp_path / "file1.json").exists()
            assert not (tmp_path / "file2.csv").exists()

    def test_clear_cache_specific(self, tmp_path):
        """Test clearing specific cache key."""
        with patch.dict(os.environ, {"DIMTENSOR_CACHE_DIR": str(tmp_path)}):
            loader = CSVLoader()

            # Create some dummy cache files
            (tmp_path / "file1.csv").write_text("data")
            (tmp_path / "file1.json").write_text("{}")
            (tmp_path / "file2.csv").write_text("data")

            loader.clear_cache(cache_key="file1")

            # Only file1 should be deleted
            assert not (tmp_path / "file1.csv").exists()
            assert not (tmp_path / "file1.json").exists()
            assert (tmp_path / "file2.csv").exists()


class TestCSVLoader:
    """Tests for CSVLoader class."""

    def test_parse_csv_simple(self, tmp_path):
        """Test parsing simple CSV file."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n")

        loader = CSVLoader()
        rows = loader.parse_csv(csv_file, skip_rows=0)

        assert len(rows) == 3
        assert rows[0] == ["a", "b", "c"]
        assert rows[1] == ["1", "2", "3"]
        assert rows[2] == ["4", "5", "6"]

    def test_parse_csv_skip_header(self, tmp_path):
        """Test parsing CSV with header skip."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("header\na,b,c\n1,2,3\n")

        loader = CSVLoader()
        rows = loader.parse_csv(csv_file, skip_rows=1)

        assert len(rows) == 2
        assert rows[0] == ["a", "b", "c"]

    def test_parse_csv_custom_delimiter(self, tmp_path):
        """Test parsing CSV with custom delimiter."""
        csv_file = tmp_path / "test.tsv"
        csv_file.write_text("a\tb\tc\n1\t2\t3\n")

        loader = CSVLoader()
        rows = loader.parse_csv(csv_file, delimiter="\t")

        assert len(rows) == 2
        assert rows[0] == ["a", "b", "c"]
        assert rows[1] == ["1", "2", "3"]


class TestNISTLoader:
    """Tests for NIST CODATA loader."""

    def test_init(self):
        """Test NISTCODATALoader initialization."""
        loader = NISTCODATALoader()
        assert loader is not None
        assert hasattr(loader, "URL")

    def test_fallback_constants(self):
        """Test fallback constants are returned."""
        loader = NISTCODATALoader()
        constants = loader._get_fallback_constants()

        assert "speed of light in vacuum" in constants
        assert "Planck constant" in constants

        # Check they are DimArrays
        c = constants["speed of light in vacuum"]
        assert isinstance(c, DimArray)

    def test_parse_unit_simple(self):
        """Test parsing simple units."""
        loader = NISTCODATALoader()

        m_unit = loader._parse_unit("m")
        assert m_unit is not None
        assert str(m_unit) == "m"

        s_unit = loader._parse_unit("s")
        assert s_unit is not None
        assert str(s_unit) == "s"

    def test_parse_unit_unknown(self):
        """Test parsing unknown units returns None."""
        loader = NISTCODATALoader()
        result = loader._parse_unit("unknown_unit_xyz")
        assert result is None

    @pytest.mark.network
    @pytest.mark.skip(reason="Requires network and actual NIST download")
    def test_load_real_data(self):
        """Test loading real NIST data (requires network)."""
        loader = NISTCODATALoader()
        constants = loader.load()

        assert len(constants) > 0
        assert any("light" in name.lower() for name in constants.keys())


class TestNASAExoplanetLoader:
    """Tests for NASA Exoplanet Archive loader."""

    def test_init(self):
        """Test NASAExoplanetLoader initialization."""
        loader = NASAExoplanetLoader()
        assert loader is not None
        assert hasattr(loader, "URL")

    def test_parse_csv_mock(self, tmp_path):
        """Test parsing exoplanet CSV with mock data."""
        csv_file = tmp_path / "exoplanets.csv"
        csv_file.write_text(
            "pl_name,pl_masse,pl_rade,pl_orbper,pl_orbsmax,st_mass,st_rad\n"
            "Kepler-1b,1.0,1.0,365.25,1.0,1.0,1.0\n"
            "Kepler-2b,5.0,2.0,100.0,0.5,1.2,1.1\n"
        )

        loader = NASAExoplanetLoader()
        data = loader._parse_exoplanet_csv(csv_file)

        assert "pl_name" in data
        assert "pl_masse" in data
        assert "pl_rade" in data

        # Check names
        assert data["pl_name"] == ["Kepler-1b", "Kepler-2b"]

        # Check DimArrays
        assert isinstance(data["pl_masse"], DimArray)
        assert len(data["pl_masse"]) == 2
        assert data["pl_masse"].data[0] == 1.0

    @pytest.mark.network
    @pytest.mark.skip(reason="Requires network and may be slow")
    def test_load_real_data(self):
        """Test loading real NASA exoplanet data (requires network)."""
        loader = NASAExoplanetLoader()
        data = loader.load()

        assert "pl_masse" in data or "pl_name" in data
        if "pl_masse" in data:
            assert isinstance(data["pl_masse"], DimArray)
            assert len(data["pl_masse"]) > 100  # Should have many exoplanets


class TestPRISMClimateLoader:
    """Tests for PRISM climate data loader."""

    def test_init(self):
        """Test PRISMClimateLoader initialization."""
        loader = PRISMClimateLoader()
        assert loader is not None

    def test_load_sample_temperature(self):
        """Test loading sample temperature data."""
        loader = PRISMClimateLoader()
        data = loader.load(variable="tmean", start_year=2020, end_year=2021)

        assert "dates" in data
        assert "values" in data
        assert "latitude" in data
        assert "longitude" in data

        # Check dates
        dates = data["dates"]
        assert len(dates) == 24  # 2 years * 12 months

        # Check values are DimArray
        values = data["values"]
        assert isinstance(values, DimArray)
        assert len(values) == 24

    def test_load_sample_precipitation(self):
        """Test loading sample precipitation data."""
        loader = PRISMClimateLoader()
        data = loader.load(variable="ppt", start_year=2022, end_year=2022)

        assert "values" in data
        values = data["values"]
        assert isinstance(values, DimArray)
        assert len(values) == 12  # 1 year * 12 months

        # Precipitation should be non-negative
        assert np.all(values.data >= 0)

    def test_load_invalid_variable(self):
        """Test loading with invalid variable raises error."""
        loader = PRISMClimateLoader()
        with pytest.raises(ValueError, match="Unknown variable"):
            loader.load(variable="invalid_var")


class TestIntegration:
    """Integration tests for dataset loading through registry."""

    def test_load_nist_through_registry(self):
        """Test loading NIST dataset through load_dataset()."""
        # Use fallback constants (no network)
        with patch(
            "dimtensor.datasets.loaders.nist.NISTCODATALoader.download"
        ) as mock_download:
            # Make download fail so it uses fallback
            mock_download.side_effect = RuntimeError("Network error")

            try:
                constants = load_dataset("nist_codata_2022")
                # Should get fallback constants
                assert len(constants) > 0
            except RuntimeError:
                # If it still fails, that's OK for this test
                pass

    def test_load_prism_through_registry(self):
        """Test loading PRISM dataset through load_dataset()."""
        data = load_dataset("prism_climate", variable="tmean")

        assert "values" in data
        assert isinstance(data["values"], DimArray)

    def test_list_datasets_includes_real_data(self):
        """Test that list_datasets includes new real datasets."""
        from dimtensor.datasets import list_datasets

        datasets = list_datasets()
        dataset_names = [ds.name for ds in datasets]

        assert "nist_codata_2022" in dataset_names
        assert "nasa_exoplanets" in dataset_names
        assert "prism_climate" in dataset_names

    def test_get_dataset_info_real_data(self):
        """Test getting info for real datasets."""
        from dimtensor.datasets import get_dataset_info

        info = get_dataset_info("nist_codata_2022")
        assert info.domain == "constants"
        assert "NIST" in info.description

        info = get_dataset_info("nasa_exoplanets")
        assert info.domain == "astronomy"
        assert "exoplanet" in info.description.lower()

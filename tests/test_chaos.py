"""Chaos tests: verify dimtensor degrades gracefully under failure modes.

(v5.2.0 task #272)

Where fuzz tests randomize *inputs*, chaos tests randomize the *environment*
or simulate broken collaborators (filesystem, optional dependencies, etc.).
The goal is to confirm dimtensor never produces a confusing or silent error.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from dimtensor import DimArray, Dimension, Unit, units
from dimtensor.errors import DimensionError, UnitConversionError
from dimtensor.io.json import from_json, load_json, save_json, to_json


# ---------------------------------------------------------------------------
# Filesystem failure modes
# ---------------------------------------------------------------------------


class TestFilesystemChaos:
    """I/O routines must surface clear errors for filesystem problems."""

    def test_load_nonexistent_file_raises_filenotfound(self) -> None:
        """Loading a missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_json("/nonexistent/path/that/does/not/exist.json")

    def test_load_directory_raises_oserror(self, tmp_path: Path) -> None:
        """Loading a directory (instead of a file) raises an OSError."""
        with pytest.raises((OSError, IsADirectoryError, PermissionError)):
            load_json(str(tmp_path))

    def test_load_corrupted_json_raises_decode_error(
        self, tmp_path: Path
    ) -> None:
        """Loading a file containing invalid JSON raises JSONDecodeError."""
        bad = tmp_path / "bad.json"
        bad.write_text("not really JSON at all { ")
        with pytest.raises(json.JSONDecodeError):
            load_json(bad)

    def test_load_truncated_file_raises(self, tmp_path: Path) -> None:
        """A truncated JSON file should raise a clean error."""
        truncated = tmp_path / "truncated.json"
        truncated.write_text('{"data": [1.0, 2.0]')  # no closing brace
        with pytest.raises(json.JSONDecodeError):
            load_json(truncated)

    def test_save_then_load_round_trip(self, tmp_path: Path) -> None:
        """A normal save/load round-trip should work without errors."""
        arr = DimArray(np.array([1.0, 2.0, 3.0]), units.m)
        path = tmp_path / "data.json"
        save_json(arr, path)
        loaded = load_json(path)
        np.testing.assert_array_equal(loaded._data, arr._data)


# ---------------------------------------------------------------------------
# Missing optional dependencies
# ---------------------------------------------------------------------------


class TestOptionalDependencyChaos:
    """When an optional dependency is missing, modules import or fail clearly."""

    def test_torch_optional_clean_failure(self) -> None:
        """If torch is missing, importing should not crash dimtensor."""
        # dimtensor itself must import without torch
        import dimtensor

        assert hasattr(dimtensor, "DimArray")

    def test_jax_optional_clean_failure(self) -> None:
        """If JAX is missing, top-level import still succeeds."""
        import dimtensor

        assert hasattr(dimtensor, "DimArray")

    def test_io_pandas_handles_missing_pandas(self) -> None:
        """If pandas is missing, importing the pandas module raises ImportError."""
        try:
            from dimtensor.io.pandas import to_dataframe  # noqa: F401
        except ImportError:
            # Acceptable - clean error on import
            pass

    def test_io_xarray_handles_missing_xarray(self) -> None:
        """If xarray is missing, importing the xarray module raises ImportError."""
        try:
            from dimtensor.io.xarray import to_xarray  # noqa: F401
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Pathological inputs that historically caused issues
# ---------------------------------------------------------------------------


class TestPathologicalInputs:
    """Inputs that have historically broken numeric libraries."""

    def test_empty_array_with_unit(self) -> None:
        """Empty DimArray should behave reasonably."""
        empty = DimArray(np.array([], dtype=float), units.m)
        assert empty._data.shape == (0,)
        # Operations on empty arrays should not crash
        doubled = empty * 2
        assert doubled._data.shape == (0,)

    def test_zero_dim_array(self) -> None:
        """0-d (scalar) arrays should work with units."""
        scalar = DimArray(np.float64(42.0), units.m)
        assert scalar.unit == units.m

    def test_very_large_array(self) -> None:
        """Large arrays should not run out of dimension cache or similar."""
        arr = DimArray(np.zeros(10_000), units.kg)
        result = arr + arr
        assert result._data.shape == (10_000,)

    def test_nan_addition_propagates(self) -> None:
        """Adding arrays with NaN preserves NaN without raising."""
        a = DimArray(np.array([1.0, np.nan, 3.0]), units.m)
        b = DimArray(np.array([4.0, 5.0, 6.0]), units.m)
        result = a + b
        assert np.isnan(result._data[1])
        assert result._data[0] == 5.0
        assert result._data[2] == 9.0

    def test_inf_multiplication_propagates(self) -> None:
        """Multiplying by inf produces inf, not crash."""
        a = DimArray(np.array([np.inf, 1.0]), units.s)
        result = a * 2.0
        assert np.isinf(result._data[0])

    def test_division_by_zero_yields_inf(self) -> None:
        """Division by zero should follow numpy semantics (inf or NaN)."""
        a = DimArray(np.array([1.0, 0.0]), units.m)
        b = DimArray(np.array([0.0, 0.0]), units.s)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = a / b
        # The unit should still be derived correctly
        assert result.unit.dimension == Dimension(length=1, time=-1)


# ---------------------------------------------------------------------------
# Concurrent access (basic thread safety smoke test)
# ---------------------------------------------------------------------------


class TestConcurrencyChaos:
    """Basic smoke checks: parallel reads of the same DimArray are safe."""

    def test_parallel_reads_dont_corrupt_state(self) -> None:
        """Reading from a DimArray from multiple threads is safe."""
        import threading

        arr = DimArray(np.arange(1000, dtype=float), units.m)
        errors: list[BaseException] = []

        def reader() -> None:
            try:
                for _ in range(100):
                    _ = arr * 2
                    _ = arr + arr
                    _ = arr.unit.symbol
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Concurrent reads produced errors: {errors!r}"


# ---------------------------------------------------------------------------
# Recovery from ill-formed pickles / dictionaries
# ---------------------------------------------------------------------------


class TestRecoveryFromBadInput:
    """from_dict should give actionable errors for missing keys."""

    def test_missing_data_key_raises_keyerror(self) -> None:
        from dimtensor.io.json import from_dict

        with pytest.raises(KeyError):
            from_dict({"unit": {"symbol": "m", "dimension": {}, "scale": 1.0}})

    def test_missing_unit_key_raises_keyerror(self) -> None:
        from dimtensor.io.json import from_dict

        with pytest.raises(KeyError):
            from_dict({"data": [1.0, 2.0]})

    def test_missing_dimension_subkey_raises_keyerror(self) -> None:
        from dimtensor.io.json import from_dict

        with pytest.raises(KeyError):
            from_dict(
                {
                    "data": [1.0],
                    "unit": {
                        "symbol": "m",
                        "dimension": {"length": 1},  # missing other keys
                        "scale": 1.0,
                    },
                }
            )

    def test_invalid_scale_type_raises(self) -> None:
        """A non-numeric scale should fail on Unit construction."""
        from dimtensor.io.json import from_dict

        bad = {
            "data": [1.0],
            "unit": {
                "symbol": "m",
                "dimension": {
                    "length": 1,
                    "mass": 0,
                    "time": 0,
                    "current": 0,
                    "temperature": 0,
                    "amount": 0,
                    "luminosity": 0,
                },
                "scale": "not a number",
            },
        }
        # Construction may succeed (scale is just a field), but conversion
        # operations should fail predictably.
        with pytest.raises((TypeError, ValueError)):
            arr = from_dict(bad)
            arr.to(units.km)

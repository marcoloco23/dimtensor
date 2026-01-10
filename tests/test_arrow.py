"""Tests for Apache Arrow integration."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from dimtensor import DimArray, units


# Skip all tests if pyarrow not installed
pytest.importorskip("pyarrow")


class TestArrowExtensionType:
    """Tests for the DimArrayType extension type."""

    def test_extension_type_serialize_deserialize(self):
        """Round-trip unit metadata through extension type serialization."""
        from dimtensor.io.arrow import to_arrow_array, from_arrow_array

        arr = DimArray([1.0, 2.0, 3.0], units.m)
        pa_arr = to_arrow_array(arr)

        # Check type has correct unit
        assert pa_arr.type.unit.symbol == "m"
        assert pa_arr.type.unit.dimension == units.m.dimension

    def test_extension_type_derived_unit(self):
        """Extension type handles derived units correctly."""
        from dimtensor.io.arrow import to_arrow_array

        arr = DimArray([9.8], units.m / units.s**2)
        pa_arr = to_arrow_array(arr)

        assert pa_arr.type.unit.dimension == arr.dimension

    def test_extension_type_scaled_unit(self):
        """Extension type preserves scale factor."""
        from dimtensor.io.arrow import to_arrow_array, from_arrow_array

        arr = DimArray([1.0, 2.0], units.km)
        pa_arr = to_arrow_array(arr)
        restored = from_arrow_array(pa_arr)

        assert restored.unit.scale == pytest.approx(units.km.scale)

    def test_extension_type_registration(self):
        """Type is recognized after registration."""
        from dimtensor.io.arrow import is_dimarray_type, to_arrow_array

        arr = DimArray([1.0], units.m)
        pa_arr = to_arrow_array(arr)

        assert is_dimarray_type(pa_arr.type)

    def test_extract_unit_from_type(self):
        """Can extract unit from DimArrayType."""
        from dimtensor.io.arrow import extract_unit_from_type, to_arrow_array

        arr = DimArray([1.0], units.kg)
        pa_arr = to_arrow_array(arr)
        unit = extract_unit_from_type(pa_arr.type)

        assert unit is not None
        assert unit.symbol == "kg"
        assert unit.dimension == units.kg.dimension


class TestArrayConversion:
    """Tests for to_arrow_array and from_arrow_array."""

    def test_to_from_arrow_array_basic(self):
        """Convert DimArray to Arrow and back."""
        from dimtensor.io.arrow import to_arrow_array, from_arrow_array

        original = DimArray([1.0, 2.0, 3.0], units.m)
        pa_arr = to_arrow_array(original)
        restored = from_arrow_array(pa_arr)

        assert np.allclose(original.data, restored.data)
        assert original.dimension == restored.dimension
        assert original.unit.symbol == restored.unit.symbol

    def test_to_from_arrow_array_with_units(self):
        """Verify units preserved through conversion."""
        from dimtensor.io.arrow import to_arrow_array, from_arrow_array

        original = DimArray([100.0], units.km)
        pa_arr = to_arrow_array(original)
        restored = from_arrow_array(pa_arr)

        assert restored.unit.scale == pytest.approx(1000.0)
        assert restored.dimension == units.m.dimension

    def test_to_from_arrow_array_dimensionless(self):
        """Handle dimensionless arrays."""
        from dimtensor.io.arrow import to_arrow_array, from_arrow_array

        original = DimArray([0.5, 1.0, 1.5], units.dimensionless)
        pa_arr = to_arrow_array(original)
        restored = from_arrow_array(pa_arr)

        assert np.allclose(original.data, restored.data)
        assert restored.is_dimensionless

    def test_to_from_arrow_array_different_dtypes(self):
        """Handle various numpy dtypes."""
        from dimtensor.io.arrow import to_arrow_array, from_arrow_array

        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            original = DimArray(np.array([1, 2, 3], dtype=dtype), units.m)
            pa_arr = to_arrow_array(original)
            restored = from_arrow_array(pa_arr)

            assert np.allclose(original.data, restored.data)

    def test_from_arrow_array_with_explicit_unit(self):
        """Provide unit explicitly for non-extension arrays."""
        import pyarrow as pa
        from dimtensor.io.arrow import from_arrow_array

        pa_arr = pa.array([1.0, 2.0, 3.0])
        arr = from_arrow_array(pa_arr, unit=units.kg)

        assert arr.unit.symbol == "kg"
        assert np.allclose(arr.data, [1.0, 2.0, 3.0])

    def test_from_arrow_array_with_shape(self):
        """Reshape array during conversion."""
        from dimtensor.io.arrow import to_arrow_array, from_arrow_array

        original = DimArray([[1.0, 2.0], [3.0, 4.0]], units.m)
        pa_arr = to_arrow_array(original)
        restored = from_arrow_array(pa_arr, shape=(2, 2))

        assert restored.shape == (2, 2)
        assert np.allclose(original.data, restored.data)


class TestTableConversion:
    """Tests for to_arrow_table and from_arrow_table."""

    def test_to_from_arrow_table(self):
        """Convert multiple arrays to table and back."""
        from dimtensor.io.arrow import to_arrow_table, from_arrow_table

        distance = DimArray([1.0, 2.0, 3.0], units.m)
        time = DimArray([0.1, 0.2, 0.3], units.s)

        table = to_arrow_table({"distance": distance, "time": time})
        restored = from_arrow_table(table)

        assert "distance" in restored
        assert "time" in restored
        assert np.allclose(restored["distance"].data, distance.data)
        assert restored["distance"].dimension == units.m.dimension
        assert restored["time"].dimension == units.s.dimension

    def test_table_with_different_units(self):
        """Table handles arrays with different units."""
        from dimtensor.io.arrow import to_arrow_table, from_arrow_table

        mass = DimArray([10.0, 20.0], units.kg)
        velocity = DimArray([5.0, 10.0], units.m / units.s)
        temperature = DimArray([300.0, 350.0], units.K)

        table = to_arrow_table({
            "mass": mass,
            "velocity": velocity,
            "temperature": temperature,
        })
        restored = from_arrow_table(table)

        assert restored["mass"].unit.symbol == "kg"
        assert restored["temperature"].unit.symbol == "K"

    def test_table_with_uncertainty(self):
        """Table preserves uncertainty."""
        from dimtensor.io.arrow import to_arrow_table, from_arrow_table

        distance = DimArray([10.0, 20.0], units.m, uncertainty=[0.5, 1.0])
        time = DimArray([1.0, 2.0], units.s)

        table = to_arrow_table({"distance": distance, "time": time})
        restored = from_arrow_table(table)

        assert restored["distance"].has_uncertainty
        assert np.allclose(restored["distance"].uncertainty, [0.5, 1.0])
        assert not restored["time"].has_uncertainty


class TestRecordBatch:
    """Tests for record batch conversion."""

    def test_record_batch_roundtrip(self):
        """Record batch conversion preserves data and units."""
        from dimtensor.io.arrow import to_record_batch, from_record_batch

        distance = DimArray([1.0, 2.0, 3.0], units.m)
        time = DimArray([0.5, 1.0, 1.5], units.s)

        batch = to_record_batch({"distance": distance, "time": time})
        restored = from_record_batch(batch)

        assert np.allclose(restored["distance"].data, distance.data)
        assert restored["distance"].dimension == units.m.dimension
        assert restored["time"].dimension == units.s.dimension

    def test_record_batch_with_uncertainty(self):
        """Record batch preserves uncertainty."""
        from dimtensor.io.arrow import to_record_batch, from_record_batch

        arr = DimArray([10.0, 20.0, 30.0], units.kg, uncertainty=[0.1, 0.2, 0.3])

        batch = to_record_batch({"mass": arr})
        restored = from_record_batch(batch)

        assert restored["mass"].has_uncertainty
        assert np.allclose(restored["mass"].uncertainty, [0.1, 0.2, 0.3])


class TestIPCStream:
    """Tests for IPC streaming functions."""

    def test_ipc_stream_roundtrip(self):
        """Stream IPC preserves data and units."""
        from dimtensor.io.arrow import write_ipc_stream, read_ipc_stream

        distance = DimArray([1.0, 2.0, 3.0], units.m)
        time = DimArray([0.5, 1.0, 1.5], units.s)

        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            path = f.name

        try:
            write_ipc_stream({"distance": distance, "time": time}, path)
            restored = read_ipc_stream(path)

            assert np.allclose(restored["distance"].data, distance.data)
            assert restored["distance"].unit.symbol == "m"
            assert restored["time"].unit.symbol == "s"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_ipc_stream_derived_units(self):
        """Stream IPC handles derived units."""
        from dimtensor.io.arrow import write_ipc_stream, read_ipc_stream

        velocity = DimArray([10.0, 20.0], units.m / units.s)
        acceleration = DimArray([9.8, 10.2], units.m / units.s**2)

        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            path = f.name

        try:
            write_ipc_stream({"v": velocity, "a": acceleration}, path)
            restored = read_ipc_stream(path)

            assert restored["v"].dimension == velocity.dimension
            assert restored["a"].dimension == acceleration.dimension
        finally:
            Path(path).unlink(missing_ok=True)

    def test_ipc_stream_with_file_object(self):
        """Stream IPC works with file objects."""
        from dimtensor.io.arrow import write_ipc_stream, read_ipc_stream
        import io

        arr = DimArray([1.0, 2.0], units.m)

        # Write to BytesIO
        buffer = io.BytesIO()
        write_ipc_stream({"x": arr}, buffer)

        # Read back
        buffer.seek(0)
        restored = read_ipc_stream(buffer)

        assert np.allclose(restored["x"].data, arr.data)
        assert restored["x"].unit.symbol == "m"


class TestIPCFile:
    """Tests for IPC file functions."""

    def test_ipc_file_roundtrip(self):
        """File IPC preserves data and units."""
        from dimtensor.io.arrow import write_ipc_file, read_ipc_file

        distance = DimArray([1.0, 2.0, 3.0], units.m)
        time = DimArray([0.5, 1.0, 1.5], units.s)

        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            path = f.name

        try:
            write_ipc_file({"distance": distance, "time": time}, path)
            restored = read_ipc_file(path)

            assert np.allclose(restored["distance"].data, distance.data)
            assert restored["distance"].unit.symbol == "m"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_ipc_file_mmap(self):
        """Memory-mapped IPC file reading."""
        from dimtensor.io.arrow import write_ipc_file, read_ipc_file_mmap

        arr = DimArray(np.arange(1000, dtype=np.float64), units.m)

        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            path = f.name

        try:
            write_ipc_file({"data": arr}, path)
            restored = read_ipc_file_mmap(path)

            assert np.allclose(restored["data"].data, arr.data)
            assert restored["data"].unit.symbol == "m"
        finally:
            Path(path).unlink(missing_ok=True)


class TestMultidimensionalArrays:
    """Tests for multidimensional array handling."""

    def test_2d_array(self):
        """Handle 2D arrays."""
        from dimtensor.io.arrow import to_arrow_table, from_arrow_table

        arr = DimArray([[1.0, 2.0], [3.0, 4.0]], units.m)

        table = to_arrow_table({"matrix": arr})
        restored = from_arrow_table(table)

        assert restored["matrix"].shape == (2, 2)
        assert np.allclose(restored["matrix"].data, arr.data)

    def test_3d_array(self):
        """Handle 3D arrays."""
        from dimtensor.io.arrow import to_arrow_table, from_arrow_table

        arr = DimArray(np.arange(24).reshape(2, 3, 4).astype(float), units.kg)

        table = to_arrow_table({"tensor": arr})
        restored = from_arrow_table(table)

        assert restored["tensor"].shape == (2, 3, 4)
        assert np.allclose(restored["tensor"].data, arr.data)


class TestUncertainty:
    """Tests for uncertainty handling."""

    def test_uncertainty_roundtrip_table(self):
        """Uncertainty preserved through table conversion."""
        from dimtensor.io.arrow import to_arrow_table, from_arrow_table

        arr = DimArray([10.0, 20.0, 30.0], units.m, uncertainty=[0.1, 0.2, 0.3])

        table = to_arrow_table({"measurement": arr})
        restored = from_arrow_table(table)

        assert restored["measurement"].has_uncertainty
        assert np.allclose(restored["measurement"].uncertainty, arr.uncertainty)

    def test_uncertainty_roundtrip_ipc(self):
        """Uncertainty preserved through IPC."""
        from dimtensor.io.arrow import write_ipc_file, read_ipc_file

        arr = DimArray([100.0], units.kg, uncertainty=[0.5])

        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            path = f.name

        try:
            write_ipc_file({"mass": arr}, path)
            restored = read_ipc_file(path)

            assert restored["mass"].has_uncertainty
            assert np.allclose(restored["mass"].uncertainty, [0.5])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_mixed_uncertainty(self):
        """Handle mix of arrays with and without uncertainty."""
        from dimtensor.io.arrow import to_arrow_table, from_arrow_table

        with_unc = DimArray([10.0], units.m, uncertainty=[0.1])
        without_unc = DimArray([20.0], units.s)

        table = to_arrow_table({"a": with_unc, "b": without_unc})
        restored = from_arrow_table(table)

        assert restored["a"].has_uncertainty
        assert not restored["b"].has_uncertainty


class TestDerivedUnits:
    """Tests for complex/derived unit expressions."""

    def test_velocity_unit(self):
        """Handle velocity unit (m/s)."""
        from dimtensor.io.arrow import to_arrow_array, from_arrow_array

        arr = DimArray([10.0, 20.0], units.m / units.s)
        pa_arr = to_arrow_array(arr)
        restored = from_arrow_array(pa_arr)

        assert restored.dimension == arr.dimension

    def test_acceleration_unit(self):
        """Handle acceleration unit (m/s^2)."""
        from dimtensor.io.arrow import to_arrow_array, from_arrow_array

        arr = DimArray([9.8], units.m / units.s**2)
        pa_arr = to_arrow_array(arr)
        restored = from_arrow_array(pa_arr)

        assert restored.dimension == arr.dimension

    def test_force_unit(self):
        """Handle force unit (N = kg*m/s^2)."""
        from dimtensor.io.arrow import to_arrow_array, from_arrow_array

        arr = DimArray([100.0], units.N)
        pa_arr = to_arrow_array(arr)
        restored = from_arrow_array(pa_arr)

        assert restored.dimension == units.N.dimension

    def test_energy_unit(self):
        """Handle energy unit (J = kg*m^2/s^2)."""
        from dimtensor.io.arrow import to_arrow_array, from_arrow_array

        arr = DimArray([1000.0], units.J)
        pa_arr = to_arrow_array(arr)
        restored = from_arrow_array(pa_arr)

        assert restored.dimension == units.J.dimension


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_array(self):
        """Handle empty arrays."""
        from dimtensor.io.arrow import to_arrow_array, from_arrow_array

        arr = DimArray([], units.m)
        pa_arr = to_arrow_array(arr)
        restored = from_arrow_array(pa_arr)

        assert len(restored.data) == 0
        assert restored.unit.symbol == "m"

    def test_single_element(self):
        """Handle single element arrays."""
        from dimtensor.io.arrow import to_arrow_array, from_arrow_array

        arr = DimArray([42.0], units.kg)
        pa_arr = to_arrow_array(arr)
        restored = from_arrow_array(pa_arr)

        assert restored.data[0] == 42.0
        assert restored.unit.symbol == "kg"

    def test_large_array(self):
        """Handle larger arrays."""
        from dimtensor.io.arrow import to_arrow_table, from_arrow_table

        arr = DimArray(np.random.randn(10000), units.m)

        table = to_arrow_table({"large": arr})
        restored = from_arrow_table(table)

        assert np.allclose(restored["large"].data, arr.data)

    def test_very_small_values(self):
        """Handle very small values."""
        from dimtensor.io.arrow import to_arrow_array, from_arrow_array

        arr = DimArray([1e-300, 1e-200, 1e-100], units.m)
        pa_arr = to_arrow_array(arr)
        restored = from_arrow_array(pa_arr)

        assert np.allclose(restored.data, arr.data)

    def test_very_large_values(self):
        """Handle very large values."""
        from dimtensor.io.arrow import to_arrow_array, from_arrow_array

        arr = DimArray([1e100, 1e200, 1e300], units.m)
        pa_arr = to_arrow_array(arr)
        restored = from_arrow_array(pa_arr)

        assert np.allclose(restored.data, arr.data)


class TestArrowBuffer:
    """Tests for zero-copy buffer conversion."""

    def test_from_arrow_buffer_basic(self):
        """Create DimArray from Arrow buffer."""
        import pyarrow as pa
        from dimtensor.io.arrow import from_arrow_buffer

        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        buffer = pa.py_buffer(data.tobytes())

        arr = from_arrow_buffer(buffer, np.float64, units.m)

        assert np.allclose(arr.data, [1.0, 2.0, 3.0])
        assert arr.unit.symbol == "m"

    def test_from_arrow_buffer_with_shape(self):
        """Reshape buffer during conversion."""
        import pyarrow as pa
        from dimtensor.io.arrow import from_arrow_buffer

        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        buffer = pa.py_buffer(data.tobytes())

        arr = from_arrow_buffer(buffer, np.float64, units.m, shape=(2, 2))

        assert arr.shape == (2, 2)
        assert np.allclose(arr.data, [[1.0, 2.0], [3.0, 4.0]])

    def test_from_arrow_buffer_invalid_shape(self):
        """Raise error for incompatible shape."""
        import pyarrow as pa
        from dimtensor.io.arrow import from_arrow_buffer

        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        buffer = pa.py_buffer(data.tobytes())

        with pytest.raises(ValueError, match="incompatible"):
            from_arrow_buffer(buffer, np.float64, units.m, shape=(2, 2))


class TestSchemaUtils:
    """Tests for schema utility functions."""

    def test_get_arrow_schema_with_units(self):
        """Get schema with unit metadata."""
        from dimtensor.io.arrow import get_arrow_schema_with_units

        arrays = {
            "distance": DimArray([1.0], units.m),
            "time": DimArray([1.0], units.s),
        }

        schema = get_arrow_schema_with_units(arrays)

        assert len(schema) == 2
        assert "distance" in schema.names
        assert "time" in schema.names

    def test_get_arrow_schema_with_uncertainty(self):
        """Schema includes uncertainty columns."""
        from dimtensor.io.arrow import get_arrow_schema_with_units

        arrays = {
            "mass": DimArray([10.0], units.kg, uncertainty=[0.1]),
        }

        schema = get_arrow_schema_with_units(arrays)

        assert "mass" in schema.names
        assert "mass_uncertainty" in schema.names

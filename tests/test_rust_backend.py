"""Tests for the Rust backend integration.

These tests verify both the Python fallback and (when available) the Rust
implementation produce correct results.
"""

import numpy as np
import pytest

from dimtensor import DimArray, units
from dimtensor._rust import (
    HAS_RUST_BACKEND,
    add_arrays,
    sub_arrays,
    mul_arrays,
    div_arrays,
    dimensions_compatible,
)
from dimtensor.core.dimensions import Dimension, DIMENSIONLESS


class TestRustBackendDetection:
    """Test Rust backend detection."""

    def test_has_rust_backend_is_bool(self):
        """HAS_RUST_BACKEND should be a boolean."""
        assert isinstance(HAS_RUST_BACKEND, bool)

    def test_fallback_available(self):
        """Python fallback should always be available."""
        # These functions should work regardless of Rust availability
        dim = Dimension(length=1)
        assert dimensions_compatible(dim, dim)


class TestDimensionCompatibility:
    """Test dimension compatibility checking."""

    def test_same_dimensions_compatible(self):
        """Same dimensions should be compatible."""
        dim = Dimension(length=1, time=-1)  # m/s
        assert dimensions_compatible(dim, dim)

    def test_different_dimensions_incompatible(self):
        """Different dimensions should not be compatible."""
        dim_a = Dimension(length=1)  # m
        dim_b = Dimension(time=1)  # s
        assert not dimensions_compatible(dim_a, dim_b)

    def test_dimensionless_compatible(self):
        """Dimensionless dimensions should be compatible."""
        assert dimensions_compatible(DIMENSIONLESS, DIMENSIONLESS)


class TestArrayOperations:
    """Test array operations with dimension checking."""

    def test_add_arrays_same_dimension(self):
        """Add arrays with same dimension."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        dim = Dimension(length=1)

        result = add_arrays(a, b, dim, dim)

        np.testing.assert_array_equal(result, [5.0, 7.0, 9.0])

    def test_add_arrays_incompatible_dimension(self):
        """Add arrays with incompatible dimensions should raise."""
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        dim_a = Dimension(length=1)
        dim_b = Dimension(time=1)

        with pytest.raises(ValueError, match="incompatible dimensions"):
            add_arrays(a, b, dim_a, dim_b)

    def test_sub_arrays_same_dimension(self):
        """Subtract arrays with same dimension."""
        a = np.array([5.0, 7.0, 9.0])
        b = np.array([1.0, 2.0, 3.0])
        dim = Dimension(mass=1)

        result = sub_arrays(a, b, dim, dim)

        np.testing.assert_array_equal(result, [4.0, 5.0, 6.0])

    def test_sub_arrays_incompatible_dimension(self):
        """Subtract arrays with incompatible dimensions should raise."""
        a = np.array([1.0])
        b = np.array([2.0])
        dim_a = Dimension(length=1)
        dim_b = Dimension(mass=1)

        with pytest.raises(ValueError, match="incompatible dimensions"):
            sub_arrays(a, b, dim_a, dim_b)

    def test_mul_arrays(self):
        """Multiply arrays and combine dimensions."""
        a = np.array([2.0, 3.0])
        b = np.array([4.0, 5.0])
        dim_a = Dimension(length=1)  # m
        dim_b = Dimension(time=-1)  # 1/s

        result, result_dim = mul_arrays(a, b, dim_a, dim_b)

        np.testing.assert_array_equal(result, [8.0, 15.0])
        assert result_dim == Dimension(length=1, time=-1)  # m/s

    def test_div_arrays(self):
        """Divide arrays and combine dimensions."""
        a = np.array([10.0, 20.0])
        b = np.array([2.0, 4.0])
        dim_a = Dimension(length=1)  # m
        dim_b = Dimension(time=1)  # s

        result, result_dim = div_arrays(a, b, dim_a, dim_b)

        np.testing.assert_array_equal(result, [5.0, 5.0])
        assert result_dim == Dimension(length=1, time=-1)  # m/s


class TestIntegrationWithDimArray:
    """Test that DimArray operations work correctly."""

    def test_dimarray_add(self):
        """DimArray addition should work."""
        a = DimArray([1.0, 2.0], units.m)
        b = DimArray([3.0, 4.0], units.m)

        result = a + b

        np.testing.assert_array_equal(result.data, [4.0, 6.0])
        assert result.unit == units.m

    def test_dimarray_mul(self):
        """DimArray multiplication should work."""
        a = DimArray([2.0, 3.0], units.m)
        b = DimArray([4.0, 5.0], units.s)

        result = a * b

        np.testing.assert_array_equal(result.data, [8.0, 15.0])
        # m * s

    def test_dimarray_div(self):
        """DimArray division should work."""
        distance = DimArray([10.0, 20.0], units.m)
        time = DimArray([2.0, 4.0], units.s)

        velocity = distance / time

        np.testing.assert_array_equal(velocity.data, [5.0, 5.0])
        # Should be m/s


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_arrays(self):
        """Empty arrays should work."""
        a = np.array([])
        b = np.array([])
        dim = DIMENSIONLESS

        result = add_arrays(a, b, dim, dim)

        assert len(result) == 0

    def test_scalar_arrays(self):
        """Scalar (0-d) arrays should work."""
        a = np.array(1.0)
        b = np.array(2.0)
        dim = Dimension(length=1)

        result = add_arrays(a, b, dim, dim)

        assert float(result) == 3.0

    def test_multidimensional_arrays(self):
        """Multi-dimensional arrays should work."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        dim = Dimension(mass=1)

        result = add_arrays(a, b, dim, dim)

        expected = np.array([[6.0, 8.0], [10.0, 12.0]])
        np.testing.assert_array_equal(result, expected)


@pytest.mark.skipif(not HAS_RUST_BACKEND, reason="Rust backend not available")
class TestRustSpecific:
    """Tests that only run when Rust backend is available."""

    def test_rust_dimension_creation(self):
        """RustDimension should be creatable."""
        from dimtensor._rust import RustDimension

        dim = RustDimension(length=1, mass=2, time=-1)
        assert dim is not None

    def test_rust_vs_python_consistency(self):
        """Rust and Python should produce same results."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        dim = Dimension(length=1)

        # Rust result
        rust_result = add_arrays(a, b, dim, dim)

        # Python result (direct numpy)
        python_result = a + b

        np.testing.assert_array_equal(rust_result, python_result)

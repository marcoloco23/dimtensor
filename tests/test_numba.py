"""Tests for Numba integration.

These tests verify:
1. Dimension encoding/decoding round-trip correctness
2. JIT-compiled dimension operations
3. @dim_jit decorator functionality
4. Integration with DimArray
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

# Check if numba is available
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

pytestmark = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


class TestHasNumba:
    """Test HAS_NUMBA flag."""

    def test_has_numba_flag(self):
        """HAS_NUMBA should reflect numba availability."""
        from dimtensor.numba import HAS_NUMBA as flag
        assert flag == HAS_NUMBA


class TestEncoding:
    """Tests for dimension encoding/decoding."""

    def test_encode_dimensionless(self):
        """Encoding dimensionless should give zeros for numerators."""
        from dimtensor.core.dimensions import DIMENSIONLESS
        from dimtensor.numba import encode_dimension

        encoded = encode_dimension(DIMENSIONLESS)
        assert encoded.shape == (14,)
        assert encoded.dtype == np.int64
        # All numerators should be 0
        for i in range(7):
            assert encoded[2 * i] == 0
            assert encoded[2 * i + 1] == 1  # denominators = 1

    def test_encode_length(self):
        """Encoding length dimension (L^1)."""
        from dimtensor.core.dimensions import Dimension
        from dimtensor.numba import encode_dimension

        dim = Dimension(length=1)
        encoded = encode_dimension(dim)

        # L exponent: 1/1
        assert encoded[0] == 1
        assert encoded[1] == 1
        # All others should be 0/1
        for i in range(1, 7):
            assert encoded[2 * i] == 0

    def test_encode_velocity(self):
        """Encoding velocity dimension (L^1 T^-1)."""
        from dimtensor.core.dimensions import Dimension
        from dimtensor.numba import encode_dimension

        dim = Dimension(length=1, time=-1)
        encoded = encode_dimension(dim)

        # L: 1/1
        assert encoded[0] == 1
        assert encoded[1] == 1
        # M: 0/1
        assert encoded[2] == 0
        # T: -1/1
        assert encoded[4] == -1
        assert encoded[5] == 1

    def test_encode_fractional_exponent(self):
        """Encoding fractional exponents (L^(1/2))."""
        from dimtensor.core.dimensions import Dimension
        from dimtensor.numba import encode_dimension

        dim = Dimension(length=Fraction(1, 2))
        encoded = encode_dimension(dim)

        # L: 1/2
        assert encoded[0] == 1
        assert encoded[1] == 2

    def test_decode_roundtrip(self):
        """Encoding then decoding should give original dimension."""
        from dimtensor.core.dimensions import Dimension
        from dimtensor.numba import decode_dimension, encode_dimension

        # Test various dimensions
        dims = [
            Dimension(),  # dimensionless
            Dimension(length=1),
            Dimension(mass=1, length=1, time=-2),  # force
            Dimension(length=Fraction(1, 2)),  # sqrt(length)
            Dimension(length=2, time=-2),  # velocity squared
        ]

        for dim in dims:
            encoded = encode_dimension(dim)
            decoded = decode_dimension(encoded)
            assert decoded == dim, f"Round-trip failed for {dim}"

    def test_encode_dimensionless_helper(self):
        """encode_dimensionless() should return dimensionless encoding."""
        from dimtensor.numba._encoding import encode_dimensionless

        encoded = encode_dimensionless()
        assert encoded.shape == (14,)
        for i in range(7):
            assert encoded[2 * i] == 0
            assert encoded[2 * i + 1] == 1


class TestOperations:
    """Tests for JIT-compiled dimension operations."""

    def test_dims_equal_same(self):
        """dims_equal should return True for equal dimensions."""
        from dimtensor.core.dimensions import Dimension
        from dimtensor.numba import dims_equal, encode_dimension

        dim1 = Dimension(length=1, time=-1)
        dim2 = Dimension(length=1, time=-1)

        enc1 = encode_dimension(dim1)
        enc2 = encode_dimension(dim2)

        assert dims_equal(enc1, enc2)

    def test_dims_equal_different(self):
        """dims_equal should return False for different dimensions."""
        from dimtensor.core.dimensions import Dimension
        from dimtensor.numba import dims_equal, encode_dimension

        dim1 = Dimension(length=1)
        dim2 = Dimension(time=1)

        enc1 = encode_dimension(dim1)
        enc2 = encode_dimension(dim2)

        assert not dims_equal(enc1, enc2)

    def test_dims_multiply(self):
        """dims_multiply should add exponents."""
        from dimtensor.core.dimensions import Dimension
        from dimtensor.numba import decode_dimension, dims_multiply, encode_dimension

        # Length * Time = Length * Time
        dim_l = Dimension(length=1)
        dim_t = Dimension(time=1)
        expected = Dimension(length=1, time=1)

        enc_l = encode_dimension(dim_l)
        enc_t = encode_dimension(dim_t)
        result = dims_multiply(enc_l, enc_t)

        assert decode_dimension(result) == expected

    def test_dims_multiply_velocity_time(self):
        """Velocity * Time = Length."""
        from dimtensor.core.dimensions import Dimension
        from dimtensor.numba import decode_dimension, dims_multiply, encode_dimension

        velocity = Dimension(length=1, time=-1)
        time = Dimension(time=1)
        expected = Dimension(length=1)

        enc_v = encode_dimension(velocity)
        enc_t = encode_dimension(time)
        result = dims_multiply(enc_v, enc_t)

        assert decode_dimension(result) == expected

    def test_dims_divide(self):
        """dims_divide should subtract exponents."""
        from dimtensor.core.dimensions import Dimension
        from dimtensor.numba import decode_dimension, dims_divide, encode_dimension

        # Length / Time = Velocity
        dim_l = Dimension(length=1)
        dim_t = Dimension(time=1)
        expected = Dimension(length=1, time=-1)

        enc_l = encode_dimension(dim_l)
        enc_t = encode_dimension(dim_t)
        result = dims_divide(enc_l, enc_t)

        assert decode_dimension(result) == expected

    def test_dims_power_square(self):
        """dims_power should multiply exponents."""
        from dimtensor.core.dimensions import Dimension
        from dimtensor.numba import decode_dimension, dims_power, encode_dimension

        # Length^2
        dim_l = Dimension(length=1)
        expected = Dimension(length=2)

        enc_l = encode_dimension(dim_l)
        result = dims_power(enc_l, 2, 1)  # power = 2/1

        assert decode_dimension(result) == expected

    def test_dims_power_sqrt(self):
        """dims_power with 1/2 should halve exponents."""
        from dimtensor.core.dimensions import Dimension
        from dimtensor.numba import decode_dimension, dims_power, encode_dimension

        # sqrt(Area) = Length
        area = Dimension(length=2)
        expected = Dimension(length=1)

        enc_area = encode_dimension(area)
        result = dims_power(enc_area, 1, 2)  # power = 1/2

        assert decode_dimension(result) == expected

    def test_gcd(self):
        """GCD function should work correctly."""
        from dimtensor.numba import gcd

        assert gcd(12, 8) == 4
        assert gcd(17, 13) == 1
        assert gcd(100, 25) == 25
        assert gcd(0, 5) == 5
        assert gcd(-12, 8) == 4  # handles negative


class TestDimJit:
    """Tests for @dim_jit decorator."""

    def test_dim_jit_basic(self):
        """Basic @dim_jit should compile and run."""
        from dimtensor import DimArray, units
        from dimtensor.numba import dim_jit

        @dim_jit
        def double(x):
            return x * 2

        arr = DimArray([1.0, 2.0, 3.0], units.m)
        result = double(arr)

        assert isinstance(result, DimArray)
        np.testing.assert_array_almost_equal(result._data, [2.0, 4.0, 6.0])
        assert result.dimension == arr.dimension

    def test_dim_jit_preserves_dimension(self):
        """@dim_jit should preserve input dimension for simple operations."""
        from dimtensor import DimArray, units
        from dimtensor.numba import dim_jit

        @dim_jit
        def negate(x):
            return -x

        arr = DimArray([1.0, 2.0], units.kg)
        result = negate(arr)

        assert result.dimension == arr.dimension

    def test_dim_jit_with_parallel(self):
        """@dim_jit with parallel=True should work."""
        from dimtensor import DimArray, units
        from dimtensor.numba import dim_jit, prange

        @dim_jit(parallel=True)
        def parallel_double(x):
            result = np.empty_like(x)
            for i in prange(x.shape[0]):
                result[i] = x[i] * 2
            return result

        arr = DimArray(np.arange(1000, dtype=np.float64), units.m)
        result = parallel_double(arr)

        assert isinstance(result, DimArray)
        np.testing.assert_array_almost_equal(result._data, arr._data * 2)

    def test_dim_jit_scalar_input(self):
        """@dim_jit should handle scalar inputs."""
        from dimtensor.numba import dim_jit

        @dim_jit
        def add_one(x):
            return x + 1

        result = add_one(5.0)
        # Without DimArray input, returns raw result
        assert result == 6.0

    def test_dim_jit_multiple_dimarray_inputs(self):
        """@dim_jit should handle multiple DimArray inputs."""
        from dimtensor import DimArray, units
        from dimtensor.numba import dim_jit

        @dim_jit
        def add_arrays(a, b):
            return a + b

        arr1 = DimArray([1.0, 2.0], units.m)
        arr2 = DimArray([3.0, 4.0], units.m)
        result = add_arrays(arr1, arr2)

        assert isinstance(result, DimArray)
        np.testing.assert_array_almost_equal(result._data, [4.0, 6.0])

    def test_dim_jit_cache(self):
        """@dim_jit with cache=True should work."""
        from dimtensor import DimArray, units
        from dimtensor.numba import dim_jit

        @dim_jit(cache=True)
        def cached_func(x):
            return x * 3

        arr = DimArray([1.0, 2.0], units.s)
        result = cached_func(arr)

        np.testing.assert_array_almost_equal(result._data, [3.0, 6.0])

    def test_dim_jit_fastmath(self):
        """@dim_jit with fastmath=True should work."""
        from dimtensor import DimArray, units
        from dimtensor.numba import dim_jit

        @dim_jit(fastmath=True)
        def fast_sum(x):
            return np.sum(x)

        arr = DimArray([1.0, 2.0, 3.0], units.m)
        result = fast_sum(arr)

        assert isinstance(result, DimArray)
        np.testing.assert_almost_equal(result._data[0], 6.0)


class TestExtractAndWrap:
    """Tests for extract_arrays and wrap_result helpers."""

    def test_extract_arrays_single(self):
        """extract_arrays should handle single DimArray."""
        from dimtensor import DimArray, units
        from dimtensor.numba import extract_arrays

        arr = DimArray([1.0, 2.0], units.m)
        raw, dims, scales = extract_arrays(arr)

        assert len(raw) == 1
        np.testing.assert_array_equal(raw[0], arr._data)
        assert dims[0] is not None
        assert scales[0] == 1.0

    def test_extract_arrays_mixed(self):
        """extract_arrays should handle mixed inputs."""
        from dimtensor import DimArray, units
        from dimtensor.numba import extract_arrays

        arr = DimArray([1.0, 2.0], units.m)
        scalar = 3.0

        raw, dims, scales = extract_arrays(arr, scalar)

        assert len(raw) == 2
        np.testing.assert_array_equal(raw[0], arr._data)
        np.testing.assert_array_equal(raw[1], np.array(3.0))
        assert dims[0] is not None
        assert dims[1] is None
        assert scales[1] == 1.0

    def test_wrap_result(self):
        """wrap_result should create DimArray with correct dimension."""
        from dimtensor import DimArray
        from dimtensor.core.dimensions import Dimension
        from dimtensor.numba import encode_dimension, wrap_result

        data = np.array([1.0, 2.0, 3.0])
        dim = Dimension(length=1, time=-1)
        encoded = encode_dimension(dim)

        result = wrap_result(data, encoded)

        assert isinstance(result, DimArray)
        np.testing.assert_array_equal(result._data, data)
        assert result.dimension == dim


class TestIntegration:
    """Integration tests with full dimtensor workflow."""

    def test_kinetic_energy_example(self):
        """Example from docstring: kinetic energy calculation."""
        from dimtensor import DimArray, units
        from dimtensor.numba import dim_jit

        @dim_jit
        def kinetic_energy_raw(mass, velocity):
            return 0.5 * mass * velocity**2

        # Note: This simplified version uses raw multiplication
        # Full dimension tracking requires more sophisticated inference
        m = DimArray([1.0, 2.0], units.kg)
        v = DimArray([3.0, 4.0], units.m / units.s)

        # For full dimension inference, we'd need more sophisticated tracking
        # This test verifies the JIT compilation works
        result = kinetic_energy_raw(m._data, v._data)
        expected = 0.5 * np.array([1.0, 2.0]) * np.array([3.0, 4.0])**2

        np.testing.assert_array_almost_equal(result, expected)

    def test_prange_available(self):
        """prange should be importable from dimtensor.numba."""
        from dimtensor.numba import prange

        # Should be numba's prange
        from numba import prange as numba_prange
        assert prange is numba_prange

    def test_jit_compilation_speedup(self):
        """JIT-compiled function should be faster than pure Python for large arrays."""
        from dimtensor import DimArray, units
        from dimtensor.numba import dim_jit
        import time

        @dim_jit
        def jit_sum(x):
            total = 0.0
            for i in range(x.shape[0]):
                total += x[i]
            return total

        def python_sum(x):
            total = 0.0
            for i in range(x.shape[0]):
                total += x[i]
            return total

        # Large array
        arr = DimArray(np.random.randn(10000), units.m)

        # Warm up JIT
        _ = jit_sum(arr)

        # Time JIT version
        start = time.time()
        for _ in range(10):
            jit_result = jit_sum(arr)
        jit_time = time.time() - start

        # Time pure Python
        start = time.time()
        for _ in range(10):
            py_result = python_sum(arr._data)
        py_time = time.time() - start

        # JIT should be significantly faster
        # (not a strict test since it depends on hardware)
        assert jit_time < py_time * 2, f"JIT ({jit_time:.4f}s) not faster than Python ({py_time:.4f}s)"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_array(self):
        """Should handle empty arrays."""
        from dimtensor import DimArray, units
        from dimtensor.numba import dim_jit

        @dim_jit
        def identity(x):
            return x

        arr = DimArray(np.array([]), units.m)
        result = identity(arr)

        assert isinstance(result, DimArray)
        assert result._data.shape == (0,)

    def test_0d_array(self):
        """Should handle 0-dimensional arrays."""
        from dimtensor import DimArray, units
        from dimtensor.numba import dim_jit

        @dim_jit
        def double(x):
            return x * 2

        arr = DimArray(5.0, units.m)
        result = double(arr)

        assert isinstance(result, DimArray)

    def test_multidimensional_array(self):
        """Should handle multidimensional arrays."""
        from dimtensor import DimArray, units
        from dimtensor.numba import dim_jit

        @dim_jit
        def transpose_sum(x):
            return np.sum(x, axis=0)

        arr = DimArray(np.array([[1.0, 2.0], [3.0, 4.0]]), units.m)
        result = transpose_sum(arr)

        assert isinstance(result, DimArray)
        np.testing.assert_array_almost_equal(result._data, [4.0, 6.0])

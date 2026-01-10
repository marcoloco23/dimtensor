"""Tests for Dask DaskDimArray integration."""

import numpy as np
import pytest

try:
    import dask
    import dask.array as da
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

pytestmark = pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")

if DASK_AVAILABLE:
    from dimtensor.dask import DaskDimArray

from dimtensor import units
from dimtensor.errors import DimensionError, UnitConversionError


class TestDaskDimArrayCreation:
    """Tests for DaskDimArray creation."""

    def test_from_numpy_array(self):
        """Create DaskDimArray from numpy array."""
        arr = np.array([1.0, 2.0, 3.0])
        dda = DaskDimArray(arr, units.m)
        assert dda.shape == (3,)
        assert dda.unit == units.m

    def test_from_dask_array(self):
        """Create DaskDimArray from dask array."""
        arr = da.from_array([1.0, 2.0, 3.0], chunks=2)
        dda = DaskDimArray(arr, units.m)
        assert dda.shape == (3,)
        assert dda.unit == units.m

    def test_from_list(self):
        """Create DaskDimArray from list."""
        dda = DaskDimArray([1.0, 2.0, 3.0], units.m)
        assert dda.shape == (3,)
        assert dda.unit == units.m

    def test_from_scalar(self):
        """Create DaskDimArray from scalar."""
        dda = DaskDimArray(5.0, units.kg)
        assert dda.size == 1
        assert dda.unit == units.kg

    def test_default_dimensionless(self):
        """Default unit is dimensionless."""
        dda = DaskDimArray(np.array([1.0]))
        assert dda.is_dimensionless

    def test_with_chunks(self):
        """Create with explicit chunk size."""
        arr = np.zeros((100, 100))
        dda = DaskDimArray(arr, units.m, chunks=(10, 10))
        assert dda.chunksize == (10, 10)


class TestDaskDimArrayProperties:
    """Tests for DaskDimArray properties."""

    def test_shape(self):
        """Test shape property."""
        dda = DaskDimArray(np.zeros((2, 3, 4)), units.m)
        assert dda.shape == (2, 3, 4)

    def test_ndim(self):
        """Test ndim property."""
        dda = DaskDimArray(np.zeros((2, 3)), units.m)
        assert dda.ndim == 2

    def test_size(self):
        """Test size property."""
        dda = DaskDimArray(np.zeros((2, 3)), units.m)
        assert dda.size == 6

    def test_dimension(self):
        """Test dimension property."""
        dda = DaskDimArray([1.0], units.m / units.s)
        assert dda.dimension == (units.m / units.s).dimension

    def test_dtype(self):
        """Test dtype property."""
        dda = DaskDimArray(np.array([1.0, 2.0], dtype=np.float32), units.m)
        assert dda.dtype == np.float32

    def test_chunks(self):
        """Test chunks property."""
        dda = DaskDimArray(np.zeros((100, 100)), units.m, chunks=(25, 50))
        assert dda.chunks == ((25, 25, 25, 25), (50, 50))


class TestLazyEvaluation:
    """Tests for lazy evaluation behavior."""

    def test_operations_are_lazy(self):
        """Operations should not trigger computation."""
        arr = np.array([1.0, 2.0, 3.0])
        dda = DaskDimArray(arr, units.m)

        # Perform operation
        result = dda * 2.0

        # Result should still be a DaskDimArray
        assert isinstance(result, DaskDimArray)
        # Should have underlying dask array
        assert isinstance(result._data, da.Array)

    def test_chained_operations_lazy(self):
        """Chained operations should remain lazy."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        dda = DaskDimArray(arr, units.m)

        # Chain of operations
        result = ((dda ** 2).sum(axis=1)).sqrt()

        # Still lazy
        assert isinstance(result, DaskDimArray)
        assert isinstance(result._data, da.Array)

    def test_compute_triggers_evaluation(self):
        """compute() should trigger evaluation."""
        from dimtensor.core.dimarray import DimArray as NumpyDimArray

        arr = np.array([1.0, 2.0, 3.0])
        dda = DaskDimArray(arr, units.m)
        result = dda * 2.0

        # Compute
        computed = result.compute()

        # Should be numpy DimArray
        assert isinstance(computed, NumpyDimArray)
        assert np.allclose(computed.data, np.array([2.0, 4.0, 6.0]))
        assert computed.unit == units.m


class TestDaskDimArrayArithmetic:
    """Tests for arithmetic operations."""

    def test_add_same_unit(self):
        """Add arrays with same unit."""
        a = DaskDimArray([1.0, 2.0], units.m)
        b = DaskDimArray([3.0, 4.0], units.m)
        c = a + b
        result = c.compute()
        assert np.allclose(result.data, np.array([4.0, 6.0]))
        assert result.unit == units.m

    def test_add_compatible_units(self):
        """Add arrays with compatible units (auto-convert)."""
        a = DaskDimArray([1000.0], units.m)
        b = DaskDimArray([1.0], units.km)
        c = a + b
        result = c.compute()
        assert np.allclose(result.data, np.array([2000.0]))
        assert result.unit == units.m

    def test_add_incompatible_raises(self):
        """Adding incompatible dimensions raises error immediately."""
        a = DaskDimArray([1.0], units.m)
        b = DaskDimArray([1.0], units.s)
        with pytest.raises(DimensionError):
            a + b

    def test_subtract(self):
        """Subtract arrays."""
        a = DaskDimArray([5.0, 6.0], units.m)
        b = DaskDimArray([1.0, 2.0], units.m)
        c = a - b
        result = c.compute()
        assert np.allclose(result.data, np.array([4.0, 4.0]))

    def test_multiply_dimensions(self):
        """Multiply arrays - dimensions multiply."""
        length = DaskDimArray([2.0], units.m)
        width = DaskDimArray([3.0], units.m)
        area = length * width
        result = area.compute()
        assert result.dimension == (units.m ** 2).dimension
        assert np.allclose(result.data, np.array([6.0]))

    def test_multiply_scalar(self):
        """Multiply by scalar."""
        dda = DaskDimArray([1.0, 2.0], units.m)
        result = (dda * 3.0).compute()
        assert np.allclose(result.data, np.array([3.0, 6.0]))
        assert result.unit == units.m

    def test_divide_dimensions(self):
        """Divide arrays - dimensions divide."""
        distance = DaskDimArray([10.0], units.m)
        time = DaskDimArray([2.0], units.s)
        velocity = distance / time
        result = velocity.compute()
        assert result.dimension == (units.m / units.s).dimension
        assert np.allclose(result.data, np.array([5.0]))

    def test_power(self):
        """Power operation scales dimension."""
        length = DaskDimArray([2.0], units.m)
        area = length ** 2
        result = area.compute()
        assert result.dimension == (units.m ** 2).dimension
        assert np.allclose(result.data, np.array([4.0]))

    def test_sqrt(self):
        """Square root halves dimension."""
        area = DaskDimArray([4.0], units.m ** 2)
        length = area.sqrt()
        result = length.compute()
        assert result.dimension == units.m.dimension
        assert np.allclose(result.data, np.array([2.0]))

    def test_negation(self):
        """Negation preserves unit."""
        dda = DaskDimArray([1.0, -2.0], units.m)
        neg = -dda
        result = neg.compute()
        assert np.allclose(result.data, np.array([-1.0, 2.0]))
        assert result.unit == units.m

    def test_abs(self):
        """Absolute value preserves unit."""
        dda = DaskDimArray([-1.0, 2.0, -3.0], units.m)
        abs_dda = abs(dda)
        result = abs_dda.compute()
        assert np.allclose(result.data, np.array([1.0, 2.0, 3.0]))
        assert result.unit == units.m


class TestDaskDimArrayReductions:
    """Tests for reduction operations."""

    def test_sum(self):
        """Sum preserves unit."""
        dda = DaskDimArray([1.0, 2.0, 3.0], units.m)
        result = dda.sum().compute()
        # Use _data for scalar comparisons (data property fails on scalars)
        assert np.allclose(result._data, 6.0)
        assert result.unit == units.m

    def test_sum_axis(self):
        """Sum along axis."""
        dda = DaskDimArray([[1.0, 2.0], [3.0, 4.0]], units.m)
        result = dda.sum(axis=0).compute()
        assert np.allclose(result.data, np.array([4.0, 6.0]))
        assert result.unit == units.m

    def test_mean(self):
        """Mean preserves unit."""
        dda = DaskDimArray([1.0, 2.0, 3.0], units.m)
        result = dda.mean().compute()
        # Use _data for scalar comparisons (data property fails on scalars)
        assert np.allclose(result._data, 2.0)
        assert result.unit == units.m

    def test_std(self):
        """Std preserves unit."""
        dda = DaskDimArray([1.0, 2.0, 3.0], units.m)
        result = dda.std().compute()
        assert result.unit == units.m

    def test_var_squares_unit(self):
        """Variance squares unit."""
        dda = DaskDimArray([1.0, 2.0, 3.0], units.m)
        result = dda.var().compute()
        assert result.dimension == (units.m ** 2).dimension

    def test_min(self):
        """Min preserves unit."""
        dda = DaskDimArray([3.0, 1.0, 2.0], units.m)
        result = dda.min().compute()
        # Use _data for scalar comparisons (data property fails on scalars)
        assert np.allclose(result._data, 1.0)
        assert result.unit == units.m

    def test_max(self):
        """Max preserves unit."""
        dda = DaskDimArray([3.0, 1.0, 2.0], units.m)
        result = dda.max().compute()
        # Use _data for scalar comparisons (data property fails on scalars)
        assert np.allclose(result._data, 3.0)
        assert result.unit == units.m


class TestDaskDimArrayReshaping:
    """Tests for reshaping operations."""

    def test_reshape(self):
        """Reshape preserves unit."""
        dda = DaskDimArray(np.zeros((2, 3)), units.m)
        reshaped = dda.reshape((6,))
        assert reshaped.shape == (6,)
        assert reshaped.unit == units.m

    def test_transpose(self):
        """Transpose preserves unit."""
        dda = DaskDimArray(np.zeros((2, 3)), units.m)
        transposed = dda.transpose()
        assert transposed.shape == (3, 2)
        assert transposed.unit == units.m

    def test_flatten(self):
        """Flatten preserves unit."""
        dda = DaskDimArray(np.zeros((2, 3)), units.m)
        flat = dda.flatten()
        assert flat.shape == (6,)
        assert flat.unit == units.m

    def test_rechunk(self):
        """Rechunk changes chunk sizes."""
        dda = DaskDimArray(np.zeros((100, 100)), units.m, chunks=(10, 10))
        rechunked = dda.rechunk((25, 25))
        assert rechunked.chunksize == (25, 25)
        assert rechunked.unit == units.m


class TestDaskDimArrayUnitConversion:
    """Tests for unit conversion."""

    def test_to_unit(self):
        """Convert to compatible unit."""
        dda = DaskDimArray([1000.0], units.m)
        dda_km = dda.to(units.km)
        result = dda_km.compute()
        assert np.allclose(result.data, np.array([1.0]))
        assert result.unit == units.km

    def test_to_unit_incompatible_raises(self):
        """Converting to incompatible unit raises error."""
        dda = DaskDimArray([1.0], units.m)
        with pytest.raises(UnitConversionError):
            dda.to(units.s)

    def test_to_base_units(self):
        """Convert to base units."""
        dda = DaskDimArray([1.0], units.km)
        base = dda.to_base_units()
        result = base.compute()
        assert np.allclose(result.data, np.array([1000.0]))

    def test_magnitude(self):
        """Get magnitude as dask array."""
        dda = DaskDimArray([1.0, 2.0], units.m)
        mag = dda.magnitude()
        assert isinstance(mag, da.Array)
        assert np.allclose(mag.compute(), np.array([1.0, 2.0]))


class TestDaskDimArrayComparison:
    """Tests for comparison operations."""

    def test_eq_same_unit(self):
        """Equality with same unit."""
        a = DaskDimArray([1.0, 2.0], units.m)
        b = DaskDimArray([1.0, 3.0], units.m)
        result = (a == b).compute()
        assert result[0] == True
        assert result[1] == False

    def test_lt(self):
        """Less than comparison."""
        a = DaskDimArray([1.0, 3.0], units.m)
        b = DaskDimArray([2.0, 2.0], units.m)
        result = (a < b).compute()
        assert result[0] == True
        assert result[1] == False

    def test_le(self):
        """Less than or equal comparison."""
        a = DaskDimArray([1.0, 2.0, 3.0], units.m)
        b = DaskDimArray([2.0, 2.0, 2.0], units.m)
        result = (a <= b).compute()
        assert result[0] == True
        assert result[1] == True
        assert result[2] == False

    def test_gt(self):
        """Greater than comparison."""
        a = DaskDimArray([1.0, 3.0], units.m)
        b = DaskDimArray([2.0, 2.0], units.m)
        result = (a > b).compute()
        assert result[0] == False
        assert result[1] == True

    def test_ge(self):
        """Greater than or equal comparison."""
        a = DaskDimArray([1.0, 2.0, 3.0], units.m)
        b = DaskDimArray([2.0, 2.0, 2.0], units.m)
        result = (a >= b).compute()
        assert result[0] == False
        assert result[1] == True
        assert result[2] == True

    def test_compare_incompatible_raises(self):
        """Comparing incompatible dimensions raises error."""
        a = DaskDimArray([1.0], units.m)
        b = DaskDimArray([1.0], units.s)
        with pytest.raises(DimensionError):
            a < b


class TestDaskDimArrayIndexing:
    """Tests for indexing operations."""

    def test_getitem(self):
        """Indexing preserves unit."""
        dda = DaskDimArray([1.0, 2.0, 3.0], units.m)
        item = dda[1]
        result = item.compute()
        # Use _data for scalar comparisons (data property fails on scalars)
        assert np.allclose(result._data, 2.0)
        assert result.unit == units.m

    def test_slice(self):
        """Slicing preserves unit."""
        dda = DaskDimArray([1.0, 2.0, 3.0, 4.0], units.m)
        sliced = dda[1:3]
        assert sliced.shape == (2,)
        assert sliced.unit == units.m

    def test_len(self):
        """Test length."""
        dda = DaskDimArray(np.zeros((5, 3)), units.m)
        assert len(dda) == 5


class TestDaskDimArrayLinearAlgebra:
    """Tests for linear algebra operations."""

    def test_dot(self):
        """Dot product multiplies dimensions."""
        a = DaskDimArray([1.0, 2.0, 3.0], units.m)
        b = DaskDimArray([4.0, 5.0, 6.0], units.s)
        c = a.dot(b)
        result = c.compute()
        assert result.dimension == (units.m * units.s).dimension
        # Use _data for scalar comparisons (data property fails on scalars)
        assert np.allclose(result._data, 32.0)

    def test_matmul(self):
        """Matrix multiplication multiplies dimensions."""
        a = DaskDimArray(np.ones((2, 3)), units.m)
        b = DaskDimArray(np.ones((3, 4)), units.s)
        c = a.matmul(b)
        assert c.shape == (2, 4)
        assert c.dimension == (units.m * units.s).dimension

    def test_matmul_operator(self):
        """@ operator works for matmul."""
        a = DaskDimArray(np.ones((2, 3)), units.m)
        b = DaskDimArray(np.ones((3, 4)), units.s)
        c = a @ b
        assert c.dimension == (units.m * units.s).dimension


class TestDaskDimArrayCreationMethods:
    """Tests for class creation methods."""

    def test_zeros(self):
        """Create array of zeros."""
        dda = DaskDimArray.zeros((3, 4), units.m)
        assert dda.shape == (3, 4)
        assert dda.unit == units.m
        result = dda.compute()
        assert np.allclose(result.data, np.zeros((3, 4)))

    def test_ones(self):
        """Create array of ones."""
        dda = DaskDimArray.ones((3, 4), units.m)
        assert dda.shape == (3, 4)
        assert dda.unit == units.m
        result = dda.compute()
        assert np.allclose(result.data, np.ones((3, 4)))

    def test_arange(self):
        """Create array with arange."""
        dda = DaskDimArray.arange(0, 10, 2, unit=units.m)
        result = dda.compute()
        assert np.allclose(result.data, np.array([0, 2, 4, 6, 8]))
        assert result.unit == units.m

    def test_from_array(self):
        """Create from array."""
        arr = np.array([1.0, 2.0, 3.0])
        dda = DaskDimArray.from_array(arr, units.m)
        assert dda.unit == units.m
        result = dda.compute()
        assert np.allclose(result.data, arr)


class TestDaskDimArrayConversion:
    """Tests for conversion methods."""

    def test_to_dimarray(self):
        """Convert to numpy DimArray."""
        from dimtensor.core.dimarray import DimArray as NumpyDimArray

        dda = DaskDimArray([1.0, 2.0], units.m)
        da_numpy = dda.to_dimarray()
        assert isinstance(da_numpy, NumpyDimArray)
        assert da_numpy.unit == units.m

    def test_from_dimarray(self):
        """Create from numpy DimArray."""
        from dimtensor.core.dimarray import DimArray as NumpyDimArray

        da_numpy = NumpyDimArray([1.0, 2.0, 3.0], units.m)
        dda = DaskDimArray.from_dimarray(da_numpy)
        assert isinstance(dda, DaskDimArray)
        assert dda.unit == units.m
        result = dda.compute()
        assert np.allclose(result.data, da_numpy.data)

    def test_roundtrip(self):
        """Roundtrip: DimArray -> DaskDimArray -> DimArray."""
        from dimtensor.core.dimarray import DimArray as NumpyDimArray

        original = NumpyDimArray([1.0, 2.0, 3.0], units.m)
        dda = DaskDimArray.from_dimarray(original)
        recovered = dda.to_dimarray()

        assert np.allclose(original.data, recovered.data)
        assert original.unit == recovered.unit


class TestDaskDimArrayStrings:
    """Tests for string representations."""

    def test_repr(self):
        """repr includes shape and unit."""
        dda = DaskDimArray(np.zeros((2, 3)), units.m)
        r = repr(dda)
        assert "DaskDimArray" in r
        assert "(2, 3)" in r
        assert "m" in r

    def test_str(self):
        """str shows shape and unit."""
        dda = DaskDimArray(np.zeros((2, 3)), units.m)
        s = str(dda)
        assert "DaskDimArray" in s
        assert "m" in s


class TestDaskDimArrayPersist:
    """Tests for persist behavior."""

    def test_persist_returns_dask_dimarray(self):
        """persist() returns DaskDimArray."""
        dda = DaskDimArray([1.0, 2.0, 3.0], units.m)
        result = dda * 2.0
        persisted = result.persist()
        assert isinstance(persisted, DaskDimArray)
        assert persisted.unit == units.m


class TestDimensionErrorBeforeCompute:
    """Tests that dimension errors are raised before compute."""

    def test_add_error_immediate(self):
        """Dimension error raised immediately on add."""
        a = DaskDimArray([1.0], units.m)
        b = DaskDimArray([1.0], units.s)
        # Error should be raised when creating the operation, not during compute
        with pytest.raises(DimensionError):
            _ = a + b

    def test_subtract_error_immediate(self):
        """Dimension error raised immediately on subtract."""
        a = DaskDimArray([1.0], units.m)
        b = DaskDimArray([1.0], units.s)
        with pytest.raises(DimensionError):
            _ = a - b

    def test_compare_error_immediate(self):
        """Dimension error raised immediately on compare."""
        a = DaskDimArray([1.0], units.m)
        b = DaskDimArray([1.0], units.s)
        with pytest.raises(DimensionError):
            _ = a < b


class TestDaskDimArrayLargeArray:
    """Tests with larger arrays to verify chunking works."""

    def test_large_array_computation(self):
        """Test computation with larger chunked array."""
        arr = np.random.randn(1000, 1000)
        dda = DaskDimArray(arr, units.m, chunks=(100, 100))

        # Compute mean along axis
        result = dda.mean(axis=0).compute()

        assert result.shape == (1000,)
        assert result.unit == units.m
        # Verify correctness
        expected = arr.mean(axis=0)
        assert np.allclose(result.data, expected)

    def test_chained_computation(self):
        """Test chained operations on large array."""
        arr = np.random.randn(500, 500)
        velocity = DaskDimArray(arr, units.m / units.s, chunks=(100, 100))

        # Compute kinetic energy equivalent (v^2/2)
        energy = (velocity ** 2) / 2.0
        mean_energy = energy.mean().compute()

        assert mean_energy.dimension == ((units.m / units.s) ** 2).dimension

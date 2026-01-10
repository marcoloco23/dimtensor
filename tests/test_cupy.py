"""Tests for CuPy DimArray integration."""

import pytest
import numpy as np

try:
    import cupy as cp
    # Check that CUDA is actually available
    cp.cuda.runtime.getDeviceCount()
    CUPY_AVAILABLE = True
except (ImportError, RuntimeError, Exception):
    CUPY_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")

if CUPY_AVAILABLE:
    from dimtensor.cupy import DimArray

from dimtensor import units
from dimtensor.errors import DimensionError, UnitConversionError


class TestCuPyDimArrayCreation:
    """Tests for CuPy DimArray creation."""

    def test_from_cupy_array(self):
        """Create DimArray from CuPy array."""
        arr = cp.array([1.0, 2.0, 3.0])
        da = DimArray(arr, units.m)
        assert da.shape == (3,)
        assert da.unit == units.m

    def test_from_numpy_array(self):
        """Create DimArray from NumPy array (auto-transfer to GPU)."""
        arr = np.array([1.0, 2.0, 3.0])
        da = DimArray(arr, units.m)
        assert da.shape == (3,)
        assert da.unit == units.m
        assert isinstance(da.data, cp.ndarray)

    def test_from_list(self):
        """Create DimArray from list."""
        da = DimArray([1.0, 2.0, 3.0], units.m)
        assert da.shape == (3,)
        assert da.unit == units.m

    def test_from_scalar(self):
        """Create DimArray from scalar."""
        da = DimArray(5.0, units.kg)
        assert da.size == 1
        assert da.unit == units.kg

    def test_default_dimensionless(self):
        """Default unit is dimensionless."""
        da = DimArray(cp.array([1.0]))
        assert da.is_dimensionless

    def test_with_uncertainty(self):
        """Create DimArray with uncertainty."""
        da = DimArray(
            cp.array([1.0, 2.0, 3.0]),
            units.m,
            uncertainty=cp.array([0.1, 0.2, 0.3])
        )
        assert da.has_uncertainty
        assert cp.allclose(da.uncertainty, cp.array([0.1, 0.2, 0.3]))

    def test_uncertainty_shape_mismatch_raises(self):
        """Uncertainty shape must match data shape."""
        with pytest.raises(ValueError, match="shape"):
            DimArray(
                cp.array([1.0, 2.0, 3.0]),
                units.m,
                uncertainty=cp.array([0.1, 0.2])
            )

    def test_from_dimarray(self):
        """Create DimArray from another DimArray."""
        da1 = DimArray(cp.array([1.0, 2.0]), units.m)
        da2 = DimArray(da1)
        assert cp.allclose(da2.data, da1.data)
        assert da2.unit == da1.unit

    def test_copy_parameter(self):
        """Test copy parameter creates independent array."""
        arr = cp.array([1.0, 2.0, 3.0])
        da = DimArray(arr, units.m, copy=True)
        arr[0] = 999.0
        assert da.data[0] == 1.0  # Should not be affected


class TestCuPyDimArrayProperties:
    """Tests for DimArray properties."""

    def test_shape(self):
        """Test shape property."""
        da = DimArray(cp.zeros((2, 3, 4)), units.m)
        assert da.shape == (2, 3, 4)

    def test_ndim(self):
        """Test ndim property."""
        da = DimArray(cp.zeros((2, 3)), units.m)
        assert da.ndim == 2

    def test_size(self):
        """Test size property."""
        da = DimArray(cp.zeros((2, 3)), units.m)
        assert da.size == 6

    def test_dimension(self):
        """Test dimension property."""
        da = DimArray([1.0], units.m / units.s)
        assert da.dimension == (units.m / units.s).dimension

    def test_device_property(self):
        """Test device property returns GPU device ID."""
        da = DimArray(cp.array([1.0]), units.m)
        assert isinstance(da.device, int)
        assert da.device >= 0

    def test_relative_uncertainty(self):
        """Test relative uncertainty calculation."""
        da = DimArray(
            cp.array([10.0, 20.0]),
            units.m,
            uncertainty=cp.array([1.0, 2.0])
        )
        rel_unc = da.relative_uncertainty
        assert cp.allclose(rel_unc, cp.array([0.1, 0.1]))


class TestCuPyDimArrayArithmetic:
    """Tests for arithmetic operations."""

    def test_add_same_unit(self):
        """Add arrays with same unit."""
        a = DimArray([1.0, 2.0], units.m)
        b = DimArray([3.0, 4.0], units.m)
        c = a + b
        assert cp.allclose(c.data, cp.array([4.0, 6.0]))
        assert c.unit == units.m

    def test_add_compatible_units(self):
        """Add arrays with compatible units (auto-convert)."""
        a = DimArray([1000.0], units.m)
        b = DimArray([1.0], units.km)
        c = a + b
        assert cp.allclose(c.data, cp.array([2000.0]))
        assert c.unit == units.m

    def test_add_incompatible_raises(self):
        """Adding incompatible dimensions raises error."""
        a = DimArray([1.0], units.m)
        b = DimArray([1.0], units.s)
        with pytest.raises(DimensionError):
            a + b

    def test_subtract(self):
        """Subtract arrays."""
        a = DimArray([5.0, 6.0], units.m)
        b = DimArray([1.0, 2.0], units.m)
        c = a - b
        assert cp.allclose(c.data, cp.array([4.0, 4.0]))

    def test_multiply_dimensions(self):
        """Multiply arrays - dimensions multiply."""
        length = DimArray([2.0], units.m)
        width = DimArray([3.0], units.m)
        area = length * width
        assert area.dimension == (units.m**2).dimension
        assert cp.allclose(area.data, cp.array([6.0]))

    def test_multiply_scalar(self):
        """Multiply by scalar."""
        da = DimArray([1.0, 2.0], units.m)
        result = da * 3.0
        assert cp.allclose(result.data, cp.array([3.0, 6.0]))
        assert result.unit == units.m

    def test_divide_dimensions(self):
        """Divide arrays - dimensions divide."""
        distance = DimArray([10.0], units.m)
        time = DimArray([2.0], units.s)
        velocity = distance / time
        assert velocity.dimension == (units.m / units.s).dimension
        assert cp.allclose(velocity.data, cp.array([5.0]))

    def test_power(self):
        """Power operation scales dimension."""
        length = DimArray([2.0], units.m)
        area = length ** 2
        assert area.dimension == (units.m**2).dimension
        assert cp.allclose(area.data, cp.array([4.0]))

    def test_sqrt(self):
        """Square root halves dimension."""
        area = DimArray([4.0], units.m**2)
        length = area.sqrt()
        assert length.dimension == units.m.dimension
        assert cp.allclose(length.data, cp.array([2.0]))

    def test_negation(self):
        """Negation preserves unit."""
        da = DimArray([1.0, -2.0], units.m)
        neg = -da
        assert cp.allclose(neg.data, cp.array([-1.0, 2.0]))
        assert neg.unit == units.m

    def test_abs(self):
        """Absolute value preserves unit."""
        da = DimArray([-1.0, 2.0, -3.0], units.m)
        result = abs(da)
        assert cp.allclose(result.data, cp.array([1.0, 2.0, 3.0]))
        assert result.unit == units.m


class TestCuPyUncertaintyPropagation:
    """Tests for uncertainty propagation."""

    def test_add_uncertainty(self):
        """Addition propagates uncertainty correctly."""
        a = DimArray([10.0], units.m, uncertainty=cp.array([1.0]))
        b = DimArray([20.0], units.m, uncertainty=cp.array([2.0]))
        c = a + b
        # sigma_z = sqrt(1^2 + 2^2) = sqrt(5)
        assert cp.allclose(c.uncertainty, cp.array([cp.sqrt(5.0)]))

    def test_sub_uncertainty(self):
        """Subtraction propagates uncertainty correctly."""
        a = DimArray([30.0], units.m, uncertainty=cp.array([3.0]))
        b = DimArray([10.0], units.m, uncertainty=cp.array([4.0]))
        c = a - b
        # sigma_z = sqrt(3^2 + 4^2) = 5.0
        assert cp.allclose(c.uncertainty, cp.array([5.0]))

    def test_mul_uncertainty(self):
        """Multiplication propagates uncertainty correctly."""
        a = DimArray([10.0], units.m, uncertainty=cp.array([1.0]))
        b = DimArray([5.0], units.s, uncertainty=cp.array([0.5]))
        c = a * b
        # rel_unc = sqrt((1/10)^2 + (0.5/5)^2) = sqrt(0.01 + 0.01) = sqrt(0.02)
        # sigma_z = 50 * sqrt(0.02)
        expected = 50.0 * cp.sqrt(0.02)
        assert cp.allclose(c.uncertainty, cp.array([expected]), rtol=1e-5)

    def test_div_uncertainty(self):
        """Division propagates uncertainty correctly."""
        a = DimArray([10.0], units.m, uncertainty=cp.array([1.0]))
        b = DimArray([2.0], units.s, uncertainty=cp.array([0.2]))
        c = a / b
        # rel_unc = sqrt((1/10)^2 + (0.2/2)^2) = sqrt(0.01 + 0.01) = sqrt(0.02)
        # result = 5.0, sigma_z = 5.0 * sqrt(0.02)
        expected = 5.0 * cp.sqrt(0.02)
        assert cp.allclose(c.uncertainty, cp.array([expected]), rtol=1e-5)

    def test_power_uncertainty(self):
        """Power propagates uncertainty correctly."""
        a = DimArray([4.0], units.m, uncertainty=cp.array([0.4]))
        c = a ** 2
        # sigma_z = |z| * |n| * sigma_x/|x| = 16 * 2 * 0.4/4 = 3.2
        assert cp.allclose(c.uncertainty, cp.array([3.2]))

    def test_scalar_mul_uncertainty(self):
        """Scalar multiplication scales uncertainty."""
        a = DimArray([10.0], units.m, uncertainty=cp.array([1.0]))
        c = a * 3.0
        # sigma_z = |3| * 1 = 3
        assert cp.allclose(c.uncertainty, cp.array([3.0]))

    def test_no_uncertainty_returns_none(self):
        """Operations without uncertainty return None for uncertainty."""
        a = DimArray([1.0], units.m)
        b = DimArray([2.0], units.m)
        c = a + b
        assert c.uncertainty is None


class TestCuPyUnitConversion:
    """Tests for unit conversion."""

    def test_to_unit(self):
        """Convert to compatible unit."""
        da = DimArray([1000.0], units.m)
        da_km = da.to(units.km)
        assert cp.allclose(da_km.data, cp.array([1.0]))
        assert da_km.unit == units.km

    def test_to_unit_with_uncertainty(self):
        """Unit conversion scales uncertainty."""
        da = DimArray([1000.0], units.m, uncertainty=cp.array([10.0]))
        da_km = da.to(units.km)
        assert cp.allclose(da_km.uncertainty, cp.array([0.01]))

    def test_to_unit_incompatible_raises(self):
        """Converting to incompatible unit raises error."""
        da = DimArray([1.0], units.m)
        with pytest.raises(UnitConversionError):
            da.to(units.s)

    def test_to_base_units(self):
        """Convert to base SI units."""
        da = DimArray([1.0], units.km)
        da_base = da.to_base_units()
        assert cp.allclose(da_base.data, cp.array([1000.0]))

    def test_magnitude(self):
        """Get magnitude as copy."""
        da = DimArray([1.0, 2.0], units.m)
        mag = da.magnitude()
        assert cp.allclose(mag, cp.array([1.0, 2.0]))
        # Should be a copy
        mag[0] = 999.0
        assert da.data[0] == 1.0


class TestCuPyReductions:
    """Tests for reduction operations."""

    def test_sum(self):
        """Sum preserves unit."""
        da = DimArray([1.0, 2.0, 3.0], units.m)
        result = da.sum()
        assert cp.allclose(result.data, cp.array([6.0]))
        assert result.unit == units.m

    def test_sum_with_uncertainty(self):
        """Sum propagates uncertainty correctly."""
        da = DimArray([1.0, 2.0, 3.0], units.m, uncertainty=cp.array([0.1, 0.2, 0.3]))
        result = da.sum()
        # sigma = sqrt(0.1^2 + 0.2^2 + 0.3^2)
        expected = cp.sqrt(0.01 + 0.04 + 0.09)
        assert cp.allclose(result.uncertainty, cp.array([expected]))

    def test_mean(self):
        """Mean preserves unit."""
        da = DimArray([1.0, 2.0, 3.0], units.m)
        result = da.mean()
        assert cp.allclose(result.data, cp.array([2.0]))
        assert result.unit == units.m

    def test_std(self):
        """Std preserves unit, no uncertainty."""
        da = DimArray([1.0, 2.0, 3.0], units.m)
        result = da.std()
        assert result.unit == units.m
        assert result.uncertainty is None

    def test_var_squares_unit(self):
        """Variance squares unit."""
        da = DimArray([1.0, 2.0, 3.0], units.m)
        result = da.var()
        assert result.dimension == (units.m**2).dimension

    def test_min(self):
        """Min preserves unit."""
        da = DimArray([3.0, 1.0, 2.0], units.m)
        result = da.min()
        assert cp.allclose(result.data, cp.array([1.0]))
        assert result.unit == units.m

    def test_max(self):
        """Max preserves unit."""
        da = DimArray([3.0, 1.0, 2.0], units.m)
        result = da.max()
        assert cp.allclose(result.data, cp.array([3.0]))
        assert result.unit == units.m

    def test_argmin(self):
        """Argmin returns index."""
        da = DimArray([3.0, 1.0, 2.0], units.m)
        result = da.argmin()
        assert int(result) == 1

    def test_argmax(self):
        """Argmax returns index."""
        da = DimArray([3.0, 1.0, 2.0], units.m)
        result = da.argmax()
        assert int(result) == 0

    def test_sum_with_axis(self):
        """Sum along axis."""
        da = DimArray(cp.array([[1.0, 2.0], [3.0, 4.0]]), units.m)
        result = da.sum(axis=0)
        assert cp.allclose(result.data, cp.array([4.0, 6.0]))
        assert result.unit == units.m


class TestCuPyReshaping:
    """Tests for reshaping operations."""

    def test_reshape(self):
        """Reshape preserves unit."""
        da = DimArray(cp.zeros((2, 3)), units.m)
        reshaped = da.reshape((6,))
        assert reshaped.shape == (6,)
        assert reshaped.unit == units.m

    def test_reshape_with_uncertainty(self):
        """Reshape preserves uncertainty."""
        da = DimArray(
            cp.array([[1.0, 2.0], [3.0, 4.0]]),
            units.m,
            uncertainty=cp.array([[0.1, 0.2], [0.3, 0.4]])
        )
        reshaped = da.reshape((4,))
        assert reshaped.shape == (4,)
        assert cp.allclose(reshaped.uncertainty, cp.array([0.1, 0.2, 0.3, 0.4]))

    def test_transpose(self):
        """Transpose preserves unit."""
        da = DimArray(cp.zeros((2, 3)), units.m)
        transposed = da.transpose()
        assert transposed.shape == (3, 2)
        assert transposed.unit == units.m

    def test_flatten(self):
        """Flatten preserves unit."""
        da = DimArray(cp.zeros((2, 3)), units.m)
        flat = da.flatten()
        assert flat.shape == (6,)
        assert flat.unit == units.m

    def test_squeeze(self):
        """Squeeze removes size-1 dimensions."""
        da = DimArray(cp.zeros((1, 3, 1)), units.m)
        squeezed = da.squeeze()
        assert squeezed.shape == (3,)
        assert squeezed.unit == units.m

    def test_expand_dims(self):
        """Expand dims adds dimension."""
        da = DimArray(cp.zeros((3,)), units.m)
        expanded = da.expand_dims(0)
        assert expanded.shape == (1, 3)
        assert expanded.unit == units.m


class TestCuPyLinearAlgebra:
    """Tests for linear algebra operations."""

    def test_dot(self):
        """Dot product multiplies dimensions."""
        a = DimArray([1.0, 2.0, 3.0], units.m)
        b = DimArray([4.0, 5.0, 6.0], units.s)
        c = a.dot(b)
        assert c.dimension == (units.m * units.s).dimension
        assert cp.allclose(c.data, cp.array([32.0]))

    def test_matmul(self):
        """Matrix multiplication multiplies dimensions."""
        a = DimArray(cp.ones((2, 3)), units.m)
        b = DimArray(cp.ones((3, 4)), units.s)
        c = a.matmul(b)
        assert c.shape == (2, 4)
        assert c.dimension == (units.m * units.s).dimension

    def test_matmul_operator(self):
        """@ operator works for matmul."""
        a = DimArray(cp.ones((2, 3)), units.m)
        b = DimArray(cp.ones((3, 4)), units.s)
        c = a @ b
        assert c.dimension == (units.m * units.s).dimension

    def test_norm(self):
        """Norm preserves unit."""
        da = DimArray([3.0, 4.0], units.m)
        result = da.norm()
        assert cp.allclose(result.data, cp.array([5.0]))
        assert result.unit == units.m


class TestCuPyComparison:
    """Tests for comparison operations."""

    def test_eq_same_unit(self):
        """Equality with same unit."""
        a = DimArray([1.0, 2.0], units.m)
        b = DimArray([1.0, 3.0], units.m)
        result = a == b
        assert bool(result[0]) is True
        assert bool(result[1]) is False

    def test_ne(self):
        """Inequality comparison."""
        a = DimArray([1.0, 2.0], units.m)
        b = DimArray([1.0, 3.0], units.m)
        result = a != b
        assert bool(result[0]) is False
        assert bool(result[1]) is True

    def test_lt(self):
        """Less than comparison."""
        a = DimArray([1.0, 3.0], units.m)
        b = DimArray([2.0, 2.0], units.m)
        result = a < b
        assert bool(result[0]) is True
        assert bool(result[1]) is False

    def test_le(self):
        """Less than or equal comparison."""
        a = DimArray([1.0, 2.0], units.m)
        b = DimArray([2.0, 2.0], units.m)
        result = a <= b
        assert bool(result[0]) is True
        assert bool(result[1]) is True

    def test_gt(self):
        """Greater than comparison."""
        a = DimArray([3.0, 1.0], units.m)
        b = DimArray([2.0, 2.0], units.m)
        result = a > b
        assert bool(result[0]) is True
        assert bool(result[1]) is False

    def test_ge(self):
        """Greater than or equal comparison."""
        a = DimArray([2.0, 3.0], units.m)
        b = DimArray([2.0, 2.0], units.m)
        result = a >= b
        assert bool(result[0]) is True
        assert bool(result[1]) is True

    def test_compare_incompatible_raises(self):
        """Comparing incompatible dimensions raises error."""
        a = DimArray([1.0], units.m)
        b = DimArray([1.0], units.s)
        with pytest.raises(DimensionError):
            a < b


class TestCuPyIndexing:
    """Tests for indexing operations."""

    def test_getitem(self):
        """Indexing preserves unit."""
        da = DimArray([1.0, 2.0, 3.0], units.m)
        item = da[1]
        assert cp.allclose(item.data, cp.array([2.0]))
        assert item.unit == units.m

    def test_slice(self):
        """Slicing preserves unit."""
        da = DimArray([1.0, 2.0, 3.0, 4.0], units.m)
        sliced = da[1:3]
        assert sliced.shape == (2,)
        assert sliced.unit == units.m

    def test_getitem_with_uncertainty(self):
        """Indexing preserves uncertainty."""
        da = DimArray([1.0, 2.0, 3.0], units.m, uncertainty=cp.array([0.1, 0.2, 0.3]))
        item = da[1]
        assert cp.allclose(item.uncertainty, cp.array([0.2]))

    def test_len(self):
        """Test __len__."""
        da = DimArray([1.0, 2.0, 3.0], units.m)
        assert len(da) == 3

    def test_iter(self):
        """Test iteration."""
        da = DimArray([1.0, 2.0, 3.0], units.m)
        values = [float(item.data[0]) for item in da]
        assert values == [1.0, 2.0, 3.0]


class TestCuPyStrings:
    """Tests for string representations."""

    def test_repr(self):
        """repr includes data and unit."""
        da = DimArray([1.0], units.m)
        r = repr(da)
        assert "DimArray" in r
        assert "m" in r

    def test_str_with_unit(self):
        """str shows data and unit."""
        da = DimArray([1.0], units.m)
        s = str(da)
        assert "m" in s

    def test_repr_with_uncertainty(self):
        """repr includes uncertainty when present."""
        da = DimArray([1.0], units.m, uncertainty=cp.array([0.1]))
        r = repr(da)
        assert "uncertainty" in r

    def test_format(self):
        """Test __format__ method."""
        da = DimArray([1.2345], units.m)
        s = f"{da:.2f}"
        assert "1.23" in s
        assert "m" in s


class TestCuPyInterop:
    """Tests for NumPy/CuPy interoperability."""

    def test_numpy_conversion(self):
        """Convert to numpy array."""
        da = DimArray([1.0, 2.0], units.m)
        arr = da.numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2,)
        assert arr[0] == pytest.approx(1.0)

    def test_cupy_method(self):
        """Get underlying CuPy array."""
        da = DimArray([1.0, 2.0], units.m)
        arr = da.cupy()
        assert isinstance(arr, cp.ndarray)

    def test_get_method(self):
        """Test get() method (CuPy convention)."""
        da = DimArray([1.0, 2.0], units.m)
        arr = da.get()
        assert isinstance(arr, np.ndarray)

    def test_set_method(self):
        """Test set() method."""
        da = DimArray([1.0, 2.0], units.m)
        da.set(np.array([3.0, 4.0]))
        assert cp.allclose(da.data, cp.array([3.0, 4.0]))

    def test_set_shape_mismatch_raises(self):
        """set() raises on shape mismatch."""
        da = DimArray([1.0, 2.0], units.m)
        with pytest.raises(ValueError, match="Shape"):
            da.set(np.array([1.0, 2.0, 3.0]))

    def test_array_protocol(self):
        """Test __array__ protocol."""
        da = DimArray([1.0, 2.0], units.m)
        arr = np.asarray(da)
        assert isinstance(arr, np.ndarray)
        assert arr[0] == pytest.approx(1.0)

    def test_cpu_method(self):
        """Test cpu() returns NumPy DimArray."""
        da = DimArray([1.0, 2.0], units.m)
        da_cpu = da.cpu()
        from dimtensor.core.dimarray import DimArray as NumpyDimArray
        assert isinstance(da_cpu, NumpyDimArray)
        assert da_cpu.unit == units.m

    def test_gpu_method(self):
        """Test gpu() returns self."""
        da = DimArray([1.0, 2.0], units.m)
        da_gpu = da.gpu()
        assert da_gpu is da


class TestCuPyDeviceManagement:
    """Tests for GPU device management."""

    def test_device_property(self):
        """Test device property."""
        da = DimArray([1.0], units.m)
        assert isinstance(da.device, int)

    def test_to_device(self):
        """Test transferring to different device (skipped if < 2 GPUs)."""
        if cp.cuda.runtime.getDeviceCount() < 2:
            pytest.skip("Multi-GPU test requires 2+ GPUs")
        da = DimArray([1.0, 2.0], units.m)
        da_dev1 = da.to_device(1)
        assert da_dev1.device == 1


class TestCuPyEdgeCases:
    """Tests for edge cases."""

    def test_empty_array(self):
        """Handle empty arrays."""
        da = DimArray(cp.array([]), units.m)
        assert da.size == 0
        assert da.shape == (0,)

    def test_scalar_array(self):
        """Handle scalar (0-d) arrays."""
        da = DimArray(5.0, units.m)
        result = da * 2.0
        assert float(result.data[0]) == 10.0

    def test_broadcasting(self):
        """Test broadcasting works correctly."""
        a = DimArray(cp.array([[1.0], [2.0], [3.0]]), units.m)  # (3, 1)
        b = DimArray(cp.array([1.0, 2.0, 3.0]), units.m)  # (3,)
        c = a + b
        assert c.shape == (3, 3)

    def test_dimensionless_operations(self):
        """Test operations with dimensionless quantities."""
        a = DimArray([1.0, 2.0], units.dimensionless)
        b = 3.0
        c = a + b
        assert cp.allclose(c.data, cp.array([4.0, 5.0]))


class TestCuPyPhysicsExamples:
    """Physics example tests."""

    def test_kinetic_energy(self):
        """Calculate kinetic energy: E = 0.5 * m * v^2."""
        m = DimArray([1.0, 2.0], units.kg)
        v = DimArray([3.0, 4.0], units.m / units.s)
        E = 0.5 * m * v**2
        assert cp.allclose(E.data, cp.array([4.5, 16.0]))
        assert E.dimension == units.J.dimension

    def test_force_calculation(self):
        """Calculate force: F = m * a."""
        m = DimArray([1.0], units.kg, uncertainty=cp.array([0.1]))
        a = DimArray([9.8], units.m / units.s**2, uncertainty=cp.array([0.1]))
        F = m * a
        assert F.dimension == units.N.dimension
        assert F.has_uncertainty

    def test_distance_from_velocity(self):
        """Calculate distance: d = v * t."""
        v = DimArray([10.0], units.m / units.s)
        t = DimArray([5.0], units.s)
        d = v * t
        assert cp.allclose(d.data, cp.array([50.0]))
        assert d.dimension == units.m.dimension

"""Tests for uncertainty propagation in DimArray.

This module tests the v0.5.0 uncertainty propagation feature, including:
- Creation of DimArrays with uncertainty
- Propagation through arithmetic operations
- Unit conversion with uncertainty
- Reduction operations
- Constant integration
- String representations
"""

import numpy as np
import pytest

from dimtensor import DimArray, units
from dimtensor.errors import DimensionError
from dimtensor.constants import c, G, h


class TestUncertaintyCreation:
    """Tests for creating DimArrays with uncertainty."""

    def test_create_with_uncertainty(self):
        """Test creating a DimArray with uncertainty."""
        arr = DimArray([10.0], units.m, uncertainty=[0.1])
        assert arr.has_uncertainty
        assert arr.uncertainty is not None
        np.testing.assert_array_almost_equal(arr.uncertainty, [0.1])

    def test_create_without_uncertainty(self):
        """Test creating a DimArray without uncertainty."""
        arr = DimArray([10.0], units.m)
        assert not arr.has_uncertainty
        assert arr.uncertainty is None

    def test_uncertainty_shape_must_match(self):
        """Test that uncertainty shape must match data shape."""
        with pytest.raises(ValueError, match="must match data shape"):
            DimArray([1.0, 2.0, 3.0], units.m, uncertainty=[0.1, 0.2])

    def test_multidimensional_uncertainty(self):
        """Test uncertainty with multidimensional arrays."""
        arr = DimArray(
            [[1.0, 2.0], [3.0, 4.0]],
            units.m,
            uncertainty=[[0.1, 0.2], [0.3, 0.4]],
        )
        assert arr.uncertainty.shape == (2, 2)

    def test_relative_uncertainty(self):
        """Test relative uncertainty calculation."""
        arr = DimArray([10.0, 20.0], units.m, uncertainty=[1.0, 2.0])
        np.testing.assert_array_almost_equal(arr.relative_uncertainty, [0.1, 0.1])

    def test_relative_uncertainty_with_zero_value(self):
        """Test relative uncertainty when value is zero."""
        arr = DimArray([0.0, 10.0], units.m, uncertainty=[0.1, 1.0])
        rel_unc = arr.relative_uncertainty
        assert np.isinf(rel_unc[0])
        np.testing.assert_almost_equal(rel_unc[1], 0.1)


class TestUncertaintyPropagation:
    """Tests for uncertainty propagation through operations."""

    def test_addition_propagation(self):
        """Test uncertainty propagation through addition."""
        a = DimArray([10.0], units.m, uncertainty=[0.1])
        b = DimArray([5.0], units.m, uncertainty=[0.2])
        result = a + b

        # sigma_z = sqrt(0.1^2 + 0.2^2) = sqrt(0.01 + 0.04) = sqrt(0.05)
        expected_unc = np.sqrt(0.1**2 + 0.2**2)
        np.testing.assert_array_almost_equal(result.data, [15.0])
        np.testing.assert_array_almost_equal(result.uncertainty, [expected_unc])

    def test_subtraction_propagation(self):
        """Test uncertainty propagation through subtraction."""
        a = DimArray([10.0], units.m, uncertainty=[0.1])
        b = DimArray([5.0], units.m, uncertainty=[0.2])
        result = a - b

        expected_unc = np.sqrt(0.1**2 + 0.2**2)
        np.testing.assert_array_almost_equal(result.data, [5.0])
        np.testing.assert_array_almost_equal(result.uncertainty, [expected_unc])

    def test_multiplication_propagation(self):
        """Test uncertainty propagation through multiplication."""
        length = DimArray([10.0], units.m, uncertainty=[0.1])
        time = DimArray([2.0], units.s, uncertainty=[0.05])
        velocity = length / time

        # For division: sigma_z/|z| = sqrt((sigma_x/x)^2 + (sigma_y/y)^2)
        # sigma_z/5.0 = sqrt((0.1/10)^2 + (0.05/2)^2) = sqrt(0.01^2 + 0.025^2)
        # sigma_z = 5.0 * sqrt(0.0001 + 0.000625) = 5.0 * 0.0269 = 0.1346
        expected_rel_unc = np.sqrt((0.1 / 10.0) ** 2 + (0.05 / 2.0) ** 2)
        expected_unc = 5.0 * expected_rel_unc

        np.testing.assert_array_almost_equal(velocity.data, [5.0])
        np.testing.assert_array_almost_equal(velocity.uncertainty, [expected_unc], decimal=4)

    def test_power_propagation(self):
        """Test uncertainty propagation through power operation."""
        arr = DimArray([4.0], units.m, uncertainty=[0.1])
        result = arr**2

        # sigma_z/|z| = |n| * sigma_x/|x|
        # sigma_z = |16| * 2 * (0.1 / 4) = 16 * 0.05 = 0.8
        expected_unc = 16.0 * 2 * (0.1 / 4.0)
        np.testing.assert_array_almost_equal(result.data, [16.0])
        np.testing.assert_array_almost_equal(result.uncertainty, [expected_unc])

    def test_sqrt_propagation(self):
        """Test uncertainty propagation through sqrt."""
        arr = DimArray([4.0], units.m**2, uncertainty=[0.2])
        result = arr.sqrt()

        # sqrt uses power(0.5): sigma_z/|z| = 0.5 * sigma_x/|x|
        # sigma_z = 2 * 0.5 * (0.2 / 4) = 0.05
        expected_unc = 2.0 * 0.5 * (0.2 / 4.0)
        np.testing.assert_array_almost_equal(result.data, [2.0])
        np.testing.assert_array_almost_equal(result.uncertainty, [expected_unc])

    def test_scalar_multiplication_preserves_uncertainty(self):
        """Test that scalar multiplication scales uncertainty."""
        arr = DimArray([10.0], units.m, uncertainty=[0.1])
        result = 2 * arr

        np.testing.assert_array_almost_equal(result.data, [20.0])
        np.testing.assert_array_almost_equal(result.uncertainty, [0.2])

    def test_scalar_division_scales_uncertainty(self):
        """Test that scalar division scales uncertainty."""
        arr = DimArray([10.0], units.m, uncertainty=[0.2])
        result = arr / 2

        np.testing.assert_array_almost_equal(result.data, [5.0])
        np.testing.assert_array_almost_equal(result.uncertainty, [0.1])

    def test_mixing_with_and_without_uncertainty(self):
        """Test combining arrays with and without uncertainty."""
        with_unc = DimArray([10.0], units.m, uncertainty=[0.1])
        without_unc = DimArray([5.0], units.m)
        result = with_unc + without_unc

        # Without uncertainty is treated as zero uncertainty
        np.testing.assert_array_almost_equal(result.data, [15.0])
        np.testing.assert_array_almost_equal(result.uncertainty, [0.1])

    def test_negation_preserves_uncertainty(self):
        """Test that negation preserves uncertainty."""
        arr = DimArray([10.0], units.m, uncertainty=[0.1])
        result = -arr

        np.testing.assert_array_almost_equal(result.data, [-10.0])
        np.testing.assert_array_almost_equal(result.uncertainty, [0.1])

    def test_abs_preserves_uncertainty(self):
        """Test that abs preserves uncertainty."""
        arr = DimArray([-10.0], units.m, uncertainty=[0.1])
        result = abs(arr)

        np.testing.assert_array_almost_equal(result.data, [10.0])
        np.testing.assert_array_almost_equal(result.uncertainty, [0.1])


class TestUncertaintyUnitConversion:
    """Tests for uncertainty through unit conversion."""

    def test_to_scales_uncertainty(self):
        """Test that unit conversion scales uncertainty."""
        arr = DimArray([1000.0], units.m, uncertainty=[10.0])
        result = arr.to(units.km)

        np.testing.assert_array_almost_equal(result.data, [1.0])
        np.testing.assert_array_almost_equal(result.uncertainty, [0.01])

    def test_to_base_units_scales_uncertainty(self):
        """Test that to_base_units scales uncertainty."""
        arr = DimArray([1.0], units.km, uncertainty=[0.01])
        result = arr.to_base_units()

        np.testing.assert_array_almost_equal(result.data, [1000.0])
        np.testing.assert_array_almost_equal(result.uncertainty, [10.0])


class TestUncertaintyReductions:
    """Tests for uncertainty in reduction operations."""

    def test_sum_propagation(self):
        """Test uncertainty propagation through sum."""
        arr = DimArray([1.0, 2.0, 3.0], units.m, uncertainty=[0.1, 0.2, 0.3])
        result = arr.sum()

        # sigma_sum = sqrt(0.1^2 + 0.2^2 + 0.3^2)
        expected_unc = np.sqrt(0.1**2 + 0.2**2 + 0.3**2)
        np.testing.assert_array_almost_equal(result.data, [6.0])
        np.testing.assert_array_almost_equal(result.uncertainty, [expected_unc])

    def test_mean_propagation(self):
        """Test uncertainty propagation through mean."""
        arr = DimArray([1.0, 2.0, 3.0], units.m, uncertainty=[0.1, 0.2, 0.3])
        result = arr.mean()

        # sigma_mean = sqrt(sum(sigma_i^2)) / N
        expected_unc = np.sqrt(0.1**2 + 0.2**2 + 0.3**2) / 3
        np.testing.assert_array_almost_equal(result.data, [2.0])
        np.testing.assert_array_almost_equal(result.uncertainty, [expected_unc])

    def test_min_takes_element_uncertainty(self):
        """Test that min takes the uncertainty of the minimum element."""
        arr = DimArray([3.0, 1.0, 2.0], units.m, uncertainty=[0.3, 0.1, 0.2])
        result = arr.min()

        np.testing.assert_array_almost_equal(result.data, [1.0])
        np.testing.assert_array_almost_equal(result.uncertainty, [0.1])

    def test_max_takes_element_uncertainty(self):
        """Test that max takes the uncertainty of the maximum element."""
        arr = DimArray([2.0, 3.0, 1.0], units.m, uncertainty=[0.2, 0.3, 0.1])
        result = arr.max()

        np.testing.assert_array_almost_equal(result.data, [3.0])
        np.testing.assert_array_almost_equal(result.uncertainty, [0.3])

    def test_std_drops_uncertainty(self):
        """Test that std drops uncertainty (too complex to propagate)."""
        arr = DimArray([1.0, 2.0, 3.0], units.m, uncertainty=[0.1, 0.2, 0.3])
        result = arr.std()

        assert result.uncertainty is None

    def test_var_drops_uncertainty(self):
        """Test that var drops uncertainty (too complex to propagate)."""
        arr = DimArray([1.0, 2.0, 3.0], units.m, uncertainty=[0.1, 0.2, 0.3])
        result = arr.var()

        assert result.uncertainty is None


class TestUncertaintyIndexing:
    """Tests for uncertainty through indexing operations."""

    def test_getitem_preserves_uncertainty(self):
        """Test that indexing preserves uncertainty."""
        arr = DimArray([1.0, 2.0, 3.0], units.m, uncertainty=[0.1, 0.2, 0.3])
        result = arr[1]

        np.testing.assert_array_almost_equal(result.data, [2.0])
        np.testing.assert_array_almost_equal(result.uncertainty, [0.2])

    def test_slice_preserves_uncertainty(self):
        """Test that slicing preserves uncertainty."""
        arr = DimArray([1.0, 2.0, 3.0], units.m, uncertainty=[0.1, 0.2, 0.3])
        result = arr[1:]

        np.testing.assert_array_almost_equal(result.data, [2.0, 3.0])
        np.testing.assert_array_almost_equal(result.uncertainty, [0.2, 0.3])

    def test_reshape_preserves_uncertainty(self):
        """Test that reshape preserves uncertainty."""
        arr = DimArray([1.0, 2.0, 3.0, 4.0], units.m, uncertainty=[0.1, 0.2, 0.3, 0.4])
        result = arr.reshape((2, 2))

        assert result.shape == (2, 2)
        assert result.uncertainty.shape == (2, 2)

    def test_flatten_preserves_uncertainty(self):
        """Test that flatten preserves uncertainty."""
        arr = DimArray(
            [[1.0, 2.0], [3.0, 4.0]],
            units.m,
            uncertainty=[[0.1, 0.2], [0.3, 0.4]],
        )
        result = arr.flatten()

        np.testing.assert_array_almost_equal(result.uncertainty, [0.1, 0.2, 0.3, 0.4])


class TestConstantUncertainty:
    """Tests for uncertainty from physical constants."""

    def test_constant_to_dimarray_with_uncertainty(self):
        """Test that constants with uncertainty transfer to DimArray."""
        # G has uncertainty
        arr = G.to_dimarray()
        if G.uncertainty > 0:
            assert arr.has_uncertainty
            np.testing.assert_array_almost_equal(arr.uncertainty, [G.uncertainty])

    def test_exact_constant_no_uncertainty(self):
        """Test that exact constants have no uncertainty."""
        # c is exact
        arr = c.to_dimarray()
        assert c.is_exact
        assert arr.uncertainty is None

    def test_constant_multiplication_propagates_uncertainty(self):
        """Test uncertainty propagation in Constant * Constant."""
        if G.uncertainty > 0:
            result = G * G
            assert result.has_uncertainty


class TestUncertaintyStringRepresentation:
    """Tests for string representation with uncertainty."""

    def test_str_with_uncertainty(self):
        """Test __str__ includes uncertainty."""
        arr = DimArray([10.0], units.m, uncertainty=[0.1])
        s = str(arr)
        assert "±" in s or "+/-" in s

    def test_str_without_uncertainty(self):
        """Test __str__ without uncertainty."""
        arr = DimArray([10.0], units.m)
        s = str(arr)
        assert "±" not in s

    def test_repr_with_uncertainty(self):
        """Test __repr__ includes uncertainty."""
        arr = DimArray([10.0], units.m, uncertainty=[0.1])
        r = repr(arr)
        assert "uncertainty" in r

    def test_format_with_uncertainty(self):
        """Test __format__ includes uncertainty."""
        arr = DimArray([10.0], units.m, uncertainty=[0.1])
        s = f"{arr:.2f}"
        assert "±" in s or "+/-" in s


class TestUncertaintyFunctions:
    """Tests for module-level functions with uncertainty."""

    def test_concatenate_with_uncertainty(self):
        """Test concatenate preserves uncertainty."""
        from dimtensor import concatenate

        a = DimArray([1.0, 2.0], units.m, uncertainty=[0.1, 0.2])
        b = DimArray([3.0, 4.0], units.m, uncertainty=[0.3, 0.4])
        result = concatenate([a, b])

        np.testing.assert_array_almost_equal(result.uncertainty, [0.1, 0.2, 0.3, 0.4])

    def test_concatenate_mixed_uncertainty(self):
        """Test concatenate with mixed uncertainty (some with, some without)."""
        from dimtensor import concatenate

        a = DimArray([1.0, 2.0], units.m, uncertainty=[0.1, 0.2])
        b = DimArray([3.0, 4.0], units.m)  # No uncertainty
        result = concatenate([a, b])

        # b's uncertainty is treated as zero
        np.testing.assert_array_almost_equal(result.uncertainty, [0.1, 0.2, 0.0, 0.0])

    def test_stack_with_uncertainty(self):
        """Test stack preserves uncertainty."""
        from dimtensor import stack

        a = DimArray([1.0, 2.0], units.m, uncertainty=[0.1, 0.2])
        b = DimArray([3.0, 4.0], units.m, uncertainty=[0.3, 0.4])
        result = stack([a, b])

        assert result.uncertainty.shape == (2, 2)

    def test_split_with_uncertainty(self):
        """Test split preserves uncertainty."""
        from dimtensor import split

        arr = DimArray([1.0, 2.0, 3.0, 4.0], units.m, uncertainty=[0.1, 0.2, 0.3, 0.4])
        parts = split(arr, 2)

        np.testing.assert_array_almost_equal(parts[0].uncertainty, [0.1, 0.2])
        np.testing.assert_array_almost_equal(parts[1].uncertainty, [0.3, 0.4])

    def test_dot_with_uncertainty(self):
        """Test dot product propagates uncertainty."""
        from dimtensor import dot

        a = DimArray([3.0, 4.0], units.m, uncertainty=[0.1, 0.2])
        b = DimArray([1.0, 2.0], units.s, uncertainty=[0.05, 0.1])
        result = dot(a, b)

        # σ² = Σ(b² σ_a² + a² σ_b²)
        #    = 1²×0.1² + 2²×0.2² + 3²×0.05² + 4²×0.1²
        #    = 0.01 + 0.16 + 0.0225 + 0.16 = 0.3525
        expected_unc = np.sqrt(0.3525)
        assert result.has_uncertainty
        np.testing.assert_almost_equal(result.uncertainty[0], expected_unc, decimal=10)

    def test_dot_one_has_uncertainty(self):
        """Test dot when only one operand has uncertainty."""
        from dimtensor import dot

        a = DimArray([3.0, 4.0], units.m, uncertainty=[0.1, 0.2])
        b = DimArray([1.0, 2.0], units.s)
        result = dot(a, b)

        # σ² = Σ(b² σ_a²) = 1²×0.1² + 2²×0.2² = 0.01 + 0.16 = 0.17
        expected_unc = np.sqrt(0.17)
        assert result.has_uncertainty
        np.testing.assert_almost_equal(result.uncertainty[0], expected_unc, decimal=10)

    def test_dot_no_uncertainty(self):
        """Test dot without uncertainty returns no uncertainty."""
        from dimtensor import dot

        a = DimArray([3.0, 4.0], units.m)
        b = DimArray([1.0, 2.0], units.s)
        result = dot(a, b)
        assert not result.has_uncertainty

    def test_matmul_with_uncertainty(self):
        """Test matmul propagates uncertainty for 2D arrays."""
        from dimtensor import matmul

        A = DimArray([[1.0, 2.0], [3.0, 4.0]], units.m, uncertainty=[[0.1, 0.2], [0.3, 0.4]])
        B = DimArray([[5.0, 6.0], [7.0, 8.0]], units.s, uncertainty=[[0.5, 0.6], [0.7, 0.8]])
        C = matmul(A, B)

        assert C.has_uncertainty
        assert C.uncertainty.shape == (2, 2)

        # Verify C[0,0]: A[0,:] @ B[:,0] = 1*5 + 2*7 = 19
        # σ²(C[0,0]) = B[0,0]²σ²(A[0,0]) + B[1,0]²σ²(A[0,1])
        #            + A[0,0]²σ²(B[0,0]) + A[0,1]²σ²(B[1,0])
        #            = 25*0.01 + 49*0.04 + 1*0.25 + 4*0.49
        #            = 0.25 + 1.96 + 0.25 + 1.96 = 4.42
        expected_unc_00 = np.sqrt(4.42)
        np.testing.assert_almost_equal(C.uncertainty[0, 0], expected_unc_00, decimal=10)

    def test_matmul_no_uncertainty(self):
        """Test matmul without uncertainty returns no uncertainty."""
        from dimtensor import matmul

        A = DimArray([[1.0, 2.0]], units.m)
        B = DimArray([[3.0], [4.0]], units.s)
        C = matmul(A, B)
        assert not C.has_uncertainty

    def test_weighted_mean_basic(self):
        """Test inverse-variance weighted mean."""
        from dimtensor import weighted_mean

        a = DimArray([10.0], units.m, uncertainty=[1.0])
        b = DimArray([12.0], units.m, uncertainty=[2.0])
        result = weighted_mean([a, b])

        # weights: 1/1² = 1, 1/2² = 0.25; total = 1.25
        # mean = (1*10 + 0.25*12) / 1.25 = 13 / 1.25 = 10.4
        # σ = 1/√1.25 ≈ 0.8944
        np.testing.assert_almost_equal(result._data[0], 10.4)
        np.testing.assert_almost_equal(result.uncertainty[0], 1.0 / np.sqrt(1.25))

    def test_weighted_mean_equal_uncertainty(self):
        """Test weighted mean with equal uncertainties gives arithmetic mean."""
        from dimtensor import weighted_mean

        a = DimArray([10.0], units.m, uncertainty=[1.0])
        b = DimArray([20.0], units.m, uncertainty=[1.0])
        result = weighted_mean([a, b])

        np.testing.assert_almost_equal(result._data[0], 15.0)

    def test_weighted_mean_zero_variance(self):
        """Test weighted mean with zero variance returns exact value."""
        from dimtensor import weighted_mean

        a = DimArray([10.0], units.m, uncertainty=[0.0])  # exact
        b = DimArray([20.0], units.m, uncertainty=[5.0])
        result = weighted_mean([a, b])

        np.testing.assert_almost_equal(result._data[0], 10.0)
        np.testing.assert_almost_equal(result.uncertainty[0], 0.0)

    def test_weighted_mean_requires_uncertainty(self):
        """Test weighted mean raises if any input lacks uncertainty."""
        from dimtensor import weighted_mean

        a = DimArray([10.0], units.m, uncertainty=[1.0])
        b = DimArray([20.0], units.m)  # no uncertainty
        with pytest.raises(ValueError, match="must have uncertainty"):
            weighted_mean([a, b])

    def test_weighted_mean_dimension_check(self):
        """Test weighted mean rejects incompatible dimensions."""
        from dimtensor import weighted_mean

        a = DimArray([10.0], units.m, uncertainty=[1.0])
        b = DimArray([20.0], units.s, uncertainty=[1.0])
        with pytest.raises(DimensionError):
            weighted_mean([a, b])

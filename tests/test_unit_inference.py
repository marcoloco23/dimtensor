"""Tests for automatic unit inference from equations."""

import pytest

from dimtensor.core.units import (
    kg, m, s, N, J, Pa, K, mol, A, V, ohm, C,
    meter, kilogram, second, newton, joule, pascal
)
from dimtensor.core.dimensions import Dimension
from dimtensor.inference import infer_units


class TestBasicInference:
    """Test basic inference cases."""

    def test_newtons_second_law(self):
        """Test F = m * a."""
        result = infer_units(
            "F = m * a",
            {"m": kg, "a": m / s**2}
        )

        assert result['is_consistent']
        assert 'F' in result['inferred']
        assert result['inferred']['F'].dimension == Dimension(mass=1, length=1, time=-2)
        assert result['confidence'] == 1.0
        assert len(result['errors']) == 0

    def test_kinetic_energy(self):
        """Test KE = 0.5 * m * v**2."""
        result = infer_units(
            "KE = 0.5 * m * v**2",
            {"m": kg, "v": m / s}
        )

        assert result['is_consistent']
        assert 'KE' in result['inferred']
        assert result['inferred']['KE'].dimension == Dimension(mass=1, length=2, time=-2)
        assert len(result['errors']) == 0

    def test_velocity_from_distance_time(self):
        """Test v = d / t."""
        result = infer_units(
            "v = d / t",
            {"d": m, "t": s}
        )

        assert result['is_consistent']
        assert 'v' in result['inferred']
        assert result['inferred']['v'].dimension == Dimension(length=1, time=-1)

    def test_work(self):
        """Test W = F * d."""
        result = infer_units(
            "W = F * d",
            {"F": N, "d": m}
        )

        assert result['is_consistent']
        assert 'W' in result['inferred']
        assert result['inferred']['W'].dimension == Dimension(mass=1, length=2, time=-2)

    def test_power(self):
        """Test P = W / t."""
        result = infer_units(
            "P = W / t",
            {"W": J, "t": s}
        )

        assert result['is_consistent']
        assert 'P' in result['inferred']
        assert result['inferred']['P'].dimension == Dimension(mass=1, length=2, time=-3)

    def test_pressure(self):
        """Test P = F / A."""
        result = infer_units(
            "P = F / A",
            {"F": N, "A": m**2}
        )

        assert result['is_consistent']
        assert 'P' in result['inferred']
        assert result['inferred']['P'].dimension == Dimension(mass=1, length=-1, time=-2)


class TestReverseInference:
    """Test inferring inputs from output."""

    def test_infer_mass(self):
        """Test F = m * a, infer m from F and a."""
        result = infer_units(
            "F = m * a",
            {"F": N, "a": m / s**2}
        )

        assert result['is_consistent']
        assert 'm' in result['inferred']
        assert result['inferred']['m'].dimension == Dimension(mass=1)

    def test_infer_acceleration(self):
        """Test F = m * a, infer a from F and m."""
        result = infer_units(
            "F = m * a",
            {"F": N, "m": kg}
        )

        assert result['is_consistent']
        assert 'a' in result['inferred']
        assert result['inferred']['a'].dimension == Dimension(length=1, time=-2)

    def test_infer_distance(self):
        """Test v = d / t, infer d from v and t."""
        result = infer_units(
            "v = d / t",
            {"v": m / s, "t": s}
        )

        assert result['is_consistent']
        assert 'd' in result['inferred']
        assert result['inferred']['d'].dimension == Dimension(length=1)


class TestPowerOperations:
    """Test power operations."""

    def test_einstein_mass_energy(self):
        """Test E = m * c**2."""
        result = infer_units(
            "E = m * c**2",
            {"m": kg, "c": m / s}
        )

        assert result['is_consistent']
        assert 'E' in result['inferred']
        assert result['inferred']['E'].dimension == Dimension(mass=1, length=2, time=-2)

    def test_gravitational_force(self):
        """Test F = G * m1 * m2 / r**2."""
        G_unit = m**3 / (kg * s**2)
        result = infer_units(
            "F = G * m1 * m2 / r**2",
            {"G": G_unit, "m1": kg, "m2": kg, "r": m}
        )

        assert result['is_consistent']
        assert 'F' in result['inferred']
        assert result['inferred']['F'].dimension == Dimension(mass=1, length=1, time=-2)

    def test_square_root_implicit(self):
        """Test v**2 = 2 * a * d."""
        result = infer_units(
            "v_squared = 2 * a * d",
            {"a": m / s**2, "d": m}
        )

        assert result['is_consistent']
        assert 'v_squared' in result['inferred']
        # v^2 should have dimension L^2 T^-2
        assert result['inferred']['v_squared'].dimension == Dimension(length=2, time=-2)


class TestAdditionSubtraction:
    """Test addition and subtraction constraints."""

    def test_addition_same_units(self):
        """Test s = a + b where a and b have same units."""
        result = infer_units(
            "s = a + b",
            {"a": m, "b": m}
        )

        assert result['is_consistent']
        assert 's' in result['inferred']
        assert result['inferred']['s'].dimension == Dimension(length=1)

    def test_subtraction_same_units(self):
        """Test delta_x = x2 - x1."""
        result = infer_units(
            "delta_x = x2 - x1",
            {"x2": m, "x1": m}
        )

        assert result['is_consistent']
        assert 'delta_x' in result['inferred']
        assert result['inferred']['delta_x'].dimension == Dimension(length=1)

    def test_addition_propagates_units(self):
        """Test that addition constraint propagates to unknown."""
        result = infer_units(
            "y = x + a",
            {"x": m, "y": m}
        )

        assert result['is_consistent']
        assert 'a' in result['inferred']
        assert result['inferred']['a'].dimension == Dimension(length=1)


class TestComplexEquations:
    """Test more complex equations."""

    def test_ohms_law(self):
        """Test V = I * R."""
        result = infer_units(
            "V = I * R",
            {"I": A, "R": ohm}
        )

        assert result['is_consistent']
        assert 'V' in result['inferred']
        assert result['inferred']['V'].dimension == Dimension(
            mass=1, length=2, time=-3, current=-1
        )

    def test_ideal_gas_law_partial(self):
        """Test P * V = n * R * T with known P, V, T."""
        # This should infer the product n*R
        R_unit = J / (mol * K)
        result = infer_units(
            "P * V = n * R * T",
            {"P": Pa, "V": m**3, "R": R_unit, "T": K}
        )

        assert result['is_consistent']
        assert 'n' in result['inferred']
        assert result['inferred']['n'].dimension == Dimension(amount=1)

    def test_capacitor_energy(self):
        """Test E = 0.5 * C * V**2."""
        F_unit = A * s / V  # Farad
        result = infer_units(
            "E = 0.5 * C * V**2",
            {"C": F_unit, "V": V}
        )

        assert result['is_consistent']
        assert 'E' in result['inferred']
        assert result['inferred']['E'].dimension == Dimension(mass=1, length=2, time=-2)


class TestErrorDetection:
    """Test error detection for invalid equations."""

    def test_addition_incompatible_units(self):
        """Test that adding kg and m/s² is detected as error."""
        result = infer_units(
            "F = m + a",
            {"m": kg, "a": m / s**2}
        )

        assert not result['is_consistent']
        assert result['confidence'] == 0.0
        assert len(result['errors']) > 0
        # Check error mentions incompatible dimensions
        error_text = ' '.join(result['errors']).lower()
        assert 'addition' in error_text or 'matching' in error_text

    def test_subtraction_incompatible_units(self):
        """Test that subtracting incompatible units is detected."""
        result = infer_units(
            "x = a - b",
            {"a": m, "b": kg}
        )

        assert not result['is_consistent']
        assert len(result['errors']) > 0

    def test_conflicting_constraints(self):
        """Test detection of conflicting constraints."""
        # This equation implies x has two different dimensions
        result = infer_units(
            "x = m + a",
            {"m": kg, "a": m / s**2}
        )

        assert not result['is_consistent']


class TestParentheses:
    """Test equations with parentheses."""

    def test_simple_parentheses(self):
        """Test a = (b + c) * d."""
        result = infer_units(
            "a = (b + c) * d",
            {"b": m, "c": m, "d": s}
        )

        assert result['is_consistent']
        assert 'a' in result['inferred']
        assert result['inferred']['a'].dimension == Dimension(length=1, time=1)

    def test_nested_parentheses(self):
        """Test complex parentheses."""
        result = infer_units(
            "F = m * (v2 - v1) / t",
            {"m": kg, "v2": m / s, "v1": m / s, "t": s}
        )

        assert result['is_consistent']
        assert 'F' in result['inferred']
        assert result['inferred']['F'].dimension == Dimension(mass=1, length=1, time=-2)


class TestEdgeCases:
    """Test edge cases."""

    def test_all_variables_known(self):
        """Test when all variables are known."""
        result = infer_units(
            "F = m * a",
            {"F": N, "m": kg, "a": m / s**2}
        )

        assert result['is_consistent']
        assert len(result['inferred']) == 0  # Nothing to infer
        assert result['confidence'] == 1.0

    def test_no_variables_known(self):
        """Test when no variables are known (can't infer anything)."""
        result = infer_units(
            "F = m * a",
            {}
        )

        # Should be consistent but unable to infer anything
        assert result['is_consistent']
        assert len(result['inferred']) == 0

    def test_single_variable(self):
        """Test x = x (trivial equation)."""
        result = infer_units(
            "x = y",
            {"y": m}
        )

        assert result['is_consistent']
        assert 'x' in result['inferred']
        assert result['inferred']['x'].dimension == Dimension(length=1)

    def test_constants_only(self):
        """Test equation with only constants."""
        result = infer_units(
            "x = 2 * 3 + 5",
            {}
        )

        assert result['is_consistent']
        assert 'x' in result['inferred']
        # Constants are dimensionless
        assert result['inferred']['x'].dimension.is_dimensionless


class TestUnaryOperators:
    """Test unary + and - operators."""

    def test_unary_minus(self):
        """Test a = -b."""
        result = infer_units(
            "a = -b",
            {"b": m}
        )

        assert result['is_consistent']
        assert 'a' in result['inferred']
        assert result['inferred']['a'].dimension == Dimension(length=1)

    def test_unary_plus(self):
        """Test a = +b."""
        result = infer_units(
            "a = +b",
            {"b": kg}
        )

        assert result['is_consistent']
        assert 'a' in result['inferred']
        assert result['inferred']['a'].dimension == Dimension(mass=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

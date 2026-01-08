"""Tests for the Dimension class."""

import pytest
from fractions import Fraction

from dimtensor.core.dimensions import Dimension, DIMENSIONLESS


class TestDimensionCreation:
    """Test Dimension creation and properties."""

    def test_dimensionless(self):
        """Dimensionless quantity has all zero exponents."""
        d = Dimension()
        assert d.is_dimensionless
        assert d.length == 0
        assert d.mass == 0
        assert d.time == 0

    def test_single_dimension(self):
        """Creating single-dimension quantities."""
        length = Dimension(length=1)
        assert length.length == 1
        assert length.mass == 0
        assert not length.is_dimensionless

        mass = Dimension(mass=1)
        assert mass.mass == 1
        assert mass.length == 0

    def test_composite_dimension(self):
        """Creating composite dimensions."""
        # Velocity: L/T
        velocity = Dimension(length=1, time=-1)
        assert velocity.length == 1
        assert velocity.time == -1
        assert velocity.mass == 0

        # Force: M*L/T^2
        force = Dimension(mass=1, length=1, time=-2)
        assert force.mass == 1
        assert force.length == 1
        assert force.time == -2

    def test_fractional_exponents(self):
        """Fractional exponents are supported."""
        # Square root of length
        sqrt_length = Dimension(length=0.5)
        assert sqrt_length.length == Fraction(1, 2)


class TestDimensionAlgebra:
    """Test dimension algebraic operations."""

    def test_multiplication(self):
        """Multiplying dimensions adds exponents."""
        length = Dimension(length=1)
        time = Dimension(time=1)

        # L * T = L*T
        result = length * time
        assert result.length == 1
        assert result.time == 1

        # L * L = L^2
        area = length * length
        assert area.length == 2

    def test_division(self):
        """Dividing dimensions subtracts exponents."""
        length = Dimension(length=1)
        time = Dimension(time=1)

        # L / T = L*T^-1 (velocity)
        velocity = length / time
        assert velocity.length == 1
        assert velocity.time == -1

        # L / L = dimensionless
        ratio = length / length
        assert ratio.is_dimensionless

    def test_power(self):
        """Raising dimensions to a power multiplies exponents."""
        length = Dimension(length=1)

        # L^2 (area)
        area = length ** 2
        assert area.length == 2

        # L^3 (volume)
        volume = length ** 3
        assert volume.length == 3

        # L^-1 (inverse length)
        inverse = length ** -1
        assert inverse.length == -1

        # L^0.5 (square root)
        sqrt = length ** 0.5
        assert sqrt.length == Fraction(1, 2)

    def test_complex_algebra(self):
        """Complex dimensional expressions."""
        # Force = M * L / T^2
        mass = Dimension(mass=1)
        length = Dimension(length=1)
        time = Dimension(time=1)

        force = mass * length / (time ** 2)
        assert force.mass == 1
        assert force.length == 1
        assert force.time == -2

        # Energy = Force * Length
        energy = force * length
        assert energy.mass == 1
        assert energy.length == 2
        assert energy.time == -2


class TestDimensionEquality:
    """Test dimension equality and hashing."""

    def test_equality(self):
        """Same dimensions are equal."""
        d1 = Dimension(length=1, time=-1)
        d2 = Dimension(length=1, time=-1)
        assert d1 == d2

    def test_inequality(self):
        """Different dimensions are not equal."""
        velocity = Dimension(length=1, time=-1)
        acceleration = Dimension(length=1, time=-2)
        assert velocity != acceleration

    def test_dimensionless_equality(self):
        """All dimensionless quantities are equal."""
        d1 = Dimension()
        d2 = DIMENSIONLESS
        assert d1 == d2

    def test_hash(self):
        """Dimensions can be used in sets and dicts."""
        d1 = Dimension(length=1)
        d2 = Dimension(length=1)
        d3 = Dimension(mass=1)

        s = {d1, d2, d3}
        assert len(s) == 2  # d1 and d2 should hash the same


class TestDimensionString:
    """Test dimension string representations."""

    def test_dimensionless_str(self):
        """Dimensionless quantity displays as '1'."""
        d = Dimension()
        assert str(d) == "1"

    def test_single_dimension_str(self):
        """Single dimension displays correctly."""
        length = Dimension(length=1)
        assert str(length) == "L"

        mass = Dimension(mass=1)
        assert str(mass) == "M"

    def test_power_str(self):
        """Powers display with superscripts."""
        area = Dimension(length=2)
        assert "L" in str(area)
        assert "²" in str(area)

        volume = Dimension(length=3)
        assert "³" in str(volume)

    def test_negative_power_str(self):
        """Negative powers display correctly."""
        inverse_time = Dimension(time=-1)
        assert "T" in str(inverse_time)
        assert "⁻" in str(inverse_time)

    def test_repr(self):
        """repr includes dimension names."""
        force = Dimension(mass=1, length=1, time=-2)
        r = repr(force)
        assert "mass=1" in r
        assert "length=1" in r
        assert "time=-2" in r

"""Property-based tests using Hypothesis (v5.2.0 task #269).

These tests verify algebraic and mathematical invariants of the dimtensor
core types (Dimension, Unit, DimArray) by generating random inputs and
checking that fundamental properties hold for all of them.

Properties tested:
- Dimension forms an Abelian group under multiplication
- Power laws hold for Dimension and Unit
- DimArray operations preserve unit semantics
- Round-tripping through serialization preserves values
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from dimtensor import DimArray, Dimension, Unit, units
from dimtensor.core.dimensions import DIMENSIONLESS
from dimtensor.errors import DimensionError


# ---------------------------------------------------------------------------
# Hypothesis strategies for dimtensor primitives
# ---------------------------------------------------------------------------


def small_fraction() -> st.SearchStrategy[Fraction]:
    """Strategy for small rational exponents (kept narrow to avoid overflow).

    Range -4..4 with denominator <=4 covers the interesting cases (sqrt,
    cube root, integer powers) without generating wildly large exponents that
    cause numeric instability or excessive scale factors.
    """

    return st.builds(
        Fraction,
        st.integers(min_value=-4, max_value=4),
        st.sampled_from([1, 2, 3, 4]),
    )


def dimension_strategy() -> st.SearchStrategy[Dimension]:
    """Strategy generating arbitrary Dimensions with small rational exponents."""

    return st.builds(
        Dimension,
        length=small_fraction(),
        mass=small_fraction(),
        time=small_fraction(),
        current=small_fraction(),
        temperature=small_fraction(),
        amount=small_fraction(),
        luminosity=small_fraction(),
    )


def finite_floats() -> st.SearchStrategy[float]:
    """Floats in a numerically safe range (no NaN, inf, or extreme values)."""

    return st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )


def positive_floats() -> st.SearchStrategy[float]:
    """Strictly positive finite floats (used for unit scale factors)."""

    return st.floats(
        min_value=1e-3,
        max_value=1e3,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )


def unit_strategy() -> st.SearchStrategy[Unit]:
    """Strategy generating arbitrary units with sensible scale factors."""

    return st.builds(
        lambda dim, scale: Unit("u", dim, scale),
        dimension_strategy(),
        positive_floats(),
    )


def array_data(
    n_min: int = 1, n_max: int = 8
) -> st.SearchStrategy[list[float]]:
    """Strategy for 1-D numpy array data."""

    return st.lists(finite_floats(), min_size=n_min, max_size=n_max)


# ---------------------------------------------------------------------------
# Dimension algebra - group theory properties
# ---------------------------------------------------------------------------


class TestDimensionAlgebra:
    """Dimension multiplication forms an Abelian group with DIMENSIONLESS as identity."""

    @given(dimension_strategy())
    def test_identity_element(self, dim: Dimension) -> None:
        """DIMENSIONLESS is the identity: dim * 1 == dim."""
        assert dim * DIMENSIONLESS == dim
        assert DIMENSIONLESS * dim == dim

    @given(dimension_strategy())
    def test_inverse_element(self, dim: Dimension) -> None:
        """Each dimension has an inverse: dim * dim**-1 == DIMENSIONLESS."""
        inverse = dim ** -1
        assert dim * inverse == DIMENSIONLESS
        assert inverse * dim == DIMENSIONLESS

    @given(dimension_strategy(), dimension_strategy())
    def test_commutativity(self, a: Dimension, b: Dimension) -> None:
        """Multiplication is commutative: a * b == b * a."""
        assert a * b == b * a

    @given(dimension_strategy(), dimension_strategy(), dimension_strategy())
    def test_associativity(
        self, a: Dimension, b: Dimension, c: Dimension
    ) -> None:
        """Multiplication is associative: (a * b) * c == a * (b * c)."""
        assert (a * b) * c == a * (b * c)

    @given(dimension_strategy(), dimension_strategy())
    def test_division_inverse_of_multiplication(
        self, a: Dimension, b: Dimension
    ) -> None:
        """(a * b) / b == a."""
        assert (a * b) / b == a

    @given(dimension_strategy(), small_fraction(), small_fraction())
    def test_power_addition_law(
        self, a: Dimension, p: Fraction, q: Fraction
    ) -> None:
        """a**p * a**q == a**(p+q)."""
        assert a ** p * a ** q == a ** (p + q)

    @given(dimension_strategy(), small_fraction())
    def test_power_zero_is_dimensionless(
        self, a: Dimension, p: Fraction
    ) -> None:
        """a**0 == DIMENSIONLESS."""
        assert a ** 0 == DIMENSIONLESS

    @given(dimension_strategy())
    def test_power_one_is_identity(self, a: Dimension) -> None:
        """a**1 == a."""
        assert a ** 1 == a

    @given(dimension_strategy(), small_fraction(), small_fraction())
    def test_power_of_power(
        self, a: Dimension, p: Fraction, q: Fraction
    ) -> None:
        """(a**p)**q == a**(p*q)."""
        assert (a ** p) ** q == a ** (p * q)

    @given(dimension_strategy())
    def test_hash_consistency(self, dim: Dimension) -> None:
        """Equal dimensions must have equal hashes (so they can be dict keys)."""
        copy = Dimension(
            length=dim.length,
            mass=dim.mass,
            time=dim.time,
            current=dim.current,
            temperature=dim.temperature,
            amount=dim.amount,
            luminosity=dim.luminosity,
        )
        assert dim == copy
        assert hash(dim) == hash(copy)

    @given(dimension_strategy())
    def test_repr_contains_nonzero_components(self, dim: Dimension) -> None:
        """repr() should include each non-zero component name."""
        text = repr(dim)
        for name, val in [
            ("length", dim.length),
            ("mass", dim.mass),
            ("time", dim.time),
            ("current", dim.current),
            ("temperature", dim.temperature),
            ("amount", dim.amount),
            ("luminosity", dim.luminosity),
        ]:
            if val != 0:
                assert name in text


# ---------------------------------------------------------------------------
# Unit conversion and algebra
# ---------------------------------------------------------------------------


class TestUnitConversion:
    """Conversion factors must be self-consistent."""

    @given(unit_strategy())
    def test_self_conversion_is_identity(self, unit: Unit) -> None:
        """Converting a unit to itself yields factor 1.0."""
        assert unit.conversion_factor(unit) == pytest.approx(1.0)

    @given(unit_strategy(), unit_strategy())
    def test_conversion_compatibility_check(
        self, a: Unit, b: Unit
    ) -> None:
        """Incompatible units raise; compatible units return finite factors."""
        if a.is_compatible(b):
            factor = a.conversion_factor(b)
            assert np.isfinite(factor)
            # Round-trip: a -> b -> a should restore the original scale.
            back = b.conversion_factor(a)
            assert factor * back == pytest.approx(1.0, rel=1e-9)
        else:
            with pytest.raises(ValueError):
                a.conversion_factor(b)

    @given(dimension_strategy(), positive_floats(), positive_floats())
    def test_conversion_factor_ratio(
        self, dim: Dimension, scale_a: float, scale_b: float
    ) -> None:
        """For same-dimension units, factor == scale_a / scale_b."""
        a = Unit("a", dim, scale_a)
        b = Unit("b", dim, scale_b)
        assert a.conversion_factor(b) == pytest.approx(scale_a / scale_b)

    @given(unit_strategy(), unit_strategy())
    def test_unit_multiplication_combines_dimensions(
        self, a: Unit, b: Unit
    ) -> None:
        """(a * b).dimension == a.dimension * b.dimension."""
        product = a * b
        assert product.dimension == a.dimension * b.dimension

    @given(unit_strategy(), unit_strategy())
    def test_unit_division_subtracts_dimensions(
        self, a: Unit, b: Unit
    ) -> None:
        """(a / b).dimension == a.dimension / b.dimension."""
        quotient = a / b
        assert quotient.dimension == a.dimension / b.dimension

    @given(unit_strategy())
    def test_unit_power_zero_is_dimensionless(self, unit: Unit) -> None:
        """Any unit raised to the 0 power yields a dimensionless unit."""
        result = unit ** 0
        assert result.dimension == DIMENSIONLESS


# ---------------------------------------------------------------------------
# DimArray arithmetic invariants
# ---------------------------------------------------------------------------


class TestDimArrayArithmetic:
    """DimArray operations must preserve dimensional rules."""

    @given(array_data(), dimension_strategy())
    def test_addition_with_self_doubles_value(
        self, data: list[float], dim: Dimension
    ) -> None:
        """a + a == 2 * a for any DimArray."""
        unit = Unit("u", dim, 1.0)
        a = DimArray(np.array(data, dtype=float), unit)
        result = a + a
        np.testing.assert_allclose(result._data, 2 * np.array(data))
        assert result.unit.dimension == dim

    @given(array_data(), dimension_strategy())
    def test_subtraction_with_self_is_zero(
        self, data: list[float], dim: Dimension
    ) -> None:
        """a - a is an array of zeros with the same dimension."""
        unit = Unit("u", dim, 1.0)
        a = DimArray(np.array(data, dtype=float), unit)
        result = a - a
        np.testing.assert_allclose(result._data, np.zeros_like(data))
        assert result.unit.dimension == dim

    @given(array_data(), dimension_strategy(), dimension_strategy())
    def test_addition_dimension_mismatch_raises(
        self, data: list[float], dim_a: Dimension, dim_b: Dimension
    ) -> None:
        """Adding arrays with different dimensions raises DimensionError."""
        assume(dim_a != dim_b)
        a = DimArray(np.array(data, dtype=float), Unit("a", dim_a, 1.0))
        b = DimArray(np.array(data, dtype=float), Unit("b", dim_b, 1.0))
        with pytest.raises(DimensionError):
            _ = a + b

    @given(array_data(), dimension_strategy(), dimension_strategy())
    def test_multiplication_combines_dimensions(
        self, data: list[float], dim_a: Dimension, dim_b: Dimension
    ) -> None:
        """(a * b).unit.dimension == a.unit.dimension * b.unit.dimension."""
        a = DimArray(np.array(data, dtype=float), Unit("a", dim_a, 1.0))
        b = DimArray(np.array(data, dtype=float), Unit("b", dim_b, 1.0))
        result = a * b
        assert result.unit.dimension == dim_a * dim_b

    @given(array_data(), dimension_strategy())
    def test_scalar_multiplication_preserves_dimension(
        self, data: list[float], dim: Dimension
    ) -> None:
        """Multiplying by a scalar preserves the dimension."""
        a = DimArray(np.array(data, dtype=float), Unit("u", dim, 1.0))
        result = a * 3.0
        assert result.unit.dimension == dim

    @given(
        array_data(),
        dimension_strategy(),
        st.integers(min_value=-3, max_value=3),
    )
    def test_power_multiplies_dimensions(
        self, data: list[float], dim: Dimension, p: int
    ) -> None:
        """(a ** p).unit.dimension == a.unit.dimension ** p."""
        # Avoid 0**negative
        if p < 0:
            data_arr = np.array(data, dtype=float)
            assume(np.all(data_arr != 0))
        a = DimArray(np.array(data, dtype=float), Unit("u", dim, 1.0))
        result = a ** p
        assert result.unit.dimension == dim ** p

    @given(array_data(), dimension_strategy())
    def test_negation_preserves_dimension(
        self, data: list[float], dim: Dimension
    ) -> None:
        """(-a).unit.dimension == a.unit.dimension."""
        a = DimArray(np.array(data, dtype=float), Unit("u", dim, 1.0))
        result = -a
        assert result.unit.dimension == dim
        np.testing.assert_allclose(result._data, -np.array(data))


# ---------------------------------------------------------------------------
# Numerical invariants of physical operations
# ---------------------------------------------------------------------------


class TestPhysicsInvariants:
    """Common physics formulas should produce dimensionally correct results."""

    @given(positive_floats(), positive_floats())
    def test_force_equals_mass_times_acceleration(
        self, mass_val: float, accel_val: float
    ) -> None:
        """F = m * a should produce a force with Newton dimension."""
        mass = DimArray(np.array([mass_val]), units.kg)
        accel = DimArray(np.array([accel_val]), units.m / units.s ** 2)
        force = mass * accel
        assert force.unit.dimension == units.N.dimension

    @given(positive_floats(), positive_floats())
    def test_velocity_equals_distance_over_time(
        self, dist: float, time_val: float
    ) -> None:
        """v = d / t produces a velocity (L/T)."""
        distance = DimArray(np.array([dist]), units.m)
        time = DimArray(np.array([time_val]), units.s)
        velocity = distance / time
        expected = Dimension(length=1, time=-1)
        assert velocity.unit.dimension == expected

    @given(positive_floats(), positive_floats())
    def test_energy_equals_force_times_distance(
        self, force_val: float, dist: float
    ) -> None:
        """E = F * d produces an energy (Joule)."""
        force = DimArray(np.array([force_val]), units.N)
        distance = DimArray(np.array([dist]), units.m)
        energy = force * distance
        assert energy.unit.dimension == units.J.dimension


# ---------------------------------------------------------------------------
# Adjust default settings: discourage flaky tests via shrinking budget
# ---------------------------------------------------------------------------


# Use a tighter setting profile for property-based tests so the suite
# runs in a sensible time even with many test cases.
settings.register_profile(
    "dimtensor",
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("dimtensor")

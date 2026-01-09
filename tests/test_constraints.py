"""Tests for validation constraints."""

import numpy as np
import pytest

from dimtensor import DimArray, units, ConstraintError
from dimtensor.validation import (
    Constraint,
    Positive,
    NonNegative,
    NonZero,
    Bounded,
    Finite,
    NotNaN,
    validate_all,
    is_all_satisfied,
    ConservationTracker,
)


class TestPositive:
    """Tests for Positive constraint."""

    def test_positive_passes_for_positive_values(self):
        """Test that positive values pass."""
        data = np.array([1.0, 2.0, 3.0])
        constraint = Positive()
        assert constraint.is_satisfied(data)

    def test_positive_fails_for_negative_values(self):
        """Test that negative values fail."""
        data = np.array([1.0, -2.0, 3.0])
        constraint = Positive()
        assert not constraint.is_satisfied(data)

    def test_positive_fails_for_zero(self):
        """Test that zero fails (not strictly positive)."""
        data = np.array([1.0, 0.0, 3.0])
        constraint = Positive()
        assert not constraint.is_satisfied(data)

    def test_positive_passes_for_inf(self):
        """Test that positive infinity passes."""
        data = np.array([1.0, np.inf])
        constraint = Positive()
        assert constraint.is_satisfied(data)

    def test_positive_fails_for_neg_inf(self):
        """Test that negative infinity fails."""
        data = np.array([1.0, -np.inf])
        constraint = Positive()
        assert not constraint.is_satisfied(data)

    def test_positive_fails_for_nan(self):
        """Test that NaN fails."""
        data = np.array([1.0, np.nan])
        constraint = Positive()
        assert not constraint.is_satisfied(data)

    def test_positive_validate_raises(self):
        """Test that validate() raises ConstraintError."""
        data = np.array([1.0, -2.0, 3.0])
        constraint = Positive()
        with pytest.raises(ConstraintError) as exc_info:
            constraint.validate(data)
        assert "Positive" in str(exc_info.value)
        assert exc_info.value.constraint == "Positive"

    def test_positive_empty_array_passes(self):
        """Test that empty array passes (vacuously true)."""
        data = np.array([])
        constraint = Positive()
        assert constraint.is_satisfied(data)


class TestNonNegative:
    """Tests for NonNegative constraint."""

    def test_nonnegative_passes_for_positive(self):
        """Test that positive values pass."""
        data = np.array([1.0, 2.0, 3.0])
        constraint = NonNegative()
        assert constraint.is_satisfied(data)

    def test_nonnegative_passes_for_zero(self):
        """Test that zero passes."""
        data = np.array([0.0, 1.0, 2.0])
        constraint = NonNegative()
        assert constraint.is_satisfied(data)

    def test_nonnegative_fails_for_negative(self):
        """Test that negative values fail."""
        data = np.array([0.0, -1.0, 2.0])
        constraint = NonNegative()
        assert not constraint.is_satisfied(data)


class TestNonZero:
    """Tests for NonZero constraint."""

    def test_nonzero_passes_for_nonzero(self):
        """Test that non-zero values pass."""
        data = np.array([1.0, -2.0, 3.0])
        constraint = NonZero()
        assert constraint.is_satisfied(data)

    def test_nonzero_fails_for_zero(self):
        """Test that zero fails."""
        data = np.array([1.0, 0.0, 3.0])
        constraint = NonZero()
        assert not constraint.is_satisfied(data)

    def test_nonzero_fails_for_nan(self):
        """Test that NaN fails (NaN != 0 is undefined)."""
        data = np.array([1.0, np.nan])
        constraint = NonZero()
        assert not constraint.is_satisfied(data)

    def test_nonzero_passes_for_inf(self):
        """Test that infinity passes (inf != 0)."""
        data = np.array([np.inf, -np.inf])
        constraint = NonZero()
        assert constraint.is_satisfied(data)


class TestBounded:
    """Tests for Bounded constraint."""

    def test_bounded_passes_within_range(self):
        """Test that values within range pass."""
        data = np.array([0.0, 0.5, 1.0])
        constraint = Bounded(0, 1)
        assert constraint.is_satisfied(data)

    def test_bounded_fails_below_min(self):
        """Test that values below min fail."""
        data = np.array([-0.1, 0.5, 1.0])
        constraint = Bounded(0, 1)
        assert not constraint.is_satisfied(data)

    def test_bounded_fails_above_max(self):
        """Test that values above max fail."""
        data = np.array([0.0, 0.5, 1.1])
        constraint = Bounded(0, 1)
        assert not constraint.is_satisfied(data)

    def test_bounded_inclusive(self):
        """Test that bounds are inclusive."""
        data = np.array([0.0, 1.0])  # Exactly on bounds
        constraint = Bounded(0, 1)
        assert constraint.is_satisfied(data)

    def test_bounded_fails_for_nan(self):
        """Test that NaN fails."""
        data = np.array([0.5, np.nan])
        constraint = Bounded(0, 1)
        assert not constraint.is_satisfied(data)

    def test_bounded_fails_for_inf(self):
        """Test that infinity fails (unless bounds are inf)."""
        data = np.array([0.5, np.inf])
        constraint = Bounded(0, 1)
        assert not constraint.is_satisfied(data)

    def test_bounded_with_inf_bounds(self):
        """Test bounded with infinite bounds."""
        data = np.array([-np.inf, 0, np.inf])
        constraint = Bounded(-np.inf, np.inf)
        # NaN still fails
        assert constraint.is_satisfied(data)

    def test_bounded_invalid_range(self):
        """Test that invalid range raises ValueError."""
        with pytest.raises(ValueError):
            Bounded(1, 0)  # min > max

    def test_bounded_repr(self):
        """Test string representation."""
        constraint = Bounded(0, 1)
        assert "Bounded(0, 1)" in repr(constraint)


class TestFinite:
    """Tests for Finite constraint."""

    def test_finite_passes_for_regular_values(self):
        """Test that regular values pass."""
        data = np.array([1.0, -2.0, 0.0])
        constraint = Finite()
        assert constraint.is_satisfied(data)

    def test_finite_fails_for_inf(self):
        """Test that infinity fails."""
        data = np.array([1.0, np.inf])
        constraint = Finite()
        assert not constraint.is_satisfied(data)

    def test_finite_fails_for_nan(self):
        """Test that NaN fails."""
        data = np.array([1.0, np.nan])
        constraint = Finite()
        assert not constraint.is_satisfied(data)


class TestNotNaN:
    """Tests for NotNaN constraint."""

    def test_notnan_passes_for_regular_values(self):
        """Test that regular values pass."""
        data = np.array([1.0, -2.0, 0.0])
        constraint = NotNaN()
        assert constraint.is_satisfied(data)

    def test_notnan_passes_for_inf(self):
        """Test that infinity passes."""
        data = np.array([1.0, np.inf, -np.inf])
        constraint = NotNaN()
        assert constraint.is_satisfied(data)

    def test_notnan_fails_for_nan(self):
        """Test that NaN fails."""
        data = np.array([1.0, np.nan])
        constraint = NotNaN()
        assert not constraint.is_satisfied(data)


class TestValidateAll:
    """Tests for validate_all function."""

    def test_validate_all_passes(self):
        """Test that valid data passes all constraints."""
        data = np.array([0.5, 0.7, 0.9])
        constraints = [Positive(), Bounded(0, 1), Finite()]
        # Should not raise
        validate_all(data, constraints)

    def test_validate_all_fails_on_first_violation(self):
        """Test that validation stops at first violation."""
        data = np.array([-0.5, 0.5])
        constraints = [Positive(), Bounded(0, 1)]
        with pytest.raises(ConstraintError) as exc_info:
            validate_all(data, constraints)
        assert exc_info.value.constraint == "Positive"


class TestIsAllSatisfied:
    """Tests for is_all_satisfied function."""

    def test_is_all_satisfied_true(self):
        """Test that is_all_satisfied returns True for valid data."""
        data = np.array([0.5, 0.7])
        constraints = [Positive(), Bounded(0, 1)]
        assert is_all_satisfied(data, constraints)

    def test_is_all_satisfied_false(self):
        """Test that is_all_satisfied returns False for invalid data."""
        data = np.array([-0.5, 0.5])
        constraints = [Positive(), Bounded(0, 1)]
        assert not is_all_satisfied(data, constraints)


class TestConstraintWithDimArray:
    """Tests for using constraints with DimArray."""

    def test_positive_with_dimarray(self):
        """Test Positive constraint with DimArray data."""
        arr = DimArray([1.0, 2.0, 3.0], units.kg)
        constraint = Positive()
        assert constraint.is_satisfied(arr.data)

    def test_bounded_probability(self):
        """Test Bounded constraint for probability values."""
        prob = DimArray([0.0, 0.5, 1.0], units.dimensionless)
        constraint = Bounded(0, 1)
        assert constraint.is_satisfied(prob.data)

    def test_validate_mass_must_be_positive(self):
        """Test that negative mass fails positive constraint."""
        mass = DimArray([-1.0], units.kg)
        constraint = Positive()
        with pytest.raises(ConstraintError):
            constraint.validate(mass.data)

    def test_multiple_constraints_on_dimarray(self):
        """Test multiple constraints on DimArray."""
        arr = DimArray([0.1, 0.5, 0.9], units.dimensionless)
        constraints = [Positive(), Bounded(0, 1), Finite()]
        assert is_all_satisfied(arr.data, constraints)


class TestDimArrayValidateMethod:
    """Tests for DimArray.validate() method."""

    def test_validate_returns_self_on_success(self):
        """Test that validate() returns self on success."""
        arr = DimArray([1.0, 2.0, 3.0], units.kg)
        result = arr.validate([Positive()])
        assert result is arr

    def test_validate_raises_on_failure(self):
        """Test that validate() raises ConstraintError on failure."""
        arr = DimArray([-1.0, 2.0], units.kg)
        with pytest.raises(ConstraintError):
            arr.validate([Positive()])

    def test_validate_with_none_is_noop(self):
        """Test that validate(None) is a no-op."""
        arr = DimArray([-1.0], units.kg)  # Would fail Positive
        result = arr.validate(None)
        assert result is arr

    def test_validate_chaining(self):
        """Test that validate() can be chained."""
        arr = DimArray([0.5, 0.7], units.dimensionless)
        # Should not raise
        result = arr.validate([Positive()]).validate([Bounded(0, 1)])
        assert result is arr

    def test_validate_multiple_constraints(self):
        """Test validating against multiple constraints."""
        arr = DimArray([0.5, 0.7], units.dimensionless)
        result = arr.validate([Positive(), Bounded(0, 1), Finite()])
        assert result is arr


class TestConstraintRepr:
    """Tests for constraint string representations."""

    def test_positive_repr(self):
        assert "Positive" in repr(Positive())

    def test_nonnegative_repr(self):
        assert "NonNegative" in repr(NonNegative())

    def test_nonzero_repr(self):
        assert "NonZero" in repr(NonZero())

    def test_bounded_repr(self):
        assert "Bounded(0, 1)" in repr(Bounded(0, 1))

    def test_finite_repr(self):
        assert "Finite" in repr(Finite())

    def test_notnan_repr(self):
        assert "NotNaN" in repr(NotNaN())


# =============================================================================
# Conservation Tracker Tests
# =============================================================================


class TestConservationTracker:
    """Tests for ConservationTracker."""

    def test_record_scalar(self):
        """Test recording scalar values."""
        tracker = ConservationTracker("Energy")
        tracker.record(100.0)
        assert len(tracker) == 1
        assert tracker.history[0] == 100.0

    def test_record_dimarray(self):
        """Test recording DimArray values."""
        tracker = ConservationTracker("Energy")
        E = DimArray([100.0], units.J)
        tracker.record(E)
        assert len(tracker) == 1
        assert tracker.history[0] == 100.0

    def test_record_array_sums(self):
        """Test that array values are summed."""
        tracker = ConservationTracker("Total Mass")
        masses = DimArray([1.0, 2.0, 3.0], units.kg)
        tracker.record(masses)
        assert tracker.history[0] == 6.0  # 1 + 2 + 3

    def test_is_conserved_true(self):
        """Test is_conserved returns True for conserved values."""
        tracker = ConservationTracker("Energy")
        tracker.record(100.0)
        tracker.record(100.0)
        tracker.record(100.0)
        assert tracker.is_conserved(rtol=1e-9)

    def test_is_conserved_within_tolerance(self):
        """Test is_conserved with values within tolerance."""
        tracker = ConservationTracker("Energy")
        tracker.record(100.0)
        tracker.record(100.0001)  # Small drift
        assert tracker.is_conserved(rtol=1e-3)  # 0.1% tolerance
        assert not tracker.is_conserved(rtol=1e-6)  # 0.0001% tolerance

    def test_is_conserved_false(self):
        """Test is_conserved returns False for non-conserved values."""
        tracker = ConservationTracker("Energy")
        tracker.record(100.0)
        tracker.record(50.0)  # Big change
        assert not tracker.is_conserved(rtol=1e-3)

    def test_is_conserved_no_values_raises(self):
        """Test is_conserved raises if no values recorded."""
        tracker = ConservationTracker("Energy")
        with pytest.raises(ValueError, match="No values recorded"):
            tracker.is_conserved()

    def test_is_conserved_single_value(self):
        """Test is_conserved returns True for single value."""
        tracker = ConservationTracker("Energy")
        tracker.record(100.0)
        assert tracker.is_conserved()

    def test_drift(self):
        """Test drift calculation."""
        tracker = ConservationTracker("Energy")
        tracker.record(100.0)
        tracker.record(101.0)
        assert abs(tracker.drift() - 0.01) < 1e-10  # 1% drift

    def test_drift_negative(self):
        """Test negative drift."""
        tracker = ConservationTracker("Energy")
        tracker.record(100.0)
        tracker.record(99.0)
        assert abs(tracker.drift() - (-0.01)) < 1e-10  # -1% drift

    def test_drift_too_few_values(self):
        """Test drift raises if fewer than 2 values."""
        tracker = ConservationTracker("Energy")
        tracker.record(100.0)
        with pytest.raises(ValueError, match="Need at least 2"):
            tracker.drift()

    def test_max_drift(self):
        """Test max_drift calculation."""
        tracker = ConservationTracker("Energy")
        tracker.record(100.0)
        tracker.record(101.0)  # 1% drift
        tracker.record(99.0)   # -1% drift
        tracker.record(102.0)  # 2% drift (max)
        assert abs(tracker.max_drift() - 0.02) < 1e-10

    def test_reset(self):
        """Test reset clears history."""
        tracker = ConservationTracker("Energy")
        tracker.record(100.0)
        tracker.record(101.0)
        tracker.reset()
        assert len(tracker) == 0

    def test_unit_consistency_check(self):
        """Test that unit mismatch raises error."""
        tracker = ConservationTracker("Energy")
        E1 = DimArray([100.0], units.J)
        E2 = DimArray([100.0], units.kg)  # Wrong unit!
        tracker.record(E1)
        with pytest.raises(ValueError, match="Unit mismatch"):
            tracker.record(E2)

    def test_repr(self):
        """Test string representation."""
        tracker = ConservationTracker("Total Energy")
        E = DimArray([100.0], units.J)
        tracker.record(E)
        r = repr(tracker)
        assert "Total Energy" in r
        assert "J" in r
        assert "records=1" in r


class TestConservationTrackerWithSimulation:
    """Integration tests for ConservationTracker in physics scenarios."""

    def test_energy_conservation_in_free_fall(self):
        """Test energy conservation for object in free fall."""
        g = 9.8  # m/s^2
        m = 1.0  # kg

        # Initial: h=10m, v=0
        h = 10.0
        v = 0.0
        KE = 0.5 * m * v**2
        PE = m * g * h
        total = KE + PE

        tracker = ConservationTracker("Mechanical Energy")
        tracker.record(total)

        # After falling: h=5m, v=sqrt(2*g*5)
        h = 5.0
        v = np.sqrt(2 * g * 5)
        KE = 0.5 * m * v**2
        PE = m * g * h
        total = KE + PE

        tracker.record(total)

        # Energy should be conserved
        assert tracker.is_conserved(rtol=1e-10)

    def test_mass_conservation(self):
        """Test mass conservation when splitting."""
        tracker = ConservationTracker("Total Mass")

        # Initial: single object
        initial_mass = DimArray([10.0], units.kg)
        tracker.record(initial_mass)

        # After splitting: two objects
        m1 = DimArray([4.0], units.kg)
        m2 = DimArray([6.0], units.kg)
        tracker.record(m1.data.sum() + m2.data.sum())

        assert tracker.is_conserved(rtol=1e-10)

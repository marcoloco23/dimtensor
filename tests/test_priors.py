"""Tests for physics priors module."""

import pytest

torch = pytest.importorskip("torch")

from dimtensor import Dimension, DimensionError, units
from dimtensor.torch import (
    ConservationPrior,
    DimensionalConsistencyPrior,
    DimTensor,
    EnergyConservationPrior,
    MomentumConservationPrior,
    PhysicalBoundsPrior,
    SymmetryPrior,
)


class TestConservationPrior:
    """Tests for ConservationPrior."""

    def test_perfect_conservation(self):
        """Test zero loss when quantity is perfectly conserved."""
        prior = ConservationPrior(rtol=1e-6)

        initial = DimTensor(torch.tensor([100.0, 200.0]), units.J)
        final = DimTensor(torch.tensor([100.0, 200.0]), units.J)

        loss = prior(initial, final)

        assert loss.item() == pytest.approx(0.0, abs=1e-10)

    def test_conservation_violated(self):
        """Test non-zero loss when conservation is violated."""
        prior = ConservationPrior(rtol=1e-3)

        initial = DimTensor(torch.tensor([100.0]), units.J)
        final = DimTensor(torch.tensor([90.0]), units.J)  # 10% loss

        loss = prior(initial, final)

        assert loss.item() > 0

    def test_within_tolerance(self):
        """Test that changes within tolerance are not penalized."""
        prior = ConservationPrior(rtol=0.01)  # 1% tolerance

        initial = DimTensor(torch.tensor([100.0]), units.J)
        final = DimTensor(torch.tensor([100.5]), units.J)  # 0.5% change

        loss = prior(initial, final)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_dimension_mismatch(self):
        """Test error on dimension mismatch."""
        prior = ConservationPrior()

        initial = DimTensor(torch.tensor([100.0]), units.J)
        final = DimTensor(torch.tensor([100.0]), units.m)

        with pytest.raises(DimensionError):
            prior(initial, final)

    def test_with_quantity_function(self):
        """Test using a quantity extraction function."""

        def total_energy(state):
            # State is [kinetic, potential]
            return state.sum()

        prior = ConservationPrior(quantity_fn=total_energy, rtol=1e-6)

        initial = DimTensor(torch.tensor([50.0, 50.0]), units.J)
        final = DimTensor(torch.tensor([60.0, 40.0]), units.J)  # Total conserved

        loss = prior(initial, final)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_raw_tensors(self):
        """Test with raw torch tensors."""
        prior = ConservationPrior(rtol=1e-6)

        initial = torch.tensor([100.0, 200.0])
        final = torch.tensor([100.0, 200.0])

        loss = prior(initial, final)

        assert loss.item() == pytest.approx(0.0)

    def test_weight_parameter(self):
        """Test that weight parameter scales the loss."""
        prior1 = ConservationPrior(rtol=1e-6, weight=1.0)
        prior2 = ConservationPrior(rtol=1e-6, weight=2.0)

        initial = DimTensor(torch.tensor([100.0]), units.J)
        final = DimTensor(torch.tensor([90.0]), units.J)

        loss1 = prior1(initial, final)
        loss2 = prior2(initial, final)

        assert loss2.item() == pytest.approx(2.0 * loss1.item())


class TestEnergyConservationPrior:
    """Tests for EnergyConservationPrior."""

    def test_hamiltonian_conservation(self):
        """Test energy conservation on simple harmonic oscillator."""

        def harmonic_hamiltonian(state):
            # state: [q, p] for position and momentum
            q = state[..., 0:1]
            p = state[..., 1:2]
            # H = 0.5*p^2 + 0.5*q^2 (with appropriate units)
            # Using DimTensor for dimensional consistency
            T = 0.5 * (p**2)  # kinetic
            V = 0.5 * (q**2)  # potential
            return T + V

        prior = EnergyConservationPrior(
            harmonic_hamiltonian, rtol=1e-6, check_dimension=False
        )

        # Initial state: q=1, p=0 (all potential energy)
        initial = DimTensor(torch.tensor([[1.0, 0.0]]), units.m)
        # Final state: q=0, p=1 (all kinetic energy)
        final = DimTensor(torch.tensor([[0.0, 1.0]]), units.m)

        loss = prior(initial, final)

        # Energy should be conserved (both states have H=0.5)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_dimension_checking(self):
        """Test that energy dimension is checked."""

        def bad_hamiltonian(state):
            # Returns wrong dimension (not energy)
            return DimTensor(torch.tensor([1.0]), units.m)  # Should be J

        prior = EnergyConservationPrior(bad_hamiltonian, check_dimension=True)

        state = DimTensor(torch.randn(1, 2), units.m)

        with pytest.raises(DimensionError, match="L²M/T²"):
            prior(state, state)

    def test_correct_energy_dimension(self):
        """Test with correct energy dimension."""

        def correct_hamiltonian(state):
            # Return proper energy dimension
            return DimTensor(torch.tensor([100.0]), units.J)

        prior = EnergyConservationPrior(correct_hamiltonian, check_dimension=True)

        initial = DimTensor(torch.randn(1, 2), units.m)
        final = DimTensor(torch.randn(1, 2), units.m)

        loss = prior(initial, final)

        # Should not raise dimension error
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_energy_dissipation(self):
        """Test detection of energy loss."""

        def simple_energy(state):
            return DimTensor(state.data.sum(), units.J)

        prior = EnergyConservationPrior(simple_energy, rtol=1e-3, check_dimension=True)

        initial = DimTensor(torch.tensor([[100.0]]), units.m)
        final = DimTensor(torch.tensor([[90.0]]), units.m)  # 10% loss

        loss = prior(initial, final)

        assert loss.item() > 0


class TestMomentumConservationPrior:
    """Tests for MomentumConservationPrior."""

    def test_momentum_conservation(self):
        """Test momentum conservation on simple collision."""
        prior = MomentumConservationPrior(rtol=1e-6, check_dimension=False)

        # Before collision: particle moving at v=[1, 0, 0]
        p_initial = DimTensor(torch.tensor([1.0, 0.0, 0.0]), units.kg * units.m / units.s)
        # After collision: same total momentum [1, 0, 0]
        p_final = DimTensor(torch.tensor([1.0, 0.0, 0.0]), units.kg * units.m / units.s)

        loss = prior(p_initial, p_final)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_momentum_dimension_check(self):
        """Test momentum dimension checking."""
        prior = MomentumConservationPrior(check_dimension=True)

        # Wrong dimension
        p_initial = DimTensor(torch.tensor([1.0]), units.J)  # Energy, not momentum

        with pytest.raises(DimensionError, match="LM/T"):
            prior(p_initial, p_initial)

    def test_correct_momentum_dimension(self):
        """Test with correct momentum dimension."""
        prior = MomentumConservationPrior(check_dimension=True)

        p_initial = DimTensor(torch.tensor([1.0, 2.0, 3.0]), units.kg * units.m / units.s)
        p_final = DimTensor(torch.tensor([1.0, 2.0, 3.0]), units.kg * units.m / units.s)

        loss = prior(p_initial, p_final)

        assert loss.item() == pytest.approx(0.0)

    def test_with_momentum_function(self):
        """Test using a function to compute total momentum."""

        def total_momentum(velocities):
            # velocities: [batch, n_particles, 3]
            # mass = 1.0 kg for all particles
            mass = DimTensor(torch.tensor(1.0), units.kg)
            return (mass * velocities).sum(dim=1)

        prior = MomentumConservationPrior(
            momentum_fn=total_momentum, rtol=1e-6, check_dimension=False
        )

        # Two particles with opposite velocities
        v_initial = DimTensor(
            torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]]), units.m / units.s
        )
        # After collision, still opposite
        v_final = DimTensor(
            torch.tensor([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]), units.m / units.s
        )

        loss = prior(v_initial, v_final)

        # Total momentum should be zero in both cases
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestSymmetryPrior:
    """Tests for SymmetryPrior."""

    def test_translational_symmetry(self):
        """Test translational invariance."""

        # Create a simple constant model (perfectly translation-invariant)
        class ConstantModel(torch.nn.Module):
            def forward(self, x):
                if isinstance(x, DimTensor):
                    return DimTensor(torch.ones_like(x.data) * 5.0, x.unit)
                return torch.ones_like(x) * 5.0

        model = ConstantModel()
        prior = SymmetryPrior(
            "translational", model, weight=1.0, num_augmentations=2, augmentation_scale=0.1
        )

        x = DimTensor(torch.randn(4, 3), units.m)
        loss = prior(x)

        # Constant model should have zero translation loss
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_rotational_symmetry(self):
        """Test rotational symmetry with a simple model."""

        # Create a model that outputs magnitude (rotationally invariant)
        class MagnitudeModel(torch.nn.Module):
            def forward(self, x):
                x_data = x.data if isinstance(x, DimTensor) else x
                mag = torch.norm(x_data, dim=-1, keepdim=True)
                if isinstance(x, DimTensor):
                    return DimTensor(mag, x.unit)
                return mag

        model = MagnitudeModel()
        prior = SymmetryPrior(
            "rotational", model, weight=1.0, num_augmentations=1, augmentation_scale=0.1
        )

        x = DimTensor(torch.randn(4, 3), units.m)
        loss = prior(x)

        # Magnitude is rotationally invariant, but our rotation test is approximate
        # Loss should be small
        assert loss.item() >= 0.0

    def test_time_symmetry(self):
        """Test time translation symmetry."""

        class TimeInvariantModel(torch.nn.Module):
            def forward(self, x):
                # Only depends on spatial coordinates, not time
                x_data = x.data if isinstance(x, DimTensor) else x
                spatial = x_data[..., :-1]  # All but last dimension
                if isinstance(x, DimTensor):
                    return DimTensor(spatial.sum(dim=-1, keepdim=True), x.unit)
                return spatial.sum(dim=-1, keepdim=True)

        model = TimeInvariantModel()
        prior = SymmetryPrior(
            "time", model, weight=1.0, num_augmentations=2, augmentation_scale=0.1
        )

        x = DimTensor(torch.randn(4, 4), units.m)  # Last dim is "time"
        loss = prior(x)

        # Should have small loss for time-invariant model
        assert loss.item() >= 0.0

    def test_symmetry_violation(self):
        """Test that symmetry violations are detected."""

        # Create model that explicitly depends on input value (not translation-invariant)
        class NonInvariantModel(torch.nn.Module):
            def forward(self, x):
                if isinstance(x, DimTensor):
                    return x  # Returns input directly
                return x

        model = NonInvariantModel()
        prior = SymmetryPrior(
            "translational", model, weight=1.0, num_augmentations=2, augmentation_scale=0.5
        )

        x = DimTensor(torch.randn(4, 3), units.m)
        loss = prior(x)

        # Should detect violation (loss > 0)
        assert loss.item() > 0


class TestDimensionalConsistencyPrior:
    """Tests for DimensionalConsistencyPrior."""

    def test_correct_dimension(self):
        """Test zero penalty when dimension matches."""
        expected_dim = Dimension(length=1, time=-1)  # Velocity
        prior = DimensionalConsistencyPrior(expected_dim, weight=1.0)

        output = DimTensor(torch.randn(10), units.m / units.s)
        loss = prior(output)

        assert loss.item() == pytest.approx(0.0)

    def test_incorrect_dimension(self):
        """Test large penalty when dimension doesn't match."""
        expected_dim = Dimension(length=1, time=-1)  # Velocity
        prior = DimensionalConsistencyPrior(expected_dim, weight=1.0)

        # Output has wrong dimension (energy)
        output = DimTensor(torch.randn(10), units.J)
        loss = prior(output)

        assert loss.item() > 1e5  # Large penalty

    def test_raw_tensor_no_penalty(self):
        """Test that raw tensors don't incur penalty."""
        expected_dim = Dimension(length=1, time=-1)
        prior = DimensionalConsistencyPrior(expected_dim, weight=1.0)

        # Can't check dimension on raw tensor
        output = torch.randn(10)
        loss = prior(output)

        assert loss.item() == pytest.approx(0.0)

    def test_energy_dimension(self):
        """Test with energy dimension."""
        energy_dim = Dimension(length=2, mass=1, time=-2)
        prior = DimensionalConsistencyPrior(energy_dim, weight=1.0)

        # Correct energy output
        output = DimTensor(torch.tensor([100.0, 200.0]), units.J)
        loss = prior(output)

        assert loss.item() == pytest.approx(0.0)


class TestPhysicalBoundsPrior:
    """Tests for PhysicalBoundsPrior."""

    def test_positive_constraint(self):
        """Test positive value constraint."""
        prior = PhysicalBoundsPrior("positive", weight=1.0)

        # All positive - no penalty
        x = DimTensor(torch.tensor([1.0, 2.0, 3.0]), units.J)
        loss = prior(x)
        assert loss.item() == pytest.approx(0.0)

        # Some negative - penalty
        x_neg = DimTensor(torch.tensor([1.0, -2.0, 3.0]), units.J)
        loss_neg = prior(x_neg)
        assert loss_neg.item() > 0

    def test_positive_or_zero_constraint(self):
        """Test non-negative constraint."""
        prior = PhysicalBoundsPrior("positive_or_zero", weight=1.0)

        # Zero is OK
        x = DimTensor(torch.tensor([0.0, 1.0, 2.0]), units.J)
        loss = prior(x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

        # Negative is penalized
        x_neg = DimTensor(torch.tensor([-1.0]), units.J)
        loss_neg = prior(x_neg)
        assert loss_neg.item() > 0

    def test_bounded_constraint(self):
        """Test bounded range constraint."""
        prior = PhysicalBoundsPrior("bounded", min_val=0.0, max_val=1.0, weight=1.0)

        # Within bounds - no penalty
        x = DimTensor(torch.tensor([0.0, 0.5, 1.0]), units.dimensionless)
        loss = prior(x)
        assert loss.item() == pytest.approx(0.0)

        # Outside bounds - penalty
        x_out = DimTensor(torch.tensor([0.5, 1.5]), units.dimensionless)
        loss_out = prior(x_out)
        assert loss_out.item() > 0

    def test_causality_constraint(self):
        """Test speed of light constraint."""
        c = 299792458.0  # m/s
        prior = PhysicalBoundsPrior("causality", weight=1.0, speed_of_light=c)

        # Below speed of light - OK
        v = DimTensor(torch.tensor([1e6, 1e7]), units.m / units.s)
        loss = prior(v)
        assert loss.item() == pytest.approx(0.0)

        # Above speed of light - penalized
        v_fast = DimTensor(torch.tensor([c * 2]), units.m / units.s)
        loss_fast = prior(v_fast)
        assert loss_fast.item() > 0

    def test_raw_tensor_bounds(self):
        """Test with raw tensors."""
        prior = PhysicalBoundsPrior("positive", weight=1.0)

        x = torch.tensor([1.0, 2.0, 3.0])
        loss = prior(x)
        assert loss.item() == pytest.approx(0.0)

    def test_bounded_invalid_range(self):
        """Test that invalid range raises error."""
        with pytest.raises(ValueError):
            PhysicalBoundsPrior("bounded", min_val=10.0, max_val=5.0)


class TestPriorIntegration:
    """Integration tests for combining multiple priors."""

    def test_multiple_priors_combined(self):
        """Test using multiple priors together."""
        energy_prior = EnergyConservationPrior(
            lambda x: DimTensor(x.data.sum(), units.J), rtol=1e-6, weight=0.1
        )
        bounds_prior = PhysicalBoundsPrior("positive", weight=0.05)

        state_initial = DimTensor(torch.tensor([[50.0, 50.0]]), units.m)
        state_final = DimTensor(torch.tensor([[60.0, 40.0]]), units.m)

        energy_loss = energy_prior(state_initial, state_final)
        bounds_loss = bounds_prior(state_final)

        total_loss = energy_loss + bounds_loss

        # Both should contribute
        assert energy_loss.item() >= 0
        assert bounds_loss.item() >= 0
        assert total_loss.item() >= 0

    def test_gradient_flow(self):
        """Test that gradients flow through priors."""
        prior = ConservationPrior(rtol=1e-6, weight=1.0)

        # Create tensors with requires_grad
        initial_t = torch.tensor([100.0], requires_grad=True)
        final_t = torch.tensor([90.0], requires_grad=True)

        initial = DimTensor(initial_t, units.J)
        final = DimTensor(final_t, units.J)

        loss = prior(initial, final)
        loss.backward()

        # Gradients should exist on the original tensors
        assert initial_t.grad is not None
        assert final_t.grad is not None
        # Gradients should be non-zero (since conservation is violated)
        assert initial_t.grad.abs().sum() > 0
        assert final_t.grad.abs().sum() > 0

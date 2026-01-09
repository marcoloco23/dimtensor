# Plan: Physics Priors Module for ML

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create a comprehensive physics priors module that enables physics-informed machine learning by enforcing conservation laws, symmetries, and physical constraints as soft priors in neural network training.

---

## Background

**Why is this needed?**

Physics-informed neural networks (PINNs) benefit greatly from incorporating physical knowledge as soft constraints during training. Current dimtensor has:
- `PhysicsLoss` for basic conservation law enforcement
- `ConservationTracker` for post-hoc validation
- Dimensional checking built into all operations

However, there's no unified system for encoding physics priors as regularizers that can guide neural network training. Physics priors help:
1. Improve sample efficiency (less training data needed)
2. Enforce physical plausibility (energy conservation, etc.)
3. Improve generalization to out-of-distribution scenarios
4. Encode domain knowledge (symmetries, invariances)

**Use cases:**
- Hamiltonian neural networks (energy-conserving dynamics)
- Lagrangian neural networks (variational mechanics)
- Physics-informed neural ODEs
- Molecular dynamics with ML potentials
- Fluid dynamics surrogates
- Climate modeling

---

## Approach

### Option A: Standalone Prior Classes

Create a new module `dimtensor.torch.priors` with prior classes that can be used independently:

**Pros:**
- Clean separation of concerns
- Priors can be used outside of loss functions
- Easy to test in isolation
- Flexible composition

**Cons:**
- Users need to manually integrate with losses
- More boilerplate code

### Option B: Extend CompositeLoss

Extend existing `CompositeLoss` to directly support physics priors:

**Pros:**
- Seamless integration with existing code
- Less boilerplate for users
- Consistent API

**Cons:**
- Tight coupling
- Less flexible for advanced use cases
- Harder to use priors outside of training loops

### Option C: Hybrid Approach (CHOSEN)

Create standalone prior classes BUT also provide helper integration with `CompositeLoss`:

**Why:**
- Best of both worlds
- Priors are reusable and testable
- Easy integration for common use cases
- Advanced users can compose manually

**Decision:** Hybrid approach - Create `priors.py` with standalone classes + convenience methods for integration with losses.

---

## Design

### Module Structure

```
src/dimtensor/torch/priors.py
```

### Base Class Hierarchy

```python
class PhysicsPrior(nn.Module):
    """Base class for physics priors."""
    def forward(self, model_output, model_input, ...) -> Tensor:
        """Compute prior loss (dimensionless scalar)."""
        ...
```

### Core Prior Classes

#### 1. ConservationPrior
Base class for all conservation law priors.

```python
class ConservationPrior(PhysicsPrior):
    """Enforce conservation of a quantity Q: dQ/dt = 0"""
    def __init__(self, quantity_fn, rtol=1e-6, weight=1.0):
        # quantity_fn: callable that extracts Q from model state
        # Returns penalty for |Q_final - Q_initial| / Q_initial
```

**Use cases:**
- Energy conservation in Hamiltonian systems
- Mass conservation in fluid dynamics
- Charge conservation in electromagnetism

#### 2. EnergyConservationPrior
Specialized for Hamiltonian systems.

```python
class EnergyConservationPrior(ConservationPrior):
    """Enforce energy conservation: H(q,p) = constant"""
    def __init__(self, hamiltonian_fn, rtol=1e-6, weight=1.0):
        # hamiltonian_fn: computes H from (q, p) or model state
```

**Features:**
- Works with DimTensor to ensure energy has correct dimension (L²M/T²)
- Can handle time-dependent Hamiltonians (for open systems)
- Supports both Cartesian and generalized coordinates

#### 3. MomentumConservationPrior
For momentum conservation in isolated systems.

```python
class MomentumConservationPrior(ConservationPrior):
    """Enforce momentum conservation: Σ p_i = constant"""
    def __init__(self, rtol=1e-6, weight=1.0):
        # Automatically sums momentum over particles
```

**Features:**
- Vector-valued conservation (3D momentum)
- Works for multi-particle systems
- Dimensional check: momentum has dimension LM/T

#### 4. SymmetryPrior
Enforce symmetry constraints (translational, rotational, gauge).

```python
class SymmetryPrior(PhysicsPrior):
    """Penalize outputs that violate symmetry."""
    def __init__(self, symmetry_type, weight=1.0):
        # symmetry_type: 'translational', 'rotational', 'time', 'gauge'
```

**Implementations:**
- **Translational**: f(x + a) ≈ f(x) for all a
- **Rotational**: f(R·x) ≈ R·f(x) for rotation matrix R
- **Time**: f(x, t) independent of absolute time
- **Gauge**: physics unchanged under gauge transformation

**How it works:**
- Evaluate model on augmented inputs (shifted, rotated)
- Penalize deviations from expected symmetry behavior
- Uses data augmentation internally

#### 5. DimensionalConsistencyPrior
Ensure outputs have physically sensible dimensions.

```python
class DimensionalConsistencyPrior(PhysicsPrior):
    """Penalize dimensional inconsistencies in model outputs."""
    def __init__(self, expected_output_dim, weight=1.0):
        # expected_output_dim: Dimension object
```

**How it works:**
- If model outputs DimTensor, check dimension matches expected
- Can also check intermediate layer outputs
- Useful for multi-task networks (each head has different dimension)

#### 6. PhysicalBoundsPrior
Enforce physical bounds (e.g., positive energy, causality).

```python
class PhysicalBoundsPrior(PhysicsPrior):
    """Penalize violations of physical bounds."""
    def __init__(self, bounds_type, weight=1.0):
        # bounds_type: 'positive_energy', 'positive_mass', 'causality', etc.
```

**Bounds types:**
- `positive_energy`: Penalize negative kinetic energy
- `positive_mass`: Penalize negative mass/density
- `causality`: Penalize faster-than-light propagation
- `thermodynamic`: Penalize violations of 2nd law (entropy)

#### 7. VariationalPrior
For Lagrangian/variational formulations.

```python
class VariationalPrior(PhysicsPrior):
    """Enforce Euler-Lagrange equations."""
    def __init__(self, lagrangian_fn, weight=1.0):
        # lagrangian_fn: L(q, q_dot, t)
        # Penalizes deviation from d/dt(∂L/∂q̇) - ∂L/∂q = 0
```

**Features:**
- Automatically computes gradients using autograd
- Works with generalized coordinates
- Can handle constraints (constrained Lagrangian)

---

## Implementation Steps

1. [ ] Create `src/dimtensor/torch/priors.py` with base class
2. [ ] Implement `PhysicsPrior` base class with common utilities
3. [ ] Implement `ConservationPrior` base class
4. [ ] Implement `EnergyConservationPrior`
5. [ ] Implement `MomentumConservationPrior`
6. [ ] Implement `SymmetryPrior` (translational, rotational)
7. [ ] Implement `DimensionalConsistencyPrior`
8. [ ] Implement `PhysicalBoundsPrior`
9. [ ] Implement `VariationalPrior`
10. [ ] Add helper function to integrate priors with `CompositeLoss`
11. [ ] Update `torch/__init__.py` to export new classes
12. [ ] Create comprehensive tests in `tests/torch/test_priors.py`
13. [ ] Add documentation and examples
14. [ ] Update ROADMAP.md to mark task complete

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/torch/priors.py` | **CREATE** - New module with all prior classes (~500 lines) |
| `src/dimtensor/torch/__init__.py` | **MODIFY** - Add exports for prior classes |
| `src/dimtensor/torch/losses.py` | **MODIFY** - Add `add_prior()` method to `CompositeLoss` (optional) |
| `tests/torch/test_priors.py` | **CREATE** - Comprehensive tests (~400 lines) |
| `docs/guides/physics-ml-advanced.md` | **CREATE** - Guide on using physics priors (optional) |
| `examples/physics_priors_demo.py` | **CREATE** - Demonstration script (optional) |

---

## Testing Strategy

### Unit Tests (test_priors.py)

#### ConservationPrior:
- [ ] Test energy conservation on simple harmonic oscillator
- [ ] Test momentum conservation on two-body collision
- [ ] Test with DimTensor inputs (correct dimensions)
- [ ] Test tolerance thresholds (rtol, atol)

#### EnergyConservationPrior:
- [ ] Test on Hamiltonian system (pendulum)
- [ ] Test gradient flow (prior should guide towards conservation)
- [ ] Test with time-dependent Hamiltonian

#### MomentumConservationPrior:
- [ ] Test on multi-particle system
- [ ] Test vector-valued conservation (3D)
- [ ] Test dimensional checking

#### SymmetryPrior:
- [ ] Test translational invariance on simple function
- [ ] Test rotational invariance on 3D function
- [ ] Test time symmetry on autonomous system

#### DimensionalConsistencyPrior:
- [ ] Test with DimTensor outputs
- [ ] Test detection of dimension mismatches
- [ ] Test multi-output networks

#### PhysicalBoundsPrior:
- [ ] Test positive energy constraint
- [ ] Test causality constraint (v < c)
- [ ] Test thermodynamic constraints

#### VariationalPrior:
- [ ] Test on simple Lagrangian (free particle)
- [ ] Test Euler-Lagrange equation enforcement
- [ ] Test with generalized coordinates

### Integration Tests:
- [ ] Test integration with `CompositeLoss`
- [ ] Test multiple priors combined
- [ ] Test in training loop (small network)
- [ ] Test gradient backpropagation through priors

### Example Scripts:
- [ ] Hamiltonian neural network with energy prior
- [ ] Molecular dynamics with conservation priors
- [ ] Symmetry-constrained regression

---

## API Examples

### Example 1: Energy Conservation in Hamiltonian NN

```python
from dimtensor.torch import DimTensor, DimLinear, CompositeLoss, DimMSELoss
from dimtensor.torch.priors import EnergyConservationPrior
from dimtensor import units
import torch

# Define Hamiltonian function
def hamiltonian(state):
    """H = kinetic + potential"""
    q, p = state.chunk(2, dim=-1)  # position, momentum
    T = 0.5 * (p ** 2)  # kinetic
    V = 0.5 * (q ** 2)  # potential (harmonic)
    return T + V

# Create prior
energy_prior = EnergyConservationPrior(
    hamiltonian_fn=hamiltonian,
    rtol=1e-5,
    weight=0.1
)

# Create composite loss
loss_fn = CompositeLoss(
    data_loss=DimMSELoss(),
    physics_losses={'energy': (energy_prior, 0.1)}
)

# Training loop
for batch in dataloader:
    initial_state, final_state, target = batch

    # Model predicts dynamics
    pred_final = model(initial_state)

    # Compute loss with energy conservation prior
    data_loss_val = data_loss(pred_final, target)
    energy_initial = hamiltonian(initial_state)
    energy_final = hamiltonian(pred_final)
    prior_loss_val = energy_prior(energy_initial, energy_final)

    total_loss = data_loss_val + 0.1 * prior_loss_val
    total_loss.backward()
```

### Example 2: Symmetry Prior for Rotational Invariance

```python
from dimtensor.torch.priors import SymmetryPrior

# Create rotational symmetry prior
rotation_prior = SymmetryPrior(
    symmetry_type='rotational',
    weight=0.05
)

# During training
for batch in dataloader:
    x, y = batch
    pred = model(x)

    # Data loss
    data_loss = mse_loss(pred, y)

    # Symmetry loss: rotate input, check output transforms correctly
    symmetry_loss = rotation_prior(model, x)

    total_loss = data_loss + 0.05 * symmetry_loss
    total_loss.backward()
```

### Example 3: Multiple Priors Combined

```python
from dimtensor.torch.priors import (
    EnergyConservationPrior,
    MomentumConservationPrior,
    PhysicalBoundsPrior
)

# Molecular dynamics simulator with multiple physics priors
energy_prior = EnergyConservationPrior(hamiltonian, weight=0.1)
momentum_prior = MomentumConservationPrior(weight=0.1)
bounds_prior = PhysicalBoundsPrior('positive_energy', weight=0.05)

# Combine all priors
def physics_loss(initial, final):
    E_init = compute_energy(initial)
    E_final = compute_energy(final)

    p_init = compute_momentum(initial)
    p_final = compute_momentum(final)

    loss = 0.0
    loss += 0.1 * energy_prior(E_init, E_final)
    loss += 0.1 * momentum_prior(p_init, p_final)
    loss += 0.05 * bounds_prior(E_final)

    return loss
```

---

## Risks / Edge Cases

### Risk 1: Conflicting Priors
**Description**: Multiple priors may conflict (e.g., energy conservation + dissipation)

**Mitigation**:
- Document which priors are compatible
- Provide warnings when conflicting priors detected
- Allow users to disable priors selectively

### Risk 2: Computational Overhead
**Description**: Computing symmetry priors requires multiple forward passes (data augmentation)

**Mitigation**:
- Make augmentation configurable (num_augmentations parameter)
- Provide "light" versions with fewer checks
- Document computational cost for each prior

### Risk 3: Gradient Pathology
**Description**: Some priors may have poor gradient landscapes

**Mitigation**:
- Use smooth penalty functions (not hard constraints)
- Provide adjustable tolerances (rtol, atol)
- Test gradient flow in unit tests

### Risk 4: Dimensional Incompatibility
**Description**: Priors may assume specific dimensional structure

**Mitigation**:
- All priors should validate input dimensions
- Provide clear error messages for dimension mismatches
- Document expected dimensions for each prior

### Edge Case 1: Time-Dependent Conservation
**Description**: Some systems have time-varying conserved quantities

**Handling**:
- `EnergyConservationPrior` accepts optional `time_dependent=True`
- Computes dH/dt and penalizes unexpected changes

### Edge Case 2: Approximate Symmetries
**Description**: Physical systems may have approximate, not exact, symmetries

**Handling**:
- All symmetry priors accept tolerance parameters
- Provide "soft" symmetry mode that allows small violations

### Edge Case 3: Discrete Systems
**Description**: Some systems are discrete (not continuous)

**Handling**:
- Priors should work with discrete time steps
- Document finite-difference approximations used

---

## Definition of Done

- [ ] All prior classes implemented and tested
- [ ] Integration with `CompositeLoss` working
- [ ] All unit tests pass (>95% coverage for priors.py)
- [ ] Documentation includes usage examples
- [ ] At least one end-to-end example (Hamiltonian NN)
- [ ] Dimensional checking works correctly
- [ ] Gradients flow properly through all priors
- [ ] CONTINUITY.md updated with completion
- [ ] torch/__init__.py exports updated

---

## Notes / Log

**Design Considerations:**

1. **Why soft priors instead of hard constraints?**
   - Hard constraints require constrained optimization (complex)
   - Soft priors work with standard SGD/Adam
   - Priors can be gradually annealed during training

2. **Why separate from losses.py?**
   - Priors are conceptually different from loss functions
   - Priors encode domain knowledge, losses measure fit
   - Cleaner separation of concerns
   - Easier to extend with new priors

3. **Computational efficiency:**
   - Priors should be computed only when needed
   - Provide "eval mode" that disables priors for inference
   - Cache intermediate computations where possible

4. **Relationship to existing code:**
   - `PhysicsLoss` in losses.py is a simple conservation check
   - New priors are more sophisticated and composable
   - Eventually, `PhysicsLoss` could be implemented using `ConservationPrior`

5. **Future extensions:**
   - Adaptive prior weighting (learn weights during training)
   - Meta-learning priors from data
   - Probabilistic priors (Bayesian formulation)
   - Integration with JAX (jit-compiled priors)

---

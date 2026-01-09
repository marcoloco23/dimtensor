"""Physics priors for physics-informed machine learning.

Provides soft constraint modules (priors) that encode physical knowledge
as regularization terms in neural network training. These priors help improve
sample efficiency, enforce physical plausibility, and improve generalization.

Priors return smooth penalty terms (not hard constraints) that can be
incorporated into loss functions and work with standard optimizers.

Example:
    >>> from dimtensor.torch import DimTensor, priors
    >>> from dimtensor import units
    >>> import torch
    >>>
    >>> # Energy conservation prior
    >>> def hamiltonian(state):
    ...     q, p = state.chunk(2, dim=-1)
    ...     T = 0.5 * (p ** 2)  # kinetic
    ...     V = 0.5 * (q ** 2)  # potential
    ...     return T + V
    >>>
    >>> prior = priors.EnergyConservationPrior(hamiltonian, rtol=1e-5)
    >>> initial_state = DimTensor(torch.randn(32, 4), units.m)
    >>> final_state = model(initial_state)
    >>> penalty = prior(initial_state, final_state)
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..core.dimensions import Dimension
from ..errors import DimensionError
from .dimtensor import DimTensor


class PhysicsPrior(nn.Module):
    """Base class for physics priors.

    All physics priors inherit from this class and implement the forward()
    method to compute a penalty term (dimensionless scalar).

    Args:
        weight: Multiplicative weight for the prior loss. Default: 1.0
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, *args, **kwargs) -> Tensor:
        """Compute prior loss (dimensionless scalar).

        Must be implemented by subclasses.

        Returns:
            Dimensionless penalty term (torch.Tensor).
        """
        raise NotImplementedError("Subclasses must implement forward()")


class ConservationPrior(PhysicsPrior):
    """Enforce conservation of a physical quantity.

    Penalizes violations of conservation laws by computing the relative
    change in a conserved quantity: penalty = |Q_final - Q_initial| / |Q_initial|

    This is a general-purpose conservation prior. For specific quantities
    (energy, momentum), use the specialized classes.

    Args:
        quantity_fn: Optional callable that extracts conserved quantity from state.
                    If None, initial and final quantities must be passed to forward().
        rtol: Relative tolerance for conservation. Changes below rtol are not penalized.
        atol: Absolute tolerance for conservation (for near-zero quantities).
        weight: Weight for the prior loss.

    Example:
        >>> # Define function to extract conserved quantity
        >>> def total_mass(state):
        ...     return state.sum()
        >>>
        >>> prior = ConservationPrior(total_mass, rtol=1e-6)
        >>> penalty = prior(initial_state, final_state)
    """

    def __init__(
        self,
        quantity_fn: Callable[[DimTensor | Tensor], DimTensor | Tensor] | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-10,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.quantity_fn = quantity_fn
        self.rtol = rtol
        self.atol = atol

    def forward(
        self,
        initial: DimTensor | Tensor,
        final: DimTensor | Tensor,
        initial_quantity: DimTensor | Tensor | None = None,
        final_quantity: DimTensor | Tensor | None = None,
    ) -> Tensor:
        """Compute conservation penalty.

        Args:
            initial: Initial state (if quantity_fn is provided) or initial quantity.
            final: Final state (if quantity_fn is provided) or final quantity.
            initial_quantity: Override initial quantity (if quantity_fn is provided).
            final_quantity: Override final quantity (if quantity_fn is provided).

        Returns:
            Dimensionless penalty (squared relative error).
        """
        # Extract quantities
        if initial_quantity is None:
            if self.quantity_fn is not None:
                initial_quantity = self.quantity_fn(initial)
            else:
                initial_quantity = initial

        if final_quantity is None:
            if self.quantity_fn is not None:
                final_quantity = self.quantity_fn(final)
            else:
                final_quantity = final

        # Check dimensions match
        if isinstance(initial_quantity, DimTensor) and isinstance(
            final_quantity, DimTensor
        ):
            if initial_quantity.dimension != final_quantity.dimension:
                raise DimensionError(
                    f"Conservation check requires same dimension: "
                    f"{initial_quantity.dimension} vs {final_quantity.dimension}"
                )
            initial_t = initial_quantity.data
            final_t = final_quantity.data
        else:
            initial_t = (
                initial_quantity.data
                if isinstance(initial_quantity, DimTensor)
                else initial_quantity
            )
            final_t = (
                final_quantity.data if isinstance(final_quantity, DimTensor) else final_quantity
            )

        # Compute relative error
        diff = torch.abs(final_t - initial_t)
        scale = torch.abs(initial_t) + self.atol
        relative_error = diff / scale

        # Quadratic penalty for exceeding tolerance
        excess = torch.relu(relative_error - self.rtol)
        penalty: Tensor = (excess**2).mean()

        return self.weight * penalty


class EnergyConservationPrior(ConservationPrior):
    """Enforce energy conservation in Hamiltonian systems.

    Specialized prior for enforcing energy conservation. Checks that energy
    has the correct physical dimension (L²M/T²) if DimTensor is used.

    Args:
        hamiltonian_fn: Callable that computes total energy H(state) from system state.
                       Should return DimTensor with energy dimension or scalar Tensor.
        rtol: Relative tolerance for energy conservation.
        atol: Absolute tolerance for energy conservation.
        weight: Weight for the prior loss.
        check_dimension: If True, verify energy has correct dimension.

    Example:
        >>> def hamiltonian(state):
        ...     q, p = state.chunk(2, dim=-1)
        ...     T = 0.5 * mass * (p ** 2)  # kinetic
        ...     V = 0.5 * k * (q ** 2)     # potential
        ...     return T + V
        >>>
        >>> prior = EnergyConservationPrior(hamiltonian, rtol=1e-5)
        >>> penalty = prior(initial_state, final_state)
    """

    def __init__(
        self,
        hamiltonian_fn: Callable[[DimTensor | Tensor], DimTensor | Tensor],
        rtol: float = 1e-6,
        atol: float = 1e-10,
        weight: float = 1.0,
        check_dimension: bool = True,
    ) -> None:
        super().__init__(quantity_fn=hamiltonian_fn, rtol=rtol, atol=atol, weight=weight)
        self.check_dimension = check_dimension
        # Energy dimension: L²M/T²
        self.energy_dim = Dimension(length=2, mass=1, time=-2)

    def forward(
        self,
        initial: DimTensor | Tensor,
        final: DimTensor | Tensor,
    ) -> Tensor:
        """Compute energy conservation penalty.

        Args:
            initial: Initial system state.
            final: Final system state.

        Returns:
            Dimensionless penalty.
        """
        # Compute energies
        E_initial = self.quantity_fn(initial)  # type: ignore
        E_final = self.quantity_fn(final)  # type: ignore

        # Check dimension if requested
        if self.check_dimension:
            if isinstance(E_initial, DimTensor):
                if E_initial.dimension != self.energy_dim:
                    raise DimensionError(
                        f"Energy must have dimension L²M/T², got {E_initial.dimension}"
                    )
            if isinstance(E_final, DimTensor):
                if E_final.dimension != self.energy_dim:
                    raise DimensionError(
                        f"Energy must have dimension L²M/T², got {E_final.dimension}"
                    )

        return super().forward(
            initial, final, initial_quantity=E_initial, final_quantity=E_final
        )


class MomentumConservationPrior(ConservationPrior):
    """Enforce momentum conservation in isolated systems.

    Specialized prior for momentum conservation. Handles vector-valued momentum
    (3D) and checks dimensional correctness (LM/T).

    Args:
        momentum_fn: Optional callable that computes total momentum from state.
                    If None, assumes input is momentum directly.
        rtol: Relative tolerance for momentum conservation.
        atol: Absolute tolerance for momentum conservation.
        weight: Weight for the prior loss.
        check_dimension: If True, verify momentum has correct dimension.

    Example:
        >>> def total_momentum(state):
        ...     # state shape: [batch, n_particles, 3] (velocities)
        ...     return (mass * state).sum(dim=1)  # sum over particles
        >>>
        >>> prior = MomentumConservationPrior(total_momentum, rtol=1e-6)
        >>> penalty = prior(initial_state, final_state)
    """

    def __init__(
        self,
        momentum_fn: Callable[[DimTensor | Tensor], DimTensor | Tensor] | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-10,
        weight: float = 1.0,
        check_dimension: bool = True,
    ) -> None:
        super().__init__(quantity_fn=momentum_fn, rtol=rtol, atol=atol, weight=weight)
        self.check_dimension = check_dimension
        # Momentum dimension: LM/T
        self.momentum_dim = Dimension(length=1, mass=1, time=-1)

    def forward(
        self,
        initial: DimTensor | Tensor,
        final: DimTensor | Tensor,
    ) -> Tensor:
        """Compute momentum conservation penalty.

        Args:
            initial: Initial state or momentum.
            final: Final state or momentum.

        Returns:
            Dimensionless penalty (averaged over vector components).
        """
        # Extract momentum
        if self.quantity_fn is not None:
            p_initial = self.quantity_fn(initial)
            p_final = self.quantity_fn(final)
        else:
            p_initial = initial
            p_final = final

        # Check dimension if requested
        if self.check_dimension:
            if isinstance(p_initial, DimTensor):
                if p_initial.dimension != self.momentum_dim:
                    raise DimensionError(
                        f"Momentum must have dimension LM/T, got {p_initial.dimension}"
                    )
            if isinstance(p_final, DimTensor):
                if p_final.dimension != self.momentum_dim:
                    raise DimensionError(
                        f"Momentum must have dimension LM/T, got {p_final.dimension}"
                    )

        return super().forward(
            initial, final, initial_quantity=p_initial, final_quantity=p_final
        )


class SymmetryPrior(PhysicsPrior):
    """Enforce symmetry constraints (translational, rotational, time).

    Penalizes model outputs that violate expected symmetries by evaluating
    the model on augmented inputs and checking transformation behavior.

    Args:
        symmetry_type: Type of symmetry to enforce:
            - 'translational': f(x + a) ≈ f(x)
            - 'rotational': f(R·x) ≈ R·f(x) for rotation R
            - 'time': f(x, t) independent of absolute time
        model: The neural network model to evaluate.
        weight: Weight for the prior loss.
        num_augmentations: Number of augmented samples to generate per input.
        augmentation_scale: Scale of perturbations (for translation/rotation).

    Example:
        >>> model = MyPhysicsNN()
        >>> prior = SymmetryPrior('translational', model, weight=0.05)
        >>> # During training
        >>> x = get_batch()
        >>> penalty = prior(x)
    """

    def __init__(
        self,
        symmetry_type: Literal["translational", "rotational", "time"],
        model: nn.Module,
        weight: float = 1.0,
        num_augmentations: int = 1,
        augmentation_scale: float = 0.1,
    ) -> None:
        super().__init__(weight)
        self.symmetry_type = symmetry_type
        self.model = model
        self.num_augmentations = num_augmentations
        self.augmentation_scale = augmentation_scale

    def forward(self, x: DimTensor | Tensor) -> Tensor:
        """Compute symmetry violation penalty.

        Args:
            x: Input tensor (batch of samples).

        Returns:
            Dimensionless penalty (mean squared symmetry violation).
        """
        # Original output
        with torch.no_grad():
            f_x = self.model(x)

        total_penalty: Tensor = torch.tensor(0.0, device=x.device if isinstance(x, Tensor) else x.data.device)

        for _ in range(self.num_augmentations):
            if self.symmetry_type == "translational":
                # Apply random translation
                x_t = x.data if isinstance(x, DimTensor) else x
                shift = torch.randn_like(x_t) * self.augmentation_scale
                x_aug = x_t + shift

                # Wrap back in DimTensor if needed
                if isinstance(x, DimTensor):
                    x_aug = DimTensor._from_tensor_and_unit(x_aug, x.unit)

                # Evaluate on translated input
                with torch.no_grad():
                    f_x_aug = self.model(x_aug)

                # Translational invariance: f(x+a) should equal f(x)
                f_x_t = f_x.data if isinstance(f_x, DimTensor) else f_x
                f_x_aug_t = f_x_aug.data if isinstance(f_x_aug, DimTensor) else f_x_aug
                penalty: Tensor = ((f_x_aug_t - f_x_t) ** 2).mean()
                total_penalty = total_penalty + penalty

            elif self.symmetry_type == "rotational":
                # Apply random rotation (2D rotation for simplicity)
                x_t = x.data if isinstance(x, DimTensor) else x

                # Generate random 2D rotation angle
                theta = torch.rand(1, device=x_t.device) * 2 * 3.14159 * self.augmentation_scale
                cos_theta = torch.cos(theta)
                sin_theta = torch.sin(theta)

                # Apply rotation to last 2 dimensions
                if x_t.shape[-1] >= 2:
                    x_rot = x_t.clone()
                    x0 = x_t[..., 0]
                    x1 = x_t[..., 1]
                    x_rot[..., 0] = cos_theta * x0 - sin_theta * x1
                    x_rot[..., 1] = sin_theta * x0 + cos_theta * x1

                    if isinstance(x, DimTensor):
                        x_rot_dim = DimTensor._from_tensor_and_unit(x_rot, x.unit)
                    else:
                        x_rot_dim = x_rot

                    # Evaluate on rotated input
                    with torch.no_grad():
                        f_x_rot = self.model(x_rot_dim)

                    # Apply same rotation to output
                    f_x_t = f_x.data if isinstance(f_x, DimTensor) else f_x
                    f_x_rot_t = f_x_rot.data if isinstance(f_x_rot, DimTensor) else f_x_rot

                    if f_x_t.shape[-1] >= 2:
                        f0 = f_x_t[..., 0]
                        f1 = f_x_t[..., 1]
                        f_expected = f_x_t.clone()
                        f_expected[..., 0] = cos_theta * f0 - sin_theta * f1
                        f_expected[..., 1] = sin_theta * f0 + cos_theta * f1

                        penalty = ((f_x_rot_t - f_expected) ** 2).mean()
                        total_penalty = total_penalty + penalty

            elif self.symmetry_type == "time":
                # Time translation: shift all time coordinates by constant
                x_t = x.data if isinstance(x, DimTensor) else x
                time_shift = torch.randn(1, device=x_t.device) * self.augmentation_scale

                # Assume last dimension is time (this is a simplification)
                x_time_shifted = x_t.clone()
                if x_t.shape[-1] > 0:
                    x_time_shifted[..., -1] = x_t[..., -1] + time_shift

                if isinstance(x, DimTensor):
                    x_shifted_dim = DimTensor._from_tensor_and_unit(x_time_shifted, x.unit)
                else:
                    x_shifted_dim = x_time_shifted

                # Time-invariant systems: output should be similar under time shift
                with torch.no_grad():
                    f_x_shifted = self.model(x_shifted_dim)

                f_x_t = f_x.data if isinstance(f_x, DimTensor) else f_x
                f_x_shifted_t = (
                    f_x_shifted.data if isinstance(f_x_shifted, DimTensor) else f_x_shifted
                )
                penalty = ((f_x_shifted_t - f_x_t) ** 2).mean()
                total_penalty = total_penalty + penalty

        # Average over augmentations
        avg_penalty: Tensor = total_penalty / max(self.num_augmentations, 1)
        return self.weight * avg_penalty


class DimensionalConsistencyPrior(PhysicsPrior):
    """Enforce dimensional consistency of model outputs.

    Checks that model outputs have the expected physical dimension and
    penalizes violations. Useful for multi-task networks where different
    outputs should have different dimensions.

    Args:
        expected_dimension: Expected dimension for the output.
        weight: Weight for the prior loss.

    Example:
        >>> from dimtensor import Dimension
        >>> # Expect velocity output (L/T)
        >>> prior = DimensionalConsistencyPrior(
        ...     expected_dimension=Dimension(length=1, time=-1)
        ... )
        >>> output = model(x)
        >>> penalty = prior(output)  # Returns 0 if dimension matches, else large penalty
    """

    def __init__(
        self,
        expected_dimension: Dimension,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.expected_dimension = expected_dimension

    def forward(self, output: DimTensor | Tensor) -> Tensor:
        """Compute dimensional consistency penalty.

        Args:
            output: Model output (DimTensor or Tensor).

        Returns:
            Dimensionless penalty (0 if dimensions match, large value otherwise).
        """
        if isinstance(output, DimTensor):
            if output.dimension != self.expected_dimension:
                # Return large penalty for dimension mismatch
                penalty: Tensor = torch.tensor(
                    1e6, device=output.data.device, dtype=output.data.dtype
                )
                return self.weight * penalty

        # Dimensions match or output is raw tensor (can't check)
        result: Tensor = torch.tensor(0.0, device=output.device if isinstance(output, Tensor) else output.data.device)
        return result


class PhysicalBoundsPrior(PhysicsPrior):
    """Enforce physical bounds on quantities.

    Penalizes violations of physical constraints like positive energy,
    positive mass, or causality (v < c).

    Args:
        bounds_type: Type of bound to enforce:
            - 'positive': Values must be > 0
            - 'positive_or_zero': Values must be >= 0
            - 'bounded': Values must be in [min_val, max_val]
            - 'causality': Speed must be < speed_of_light
        min_val: Minimum value (for 'bounded' type).
        max_val: Maximum value (for 'bounded' type).
        weight: Weight for the prior loss.
        speed_of_light: Speed of light in appropriate units (for 'causality').

    Example:
        >>> # Enforce positive energy
        >>> prior = PhysicalBoundsPrior('positive', weight=0.05)
        >>> energy = compute_energy(state)
        >>> penalty = prior(energy)
    """

    def __init__(
        self,
        bounds_type: Literal["positive", "positive_or_zero", "bounded", "causality"],
        min_val: float | None = None,
        max_val: float | None = None,
        weight: float = 1.0,
        speed_of_light: float = 299792458.0,  # m/s
    ) -> None:
        super().__init__(weight)
        self.bounds_type = bounds_type
        self.min_val = min_val
        self.max_val = max_val
        self.speed_of_light = speed_of_light

        if bounds_type == "bounded":
            if min_val is None or max_val is None:
                raise ValueError("min_val and max_val must be specified for 'bounded' type")
            if min_val > max_val:
                raise ValueError(f"min_val ({min_val}) must be <= max_val ({max_val})")

    def forward(self, x: DimTensor | Tensor) -> Tensor:
        """Compute bounds violation penalty.

        Args:
            x: Values to check.

        Returns:
            Dimensionless penalty (squared violation).
        """
        # Extract tensor
        x_t: Tensor = x.data if isinstance(x, DimTensor) else x

        if self.bounds_type == "positive":
            # Penalize negative values: penalty = max(0, -x)^2
            violation = torch.relu(-x_t)
            penalty: Tensor = (violation**2).mean()

        elif self.bounds_type == "positive_or_zero":
            # Penalize strictly negative values
            violation = torch.relu(-x_t - 1e-8)
            penalty = (violation**2).mean()

        elif self.bounds_type == "bounded":
            # Penalize values outside [min_val, max_val]
            assert self.min_val is not None and self.max_val is not None
            lower_violation = torch.relu(self.min_val - x_t)
            upper_violation = torch.relu(x_t - self.max_val)
            penalty = ((lower_violation**2 + upper_violation**2)).mean()

        elif self.bounds_type == "causality":
            # Penalize speeds exceeding speed of light
            # Assumes x is velocity magnitude
            speed = torch.abs(x_t)
            violation = torch.relu(speed - self.speed_of_light)
            penalty = (violation**2).mean()

        else:
            raise ValueError(f"Unknown bounds_type: {self.bounds_type}")

        return self.weight * penalty


# Convenience exports
__all__ = [
    "PhysicsPrior",
    "ConservationPrior",
    "EnergyConservationPrior",
    "MomentumConservationPrior",
    "SymmetryPrior",
    "DimensionalConsistencyPrior",
    "PhysicalBoundsPrior",
]

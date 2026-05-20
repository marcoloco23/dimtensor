"""Sensitivity analysis tools for unit-aware models.

This module provides local sensitivity analysis (gradient computation) with
automatic dimensional correctness. Supports numerical finite differences
and optional automatic differentiation for PyTorch and JAX backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np

from ..core.dimarray import DimArray
from ..core.units import Unit, dimensionless
from ..errors import DimensionError


# Type aliases for clarity
ParamDict = Dict[str, DimArray]
SensitivityDict = Dict[str, DimArray]


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis.

    Attributes:
        sensitivities: Dictionary mapping parameter names to their sensitivities.
                      Each sensitivity has units of output/parameter.
        normalized: Dictionary of normalized (dimensionless) sensitivity indices.
        ranking: List of (parameter_name, importance_score) tuples, sorted by importance.
        output: The model output value at the evaluation point.
    """
    sensitivities: SensitivityDict
    normalized: Dict[str, float]
    ranking: List[Tuple[str, float]]
    output: DimArray


def local_sensitivity(
    func: Callable[..., DimArray],
    param: DimArray,
    args: tuple = (),
    kwargs: dict | None = None,
    relative_step: float = 1e-5,
    method: str = "central",
    use_autograd: bool = False,
) -> DimArray:
    """Compute local sensitivity (gradient) of a function with respect to a parameter.

    Uses finite differences by default. For PyTorch DimTensor or JAX DimArray,
    can use automatic differentiation if use_autograd=True.

    The sensitivity ∂f/∂p has units of [f]/[p].

    Args:
        func: Function to analyze. Should accept param as first argument and return DimArray.
        param: Parameter to compute sensitivity with respect to.
        args: Additional positional arguments to pass to func.
        kwargs: Additional keyword arguments to pass to func.
        relative_step: Relative step size for finite differences (default 1e-5).
        method: Finite difference method - "central" (default), "forward", or "backward".
        use_autograd: If True, use automatic differentiation (PyTorch/JAX only).

    Returns:
        Sensitivity ∂f/∂param as a DimArray with appropriate units.

    Raises:
        ValueError: If method is not recognized or autograd requested but unavailable.
        DimensionError: If func returns incompatible dimensions.

    Examples:
        >>> from dimtensor import DimArray, units
        >>> def kinetic_energy(v, m):
        ...     return 0.5 * m * v**2
        >>> v = DimArray(10.0, units.m / units.s)
        >>> m = DimArray(2.0, units.kg)
        >>> # Sensitivity of E with respect to v: ∂E/∂v = m * v
        >>> sens = local_sensitivity(kinetic_energy, v, args=(m,))
        >>> print(sens)  # Should be ~20 kg·m/s
        >>> print(sens.unit)  # J·s/m = kg·m/s
    """
    if kwargs is None:
        kwargs = {}

    # Check if autograd is requested
    if use_autograd:
        return _autograd_sensitivity(func, param, args, kwargs)

    # Finite difference methods
    if method not in ["central", "forward", "backward"]:
        raise ValueError(f"Unknown method '{method}'. Use 'central', 'forward', or 'backward'.")

    # Compute step size
    param_value = param._data
    param_shape = param_value.shape

    # For vector/array parameters, compute gradient element by element
    if param_value.size > 1:
        # Initialize gradient array
        f_base = func(param, *args, **kwargs)
        grad_data = np.zeros_like(param_value)

        # Compute gradient for each element
        for idx in np.ndindex(param_shape):
            # Compute step for this element
            if param_value[idx] == 0:
                delta_elem = relative_step
            else:
                delta_elem = np.abs(param_value[idx]) * relative_step

            # Create perturbation array
            perturbation = np.zeros_like(param_value)
            perturbation[idx] = delta_elem

            if method == "central":
                param_plus = DimArray._from_data_and_unit(param_value + perturbation, param._unit)
                param_minus = DimArray._from_data_and_unit(param_value - perturbation, param._unit)
                f_plus = func(param_plus, *args, **kwargs)
                f_minus = func(param_minus, *args, **kwargs)
                diff = np.asarray(f_plus._data - f_minus._data) / (2 * delta_elem)
                grad_data[idx] = diff.item() if diff.size == 1 else diff
            elif method == "forward":
                param_plus = DimArray._from_data_and_unit(param_value + perturbation, param._unit)
                f_plus = func(param_plus, *args, **kwargs)
                diff = np.asarray(f_plus._data - f_base._data) / delta_elem
                grad_data[idx] = diff.item() if diff.size == 1 else diff
            else:  # backward
                param_minus = DimArray._from_data_and_unit(param_value - perturbation, param._unit)
                f_minus = func(param_minus, *args, **kwargs)
                diff = np.asarray(f_base._data - f_minus._data) / delta_elem
                grad_data[idx] = diff.item() if diff.size == 1 else diff

        sensitivity_unit = f_base._unit / param._unit
        return DimArray._from_data_and_unit(grad_data, sensitivity_unit)

    # For scalar parameters, use simpler approach
    if param_value == 0:
        delta = relative_step
    else:
        delta = np.abs(param_value) * relative_step

    # Create perturbed parameters
    if method == "central":
        # Central difference: (f(x+h) - f(x-h)) / 2h
        param_plus = DimArray._from_data_and_unit(param_value + delta, param._unit)
        param_minus = DimArray._from_data_and_unit(param_value - delta, param._unit)

        f_plus = func(param_plus, *args, **kwargs)
        f_minus = func(param_minus, *args, **kwargs)

        df = f_plus - f_minus
        denom = 2 * delta

    elif method == "forward":
        # Forward difference: (f(x+h) - f(x)) / h
        param_plus = DimArray._from_data_and_unit(param_value + delta, param._unit)
        param_base = param

        f_plus = func(param_plus, *args, **kwargs)
        f_base = func(param_base, *args, **kwargs)

        df = f_plus - f_base
        denom = delta

    else:  # backward
        # Backward difference: (f(x) - f(x-h)) / h
        param_base = param
        param_minus = DimArray._from_data_and_unit(param_value - delta, param._unit)

        f_base = func(param_base, *args, **kwargs)
        f_minus = func(param_minus, *args, **kwargs)

        df = f_base - f_minus
        denom = delta

    # Compute derivative: df/dparam
    # Units: [f] / [param]
    sensitivity_value = df._data / denom
    sensitivity_unit = df._unit / param._unit

    return DimArray._from_data_and_unit(sensitivity_value, sensitivity_unit)


def _autograd_sensitivity(
    func: Callable[..., DimArray],
    param: DimArray,
    args: tuple,
    kwargs: dict,
) -> DimArray:
    """Compute sensitivity using automatic differentiation.

    Attempts to use PyTorch or JAX autograd if available.
    """
    # Check if param is a PyTorch DimTensor
    try:
        from ..torch.dimtensor import DimTensor
        if isinstance(param, DimTensor):
            return _pytorch_sensitivity(func, param, args, kwargs)
    except ImportError:
        pass

    # Check if JAX is available and param is JAX-backed
    try:
        import jax
        import jax.numpy as jnp
        if isinstance(param._data, jnp.ndarray):
            return _jax_sensitivity(func, param, args, kwargs)
    except ImportError:
        pass

    raise ValueError(
        "use_autograd=True but parameter is not a PyTorch DimTensor or JAX DimArray. "
        "Use finite differences instead."
    )


def _pytorch_sensitivity(
    func: Callable[..., Any],
    param: Any,
    args: tuple,
    kwargs: dict,
) -> DimArray:
    """Compute sensitivity using PyTorch autograd."""
    import torch

    # Ensure parameter requires grad
    if not param._data.requires_grad:
        param._data.requires_grad_(True)

    # Forward pass
    output = func(param, *args, **kwargs)

    # For scalar output, use simple backward
    if output._data.numel() == 1:
        output._data.backward()
        grad = param._data.grad
        sensitivity_unit = output._unit / param._unit
        return DimArray._from_data_and_unit(grad.detach().cpu().numpy(), sensitivity_unit)
    else:
        # For vector output, sum and compute gradient
        # This gives the gradient of sum(output) w.r.t. param
        loss = output._data.sum()
        loss.backward()
        grad = param._data.grad
        sensitivity_unit = output._unit / param._unit
        return DimArray._from_data_and_unit(grad.detach().cpu().numpy(), sensitivity_unit)


def _jax_sensitivity(
    func: Callable[..., DimArray],
    param: DimArray,
    args: tuple,
    kwargs: dict,
) -> DimArray:
    """Compute sensitivity using JAX autograd."""
    import jax

    # Create a scalar-valued function for grad
    def scalar_func(p_data):
        p = DimArray._from_data_and_unit(p_data, param._unit)
        output = func(p, *args, **kwargs)
        # Sum to get scalar
        return output._data.sum()

    # Compute gradient
    grad_fn = jax.grad(scalar_func)
    grad = grad_fn(param._data)

    # Get output to determine units
    output = func(param, *args, **kwargs)
    sensitivity_unit = output._unit / param._unit

    return DimArray._from_data_and_unit(np.array(grad), sensitivity_unit)


def sensitivity_matrix(
    func: Callable[..., DimArray],
    params: ParamDict,
    relative_step: float = 1e-5,
    method: str = "central",
    use_autograd: bool = False,
) -> SensitivityDict:
    """Compute sensitivity with respect to multiple parameters.

    For a function f with parameters {p1, p2, ..., pn}, computes the sensitivity
    of f with respect to each parameter: {∂f/∂p1, ∂f/∂p2, ..., ∂f/∂pn}.

    Args:
        func: Function to analyze. Should accept keyword arguments matching param keys.
        params: Dictionary mapping parameter names to their DimArray values.
        relative_step: Relative step size for finite differences.
        method: Finite difference method - "central", "forward", or "backward".
        use_autograd: If True, use automatic differentiation when available.

    Returns:
        Dictionary mapping parameter names to their sensitivities (each a DimArray).

    Examples:
        >>> from dimtensor import DimArray, units
        >>> def kinetic_energy(mass, velocity):
        ...     return 0.5 * mass * velocity**2
        >>> params = {
        ...     "mass": DimArray(2.0, units.kg),
        ...     "velocity": DimArray(10.0, units.m / units.s),
        ... }
        >>> sens = sensitivity_matrix(kinetic_energy, params)
        >>> print(sens["mass"])  # ∂E/∂m = 0.5 * v^2
        >>> print(sens["velocity"])  # ∂E/∂v = m * v
    """
    sensitivities = {}

    for param_name, param_value in params.items():
        # Create wrapper function that accepts param as first arg.
        # Bind `param_name` via default argument so the closure captures the
        # iteration's value, not a late-bound reference to the loop variable.
        def func_wrapper(p, *, _name=param_name, **kwargs_with_others):
            all_params = params.copy()
            all_params[_name] = p
            return func(**all_params)

        # Compute sensitivity
        sens = local_sensitivity(
            func_wrapper,
            param_value,
            relative_step=relative_step,
            method=method,
            use_autograd=use_autograd,
        )
        sensitivities[param_name] = sens

    return sensitivities


def rank_parameters(
    func: Callable[..., DimArray],
    params: ParamDict,
    normalization: str = "relative",
    relative_step: float = 1e-5,
    method: str = "central",
    use_autograd: bool = False,
) -> SensitivityResult:
    """Rank parameters by their importance (sensitivity magnitude).

    Computes normalized sensitivity coefficients and ranks parameters from
    most to least important. Handles dimensional correctness by normalizing
    sensitivities to dimensionless importance scores.

    Args:
        func: Function to analyze. Should accept keyword arguments matching param keys.
        params: Dictionary mapping parameter names to their DimArray values.
        normalization: How to normalize sensitivities:
                      - "relative": (∂f/∂p) * (p/f) - relative sensitivity
                      - "absolute": |∂f/∂p| - absolute magnitude
                      - "scaled": |∂f/∂p| * |p| - scaled by parameter magnitude
        relative_step: Relative step size for finite differences.
        method: Finite difference method.
        use_autograd: If True, use automatic differentiation when available.

    Returns:
        SensitivityResult containing sensitivities, normalized values, and ranking.

    Raises:
        ValueError: If normalization method is not recognized.
        ZeroDivisionError: If relative normalization requested but f or p is near zero.

    Examples:
        >>> from dimtensor import DimArray, units
        >>> def kinetic_energy(mass, velocity):
        ...     return 0.5 * mass * velocity**2
        >>> params = {
        ...     "mass": DimArray(2.0, units.kg),
        ...     "velocity": DimArray(10.0, units.m / units.s),
        ... }
        >>> result = rank_parameters(kinetic_energy, params)
        >>> print(result.ranking)  # [("velocity", 2.0), ("mass", 1.0)]
        >>> # velocity has twice the relative importance
    """
    if normalization not in ["relative", "absolute", "scaled"]:
        raise ValueError(
            f"Unknown normalization '{normalization}'. "
            "Use 'relative', 'absolute', or 'scaled'."
        )

    # Compute sensitivities
    sensitivities = sensitivity_matrix(
        func, params, relative_step, method, use_autograd
    )

    # Evaluate function at current parameters
    output = func(**params)

    # Compute normalized importance scores
    normalized = {}

    for param_name, sensitivity in sensitivities.items():
        param_value = params[param_name]

        if normalization == "relative":
            # Relative sensitivity: (∂f/∂p) * (p/f)
            # This is dimensionless: ([f]/[p]) * ([p]/[f]) = 1

            # Check for near-zero values
            if np.any(np.abs(output._data) < 1e-10):
                raise ZeroDivisionError(
                    f"Cannot compute relative sensitivity: output value near zero. "
                    f"Use normalization='absolute' or 'scaled' instead."
                )
            if np.any(np.abs(param_value._data) < 1e-10):
                raise ZeroDivisionError(
                    f"Cannot compute relative sensitivity for parameter '{param_name}': "
                    f"value near zero. Use normalization='absolute' instead."
                )

            # (∂f/∂p) * p / f
            importance = (sensitivity * param_value) / output
            # Extract scalar magnitude
            normalized[param_name] = float(np.abs(importance._data).sum())

        elif normalization == "absolute":
            # Absolute sensitivity: |∂f/∂p|
            # Units: [f]/[p], so we need to make it dimensionless somehow
            # Use the magnitude as-is (user must interpret units)
            normalized[param_name] = float(np.abs(sensitivity._data).sum())

        else:  # scaled
            # Scaled sensitivity: |∂f/∂p| * |p|
            # Units: [f], so this has same units as output
            importance = sensitivity * param_value
            normalized[param_name] = float(np.abs(importance._data).sum())

    # Rank parameters by importance (descending)
    ranking = sorted(normalized.items(), key=lambda x: x[1], reverse=True)

    return SensitivityResult(
        sensitivities=sensitivities,
        normalized=normalized,
        ranking=ranking,
        output=output,
    )


def normalized_sensitivity(
    sensitivity: DimArray,
    param: DimArray,
    output: DimArray,
) -> float:
    """Compute normalized (dimensionless) sensitivity coefficient.

    Computes the relative sensitivity: (∂f/∂p) * (p/f), which is dimensionless
    and represents the relative change in output for a relative change in parameter.

    Args:
        sensitivity: Sensitivity ∂f/∂p (units: [f]/[p]).
        param: Parameter value p (units: [p]).
        output: Function output f (units: [f]).

    Returns:
        Normalized sensitivity as a scalar float.

    Raises:
        ZeroDivisionError: If output is near zero.

    Examples:
        >>> from dimtensor import DimArray, units
        >>> # E = 0.5 * m * v^2, so ∂E/∂v = m * v
        >>> dE_dv = DimArray(20.0, units.kg * units.m / units.s)
        >>> v = DimArray(10.0, units.m / units.s)
        >>> E = DimArray(100.0, units.J)
        >>> norm_sens = normalized_sensitivity(dE_dv, v, E)
        >>> print(norm_sens)  # (20 * 10) / 100 = 2.0
    """
    if np.any(np.abs(output._data) < 1e-10):
        raise ZeroDivisionError(
            "Cannot compute normalized sensitivity: output value near zero."
        )

    # (∂f/∂p) * p / f - dimensionless
    result = (sensitivity * param) / output

    # Return scalar magnitude
    return float(np.abs(result._data).sum())


def tornado_diagram_data(
    func: Callable[..., DimArray],
    params: ParamDict,
    variations: Dict[str, Tuple[DimArray, DimArray]] | None = None,
    relative_variation: float = 0.1,
) -> Dict[str, Dict[str, float]]:
    """Generate data for a tornado diagram (one-at-a-time sensitivity).

    For each parameter, evaluates the function at low and high values while
    keeping other parameters constant. Returns the output variation for each parameter.

    Args:
        func: Function to analyze.
        params: Dictionary of parameter values (baseline).
        variations: Optional dictionary mapping parameter names to (low, high) tuples.
                   If None, uses ±relative_variation around baseline.
        relative_variation: Relative variation to use if variations not specified (default 0.1 = 10%).

    Returns:
        Dictionary mapping parameter names to {"low": value, "high": value, "range": value}.
        Values are the function output (converted to float).

    Examples:
        >>> from dimtensor import DimArray, units
        >>> def kinetic_energy(mass, velocity):
        ...     return 0.5 * mass * velocity**2
        >>> params = {
        ...     "mass": DimArray(2.0, units.kg),
        ...     "velocity": DimArray(10.0, units.m / units.s),
        ... }
        >>> data = tornado_diagram_data(kinetic_energy, params, relative_variation=0.2)
        >>> # data["velocity"]["range"] will be larger than data["mass"]["range"]
    """
    baseline_output = func(**params)
    baseline_value = float(baseline_output._data.sum())

    results = {}

    for param_name, param_value in params.items():
        # Determine low and high values
        if variations and param_name in variations:
            low_val, high_val = variations[param_name]
        else:
            # Use relative variation
            delta = param_value._data * relative_variation
            low_val = DimArray._from_data_and_unit(
                param_value._data - delta, param_value._unit
            )
            high_val = DimArray._from_data_and_unit(
                param_value._data + delta, param_value._unit
            )

        # Evaluate at low value
        params_low = params.copy()
        params_low[param_name] = low_val
        output_low = func(**params_low)

        # Evaluate at high value
        params_high = params.copy()
        params_high[param_name] = high_val
        output_high = func(**params_high)

        # Store results
        low_value = float(output_low._data.sum())
        high_value = float(output_high._data.sum())

        results[param_name] = {
            "low": low_value,
            "high": high_value,
            "range": abs(high_value - low_value),
            "baseline": baseline_value,
        }

    return results

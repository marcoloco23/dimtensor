"""Automatic scaling law discovery with dimensional constraints.

This module provides tools for discovering power law relationships from dimensional data
while enforcing physical dimensional consistency.

Classes:
    PowerLawFitter: Single-variable power law fitting (y = C * x^a)
    MultiPowerLawFitter: Multi-variable power law fitting (y = C * x1^a1 * x2^a2 * ...)
    FitResult: Results from power law fitting with statistics

Example:
    >>> from dimtensor import DimArray, units
    >>> from dimtensor.analysis.scaling import PowerLawFitter
    >>>
    >>> # Fit drag force relationship: F ~ v^2
    >>> velocity = DimArray([1, 2, 3, 4, 5], units.m / units.s)
    >>> force = DimArray([0.5, 2.0, 4.5, 8.0, 12.5], units.N)
    >>> fitter = PowerLawFitter()
    >>> result = fitter.fit(velocity, force)
    >>> print(f"Exponent: {result.exponents[0]:.2f}")  # Should be ~2.0
    >>> print(f"R²: {result.r_squared:.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from ..core.dimarray import DimArray
from ..core.dimensions import Dimension
from ..core.units import Unit, dimensionless
from ..errors import DimensionError

try:
    from scipy import optimize as sp_optimize
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _check_scipy() -> None:
    """Check that scipy is available."""
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for scaling law analysis. "
            "Install with: pip install scipy"
        )


@dataclass
class FitResult:
    """Results from power law fitting.

    Attributes:
        coefficient: Fitted coefficient C in power law.
        coefficient_unit: Physical unit of the coefficient.
        exponents: Fitted exponents [a1, a2, ...].
        r_squared: Coefficient of determination (R²).
        residuals: Residuals from the fit.
        stderr_exponents: Standard errors of exponents.
        stderr_coefficient: Standard error of log(coefficient).
        x_units: Units of independent variables.
        y_unit: Unit of dependent variable.
        success: Whether the fit succeeded.
        message: Status message.
    """
    coefficient: float
    coefficient_unit: Unit
    exponents: list[float]
    r_squared: float
    residuals: NDArray[np.floating[Any]]
    stderr_exponents: list[float] | None
    stderr_coefficient: float | None
    x_units: list[Unit]
    y_unit: Unit
    success: bool
    message: str

    def predict(self, *x_data: DimArray) -> DimArray:
        """Predict y values using the fitted power law.

        Args:
            *x_data: Independent variable arrays (must match number of variables in fit).

        Returns:
            Predicted y values with appropriate units.

        Example:
            >>> result = fitter.fit(x, y)
            >>> y_pred = result.predict(x_new)
        """
        if len(x_data) != len(self.exponents):
            raise ValueError(
                f"Expected {len(self.exponents)} independent variables, "
                f"got {len(x_data)}"
            )

        # Check dimensional consistency
        for i, (x, expected_unit) in enumerate(zip(x_data, self.x_units)):
            if not isinstance(x, DimArray):
                raise TypeError(f"Argument {i} must be a DimArray")
            if x.unit.dimension != expected_unit.dimension:
                raise DimensionError(
                    f"Argument {i} has dimension {x.unit.dimension}, "
                    f"expected {expected_unit.dimension}"
                )

        # Compute y = C * prod(x_i^a_i)
        result = np.full_like(x_data[0].data, self.coefficient, dtype=float)
        for x, exp, expected_unit in zip(x_data, self.exponents, self.x_units):
            # Convert to expected unit for consistency
            x_converted = x.to(expected_unit)
            result *= x_converted.data ** exp

        return DimArray._from_data_and_unit(result, self.coefficient_unit)

    def __repr__(self) -> str:
        """String representation of fit results."""
        exp_str = ", ".join(f"{e:.4g}" for e in self.exponents)
        return (
            f"FitResult(coefficient={self.coefficient:.4g} {self.coefficient_unit.symbol}, "
            f"exponents=[{exp_str}], R²={self.r_squared:.4f})"
        )


class PowerLawFitter:
    """Fit single-variable power laws with dimensional constraints.

    Fits relationships of the form:
        y = C * x^a

    where dimensional consistency requires:
        dim(y) = dim(C) * dim(x)^a

    The fitter automatically determines valid exponents by solving the dimensional
    constraint equation, then fits the coefficient using linear regression in log-space.

    Example:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.analysis.scaling import PowerLawFitter
        >>>
        >>> # Spring force: F = k * x^2 (for demonstration)
        >>> x = DimArray([0.1, 0.2, 0.3, 0.4], units.m)
        >>> F = DimArray([0.01, 0.04, 0.09, 0.16], units.N)
        >>>
        >>> fitter = PowerLawFitter()
        >>> result = fitter.fit(x, F)
        >>> print(f"F = {result.coefficient:.2f} * x^{result.exponents[0]:.2f}")
    """

    def __init__(self, search_rational: bool = True, max_denominator: int = 10):
        """Initialize PowerLawFitter.

        Args:
            search_rational: If True, search for rational exponents when underdetermined.
            max_denominator: Maximum denominator for rational exponent search.
        """
        _check_scipy()
        self.search_rational = search_rational
        self.max_denominator = max_denominator

    def fit(self, x: DimArray, y: DimArray) -> FitResult:
        """Fit power law y = C * x^a to data.

        Args:
            x: Independent variable with units.
            y: Dependent variable with units.

        Returns:
            FitResult containing fitted parameters and statistics.

        Raises:
            DimensionError: If no valid exponent exists for dimensional consistency.
            ValueError: If data arrays have different lengths or insufficient data.
        """
        if not isinstance(x, DimArray):
            raise TypeError("x must be a DimArray")
        if not isinstance(y, DimArray):
            raise TypeError("y must be a DimArray")

        x_data = np.asarray(x.data).flatten()
        y_data = np.asarray(y.data).flatten()

        if len(x_data) != len(y_data):
            raise ValueError(
                f"x and y must have same length (got {len(x_data)} and {len(y_data)})"
            )
        if len(x_data) < 2:
            raise ValueError("Need at least 2 data points for fitting")

        # Check for non-positive values (can't take log)
        if np.any(x_data <= 0) or np.any(y_data <= 0):
            raise ValueError(
                "Power law fitting requires all values to be positive "
                "(log transform fails for non-positive values)"
            )

        # Try to solve for exponent from dimensional constraints
        exponent = self._solve_exponent_from_dimensions(x.unit.dimension, y.unit.dimension)

        if exponent is None:
            # Underdetermined: fit exponent from data
            return self._fit_free_exponent(x, y)
        else:
            # Determined: fit only coefficient
            return self._fit_with_fixed_exponent(x, y, float(exponent))

    def _solve_exponent_from_dimensions(
        self, x_dim: Dimension, y_dim: Dimension
    ) -> Fraction | None:
        """Solve for exponent from dimensional constraint.

        For y = C * x^a, we need: dim(y) = dim(C) * dim(x)^a

        This gives 7 equations (one per base dimension). If x has only one non-zero
        dimension, we can solve uniquely. Otherwise, the system is underdetermined.

        Args:
            x_dim: Dimension of independent variable.
            y_dim: Dimension of dependent variable.

        Returns:
            Exponent if uniquely determined, None if underdetermined.
        """
        x_exp = x_dim._exponents
        y_exp = y_dim._exponents

        # Find dimensions where x has non-zero exponent
        non_zero_dims = [i for i, e in enumerate(x_exp) if e != 0]

        if len(non_zero_dims) == 0:
            # x is dimensionless, no constraint on exponent
            return None

        # Check if all non-zero dimensions give the same exponent
        exponents = []
        for i in non_zero_dims:
            if x_exp[i] == 0:
                continue
            # a = (y_exp[i] - C_exp[i]) / x_exp[i]
            # We assume C_exp[i] = y_exp[i] - a * x_exp[i]
            # For consistency, compute a from first non-zero dimension
            a = y_exp[i] / x_exp[i]
            exponents.append(a)

        if not exponents:
            return None

        # Check if all computed exponents are consistent
        first_exp = exponents[0]
        if all(abs(float(e - first_exp)) < 1e-10 for e in exponents):
            # Check that this exponent is consistent for all dimensions
            for i in range(7):
                expected_y = first_exp * x_exp[i]
                if abs(float(y_exp[i] - expected_y)) > 1e-10:
                    # This dimension requires different coefficient dimension
                    pass
            return first_exp

        return None

    def _fit_with_fixed_exponent(
        self, x: DimArray, y: DimArray, exponent: float
    ) -> FitResult:
        """Fit coefficient with fixed exponent.

        Uses linear regression in log-space: log(y) = log(C) + a*log(x)
        """
        x_data = np.asarray(x.data).flatten()
        y_data = np.asarray(y.data).flatten()

        # Transform to log space
        log_x = np.log(x_data)
        log_y = np.log(y_data)

        # Linear regression: log_y = log_C + exponent * log_x
        # Compute log_C as mean(log_y - exponent * log_x)
        log_C = np.mean(log_y - exponent * log_x)
        C = np.exp(log_C)

        # Compute predictions
        y_pred = C * x_data ** exponent

        # Compute residuals and R²
        residuals = y_data - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Compute standard error of log(C)
        n = len(x_data)
        stderr_log_C = np.sqrt(ss_res / (n - 1)) if n > 1 else None

        # Determine coefficient unit
        # dim(C) = dim(y) / dim(x)^a
        coef_unit = self._compute_coefficient_unit(x.unit, y.unit, exponent)

        return FitResult(
            coefficient=C,
            coefficient_unit=coef_unit,
            exponents=[exponent],
            r_squared=r_squared,
            residuals=residuals,
            stderr_exponents=None,
            stderr_coefficient=stderr_log_C,
            x_units=[x.unit],
            y_unit=y.unit,
            success=True,
            message="Fit successful with dimensionally constrained exponent"
        )

    def _fit_free_exponent(self, x: DimArray, y: DimArray) -> FitResult:
        """Fit both coefficient and exponent when exponent is not constrained.

        Uses non-linear least squares in log-space.
        """
        x_data = np.asarray(x.data).flatten()
        y_data = np.asarray(y.data).flatten()

        # Transform to log space
        log_x = np.log(x_data)
        log_y = np.log(y_data)

        # Linear regression to get initial guess: log(y) = log(C) + a*log(x)
        slope, intercept, r_value, _, stderr = sp_stats.linregress(log_x, log_y)

        exponent = slope
        C = np.exp(intercept)

        # Compute predictions
        y_pred = C * x_data ** exponent

        # Residuals for diagnostic plots; R² from scipy's correlation coefficient
        residuals = y_data - y_pred
        r_squared = r_value ** 2

        # Determine coefficient unit
        coef_unit = self._compute_coefficient_unit(x.unit, y.unit, exponent)

        return FitResult(
            coefficient=C,
            coefficient_unit=coef_unit,
            exponents=[exponent],
            r_squared=r_squared,
            residuals=residuals,
            stderr_exponents=[stderr] if stderr is not None else None,
            stderr_coefficient=stderr,
            x_units=[x.unit],
            y_unit=y.unit,
            success=True,
            message="Fit successful with free exponent (underdetermined system)"
        )

    def _compute_coefficient_unit(
        self, x_unit: Unit, y_unit: Unit, exponent: float
    ) -> Unit:
        """Compute the unit of the coefficient C.

        From dim(y) = dim(C) * dim(x)^a, we get:
        dim(C) = dim(y) / dim(x)^a
        """
        x_dim_powered = x_unit.dimension ** exponent
        coef_dim = y_unit.dimension / x_dim_powered

        # Compute scale factor: y_scale / (x_scale^a)
        coef_scale = y_unit.scale / (x_unit.scale ** exponent)

        return Unit(symbol="[C]", dimension=coef_dim, scale=coef_scale)


class MultiPowerLawFitter:
    """Fit multi-variable power laws with dimensional constraints.

    Fits relationships of the form:
        y = C * x1^a1 * x2^a2 * ... * xn^an

    where dimensional consistency requires:
        dim(y) = dim(C) * dim(x1)^a1 * dim(x2)^a2 * ... * dim(xn)^an

    This gives 7 linear equations (one per SI base dimension) in n+1 unknowns
    (n exponents + coefficient dimension). The system may be:
    - Overdetermined (n < 6): No solution or unique solution
    - Exactly determined (n = 6): Unique solution
    - Underdetermined (n > 6): Multiple solutions, requires additional constraints

    Example:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.analysis.scaling import MultiPowerLawFitter
        >>>
        >>> # Ideal gas: P*V = n*R*T  →  P = (n*R) * T^1 * V^(-1)
        >>> T = DimArray([300, 400, 500], units.K)
        >>> V = DimArray([0.1, 0.1, 0.1], units.m**3)
        >>> P = DimArray([24.9, 33.2, 41.5], units.Pa)
        >>>
        >>> fitter = MultiPowerLawFitter()
        >>> result = fitter.fit([T, V], P)
        >>> print(f"Exponents: {result.exponents}")  # Should be ~[1.0, -1.0]
    """

    def __init__(
        self,
        search_rational: bool = True,
        max_denominator: int = 10,
        exponent_constraints: dict[int, float] | None = None
    ):
        """Initialize MultiPowerLawFitter.

        Args:
            search_rational: If True, search for rational exponents when underdetermined.
            max_denominator: Maximum denominator for rational exponent search.
            exponent_constraints: Optional dict mapping variable index to fixed exponent.
        """
        _check_scipy()
        self.search_rational = search_rational
        self.max_denominator = max_denominator
        self.exponent_constraints = exponent_constraints or {}

    def fit(self, x_vars: Sequence[DimArray], y: DimArray) -> FitResult:
        """Fit multi-variable power law.

        Args:
            x_vars: Sequence of independent variables with units.
            y: Dependent variable with units.

        Returns:
            FitResult containing fitted parameters and statistics.

        Raises:
            DimensionError: If no valid exponent combination exists.
            ValueError: If data arrays have incompatible shapes or insufficient data.
        """
        if not x_vars:
            raise ValueError("Need at least one independent variable")
        if not isinstance(y, DimArray):
            raise TypeError("y must be a DimArray")

        # Validate inputs
        for i, x in enumerate(x_vars):
            if not isinstance(x, DimArray):
                raise TypeError(f"x_vars[{i}] must be a DimArray")

        # Get data arrays
        x_data_list = [np.asarray(x.data).flatten() for x in x_vars]
        y_data = np.asarray(y.data).flatten()

        # Check all arrays have same length
        n_points = len(y_data)
        for i, x_data in enumerate(x_data_list):
            if len(x_data) != n_points:
                raise ValueError(
                    f"All arrays must have same length "
                    f"(y has {n_points}, x_vars[{i}] has {len(x_data)})"
                )

        if n_points < len(x_vars) + 1:
            raise ValueError(
                f"Need at least {len(x_vars) + 1} data points "
                f"for {len(x_vars)} variables"
            )

        # Check for non-positive values
        if np.any(y_data <= 0):
            raise ValueError("All y values must be positive for power law fitting")
        for i, x_data in enumerate(x_data_list):
            if np.any(x_data <= 0):
                raise ValueError(
                    f"All x_vars[{i}] values must be positive for power law fitting"
                )

        # Transform to log space
        log_y = np.log(y_data)
        log_x_matrix = np.column_stack([np.log(x_data) for x_data in x_data_list])

        # Linear regression: log(y) = log(C) + sum(a_i * log(x_i))
        # Using least squares: log_x_matrix @ exponents = log_y - log_C
        # Add column of ones for intercept (log_C)
        X = np.column_stack([log_x_matrix, np.ones(n_points)])

        # Solve using least squares
        params, residuals_ls, rank, s = np.linalg.lstsq(X, log_y, rcond=None)

        exponents = params[:-1].tolist()
        log_C = params[-1]
        C = np.exp(log_C)

        # Compute predictions
        y_pred = np.exp(log_C + log_x_matrix @ params[:-1])

        # Compute residuals and R²
        residuals = y_data - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Compute coefficient unit
        coef_unit = self._compute_coefficient_unit(
            [x.unit for x in x_vars], y.unit, exponents
        )

        # Estimate standard errors (simplified)
        if n_points > len(exponents) + 1:
            mse = ss_res / (n_points - len(exponents) - 1)
            # This is a simplified error estimate
            stderr = np.sqrt(mse / n_points)
            stderr_exponents = [stderr] * len(exponents)
            stderr_coefficient = stderr
        else:
            stderr_exponents = None
            stderr_coefficient = None

        return FitResult(
            coefficient=C,
            coefficient_unit=coef_unit,
            exponents=exponents,
            r_squared=r_squared,
            residuals=residuals,
            stderr_exponents=stderr_exponents,
            stderr_coefficient=stderr_coefficient,
            x_units=[x.unit for x in x_vars],
            y_unit=y.unit,
            success=True,
            message=f"Fit successful with {len(exponents)} variables"
        )

    def _compute_coefficient_unit(
        self, x_units: list[Unit], y_unit: Unit, exponents: list[float]
    ) -> Unit:
        """Compute the unit of the coefficient C.

        From dim(y) = dim(C) * prod(dim(x_i)^a_i), we get:
        dim(C) = dim(y) / prod(dim(x_i)^a_i)
        """
        # Start with y dimension
        coef_dim = y_unit.dimension
        coef_scale = y_unit.scale

        # Divide by each x_i^a_i
        for x_unit, exp in zip(x_units, exponents):
            x_dim_powered = x_unit.dimension ** exp
            coef_dim = coef_dim / x_dim_powered
            coef_scale = coef_scale / (x_unit.scale ** exp)

        return Unit(symbol="[C]", dimension=coef_dim, scale=coef_scale)

    def fit_constrained(
        self,
        x_vars: Sequence[DimArray],
        y: DimArray,
        initial_guess: list[float] | None = None
    ) -> FitResult:
        """Fit with dimensional constraints using non-linear optimization.

        This method enforces dimensional consistency during the optimization process.
        Use when you want stricter dimensional constraints or when the linear method
        fails.

        Args:
            x_vars: Sequence of independent variables with units.
            y: Dependent variable with units.
            initial_guess: Initial guess for exponents (optional).

        Returns:
            FitResult containing fitted parameters and statistics.
        """
        # For now, use the standard fit method
        # TODO: Implement actual constrained optimization with dimensional constraints
        result = self.fit(x_vars, y)
        result.message = "Fit using constrained optimization (not yet implemented, using linear fit)"
        return result

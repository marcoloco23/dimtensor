"""Monte Carlo uncertainty propagation for DimArray.

This module provides statistical uncertainty propagation through random sampling,
enabling analysis of:
- Nonlinear operations where analytical propagation fails
- Correlated input parameters
- Non-Gaussian output distributions
- Higher-order uncertainty effects

Supports multiple sampling strategies:
- Random: Standard Monte Carlo with pseudo-random sampling
- LHS: Latin Hypercube Sampling for better coverage
- Sobol: Quasi-Monte Carlo with low-discrepancy sequences
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.stats import qmc  # type: ignore

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


SamplingMethod = Literal["random", "lhs", "sobol"]


@dataclass
class MCResult:
    """Results from Monte Carlo uncertainty propagation.

    Attributes:
        mean: Mean of the output distribution
        std: Standard deviation of the output distribution
        samples: All samples from the Monte Carlo simulation (shape: (n_samples, ...))
        percentiles: Dictionary of percentile values (e.g., {5: val, 95: val})
        n_samples: Number of samples used
        method: Sampling method used
        converged: Whether the simulation converged (if diagnostics enabled)
    """

    mean: NDArray[np.floating]
    std: NDArray[np.floating]
    samples: NDArray[np.floating]
    percentiles: dict[float, NDArray[np.floating]]
    n_samples: int
    method: SamplingMethod
    converged: bool = True

    def percentile(self, q: float) -> NDArray[np.floating]:
        """Get the q-th percentile of the distribution.

        Args:
            q: Percentile to compute (0-100)

        Returns:
            Array of percentile values
        """
        return np.percentile(self.samples, q, axis=0)

    def confidence_interval(
        self, confidence: float = 0.95
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Get symmetric confidence interval.

        Args:
            confidence: Confidence level (0-1)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        alpha = 1 - confidence
        lower = self.percentile(100 * alpha / 2)
        upper = self.percentile(100 * (1 - alpha / 2))
        return lower, upper

    def histogram(self, bins: int = 50) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Compute histogram of the distribution.

        Args:
            bins: Number of histogram bins

        Returns:
            Tuple of (counts, bin_edges)
        """
        # For scalar output
        if self.samples.ndim == 1 or (
            self.samples.ndim == 2 and self.samples.shape[1] == 1
        ):
            samples_flat = self.samples.flatten()
            return np.histogram(samples_flat, bins=bins)  # type: ignore
        else:
            # For array output, return histogram of first element
            samples_flat = self.samples[:, 0].flatten()
            return np.histogram(samples_flat, bins=bins)  # type: ignore

    def standard_error(self) -> NDArray[np.floating]:
        """Compute standard error of the mean estimate.

        Returns:
            Standard error: std / sqrt(n_samples)
        """
        return self.std / np.sqrt(self.n_samples)


class RandomSampler:
    """Simple random sampling using numpy's random number generator."""

    def __init__(self, seed: int | None = None):
        """Initialize random sampler.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        n_samples: int,
        means: NDArray[np.floating],
        stds: NDArray[np.floating],
        correlation: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Generate random samples from Gaussian distributions.

        Args:
            n_samples: Number of samples to generate
            means: Mean values for each input (shape: (n_inputs,))
            stds: Standard deviations for each input (shape: (n_inputs,))
            correlation: Optional correlation matrix (shape: (n_inputs, n_inputs))

        Returns:
            Array of samples (shape: (n_samples, n_inputs))
        """
        n_inputs = len(means)

        if correlation is not None:
            # Generate correlated samples using Cholesky decomposition
            cov = _correlation_to_covariance(correlation, stds)
            samples = self.rng.multivariate_normal(means, cov, size=n_samples)
        else:
            # Generate independent samples
            samples = np.zeros((n_samples, n_inputs))
            for i in range(n_inputs):
                samples[:, i] = self.rng.normal(means[i], stds[i], n_samples)

        return samples


class LHSSampler:
    """Latin Hypercube Sampling for improved coverage."""

    def __init__(self, seed: int | None = None):
        """Initialize LHS sampler.

        Args:
            seed: Random seed for reproducibility

        Raises:
            ImportError: If scipy is not installed
        """
        if not HAS_SCIPY:
            raise ImportError(
                "Latin Hypercube Sampling requires scipy. "
                "Install with: pip install scipy"
            )
        self.seed = seed

    def sample(
        self,
        n_samples: int,
        means: NDArray[np.floating],
        stds: NDArray[np.floating],
        correlation: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Generate LHS samples from Gaussian distributions.

        Args:
            n_samples: Number of samples to generate
            means: Mean values for each input (shape: (n_inputs,))
            stds: Standard deviations for each input (shape: (n_inputs,))
            correlation: Optional correlation matrix (shape: (n_inputs, n_inputs))

        Returns:
            Array of samples (shape: (n_samples, n_inputs))
        """
        n_inputs = len(means)

        # Generate LHS samples in [0, 1] hypercube
        sampler = qmc.LatinHypercube(d=n_inputs, seed=self.seed)
        uniform_samples = sampler.random(n=n_samples)

        # Transform to standard normal using inverse CDF
        from scipy import stats

        normal_samples = stats.norm.ppf(uniform_samples)

        # Scale and shift to desired mean/std
        if correlation is not None:
            # Apply correlation using Cholesky decomposition
            L = np.linalg.cholesky(correlation)
            correlated_samples = normal_samples @ L.T
            samples = correlated_samples * stds + means
        else:
            samples = normal_samples * stds + means

        return samples


class SobolSampler:
    """Quasi-Monte Carlo sampling using Sobol sequences."""

    def __init__(self, seed: int | None = None):
        """Initialize Sobol sampler.

        Args:
            seed: Random seed for reproducibility

        Raises:
            ImportError: If scipy is not installed
        """
        if not HAS_SCIPY:
            raise ImportError(
                "Sobol sampling requires scipy. " "Install with: pip install scipy"
            )
        self.seed = seed

    def sample(
        self,
        n_samples: int,
        means: NDArray[np.floating],
        stds: NDArray[np.floating],
        correlation: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Generate Sobol samples from Gaussian distributions.

        Args:
            n_samples: Number of samples to generate
            means: Mean values for each input (shape: (n_inputs,))
            stds: Standard deviations for each input (shape: (n_inputs,))
            correlation: Optional correlation matrix (shape: (n_inputs, n_inputs))

        Returns:
            Array of samples (shape: (n_samples, n_inputs))
        """
        n_inputs = len(means)

        # Generate Sobol samples in [0, 1] hypercube
        sampler = qmc.Sobol(d=n_inputs, scramble=True, seed=self.seed)
        uniform_samples = sampler.random(n=n_samples)

        # Transform to standard normal using inverse CDF
        from scipy import stats

        normal_samples = stats.norm.ppf(uniform_samples)

        # Scale and shift to desired mean/std
        if correlation is not None:
            # Apply correlation using Cholesky decomposition
            L = np.linalg.cholesky(correlation)
            correlated_samples = normal_samples @ L.T
            samples = correlated_samples * stds + means
        else:
            samples = normal_samples * stds + means

        return samples


def _get_sampler(method: SamplingMethod, seed: int | None = None):
    """Get sampler instance for the specified method.

    Args:
        method: Sampling method ("random", "lhs", "sobol")
        seed: Random seed for reproducibility

    Returns:
        Sampler instance

    Raises:
        ValueError: If method is not recognized
    """
    if method == "random":
        return RandomSampler(seed=seed)
    elif method == "lhs":
        return LHSSampler(seed=seed)
    elif method == "sobol":
        return SobolSampler(seed=seed)
    else:
        raise ValueError(
            f"Unknown sampling method: {method}. "
            f"Choose from: 'random', 'lhs', 'sobol'"
        )


def _correlation_to_covariance(
    correlation: NDArray[np.floating], stds: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Convert correlation matrix to covariance matrix.

    Args:
        correlation: Correlation matrix (n_inputs, n_inputs)
        stds: Standard deviations (n_inputs,)

    Returns:
        Covariance matrix (n_inputs, n_inputs)
    """
    D = np.diag(stds)
    return D @ correlation @ D


def _validate_correlation_matrix(correlation: NDArray[np.floating]) -> None:
    """Validate that correlation matrix is valid.

    Args:
        correlation: Correlation matrix to validate

    Raises:
        ValueError: If matrix is invalid
    """
    # Check square
    if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
        raise ValueError(
            f"Correlation matrix must be square, got shape {correlation.shape}"
        )

    # Check symmetric
    if not np.allclose(correlation, correlation.T):
        raise ValueError("Correlation matrix must be symmetric")

    # Check diagonal is 1
    if not np.allclose(np.diag(correlation), 1.0):
        raise ValueError("Correlation matrix diagonal must be 1")

    # Check values in [-1, 1]
    if np.any(correlation < -1) or np.any(correlation > 1):
        raise ValueError("Correlation matrix values must be in [-1, 1]")

    # Check positive semi-definite (eigenvalues >= 0)
    eigenvalues = np.linalg.eigvalsh(correlation)
    if np.any(eigenvalues < -1e-10):
        raise ValueError(
            f"Correlation matrix must be positive semi-definite. "
            f"Minimum eigenvalue: {eigenvalues.min()}"
        )


def _check_convergence(
    samples: NDArray[np.floating], window: int = 100
) -> tuple[bool, float]:
    """Check if Monte Carlo simulation has converged.

    Uses running mean stability as convergence criterion.

    Args:
        samples: Array of samples (shape: (n_samples, ...))
        window: Window size for running mean calculation

    Returns:
        Tuple of (converged, relative_change)
    """
    n_samples = samples.shape[0]
    if n_samples < 2 * window:
        return False, float("inf")

    # Compute mean of first and last window
    first_window_mean = np.mean(samples[:window], axis=0)
    last_window_mean = np.mean(samples[-window:], axis=0)

    # Compute relative change
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_change = np.abs(
            (last_window_mean - first_window_mean)
            / (np.abs(first_window_mean) + 1e-10)
        )

    # Consider converged if relative change < 1%
    converged = np.all(rel_change < 0.01)
    max_rel_change = float(np.max(rel_change))

    return converged, max_rel_change


def monte_carlo(
    func: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    means: NDArray[np.floating] | list[float],
    stds: NDArray[np.floating] | list[float],
    n_samples: int = 10000,
    method: SamplingMethod = "random",
    correlation: NDArray[np.floating] | None = None,
    percentiles: list[float] | None = None,
    seed: int | None = None,
    check_convergence: bool = True,
) -> MCResult:
    """Propagate uncertainty using Monte Carlo sampling.

    Args:
        func: Function to propagate uncertainty through.
              Takes array of shape (n_inputs,) and returns array of shape (...).
        means: Mean values for each input (length n_inputs)
        stds: Standard deviations for each input (length n_inputs)
        n_samples: Number of Monte Carlo samples
        method: Sampling method ("random", "lhs", "sobol")
        correlation: Optional correlation matrix (shape: (n_inputs, n_inputs))
        percentiles: List of percentiles to compute (e.g., [5, 95])
        seed: Random seed for reproducibility
        check_convergence: If True, check convergence and warn if not converged

    Returns:
        MCResult with mean, std, samples, and percentiles

    Raises:
        ValueError: If inputs are invalid or correlation matrix is invalid

    Examples:
        >>> # Simple function with 2 inputs
        >>> def f(x):
        ...     return x[0]**2 + x[1]**2
        >>> result = monte_carlo(f, means=[1.0, 2.0], stds=[0.1, 0.2])
        >>> print(f"Mean: {result.mean:.3f} ± {result.std:.3f}")

        >>> # With correlation
        >>> corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        >>> result = monte_carlo(f, means=[1.0, 2.0], stds=[0.1, 0.2],
        ...                      correlation=corr, method="lhs")
    """
    # Convert to numpy arrays
    means_arr = np.asarray(means, dtype=float)
    stds_arr = np.asarray(stds, dtype=float)

    # Validate inputs
    if means_arr.ndim != 1 or stds_arr.ndim != 1:
        raise ValueError("means and stds must be 1-dimensional")
    if len(means_arr) != len(stds_arr):
        raise ValueError(
            f"means and stds must have same length, "
            f"got {len(means_arr)} and {len(stds_arr)}"
        )
    if np.any(stds_arr < 0):
        raise ValueError("Standard deviations must be non-negative")
    if n_samples < 2:
        raise ValueError(f"n_samples must be at least 2, got {n_samples}")

    # Validate correlation matrix if provided
    if correlation is not None:
        correlation_arr = np.asarray(correlation, dtype=float)
        if correlation_arr.shape != (len(means_arr), len(means_arr)):
            raise ValueError(
                f"Correlation matrix shape {correlation_arr.shape} "
                f"must match (n_inputs, n_inputs) = ({len(means_arr)}, {len(means_arr)})"
            )
        _validate_correlation_matrix(correlation_arr)
    else:
        correlation_arr = None

    # Handle zero uncertainties (fixed parameters)
    zero_std_mask = stds_arr == 0
    if np.all(zero_std_mask):
        # All parameters fixed, return deterministic result
        output = func(means_arr)
        output_arr = np.atleast_1d(output)
        samples_arr = np.tile(output_arr, (n_samples, 1))
        percentiles_dict = {}
        if percentiles is not None:
            for p in percentiles:
                percentiles_dict[p] = output_arr
        return MCResult(
            mean=output_arr,
            std=np.zeros_like(output_arr),
            samples=samples_arr,
            percentiles=percentiles_dict,
            n_samples=n_samples,
            method=method,
            converged=True,
        )

    # Get sampler
    sampler = _get_sampler(method, seed=seed)

    # Generate input samples
    input_samples = sampler.sample(n_samples, means_arr, stds_arr, correlation_arr)

    # Evaluate function on all samples
    output_samples_list = []
    for i in range(n_samples):
        output = func(input_samples[i])
        output_samples_list.append(np.atleast_1d(output))

    # Stack output samples (shape: (n_samples, output_shape...))
    output_samples = np.array(output_samples_list)

    # Compute statistics
    mean = np.mean(output_samples, axis=0)
    std = np.std(output_samples, axis=0, ddof=1)

    # Compute percentiles
    percentiles_dict = {}
    if percentiles is not None:
        for p in percentiles:
            percentiles_dict[p] = np.percentile(output_samples, p, axis=0)

    # Check convergence
    converged = True
    if check_convergence:
        converged, rel_change = _check_convergence(output_samples)
        if not converged:
            import warnings

            warnings.warn(
                f"Monte Carlo simulation may not have converged. "
                f"Relative change in mean: {rel_change:.3%}. "
                f"Consider increasing n_samples.",
                RuntimeWarning,
                stacklevel=2,
            )

    return MCResult(
        mean=mean,
        std=std,
        samples=output_samples,
        percentiles=percentiles_dict,
        n_samples=n_samples,
        method=method,
        converged=converged,
    )


def monte_carlo_dimarray(
    func: Callable,
    dimarrays: list,
    n_samples: int = 10000,
    method: SamplingMethod = "random",
    correlation: NDArray[np.floating] | None = None,
    percentiles: list[float] | None = None,
    seed: int | None = None,
    check_convergence: bool = True,
):
    """Monte Carlo uncertainty propagation for DimArray objects.

    This is a convenience wrapper around monte_carlo() that handles DimArray
    inputs and preserves units in the output.

    Args:
        func: Function that takes DimArray objects and returns a DimArray
        dimarrays: List of DimArray objects with uncertainty
        n_samples: Number of Monte Carlo samples
        method: Sampling method ("random", "lhs", "sobol")
        correlation: Optional correlation matrix
        percentiles: List of percentiles to compute
        seed: Random seed for reproducibility
        check_convergence: If True, check convergence and warn if not converged

    Returns:
        MCResult with mean, std, samples (all as DimArrays with appropriate units)

    Raises:
        ValueError: If any DimArray lacks uncertainty
        ImportError: If DimArray is not available (circular import protection)

    Examples:
        >>> from dimtensor import DimArray, units
        >>> # Two lengths with uncertainty
        >>> a = DimArray([1.0], units.m, uncertainty=[0.1])
        >>> b = DimArray([2.0], units.m, uncertainty=[0.2])
        >>> # Propagate through addition
        >>> result = monte_carlo_dimarray(lambda: a + b, [a, b])
        >>> print(f"Result: {result.mean} ± {result.std}")
    """
    # Import here to avoid circular dependency
    from ..core.dimarray import DimArray

    # Validate inputs
    if not all(isinstance(arr, DimArray) for arr in dimarrays):
        raise ValueError("All inputs must be DimArray objects")

    if not all(arr.has_uncertainty for arr in dimarrays):
        raise ValueError("All DimArray objects must have uncertainty")

    # Extract means, stds, and units
    means_list = []
    stds_list = []
    units_list = []

    for arr in dimarrays:
        # For now, only support scalar DimArrays
        if arr.data.size != 1:
            raise ValueError(
                "Monte Carlo for DimArray currently only supports scalar inputs"
            )
        means_list.append(float(arr.data.flat[0]))
        stds_list.append(float(arr.uncertainty.flat[0]))  # type: ignore
        units_list.append(arr.unit)

    means = np.array(means_list)
    stds = np.array(stds_list)

    # Create wrapper function that converts arrays to DimArrays
    def wrapped_func(x: NDArray[np.floating]) -> NDArray[np.floating]:
        # Convert input array to DimArrays
        dim_inputs = [
            DimArray([x[i]], unit=units_list[i]) for i in range(len(dimarrays))
        ]
        # Call user function
        result = func(*dim_inputs)
        # Extract numerical value
        if isinstance(result, DimArray):
            return np.array([float(result.data.flat[0])])
        else:
            return np.atleast_1d(result)

    # Run Monte Carlo
    mc_result = monte_carlo(
        wrapped_func,
        means,
        stds,
        n_samples=n_samples,
        method=method,
        correlation=correlation,
        percentiles=percentiles,
        seed=seed,
        check_convergence=check_convergence,
    )

    # Get output unit from a test evaluation
    test_inputs = [DimArray([means[i]], unit=units_list[i]) for i in range(len(means))]
    test_output = func(*test_inputs)
    if isinstance(test_output, DimArray):
        output_unit = test_output.unit
    else:
        output_unit = None

    # Convert result arrays back to DimArrays if we have a unit
    if output_unit is not None:
        mc_result.mean = DimArray(mc_result.mean, unit=output_unit).data  # type: ignore
        mc_result.std = DimArray(mc_result.std, unit=output_unit).data  # type: ignore

    return mc_result

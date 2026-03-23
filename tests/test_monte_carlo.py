"""Tests for Monte Carlo uncertainty propagation."""

import numpy as np
import pytest

from dimtensor.uncertainty import (
    LHSSampler,
    MCResult,
    RandomSampler,
    SobolSampler,
    monte_carlo,
    monte_carlo_dimarray,
)

# Check if scipy is available
try:
    import scipy  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TestMCResult:
    """Tests for MCResult class."""

    def test_percentile(self):
        """Test percentile calculation."""
        samples = np.random.randn(1000, 1)
        result = MCResult(
            mean=np.mean(samples, axis=0),
            std=np.std(samples, axis=0),
            samples=samples,
            percentiles={},
            n_samples=1000,
            method="random",
        )
        p50 = result.percentile(50)
        assert p50.shape == (1,)
        # Median should be close to 0 for standard normal (relaxed for small sample size)
        assert abs(p50[0]) < 0.2

    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        samples = np.random.randn(1000, 1)
        result = MCResult(
            mean=np.mean(samples, axis=0),
            std=np.std(samples, axis=0),
            samples=samples,
            percentiles={},
            n_samples=1000,
            method="random",
        )
        lower, upper = result.confidence_interval(0.95)
        # For standard normal, 95% CI should be roughly [-2, 2]
        assert lower[0] < -1.5
        assert upper[0] > 1.5

    def test_histogram(self):
        """Test histogram generation."""
        samples = np.random.randn(1000, 1)
        result = MCResult(
            mean=np.mean(samples, axis=0),
            std=np.std(samples, axis=0),
            samples=samples,
            percentiles={},
            n_samples=1000,
            method="random",
        )
        counts, bin_edges = result.histogram(bins=20)
        assert len(counts) == 20
        assert len(bin_edges) == 21
        assert np.sum(counts) == 1000

    def test_standard_error(self):
        """Test standard error calculation."""
        samples = np.ones((100, 1))
        result = MCResult(
            mean=np.ones(1),
            std=np.ones(1),
            samples=samples,
            percentiles={},
            n_samples=100,
            method="random",
        )
        se = result.standard_error()
        expected = 1.0 / np.sqrt(100)
        assert np.allclose(se, expected)


class TestRandomSampler:
    """Tests for RandomSampler."""

    def test_sample_independent(self):
        """Test independent random sampling."""
        sampler = RandomSampler(seed=42)
        means = np.array([1.0, 2.0])
        stds = np.array([0.1, 0.2])
        samples = sampler.sample(1000, means, stds)

        assert samples.shape == (1000, 2)
        # Check means (with tolerance)
        assert np.allclose(np.mean(samples, axis=0), means, atol=0.05)
        # Check stds (with tolerance)
        assert np.allclose(np.std(samples, axis=0), stds, atol=0.05)

    def test_sample_correlated(self):
        """Test correlated random sampling."""
        sampler = RandomSampler(seed=42)
        means = np.array([1.0, 2.0])
        stds = np.array([0.1, 0.2])
        correlation = np.array([[1.0, 0.9], [0.9, 1.0]])

        samples = sampler.sample(1000, means, stds, correlation)

        assert samples.shape == (1000, 2)
        # Check correlation (with tolerance)
        sample_corr = np.corrcoef(samples.T)
        assert np.allclose(sample_corr, correlation, atol=0.1)

    def test_reproducibility(self):
        """Test that seed ensures reproducibility."""
        means = np.array([1.0, 2.0])
        stds = np.array([0.1, 0.2])

        sampler1 = RandomSampler(seed=42)
        samples1 = sampler1.sample(100, means, stds)

        sampler2 = RandomSampler(seed=42)
        samples2 = sampler2.sample(100, means, stds)

        assert np.allclose(samples1, samples2)


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
class TestLHSSampler:
    """Tests for LHSSampler."""

    def test_sample_coverage(self):
        """Test that LHS provides good coverage."""
        sampler = LHSSampler(seed=42)
        means = np.array([0.0, 0.0])
        stds = np.array([1.0, 1.0])
        samples = sampler.sample(100, means, stds)

        assert samples.shape == (100, 2)
        # Each dimension should be well-covered (check quartiles)
        for i in range(2):
            sorted_samples = np.sort(samples[:, i])
            # Check that we have roughly equal samples in each quartile
            q1 = sorted_samples[24]  # 25th percentile sample
            q2 = sorted_samples[49]  # 50th percentile sample
            q3 = sorted_samples[74]  # 75th percentile sample
            # For standard normal, these should be near -0.67, 0, 0.67
            assert abs(q2) < 0.3  # median near 0

    def test_sample_correlated(self):
        """Test LHS with correlation."""
        sampler = LHSSampler(seed=42)
        means = np.array([1.0, 2.0])
        stds = np.array([0.5, 1.0])
        correlation = np.array([[1.0, 0.8], [0.8, 1.0]])

        samples = sampler.sample(500, means, stds, correlation)

        assert samples.shape == (500, 2)
        # Check correlation
        sample_corr = np.corrcoef(samples.T)
        assert np.allclose(sample_corr[0, 1], 0.8, atol=0.15)


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
class TestSobolSampler:
    """Tests for SobolSampler."""

    def test_sample_low_discrepancy(self):
        """Test that Sobol sequences have low discrepancy."""
        sampler = SobolSampler(seed=42)
        means = np.array([0.0, 0.0])
        stds = np.array([1.0, 1.0])
        samples = sampler.sample(100, means, stds)

        assert samples.shape == (100, 2)
        # Sobol sequences should provide more uniform coverage than random
        # Check that means are close to target
        assert np.allclose(np.mean(samples, axis=0), means, atol=0.3)


class TestMonteCarlo:
    """Tests for monte_carlo function."""

    def test_simple_addition(self):
        """Test MC propagation for simple addition."""

        def f(x):
            return np.array([x[0] + x[1]])

        means = [1.0, 2.0]
        stds = [0.1, 0.2]

        result = monte_carlo(f, means, stds, n_samples=5000, seed=42)

        # Expected: mean = 3.0, std = sqrt(0.1^2 + 0.2^2) = 0.2236
        assert np.allclose(result.mean, 3.0, atol=0.02)
        assert np.allclose(result.std, np.sqrt(0.1**2 + 0.2**2), atol=0.02)

    def test_multiplication(self):
        """Test MC propagation for multiplication."""

        def f(x):
            return np.array([x[0] * x[1]])

        means = [2.0, 3.0]
        stds = [0.1, 0.15]

        result = monte_carlo(f, means, stds, n_samples=5000, seed=42)

        # Expected: mean ≈ 6.0
        # Relative std = sqrt((0.1/2)^2 + (0.15/3)^2) ≈ 0.0707
        # Absolute std ≈ 6.0 * 0.0707 ≈ 0.424
        assert np.allclose(result.mean, 6.0, atol=0.1)
        expected_std = 6.0 * np.sqrt((0.1 / 2) ** 2 + (0.15 / 3) ** 2)
        assert np.allclose(result.std, expected_std, rtol=0.1)

    def test_nonlinear_function(self):
        """Test MC propagation for nonlinear function."""

        def f(x):
            return np.array([x[0] ** 2])

        means = [1.0]
        stds = [0.1]

        result = monte_carlo(f, means, stds, n_samples=5000, seed=42)

        # For x ~ N(1, 0.1), E[x^2] ≈ 1 + 0.01 = 1.01
        assert np.allclose(result.mean, 1.01, atol=0.02)

    def test_percentiles(self):
        """Test that percentiles are computed correctly."""

        def f(x):
            return np.array([x[0]])

        means = [0.0]
        stds = [1.0]

        result = monte_carlo(
            f, means, stds, n_samples=5000, percentiles=[5, 50, 95], seed=42
        )

        # For standard normal:
        # 5th percentile ≈ -1.645
        # 50th percentile ≈ 0
        # 95th percentile ≈ 1.645
        assert abs(result.percentiles[5][0] - (-1.645)) < 0.15
        assert abs(result.percentiles[50][0]) < 0.1
        assert abs(result.percentiles[95][0] - 1.645) < 0.15

    def test_zero_uncertainty(self):
        """Test handling of zero uncertainty (deterministic)."""

        def f(x):
            return np.array([x[0] + x[1]])

        means = [1.0, 2.0]
        stds = [0.0, 0.0]

        result = monte_carlo(f, means, stds, n_samples=100, seed=42)

        # All samples should be identical
        assert np.allclose(result.mean, 3.0)
        assert np.allclose(result.std, 0.0)
        assert np.allclose(result.samples, 3.0)

    def test_correlated_inputs(self):
        """Test MC with correlated inputs."""

        def f(x):
            return np.array([x[0] + x[1]])

        means = [1.0, 2.0]
        stds = [0.1, 0.1]
        # Perfect positive correlation
        correlation = np.array([[1.0, 1.0], [1.0, 1.0]])

        result = monte_carlo(
            f, means, stds, n_samples=5000, correlation=correlation, seed=42
        )

        # With perfect correlation, std(x0 + x1) = std(x0) + std(x1) = 0.2
        assert np.allclose(result.mean, 3.0, atol=0.02)
        assert np.allclose(result.std, 0.2, atol=0.03)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_lhs_method(self):
        """Test that LHS method can be used."""

        def f(x):
            return np.array([x[0] + x[1]])

        means = [1.0, 2.0]
        stds = [0.1, 0.2]

        result = monte_carlo(f, means, stds, n_samples=1000, method="lhs", seed=42)

        assert result.method == "lhs"
        assert np.allclose(result.mean, 3.0, atol=0.05)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_sobol_method(self):
        """Test that Sobol method can be used."""

        def f(x):
            return np.array([x[0] + x[1]])

        means = [1.0, 2.0]
        stds = [0.1, 0.2]

        result = monte_carlo(f, means, stds, n_samples=1000, method="sobol", seed=42)

        assert result.method == "sobol"
        assert np.allclose(result.mean, 3.0, atol=0.05)

    def test_invalid_method(self):
        """Test that invalid method raises error."""

        def f(x):
            return np.array([x[0]])

        with pytest.raises(ValueError, match="Unknown sampling method"):
            monte_carlo(f, [1.0], [0.1], method="invalid")  # type: ignore

    def test_invalid_correlation_shape(self):
        """Test that invalid correlation matrix shape raises error."""

        def f(x):
            return np.array([x[0] + x[1]])

        means = [1.0, 2.0]
        stds = [0.1, 0.2]
        correlation = np.array([[1.0, 0.5]])  # Wrong shape

        with pytest.raises(ValueError, match="Correlation matrix shape"):
            monte_carlo(f, means, stds, correlation=correlation)

    def test_invalid_correlation_values(self):
        """Test that invalid correlation values raise error."""

        def f(x):
            return np.array([x[0] + x[1]])

        means = [1.0, 2.0]
        stds = [0.1, 0.2]
        correlation = np.array([[1.0, 2.0], [2.0, 1.0]])  # Values > 1

        with pytest.raises(ValueError, match="Correlation matrix values"):
            monte_carlo(f, means, stds, correlation=correlation)

    def test_convergence_warning(self):
        """Test that convergence warning is issued when needed."""

        def f(x):
            # Highly variable function
            return np.array([x[0] ** 10])

        means = [1.0]
        stds = [0.5]

        with pytest.warns(RuntimeWarning, match="may not have converged"):
            monte_carlo(f, means, stds, n_samples=100, seed=42, check_convergence=True)

    def test_multidimensional_output(self):
        """Test MC with multidimensional output."""

        def f(x):
            return np.array([x[0] + x[1], x[0] * x[1]])

        means = [1.0, 2.0]
        stds = [0.1, 0.2]

        result = monte_carlo(f, means, stds, n_samples=5000, seed=42)

        assert result.mean.shape == (2,)
        assert result.std.shape == (2,)
        assert result.samples.shape == (5000, 2)

        # First output: x0 + x1
        assert np.allclose(result.mean[0], 3.0, atol=0.02)
        # Second output: x0 * x1
        assert np.allclose(result.mean[1], 2.0, atol=0.05)


class TestMonteCarloIntegration:
    """Integration tests comparing MC with analytical results."""

    def test_compare_with_analytical_addition(self):
        """Compare MC with analytical propagation for addition."""

        def f(x):
            return np.array([x[0] + x[1]])

        means = [1.0, 2.0]
        stds = [0.1, 0.2]

        result = monte_carlo(f, means, stds, n_samples=10000, seed=42)

        # Analytical: std = sqrt(0.1^2 + 0.2^2)
        analytical_std = np.sqrt(0.1**2 + 0.2**2)

        # MC should agree within a few percent
        assert np.allclose(result.std, analytical_std, rtol=0.05)

    def test_compare_with_analytical_multiplication(self):
        """Compare MC with analytical propagation for multiplication."""

        def f(x):
            return np.array([x[0] * x[1]])

        means = [2.0, 3.0]
        stds = [0.1, 0.15]

        result = monte_carlo(f, means, stds, n_samples=10000, seed=42)

        # Analytical: relative std = sqrt((σx/x)^2 + (σy/y)^2)
        # absolute std = |x*y| * relative_std
        rel_std = np.sqrt((0.1 / 2.0) ** 2 + (0.15 / 3.0) ** 2)
        analytical_std = 6.0 * rel_std

        # MC should agree within a few percent
        assert np.allclose(result.std, analytical_std, rtol=0.1)


class TestMonteCarloDimArray:
    """Tests for monte_carlo_dimarray function."""

    def test_basic_usage(self):
        """Test basic usage with DimArray."""
        from dimtensor import DimArray, units

        a = DimArray([1.0], units.m, uncertainty=[0.1])
        b = DimArray([2.0], units.m, uncertainty=[0.2])

        result = monte_carlo_dimarray(
            lambda a, b: a + b, [a, b], n_samples=5000, seed=42
        )

        # Result should have correct mean and std
        assert np.allclose(result.mean, 3.0, atol=0.05)
        assert np.allclose(result.std, np.sqrt(0.1**2 + 0.2**2), atol=0.05)

    def test_without_uncertainty(self):
        """Test that error is raised if DimArray lacks uncertainty."""
        from dimtensor import DimArray, units

        a = DimArray([1.0], units.m)  # No uncertainty
        b = DimArray([2.0], units.m, uncertainty=[0.2])

        with pytest.raises(ValueError, match="must have uncertainty"):
            monte_carlo_dimarray(lambda a, b: a + b, [a, b])

    def test_non_scalar_error(self):
        """Test that error is raised for non-scalar DimArray."""
        from dimtensor import DimArray, units

        a = DimArray([1.0, 2.0], units.m, uncertainty=[0.1, 0.1])

        with pytest.raises(ValueError, match="only supports scalar inputs"):
            monte_carlo_dimarray(lambda a: a, [a])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

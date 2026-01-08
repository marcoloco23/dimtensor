"""Tests for benchmark module."""

import pytest
from dimtensor.benchmarks import (
    BenchmarkResult,
    time_operation,
    benchmark_creation,
    benchmark_addition,
    benchmark_multiplication,
    benchmark_division,
    benchmark_power,
    benchmark_reduction,
    benchmark_indexing,
    benchmark_chained_ops,
    quick_benchmark,
)


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_overhead_calculation(self):
        """Test overhead calculation."""
        result = BenchmarkResult("test", 0.001, 0.002, 1000, 100)
        assert result.overhead == pytest.approx(2.0)
        assert result.overhead_percent == pytest.approx(100.0)

    def test_overhead_zero_numpy(self):
        """Test overhead with zero numpy time."""
        result = BenchmarkResult("test", 0.0, 0.001, 1000, 100)
        assert result.overhead == float('inf')


class TestTimeOperation:
    """Tests for time_operation function."""

    def test_times_simple_operation(self):
        """Time a simple operation."""
        def simple():
            x = 1 + 1
            return x

        time_taken = time_operation(simple, iterations=100)
        assert time_taken > 0
        assert time_taken < 0.01  # Should be very fast


class TestBenchmarks:
    """Tests for individual benchmarks."""

    def test_benchmark_creation(self):
        """Benchmark creation returns valid result."""
        result = benchmark_creation(size=100, iterations=10)
        assert result.name == "creation"
        assert result.numpy_time > 0
        assert result.dimarray_time > 0
        assert result.overhead > 0

    def test_benchmark_addition(self):
        """Benchmark addition returns valid result."""
        result = benchmark_addition(size=100, iterations=10)
        assert result.name == "addition"
        assert result.overhead > 0

    def test_benchmark_multiplication(self):
        """Benchmark multiplication returns valid result."""
        result = benchmark_multiplication(size=100, iterations=10)
        assert result.name == "multiplication"
        assert result.overhead > 0

    def test_benchmark_division(self):
        """Benchmark division returns valid result."""
        result = benchmark_division(size=100, iterations=10)
        assert result.name == "division"
        assert result.overhead > 0

    def test_benchmark_power(self):
        """Benchmark power returns valid result."""
        result = benchmark_power(size=100, iterations=10)
        assert result.name == "power"
        assert result.overhead > 0

    def test_benchmark_reduction(self):
        """Benchmark reduction returns valid result."""
        result = benchmark_reduction(size=100, iterations=10)
        assert result.name == "sum"
        assert result.overhead > 0

    def test_benchmark_indexing(self):
        """Benchmark indexing returns valid result."""
        result = benchmark_indexing(size=100, iterations=10)
        assert result.name == "indexing"
        assert result.overhead > 0

    def test_benchmark_chained_ops(self):
        """Benchmark chained operations returns valid result."""
        result = benchmark_chained_ops(size=100, iterations=10)
        assert result.name == "chained_ops"
        assert result.overhead > 0


class TestQuickBenchmark:
    """Tests for quick_benchmark function."""

    def test_returns_all_operations(self):
        """Quick benchmark returns all operation types."""
        results = quick_benchmark()
        assert "creation" in results
        assert "addition" in results
        assert "multiplication" in results
        assert "division" in results
        assert "power" in results
        assert "sum" in results

    def test_overhead_reasonable(self):
        """Overhead should be reasonable (<100x for most operations)."""
        results = quick_benchmark()
        for op, overhead in results.items():
            # Most operations should have <50x overhead
            # Creation is typically higher due to unit handling
            if op != "creation":
                assert overhead < 100, f"{op} has excessive overhead: {overhead}x"

"""Performance benchmarks for dimtensor.

This module provides utilities for benchmarking dimtensor operations
against raw numpy arrays to measure the overhead of unit tracking.

Example:
    >>> from dimtensor.benchmarks import benchmark_suite, print_results
    >>> results = benchmark_suite()
    >>> print_results(results)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .core.dimarray import DimArray
from .core.units import m, s, kg, dimensionless


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    numpy_time: float
    dimarray_time: float
    array_size: int
    iterations: int

    @property
    def overhead(self) -> float:
        """Overhead factor (dimarray_time / numpy_time)."""
        if self.numpy_time == 0:
            return float('inf')
        return self.dimarray_time / self.numpy_time

    @property
    def overhead_percent(self) -> float:
        """Overhead as percentage."""
        return (self.overhead - 1) * 100


def time_operation(func: Callable[[], None], iterations: int = 1000) -> float:
    """Time an operation over multiple iterations.

    Args:
        func: Zero-argument function to time.
        iterations: Number of iterations.

    Returns:
        Average time per iteration in seconds.
    """
    # Warmup
    for _ in range(min(10, iterations)):
        func()

    start = time.perf_counter()
    for _ in range(iterations):
        func()
    end = time.perf_counter()

    return (end - start) / iterations


def benchmark_creation(size: int = 10000, iterations: int = 1000) -> BenchmarkResult:
    """Benchmark array creation."""
    data = list(range(size))

    def numpy_create() -> None:
        np.array(data)

    def dimarray_create() -> None:
        DimArray(data, m)

    numpy_time = time_operation(numpy_create, iterations)
    dimarray_time = time_operation(dimarray_create, iterations)

    return BenchmarkResult("creation", numpy_time, dimarray_time, size, iterations)


def benchmark_addition(size: int = 10000, iterations: int = 1000) -> BenchmarkResult:
    """Benchmark element-wise addition."""
    np_a = np.random.randn(size)
    np_b = np.random.randn(size)
    da_a = DimArray(np_a, m)
    da_b = DimArray(np_b, m)

    def numpy_add() -> None:
        np_a + np_b

    def dimarray_add() -> None:
        da_a + da_b

    numpy_time = time_operation(numpy_add, iterations)
    dimarray_time = time_operation(dimarray_add, iterations)

    return BenchmarkResult("addition", numpy_time, dimarray_time, size, iterations)


def benchmark_multiplication(size: int = 10000, iterations: int = 1000) -> BenchmarkResult:
    """Benchmark element-wise multiplication."""
    np_a = np.random.randn(size)
    np_b = np.random.randn(size)
    da_a = DimArray(np_a, m)
    da_b = DimArray(np_b, s)

    def numpy_mul() -> None:
        np_a * np_b

    def dimarray_mul() -> None:
        da_a * da_b

    numpy_time = time_operation(numpy_mul, iterations)
    dimarray_time = time_operation(dimarray_mul, iterations)

    return BenchmarkResult("multiplication", numpy_time, dimarray_time, size, iterations)


def benchmark_division(size: int = 10000, iterations: int = 1000) -> BenchmarkResult:
    """Benchmark element-wise division."""
    np_a = np.random.randn(size)
    np_b = np.random.randn(size) + 1  # Avoid division by zero
    da_a = DimArray(np_a, m)
    da_b = DimArray(np_b, s)

    def numpy_div() -> None:
        np_a / np_b

    def dimarray_div() -> None:
        da_a / da_b

    numpy_time = time_operation(numpy_div, iterations)
    dimarray_time = time_operation(dimarray_div, iterations)

    return BenchmarkResult("division", numpy_time, dimarray_time, size, iterations)


def benchmark_power(size: int = 10000, iterations: int = 1000) -> BenchmarkResult:
    """Benchmark power operation."""
    np_a = np.abs(np.random.randn(size)) + 0.1
    da_a = DimArray(np_a, m)

    def numpy_pow() -> None:
        np_a ** 2

    def dimarray_pow() -> None:
        da_a ** 2

    numpy_time = time_operation(numpy_pow, iterations)
    dimarray_time = time_operation(dimarray_pow, iterations)

    return BenchmarkResult("power", numpy_time, dimarray_time, size, iterations)


def benchmark_reduction(size: int = 10000, iterations: int = 1000) -> BenchmarkResult:
    """Benchmark reduction operations (sum)."""
    np_a = np.random.randn(size)
    da_a = DimArray(np_a, m)

    def numpy_sum() -> None:
        np_a.sum()

    def dimarray_sum() -> None:
        da_a.sum()

    numpy_time = time_operation(numpy_sum, iterations)
    dimarray_time = time_operation(dimarray_sum, iterations)

    return BenchmarkResult("sum", numpy_time, dimarray_time, size, iterations)


def benchmark_matmul(size: int = 100, iterations: int = 100) -> BenchmarkResult:
    """Benchmark matrix multiplication."""
    np_a = np.random.randn(size, size)
    np_b = np.random.randn(size, size)
    da_a = DimArray(np_a, m)
    da_b = DimArray(np_b, s)

    def numpy_matmul() -> None:
        np_a @ np_b

    def dimarray_matmul() -> None:
        # Use functions module
        from .functions import matmul
        matmul(da_a, da_b)

    numpy_time = time_operation(numpy_matmul, iterations)
    dimarray_time = time_operation(dimarray_matmul, iterations)

    return BenchmarkResult("matmul", numpy_time, dimarray_time, size * size, iterations)


def benchmark_indexing(size: int = 10000, iterations: int = 1000) -> BenchmarkResult:
    """Benchmark array indexing."""
    np_a = np.random.randn(size)
    da_a = DimArray(np_a, m)
    indices = np.random.randint(0, size, 100)

    def numpy_index() -> None:
        np_a[indices]

    def dimarray_index() -> None:
        da_a[indices]

    numpy_time = time_operation(numpy_index, iterations)
    dimarray_time = time_operation(dimarray_index, iterations)

    return BenchmarkResult("indexing", numpy_time, dimarray_time, size, iterations)


def benchmark_chained_ops(size: int = 10000, iterations: int = 1000) -> BenchmarkResult:
    """Benchmark chained operations."""
    np_a = np.random.randn(size)
    np_b = np.random.randn(size)
    np_c = np.random.randn(size) + 1
    da_a = DimArray(np_a, m)
    da_b = DimArray(np_b, m)
    da_c = DimArray(np_c, s)

    def numpy_chain() -> None:
        ((np_a + np_b) * 2.0) / np_c

    def dimarray_chain() -> None:
        ((da_a + da_b) * 2.0) / da_c

    numpy_time = time_operation(numpy_chain, iterations)
    dimarray_time = time_operation(dimarray_chain, iterations)

    return BenchmarkResult("chained_ops", numpy_time, dimarray_time, size, iterations)


def benchmark_suite(
    sizes: list[int] | None = None,
    iterations: int = 1000,
) -> list[BenchmarkResult]:
    """Run the full benchmark suite.

    Args:
        sizes: List of array sizes to test. Defaults to [1000, 10000, 100000].
        iterations: Number of iterations per benchmark.

    Returns:
        List of benchmark results.
    """
    if sizes is None:
        sizes = [1000, 10000, 100000]

    results = []

    for size in sizes:
        # Adjust iterations for larger arrays
        iters = iterations if size <= 10000 else iterations // 10
        matmul_size = min(size, 100)  # Cap matmul size

        results.append(benchmark_creation(size, iters))
        results.append(benchmark_addition(size, iters))
        results.append(benchmark_multiplication(size, iters))
        results.append(benchmark_division(size, iters))
        results.append(benchmark_power(size, iters))
        results.append(benchmark_reduction(size, iters))
        results.append(benchmark_matmul(matmul_size, iters))
        results.append(benchmark_indexing(size, iters))
        results.append(benchmark_chained_ops(size, iters))

    return results


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a formatted table.

    Args:
        results: List of benchmark results.
    """
    print(f"{'Operation':<15} {'Size':>10} {'NumPy (us)':>12} {'DimArray (us)':>14} {'Overhead':>10}")
    print("-" * 65)

    for r in results:
        numpy_us = r.numpy_time * 1_000_000
        dimarray_us = r.dimarray_time * 1_000_000
        overhead_str = f"{r.overhead:.2f}x"
        print(f"{r.name:<15} {r.array_size:>10} {numpy_us:>12.2f} {dimarray_us:>14.2f} {overhead_str:>10}")


def quick_benchmark() -> dict[str, float]:
    """Run a quick benchmark and return overhead factors.

    Returns:
        Dictionary mapping operation names to overhead factors.
    """
    size = 10000
    iterations = 100

    return {
        "creation": benchmark_creation(size, iterations).overhead,
        "addition": benchmark_addition(size, iterations).overhead,
        "multiplication": benchmark_multiplication(size, iterations).overhead,
        "division": benchmark_division(size, iterations).overhead,
        "power": benchmark_power(size, iterations).overhead,
        "sum": benchmark_reduction(size, iterations).overhead,
    }

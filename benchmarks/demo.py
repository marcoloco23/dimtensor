#!/usr/bin/env python
"""Demo script showing how to use the benchmark suite.

This script demonstrates:
1. Running individual benchmarks
2. Getting timing results
3. Comparing overhead
"""

import sys
sys.path.insert(0, '.')

from suites.numpy_ops import ArrayCreation, ArithmeticOps, ReductionOps

def demo_array_creation():
    """Demo array creation benchmarks."""
    print("=" * 70)
    print("Array Creation Benchmarks")
    print("=" * 70)

    bench = ArrayCreation()

    for size in [100, 10_000, 1_000_000]:
        bench.setup(size)

        # Time numpy creation
        import time
        start = time.perf_counter()
        for _ in range(100):
            bench.time_from_list_numpy(size)
        numpy_time = (time.perf_counter() - start) / 100

        # Time dimarray creation
        start = time.perf_counter()
        for _ in range(100):
            bench.time_from_list_dimarray(size)
        dimarray_time = (time.perf_counter() - start) / 100

        overhead = dimarray_time / numpy_time

        print(f"\nSize: {size:>10,} elements")
        print(f"  NumPy:    {numpy_time*1e6:>8.2f} μs")
        print(f"  DimArray: {dimarray_time*1e6:>8.2f} μs")
        print(f"  Overhead: {overhead:>8.2f}x")


def demo_arithmetic():
    """Demo arithmetic operation benchmarks."""
    print("\n" + "=" * 70)
    print("Arithmetic Operations Benchmarks")
    print("=" * 70)

    bench = ArithmeticOps()

    for size in [100, 10_000, 1_000_000]:
        bench.setup(size)

        # Test addition
        import time
        start = time.perf_counter()
        for _ in range(100):
            bench.time_add_numpy(size)
        numpy_time = (time.perf_counter() - start) / 100

        start = time.perf_counter()
        for _ in range(100):
            bench.time_add_dimarray(size)
        dimarray_time = (time.perf_counter() - start) / 100

        overhead = dimarray_time / numpy_time

        print(f"\nSize: {size:>10,} elements (addition)")
        print(f"  NumPy:    {numpy_time*1e6:>8.2f} μs")
        print(f"  DimArray: {dimarray_time*1e6:>8.2f} μs")
        print(f"  Overhead: {overhead:>8.2f}x")


def demo_reductions():
    """Demo reduction operation benchmarks."""
    print("\n" + "=" * 70)
    print("Reduction Operations Benchmarks")
    print("=" * 70)

    bench = ReductionOps()

    for size in [100, 10_000, 1_000_000]:
        bench.setup(size)

        # Test sum
        import time
        start = time.perf_counter()
        for _ in range(100):
            bench.time_sum_numpy(size)
        numpy_time = (time.perf_counter() - start) / 100

        start = time.perf_counter()
        for _ in range(100):
            bench.time_sum_dimarray(size)
        dimarray_time = (time.perf_counter() - start) / 100

        overhead = dimarray_time / numpy_time

        print(f"\nSize: {size:>10,} elements (sum)")
        print(f"  NumPy:    {numpy_time*1e6:>8.2f} μs")
        print(f"  DimArray: {dimarray_time*1e6:>8.2f} μs")
        print(f"  Overhead: {overhead:>8.2f}x")


if __name__ == "__main__":
    print("\nDimtensor Benchmark Suite Demo")
    print("This demonstrates the new ASV benchmark infrastructure\n")

    demo_array_creation()
    demo_arithmetic()
    demo_reductions()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nThe benchmark suite includes:")
    print("  - NumPy operations (creation, arithmetic, reductions, etc.)")
    print("  - PyTorch operations (CPU, GPU, autograd)")
    print("  - Competitor comparisons (pint, astropy, unyt)")
    print("\nTo run full benchmarks:")
    print("  asv run --python=same")
    print("\nTo generate HTML reports:")
    print("  asv publish")
    print("  asv preview")
    print()

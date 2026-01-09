"""Performance benchmarks for DimTensor CUDA operations.

This module provides utilities for benchmarking DimTensor operations
against raw PyTorch tensors to measure the overhead of unit tracking
on GPU operations.

Example:
    >>> from dimtensor.torch.benchmarks import benchmark_cuda_suite, print_cuda_results
    >>> import torch
    >>> if torch.cuda.is_available():
    ...     results = benchmark_cuda_suite()
    ...     print_cuda_results(results)
"""

from __future__ import annotations

import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Literal

import torch
from torch import Tensor

from ..core.units import dimensionless, kg, m, s
from .dimtensor import DimTensor


@dataclass
class CudaBenchmarkResult:
    """Result of a CUDA benchmark run."""

    name: str
    pytorch_time: float  # GPU time in seconds
    dimtensor_time: float  # GPU time in seconds
    tensor_size: int
    device: str
    dtype: str
    iterations: int
    mode: Literal["forward", "backward", "mixed"] = "forward"

    @property
    def overhead(self) -> float:
        """Overhead factor (dimtensor_time / pytorch_time)."""
        if self.pytorch_time == 0:
            return float("inf")
        return self.dimtensor_time / self.pytorch_time

    @property
    def overhead_percent(self) -> float:
        """Overhead as percentage."""
        return (self.overhead - 1) * 100


def cuda_available() -> bool:
    """Check if CUDA is available."""
    return bool(torch.cuda.is_available())


def get_device_info() -> dict[str, str]:
    """Get CUDA device information.

    Returns:
        Dictionary with device name, CUDA version, PyTorch version.
    """
    info = {"pytorch_version": torch.__version__}

    if cuda_available():
        info["cuda_available"] = "True"
        info["cuda_version"] = torch.version.cuda or "unknown"
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = str(torch.cuda.device_count())
    else:
        info["cuda_available"] = "False"

    return info


@contextmanager
def cuda_timer(device: str = "cuda") -> Iterator[list[float]]:
    """Context manager for precise CUDA timing using Events.

    Args:
        device: Device to time on ('cuda' or 'cpu').

    Yields:
        List that will contain elapsed time in seconds.

    Example:
        >>> with cuda_timer() as elapsed:
        ...     result = tensor1 + tensor2
        >>> print(f"Operation took {elapsed[0]:.6f} seconds")
    """
    elapsed = [0.0]

    if device == "cuda" and cuda_available():
        # CUDA Events for precise GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Synchronize before starting
        torch.cuda.synchronize()
        start_event.record()

        yield elapsed

        end_event.record()
        torch.cuda.synchronize()  # Wait for completion
        elapsed[0] = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to s
    else:
        # CPU timing fallback
        start = time.perf_counter()
        yield elapsed
        end = time.perf_counter()
        elapsed[0] = end - start


def cuda_time_operation(
    func: Callable[[], None],
    iterations: int = 100,
    warmup: int = 10,
    device: str = "cuda",
) -> float:
    """Time a GPU operation over multiple iterations.

    Args:
        func: Zero-argument function to time.
        iterations: Number of iterations to average over.
        warmup: Number of warmup iterations (discarded).
        device: Device to run on ('cuda' or 'cpu').

    Returns:
        Average time per iteration in seconds.
    """
    # Warmup iterations to initialize CUDA context
    for _ in range(warmup):
        func()

    if device == "cuda" and cuda_available():
        torch.cuda.synchronize()

    total_time = 0.0

    for _ in range(iterations):
        with cuda_timer(device) as elapsed:
            func()
        total_time += elapsed[0]

    return total_time / iterations


# =============================================================================
# Individual Benchmarks
# =============================================================================


def benchmark_creation_cuda(
    size: int = 10000,
    iterations: int = 100,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> CudaBenchmarkResult:
    """Benchmark tensor creation on GPU.

    Args:
        size: Number of elements in tensor.
        iterations: Number of iterations.
        device: Device to use ('cuda' or 'cpu').
        dtype: Data type for tensors.

    Returns:
        Benchmark result.
    """
    data = torch.randn(size, dtype=dtype, device="cpu")

    def pytorch_create() -> None:
        data.to(device)

    def dimtensor_create() -> None:
        DimTensor(data, m, device=device)

    pytorch_time = cuda_time_operation(pytorch_create, iterations, device=device)
    dimtensor_time = cuda_time_operation(dimtensor_create, iterations, device=device)

    return CudaBenchmarkResult(
        name="creation",
        pytorch_time=pytorch_time,
        dimtensor_time=dimtensor_time,
        tensor_size=size,
        device=device,
        dtype=str(dtype),
        iterations=iterations,
    )


def benchmark_device_transfer(
    size: int = 10000,
    iterations: int = 100,
    direction: Literal["cpu_to_cuda", "cuda_to_cpu"] = "cpu_to_cuda",
    dtype: torch.dtype = torch.float32,
) -> CudaBenchmarkResult:
    """Benchmark device transfer overhead.

    Args:
        size: Number of elements in tensor.
        iterations: Number of iterations.
        direction: Transfer direction.
        dtype: Data type for tensors.

    Returns:
        Benchmark result.
    """
    if not cuda_available():
        warnings.warn("CUDA not available, skipping device transfer benchmark")
        return CudaBenchmarkResult(
            name=f"transfer_{direction}",
            pytorch_time=0.0,
            dimtensor_time=0.0,
            tensor_size=size,
            device="cpu",
            dtype=str(dtype),
            iterations=iterations,
        )

    if direction == "cpu_to_cuda":
        pt_tensor = torch.randn(size, dtype=dtype, device="cpu")
        dt_tensor = DimTensor(torch.randn(size, dtype=dtype, device="cpu"), m)

        def pytorch_transfer() -> None:
            pt_tensor.cuda()

        def dimtensor_transfer() -> None:
            dt_tensor.cuda()

    else:  # cuda_to_cpu
        pt_tensor = torch.randn(size, dtype=dtype, device="cuda")
        dt_tensor = DimTensor(torch.randn(size, dtype=dtype, device="cuda"), m)

        def pytorch_transfer() -> None:
            pt_tensor.cpu()

        def dimtensor_transfer() -> None:
            dt_tensor.cpu()

    pytorch_time = cuda_time_operation(pytorch_transfer, iterations, device="cuda")
    dimtensor_time = cuda_time_operation(dimtensor_transfer, iterations, device="cuda")

    return CudaBenchmarkResult(
        name=f"transfer_{direction}",
        pytorch_time=pytorch_time,
        dimtensor_time=dimtensor_time,
        tensor_size=size,
        device="cuda",
        dtype=str(dtype),
        iterations=iterations,
    )


def benchmark_addition_cuda(
    size: int = 10000,
    iterations: int = 100,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> CudaBenchmarkResult:
    """Benchmark element-wise addition on GPU.

    Args:
        size: Number of elements in tensor.
        iterations: Number of iterations.
        device: Device to use.
        dtype: Data type for tensors.

    Returns:
        Benchmark result.
    """
    pt_a = torch.randn(size, dtype=dtype, device=device)
    pt_b = torch.randn(size, dtype=dtype, device=device)
    dt_a = DimTensor(torch.randn(size, dtype=dtype, device=device), m)
    dt_b = DimTensor(torch.randn(size, dtype=dtype, device=device), m)

    def pytorch_add() -> None:
        pt_a + pt_b

    def dimtensor_add() -> None:
        dt_a + dt_b

    pytorch_time = cuda_time_operation(pytorch_add, iterations, device=device)
    dimtensor_time = cuda_time_operation(dimtensor_add, iterations, device=device)

    return CudaBenchmarkResult(
        name="addition",
        pytorch_time=pytorch_time,
        dimtensor_time=dimtensor_time,
        tensor_size=size,
        device=device,
        dtype=str(dtype),
        iterations=iterations,
    )


def benchmark_multiplication_cuda(
    size: int = 10000,
    iterations: int = 100,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> CudaBenchmarkResult:
    """Benchmark element-wise multiplication on GPU.

    Args:
        size: Number of elements in tensor.
        iterations: Number of iterations.
        device: Device to use.
        dtype: Data type for tensors.

    Returns:
        Benchmark result.
    """
    pt_a = torch.randn(size, dtype=dtype, device=device)
    pt_b = torch.randn(size, dtype=dtype, device=device)
    dt_a = DimTensor(torch.randn(size, dtype=dtype, device=device), m)
    dt_b = DimTensor(torch.randn(size, dtype=dtype, device=device), s)

    def pytorch_mul() -> None:
        pt_a * pt_b

    def dimtensor_mul() -> None:
        dt_a * dt_b

    pytorch_time = cuda_time_operation(pytorch_mul, iterations, device=device)
    dimtensor_time = cuda_time_operation(dimtensor_mul, iterations, device=device)

    return CudaBenchmarkResult(
        name="multiplication",
        pytorch_time=pytorch_time,
        dimtensor_time=dimtensor_time,
        tensor_size=size,
        device=device,
        dtype=str(dtype),
        iterations=iterations,
    )


def benchmark_division_cuda(
    size: int = 10000,
    iterations: int = 100,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> CudaBenchmarkResult:
    """Benchmark element-wise division on GPU.

    Args:
        size: Number of elements in tensor.
        iterations: Number of iterations.
        device: Device to use.
        dtype: Data type for tensors.

    Returns:
        Benchmark result.
    """
    pt_a = torch.randn(size, dtype=dtype, device=device)
    pt_b = torch.randn(size, dtype=dtype, device=device) + 1.0
    dt_a = DimTensor(torch.randn(size, dtype=dtype, device=device), m)
    dt_b = DimTensor(torch.randn(size, dtype=dtype, device=device) + 1.0, s)

    def pytorch_div() -> None:
        pt_a / pt_b

    def dimtensor_div() -> None:
        dt_a / dt_b

    pytorch_time = cuda_time_operation(pytorch_div, iterations, device=device)
    dimtensor_time = cuda_time_operation(dimtensor_div, iterations, device=device)

    return CudaBenchmarkResult(
        name="division",
        pytorch_time=pytorch_time,
        dimtensor_time=dimtensor_time,
        tensor_size=size,
        device=device,
        dtype=str(dtype),
        iterations=iterations,
    )


def benchmark_matmul_cuda(
    size: int = 128,
    iterations: int = 100,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> CudaBenchmarkResult:
    """Benchmark matrix multiplication on GPU.

    Args:
        size: Matrix dimension (size x size).
        iterations: Number of iterations.
        device: Device to use.
        dtype: Data type for tensors.

    Returns:
        Benchmark result.
    """
    pt_a = torch.randn(size, size, dtype=dtype, device=device)
    pt_b = torch.randn(size, size, dtype=dtype, device=device)
    dt_a = DimTensor(torch.randn(size, size, dtype=dtype, device=device), m)
    dt_b = DimTensor(torch.randn(size, size, dtype=dtype, device=device), s)

    def pytorch_matmul() -> None:
        torch.matmul(pt_a, pt_b)

    def dimtensor_matmul() -> None:
        dt_a.matmul(dt_b)

    pytorch_time = cuda_time_operation(pytorch_matmul, iterations, device=device)
    dimtensor_time = cuda_time_operation(dimtensor_matmul, iterations, device=device)

    return CudaBenchmarkResult(
        name="matmul",
        pytorch_time=pytorch_time,
        dimtensor_time=dimtensor_time,
        tensor_size=size * size,
        device=device,
        dtype=str(dtype),
        iterations=iterations,
    )


def benchmark_reduction_cuda(
    size: int = 10000,
    iterations: int = 100,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> CudaBenchmarkResult:
    """Benchmark reduction operations (sum) on GPU.

    Args:
        size: Number of elements in tensor.
        iterations: Number of iterations.
        device: Device to use.
        dtype: Data type for tensors.

    Returns:
        Benchmark result.
    """
    pt_a = torch.randn(size, dtype=dtype, device=device)
    dt_a = DimTensor(torch.randn(size, dtype=dtype, device=device), m)

    def pytorch_sum() -> None:
        pt_a.sum()

    def dimtensor_sum() -> None:
        dt_a.sum()

    pytorch_time = cuda_time_operation(pytorch_sum, iterations, device=device)
    dimtensor_time = cuda_time_operation(dimtensor_sum, iterations, device=device)

    return CudaBenchmarkResult(
        name="sum",
        pytorch_time=pytorch_time,
        dimtensor_time=dimtensor_time,
        tensor_size=size,
        device=device,
        dtype=str(dtype),
        iterations=iterations,
    )


def benchmark_autograd_forward(
    size: int = 10000,
    iterations: int = 100,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> CudaBenchmarkResult:
    """Benchmark forward pass with autograd enabled.

    Args:
        size: Number of elements in tensor.
        iterations: Number of iterations.
        device: Device to use.
        dtype: Data type for tensors.

    Returns:
        Benchmark result.
    """
    pt_a = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
    pt_b = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
    dt_a = DimTensor(
        torch.randn(size, dtype=dtype, device=device), m, requires_grad=True
    )
    dt_b = DimTensor(
        torch.randn(size, dtype=dtype, device=device), s, requires_grad=True
    )

    def pytorch_forward() -> None:
        result = (pt_a * pt_b).sum()
        # Don't call backward to avoid accumulating gradients

    def dimtensor_forward() -> None:
        result = (dt_a * dt_b).sum()

    pytorch_time = cuda_time_operation(pytorch_forward, iterations, device=device)
    dimtensor_time = cuda_time_operation(dimtensor_forward, iterations, device=device)

    return CudaBenchmarkResult(
        name="autograd_forward",
        pytorch_time=pytorch_time,
        dimtensor_time=dimtensor_time,
        tensor_size=size,
        device=device,
        dtype=str(dtype),
        iterations=iterations,
        mode="forward",
    )


def benchmark_autograd_backward(
    size: int = 10000,
    iterations: int = 100,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> CudaBenchmarkResult:
    """Benchmark backward pass with autograd.

    Args:
        size: Number of elements in tensor.
        iterations: Number of iterations.
        device: Device to use.
        dtype: Data type for tensors.

    Returns:
        Benchmark result.
    """

    def pytorch_backward() -> None:
        pt_a = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        pt_b = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        result = (pt_a * pt_b).sum()
        result.backward()

    def dimtensor_backward() -> None:
        dt_a = DimTensor(
            torch.randn(size, dtype=dtype, device=device), m, requires_grad=True
        )
        dt_b = DimTensor(
            torch.randn(size, dtype=dtype, device=device), s, requires_grad=True
        )
        result = (dt_a * dt_b).sum()
        result.backward()

    pytorch_time = cuda_time_operation(pytorch_backward, iterations, device=device)
    dimtensor_time = cuda_time_operation(dimtensor_backward, iterations, device=device)

    return CudaBenchmarkResult(
        name="autograd_backward",
        pytorch_time=pytorch_time,
        dimtensor_time=dimtensor_time,
        tensor_size=size,
        device=device,
        dtype=str(dtype),
        iterations=iterations,
        mode="backward",
    )


def benchmark_chained_ops_cuda(
    size: int = 10000,
    iterations: int = 100,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> CudaBenchmarkResult:
    """Benchmark chained operations on GPU.

    Args:
        size: Number of elements in tensor.
        iterations: Number of iterations.
        device: Device to use.
        dtype: Data type for tensors.

    Returns:
        Benchmark result.
    """
    pt_a = torch.randn(size, dtype=dtype, device=device)
    pt_b = torch.randn(size, dtype=dtype, device=device)
    pt_c = torch.randn(size, dtype=dtype, device=device) + 1.0
    dt_a = DimTensor(torch.randn(size, dtype=dtype, device=device), m)
    dt_b = DimTensor(torch.randn(size, dtype=dtype, device=device), m)
    dt_c = DimTensor(torch.randn(size, dtype=dtype, device=device) + 1.0, s)

    def pytorch_chain() -> None:
        ((pt_a + pt_b) * 2.0) / pt_c

    def dimtensor_chain() -> None:
        ((dt_a + dt_b) * 2.0) / dt_c

    pytorch_time = cuda_time_operation(pytorch_chain, iterations, device=device)
    dimtensor_time = cuda_time_operation(dimtensor_chain, iterations, device=device)

    return CudaBenchmarkResult(
        name="chained_ops",
        pytorch_time=pytorch_time,
        dimtensor_time=dimtensor_time,
        tensor_size=size,
        device=device,
        dtype=str(dtype),
        iterations=iterations,
        mode="mixed",
    )


# =============================================================================
# Benchmark Suite
# =============================================================================


def benchmark_cuda_suite(
    sizes: list[int] | None = None,
    iterations: int = 100,
    device: str = "cuda",
    dtypes: list[torch.dtype] | None = None,
    include_transfers: bool = True,
    include_autograd: bool = True,
) -> list[CudaBenchmarkResult]:
    """Run the full CUDA benchmark suite.

    Args:
        sizes: List of tensor sizes to test.
        iterations: Number of iterations per benchmark.
        device: Device to use ('cuda' or 'cpu').
        dtypes: List of data types to test.
        include_transfers: Whether to benchmark device transfers.
        include_autograd: Whether to benchmark autograd operations.

    Returns:
        List of benchmark results.
    """
    if not cuda_available() and device == "cuda":
        warnings.warn(
            "CUDA not available, running CPU benchmarks instead. "
            "Results may not be representative of GPU performance."
        )
        device = "cpu"

    if sizes is None:
        sizes = [1000, 10000, 100000]

    if dtypes is None:
        dtypes = [torch.float32]

    results = []

    for size in sizes:
        for dtype in dtypes:
            # Adjust iterations for larger arrays
            iters = iterations if size <= 10000 else iterations // 10
            matmul_size = min(int(size**0.5), 128)  # Cap matmul size

            results.append(benchmark_creation_cuda(size, iters, device, dtype))
            results.append(benchmark_addition_cuda(size, iters, device, dtype))
            results.append(benchmark_multiplication_cuda(size, iters, device, dtype))
            results.append(benchmark_division_cuda(size, iters, device, dtype))
            results.append(benchmark_matmul_cuda(matmul_size, iters, device, dtype))
            results.append(benchmark_reduction_cuda(size, iters, device, dtype))
            results.append(benchmark_chained_ops_cuda(size, iters, device, dtype))

            if include_autograd:
                results.append(
                    benchmark_autograd_forward(size, iters // 2, device, dtype)
                )
                results.append(
                    benchmark_autograd_backward(size, iters // 2, device, dtype)
                )

    # Device transfers (only for CUDA)
    if include_transfers and cuda_available():
        for size in sizes:
            for dtype in dtypes:
                iters = iterations if size <= 10000 else iterations // 10
                results.append(
                    benchmark_device_transfer(size, iters, "cpu_to_cuda", dtype)
                )
                results.append(
                    benchmark_device_transfer(size, iters, "cuda_to_cpu", dtype)
                )

    return results


# =============================================================================
# Profiler Integration
# =============================================================================


def profile_dimtensor(
    workload: Callable[[], None],
    name: str = "dimtensor_workload",
    use_cuda: bool = True,
    record_shapes: bool = True,
    with_stack: bool = False,
) -> None:
    """Profile a DimTensor workload using torch.profiler.

    Args:
        workload: Function containing the workload to profile.
        name: Name for the profiling trace.
        use_cuda: Whether to profile CUDA operations.
        record_shapes: Whether to record tensor shapes.
        with_stack: Whether to record Python call stack (slower).

    Example:
        >>> def my_workload():
        ...     a = DimTensor(torch.randn(1000, device='cuda'), units.m)
        ...     b = DimTensor(torch.randn(1000, device='cuda'), units.s)
        ...     c = a * b
        ...     c.sum()
        >>> profile_dimtensor(my_workload)
    """
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if use_cuda and cuda_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=record_shapes,
        with_stack=with_stack,
    ) as prof:
        workload()

    print(f"\n=== Profiler Results for {name} ===")
    print(prof.key_averages().table(sort_by="cuda_time_total" if use_cuda else "cpu_time_total", row_limit=20))


# =============================================================================
# Results Display
# =============================================================================


def print_cuda_results(results: list[CudaBenchmarkResult]) -> None:
    """Print CUDA benchmark results in a formatted table.

    Args:
        results: List of benchmark results.
    """
    print(f"{'Operation':<20} {'Size':>10} {'Device':<8} {'PyTorch (us)':>14} {'DimTensor (us)':>15} {'Overhead':>10}")
    print("-" * 85)

    for r in results:
        pytorch_us = r.pytorch_time * 1_000_000
        dimtensor_us = r.dimtensor_time * 1_000_000
        overhead_str = f"{r.overhead:.2f}x"
        print(
            f"{r.name:<20} {r.tensor_size:>10} {r.device:<8} "
            f"{pytorch_us:>14.2f} {dimtensor_us:>15.2f} {overhead_str:>10}"
        )


def print_device_info() -> None:
    """Print CUDA device information."""
    info = get_device_info()
    print("=== Device Information ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    print()


def quick_cuda_benchmark() -> dict[str, float]:
    """Run a quick CUDA benchmark and return overhead factors.

    Returns:
        Dictionary mapping operation names to overhead factors.
    """
    device = "cuda" if cuda_available() else "cpu"
    size = 10000
    iterations = 50

    return {
        "creation": benchmark_creation_cuda(size, iterations, device).overhead,
        "addition": benchmark_addition_cuda(size, iterations, device).overhead,
        "multiplication": benchmark_multiplication_cuda(size, iterations, device).overhead,
        "division": benchmark_division_cuda(size, iterations, device).overhead,
        "matmul": benchmark_matmul_cuda(64, iterations, device).overhead,
        "sum": benchmark_reduction_cuda(size, iterations, device).overhead,
    }

"""Tests for DimTensor CUDA benchmarks."""

import pytest
import torch

from dimtensor.torch.benchmarks import (
    CudaBenchmarkResult,
    benchmark_addition_cuda,
    benchmark_autograd_backward,
    benchmark_autograd_forward,
    benchmark_chained_ops_cuda,
    benchmark_creation_cuda,
    benchmark_cuda_suite,
    benchmark_division_cuda,
    benchmark_matmul_cuda,
    benchmark_multiplication_cuda,
    benchmark_reduction_cuda,
    cuda_available,
    cuda_time_operation,
    cuda_timer,
    get_device_info,
    print_cuda_results,
    print_device_info,
    quick_cuda_benchmark,
)

# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()
skip_if_no_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


class TestCudaUtilities:
    """Test CUDA utility functions."""

    def test_cuda_available(self) -> None:
        """Test CUDA availability check."""
        result = cuda_available()
        assert isinstance(result, bool)
        assert result == torch.cuda.is_available()

    def test_get_device_info(self) -> None:
        """Test device info retrieval."""
        info = get_device_info()
        assert isinstance(info, dict)
        assert "pytorch_version" in info
        assert "cuda_available" in info

        if CUDA_AVAILABLE:
            assert "cuda_version" in info
            assert "device_name" in info
            assert "device_count" in info

    def test_cuda_timer_cpu(self) -> None:
        """Test CUDA timer on CPU."""
        with cuda_timer(device="cpu") as elapsed:
            # Simple operation
            a = torch.randn(100)
            b = torch.randn(100)
            c = a + b

        assert len(elapsed) == 1
        assert elapsed[0] >= 0
        assert isinstance(elapsed[0], float)

    @skip_if_no_cuda
    def test_cuda_timer_gpu(self) -> None:
        """Test CUDA timer on GPU."""
        with cuda_timer(device="cuda") as elapsed:
            a = torch.randn(1000, device="cuda")
            b = torch.randn(1000, device="cuda")
            c = a + b
            torch.cuda.synchronize()

        assert len(elapsed) == 1
        assert elapsed[0] >= 0
        assert isinstance(elapsed[0], float)

    def test_cuda_time_operation_cpu(self) -> None:
        """Test timing operations on CPU."""

        def operation() -> None:
            a = torch.randn(100)
            b = torch.randn(100)
            c = a + b

        time_taken = cuda_time_operation(operation, iterations=10, warmup=2, device="cpu")
        assert time_taken >= 0
        assert isinstance(time_taken, float)

    @skip_if_no_cuda
    def test_cuda_time_operation_gpu(self) -> None:
        """Test timing operations on GPU."""

        def operation() -> None:
            a = torch.randn(1000, device="cuda")
            b = torch.randn(1000, device="cuda")
            c = a + b

        time_taken = cuda_time_operation(operation, iterations=10, warmup=2, device="cuda")
        assert time_taken >= 0
        assert isinstance(time_taken, float)


class TestBenchmarkResult:
    """Test CudaBenchmarkResult dataclass."""

    def test_benchmark_result_creation(self) -> None:
        """Test creating a benchmark result."""
        result = CudaBenchmarkResult(
            name="test",
            pytorch_time=1.0,
            dimtensor_time=1.5,
            tensor_size=1000,
            device="cuda",
            dtype="torch.float32",
            iterations=100,
        )

        assert result.name == "test"
        assert result.pytorch_time == 1.0
        assert result.dimtensor_time == 1.5
        assert result.tensor_size == 1000
        assert result.device == "cuda"
        assert result.dtype == "torch.float32"
        assert result.iterations == 100

    def test_overhead_calculation(self) -> None:
        """Test overhead calculation."""
        result = CudaBenchmarkResult(
            name="test",
            pytorch_time=1.0,
            dimtensor_time=1.5,
            tensor_size=1000,
            device="cuda",
            dtype="torch.float32",
            iterations=100,
        )

        assert result.overhead == 1.5
        assert result.overhead_percent == 50.0

    def test_overhead_zero_time(self) -> None:
        """Test overhead when pytorch_time is zero."""
        result = CudaBenchmarkResult(
            name="test",
            pytorch_time=0.0,
            dimtensor_time=1.5,
            tensor_size=1000,
            device="cuda",
            dtype="torch.float32",
            iterations=100,
        )

        assert result.overhead == float("inf")


class TestIndividualBenchmarks:
    """Test individual benchmark functions."""

    def test_benchmark_creation_cpu(self) -> None:
        """Test creation benchmark on CPU."""
        result = benchmark_creation_cuda(size=100, iterations=5, device="cpu")

        assert isinstance(result, CudaBenchmarkResult)
        assert result.name == "creation"
        assert result.pytorch_time >= 0
        assert result.dimtensor_time >= 0
        assert result.tensor_size == 100
        assert result.device == "cpu"
        assert result.iterations == 5

    @skip_if_no_cuda
    def test_benchmark_creation_cuda(self) -> None:
        """Test creation benchmark on CUDA."""
        result = benchmark_creation_cuda(size=100, iterations=5, device="cuda")

        assert isinstance(result, CudaBenchmarkResult)
        assert result.name == "creation"
        assert result.pytorch_time >= 0
        assert result.dimtensor_time >= 0
        assert result.tensor_size == 100
        assert result.device == "cuda"

    def test_benchmark_addition_cpu(self) -> None:
        """Test addition benchmark on CPU."""
        result = benchmark_addition_cuda(size=100, iterations=5, device="cpu")

        assert isinstance(result, CudaBenchmarkResult)
        assert result.name == "addition"
        assert result.pytorch_time >= 0
        assert result.dimtensor_time >= 0

    @skip_if_no_cuda
    def test_benchmark_addition_cuda(self) -> None:
        """Test addition benchmark on CUDA."""
        result = benchmark_addition_cuda(size=100, iterations=5, device="cuda")

        assert isinstance(result, CudaBenchmarkResult)
        assert result.name == "addition"
        assert result.device == "cuda"

    def test_benchmark_multiplication_cpu(self) -> None:
        """Test multiplication benchmark on CPU."""
        result = benchmark_multiplication_cuda(size=100, iterations=5, device="cpu")

        assert isinstance(result, CudaBenchmarkResult)
        assert result.name == "multiplication"
        assert result.pytorch_time >= 0
        assert result.dimtensor_time >= 0

    def test_benchmark_division_cpu(self) -> None:
        """Test division benchmark on CPU."""
        result = benchmark_division_cuda(size=100, iterations=5, device="cpu")

        assert isinstance(result, CudaBenchmarkResult)
        assert result.name == "division"
        assert result.pytorch_time >= 0
        assert result.dimtensor_time >= 0

    def test_benchmark_matmul_cpu(self) -> None:
        """Test matmul benchmark on CPU."""
        result = benchmark_matmul_cuda(size=16, iterations=5, device="cpu")

        assert isinstance(result, CudaBenchmarkResult)
        assert result.name == "matmul"
        assert result.pytorch_time >= 0
        assert result.dimtensor_time >= 0

    @skip_if_no_cuda
    def test_benchmark_matmul_cuda(self) -> None:
        """Test matmul benchmark on CUDA."""
        result = benchmark_matmul_cuda(size=16, iterations=5, device="cuda")

        assert isinstance(result, CudaBenchmarkResult)
        assert result.name == "matmul"
        assert result.device == "cuda"

    def test_benchmark_reduction_cpu(self) -> None:
        """Test reduction benchmark on CPU."""
        result = benchmark_reduction_cuda(size=100, iterations=5, device="cpu")

        assert isinstance(result, CudaBenchmarkResult)
        assert result.name == "sum"
        assert result.pytorch_time >= 0
        assert result.dimtensor_time >= 0

    def test_benchmark_chained_ops_cpu(self) -> None:
        """Test chained operations benchmark on CPU."""
        result = benchmark_chained_ops_cuda(size=100, iterations=5, device="cpu")

        assert isinstance(result, CudaBenchmarkResult)
        assert result.name == "chained_ops"
        assert result.pytorch_time >= 0
        assert result.dimtensor_time >= 0
        assert result.mode == "mixed"

    def test_benchmark_autograd_forward_cpu(self) -> None:
        """Test autograd forward benchmark on CPU."""
        result = benchmark_autograd_forward(size=100, iterations=5, device="cpu")

        assert isinstance(result, CudaBenchmarkResult)
        assert result.name == "autograd_forward"
        assert result.pytorch_time >= 0
        assert result.dimtensor_time >= 0
        assert result.mode == "forward"

    def test_benchmark_autograd_backward_cpu(self) -> None:
        """Test autograd backward benchmark on CPU."""
        result = benchmark_autograd_backward(size=100, iterations=5, device="cpu")

        assert isinstance(result, CudaBenchmarkResult)
        assert result.name == "autograd_backward"
        assert result.pytorch_time >= 0
        assert result.dimtensor_time >= 0
        assert result.mode == "backward"

    @skip_if_no_cuda
    def test_benchmark_autograd_forward_cuda(self) -> None:
        """Test autograd forward benchmark on CUDA."""
        result = benchmark_autograd_forward(size=100, iterations=5, device="cuda")

        assert isinstance(result, CudaBenchmarkResult)
        assert result.name == "autograd_forward"
        assert result.device == "cuda"

    @skip_if_no_cuda
    def test_benchmark_autograd_backward_cuda(self) -> None:
        """Test autograd backward benchmark on CUDA."""
        result = benchmark_autograd_backward(size=100, iterations=5, device="cuda")

        assert isinstance(result, CudaBenchmarkResult)
        assert result.name == "autograd_backward"
        assert result.device == "cuda"


class TestBenchmarkSuite:
    """Test the benchmark suite."""

    def test_benchmark_suite_cpu(self) -> None:
        """Test running benchmark suite on CPU."""
        results = benchmark_cuda_suite(
            sizes=[100],
            iterations=5,
            device="cpu",
            dtypes=[torch.float32],
            include_transfers=False,
            include_autograd=True,
        )

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, CudaBenchmarkResult) for r in results)
        assert all(r.device == "cpu" for r in results)

    @skip_if_no_cuda
    def test_benchmark_suite_cuda(self) -> None:
        """Test running benchmark suite on CUDA."""
        results = benchmark_cuda_suite(
            sizes=[100],
            iterations=5,
            device="cuda",
            dtypes=[torch.float32],
            include_transfers=True,
            include_autograd=True,
        )

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, CudaBenchmarkResult) for r in results)
        # Most should be CUDA (some transfers might show differently)
        cuda_results = [r for r in results if r.device == "cuda"]
        assert len(cuda_results) > 0

    def test_benchmark_suite_multiple_sizes(self) -> None:
        """Test benchmark suite with multiple sizes."""
        results = benchmark_cuda_suite(
            sizes=[50, 100],
            iterations=3,
            device="cpu",
            dtypes=[torch.float32],
            include_transfers=False,
            include_autograd=False,
        )

        sizes_tested = {r.tensor_size for r in results if r.name != "matmul"}
        assert 50 in sizes_tested or 100 in sizes_tested

    def test_benchmark_suite_multiple_dtypes(self) -> None:
        """Test benchmark suite with multiple dtypes."""
        results = benchmark_cuda_suite(
            sizes=[100],
            iterations=3,
            device="cpu",
            dtypes=[torch.float32, torch.float64],
            include_transfers=False,
            include_autograd=False,
        )

        dtypes_tested = {r.dtype for r in results}
        assert "torch.float32" in dtypes_tested or "torch.float64" in dtypes_tested


class TestOutputFunctions:
    """Test output and printing functions."""

    def test_print_device_info(self, capsys) -> None:
        """Test printing device info."""
        print_device_info()
        captured = capsys.readouterr()
        assert "Device Information" in captured.out
        assert "pytorch_version" in captured.out

    def test_print_cuda_results(self, capsys) -> None:
        """Test printing benchmark results."""
        results = [
            CudaBenchmarkResult(
                name="test1",
                pytorch_time=1.0e-6,
                dimtensor_time=1.5e-6,
                tensor_size=1000,
                device="cuda",
                dtype="torch.float32",
                iterations=100,
            ),
            CudaBenchmarkResult(
                name="test2",
                pytorch_time=2.0e-6,
                dimtensor_time=3.0e-6,
                tensor_size=2000,
                device="cuda",
                dtype="torch.float32",
                iterations=100,
            ),
        ]

        print_cuda_results(results)
        captured = capsys.readouterr()

        assert "Operation" in captured.out
        assert "PyTorch" in captured.out
        assert "DimTensor" in captured.out
        assert "Overhead" in captured.out
        assert "test1" in captured.out
        assert "test2" in captured.out

    def test_quick_cuda_benchmark(self) -> None:
        """Test quick benchmark function."""
        result = quick_cuda_benchmark()

        assert isinstance(result, dict)
        assert "creation" in result
        assert "addition" in result
        assert "multiplication" in result
        assert "division" in result
        assert "matmul" in result
        assert "sum" in result

        # All overhead values should be positive numbers
        for value in result.values():
            assert isinstance(value, float)
            assert value >= 0

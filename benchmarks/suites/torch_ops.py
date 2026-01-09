"""PyTorch DimTensor benchmarks.

Benchmarks PyTorch backend operations, including CPU/GPU and autograd.
"""

from __future__ import annotations

import numpy as np

# Import after setup to allow asv to install the package
try:
    import torch
    from dimtensor.torch.dimtensor import DimTensor
    from dimtensor.core.units import m, s, kg, dimensionless
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    DimTensor = None
    m = s = kg = dimensionless = None


class TorchBasicOps:
    """Benchmark basic PyTorch operations (CPU)."""

    param_names = ['size']
    params = [[1, 100, 10_000, 1_000_000]]

    def setup(self, size):
        """Set up test tensors."""
        if not TORCH_AVAILABLE:
            raise NotImplementedError("PyTorch not available")

        self.torch_a = torch.randn(size)
        self.torch_b = torch.randn(size)

        self.dt_a = DimTensor(self.torch_a, m)
        self.dt_b = DimTensor(self.torch_b, m)
        self.dt_c = DimTensor(torch.randn(size), s)

    def time_add_torch(self, size):
        """Time torch addition."""
        self.torch_a + self.torch_b

    def time_add_dimtensor(self, size):
        """Time DimTensor addition."""
        self.dt_a + self.dt_b

    def time_multiply_torch(self, size):
        """Time torch multiplication."""
        self.torch_a * self.torch_b

    def time_multiply_dimtensor(self, size):
        """Time DimTensor multiplication."""
        self.dt_a * self.dt_c

    def time_sum_torch(self, size):
        """Time torch sum."""
        self.torch_a.sum()

    def time_sum_dimtensor(self, size):
        """Time DimTensor sum."""
        self.dt_a.sum()


class TorchMatrixOps:
    """Benchmark PyTorch matrix operations."""

    param_names = ['size']
    params = [[10, 100, 500]]

    def setup(self, size):
        """Set up test matrices."""
        if not TORCH_AVAILABLE:
            raise NotImplementedError("PyTorch not available")

        self.torch_a = torch.randn(size, size)
        self.torch_b = torch.randn(size, size)

        self.dt_a = DimTensor(self.torch_a, m)
        self.dt_b = DimTensor(self.torch_b, s)

    def time_matmul_torch(self, size):
        """Time torch matrix multiplication."""
        self.torch_a @ self.torch_b

    def time_matmul_dimtensor(self, size):
        """Time DimTensor matrix multiplication."""
        self.dt_a @ self.dt_b


class TorchAutogradOps:
    """Benchmark PyTorch autograd operations."""

    param_names = ['size']
    params = [[100, 10_000, 1_000_000]]

    def setup(self, size):
        """Set up test tensors with gradients."""
        if not TORCH_AVAILABLE:
            raise NotImplementedError("PyTorch not available")

        self.torch_a = torch.randn(size, requires_grad=True)
        self.torch_b = torch.randn(size, requires_grad=True)

        self.dt_a = DimTensor(torch.randn(size, requires_grad=True), m)
        self.dt_b = DimTensor(torch.randn(size, requires_grad=True), m)

    def time_forward_torch(self, size):
        """Time torch forward pass."""
        result = (self.torch_a + self.torch_b) * 2.0
        result.sum()

    def time_forward_dimtensor(self, size):
        """Time DimTensor forward pass."""
        result = (self.dt_a + self.dt_b) * 2.0
        result.sum()

    def time_backward_torch(self, size):
        """Time torch backward pass."""
        result = (self.torch_a + self.torch_b) * 2.0
        loss = result.sum()
        loss.backward()

    def time_backward_dimtensor(self, size):
        """Time DimTensor backward pass."""
        result = (self.dt_a + self.dt_b) * 2.0
        loss = result.sum()
        loss.backward()


class TorchGPUOps:
    """Benchmark PyTorch GPU operations (if available)."""

    param_names = ['size']
    params = [[100, 10_000, 1_000_000]]

    def setup(self, size):
        """Set up test tensors on GPU."""
        if not TORCH_AVAILABLE:
            raise NotImplementedError("PyTorch not available")
        if not torch.cuda.is_available():
            raise NotImplementedError("CUDA not available")

        self.torch_a = torch.randn(size, device='cuda')
        self.torch_b = torch.randn(size, device='cuda')

        self.dt_a = DimTensor(torch.randn(size, device='cuda'), m)
        self.dt_b = DimTensor(torch.randn(size, device='cuda'), m)
        self.dt_c = DimTensor(torch.randn(size, device='cuda'), s)

    def time_add_torch_gpu(self, size):
        """Time torch addition on GPU."""
        self.torch_a + self.torch_b

    def time_add_dimtensor_gpu(self, size):
        """Time DimTensor addition on GPU."""
        self.dt_a + self.dt_b

    def time_multiply_torch_gpu(self, size):
        """Time torch multiplication on GPU."""
        self.torch_a * self.torch_b

    def time_multiply_dimtensor_gpu(self, size):
        """Time DimTensor multiplication on GPU."""
        self.dt_a * self.dt_c

    def time_sum_torch_gpu(self, size):
        """Time torch sum on GPU."""
        self.torch_a.sum()

    def time_sum_dimtensor_gpu(self, size):
        """Time DimTensor sum on GPU."""
        self.dt_a.sum()


class TorchDeviceTransfers:
    """Benchmark device transfer operations."""

    param_names = ['size']
    params = [[100, 10_000, 1_000_000]]

    def setup(self, size):
        """Set up test tensors."""
        if not TORCH_AVAILABLE:
            raise NotImplementedError("PyTorch not available")
        if not torch.cuda.is_available():
            raise NotImplementedError("CUDA not available")

        self.torch_cpu = torch.randn(size)
        self.dt_cpu = DimTensor(torch.randn(size), m)

        self.torch_gpu = torch.randn(size, device='cuda')
        self.dt_gpu = DimTensor(torch.randn(size, device='cuda'), m)

    def time_cpu_to_gpu_torch(self, size):
        """Time torch CPU to GPU transfer."""
        self.torch_cpu.to('cuda')

    def time_cpu_to_gpu_dimtensor(self, size):
        """Time DimTensor CPU to GPU transfer."""
        self.dt_cpu.to('cuda')

    def time_gpu_to_cpu_torch(self, size):
        """Time torch GPU to CPU transfer."""
        self.torch_gpu.to('cpu')

    def time_gpu_to_cpu_dimtensor(self, size):
        """Time DimTensor GPU to CPU transfer."""
        self.dt_gpu.to('cpu')

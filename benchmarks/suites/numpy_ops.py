"""NumPy DimArray benchmarks.

Benchmarks core operations on DimArray (NumPy backend) and compares
overhead against raw NumPy arrays.
"""

from __future__ import annotations

import numpy as np

# Import after setup to allow asv to install the package
try:
    from dimtensor.core.dimarray import DimArray
    from dimtensor.core.units import m, s, kg, dimensionless
    from dimtensor.functions import matmul
except ImportError:
    # Dummy classes for asv discovery phase
    DimArray = None
    m = s = kg = dimensionless = None
    matmul = None


class ArrayCreation:
    """Benchmark array creation overhead."""

    param_names = ['size']
    params = [[1, 100, 10_000, 1_000_000]]

    def setup(self, size):
        """Set up test data."""
        self.data = list(range(size))
        self.np_data = np.array(self.data)

    def time_from_list_numpy(self, size):
        """Time numpy array creation from list."""
        np.array(self.data)

    def time_from_list_dimarray(self, size):
        """Time DimArray creation from list."""
        DimArray(self.data, m)

    def time_from_numpy_dimarray(self, size):
        """Time DimArray creation from numpy array."""
        DimArray(self.np_data, m)

    def time_copy_dimarray(self, size):
        """Time DimArray copy."""
        da = DimArray(self.np_data, m)
        da.copy()


class ArithmeticOps:
    """Benchmark arithmetic operations."""

    param_names = ['size']
    params = [[1, 100, 10_000, 1_000_000]]

    def setup(self, size):
        """Set up test arrays."""
        self.np_a = np.random.randn(size)
        self.np_b = np.random.randn(size)
        self.np_c = np.random.randn(size) + 1  # Avoid division by zero

        self.da_a = DimArray(self.np_a, m)
        self.da_b = DimArray(self.np_b, m)
        self.da_c = DimArray(self.np_c, s)
        self.da_b_same_unit = DimArray(self.np_b, m)

    def time_add_numpy(self, size):
        """Time numpy addition."""
        self.np_a + self.np_b

    def time_add_dimarray(self, size):
        """Time DimArray addition."""
        self.da_a + self.da_b_same_unit

    def time_subtract_numpy(self, size):
        """Time numpy subtraction."""
        self.np_a - self.np_b

    def time_subtract_dimarray(self, size):
        """Time DimArray subtraction."""
        self.da_a - self.da_b_same_unit

    def time_multiply_numpy(self, size):
        """Time numpy multiplication."""
        self.np_a * self.np_b

    def time_multiply_dimarray(self, size):
        """Time DimArray multiplication."""
        self.da_a * self.da_c

    def time_divide_numpy(self, size):
        """Time numpy division."""
        self.np_a / self.np_c

    def time_divide_dimarray(self, size):
        """Time DimArray division."""
        self.da_a / self.da_c

    def time_power_numpy(self, size):
        """Time numpy power."""
        self.np_a ** 2

    def time_power_dimarray(self, size):
        """Time DimArray power."""
        self.da_a ** 2


class ReductionOps:
    """Benchmark reduction operations."""

    param_names = ['size']
    params = [[1, 100, 10_000, 1_000_000]]

    def setup(self, size):
        """Set up test arrays."""
        self.np_a = np.random.randn(size)
        self.da_a = DimArray(self.np_a, m)

    def time_sum_numpy(self, size):
        """Time numpy sum."""
        self.np_a.sum()

    def time_sum_dimarray(self, size):
        """Time DimArray sum."""
        self.da_a.sum()

    def time_mean_numpy(self, size):
        """Time numpy mean."""
        self.np_a.mean()

    def time_mean_dimarray(self, size):
        """Time DimArray mean."""
        self.da_a.mean()

    def time_std_numpy(self, size):
        """Time numpy std."""
        self.np_a.std()

    def time_std_dimarray(self, size):
        """Time DimArray std."""
        self.da_a.std()

    def time_min_numpy(self, size):
        """Time numpy min."""
        self.np_a.min()

    def time_min_dimarray(self, size):
        """Time DimArray min."""
        self.da_a.min()

    def time_max_numpy(self, size):
        """Time numpy max."""
        self.np_a.max()

    def time_max_dimarray(self, size):
        """Time DimArray max."""
        self.da_a.max()


class MatrixOps:
    """Benchmark matrix operations."""

    param_names = ['size']
    params = [[10, 100, 500]]

    def setup(self, size):
        """Set up test matrices."""
        self.np_a = np.random.randn(size, size)
        self.np_b = np.random.randn(size, size)

        self.da_a = DimArray(self.np_a, m)
        self.da_b = DimArray(self.np_b, s)

    def time_matmul_numpy(self, size):
        """Time numpy matrix multiplication."""
        self.np_a @ self.np_b

    def time_matmul_dimarray(self, size):
        """Time DimArray matrix multiplication."""
        matmul(self.da_a, self.da_b)

    def time_transpose_numpy(self, size):
        """Time numpy transpose."""
        self.np_a.T

    def time_transpose_dimarray(self, size):
        """Time DimArray transpose."""
        self.da_a.T


class IndexingOps:
    """Benchmark indexing operations."""

    param_names = ['size']
    params = [[100, 10_000, 1_000_000]]

    def setup(self, size):
        """Set up test arrays."""
        self.np_a = np.random.randn(size)
        self.da_a = DimArray(self.np_a, m)

        # Different indexing patterns
        self.scalar_idx = size // 2
        self.slice_idx = slice(10, size // 2)
        self.fancy_idx = np.random.randint(0, size, 100)
        self.bool_idx = self.np_a > 0

    def time_scalar_numpy(self, size):
        """Time numpy scalar indexing."""
        self.np_a[self.scalar_idx]

    def time_scalar_dimarray(self, size):
        """Time DimArray scalar indexing."""
        self.da_a[self.scalar_idx]

    def time_slice_numpy(self, size):
        """Time numpy slice indexing."""
        self.np_a[self.slice_idx]

    def time_slice_dimarray(self, size):
        """Time DimArray slice indexing."""
        self.da_a[self.slice_idx]

    def time_fancy_numpy(self, size):
        """Time numpy fancy indexing."""
        self.np_a[self.fancy_idx]

    def time_fancy_dimarray(self, size):
        """Time DimArray fancy indexing."""
        self.da_a[self.fancy_idx]

    def time_boolean_numpy(self, size):
        """Time numpy boolean indexing."""
        self.np_a[self.bool_idx]

    def time_boolean_dimarray(self, size):
        """Time DimArray boolean indexing."""
        self.da_a[self.bool_idx]


class ChainedOps:
    """Benchmark chained operations."""

    param_names = ['size']
    params = [[100, 10_000, 1_000_000]]

    def setup(self, size):
        """Set up test arrays."""
        self.np_a = np.random.randn(size)
        self.np_b = np.random.randn(size)
        self.np_c = np.random.randn(size) + 1

        self.da_a = DimArray(self.np_a, m)
        self.da_b = DimArray(self.np_b, m)
        self.da_c = DimArray(self.np_c, s)

    def time_chain_numpy(self, size):
        """Time numpy chained operations."""
        ((self.np_a + self.np_b) * 2.0) / self.np_c

    def time_chain_dimarray(self, size):
        """Time DimArray chained operations."""
        ((self.da_a + self.da_b) * 2.0) / self.da_c


class ShapeOps:
    """Benchmark shape manipulation operations."""

    param_names = ['size']
    params = [[100, 10_000, 1_000_000]]

    def setup(self, size):
        """Set up test arrays."""
        # Create 2D arrays for reshape tests
        side = int(np.sqrt(size))
        actual_size = side * side

        self.np_a = np.random.randn(actual_size)
        self.np_2d = np.random.randn(side, side)

        self.da_a = DimArray(self.np_a, m)
        self.da_2d = DimArray(self.np_2d, m)

        self.new_shape = (side, side)

    def time_reshape_numpy(self, size):
        """Time numpy reshape."""
        self.np_a.reshape(self.new_shape)

    def time_reshape_dimarray(self, size):
        """Time DimArray reshape."""
        self.da_a.reshape(self.new_shape)

    def time_flatten_numpy(self, size):
        """Time numpy flatten."""
        self.np_2d.flatten()

    def time_flatten_dimarray(self, size):
        """Time DimArray flatten."""
        self.da_2d.flatten()

    def time_squeeze_numpy(self, size):
        """Time numpy squeeze."""
        self.np_a[:, np.newaxis].squeeze()

    def time_squeeze_dimarray(self, size):
        """Time DimArray squeeze."""
        self.da_a[:, np.newaxis].squeeze()


class BroadcastingOps:
    """Benchmark broadcasting operations."""

    param_names = ['size']
    params = [[100, 1_000, 10_000]]

    def setup(self, size):
        """Set up test arrays with broadcasting shapes."""
        self.np_a = np.random.randn(size, 1)
        self.np_b = np.random.randn(1, size)

        self.da_a = DimArray(self.np_a, m)
        self.da_b = DimArray(self.np_b, s)

    def time_broadcast_numpy(self, size):
        """Time numpy broadcasting."""
        self.np_a * self.np_b

    def time_broadcast_dimarray(self, size):
        """Time DimArray broadcasting."""
        self.da_a * self.da_b

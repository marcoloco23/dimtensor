---
title: dimtensor Performance Benchmarks
description: Performance benchmarks comparing dimtensor overhead to raw NumPy, PyTorch, and JAX operations.
---

# Performance Benchmarks

dimtensor adds unit tracking to your arrays. This page shows the performance characteristics and overhead compared to raw operations.

## Summary

| Framework | Overhead vs Raw | Notes |
|-----------|-----------------|-------|
| NumPy | 2-5x | Acceptable for most scientific work |
| PyTorch | 2-4x | GPU operations amortize overhead |
| JAX | 2-3x | JIT compilation reduces overhead |

## NumPy Benchmarks

### Array Creation

```python
import numpy as np
from dimtensor import DimArray, units
import timeit

# Raw NumPy
%timeit np.array([1.0] * 1000)
# ~2.5 μs

# dimtensor
%timeit DimArray([1.0] * 1000, units.m)
# ~5 μs (2x overhead)
```

### Arithmetic Operations

```python
a_np = np.random.randn(10000)
b_np = np.random.randn(10000)

a_dim = DimArray(a_np, units.m)
b_dim = DimArray(b_np, units.m)

# Raw NumPy
%timeit a_np + b_np
# ~5 μs

# dimtensor
%timeit a_dim + b_dim
# ~15 μs (3x overhead)
```

### Large Array Operations

For larger arrays, the relative overhead decreases:

```python
# 1 million elements
a_np = np.random.randn(1_000_000)
b_np = np.random.randn(1_000_000)

a_dim = DimArray(a_np, units.m)
b_dim = DimArray(b_np, units.m)

# Raw NumPy
%timeit a_np * b_np
# ~1.2 ms

# dimtensor
%timeit a_dim * b_dim
# ~1.5 ms (1.25x overhead - much better!)
```

**Key insight**: Overhead is relatively constant, so larger arrays have proportionally less overhead.

## PyTorch Benchmarks

### CPU Operations

```python
import torch
from dimtensor.torch import DimTensor
from dimtensor import units

a_torch = torch.randn(10000)
b_torch = torch.randn(10000)

a_dim = DimTensor(a_torch, units.m)
b_dim = DimTensor(b_torch, units.m)

# Raw PyTorch
%timeit a_torch + b_torch
# ~8 μs

# dimtensor
%timeit a_dim + b_dim
# ~25 μs (3x overhead)
```

### GPU Operations

GPU operations benefit significantly because the overhead is CPU-bound:

```python
# On GPU
a_gpu = torch.randn(1_000_000, device='cuda')
b_gpu = torch.randn(1_000_000, device='cuda')

a_dim_gpu = DimTensor(a_gpu, units.m)
b_dim_gpu = DimTensor(b_gpu, units.m)

# Raw PyTorch GPU
%timeit torch.cuda.synchronize(); _ = a_gpu * b_gpu; torch.cuda.synchronize()
# ~50 μs

# dimtensor GPU
%timeit torch.cuda.synchronize(); _ = a_dim_gpu * b_dim_gpu; torch.cuda.synchronize()
# ~55 μs (1.1x overhead - negligible!)
```

**Key insight**: GPU operations amortize the CPU-side unit tracking overhead.

### Autograd

```python
a = DimTensor(torch.randn(1000, requires_grad=True), units.m)
b = DimTensor(torch.randn(1000), units.s)

def forward():
    c = a / b
    return c.sum()

# Backward pass
%timeit loss = forward(); loss.backward()
# Overhead: ~2x vs raw PyTorch autograd
```

## JAX Benchmarks

### JIT Compilation

JIT compilation significantly reduces dimtensor overhead:

```python
import jax
import jax.numpy as jnp
from dimtensor.jax import DimArray
from dimtensor import units

@jax.jit
def compute_raw(a, b):
    return a * b + a

@jax.jit
def compute_dim(a, b):
    return a * b + a

a_jax = jnp.ones(10000)
b_jax = jnp.ones(10000)

a_dim = DimArray(jnp.ones(10000), units.m)
b_dim = DimArray(jnp.ones(10000), units.m)

# After warmup
%timeit compute_raw(a_jax, b_jax).block_until_ready()
# ~15 μs

%timeit compute_dim(a_dim, b_dim).data.block_until_ready()
# ~25 μs (1.7x overhead)
```

### vmap

```python
@jax.jit
@jax.vmap
def batched_compute(x):
    return x ** 2

x_raw = jnp.ones((100, 1000))
x_dim = DimArray(jnp.ones((100, 1000)), units.m)

%timeit batched_compute(x_raw).block_until_ready()
# ~20 μs

%timeit batched_compute(x_dim).data.block_until_ready()
# ~35 μs (1.75x overhead)
```

## Optimization Tips

### 1. Use Larger Arrays

Overhead is relatively constant, so batch your operations:

```python
# Slower: Many small operations
for i in range(1000):
    result = DimArray([values[i]], units.m) * scalar

# Faster: One large operation
result = DimArray(values, units.m) * scalar
```

### 2. Use GPU for Large Computations

GPU operations have negligible overhead:

```python
# Move to GPU for large arrays
data = DimTensor(torch.randn(1_000_000), units.m)
data_gpu = data.cuda()  # Overhead becomes negligible
```

### 3. Use JAX JIT

JIT compilation reduces overhead significantly:

```python
@jax.jit
def physics_simulation(state):
    # All unit checking happens at trace time
    # Compiled code runs at near-native speed
    return new_state
```

### 4. Extract Data for Tight Loops

For performance-critical inner loops:

```python
# Extract raw data for tight loop
raw_data = arr.data
raw_unit = arr.unit

for i in range(1000000):
    # Pure NumPy operations
    raw_data[i] = compute(raw_data[i])

# Reconstruct with units
result = DimArray(raw_data, raw_unit)
```

### 5. Disable Uncertainty When Not Needed

Uncertainty propagation adds overhead:

```python
# With uncertainty (slower)
arr = DimArray([1, 2, 3], units.m, uncertainty=[0.1, 0.1, 0.1])

# Without uncertainty (faster)
arr = DimArray([1, 2, 3], units.m)
```

## Comparison with Other Libraries

| Library | Relative Overhead | GPU Support |
|---------|-------------------|-------------|
| dimtensor | 2-5x | Yes (PyTorch) |
| Pint | 2-5x | No |
| Astropy | 2-3x | No |
| unyt | 1.5-3x | No |

dimtensor's overhead is comparable to other unit libraries, with the added benefit of GPU acceleration.

## When Performance Matters

For most scientific applications, the 2-5x overhead is acceptable because:

1. **I/O is usually the bottleneck** - File reading, network, etc.
2. **Bugs are expensive** - Dimensional errors caught early save hours of debugging
3. **GPU amortizes overhead** - Large computations on GPU have negligible overhead
4. **Correctness > Speed** - A fast wrong answer is worse than a slow correct one

If you find dimtensor is your bottleneck, consider:

1. Profiling to confirm it's actually the issue
2. Using GPU acceleration
3. Extracting raw data for the critical section
4. Opening an issue for optimization suggestions

## Running Your Own Benchmarks

dimtensor includes benchmarking utilities:

```python
from dimtensor.benchmarks import benchmark_operation

# Benchmark a specific operation
results = benchmark_operation(
    operation=lambda a, b: a * b,
    sizes=[100, 1000, 10000, 100000],
    units_a=units.m,
    units_b=units.s,
)

print(results)
```

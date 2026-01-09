# Getting Started

## Installation

### Basic Installation

Install dimtensor from PyPI:

```bash
pip install dimtensor
```

This gives you the core library with NumPy support, physical constants, and basic I/O.

### Framework-Specific Installation

Choose the installation that matches your ML framework:

```bash
# PyTorch integration (DimTensor with autograd)
pip install dimtensor[torch]

# JAX integration (pytree support for jit/vmap/grad)
pip install dimtensor[jax]

# Both frameworks
pip install dimtensor[torch,jax]
```

### Use Case Installation

Install additional features based on your needs:

```bash
# Data science tools
pip install dimtensor[pandas]      # Pandas integration
pip install dimtensor[polars]      # Polars integration
pip install dimtensor[xarray]      # xarray integration

# File formats
pip install dimtensor[h5py]        # HDF5 support
pip install dimtensor[netcdf]      # NetCDF support
pip install dimtensor[parquet]     # Parquet support

# Visualization
pip install dimtensor[matplotlib]  # Matplotlib plots
pip install dimtensor[plotly]      # Plotly interactive plots

# Scientific computing
pip install dimtensor[scipy]       # SciPy integration
pip install dimtensor[sympy]       # Symbolic math
pip install dimtensor[sklearn]     # Scikit-learn integration

# Everything (recommended for exploration)
pip install dimtensor[all]

# Development (includes pytest, mypy, ruff)
pip install dimtensor[dev]
```

## Quick Start

### Creating DimArrays

A `DimArray` is a NumPy array with an attached physical unit:

```python
from dimtensor import DimArray, units

# From a list
velocity = DimArray([10, 20, 30], units.m / units.s)

# From a NumPy array
import numpy as np
data = np.array([1.0, 2.0, 3.0])
distance = DimArray(data, units.km)

# Scalar values
mass = DimArray(5.0, units.kg)
```

### Automatic Dimension Checking

dimtensor catches dimension errors immediately:

```python
from dimtensor import DimArray, units

# Multiplication combines dimensions
velocity = DimArray([10], units.m / units.s)
time = DimArray([5], units.s)
distance = velocity * time
print(distance)  # [50] m

# Addition requires same dimensions
a = DimArray([1, 2], units.m)
b = DimArray([3, 4], units.m)
print(a + b)  # [4 6] m

# Incompatible dimensions raise errors
acceleration = DimArray([9.8], units.m / units.s**2)
velocity + acceleration  # DimensionError: cannot add m/s to m/s^2
```

### Unit Conversion

Convert between compatible units effortlessly:

```python
distance = DimArray([1.0], units.km)

# Convert to meters
in_meters = distance.to(units.m)
print(in_meters)  # [1000.] m

# Convert to miles
in_miles = distance.to(units.mile)
print(in_miles)  # [0.62137119] mi

# Units simplify automatically
force = DimArray([10], units.kg * units.m / units.s**2)
print(force)  # [10] N (newton)
```

### PyTorch Integration

If you installed `dimtensor[torch]`, use unit-aware tensors with autograd:

```python
import torch
from dimtensor.torch import DimTensor
from dimtensor import units

# Create tensors with units
v = DimTensor(
    torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
    units.m / units.s
)
t = DimTensor(torch.tensor([0.5, 1.0, 1.5]), units.s)

# Operations preserve units
d = v * t  # distance in meters

# Gradients flow through
d.sum().backward()
print(v.grad)  # tensor([0.5000, 1.0000, 1.5000])

# GPU support
v_cuda = v.cuda()
```

### JAX Integration

If you installed `dimtensor[jax]`, DimArrays work with JIT, vmap, and grad:

```python
import jax
import jax.numpy as jnp
from dimtensor.jax import DimArray
from dimtensor import units

@jax.jit
def kinetic_energy(mass, velocity):
    return 0.5 * mass * velocity**2

m = DimArray(jnp.array([1.0, 2.0]), units.kg)
v = DimArray(jnp.array([10.0, 20.0]), units.m / units.s)
E = kinetic_energy(m, v)  # JIT-compiled, units preserved
print(E)  # [50. 400.] J
```

### Physical Constants

Access CODATA 2022 constants with proper units:

```python
from dimtensor import constants, DimArray, units

# Speed of light
print(constants.c)  # 299792458.0 m/s (exact)

# Planck constant with uncertainty
print(constants.h)  # 6.62607015e-34 J·s

# Use in calculations
mass = DimArray([1.0], units.kg)
energy = mass * constants.c**2  # E = mc²
print(energy)  # [8.98755179e+16] J
```

### Uncertainty Propagation

Track measurement uncertainties through calculations:

```python
from dimtensor import DimArray, units

length = DimArray([10.0], units.m, uncertainty=[0.1])
width = DimArray([5.0], units.m, uncertainty=[0.05])

# Uncertainties propagate automatically
area = length * width
print(area)  # 50.0 ± 0.71 m² (quadrature sum)
```

### Save and Load with Units

Units are preserved during I/O operations:

```python
from dimtensor import DimArray, units
from dimtensor.io import save_json, load_json

arr = DimArray([1.0, 2.0, 3.0], units.m / units.s)

# Save with units
save_json(arr, "velocities.json")

# Load with units intact
loaded = load_json("velocities.json")
print(loaded.unit)  # m/s
```

## Key Features

- **Dimensional Safety** - Catch unit errors at operation time, not after hours of computation
- **Unit Conversion** - Convert between compatible units with `.to()`
- **Framework Support** - NumPy, PyTorch (with autograd), and JAX (with JIT)
- **Physical Constants** - CODATA 2022 constants with uncertainties
- **Uncertainty Propagation** - Automatic uncertainty tracking through all operations
- **Rich I/O** - JSON, HDF5, Parquet, NetCDF, pandas, xarray
- **Visualization** - Matplotlib and Plotly with automatic unit labels
- **Domain Units** - Astronomy, chemistry, and engineering units
- **Type Safety** - Full type hints for IDE support

## Accessing Data

```python
arr = DimArray([1, 2, 3], units.m)

# Get the underlying NumPy array (read-only)
arr.data  # array([1, 2, 3])

# Get the unit
arr.unit  # Unit(m)

# Array properties
arr.shape  # (3,)
arr.ndim   # 1
arr.size   # 3
arr.dtype  # dtype('int64')
```

## What's Next?

- **[Working with Units](guide/units.md)** - Available units, creating custom units, and unit systems
- **[Array Operations](guide/operations.md)** - Reshaping, broadcasting, linear algebra, and advanced operations
- **[Examples](guide/examples.md)** - Real-world physics calculations and machine learning examples
- **[API Reference](api/index.md)** - Complete API documentation
- **[GitHub Repository](https://github.com/marcoloco23/dimtensor)** - Source code, issues, and discussions

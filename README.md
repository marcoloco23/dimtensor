<div align="center">

# dimtensor

**Unit-aware tensors for physics and scientific machine learning**

[![PyPI version](https://img.shields.io/pypi/v/dimtensor.svg)](https://pypi.org/project/dimtensor/)
[![Python versions](https://img.shields.io/pypi/pyversions/dimtensor.svg)](https://pypi.org/project/dimtensor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/dimtensor/month)](https://pepy.tech/project/dimtensor)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://marcoloco23.github.io/dimtensor)

[Documentation](https://marcoloco23.github.io/dimtensor) |
[PyPI](https://pypi.org/project/dimtensor/) |
[Changelog](https://github.com/marcoloco23/dimtensor/blob/main/CHANGELOG.md) |
[Contributing](https://github.com/marcoloco23/dimtensor/blob/main/CONTRIBUTING.md)

</div>

---

**dimtensor** catches dimensional errors at operation time, not after hours of computation.

```python
from dimtensor import DimArray, units

# Operations check dimensions automatically
velocity = DimArray([10, 20, 30], units.m / units.s)
time = DimArray([1, 2, 3], units.s)
distance = velocity * time  # [10 40 90] m

# Errors caught immediately
acceleration = DimArray([9.8], units.m / units.s**2)
velocity + acceleration  # DimensionError: cannot add m/s to m/s^2
```

## Why dimtensor?

| Problem | Solution |
|---------|----------|
| Silent unit errors waste compute time | Immediate `DimensionError` at operation time |
| PyTorch has no unit support | Native `DimTensor` with full autograd and GPU support |
| JAX incompatible with unit libraries | `DimArray` registered as pytree for jit/vmap/grad |
| Uncertainty handled separately | Built-in uncertainty propagation through all operations |
| Units lost during I/O | Save/load with units to JSON, HDF5, Parquet, NetCDF |

## Features

- **Dimensional Safety** - Operations between incompatible dimensions raise `DimensionError`
- **Unit Conversion** - Convert between compatible units with `.to()`
- **NumPy/PyTorch/JAX** - Full integration with all three frameworks
- **Physical Constants** - CODATA 2022 constants with proper units and uncertainties
- **Uncertainty Propagation** - Track and propagate measurement uncertainties
- **I/O Support** - JSON, HDF5, Parquet, NetCDF, pandas, xarray
- **Visualization** - Matplotlib and Plotly with automatic unit labels
- **Domain Units** - Astronomy, chemistry, and engineering units
- **Dimensional Inference** - Infer dimensions from variable names and equations
- **Dimensional Linting** - Static analysis CLI for finding unit errors
- **Optional Rust Backend** - Accelerated operations when built from source

## Installation

```bash
pip install dimtensor
```

For framework-specific support:
```bash
pip install dimtensor[torch]  # PyTorch integration
pip install dimtensor[jax]    # JAX integration
pip install dimtensor[all]    # All optional dependencies
```

### Optional: Rust Backend (v2.0+)

For improved performance, build the optional Rust backend:

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin

# Build and install the Rust extension
cd /path/to/dimtensor/rust
maturin build --release
pip install target/wheels/dimtensor_core-*.whl
```

The library automatically uses the Rust backend when available, with a pure Python fallback otherwise. Check availability:

```python
from dimtensor._rust import HAS_RUST_BACKEND
print(f"Rust backend: {HAS_RUST_BACKEND}")
```

## Quick Start

### NumPy

```python
from dimtensor import DimArray, units

v = DimArray([10], units.m / units.s)  # velocity
t = DimArray([5], units.s)              # time
d = v * t                               # distance = 50 m
```

### PyTorch

```python
import torch
from dimtensor.torch import DimTensor
from dimtensor import units

# Unit-aware tensors with autograd
v = DimTensor(torch.tensor([1.0, 2.0, 3.0], requires_grad=True), units.m / units.s)
t = DimTensor(torch.tensor([0.5, 1.0, 1.5]), units.s)
d = v * t  # distance in meters

# Gradients flow through
d.sum().backward()
print(v.grad)

# GPU support
v_cuda = v.cuda()
```

### JAX

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
E = kinetic_energy(m, v)  # JIT-compiled, units preserved: [50. 400.] J
```

### Physical Constants

```python
from dimtensor import constants, DimArray, units

print(constants.c)   # Speed of light: 299792458.0 m/s
print(constants.h)   # Planck constant with uncertainty

E = constants.c**2 * DimArray([1.0], units.kg)  # E = mc^2
```

### Uncertainty Propagation

```python
from dimtensor import DimArray, units

length = DimArray([10.0], units.m, uncertainty=[0.1])
width = DimArray([5.0], units.m, uncertainty=[0.05])

area = length * width  # 50 +/- 0.71 m^2 (propagated in quadrature)
```

### I/O

```python
from dimtensor import DimArray, units
from dimtensor.io import save_json, load_json, save_hdf5, load_hdf5

arr = DimArray([1.0, 2.0, 3.0], units.m)

# JSON
save_json(arr, "data.json")
loaded = load_json("data.json")  # Units preserved

# HDF5
save_hdf5(arr, "data.h5", compression="gzip")
loaded = load_hdf5("data.h5")
```

### Visualization

```python
from dimtensor import DimArray, units
from dimtensor.visualization import plot

time = DimArray([0, 1, 2, 3], units.s)
distance = DimArray([0, 10, 40, 90], units.m)

plot(time, distance)  # Axes labeled automatically: [s], [m]
```

### Domain-Specific Units

```python
from dimtensor import DimArray
from dimtensor.domains.astronomy import parsec, light_year, solar_mass
from dimtensor.domains.chemistry import molar, dalton
from dimtensor.domains.engineering import MPa, hp, kWh

# Astronomy
distance = DimArray([4.24], light_year).to(parsec)  # ~1.3 pc

# Chemistry
concentration = DimArray([0.1], molar)  # 0.1 M

# Engineering
stress = DimArray([250], MPa)
power = DimArray([100], hp)
```

### Dimensional Inference (v2.0+)

```python
from dimtensor.inference import infer_dimension, get_equations_by_domain

# Infer dimension from variable name
result = infer_dimension("velocity")
print(result.dimension)    # L·T⁻¹ (length/time)
print(result.confidence)   # 0.9

# Works with prefixes and suffixes
result = infer_dimension("initial_velocity_x")
print(result.dimension)    # L·T⁻¹

# Query physics equations
mechanics = get_equations_by_domain("mechanics")
for eq in mechanics[:3]:
    print(f"{eq.name}: {eq.formula}")
# Newton's Second Law: F = ma
# Kinetic Energy: KE = ½mv²
# Gravitational Force: F = Gm₁m₂/r²
```

### Dimensional Linting (v2.1+)

```bash
# Lint a file for dimensional issues
dimtensor lint physics_simulation.py

# Example output:
# physics.py:15:4: W002 Potential dimension mismatch: LT⁻¹ + LT⁻²
#   velocity + acceleration
#   Suggestion: Cannot add/subtract LT⁻¹ and LT⁻². Check your units.

# JSON output for IDE integration
dimtensor lint --format=json src/

# Strict mode (report all inferences)
dimtensor lint --strict script.py
```

## Useful Links

- [Documentation](https://marcoloco23.github.io/dimtensor)
- [Source Code](https://github.com/marcoloco23/dimtensor)
- [Issue Tracker](https://github.com/marcoloco23/dimtensor/issues)
- [Changelog](https://github.com/marcoloco23/dimtensor/blob/main/CHANGELOG.md)
- [PyPI](https://pypi.org/project/dimtensor/)

## Call for Contributions

dimtensor is an open source project and welcomes contributions of all kinds. Here are ways to get involved:

- **Report bugs** - [Open an issue](https://github.com/marcoloco23/dimtensor/issues/new)
- **Request features** - Share ideas in [discussions](https://github.com/marcoloco23/dimtensor/discussions)
- **Contribute code** - See our [contributing guide](CONTRIBUTING.md)
- **Improve docs** - Fix typos, add examples, clarify explanations
- **Share use cases** - Write tutorials or blog posts

Writing code isn't the only way to contribute. Good issues, documentation improvements, and community engagement are just as valuable.

## License

MIT

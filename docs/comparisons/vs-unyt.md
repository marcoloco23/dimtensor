---
title: dimtensor vs unyt - Python Unit Libraries Compared
description: Compare dimtensor and unyt for scientific computing with units. Learn which library fits your astrophysics and simulation needs.
---

# dimtensor vs unyt

[unyt](https://unyt.readthedocs.io/) is the units library from the yt project, designed for astrophysical simulations. This page compares it with dimtensor.

## TL;DR

| Feature | dimtensor | unyt |
|---------|-----------|------|
| PyTorch support | Native | None |
| JAX support | Native | None |
| GPU support | CUDA, MPS | Dask arrays |
| Uncertainty | Built-in | No |
| yt integration | No | Native |
| Performance | 2-5x NumPy | Optimized |
| Cosmology units | No | Yes |

## When to Choose dimtensor

**Choose dimtensor if you need:**

- Machine learning with PyTorch or JAX
- GPU acceleration with unit tracking
- Built-in uncertainty propagation
- Multiple I/O formats (HDF5, NetCDF, Parquet)

```python
# dimtensor: PyTorch with autograd
from dimtensor.torch import DimTensor
from dimtensor import units
import torch

v = DimTensor(torch.tensor([1.0, 2.0], requires_grad=True), units.m / units.s)
energy = 0.5 * v ** 2
energy.sum().backward()  # Gradients work
```

## When to Choose unyt

**Choose unyt if you need:**

- Integration with yt for astrophysical visualizations
- Cosmological unit conversions
- Optimized performance for large simulations
- Dask array support for out-of-core computation

```python
# unyt: Cosmological units
from unyt import unyt_array
from unyt.unit_systems import cgs_unit_system

# Comoving coordinates
distance = unyt_array([100], 'Mpc/h')

# yt integration
import yt
ds = yt.load("simulation.hdf5")
# Units handled automatically
```

## Feature Comparison

### Basic Usage

=== "dimtensor"

    ```python
    from dimtensor import DimArray, units

    velocity = DimArray([10, 20, 30], units.m / units.s)
    time = DimArray([1, 2, 3], units.s)
    distance = velocity * time
    ```

=== "unyt"

    ```python
    from unyt import unyt_array

    velocity = unyt_array([10, 20, 30], 'm/s')
    time = unyt_array([1, 2, 3], 's')
    distance = velocity * time
    ```

### Machine Learning

=== "dimtensor"

    ```python
    import jax
    import jax.numpy as jnp
    from dimtensor.jax import DimArray
    from dimtensor import units

    @jax.jit
    def simulate(mass, velocity):
        return 0.5 * mass * velocity**2

    m = DimArray(jnp.array([1.0, 2.0]), units.kg)
    v = DimArray(jnp.array([10.0, 20.0]), units.m / units.s)
    E = simulate(m, v)  # JIT works with units
    ```

=== "unyt"

    ```python
    # unyt doesn't support JAX or PyTorch
    # Must extract values for ML frameworks
    from unyt import unyt_array

    mass = unyt_array([1.0, 2.0], 'kg')
    mass_values = mass.v  # Extract values for ML
    ```

### GPU Support

=== "dimtensor"

    ```python
    from dimtensor.torch import DimTensor
    from dimtensor import units
    import torch

    # Native CUDA support
    arr = DimTensor(torch.randn(1000, 1000), units.m)
    arr_gpu = arr.cuda()  # Units preserved on GPU
    result = arr_gpu @ arr_gpu.T  # GPU computation with units
    ```

=== "unyt"

    ```python
    # unyt uses Dask for distributed computing
    # No direct GPU support
    from unyt import unyt_array
    import dask.array as da

    # Dask for out-of-core computation
    data = da.from_array(large_array, chunks=(1000, 1000))
    # Limited unit support with Dask
    ```

### Uncertainty Propagation

=== "dimtensor"

    ```python
    from dimtensor import DimArray, units

    # Built-in uncertainty
    measurement = DimArray([10.0], units.m, uncertainty=[0.1])
    area = measurement ** 2  # Uncertainty propagates
    print(area)  # 100 +/- 2.0 m^2
    ```

=== "unyt"

    ```python
    # unyt doesn't have built-in uncertainty
    # Must track manually or use separate library
    from unyt import unyt_array
    import uncertainties

    # Manual tracking required
    ```

### I/O Support

| Format | dimtensor | unyt |
|--------|-----------|------|
| JSON | Yes | Yes |
| HDF5 | Yes | Yes (yt native) |
| Parquet | Yes | No |
| NetCDF | Yes | No |
| pandas | Yes | Limited |
| yt datasets | No | Native |

## Performance Comparison

Both libraries prioritize performance for scientific computing:

| Operation | dimtensor | unyt |
|-----------|-----------|------|
| Array creation | ~2x NumPy | ~1.5x NumPy |
| Arithmetic | ~2-5x NumPy | ~2-3x NumPy |
| Unit conversion | Fast | Fast |
| GPU operations | Native | N/A |

unyt is slightly more optimized for pure NumPy operations, while dimtensor provides GPU acceleration.

## Migration from unyt

If you're adding ML to simulation code:

```python
# unyt code
from unyt import unyt_array

mass = unyt_array([1e10, 2e10], 'Msun')
velocity = unyt_array([100, 200], 'km/s')
kinetic_energy = 0.5 * mass * velocity**2

# dimtensor equivalent
from dimtensor import DimArray
from dimtensor.domains.astronomy import solar_mass
from dimtensor import units

mass = DimArray([1e10, 2e10], solar_mass)
velocity = DimArray([100, 200], units.km / units.s)
kinetic_energy = 0.5 * mass * velocity**2
# Now works with PyTorch/JAX
```

## Using Both Together

For yt-based projects needing ML:

```python
import yt
from dimtensor.torch import DimTensor
from dimtensor import units
import torch

# Load simulation with yt (uses unyt internally)
ds = yt.load("simulation.hdf5")
ad = ds.all_data()

# Extract data for ML
density_values = ad['gas', 'density'].to('g/cm**3').v
temperature_values = ad['gas', 'temperature'].to('K').v

# Convert to dimtensor for ML pipeline
density = DimTensor(
    torch.tensor(density_values, dtype=torch.float32),
    units.g / units.cm**3
)
temperature = DimTensor(
    torch.tensor(temperature_values, dtype=torch.float32),
    units.K
)

# Now use in your neural network...
```

## Conclusion

- **Use dimtensor** for ML workflows, GPU computing, or uncertainty propagation
- **Use unyt** for yt integration, cosmological simulations, or Dask-based workflows
- **Use both** when you need yt's analysis tools plus ML capabilities

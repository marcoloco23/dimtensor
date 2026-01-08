# dimtensor

Unit-aware tensors for physics and scientific machine learning.

`dimtensor` wraps NumPy, PyTorch, and JAX arrays with physical unit tracking, catching dimensional errors at operation time rather than after hours of computation.

## Installation

```bash
pip install dimtensor
```

For PyTorch or JAX support:
```bash
pip install dimtensor[torch]  # PyTorch integration
pip install dimtensor[jax]    # JAX integration
pip install dimtensor[all]    # All optional dependencies
```

## Quick Start

```python
from dimtensor import DimArray, units

# Create dimension-aware arrays
velocity = DimArray([10, 20, 30], units.m / units.s)
time = DimArray([1, 2, 3], units.s)

# Operations preserve/check dimensions
distance = velocity * time
print(distance)  # [10 40 90] m

# Catch errors early
acceleration = DimArray([9.8], units.m / units.s**2)
velocity + acceleration  # DimensionError: cannot add m/s to m/s^2
```

## Features

- **Dimensional Safety**: Operations between incompatible dimensions raise `DimensionError` immediately
- **Unit Conversion**: Convert between compatible units with `.to()`
- **SI Units**: Full support for SI base and derived units
- **NumPy Integration**: Works with NumPy arrays and supports common operations
- **PyTorch Integration**: `DimTensor` wraps PyTorch tensors with autograd support
- **JAX Integration**: `DimArray` registered as JAX pytree for jit/vmap/grad
- **Physical Constants**: CODATA constants with proper units and uncertainties
- **Uncertainty Propagation**: Track and propagate measurement uncertainties
- **I/O Support**: Save/load to JSON, HDF5, and pandas DataFrames

## PyTorch Integration

```python
import torch
from dimtensor.torch import DimTensor
from dimtensor import units

# Unit-aware tensors with autograd
v = DimTensor(torch.tensor([1.0, 2.0, 3.0], requires_grad=True), units.m / units.s)
t = DimTensor(torch.tensor([0.5, 1.0, 1.5]), units.s)
d = v * t  # distance in meters

# Gradients work
loss = d.sum()
loss.backward()
print(v.grad)  # Gradients flow through

# GPU support
v_cuda = v.cuda()  # Move to GPU, preserving units
```

## JAX Integration

```python
import jax
import jax.numpy as jnp
from dimtensor.jax import DimArray
from dimtensor import units

# Unit-aware arrays compatible with JAX transformations
@jax.jit
def kinetic_energy(mass, velocity):
    return 0.5 * mass * velocity**2

m = DimArray(jnp.array([1.0, 2.0]), units.kg)
v = DimArray(jnp.array([10.0, 20.0]), units.m / units.s)
E = kinetic_energy(m, v)  # JIT-compiled, units preserved
print(E)  # [50. 400.] J
```

## Physical Constants

```python
from dimtensor import constants

# CODATA physical constants with proper units
print(constants.c)           # Speed of light: 299792458.0 m/s
print(constants.h)           # Planck constant with uncertainty
print(constants.G)           # Gravitational constant

# Use in calculations
E = constants.c**2 * DimArray([1.0], units.kg)  # E = mc^2
```

## Uncertainty Propagation

```python
from dimtensor import DimArray, units

# Measurement with uncertainty
length = DimArray([10.0], units.m, uncertainty=[0.1])
width = DimArray([5.0], units.m, uncertainty=[0.05])

# Uncertainties propagate through operations
area = length * width
print(area)  # 50 +/- 0.71 m^2 (propagated in quadrature)
```

## I/O Support

### JSON

```python
from dimtensor.io import save_json, load_json

arr = DimArray([1.0, 2.0, 3.0], units.m)
save_json(arr, "data.json")
loaded = load_json("data.json")  # Units preserved
```

### HDF5

```python
from dimtensor.io import save_hdf5, load_hdf5

arr = DimArray([1.0, 2.0, 3.0], units.m)
save_hdf5(arr, "data.h5", compression="gzip")
loaded = load_hdf5("data.h5")
```

### Pandas

```python
from dimtensor.io import to_dataframe, from_dataframe

data = {
    "position": DimArray([1, 2, 3], units.m),
    "velocity": DimArray([10, 20, 30], units.m / units.s)
}
df = to_dataframe(data)  # Unit metadata in attrs
arrays = from_dataframe(df)  # Reconstruct DimArrays
```

### NetCDF

```python
from dimtensor.io import save_netcdf, load_netcdf

arr = DimArray([1.0, 2.0, 3.0], units.m)
save_netcdf(arr, "data.nc")
loaded = load_netcdf("data.nc")
```

### Parquet

```python
from dimtensor.io import save_parquet, load_parquet

arr = DimArray([1.0, 2.0, 3.0], units.m)
save_parquet(arr, "data.parquet", compression="snappy")
loaded = load_parquet("data.parquet")
```

### xarray

```python
from dimtensor.io import to_xarray, from_xarray

arr = DimArray([1.0, 2.0, 3.0], units.m)
da = to_xarray(arr, name="distance", dims=("x",))
restored = from_xarray(da)  # Back to DimArray with units
```

## Examples

### Kinematics

```python
from dimtensor import DimArray, units

v = DimArray([10], units.m / units.s)  # velocity
t = DimArray([5], units.s)              # time
d = v * t                               # distance = 50 m
```

### Force and Energy

```python
m = DimArray([2], units.kg)              # mass
g = DimArray([9.8], units.m / units.s**2)  # gravity
h = DimArray([10], units.m)              # height

# Potential energy
PE = m * g * h  # 196 J

# Force
F = m * g  # 19.6 N
```

### Unit Conversion

```python
distance = DimArray([1], units.km)
in_meters = distance.to(units.m)  # 1000 m
in_miles = distance.to(units.mile)  # ~0.621 mi
```

## Why dimtensor?

1. **Catch errors early**: Don't waste compute on dimensionally invalid calculations
2. **Self-documenting code**: Units make physics code clearer
3. **ML-ready**: PyTorch autograd and JAX transformations fully supported
4. **Scientific computing**: Physical constants and uncertainty propagation
5. **Lightweight**: Just metadata tracking, minimal overhead

## License

MIT

---
title: Common dimtensor Errors and Solutions
description: Troubleshooting guide for common errors when using dimtensor for unit-aware computation in Python.
---

# Common Errors and How to Fix Them

This page covers the most common errors you'll encounter when using dimtensor and how to resolve them.

## DimensionError

### Cannot add/subtract incompatible dimensions

**Error:**
```
DimensionError: cannot add m/s to m/s^2
```

**Cause:** You're trying to add or subtract quantities with different dimensions. This is a physics error - you can't add velocity to acceleration.

**Solution:** Ensure both operands have the same dimensions:

```python
from dimtensor import DimArray, units

velocity = DimArray([10], units.m / units.s)
acceleration = DimArray([9.8], units.m / units.s**2)
time = DimArray([2], units.s)

# Wrong: Adding velocity to acceleration
# velocity + acceleration  # DimensionError!

# Right: Convert acceleration to velocity change first
velocity_change = acceleration * time  # Now has units m/s
total_velocity = velocity + velocity_change  # Works!
```

### Cannot compare incompatible dimensions

**Error:**
```
DimensionError: cannot compare m to s
```

**Cause:** Comparing quantities with different dimensions doesn't make physical sense.

**Solution:** Only compare quantities with the same dimensions:

```python
# Wrong
length = DimArray([10], units.m)
time = DimArray([5], units.s)
# length > time  # DimensionError!

# Right: Compare same dimensions
length1 = DimArray([10], units.m)
length2 = DimArray([5], units.m)
length1 > length2  # True
```

## UnitConversionError

### Cannot convert between incompatible units

**Error:**
```
UnitConversionError: cannot convert m to s
```

**Cause:** Attempting to convert between units that measure different physical quantities.

**Solution:** Only convert between compatible units:

```python
from dimtensor import DimArray, units

distance = DimArray([1000], units.m)

# Wrong: meters to seconds (different dimensions)
# distance.to(units.s)  # UnitConversionError!

# Right: meters to kilometers (same dimension)
distance.to(units.km)  # 1 km
```

### Unit not found

**Error:**
```
AttributeError: module 'dimtensor.units' has no attribute 'xyz'
```

**Cause:** The unit you're trying to use doesn't exist in dimtensor.

**Solution:** Check the available units or use domain-specific units:

```python
from dimtensor import units

# Common SI units
units.m      # meter
units.kg     # kilogram
units.s      # second
units.A      # ampere
units.K      # kelvin
units.mol    # mole
units.cd     # candela

# Derived units
units.N      # newton
units.J      # joule
units.W      # watt
units.Pa     # pascal
units.Hz     # hertz

# For domain-specific units
from dimtensor.domains.astronomy import parsec, solar_mass
from dimtensor.domains.chemistry import molar, dalton
from dimtensor.domains.engineering import MPa, hp
```

## Shape Errors

### Incompatible shapes for operation

**Error:**
```
ValueError: operands could not be broadcast together with shapes (3,) (4,)
```

**Cause:** NumPy broadcasting rules still apply. Arrays must have compatible shapes.

**Solution:** Ensure arrays can be broadcast together:

```python
from dimtensor import DimArray, units

a = DimArray([1, 2, 3], units.m)      # shape (3,)
b = DimArray([1, 2, 3, 4], units.m)   # shape (4,)

# Wrong: Incompatible shapes
# a + b  # ValueError!

# Right: Use compatible shapes
b = DimArray([1, 2, 3], units.m)      # shape (3,)
a + b  # Works!

# Or use broadcasting
a = DimArray([[1], [2], [3]], units.m)    # shape (3, 1)
b = DimArray([1, 2, 3, 4], units.m)       # shape (4,)
a + b  # Broadcasting works: shape (3, 4)
```

## PyTorch-Specific Errors

### Tensor dtype mismatch

**Error:**
```
RuntimeError: expected scalar type Float but found Double
```

**Solution:** Ensure consistent dtypes:

```python
from dimtensor.torch import DimTensor
from dimtensor import units
import torch

# Specify dtype explicitly
a = DimTensor(torch.tensor([1.0, 2.0], dtype=torch.float32), units.m)
b = DimTensor(torch.tensor([3.0, 4.0], dtype=torch.float32), units.m)
a + b  # Works!
```

### Gradient computation on integer tensor

**Error:**
```
RuntimeError: Only Tensors of floating point dtype can require gradients
```

**Solution:** Use floating point tensors for autograd:

```python
from dimtensor.torch import DimTensor
from dimtensor import units
import torch

# Wrong: Integer tensor
# v = DimTensor(torch.tensor([1, 2, 3], requires_grad=True), units.m)

# Right: Float tensor
v = DimTensor(torch.tensor([1.0, 2.0, 3.0], requires_grad=True), units.m)
```

### Device mismatch

**Error:**
```
RuntimeError: Expected all tensors to be on the same device
```

**Solution:** Move all tensors to the same device:

```python
from dimtensor.torch import DimTensor
from dimtensor import units
import torch

a = DimTensor(torch.tensor([1.0, 2.0]), units.m).cuda()
b = DimTensor(torch.tensor([3.0, 4.0]), units.m)  # On CPU

# Wrong: Mixed devices
# a + b  # RuntimeError!

# Right: Same device
b = b.cuda()
a + b  # Works!
```

## JAX-Specific Errors

### JIT compilation issues

**Error:**
```
jax.errors.TracerArrayConversionError: ...
```

**Cause:** JAX's tracing doesn't work with certain Python operations on traced values.

**Solution:** Ensure your function uses JAX operations:

```python
import jax
import jax.numpy as jnp
from dimtensor.jax import DimArray
from dimtensor import units

# Wrong: Python conditionals on traced values
@jax.jit
def bad_function(x):
    if x.data[0] > 0:  # Python if on traced value
        return x * 2
    return x

# Right: Use JAX conditionals
@jax.jit
def good_function(x):
    return jax.lax.cond(
        x.data[0] > 0,
        lambda: x * 2,
        lambda: x
    )
```

## Uncertainty Errors

### Uncertainty shape mismatch

**Error:**
```
ValueError: uncertainty must have same shape as data
```

**Solution:** Ensure uncertainty array matches data shape:

```python
from dimtensor import DimArray, units

# Wrong: Mismatched shapes
# arr = DimArray([1, 2, 3], units.m, uncertainty=[0.1, 0.2])

# Right: Same shape
arr = DimArray([1, 2, 3], units.m, uncertainty=[0.1, 0.2, 0.3])
```

## I/O Errors

### File format not recognized

**Error:**
```
ValueError: Unknown file format
```

**Solution:** Use the correct I/O function for your file format:

```python
from dimtensor.io import (
    save_json, load_json,
    save_hdf5, load_hdf5,
    save_parquet, load_parquet,
    save_netcdf, load_netcdf,
    to_dataframe, from_dataframe,
    to_xarray, from_xarray
)

# Use the right function for each format
save_json(arr, "data.json")
save_hdf5(arr, "data.h5")
save_parquet(arr, "data.parquet")
save_netcdf(arr, "data.nc")
```

### Missing optional dependency

**Error:**
```
ImportError: HDF5 support requires h5py. Install with: pip install h5py
```

**Solution:** Install the required optional dependency:

```bash
pip install dimtensor[hdf5]    # For HDF5 support
pip install dimtensor[parquet] # For Parquet support
pip install dimtensor[netcdf]  # For NetCDF support
pip install dimtensor[all]     # All optional dependencies
```

## Still Having Issues?

If you can't find your error here:

1. **Check the [FAQ](faq.md)** for common questions
2. **Search [GitHub Issues](https://github.com/marcoloco23/dimtensor/issues)** for similar problems
3. **Open a new issue** with:
   - Your dimtensor version (`pip show dimtensor`)
   - Python version
   - Minimal code to reproduce the error
   - Full error traceback

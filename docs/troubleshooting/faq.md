---
title: dimtensor FAQ - Frequently Asked Questions
description: Answers to frequently asked questions about dimtensor, the Python library for unit-aware tensors.
---

# Frequently Asked Questions

## General

### What is dimtensor?

dimtensor is a Python library that adds physical unit tracking to NumPy, PyTorch, and JAX arrays. It catches dimensional errors at operation time, preventing bugs that would otherwise only be discovered after expensive computations.

### How does dimtensor compare to Pint?

See our [detailed comparison](../comparisons/vs-pint.md). The key differences:

- **dimtensor** has native PyTorch/JAX support with autograd
- **Pint** has more flexible unit customization

### Is dimtensor production-ready?

Yes. dimtensor is at version 1.x with comprehensive test coverage, type hints, and stable API.

## Units

### What units are available?

dimtensor includes:

- **SI base units**: m, kg, s, A, K, mol, cd
- **SI derived units**: N, J, W, Pa, Hz, V, C, F, etc.
- **Common units**: km, cm, mm, g, mg, min, hr, etc.
- **Domain-specific units**:
  - Astronomy: parsec, AU, light_year, solar_mass
  - Chemistry: molar, dalton, ppm
  - Engineering: MPa, hp, BTU, kWh

```python
from dimtensor import units
from dimtensor.domains.astronomy import parsec
from dimtensor.domains.chemistry import molar
from dimtensor.domains.engineering import MPa
```

### How do I create custom units?

Currently, dimtensor focuses on standard scientific units. For highly custom unit systems, consider using Pint alongside dimtensor.

### How do I convert between units?

Use the `.to()` method:

```python
from dimtensor import DimArray, units

distance = DimArray([1000], units.m)
in_km = distance.to(units.km)  # 1 km
in_miles = distance.to(units.mile)  # ~0.621 mi
```

### Why can't I convert meters to seconds?

Units can only be converted to other units with the same dimensions. Meters (length) and seconds (time) measure different physical quantities. This is a feature, not a bug - it prevents physically meaningless conversions.

## Performance

### What's the performance overhead?

dimtensor adds approximately 2-5x overhead compared to raw NumPy for typical operations. This overhead comes from:

- Unit checking during operations
- Creating new DimArray wrappers for results

For most scientific applications, this overhead is negligible compared to the bugs it prevents.

### How can I improve performance?

1. **Use larger arrays** - Overhead is relatively smaller for bigger arrays
2. **Batch operations** - Fewer operations means less overhead
3. **Use GPU** - For PyTorch, GPU operations amortize the overhead
4. **Strip units in tight loops** - For performance-critical inner loops, extract raw data:

```python
# For tight loops, extract raw data
raw_data = arr.data
# ... perform operations on raw_data ...
result = DimArray(raw_data, arr.unit)
```

### Does dimtensor support GPU?

Yes, via PyTorch integration:

```python
from dimtensor.torch import DimTensor
from dimtensor import units
import torch

arr = DimTensor(torch.randn(1000, 1000), units.m)
arr_gpu = arr.cuda()  # Units preserved on GPU
```

## Integration

### Can I use dimtensor with pandas?

Yes, dimtensor supports pandas DataFrame conversion:

```python
from dimtensor import DimArray, units
from dimtensor.io import to_dataframe, from_dataframe

data = {
    "position": DimArray([1, 2, 3], units.m),
    "velocity": DimArray([10, 20, 30], units.m / units.s)
}
df = to_dataframe(data)
arrays = from_dataframe(df)
```

### Can I use dimtensor with xarray?

Yes:

```python
from dimtensor import DimArray, units
from dimtensor.io import to_xarray, from_xarray

arr = DimArray([1.0, 2.0, 3.0], units.m)
da = to_xarray(arr, name="distance", dims=("x",))
restored = from_xarray(da)
```

### Does dimtensor work with NumPy functions?

Yes, most NumPy functions work with DimArray:

```python
import numpy as np
from dimtensor import DimArray, units

arr = DimArray([1, 4, 9], units.m**2)
result = np.sqrt(arr)  # Returns DimArray with units m

# Functions requiring dimensionless input
angle = DimArray([0, np.pi/2, np.pi], units.dimensionless)
np.sin(angle)  # Works
```

### Does JAX JIT work with dimtensor?

Yes, DimArray is registered as a JAX pytree:

```python
import jax
from dimtensor.jax import DimArray
from dimtensor import units

@jax.jit
def kinetic_energy(m, v):
    return 0.5 * m * v**2

m = DimArray([1.0], units.kg)
v = DimArray([10.0], units.m / units.s)
E = kinetic_energy(m, v)  # JIT works, units preserved
```

## Uncertainty

### How does uncertainty propagation work?

dimtensor propagates uncertainties in quadrature through all operations:

```python
from dimtensor import DimArray, units

x = DimArray([10.0], units.m, uncertainty=[0.1])
y = DimArray([5.0], units.m, uncertainty=[0.2])

# Addition: uncertainty adds in quadrature
z = x + y  # 15.0 +/- 0.224 m

# Multiplication: relative uncertainties add in quadrature
a = x * y  # 50.0 +/- 1.12 m^2
```

### Can I disable uncertainty propagation?

Create arrays without the `uncertainty` parameter:

```python
# With uncertainty
arr = DimArray([10.0], units.m, uncertainty=[0.1])

# Without uncertainty (faster)
arr = DimArray([10.0], units.m)
```

## I/O

### What file formats are supported?

- JSON
- HDF5 (requires `h5py`)
- Parquet (requires `pyarrow`)
- NetCDF (requires `netcdf4`)
- pandas DataFrame
- xarray DataArray

### How do I install optional dependencies?

```bash
pip install dimtensor[hdf5]    # HDF5 support
pip install dimtensor[parquet] # Parquet support
pip install dimtensor[netcdf]  # NetCDF support
pip install dimtensor[all]     # All dependencies
```

### Are units preserved when saving/loading?

Yes, all I/O functions preserve units:

```python
from dimtensor import DimArray, units
from dimtensor.io import save_json, load_json

arr = DimArray([1, 2, 3], units.m / units.s)
save_json(arr, "velocity.json")
loaded = load_json("velocity.json")
print(loaded.unit)  # m/s
```

## Troubleshooting

### Why am I getting DimensionError?

This means you're trying to combine quantities with incompatible dimensions (like adding meters to seconds). This is dimtensor working correctly - it's catching a physics error. See [Common Errors](errors.md) for solutions.

### Why doesn't my custom NumPy function work?

Some NumPy functions aren't supported yet. You can:

1. Extract raw data, apply the function, and reconstruct:
   ```python
   result_data = custom_function(arr.data)
   result = DimArray(result_data, expected_unit)
   ```

2. [Open an issue](https://github.com/marcoloco23/dimtensor/issues) requesting support for that function.

### Where can I get help?

1. Check this FAQ and [Common Errors](errors.md)
2. Search [GitHub Issues](https://github.com/marcoloco23/dimtensor/issues)
3. Open a new issue with your question
4. Join [GitHub Discussions](https://github.com/marcoloco23/dimtensor/discussions)

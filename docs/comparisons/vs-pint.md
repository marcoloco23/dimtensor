---
title: dimtensor vs Pint - Python Unit Libraries Compared
description: Comprehensive comparison of dimtensor and Pint for unit-aware computation in Python. Learn when to use each library.
---

# dimtensor vs Pint

[Pint](https://pint.readthedocs.io/) is the most popular Python library for physical units. This page compares it with dimtensor to help you choose the right tool.

## TL;DR

| Feature | dimtensor | Pint |
|---------|-----------|------|
| PyTorch integration | Native autograd | Limited wrapping |
| JAX integration | Native pytree | None |
| GPU support | CUDA, MPS | No |
| Uncertainty | Built-in | Via `pint[uncertainties]` |
| Unit definitions | SI + domains | Highly customizable |
| Performance | 2-5x NumPy | Similar |

## When to Choose dimtensor

**Choose dimtensor if you need:**

- Machine learning with PyTorch or JAX
- Automatic differentiation through unit-aware operations
- GPU acceleration (CUDA or Apple Silicon)
- Built-in uncertainty propagation
- Multiple I/O formats (HDF5, NetCDF, Parquet)

```python
# dimtensor: Native PyTorch autograd
from dimtensor.torch import DimTensor
from dimtensor import units
import torch

v = DimTensor(torch.tensor([1.0], requires_grad=True), units.m / units.s)
t = DimTensor(torch.tensor([2.0]), units.s)
d = v * t
d.backward()  # Gradients flow through
print(v.grad)  # Works!
```

## When to Choose Pint

**Choose Pint if you need:**

- Complex custom unit systems
- Non-SI units with unusual conversions
- Mature ecosystem with extensive documentation
- Pandas integration via `pint-pandas`
- Maximum flexibility in unit definitions

```python
# Pint: Custom unit systems
import pint
ureg = pint.UnitRegistry()

# Define custom units
ureg.define('smoot = 1.7018 * meter')
distance = 100 * ureg.smoot
print(distance.to('meter'))  # 170.18 meter
```

## Feature Comparison

### NumPy Integration

Both libraries wrap NumPy arrays effectively:

=== "dimtensor"

    ```python
    from dimtensor import DimArray, units
    import numpy as np

    arr = DimArray(np.array([1, 2, 3]), units.m)
    result = np.sqrt(arr)  # Returns DimArray with m^0.5
    ```

=== "Pint"

    ```python
    import pint
    import numpy as np

    ureg = pint.UnitRegistry()
    arr = np.array([1, 2, 3]) * ureg.meter
    result = np.sqrt(arr)  # Returns Quantity with m^0.5
    ```

### Machine Learning

dimtensor provides native PyTorch and JAX support. Pint requires workarounds:

=== "dimtensor"

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

=== "Pint"

    ```python
    # Pint doesn't support JAX
    # For PyTorch, you must strip units before operations
    # and manually track dimensions
    ```

### Uncertainty Propagation

=== "dimtensor"

    ```python
    from dimtensor import DimArray, units

    # Built-in uncertainty support
    length = DimArray([10.0], units.m, uncertainty=[0.1])
    area = length ** 2  # Uncertainty propagates automatically
    ```

=== "Pint"

    ```python
    import pint
    from uncertainties import ufloat

    ureg = pint.UnitRegistry()
    # Requires uncertainties package
    length = ufloat(10.0, 0.1) * ureg.meter
    area = length ** 2
    ```

### I/O Support

| Format | dimtensor | Pint |
|--------|-----------|------|
| JSON | Yes | Manual |
| HDF5 | Yes | Manual |
| Parquet | Yes | No |
| NetCDF | Yes | No |
| pandas | Yes | pint-pandas |
| xarray | Yes | pint-xarray |

## Migration from Pint

If you're moving from Pint to dimtensor:

```python
# Pint code
import pint
ureg = pint.UnitRegistry()
velocity = 10 * ureg.meter / ureg.second
time = 5 * ureg.second
distance = velocity * time

# dimtensor equivalent
from dimtensor import DimArray, units
velocity = DimArray([10], units.m / units.s)
time = DimArray([5], units.s)
distance = velocity * time
```

Key differences:
- dimtensor uses `DimArray()` constructor instead of multiplication
- Units accessed via `units.m` instead of `ureg.meter`
- Both use `/` and `**` for compound units

## Performance

Both libraries have similar overhead for NumPy operations (2-5x raw NumPy). dimtensor's advantage appears in:

- **GPU operations** - Native CUDA support
- **Autograd** - No unit stripping/reattaching overhead
- **Batch operations** - Better optimization for large arrays

## Conclusion

- **Use dimtensor** for ML, GPU computing, or projects needing PyTorch/JAX
- **Use Pint** for complex unit systems, maximum flexibility, or pure data analysis

Both are excellent libraries. The choice depends on your specific requirements.

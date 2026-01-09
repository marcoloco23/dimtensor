---
title: Migration Guide - Moving to dimtensor
description: How to migrate your Python code from Pint, Astropy units, unyt, or raw NumPy to dimtensor.
---

# Migration Guide

This guide helps you migrate existing code to dimtensor from other libraries or raw NumPy.

## From Raw NumPy

If you're adding units to existing NumPy code:

### Before (Raw NumPy)

```python
import numpy as np

# No unit safety - bugs can go unnoticed
velocity = np.array([10, 20, 30])  # m/s (implicit)
time = np.array([1, 2, 3])          # s (implicit)
distance = velocity * time          # Hopefully correct...

# This bug compiles fine but is physically wrong
acceleration = np.array([9.8])      # m/s^2 (implicit)
wrong = velocity + acceleration     # No error! But meaningless.
```

### After (dimtensor)

```python
from dimtensor import DimArray, units

# Units are explicit and checked
velocity = DimArray([10, 20, 30], units.m / units.s)
time = DimArray([1, 2, 3], units.s)
distance = velocity * time  # [10 40 90] m

# Bug caught immediately
acceleration = DimArray([9.8], units.m / units.s**2)
velocity + acceleration  # DimensionError: cannot add m/s to m/s^2
```

### Migration Steps

1. **Import dimtensor**
   ```python
   from dimtensor import DimArray, units
   ```

2. **Wrap arrays with units**
   ```python
   # Before
   data = np.array([1, 2, 3])

   # After
   data = DimArray([1, 2, 3], units.m)
   ```

3. **Access raw data when needed**
   ```python
   raw = data.data  # Get underlying numpy array
   ```

4. **Update function signatures**
   ```python
   # Before
   def calculate_energy(mass, velocity):
       return 0.5 * mass * velocity**2

   # After (works the same, but now type-safe)
   def calculate_energy(mass: DimArray, velocity: DimArray) -> DimArray:
       return 0.5 * mass * velocity**2
   ```

---

## From Pint

### Key Differences

| Pint | dimtensor |
|------|-----------|
| `ureg = pint.UnitRegistry()` | `from dimtensor import units` |
| `10 * ureg.meter` | `DimArray([10], units.m)` |
| `value.magnitude` | `arr.data` |
| `value.units` | `arr.unit` |
| `value.to('km')` | `arr.to(units.km)` |

### Before (Pint)

```python
import pint

ureg = pint.UnitRegistry()

velocity = 10 * ureg.meter / ureg.second
time = 5 * ureg.second
distance = velocity * time

# Convert units
distance_km = distance.to('kilometer')

# Get magnitude
value = distance.magnitude
```

### After (dimtensor)

```python
from dimtensor import DimArray, units

velocity = DimArray([10], units.m / units.s)
time = DimArray([5], units.s)
distance = velocity * time

# Convert units
distance_km = distance.to(units.km)

# Get raw data
value = distance.data
```

### Migration Script

```python
# Quick find-and-replace patterns:
# ureg.meter -> units.m
# ureg.second -> units.s
# ureg.kilogram -> units.kg
# .magnitude -> .data
# .units -> .unit
# * ureg.unit -> DimArray([...], units.unit)
```

---

## From Astropy

### Key Differences

| Astropy | dimtensor |
|---------|-----------|
| `from astropy import units as u` | `from dimtensor import units` |
| `10 * u.m` | `DimArray([10], units.m)` |
| `value.value` | `arr.data` |
| `value.unit` | `arr.unit` |
| `value.to(u.km)` | `arr.to(units.km)` |

### Before (Astropy)

```python
from astropy import units as u
from astropy import constants as const

mass = 1.0 * u.kg
velocity = 10 * u.m / u.s
energy = 0.5 * mass * velocity**2

# Physical constants
c = const.c
E = mass * c**2
```

### After (dimtensor)

```python
from dimtensor import DimArray, units, constants

mass = DimArray([1.0], units.kg)
velocity = DimArray([10], units.m / units.s)
energy = 0.5 * mass * velocity**2

# Physical constants
c = constants.c
E = mass * c**2
```

### Astronomy Units

```python
# Astropy
from astropy import units as u
distance = 10 * u.pc

# dimtensor
from dimtensor import DimArray
from dimtensor.domains.astronomy import parsec
distance = DimArray([10], parsec)
```

---

## From unyt

### Key Differences

| unyt | dimtensor |
|------|-----------|
| `unyt_array([1,2,3], 'm')` | `DimArray([1,2,3], units.m)` |
| `arr.v` or `arr.value` | `arr.data` |
| `arr.units` | `arr.unit` |
| `arr.to('km')` | `arr.to(units.km)` |

### Before (unyt)

```python
from unyt import unyt_array

velocity = unyt_array([10, 20, 30], 'm/s')
time = unyt_array([1, 2, 3], 's')
distance = velocity * time

# Get values
values = velocity.v
```

### After (dimtensor)

```python
from dimtensor import DimArray, units

velocity = DimArray([10, 20, 30], units.m / units.s)
time = DimArray([1, 2, 3], units.s)
distance = velocity * time

# Get values
values = velocity.data
```

---

## PyTorch Migration

If you're adding units to PyTorch code:

### Before (Raw PyTorch)

```python
import torch

mass = torch.tensor([1.0, 2.0], requires_grad=True)
velocity = torch.tensor([10.0, 20.0])
energy = 0.5 * mass * velocity**2

loss = energy.sum()
loss.backward()
```

### After (dimtensor)

```python
import torch
from dimtensor.torch import DimTensor
from dimtensor import units

mass = DimTensor(torch.tensor([1.0, 2.0], requires_grad=True), units.kg)
velocity = DimTensor(torch.tensor([10.0, 20.0]), units.m / units.s)
energy = 0.5 * mass * velocity**2  # Returns Joules

loss = energy.sum()
loss.backward()  # Gradients work!
```

---

## JAX Migration

### Before (Raw JAX)

```python
import jax
import jax.numpy as jnp

@jax.jit
def kinetic_energy(mass, velocity):
    return 0.5 * mass * velocity**2

m = jnp.array([1.0, 2.0])
v = jnp.array([10.0, 20.0])
E = kinetic_energy(m, v)
```

### After (dimtensor)

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
E = kinetic_energy(m, v)  # JIT works, returns Joules
```

---

## Common Patterns

### Wrapping External Data

```python
import numpy as np
from dimtensor import DimArray, units

# Data from file, API, etc.
raw_data = np.loadtxt("measurements.csv")

# Add units based on what you know about the data
temperatures = DimArray(raw_data[:, 0], units.K)
pressures = DimArray(raw_data[:, 1], units.Pa)
```

### Interfacing with Unit-Unaware Libraries

```python
from dimtensor import DimArray, units
import scipy.optimize

def objective(x_raw):
    # Convert raw values back to DimArray for physics
    x = DimArray(x_raw, units.m)
    # ... do unit-aware calculation ...
    return result.data  # Return raw for scipy

# scipy needs raw arrays
x0 = initial_guess.data
result = scipy.optimize.minimize(objective, x0)
```

### Gradual Migration

You don't have to convert everything at once:

```python
from dimtensor import DimArray, units
import numpy as np

# Mix of old and new code
old_array = np.array([1, 2, 3])  # Legacy code
new_array = DimArray([4, 5, 6], units.m)  # New code

# Convert when needed
old_with_units = DimArray(old_array, units.m)
combined = old_with_units + new_array
```

---

## Getting Help

If you run into issues during migration:

1. Check [Common Errors](errors.md)
2. Check the [FAQ](faq.md)
3. [Open an issue](https://github.com/marcoloco23/dimtensor/issues) with your specific migration question

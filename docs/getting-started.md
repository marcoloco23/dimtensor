# Getting Started

## Installation

Install dimtensor from PyPI:

```bash
pip install dimtensor
```

For development:

```bash
pip install dimtensor[dev]
```

## Basic Usage

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

### Arithmetic Operations

Operations automatically track dimensions:

```python
# Multiplication combines dimensions
force = mass * DimArray([9.8], units.m / units.s**2)
print(force)  # 49.0 N

# Addition requires same dimensions
a = DimArray([1, 2], units.m)
b = DimArray([3, 4], units.m)
print(a + b)  # [4 6] m

# Incompatible dimensions raise errors
c = DimArray([1], units.s)
a + c  # DimensionError!
```

### Unit Conversion

Convert between compatible units:

```python
distance = DimArray([1.0], units.km)

# Convert to meters
in_meters = distance.to(units.m)
print(in_meters)  # [1000.] m

# Convert to miles
in_miles = distance.to(units.mile)
print(in_miles)  # [0.62137119] mi
```

### Accessing Data

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

- [Working with Units](guide/units.md) - Available units and creating custom units
- [Array Operations](guide/operations.md) - Reshaping, linear algebra, and more
- [Examples](guide/examples.md) - Real physics calculations

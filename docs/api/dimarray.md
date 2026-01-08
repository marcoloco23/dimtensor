# DimArray

The core class for dimension-aware arrays.

## Overview

`DimArray` wraps a NumPy array and tracks its physical dimensions through all operations. Operations between incompatible dimensions raise `DimensionError` immediately.

```python
from dimtensor import DimArray, units

# Create a DimArray
velocity = DimArray([10, 20, 30], units.m / units.s)
```

## API Reference

::: dimtensor.DimArray
    options:
      members:
        - __init__
        - data
        - unit
        - dimension
        - shape
        - ndim
        - size
        - dtype
        - is_dimensionless
        - to
        - to_base_units
        - magnitude
        - sum
        - mean
        - std
        - var
        - min
        - max
        - argmin
        - argmax
        - reshape
        - transpose
        - flatten
        - sqrt

# dimtensor

**Unit-aware tensors for physics and scientific machine learning.**

dimtensor wraps NumPy arrays with physical unit tracking, catching dimensional errors at operation time rather than after hours of computation.

## Why dimtensor?

- **Catch errors early**: Don't waste compute on dimensionally invalid calculations
- **Self-documenting code**: Units make physics code clearer
- **NumPy compatible**: Works seamlessly with existing NumPy code
- **Lightweight**: Just metadata tracking, minimal overhead

## Quick Example

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

## Installation

```bash
pip install dimtensor
```

## Features

| Feature | Description |
|---------|-------------|
| **Dimensional Safety** | Operations between incompatible dimensions raise `DimensionError` |
| **Unit Conversion** | Convert between compatible units with `.to()` |
| **SI Units** | Full support for SI base and derived units |
| **Unit Simplification** | `kg*m/s^2` automatically displays as `N` |
| **NumPy Integration** | Works with ufuncs like `np.sin`, `np.sqrt` |
| **Array Functions** | `concatenate`, `stack`, `split`, `dot`, `matmul`, `norm` |

## Next Steps

- [Getting Started](getting-started.md) - Installation and first steps
- [Working with Units](guide/units.md) - Learn about the unit system
- [Examples](guide/examples.md) - Real-world physics calculations
- [API Reference](api/dimarray.md) - Complete API documentation

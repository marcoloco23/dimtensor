# dimtensor

Unit-aware tensors for physics and scientific machine learning.

`dimtensor` wraps NumPy arrays with physical unit tracking, catching dimensional errors at operation time rather than after hours of computation.

## Installation

```bash
pip install dimtensor
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
- **Lightweight**: Minimal overhead, just metadata tracking

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
3. **Designed for ML**: Built with PyTorch/JAX integration in mind (coming soon)
4. **Lightweight**: Just metadata tracking, minimal overhead

## License

MIT

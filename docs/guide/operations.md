# Array Operations

## Arithmetic Operations

All standard arithmetic operations preserve dimensional correctness:

```python
from dimtensor import DimArray, units

a = DimArray([1, 2, 3], units.m)
b = DimArray([4, 5, 6], units.m)

# Addition/subtraction (requires same dimension)
a + b  # [5 7 9] m
a - b  # [-3 -3 -3] m

# Multiplication (dimensions combine)
c = DimArray([2], units.s)
a * c  # [2 4 6] m*s

# Division (dimensions divide)
a / c  # [0.5 1.0 1.5] m/s

# Power (dimension exponentiated)
a ** 2  # [1 4 9] m^2
```

## Reduction Operations

Reduce arrays while preserving units:

```python
arr = DimArray([[1, 2, 3], [4, 5, 6]], units.m)

# Sum
arr.sum()           # [21] m
arr.sum(axis=0)     # [5 7 9] m
arr.sum(axis=1)     # [6 15] m

# Mean
arr.mean()          # [3.5] m

# Min/Max
arr.min()           # [1] m
arr.max()           # [6] m

# Standard deviation (preserves unit)
arr.std()           # [...] m

# Variance (squares the unit)
arr.var()           # [...] m^2
```

## Searching

Find indices of minimum/maximum values:

```python
arr = DimArray([3, 1, 4, 1, 5], units.m)

# Returns plain numpy integers/arrays (not DimArray)
arr.argmin()        # 1
arr.argmax()        # 4

# With axis
arr2d = DimArray([[3, 1], [2, 4]], units.m)
arr2d.argmin(axis=0)  # [1, 0]
arr2d.argmax(axis=1)  # [0, 1]
```

## Reshaping

Reshape operations preserve units:

```python
arr = DimArray([1, 2, 3, 4, 5, 6], units.m)

# Reshape
arr.reshape((2, 3))      # [[1 2 3] [4 5 6]] m
arr.reshape((3, -1))     # [[1 2] [3 4] [5 6]] m

# Transpose
arr2d = DimArray([[1, 2, 3], [4, 5, 6]], units.m)
arr2d.transpose()        # [[1 4] [2 5] [3 6]] m

# Flatten
arr2d.flatten()          # [1 2 3 4 5 6] m
```

## Array Functions

Module-level functions for combining arrays:

```python
from dimtensor import DimArray, units, concatenate, stack, split

a = DimArray([1, 2], units.m)
b = DimArray([3, 4], units.m)

# Concatenate (requires same dimension)
concatenate([a, b])      # [1 2 3 4] m

# Stack (creates new axis)
stack([a, b])            # [[1 2] [3 4]] m
stack([a, b], axis=1)    # [[1 3] [2 4]] m

# Split
arr = DimArray([1, 2, 3, 4], units.m)
split(arr, 2)            # [[1 2] m, [3 4] m]
```

## Linear Algebra

Linear algebra functions with proper dimension handling:

```python
from dimtensor import DimArray, units, dot, matmul, norm

# Dot product (dimensions multiply)
length = DimArray([1, 2, 3], units.m)
force = DimArray([4, 5, 6], units.N)
work = dot(length, force)  # [32] J (m * N = J)

# Matrix multiplication
A = DimArray([[1, 2], [3, 4]], units.m)
B = DimArray([[5, 6], [7, 8]], units.s)
matmul(A, B)  # Result has dimension m*s

# Norm (preserves unit)
velocity = DimArray([3, 4], units.m / units.s)
speed = norm(velocity)  # [5] m/s
```

## NumPy ufunc Integration

Use NumPy functions directly:

```python
import numpy as np
from dimtensor import DimArray, units

# Trigonometric functions (require dimensionless)
angle = DimArray([0, 3.14159/2], units.rad)
np.sin(angle)  # [0, 1]
np.cos(angle)  # [1, 0]

# Exponential/log (require dimensionless)
x = DimArray([0, 1, 2], units.rad)  # dimensionless
np.exp(x)
np.log(x + 1)

# sqrt (halves dimension exponents)
area = DimArray([4, 9, 16], units.m**2)
np.sqrt(area)  # [2 3 4] m

# abs (preserves unit)
velocity = DimArray([-1, 2, -3], units.m / units.s)
np.abs(velocity)  # [1 2 3] m/s
```

!!! note "Dimensional Requirements"
    Functions like `sin`, `cos`, `exp`, `log` require dimensionless input.
    Attempting to use them with dimensional quantities raises `DimensionError`.

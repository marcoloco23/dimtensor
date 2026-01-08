# Functions

Module-level functions for array operations.

## Array Manipulation

### concatenate

Join arrays along an existing axis.

```python
from dimtensor import DimArray, units, concatenate

a = DimArray([1, 2], units.m)
b = DimArray([3, 4], units.m)
result = concatenate([a, b])  # [1 2 3 4] m
```

All arrays must have the same dimension. Arrays with compatible units are converted to the first array's unit.

::: dimtensor.concatenate

### stack

Stack arrays along a new axis.

```python
from dimtensor import DimArray, units, stack

a = DimArray([1, 2], units.m)
b = DimArray([3, 4], units.m)
result = stack([a, b])  # [[1 2] [3 4]] m
```

::: dimtensor.stack

### split

Split an array into sub-arrays.

```python
from dimtensor import DimArray, units, split

arr = DimArray([1, 2, 3, 4], units.m)
parts = split(arr, 2)  # [[1 2] m, [3 4] m]
```

::: dimtensor.split

## Linear Algebra

### dot

Dot product of two arrays. Dimensions multiply.

```python
from dimtensor import DimArray, units, dot

# Work = Force . Displacement
force = DimArray([10, 0, 0], units.N)
displacement = DimArray([5, 0, 0], units.m)
work = dot(force, displacement)  # [50] J
```

::: dimtensor.dot

### matmul

Matrix multiplication. Dimensions multiply.

```python
from dimtensor import DimArray, units, matmul

A = DimArray([[1, 2], [3, 4]], units.m)
B = DimArray([[5, 6], [7, 8]], units.s)
C = matmul(A, B)  # Result has unit m*s
```

::: dimtensor.matmul

### norm

Compute the norm of an array. Preserves the original unit.

```python
from dimtensor import DimArray, units, norm

velocity = DimArray([3, 4], units.m / units.s)
speed = norm(velocity)  # [5] m/s
```

::: dimtensor.norm

## Exceptions

### DimensionError

Raised when dimensions are incompatible for an operation.

```python
from dimtensor import DimArray, DimensionError, units

try:
    a = DimArray([1], units.m)
    b = DimArray([1], units.s)
    a + b  # Cannot add m to s
except DimensionError as e:
    print(e)  # "Cannot add quantities with dimensions L and T"
```

::: dimtensor.DimensionError

### UnitConversionError

Raised when unit conversion is not possible.

```python
from dimtensor import DimArray, UnitConversionError, units

try:
    distance = DimArray([1], units.m)
    distance.to(units.s)  # Cannot convert m to s
except UnitConversionError as e:
    print(e)
```

::: dimtensor.UnitConversionError

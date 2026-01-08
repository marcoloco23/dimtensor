# Working with Units

## Available Units

dimtensor provides a comprehensive set of SI and common non-SI units.

### SI Base Units

| Unit | Symbol | Dimension |
|------|--------|-----------|
| `units.m` | m | Length |
| `units.kg` | kg | Mass |
| `units.s` | s | Time |
| `units.A` | A | Electric current |
| `units.K` | K | Temperature |
| `units.mol` | mol | Amount of substance |
| `units.cd` | cd | Luminous intensity |

### SI Derived Units

| Unit | Symbol | Definition |
|------|--------|------------|
| `units.N` | N | Newton (kg*m/s^2) |
| `units.J` | J | Joule (kg*m^2/s^2) |
| `units.W` | W | Watt (kg*m^2/s^3) |
| `units.Pa` | Pa | Pascal (kg/m/s^2) |
| `units.Hz` | Hz | Hertz (1/s) |
| `units.V` | V | Volt |
| `units.ohm` | ohm | Ohm |
| `units.C` | C | Coulomb |

### Common Non-SI Units

| Unit | Symbol | Relation to SI |
|------|--------|----------------|
| `units.km` | km | 1000 m |
| `units.cm` | cm | 0.01 m |
| `units.mm` | mm | 0.001 m |
| `units.mile` | mi | 1609.344 m |
| `units.hour` | h | 3600 s |
| `units.minute` | min | 60 s |
| `units.eV` | eV | 1.602e-19 J |
| `units.atm` | atm | 101325 Pa |
| `units.rad` | rad | Dimensionless |
| `units.deg` | deg | pi/180 rad |

## Creating Compound Units

Build complex units using arithmetic:

```python
from dimtensor import units

# Velocity
velocity_unit = units.m / units.s

# Acceleration
accel_unit = units.m / units.s**2

# Energy density
energy_density = units.J / units.m**3

# Pressure (alternative)
pressure = units.kg / (units.m * units.s**2)
```

## Unit Simplification

dimtensor automatically simplifies compound units to their SI derived equivalents:

```python
from dimtensor import DimArray, units

# Force calculation
mass = DimArray([2.0], units.kg)
accel = DimArray([9.8], units.m / units.s**2)
force = mass * accel

print(force)  # [19.6] N  (not kg*m/s^2)

# Energy calculation
distance = DimArray([10.0], units.m)
work = force * distance

print(work)  # [196.] J  (not kg*m^2/s^2)
```

## Unit Conversion

Convert between compatible units with `.to()`:

```python
# Length conversions
distance = DimArray([5.0], units.km)
print(distance.to(units.m))     # [5000.] m
print(distance.to(units.mile))  # [3.10685596] mi

# Time conversions
time = DimArray([3600.0], units.s)
print(time.to(units.hour))  # [1.] h

# Energy conversions
energy = DimArray([1.0], units.J)
print(energy.to(units.eV))  # [6.242e+18] eV
```

!!! warning "Incompatible Conversions"
    Attempting to convert between incompatible dimensions raises `UnitConversionError`:

    ```python
    distance = DimArray([1.0], units.m)
    distance.to(units.s)  # UnitConversionError!
    ```

## Dimensionless Quantities

Some quantities have no physical dimension:

```python
# Angles
angle = DimArray([3.14159], units.rad)
print(angle.is_dimensionless)  # True

# Ratios
ratio = distance1 / distance2  # Dimensionless

# Works with trig functions
import numpy as np
np.sin(angle)  # Works because angle is dimensionless
```

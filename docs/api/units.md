# Units

Physical units for dimensional analysis.

## Usage

```python
from dimtensor import units

# Access units
units.m      # meter
units.kg     # kilogram
units.s      # second

# Create compound units
velocity = units.m / units.s
acceleration = units.m / units.s**2
force = units.kg * units.m / units.s**2  # Same as units.N
```

## SI Base Units

| Attribute | Symbol | Dimension |
|-----------|--------|-----------|
| `units.m` | m | Length |
| `units.kg` | kg | Mass |
| `units.s` | s | Time |
| `units.A` | A | Electric current |
| `units.K` | K | Temperature |
| `units.mol` | mol | Amount of substance |
| `units.cd` | cd | Luminous intensity |

## SI Derived Units

| Attribute | Symbol | Equivalent |
|-----------|--------|------------|
| `units.N` | N | kg*m/s^2 |
| `units.J` | J | kg*m^2/s^2 |
| `units.W` | W | kg*m^2/s^3 |
| `units.Pa` | Pa | kg/(m*s^2) |
| `units.Hz` | Hz | 1/s |
| `units.C` | C | A*s |
| `units.V` | V | kg*m^2/(A*s^3) |
| `units.F` | F | A^2*s^4/(kg*m^2) |
| `units.ohm` | ohm | kg*m^2/(A^2*s^3) |

## Length Units

| Attribute | Symbol | SI Value |
|-----------|--------|----------|
| `units.m` | m | 1 m |
| `units.km` | km | 1000 m |
| `units.cm` | cm | 0.01 m |
| `units.mm` | mm | 0.001 m |
| `units.um` | um | 1e-6 m |
| `units.nm` | nm | 1e-9 m |
| `units.mile` | mi | 1609.344 m |
| `units.ft` | ft | 0.3048 m |
| `units.inch` | in | 0.0254 m |

## Time Units

| Attribute | Symbol | SI Value |
|-----------|--------|----------|
| `units.s` | s | 1 s |
| `units.ms` | ms | 0.001 s |
| `units.us` | us | 1e-6 s |
| `units.ns` | ns | 1e-9 s |
| `units.minute` | min | 60 s |
| `units.hour` | h | 3600 s |
| `units.day` | d | 86400 s |

## Energy Units

| Attribute | Symbol | SI Value |
|-----------|--------|----------|
| `units.J` | J | 1 J |
| `units.kJ` | kJ | 1000 J |
| `units.eV` | eV | 1.602e-19 J |
| `units.cal` | cal | 4.184 J |
| `units.kcal` | kcal | 4184 J |

## Pressure Units

| Attribute | Symbol | SI Value |
|-----------|--------|----------|
| `units.Pa` | Pa | 1 Pa |
| `units.kPa` | kPa | 1000 Pa |
| `units.bar` | bar | 1e5 Pa |
| `units.atm` | atm | 101325 Pa |

## Angle Units

| Attribute | Symbol | Note |
|-----------|--------|------|
| `units.rad` | rad | Dimensionless |
| `units.deg` | deg | pi/180 rad |

## API Reference

::: dimtensor.Unit
    options:
      members:
        - symbol
        - dimension
        - scale

::: dimtensor.Dimension
    options:
      members:
        - length
        - mass
        - time
        - current
        - temperature
        - amount
        - luminosity

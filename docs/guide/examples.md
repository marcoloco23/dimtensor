# Examples

Real-world physics calculations with dimtensor.

## Mechanics

### Kinematics

```python
from dimtensor import DimArray, units

# Initial velocity and acceleration
v0 = DimArray([10], units.m / units.s)
a = DimArray([2], units.m / units.s**2)
t = DimArray([5], units.s)

# Final velocity: v = v0 + a*t
v = v0 + a * t
print(f"Final velocity: {v}")  # 20.0 m/s

# Distance: d = v0*t + 0.5*a*t^2
d = v0 * t + 0.5 * a * t**2
print(f"Distance: {d}")  # 75.0 m
```

### Force and Energy

```python
from dimtensor import DimArray, units, dot, norm

# Mass and gravity
m = DimArray([2.0], units.kg)
g = DimArray([9.8], units.m / units.s**2)

# Weight (force)
F = m * g
print(f"Weight: {F}")  # 19.6 N

# Potential energy at height h
h = DimArray([10.0], units.m)
PE = m * g * h
print(f"Potential energy: {PE}")  # 196.0 J

# Kinetic energy
v = DimArray([5.0], units.m / units.s)
KE = 0.5 * m * v**2
print(f"Kinetic energy: {KE}")  # 25.0 J
```

### Projectile Motion

```python
import numpy as np
from dimtensor import DimArray, units, norm

# Launch parameters
v0 = DimArray([20], units.m / units.s)  # initial speed
theta = DimArray([45 * np.pi / 180], units.rad)  # launch angle
g = DimArray([9.8], units.m / units.s**2)

# Initial velocity components
vx = v0 * np.cos(theta)
vy = v0 * np.sin(theta)

# Time of flight
t_flight = 2 * vy / g
print(f"Time of flight: {t_flight.to(units.s)}")

# Maximum height
h_max = vy**2 / (2 * g)
print(f"Maximum height: {h_max}")

# Range
R = vx * t_flight
print(f"Range: {R}")
```

## Electricity

### Ohm's Law

```python
from dimtensor import DimArray, units

# Voltage and current
V = DimArray([12.0], units.V)
I = DimArray([2.0], units.A)

# Resistance: R = V/I
R = V / I
print(f"Resistance: {R}")  # 6.0 ohm

# Power: P = V*I
P = V * I
print(f"Power: {P}")  # 24.0 W
```

### RC Circuit

```python
from dimtensor import DimArray, units
import numpy as np

# Circuit parameters
R = DimArray([1000], units.ohm)  # 1 kOhm
C = DimArray([1e-6], units.F)     # 1 uF
V0 = DimArray([5.0], units.V)     # Initial voltage

# Time constant
tau = R * C
print(f"Time constant: {tau.to(units.s)}")  # 0.001 s = 1 ms

# Voltage decay at t = 2*tau
t = 2 * tau
V_t = V0 * np.exp((-t / tau).to(units.rad))  # dimensionless exponent
print(f"Voltage at t=2*tau: {V_t}")
```

## Thermodynamics

### Ideal Gas Law

```python
from dimtensor import DimArray, units

# Gas constant (you would define this)
R_gas = DimArray([8.314], units.J / (units.mol * units.K))

# State variables
n = DimArray([1.0], units.mol)
T = DimArray([300], units.K)
V = DimArray([0.025], units.m**3)

# Pressure: P = nRT/V
P = n * R_gas * T / V
print(f"Pressure: {P.to(units.Pa)}")
print(f"Pressure: {P.to(units.atm)}")
```

## Working with Vectors

### 3D Velocity and Speed

```python
from dimtensor import DimArray, units, norm, dot

# 3D velocity vector
velocity = DimArray([3.0, 4.0, 0.0], units.m / units.s)

# Speed (magnitude)
speed = norm(velocity)
print(f"Speed: {speed}")  # 5.0 m/s

# Kinetic energy
m = DimArray([2.0], units.kg)
KE = 0.5 * m * dot(velocity, velocity)
print(f"KE: {KE}")  # 25.0 J
```

### Work Done by Force

```python
from dimtensor import DimArray, units, dot

# Force vector
F = DimArray([10.0, 5.0, 0.0], units.N)

# Displacement vector
d = DimArray([3.0, 4.0, 0.0], units.m)

# Work: W = F . d
W = dot(F, d)
print(f"Work done: {W}")  # 50.0 J
```

## Unit Conversions

### Distance and Speed

```python
from dimtensor import DimArray, units

# Marathon distance
marathon = DimArray([42.195], units.km)
print(f"Marathon: {marathon.to(units.m)}")      # 42195 m
print(f"Marathon: {marathon.to(units.mile)}")   # ~26.2 mi

# Speed conversion
speed_ms = DimArray([10], units.m / units.s)
speed_kmh = speed_ms.to(units.km / units.hour)
print(f"Speed: {speed_kmh}")  # 36 km/h
```

### Energy Units

```python
from dimtensor import DimArray, units

# Energy in different units
energy_J = DimArray([1.0], units.J)
energy_eV = energy_J.to(units.eV)
print(f"1 J = {energy_eV} eV")  # ~6.24e18 eV
```

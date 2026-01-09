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

## Physics-Informed Machine Learning

### DimTensor: PyTorch Integration

```python
import torch
from dimtensor.torch import DimTensor
from dimtensor import units

# Create DimTensor with autograd support
position = DimTensor(torch.tensor([1.0, 2.0, 3.0], requires_grad=True), units.m)
time = DimTensor(torch.tensor([0.5, 1.0, 1.5]), units.s)

# Compute velocity (dimensions checked automatically)
velocity = position / time
print(f"Velocity: {velocity}")  # [2.0, 2.0, 2.0] m/s

# Backpropagation preserves units
loss = (velocity ** 2).sum()
loss.backward()
print(f"Gradient shape: {position.grad.shape}")
```

### Dimension-Aware Neural Network Layers

```python
import torch
import torch.nn as nn
from dimtensor.torch import DimTensor, DimLinear, DimSequential
from dimtensor import units, Dimension

# Define dimensions
L = Dimension(length=1)           # position [m]
V = Dimension(length=1, time=-1)  # velocity [m/s]
A = Dimension(length=1, time=-2)  # acceleration [m/s²]

# Build a physics-informed network
# Input: position -> Hidden: velocity -> Output: acceleration
model = DimSequential(
    DimLinear(in_features=3, out_features=16, input_dim=L, output_dim=V),
    DimLinear(in_features=16, out_features=3, input_dim=V, output_dim=A)
)

# Input with position units
x = DimTensor(torch.randn(10, 3), units.m)

# Forward pass - output automatically has acceleration units
acceleration = model(x)
print(f"Output dimension: {acceleration.dimension}")  # L=1, T=-2

# Dimension errors caught at runtime!
# model(DimTensor(torch.randn(10, 3), units.s))  # Raises DimensionError!
```

### Physics-Informed Convolutional Layers

```python
import torch
from dimtensor.torch import DimTensor, DimConv2d
from dimtensor import units, Dimension

# Process a temperature field (e.g., heat equation simulation)
conv = DimConv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=3,
    input_dim=Dimension(temperature=1),   # Temperature [K]
    output_dim=Dimension(temperature=1),  # Temperature [K]
)

# Temperature field: 16 samples, 1 channel, 64x64 grid
T_field = DimTensor(torch.randn(16, 1, 64, 64) * 300, units.K)

# Convolve - output maintains temperature dimension
T_features = conv(T_field)
print(f"Output shape: {T_features.shape}")      # (16, 8, 62, 62)
print(f"Output unit: {T_features.unit.symbol}") # K
```

## Uncertainty Propagation

### Automatic Error Propagation

```python
from dimtensor import DimArray, units
import numpy as np

# Measurements with uncertainty
length = DimArray([10.0], units.m, uncertainty=[0.1])  # ±0.1 m
width = DimArray([5.0], units.m, uncertainty=[0.05])   # ±0.05 m

# Area calculation - uncertainty propagates automatically
area = length * width
print(f"Area: {area._data[0]:.2f} ± {area.uncertainty[0]:.3f} {area.unit.symbol}")
# Area: 50.00 ± 0.707 m²

# Relative uncertainty
print(f"Relative uncertainty: {area.relative_uncertainty[0]:.2%}")
# ~1.41%
```

### Physics Calculation with Errors

```python
from dimtensor import DimArray, units
import numpy as np

# Pendulum period: T = 2π√(L/g)
L = DimArray([1.0], units.m, uncertainty=[0.01])        # Length ±1 cm
g = DimArray([9.81], units.m/units.s**2, uncertainty=[0.01])  # g ±0.01 m/s²

# Calculate period
T = 2 * np.pi * (L / g) ** 0.5
print(f"Period: {T._data[0]:.3f} ± {T.uncertainty[0]:.4f} {T.unit.symbol}")

# Check uncertainty is tracked
assert T.has_uncertainty
```

## Constraints and Validation

### Value Constraints

```python
from dimtensor import DimArray, units
from dimtensor.validation import Positive, NonNegative, Bounded, Finite

# Mass must be positive
mass = DimArray([2.5, 3.0, 1.8], units.kg)
positive_check = Positive()
positive_check.validate(mass._data)  # OK

# Probability must be in [0, 1]
probability = DimArray([0.3, 0.7, 0.95], units.dimensionless)
bounded_check = Bounded(0, 1)
bounded_check.validate(probability._data)  # OK

# Temperature must be finite (no inf/nan)
temperature = DimArray([273.15, 298.15, 310.0], units.K)
finite_check = Finite()
finite_check.validate(temperature._data)  # OK

# Example: catching invalid values
try:
    invalid_mass = DimArray([-1.0], units.kg)
    positive_check.validate(invalid_mass._data)
except Exception as e:
    print(f"Caught error: {type(e).__name__}")
```

### Physics Constraints

```python
from dimtensor import DimArray, units
from dimtensor.validation import Positive, Bounded

# Energy conservation: ensure all energies are positive
KE = DimArray([100.0, 200.0], units.J)
PE = DimArray([50.0, 150.0], units.J)
total_energy = KE + PE

# Validate total energy is positive and finite
Positive().validate(total_energy._data)
Finite().validate(total_energy._data)

print(f"Total energy: {total_energy} (validated)")
```

## Dataset Loaders

### Loading NIST Physical Constants

```python
from dimtensor.datasets.loaders import NISTCODATALoader

# Download and cache NIST CODATA constants
loader = NISTCODATALoader()
constants = loader.load()

# Access physical constants
c = constants['speed of light in vacuum']
print(f"Speed of light: {c}")

h = constants['Planck constant']
print(f"Planck constant: {h}")
```

### Loading NASA Exoplanet Data

```python
from dimtensor.datasets.loaders import NASAExoplanetLoader

# Load confirmed exoplanet catalog
loader = NASAExoplanetLoader()
exoplanets = loader.load()

# DataFrame with physical units in columns
print(f"Loaded {len(exoplanets)} exoplanets")
print(exoplanets[['pl_name', 'pl_masse', 'pl_rade']].head())
```

### Loading Climate Data

```python
from dimtensor.datasets.loaders import NOAAClimateLoader

# Download NOAA climate indices
loader = NOAAClimateLoader()
climate_data = loader.load()

print(f"Climate data: {climate_data.head()}")
```

## Equation Database

### Searching Physics Equations

```python
from dimtensor.equations.database import search_equations, get_equation, list_domains

# Search for equations about energy
energy_equations = search_equations("energy")
print(f"Found {len(energy_equations)} equations about energy:")
for eq in energy_equations[:3]:
    print(f"  - {eq.name}: {eq.formula}")

# Get a specific equation
newtons_law = get_equation("Newton's Second Law")
print(f"\n{newtons_law.name}")
print(f"Formula: {newtons_law.formula}")
print(f"Variables: {newtons_law.variables}")
print(f"LaTeX: {newtons_law.latex}")

# List all physics domains
domains = list_domains()
print(f"\nAvailable domains: {domains}")
```

### Using Equations for Validation

```python
from dimtensor.equations.database import get_equation
from dimtensor import DimArray, units

# Get kinetic energy equation
ke_eq = get_equation("Kinetic Energy")
print(f"{ke_eq.formula}")  # KE = (1/2)mv^2

# Verify our calculation matches expected dimensions
m = DimArray([2.0], units.kg)
v = DimArray([10.0], units.m / units.s)
KE = 0.5 * m * v**2

# Check dimension matches
expected_dim = ke_eq.variables['KE']
actual_dim = KE.dimension
print(f"Expected: {expected_dim}")
print(f"Actual: {actual_dim}")
assert actual_dim == expected_dim, "Dimension mismatch!"
```

### Browsing Equations by Domain

```python
from dimtensor.equations.database import get_equations

# Get all thermodynamics equations
thermo_eqs = get_equations(domain="thermodynamics")
print(f"Thermodynamics equations ({len(thermo_eqs)}):")
for eq in thermo_eqs:
    print(f"  - {eq.name}: {eq.formula}")

# Get equations with specific tags
fundamental_eqs = get_equations(tags=["fundamental"])
print(f"\nFundamental equations: {len(fundamental_eqs)}")
```

## Unit Inference

### Automatic Unit Inference from Equations

```python
from dimtensor.inference.solver import infer_units
from dimtensor import units

# Given F = m * a, infer F's units from m and a
result = infer_units(
    equation="F = m * a",
    known_units={
        "m": units.kg,
        "a": units.m / units.s**2
    }
)

print(f"Is consistent? {result['is_consistent']}")
print(f"Confidence: {result['confidence']}")
print(f"Inferred F: {result['inferred']['F']}")  # kg·m/s² = N

# More complex equation: E = (1/2) * m * v**2
result = infer_units(
    equation="E = 0.5 * m * v**2",
    known_units={
        "m": units.kg,
        "v": units.m / units.s
    }
)

print(f"\nInferred E: {result['inferred']['E']}")  # kg·m²/s² = J
```

### Detecting Dimensional Errors

```python
from dimtensor.inference.solver import infer_units
from dimtensor import units

# Invalid equation: trying to add length and time
result = infer_units(
    equation="x = L + t",
    known_units={
        "L": units.m,
        "t": units.s
    }
)

print(f"Is consistent? {result['is_consistent']}")  # False
print(f"Errors: {result['errors']}")
```

## Visualization with Matplotlib

### Basic Plotting with Unit Labels

```python
from dimtensor import DimArray, units
from dimtensor.visualization import plot
import matplotlib.pyplot as plt

# Time series data
time = DimArray([0, 1, 2, 3, 4, 5], units.s)
position = DimArray([0, 5, 20, 45, 80, 125], units.m)

# Plot with automatic axis labels
plot(time, position)
plt.title("Position vs Time")
plt.grid(True)
plt.show()
# X-axis automatically labeled: [s]
# Y-axis automatically labeled: [m]
```

### Scatter Plot with Units

```python
from dimtensor import DimArray, units
from dimtensor.visualization import scatter
import matplotlib.pyplot as plt

# Mass vs Force data
mass = DimArray([1, 2, 3, 4, 5], units.kg)
force = DimArray([9.8, 19.6, 29.4, 39.2, 49.0], units.N)

# Scatter plot with automatic labels
scatter(mass, force, s=100, alpha=0.7)
plt.title("Force vs Mass (g = 9.8 m/s²)")
plt.grid(True)
plt.show()
```

### Error Bars with Uncertainty

```python
from dimtensor import DimArray, units
from dimtensor.visualization import errorbar
import matplotlib.pyplot as plt
import numpy as np

# Experimental data with uncertainty
time = DimArray([0, 1, 2, 3, 4], units.s)
distance = DimArray([0, 10, 19, 32, 48], units.m,
                    uncertainty=[0, 0.5, 0.7, 1.0, 1.2])

# Plot with error bars (automatically uses uncertainty)
errorbar(time, distance, fmt='o-', capsize=5)
plt.title("Distance Measurement with Uncertainty")
plt.grid(True)
plt.show()
```

### Multiple Unit Conversions in Plots

```python
from dimtensor import DimArray, units
from dimtensor.visualization import plot
import matplotlib.pyplot as plt

# Create data in meters
distance_m = DimArray([0, 1000, 2000, 3000, 4000, 5000], units.m)
time_s = DimArray([0, 100, 200, 300, 400, 500], units.s)

# Plot in kilometers
plot(time_s, distance_m, y_unit=units.km, label='Distance')
plt.title("Distance Traveled")
plt.legend()
plt.grid(True)
plt.show()
# Y-axis will show [km]
```

## SciPy Integration

### Numerical Integration with Units

```python
from dimtensor import DimArray, units
from dimtensor.scipy import quad

# Integrate velocity to get distance: d = ∫v dt
def velocity(t):
    # v(t) = 10 m/s (constant)
    return DimArray([10.0], units.m / units.s)

# Integrate from t=0 to t=5 seconds
distance, error = quad(velocity, 0.0, 5.0)
print(f"Distance: {distance}")  # 50 m
print(f"Integration error: {error}")
```

### ODE Solving with Physical Units

```python
import torch
from dimtensor import DimArray, units
from dimtensor.scipy import solve_ivp
import numpy as np

# Simple harmonic oscillator: d²x/dt² = -ω²x
# State vector: [position, velocity]
def harmonic_oscillator(t, y):
    omega = 2.0  # rad/s
    position = y._data[0]
    velocity = y._data[1]

    # dy/dt = [velocity, -omega^2 * position]
    dydt = DimArray([velocity, -omega**2 * position], y.unit)
    return dydt

# Initial conditions: x0 = 1m, v0 = 0 m/s
y0 = DimArray([1.0, 0.0], units.m)
t_span = (0.0, 10.0)  # seconds

# Solve ODE
sol = solve_ivp(harmonic_oscillator, t_span, y0, t_eval=np.linspace(0, 10, 100))

print(f"Solution times shape: {sol.t.shape}")
print(f"Solution values shape: {sol.y.shape}")
print(f"Final position: {sol.y._data[0, -1]:.3f} {sol.y.unit.symbol}")
```

## Scikit-Learn Integration

### Dimension-Aware Standard Scaler

```python
from dimtensor import DimArray, units
from dimtensor.sklearn import DimStandardScaler
import numpy as np

# Feature matrix with physical units (e.g., temperature measurements)
X = DimArray([
    [273.15, 298.15, 310.15],
    [275.0, 300.0, 312.0],
    [270.0, 295.0, 308.0]
], units.K)

# Create and fit scaler
scaler = DimStandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Original data:\n{X._data}")
print(f"Scaled data (z-scores):\n{X_scaled}")

# Inverse transform restores original scale and units
X_restored = scaler.inverse_transform(X_scaled)
print(f"Restored data:\n{X_restored._data}")
print(f"Units preserved: {X_restored.unit.symbol}")
```

### Min-Max Scaler with Units

```python
from dimtensor import DimArray, units
from dimtensor.sklearn import DimMinMaxScaler

# Pressure measurements
pressures = DimArray([
    [101325, 150000, 200000],  # Pa
    [110000, 160000, 210000],
    [105000, 155000, 205000]
], units.Pa)

# Scale to [0, 1]
scaler = DimMinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(pressures)

print(f"Original range: {pressures._data.min():.0f} - {pressures._data.max():.0f} Pa")
print(f"Scaled range: {scaled.min():.2f} - {scaled.max():.2f}")

# Inverse transform
restored = scaler.inverse_transform(scaled)
print(f"Restored units: {restored.unit.symbol}")
```

## SymPy Integration

### Converting to Symbolic Expressions

```python
from dimtensor import DimArray, units
from dimtensor.sympy import to_sympy, from_sympy
import sympy as sp

# Convert numeric DimArray to SymPy
distance = DimArray([100], units.m)
expr = to_sympy(distance)
print(f"SymPy expression: {expr}")  # 100*meter

# Create symbolic expression with a symbol
symbolic_distance = to_sympy(distance, symbol="d")
print(f"Symbolic: {symbolic_distance}")  # d*meter

# Symbolic algebra
time_sym = to_sympy(DimArray([10], units.s), symbol="t")
velocity_expr = symbolic_distance / time_sym
print(f"Velocity: {velocity_expr}")  # d*meter/(t*second)
```

### Converting from SymPy to DimArray

```python
from dimtensor.sympy import from_sympy
from dimtensor import units
import sympy as sp
from sympy.physics.units import meter, second, joule

# SymPy expression with units
expr = 100 * meter / second

# Convert to DimArray
arr = from_sympy(expr)
print(f"DimArray: {arr}")  # [100.] m/s

# Energy expression
energy_expr = 500 * joule
energy = from_sympy(energy_expr)
print(f"Energy: {energy}")  # [500.] J

# Can specify target unit
energy_eV = from_sympy(energy_expr, target_unit=units.eV)
print(f"Energy in eV: {energy_eV}")
```

### Symbolic Differentiation with Units

```python
from dimtensor.sympy import to_sympy, from_sympy
from dimtensor import DimArray, units
import sympy as sp

# Position as function of time: x(t) = x0 + v*t + (1/2)*a*t²
x0 = to_sympy(DimArray([0], units.m), symbol="x0")
v = to_sympy(DimArray([10], units.m/units.s), symbol="v")
a = to_sympy(DimArray([2], units.m/units.s**2), symbol="a")
t = sp.Symbol('t', real=True, positive=True)

# Position equation (symbolic)
position = x0 + v*t + sp.Rational(1, 2)*a*t**2
print(f"Position: {position}")

# Differentiate to get velocity
velocity = sp.diff(position, t)
print(f"Velocity: {velocity}")  # v + a*t

# Differentiate again to get acceleration
acceleration = sp.diff(velocity, t)
print(f"Acceleration: {acceleration}")  # a
```

## Advanced Examples

### Physics Simulation with Units

```python
from dimtensor import DimArray, units
import numpy as np

# Projectile motion simulation
g = DimArray([9.8], units.m / units.s**2)
v0 = DimArray([50], units.m / units.s)
theta = DimArray([np.pi/4], units.rad)  # 45 degrees

# Initial velocity components
vx = v0 * np.cos(theta)
vy = v0 * np.sin(theta)

# Time of flight
t_flight = 2 * vy / g
print(f"Time of flight: {t_flight.to(units.s)}")

# Maximum height
h_max = (vy ** 2) / (2 * g)
print(f"Maximum height: {h_max.to(units.m)}")

# Range
R = vx * t_flight
print(f"Range: {R.to(units.m)}")

# Trajectory at t = 2 seconds
t = DimArray([2.0], units.s)
x = vx * t
y = vy * t - 0.5 * g * t**2
print(f"Position at t=2s: ({x}, {y})")
```

### Thermodynamics with Validation

```python
from dimtensor import DimArray, units
from dimtensor.validation import Positive

# Ideal gas law: PV = nRT
R = DimArray([8.314], units.J / (units.mol * units.K))  # Gas constant
n = DimArray([2.0], units.mol)
T = DimArray([300], units.K)
V = DimArray([0.05], units.m**3)

# Calculate pressure
P = (n * R * T) / V
print(f"Pressure: {P.to(units.Pa)}")
print(f"Pressure: {P.to(units.atm)}")

# Validate: pressure must be positive
Positive().validate(P._data)

# Energy validation
U = DimArray([1000], units.J)  # Internal energy
Q = DimArray([500], units.J)   # Heat added
W = DimArray([300], units.J)   # Work done

# First law: ΔU = Q - W
delta_U = Q - W
print(f"Change in internal energy: {delta_U}")

# All energies must be finite
from dimtensor.validation import Finite
Finite().validate(U._data)
Finite().validate(Q._data)
Finite().validate(W._data)
```

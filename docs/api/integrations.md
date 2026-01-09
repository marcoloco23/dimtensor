# Scientific Computing Integrations

Dimension-aware wrappers for SciPy, scikit-learn, and SymPy.

## Overview

The dimtensor integration modules provide seamless interoperability with popular scientific computing libraries while preserving physical dimensions:

- **SciPy**: ODE solving, integration, interpolation, optimization
- **scikit-learn**: Transformers and pipelines preserving units
- **SymPy**: Symbolic computation with dimensional analysis

```python
# SciPy integration
from dimtensor.scipy import solve_ivp, quad
from dimtensor import DimArray, units

def f(t, y):
    return -0.5 * y  # Exponential decay

sol = solve_ivp(f, (0, 10), DimArray([1.0], units.kg))

# scikit-learn integration
from dimtensor.sklearn import DimStandardScaler

scaler = DimStandardScaler()
X_scaled = scaler.fit_transform(X)  # Preserves units

# SymPy integration
from dimtensor.sympy import to_sympy, from_sympy

expr = to_sympy(mass * velocity)  # SymPy expression with units
```

## SciPy Integration

### ODE Solving

#### solve_ivp

Dimension-aware initial value problem solver. Wraps `scipy.integrate.solve_ivp` to preserve units through ODE solving.

```python
from dimtensor import DimArray, units
from dimtensor.scipy import solve_ivp
import numpy as np

def harmonic_oscillator(t, y):
    """Simple harmonic oscillator: d²x/dt² = -ω²x

    y = [position, velocity]
    dy/dt = [velocity, -ω² * position]
    """
    omega = 2.0  # rad/s
    position = y[0]
    velocity = y[1]

    dx_dt = velocity
    dv_dt = -omega**2 * position

    return DimArray([dx_dt.data, dv_dt.data], y.unit)

# Initial conditions: position = 1m, velocity = 0 m/s
y0 = DimArray([1.0, 0.0], units.m)

# Solve over 10 seconds
sol = solve_ivp(harmonic_oscillator, (0.0, 10.0), y0, dense_output=True)

# Access solution
t = sol.t  # Time points
y = sol.y  # State (position, velocity) with units
```

::: dimtensor.scipy.solve_ivp
    options:
      members:
        - fun
        - t_span
        - y0
        - method
        - t_eval

#### odeint

Backward-compatible wrapper for `scipy.integrate.odeint`.

```python
from dimtensor.scipy import odeint

def deriv(y, t):
    return -0.5 * y

t = np.linspace(0, 10, 100)
y0 = DimArray([1.0], units.kg)
sol = odeint(deriv, y0, t)
```

::: dimtensor.scipy.odeint

### Integration

#### quad

Dimension-aware numerical integration.

```python
from dimtensor.scipy import quad
from dimtensor import units

def integrand(x):
    # Force as function of position: F = -kx
    k = 10.0  # N/m
    return DimArray(-k * x, units.N)

# Integrate from x=0 to x=1 meter
result, error = quad(integrand, 0.0, 1.0)
print(result)  # Work in Joules (N*m)
```

::: dimtensor.scipy.quad

### Interpolation

#### interp1d

Dimension-aware 1D interpolation.

```python
from dimtensor.scipy import interp1d
from dimtensor import DimArray, units

# Measured temperature vs time
time = DimArray([0, 1, 2, 3, 4], units.s)
temp = DimArray([300, 310, 305, 315, 320], units.K)

# Create interpolator
f = interp1d(time, temp, kind='cubic')

# Interpolate at new points
t_new = DimArray([0.5, 1.5, 2.5], units.s)
T_new = f(t_new)  # DimArray with units.K
```

::: dimtensor.scipy.interp1d

### Optimization

#### minimize

Dimension-aware optimization wrapper.

```python
from dimtensor.scipy import minimize
from dimtensor import DimArray, units

def objective(x):
    """Minimize potential energy: U = 0.5*k*x^2"""
    k = 10.0  # N/m
    return 0.5 * k * x.data**2

x0 = DimArray([1.0], units.m)
result = minimize(objective, x0, method='BFGS')
x_min = result.x  # DimArray with units
```

::: dimtensor.scipy.minimize

## scikit-learn Integration

### Transformers

All transformers preserve physical units through fit/transform operations.

#### DimStandardScaler

Standardizes features by removing mean and scaling to unit variance, preserving physical units.

```python
from dimtensor.sklearn import DimStandardScaler
from dimtensor import DimArray, units
import numpy as np

# Training data with units
X = DimArray(np.random.randn(100, 3), units.m)

# Fit and transform
scaler = DimStandardScaler()
X_scaled = scaler.fit_transform(X)  # Centered and scaled, still [m]

# Inverse transform
X_original = scaler.inverse_transform(X_scaled)  # Recovers original scale
```

::: dimtensor.sklearn.DimStandardScaler
    options:
      members:
        - __init__
        - fit
        - transform
        - fit_transform
        - inverse_transform

#### DimMinMaxScaler

Scales features to a given range [0, 1] while preserving units.

```python
from dimtensor.sklearn import DimMinMaxScaler

scaler = DimMinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)  # Values in [0, 1], units preserved
```

::: dimtensor.sklearn.DimMinMaxScaler

#### DimRobustScaler

Scales features using statistics robust to outliers, preserving units.

```python
from dimtensor.sklearn import DimRobustScaler

# Uses median and IQR instead of mean and std
scaler = DimRobustScaler()
X_scaled = scaler.fit_transform(X)
```

::: dimtensor.sklearn.DimRobustScaler

#### DimPCA

Principal Component Analysis preserving physical dimensions.

```python
from dimtensor.sklearn import DimPCA

pca = DimPCA(n_components=2)
X_transformed = pca.fit_transform(X)  # Reduced dimensions, units preserved
```

::: dimtensor.sklearn.DimPCA
    options:
      members:
        - __init__
        - fit
        - transform
        - fit_transform
        - inverse_transform
        - explained_variance_
        - explained_variance_ratio_

### Pipelines

Combine multiple transformers in a pipeline:

```python
from dimtensor.sklearn import DimStandardScaler, DimPCA
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', DimStandardScaler()),
    ('pca', DimPCA(n_components=2))
])

X_transformed = pipeline.fit_transform(X)
```

## SymPy Integration

### Conversion Functions

#### to_sympy

Convert DimArray to SymPy expression with units.

```python
from dimtensor import DimArray, units
from dimtensor.sympy import to_sympy
import sympy as sp

# Create dimensional array
mass = DimArray([2.0], units.kg)
velocity = DimArray([10.0], units.m / units.s)

# Convert to SymPy
m_sym = to_sympy(mass)
v_sym = to_sympy(velocity)

# Symbolic computation
KE_sym = sp.Rational(1, 2) * m_sym * v_sym**2
print(KE_sym)  # 100.0*joule

# Simplify
KE_simplified = sp.simplify(KE_sym)
```

::: dimtensor.sympy.to_sympy

#### from_sympy

Convert SymPy expression with units back to DimArray.

```python
from dimtensor.sympy import from_sympy
import sympy as sp
from sympy.physics.units import meter, second

# SymPy expression
expr = 10 * meter / second

# Convert to DimArray
arr = from_sympy(expr)
print(arr.unit)  # m/s
```

::: dimtensor.sympy.from_sympy

#### sympy_unit_for

Get SymPy unit expression for a dimtensor Unit.

```python
from dimtensor import units
from dimtensor.sympy import sympy_unit_for

sympy_force = sympy_unit_for(units.N)
print(sympy_force)  # newton

sympy_energy = sympy_unit_for(units.J)
print(sympy_energy)  # joule
```

::: dimtensor.sympy.sympy_unit_for

### Symbolic Calculus

#### differentiate

Symbolic differentiation with dimensional checking.

```python
from dimtensor.sympy import differentiate
from dimtensor import units
import sympy as sp

# Define symbolic variable with units
x = sp.Symbol('x') * units.m
t = sp.Symbol('t') * units.s

# Position as function of time: x(t) = x0 + v*t
x0 = 10.0 * units.m
v = 5.0 * units.m / units.s
x_of_t = x0 + v * t

# Differentiate to get velocity
v_result = differentiate(x_of_t, t)
print(v_result)  # 5.0 m/s
```

::: dimtensor.sympy.differentiate

#### integrate_symbolic

Symbolic integration with dimensional checking.

```python
from dimtensor.sympy import integrate_symbolic
import sympy as sp

# Integrate force over distance to get work
F = sp.Symbol('F')  # Force (N)
x = sp.Symbol('x')  # Distance (m)

work = integrate_symbolic(F, x)  # F*x (N*m = J)
```

::: dimtensor.sympy.integrate_symbolic

## Examples

### Physics Simulation with SciPy

```python
from dimtensor import DimArray, units
from dimtensor.scipy import solve_ivp
import matplotlib.pyplot as plt

def projectile_motion(t, y):
    """2D projectile motion with air resistance.

    y = [x, vx, z, vz]
    """
    x, vx, z, vz = y[0], y[1], y[2], y[3]

    # Constants
    g = 9.81  # m/s^2
    drag_coeff = 0.1  # 1/s

    # Derivatives
    dx_dt = vx
    dvx_dt = -drag_coeff * vx
    dz_dt = vz
    dvz_dt = -g - drag_coeff * vz

    return DimArray([dx_dt.data, dvx_dt.data, dz_dt.data, dvz_dt.data], y.unit)

# Initial conditions: position (0, 0), velocity (10, 10) m/s
y0 = DimArray([0.0, 10.0, 0.0, 10.0], units.m)

# Solve
sol = solve_ivp(projectile_motion, (0.0, 3.0), y0, dense_output=True)

# Plot trajectory
t_plot = np.linspace(0, 3, 300)
y_plot = sol.sol(t_plot)
plt.plot(y_plot[0], y_plot[2])  # x vs z
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.title('Projectile Motion with Air Resistance')
```

### Machine Learning Pipeline

```python
from dimtensor import DimArray, units
from dimtensor.sklearn import DimStandardScaler, DimPCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Generate training data with units
n_samples = 1000
X = DimArray(np.random.randn(n_samples, 10), units.m)  # Features
y = DimArray(np.random.randn(n_samples), units.kg)      # Target

# Create pipeline
pipeline = Pipeline([
    ('scaler', DimStandardScaler()),
    ('pca', DimPCA(n_components=5)),
    # Note: RF operates on scaled data, doesn't track units
    ('regressor', RandomForestRegressor())
])

# Train
# Extract data for sklearn regressor (which doesn't understand units)
from dimtensor.sklearn import make_pipeline
pipe = make_pipeline(
    DimStandardScaler(),
    DimPCA(n_components=5)
)

X_processed = pipe.fit_transform(X)

# Now use with standard sklearn
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_processed, y.data)

# Predict (need to manually wrap with units)
X_test = DimArray(np.random.randn(10, 10), units.m)
X_test_processed = pipe.transform(X_test)
y_pred_data = rf.predict(X_test_processed)
y_pred = DimArray(y_pred_data, units.kg)
```

### Symbolic Physics with SymPy

```python
from dimtensor import DimArray, units
from dimtensor.sympy import to_sympy, from_sympy, differentiate
import sympy as sp

# Define symbolic variables
t = sp.Symbol('t', real=True, positive=True)
m = sp.Symbol('m', real=True, positive=True)
v0 = sp.Symbol('v0', real=True)
theta = sp.Symbol('theta', real=True)
g = sp.Symbol('g', real=True, positive=True)

# Projectile motion equations
x = v0 * sp.cos(theta) * t
y = v0 * sp.sin(theta) * t - sp.Rational(1, 2) * g * t**2

# Differentiate to get velocity components
vx = sp.diff(x, t)
vy = sp.diff(y, t)

# Speed
speed = sp.sqrt(vx**2 + vy**2)
speed_simplified = sp.simplify(speed)

print(f"vx = {vx}")
print(f"vy = {vy}")
print(f"speed = {speed_simplified}")

# Substitute numeric values with units
subs_dict = {
    v0: 10,  # m/s
    theta: sp.pi / 4,  # 45 degrees
    g: 9.81,  # m/s^2
    t: 1.0  # s
}

x_numeric = x.subs(subs_dict)
y_numeric = y.subs(subs_dict)

print(f"x(1s) = {x_numeric:.2f} m")
print(f"y(1s) = {y_numeric:.2f} m")
```

### Optimization with Constraints

```python
from dimtensor import DimArray, units
from dimtensor.scipy import minimize
import numpy as np

def rocket_trajectory_cost(controls):
    """Minimize fuel for reaching target.

    controls: thrust vector [Fx, Fz] in Newtons
    """
    Fx, Fz = controls[0], controls[1]

    # Fuel consumption proportional to thrust magnitude
    fuel = np.sqrt(Fx**2 + Fz**2)

    return fuel

def constraint_reach_target(controls):
    """Constraint: must reach target position."""
    # ... compute final position from controls ...
    # Return 0 if constraint satisfied
    return final_position - target_position

# Initial guess
x0 = DimArray([100.0, 100.0], units.N)

# Minimize with constraint
from scipy.optimize import NonlinearConstraint
constraint = NonlinearConstraint(constraint_reach_target, 0, 0)

result = minimize(
    rocket_trajectory_cost,
    x0,
    method='SLSQP',
    constraints=[constraint]
)

optimal_thrust = result.x  # DimArray in Newtons
```

### Dimensionality Reduction

```python
from dimtensor import DimArray, units
from dimtensor.sklearn import DimPCA, DimStandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Generate high-dimensional physics data
n_samples = 1000
n_features = 50

# Features: various physical quantities
data = np.random.randn(n_samples, n_features)
X = DimArray(data, units.m)  # All in meters for simplicity

# Standardize
scaler = DimStandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = DimPCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X_reduced[:, 0].data, X_reduced[:, 1].data, alpha=0.5)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Physics Data in PC Space')

# Check explained variance
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

## Advanced Integration

### Custom SciPy Functions

```python
from dimtensor import DimArray
from dimtensor.scipy import solve_ivp
import scipy.integrate as integrate

def coupled_oscillators(t, y, k, m):
    """System of coupled harmonic oscillators."""
    # y = [x1, v1, x2, v2]
    x1, v1, x2, v2 = y[0], y[1], y[2], y[3]

    # Coupling force
    F_coupling = k * (x2 - x1)

    # Derivatives
    dx1_dt = v1
    dv1_dt = F_coupling / m
    dx2_dt = v2
    dv2_dt = -F_coupling / m

    return DimArray([dx1_dt.data, dv1_dt.data, dx2_dt.data, dv2_dt.data], y.unit)

# Solve with parameters
k = 10.0  # N/m
m = 1.0   # kg
y0 = DimArray([1.0, 0.0, -1.0, 0.0], units.m)

sol = solve_ivp(
    lambda t, y: coupled_oscillators(t, y, k, m),
    (0, 10),
    y0,
    method='DOP853',  # High-accuracy method
    rtol=1e-10,
    atol=1e-12
)
```

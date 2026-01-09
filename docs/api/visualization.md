# Visualization

Plotting functions with automatic unit labeling for matplotlib and plotly.

## Overview

The `dimtensor.visualization` module provides plotting functions that automatically handle physical units:

- **Matplotlib Integration**: Automatic axis labeling with units
- **Plotly Integration**: Interactive plots with unit-aware tooltips
- **Seamless Conversion**: Plots work with both DimArray and regular arrays

```python
from dimtensor import DimArray, units
from dimtensor.visualization import plot, scatter

# Data with units
time = DimArray([0, 1, 2, 3, 4], units.s)
distance = DimArray([0, 5, 20, 45, 80], units.m)

# Plot automatically labels axes
plot(time, distance)  # X-axis: "Time [s]", Y-axis: "Distance [m]"
```

## Matplotlib Integration

### Setup

Enable automatic unit handling for matplotlib:

```python
from dimtensor.visualization import setup_matplotlib

# Enable automatic unit labels
setup_matplotlib()

# Now matplotlib.pyplot automatically handles DimArrays
import matplotlib.pyplot as plt
plt.plot(time, distance)  # Axes labeled automatically
```

::: dimtensor.visualization.setup_matplotlib

### Plotting Functions

#### plot

Line plot with automatic unit labeling.

```python
from dimtensor.visualization import plot
from dimtensor import DimArray, units

time = DimArray([0, 1, 2, 3], units.s)
velocity = DimArray([0, 10, 20, 30], units.m / units.s)

plot(time, velocity, label='Velocity')
plt.title('Object Motion')
plt.legend()
plt.show()
```

::: dimtensor.visualization.plot

#### scatter

Scatter plot with unit labels.

```python
from dimtensor.visualization import scatter

pressure = DimArray([100, 110, 120, 130], units.kPa)
temperature = DimArray([300, 320, 340, 360], units.K)

scatter(pressure, temperature, c='red', marker='o')
plt.title('P-T Diagram')
```

::: dimtensor.visualization.scatter

#### errorbar

Error bar plot preserving units.

```python
from dimtensor.visualization import errorbar

x = DimArray([1, 2, 3, 4], units.s)
y = DimArray([10, 20, 25, 30], units.m)
yerr = DimArray([1, 2, 1.5, 2], units.m)

errorbar(x, y, yerr=yerr, fmt='o-', capsize=5)
plt.title('Measurement with Uncertainty')
```

::: dimtensor.visualization.errorbar

#### hist

Histogram with unit-labeled axes.

```python
from dimtensor.visualization import hist

velocities = DimArray(np.random.randn(1000) * 10 + 50, units.m / units.s)

hist(velocities, bins=30, alpha=0.7)
plt.title('Velocity Distribution')
```

::: dimtensor.visualization.hist

#### contour

Contour plot for 2D fields with units.

```python
from dimtensor.visualization import contour

X = DimArray(np.linspace(0, 10, 50), units.m)
Y = DimArray(np.linspace(0, 10, 50), units.m)
Z = DimArray(np.outer(X.data, Y.data), units.Pa)  # Pressure field

contour(X, Y, Z, levels=20)
plt.title('Pressure Field')
```

::: dimtensor.visualization.contour

#### quiver

Vector field plot with units.

```python
from dimtensor.visualization import quiver

# Velocity field
X, Y = np.meshgrid(np.linspace(0, 5, 20), np.linspace(0, 5, 20))
U = DimArray(-Y, units.m / units.s)
V = DimArray(X, units.m / units.s)
x_pos = DimArray(X, units.m)
y_pos = DimArray(Y, units.m)

quiver(x_pos, y_pos, U, V)
plt.title('Velocity Field')
```

::: dimtensor.visualization.quiver

### Advanced Matplotlib Features

#### Multiple Subplots

```python
from dimtensor.visualization import plot, scatter
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left plot
plt.sca(ax1)
plot(time, position)
plt.title('Position vs Time')

# Right plot
plt.sca(ax2)
plot(time, velocity)
plt.title('Velocity vs Time')

plt.tight_layout()
plt.show()
```

#### Custom Styling

```python
from dimtensor.visualization import plot

plot(time, distance,
     color='blue',
     linewidth=2,
     linestyle='--',
     marker='o',
     markersize=8)
plt.grid(True, alpha=0.3)
plt.title('Styled Plot')
```

## Plotly Integration

Interactive plots with unit-aware tooltips and automatic labeling.

### Line Plots

#### line

Create interactive line plot.

```python
from dimtensor.visualization.plotly import line
from dimtensor import DimArray, units

time = DimArray([0, 1, 2, 3, 4], units.s)
position = DimArray([0, 10, 40, 90, 160], units.m)

fig = line(time, position,
           title='Position vs Time',
           x_title='Time',
           y_title='Position')
fig.show()
```

::: dimtensor.visualization.plotly.line

#### scatter

Interactive scatter plot.

```python
from dimtensor.visualization.plotly import scatter

mass = DimArray([1, 2, 3, 4, 5], units.kg)
energy = DimArray([10, 40, 90, 160, 250], units.J)

fig = scatter(mass, energy,
              title='Energy vs Mass',
              size_max=20)
fig.show()
```

::: dimtensor.visualization.plotly.scatter

### 3D Visualization

#### scatter_3d

3D scatter plot with units.

```python
from dimtensor.visualization.plotly import scatter_3d

x = DimArray([1, 2, 3, 4], units.m)
y = DimArray([2, 3, 1, 4], units.m)
z = DimArray([5, 1, 3, 2], units.m)

fig = scatter_3d(x, y, z, title='3D Particle Positions')
fig.show()
```

::: dimtensor.visualization.plotly.scatter_3d

#### surface

3D surface plot for fields.

```python
from dimtensor.visualization.plotly import surface
import numpy as np

x = DimArray(np.linspace(-5, 5, 50), units.m)
y = DimArray(np.linspace(-5, 5, 50), units.m)
X, Y = np.meshgrid(x.data, y.data)
Z = DimArray(np.sin(np.sqrt(X**2 + Y**2)), units.dimensionless)

fig = surface(x, y, Z, title='Wave Pattern')
fig.show()
```

::: dimtensor.visualization.plotly.surface

### Heatmaps

#### heatmap

2D heatmap with unit labels.

```python
from dimtensor.visualization.plotly import heatmap

# Temperature field over space and time
x = DimArray(np.linspace(0, 10, 20), units.m)
t = DimArray(np.linspace(0, 5, 30), units.s)
T = DimArray(np.random.randn(30, 20) * 10 + 300, units.K)

fig = heatmap(x, t, T,
              x_title='Position',
              y_title='Time',
              title='Temperature Evolution')
fig.show()
```

::: dimtensor.visualization.plotly.heatmap

## Examples

### Time Series Visualization

```python
from dimtensor import DimArray, units
from dimtensor.visualization import plot
import numpy as np
import matplotlib.pyplot as plt

# Generate time series data
t = DimArray(np.linspace(0, 10, 1000), units.s)
position = DimArray(np.sin(2 * np.pi * t.data), units.m)
velocity = DimArray(2 * np.pi * np.cos(2 * np.pi * t.data), units.m / units.s)
acceleration = DimArray(-4 * np.pi**2 * np.sin(2 * np.pi * t.data), units.m / units.s**2)

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

plt.sca(axes[0])
plot(t, position)
plt.title('Harmonic Oscillator')
plt.ylabel('Position [m]')
plt.grid(True)

plt.sca(axes[1])
plot(t, velocity, color='orange')
plt.ylabel('Velocity [m/s]')
plt.grid(True)

plt.sca(axes[2])
plot(t, acceleration, color='red')
plt.ylabel('Acceleration [m/s²]')
plt.xlabel('Time [s]')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Phase Space Plot

```python
from dimtensor.visualization import plot, scatter
import matplotlib.pyplot as plt

# Generate phase space trajectory
theta = DimArray(np.linspace(0, 4*np.pi, 1000), units.rad)
omega = DimArray(np.cos(theta.data), units.rad / units.s)

scatter(theta, omega, s=1, c=np.linspace(0, 1, 1000), cmap='viridis')
plt.title('Phase Space: Pendulum')
plt.xlabel('Angle [rad]')
plt.ylabel('Angular Velocity [rad/s]')
plt.colorbar(label='Time (normalized)')
plt.grid(True)
plt.show()
```

### Pressure-Volume Diagram

```python
from dimtensor import DimArray, units
from dimtensor.visualization import plot
import numpy as np

# Isothermal process: PV = constant
V = DimArray(np.linspace(0.1, 2.0, 100), units.m**3)
n = 1.0  # mol
R = 8.314  # J/(mol*K)
T = DimArray([300, 400, 500], units.K)

plt.figure(figsize=(10, 6))

for temp in T.data:
    P = DimArray(n * R * temp / V.data, units.Pa)
    plot(V, P, label=f'T = {temp} K')

plt.title('Ideal Gas: P-V Diagram')
plt.xlabel('Volume [m³]')
plt.ylabel('Pressure [Pa]')
plt.legend()
plt.grid(True)
plt.show()
```

### Vector Field Visualization

```python
from dimtensor import DimArray, units
from dimtensor.visualization import quiver
import numpy as np
import matplotlib.pyplot as plt

# Electric field around a point charge
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)

# Field components (proportional to 1/r^2)
r_squared = X**2 + Y**2 + 0.1  # Add small value to avoid singularity
Ex = X / r_squared
Ey = Y / r_squared

x_pos = DimArray(X, units.m)
y_pos = DimArray(Y, units.m)
E_x = DimArray(Ex, units.V / units.m)
E_y = DimArray(Ey, units.V / units.m)

plt.figure(figsize=(8, 8))
quiver(x_pos, y_pos, E_x, E_y)
plt.title('Electric Field around Point Charge')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.axis('equal')
plt.show()
```

### Interactive 3D Trajectory (Plotly)

```python
from dimtensor import DimArray, units
from dimtensor.visualization.plotly import scatter_3d
import numpy as np

# Generate 3D spiral trajectory
t = np.linspace(0, 4*np.pi, 1000)
x = DimArray(np.cos(t) * np.exp(-t/10), units.m)
y = DimArray(np.sin(t) * np.exp(-t/10), units.m)
z = DimArray(t, units.m)

fig = scatter_3d(x, y, z,
                 title='Damped Spiral Trajectory',
                 color=t,
                 color_continuous_scale='viridis')

fig.update_traces(marker=dict(size=2))
fig.show()
```

### Temperature Field Animation (Plotly)

```python
from dimtensor import DimArray, units
from dimtensor.visualization.plotly import heatmap
import plotly.graph_objects as go
import numpy as np

# Generate temperature field evolution
x = DimArray(np.linspace(0, 10, 50), units.m)
t_steps = 20

frames = []
for i in range(t_steps):
    # Heat diffusion (simplified)
    T = DimArray(
        300 + 50 * np.exp(-x.data/3) * np.sin(2*np.pi*i/t_steps),
        units.K
    )
    frame_data = go.Frame(
        data=[go.Heatmap(z=T.data[np.newaxis, :])],
        name=f'frame{i}'
    )
    frames.append(frame_data)

# Create figure with first frame
T0 = DimArray(300 + 50 * np.exp(-x.data/3), units.K)
fig = go.Figure(
    data=[go.Heatmap(z=T0.data[np.newaxis, :])],
    frames=frames
)

# Add animation controls
fig.update_layout(
    title='Temperature Diffusion',
    xaxis_title='Position [m]',
    yaxis_title='',
    updatemenus=[{
        'buttons': [
            {'args': [None, {'frame': {'duration': 100}}],
             'label': 'Play',
             'method': 'animate'}
        ],
        'type': 'buttons'
    }]
)

fig.show()
```

### Multiple Unit Conversions

```python
from dimtensor import DimArray, units
from dimtensor.visualization import plot
import matplotlib.pyplot as plt

# Same data in different units
time_s = DimArray([0, 60, 120, 180], units.s)
distance_m = DimArray([0, 100, 400, 900], units.m)

# Convert for display
time_min = time_s.to(units.minute)
distance_km = distance_m.to(units.km)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: SI units
plt.sca(ax1)
plot(time_s, distance_m, 'b-o')
plt.title('Motion (SI Units)')
plt.grid(True)

# Plot 2: Converted units
plt.sca(ax2)
plot(time_min, distance_km, 'r-o')
plt.title('Motion (Converted Units)')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Best Practices

### Unit Consistency

Always ensure data has compatible units when plotting on the same axes:

```python
# Good: Same dimension
velocity1 = DimArray([10, 20, 30], units.m / units.s)
velocity2 = DimArray([15, 25, 35], units.km / units.hour)  # Will convert

plot(time, velocity1, label='Car 1')
plot(time, velocity2.to(units.m / units.s), label='Car 2')  # Explicit conversion

# Bad: Different dimensions (will error or produce misleading plot)
# plot(time_s, mass_kg)  # Don't mix time and mass!
```

### Labeling

Provide clear titles and labels even when units are automatic:

```python
plot(time, velocity, label='Measured')
plot(time, velocity_predicted, label='Model')
plt.title('Velocity Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
```

### Interactive vs Static

- Use **matplotlib** for publication-quality static figures
- Use **plotly** for interactive exploration and dashboards

```python
# For publication: matplotlib
from dimtensor.visualization import plot
plot(x, y)
plt.savefig('figure.pdf', dpi=300, bbox_inches='tight')

# For exploration: plotly
from dimtensor.visualization.plotly import line
fig = line(x, y)
fig.write_html('interactive_plot.html')
```

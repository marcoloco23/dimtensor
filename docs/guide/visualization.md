# Visualization with DimArray

Create publication-quality plots with automatic unit labeling using matplotlib or plotly.

## Why Automatic Unit Labels Matter

In scientific plotting, axis labels must include units to be meaningful. dimtensor's visualization modules automatically extract units from DimArray objects and add them to plot axes, reducing errors and saving time.

**Traditional approach (error-prone):**
```python
plt.plot(time_data, distance_data)
plt.xlabel("Time (s)")  # Must manually specify units
plt.ylabel("Distance (m)")
```

**With dimtensor (automatic):**
```python
from dimtensor.visualization import plot
plot(time, distance)  # Units extracted automatically!
# x-axis: [s], y-axis: [m]
```

## Installation

The visualization modules require matplotlib or plotly:

```bash
# For matplotlib support
pip install matplotlib

# For plotly support
pip install plotly

# Install both
pip install dimtensor[all]
```

## Quick Start

### Matplotlib Quick Example

```python
from dimtensor import DimArray, units
from dimtensor.visualization import plot
import matplotlib.pyplot as plt

time = DimArray([0, 1, 2, 3, 4], units.s)
position = DimArray([0, 5, 20, 45, 80], units.m)

plot(time, position)
plt.title("Projectile Motion")
plt.show()
# Axes automatically labeled: [s] and [m]
```

### Plotly Quick Example

```python
from dimtensor import DimArray, units
from dimtensor.visualization.plotly import line

time = DimArray([0, 1, 2, 3, 4], units.s)
position = DimArray([0, 5, 20, 45, 80], units.m)

fig = line(time, position, title="Projectile Motion")
fig.show()
# Interactive plot with automatic unit labels
```

## Matplotlib Integration

dimtensor provides two ways to use matplotlib: direct integration or wrapper functions.

### Mode 1: Direct Integration with setup_matplotlib()

Call `setup_matplotlib()` once to enable automatic unit handling with standard matplotlib functions:

```python
from dimtensor import DimArray, units
from dimtensor.visualization import setup_matplotlib
import matplotlib.pyplot as plt

setup_matplotlib()  # Enable integration

# Now use standard matplotlib functions
time = DimArray([0, 1, 2, 3], units.s)
distance = DimArray([0, 10, 40, 90], units.m)

plt.plot(time, distance, 'o-')
plt.title("Position vs Time")
plt.show()
# Axes automatically labeled with [s] and [m]
```

This works with `plt.plot()`, `plt.scatter()`, and other matplotlib functions that accept array-like data.

### Mode 2: Wrapper Functions

Use dimtensor's wrapper functions for more control over unit conversion:

#### plot() - Line Plots

```python
from dimtensor import DimArray, units
from dimtensor.visualization import plot
import matplotlib.pyplot as plt

# Physics example: kinetic energy vs velocity
velocity = DimArray([0, 10, 20, 30, 40], units.m / units.s)
mass = DimArray([2.0], units.kg)
kinetic_energy = 0.5 * mass * velocity**2

plot(velocity, kinetic_energy, 'o-', color='blue')
plt.title("Kinetic Energy")
plt.grid(True)
plt.show()
# x-axis: [m/s], y-axis: [J]
```

#### scatter() - Scatter Plots

```python
from dimtensor import DimArray, units
from dimtensor.visualization import scatter
import matplotlib.pyplot as plt

# Experimental data
mass = DimArray([1.0, 2.0, 3.0, 4.0, 5.0], units.kg)
force = DimArray([9.7, 19.8, 29.5, 39.1, 49.3], units.N)

scatter(mass, force, s=100, alpha=0.6, color='red')
plt.title("Force vs Mass")
plt.show()
```

#### bar() - Bar Charts

```python
from dimtensor import DimArray, units
from dimtensor.visualization import bar
import matplotlib.pyplot as plt

# Comparing heights
categories = ["Building A", "Building B", "Building C", "Building D"]
heights = DimArray([45, 60, 38, 52], units.m)

bar(categories, heights, color=['blue', 'green', 'red', 'orange'])
plt.title("Building Heights")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

#### hist() - Histograms

```python
from dimtensor import DimArray, units
from dimtensor.visualization import hist
import matplotlib.pyplot as plt
import numpy as np

# Measurement distribution
measurements = DimArray(np.random.normal(100, 5, 100), units.cm)

hist(measurements, bins=20, alpha=0.7, color='purple', edgecolor='black')
plt.title("Measurement Distribution")
plt.show()
# x-axis automatically labeled with [cm]
```

#### errorbar() - Error Bar Plots

```python
from dimtensor import DimArray, units
from dimtensor.visualization import errorbar
import matplotlib.pyplot as plt

# Experimental data with uncertainty
time = DimArray([1, 2, 3, 4, 5], units.s)
temperature = DimArray(
    [20.1, 35.2, 50.5, 64.8, 80.3],
    units.K,
    uncertainty=[0.5, 0.6, 0.4, 0.7, 0.5]
)

errorbar(time, temperature, fmt='o-', capsize=5, capthick=2)
plt.title("Temperature vs Time")
plt.grid(True, alpha=0.3)
plt.show()
# Error bars automatically extracted from temperature.uncertainty
```

### Unit Conversion in Plots

Convert units for display using `x_unit` and `y_unit` parameters:

```python
from dimtensor import DimArray, units
from dimtensor.visualization import plot
import matplotlib.pyplot as plt

# Data in SI units
distance = DimArray([0, 1000, 2000, 3000, 4000], units.m)
time = DimArray([0, 10, 20, 30, 40], units.s)

# Display in different units
plot(time, distance, y_unit=units.km)
plt.title("Distance Over Time")
plt.show()
# x-axis: [s], y-axis: [km] (automatically converted!)
```

**Converting both axes:**

```python
# Data in SI base units
time = DimArray([0, 3600, 7200, 10800], units.s)
distance = DimArray([0, 50000, 100000, 150000], units.m)

# Display in more convenient units
plot(time, distance, x_unit=units.hour, y_unit=units.km)
plt.title("Long Distance Travel")
plt.show()
# x-axis: [h], y-axis: [km]
```

### Working with Subplots

Use the `ax` parameter to create multi-panel figures:

```python
from dimtensor import DimArray, units
from dimtensor.visualization import plot
import matplotlib.pyplot as plt
import numpy as np

time = DimArray(np.linspace(0, 10, 100), units.s)
position = DimArray(5 * time._data**2, units.m)
velocity = DimArray(10 * time._data, units.m / units.s)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Plot position
plot(time, position, ax=ax1, color='blue')
ax1.set_title("Position vs Time")
ax1.grid(True, alpha=0.3)

# Plot velocity
plot(time, velocity, ax=ax2, color='red')
ax2.set_title("Velocity vs Time")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Dimensionless Quantities

Dimensionless quantities don't show unit labels:

```python
from dimtensor import DimArray, units
from dimtensor.visualization import plot
import matplotlib.pyplot as plt
import numpy as np

# Angle (dimensionless)
angle = DimArray(np.linspace(0, 2*np.pi, 100), units.rad)
sine = DimArray(np.sin(angle._data), units.dimensionless)

plot(angle, sine)
plt.title("Sine Wave")
# No unit labels because both are dimensionless
plt.show()
```

## Plotly Integration

Plotly creates interactive, web-based visualizations with automatic unit labels.

### Import Structure

```python
from dimtensor.visualization.plotly import line, scatter, bar, histogram, scatter_with_errors
```

### line() - Interactive Line Plots

```python
from dimtensor import DimArray, units
from dimtensor.visualization.plotly import line

time = DimArray([0, 1, 2, 3, 4], units.s)
velocity = DimArray([0, 9.8, 19.6, 29.4, 39.2], units.m / units.s)

fig = line(
    time,
    velocity,
    title="Velocity Over Time",
    x_title="Time",
    y_title="Velocity"
)
fig.show()
# Interactive plot with hover tooltips
# Axes: Time [s] and Velocity [m/s]
```

### scatter() - Interactive Scatter Plots

```python
from dimtensor import DimArray, units
from dimtensor.visualization.plotly import scatter

# Experimental relationship
voltage = DimArray([1.0, 2.0, 3.0, 4.0, 5.0], units.V)
current = DimArray([0.1, 0.21, 0.29, 0.42, 0.49], units.A)

fig = scatter(
    voltage,
    current,
    title="Ohm's Law Verification",
    x_title="Voltage",
    y_title="Current"
)
fig.update_traces(marker=dict(size=12, color='red'))
fig.show()
```

### bar() - Interactive Bar Charts

```python
from dimtensor import DimArray, units
from dimtensor.visualization.plotly import bar

planets = ["Mercury", "Venus", "Earth", "Mars"]
gravity = DimArray([3.7, 8.87, 9.81, 3.71], units.m / units.s**2)

fig = bar(
    planets,
    gravity,
    title="Surface Gravity by Planet",
    y_title="Acceleration"
)
fig.update_traces(marker_color=['gray', 'yellow', 'blue', 'red'])
fig.show()
# y-axis automatically shows [m/s²]
```

### histogram() - Distribution Plots

```python
from dimtensor import DimArray, units
from dimtensor.visualization.plotly import histogram
import numpy as np

# Sample measurements
samples = DimArray(np.random.normal(9.81, 0.1, 200), units.m / units.s**2)

fig = histogram(
    samples,
    title="Measured Gravitational Acceleration",
    x_title="Acceleration",
    nbins=30
)
fig.show()
# x-axis: Acceleration [m/s²]
```

### scatter_with_errors() - Error Bars

```python
from dimtensor import DimArray, units
from dimtensor.visualization.plotly import scatter_with_errors

# Data with measurement uncertainty
voltage = DimArray([0, 1, 2, 3, 4, 5], units.V)
resistance = DimArray(
    [10.1, 10.3, 9.9, 10.2, 10.0, 9.8],
    units.ohm,
    uncertainty=[0.2, 0.3, 0.2, 0.2, 0.3, 0.2]
)

fig = scatter_with_errors(
    voltage,
    resistance,
    title="Resistance Measurement",
    x_title="Applied Voltage",
    y_title="Resistance"
)
fig.show()
# Error bars automatically from resistance.uncertainty
```

### Unit Conversion with Plotly

```python
from dimtensor import DimArray, units
from dimtensor.visualization.plotly import line

# Data in meters
altitude = DimArray([0, 1000, 2000, 3000, 4000], units.m)
time = DimArray([0, 60, 120, 180, 240], units.s)

# Display in kilometers and minutes
fig = line(
    time,
    altitude,
    x_unit=units.minute,
    y_unit=units.km,
    title="Rocket Ascent",
    x_title="Time",
    y_title="Altitude"
)
fig.show()
# Automatically converted: Time [min], Altitude [km]
```

### Custom Styling

```python
from dimtensor import DimArray, units
from dimtensor.visualization.plotly import scatter

mass = DimArray([1, 2, 3, 4, 5], units.kg)
kinetic_energy = DimArray([5, 20, 45, 80, 125], units.J)

fig = scatter(mass, kinetic_energy, title="Kinetic Energy vs Mass")

# Customize appearance
fig.update_traces(
    marker=dict(size=15, color='darkblue', symbol='diamond'),
    line=dict(width=2, color='lightblue')
)

fig.update_layout(
    font=dict(size=14),
    plot_bgcolor='lightgray',
    width=800,
    height=600
)

fig.show()
```

## Error Bars and Uncertainty

Both matplotlib and plotly support automatic extraction of uncertainty from DimArray objects.

### Creating DimArray with Uncertainty

```python
from dimtensor import DimArray, units

# Method 1: Specify uncertainty in constructor
temperature = DimArray(
    [20.1, 25.3, 30.5],
    units.K,
    uncertainty=[0.5, 0.4, 0.6]
)

# Method 2: Use with_uncertainty method
distance = DimArray([10.0, 20.0, 30.0], units.m)
distance_with_error = distance.with_uncertainty([0.1, 0.2, 0.15])

# Check if uncertainty is present
print(temperature.has_uncertainty)  # True
print(temperature.uncertainty)  # [0.5, 0.4, 0.6]
```

### Matplotlib Error Bars

The `errorbar()` function automatically extracts uncertainty:

```python
from dimtensor import DimArray, units
from dimtensor.visualization import errorbar
import matplotlib.pyplot as plt

time = DimArray([0, 1, 2, 3, 4], units.s)
position = DimArray(
    [0.0, 4.9, 19.6, 44.1, 78.4],
    units.m,
    uncertainty=[0.1, 0.2, 0.3, 0.4, 0.5]
)

errorbar(time, position, fmt='o-', capsize=5, capthick=2, label='Measured')
plt.title("Position with Measurement Uncertainty")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Explicitly specifying errors:**

```python
from dimtensor import DimArray, units
from dimtensor.visualization import errorbar
import matplotlib.pyplot as plt

x = DimArray([1, 2, 3, 4], units.s)
y = DimArray([10, 20, 30, 40], units.m)
y_errors = DimArray([1, 1.5, 1.2, 1.8], units.m)

errorbar(x, y, yerr=y_errors, fmt='s-', capsize=4)
plt.title("Explicit Error Specification")
plt.show()
```

### Plotly Error Bars

The `scatter_with_errors()` function handles uncertainty:

```python
from dimtensor import DimArray, units
from dimtensor.visualization.plotly import scatter_with_errors

time = DimArray([0, 1, 2, 3, 4], units.s)
voltage = DimArray(
    [0.0, 1.5, 3.1, 4.4, 6.0],
    units.V,
    uncertainty=[0.1, 0.15, 0.2, 0.15, 0.1]
)

fig = scatter_with_errors(
    time,
    voltage,
    title="Voltage Measurement Over Time",
    x_title="Time",
    y_title="Voltage"
)
fig.update_traces(marker=dict(size=10))
fig.show()
```

**Error bars on both axes:**

```python
from dimtensor import DimArray, units
from dimtensor.visualization.plotly import scatter_with_errors

x = DimArray([1.0, 2.0, 3.0], units.s, uncertainty=[0.1, 0.1, 0.1])
y = DimArray([10.0, 20.0, 30.0], units.m, uncertainty=[0.5, 0.6, 0.5])

fig = scatter_with_errors(
    x, y,
    title="Errors on Both Axes",
    x_title="Time",
    y_title="Distance"
)
fig.show()
# Both x and y error bars shown
```

### Unit Conversion of Error Bars

Error bars are automatically converted along with the data:

```python
from dimtensor import DimArray, units
from dimtensor.visualization import errorbar
import matplotlib.pyplot as plt

# Data in meters with uncertainty
distance = DimArray(
    [1000, 2000, 3000, 4000],
    units.m,
    uncertainty=[50, 60, 55, 70]
)
time = DimArray([10, 20, 30, 40], units.s)

# Convert to kilometers for display
errorbar(time, distance, y_unit=units.km, fmt='o-', capsize=5)
plt.title("Distance with Converted Error Bars")
plt.show()
# y-axis: [km], error bars automatically scaled to km
```

## Customization

### Matplotlib Customization

All standard matplotlib parameters work with wrapper functions:

```python
from dimtensor import DimArray, units
from dimtensor.visualization import plot, scatter
import matplotlib.pyplot as plt

time = DimArray([0, 1, 2, 3, 4], units.s)
position = DimArray([0, 5, 20, 45, 80], units.m)

# Line style and color
plot(time, position,
     color='darkblue',
     linestyle='--',
     linewidth=2,
     marker='o',
     markersize=8,
     label='Trajectory')

plt.title("Customized Plot", fontsize=16, fontweight='bold')
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
```

**Combining multiple data series:**

```python
from dimtensor import DimArray, units
from dimtensor.visualization import plot
import matplotlib.pyplot as plt
import numpy as np

time = DimArray(np.linspace(0, 5, 50), units.s)

# Multiple trajectories
y1 = DimArray(4.9 * time._data**2, units.m)
y2 = DimArray(3.5 * time._data**2, units.m)
y3 = DimArray(6.0 * time._data**2, units.m)

plot(time, y1, 'r-', label='g = 9.8 m/s²')
plot(time, y2, 'g--', label='g = 7.0 m/s²')
plot(time, y3, 'b:', label='g = 12.0 m/s²')

plt.title("Free Fall with Different Gravities")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Plotly Customization

Customize plotly figures with `update_traces()` and `update_layout()`:

```python
from dimtensor import DimArray, units
from dimtensor.visualization.plotly import line
import numpy as np

time = DimArray(np.linspace(0, 10, 100), units.s)
signal = DimArray(5 * np.sin(2 * np.pi * time._data / 5), units.V)

fig = line(time, signal, title="Sinusoidal Voltage")

# Customize trace
fig.update_traces(
    line=dict(color='purple', width=3),
    hovertemplate='Time: %{x:.2f}<br>Voltage: %{y:.2f}<extra></extra>'
)

# Customize layout
fig.update_layout(
    font=dict(family='Arial', size=14),
    plot_bgcolor='#f0f0f0',
    hovermode='x unified',
    width=1000,
    height=600,
    showlegend=True
)

fig.show()
```

## Best Practices

### 1. Use Convenient Units for Display

Convert data to units appropriate for the scale:

```python
# Good: Long distances in km
distance = DimArray([5000, 10000, 15000], units.m)
plot(time, distance, y_unit=units.km)  # Displays 5, 10, 15 km

# Avoid: Keeping in meters makes values harder to read
plot(time, distance)  # Displays 5000, 10000, 15000 m
```

### 2. Leverage Uncertainty Auto-Extraction

Let dimtensor extract uncertainty automatically:

```python
# Good: Uncertainty automatically handled
data_with_error = DimArray([10, 20, 30], units.m, uncertainty=[0.5, 0.6, 0.5])
errorbar(x, data_with_error)

# Avoid: Manually specifying when uncertainty is already present
errorbar(x, data_with_error, yerr=data_with_error.uncertainty)  # Redundant
```

### 3. Add Titles and Context

Always add descriptive titles for publication-quality plots:

```python
from dimtensor.visualization import plot
import matplotlib.pyplot as plt

plot(time, distance)
plt.title("Projectile Motion Under Constant Acceleration", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(['Experimental Data'], loc='best')
plt.tight_layout()
```

### 4. Use Subplots for Related Data

Group related measurements in multi-panel figures:

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plot(time, position, ax=axes[0, 0])
axes[0, 0].set_title("Position")

plot(time, velocity, ax=axes[0, 1])
axes[0, 1].set_title("Velocity")

plot(time, acceleration, ax=axes[1, 0])
axes[1, 0].set_title("Acceleration")

plot(time, energy, ax=axes[1, 1])
axes[1, 1].set_title("Energy")

plt.tight_layout()
```

### 5. Choose the Right Plot Type

- **Line plots**: Continuous data, time series, functions
- **Scatter plots**: Discrete measurements, relationships
- **Bar charts**: Categorical comparisons, discrete categories
- **Histograms**: Distributions, frequency analysis
- **Error bars**: Experimental data with uncertainty

### 6. Save Figures for Publications

```python
import matplotlib.pyplot as plt
from dimtensor.visualization import plot

plot(time, distance)
plt.title("Experimental Results")
plt.savefig('results.png', dpi=300, bbox_inches='tight')
plt.savefig('results.pdf', bbox_inches='tight')  # Vector format for papers
```

## Comparison: Matplotlib vs Plotly

| Feature | Matplotlib | Plotly |
|---------|------------|--------|
| Output | Static images | Interactive HTML |
| Use case | Publications, reports | Web dashboards, exploration |
| Setup | `setup_matplotlib()` or wrappers | Import from `.plotly` |
| Customization | Extensive (mature ecosystem) | Modern, web-focused styling |
| File formats | PNG, PDF, SVG, etc | HTML, PNG (static export) |
| Performance | Fast for static plots | Slower for very large datasets |
| Jupyter | Inline display | Inline interactive widgets |

**When to use matplotlib:**
- Creating figures for academic papers
- Generating multiple subplots
- Need precise control over every element
- Creating static, high-resolution images

**When to use plotly:**
- Building interactive dashboards
- Exploring data interactively
- Sharing results via web pages
- Want hover tooltips and zoom capabilities

## Complete Example: Projectile Motion Analysis

```python
from dimtensor import DimArray, units
from dimtensor.visualization import plot, errorbar
import matplotlib.pyplot as plt
import numpy as np

# Simulate projectile motion with measurement uncertainty
t = np.linspace(0, 3, 30)
time = DimArray(t, units.s)

# True trajectory
v0 = DimArray([30.0], units.m / units.s)
angle = DimArray([45.0], units.deg)
g = DimArray([9.81], units.m / units.s**2)

# Calculate position (simplified 1D)
true_height = v0 * np.sin(angle._data * np.pi / 180) * time - 0.5 * g * time**2

# Add measurement noise
noise = np.random.normal(0, 0.5, len(t))
measured_height = DimArray(
    true_height._data + noise,
    units.m,
    uncertainty=0.5 * np.ones(len(t))
)

# Create publication-quality figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Top panel: trajectory with error bars
errorbar(time, measured_height, ax=ax1, fmt='o', capsize=4,
         alpha=0.6, label='Measured', color='blue')
plot(time, true_height, ax=ax1, linestyle='--',
     color='red', linewidth=2, label='Theoretical')
ax1.set_title("Projectile Motion: Height vs Time", fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Bottom panel: residuals
residuals = measured_height._data - true_height._data
residual_array = DimArray(residuals, units.m, uncertainty=measured_height.uncertainty)
errorbar(time, residual_array, ax=ax2, fmt='o', capsize=4, color='green')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.set_title("Measurement Residuals", fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('projectile_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Mean absolute error: {np.mean(np.abs(residuals)):.3f} m")
print(f"RMS error: {np.sqrt(np.mean(residuals**2)):.3f} m")
```

## Troubleshooting

### "matplotlib is not available"

```bash
pip install matplotlib
```

### "plotly is not available"

```bash
pip install plotly
```

### Units not showing on axes

Make sure you're using DimArray objects, not raw numpy arrays:

```python
# Wrong: raw numpy arrays have no units
plot(np.array([1, 2, 3]), np.array([4, 5, 6]))

# Correct: use DimArray
plot(DimArray([1, 2, 3], units.s), DimArray([4, 5, 6], units.m))
```

### UnitConversionError when converting

Ensure target units have compatible dimensions:

```python
# Wrong: incompatible dimensions
distance = DimArray([100], units.m)
plot(time, distance, y_unit=units.s)  # Error!

# Correct: compatible dimensions
plot(time, distance, y_unit=units.km)  # OK: m -> km
```

### Error bars not showing

Check that your DimArray has uncertainty:

```python
data = DimArray([1, 2, 3], units.m)
print(data.has_uncertainty)  # Should be True

# If False, add uncertainty:
data_with_error = data.with_uncertainty([0.1, 0.1, 0.1])
```

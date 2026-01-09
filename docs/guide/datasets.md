# Working with Physics Datasets

dimtensor includes a powerful dataset module for physics-aware machine learning. It provides built-in synthetic physics datasets, real-world data loaders, and tools for creating custom dataset loaders with automatic caching and unit tracking.

## Quick Start

```python
from dimtensor.datasets import list_datasets, load_dataset, get_dataset_info

# Discover available datasets
datasets = list_datasets()
print(f"Found {len(datasets)} datasets")

# Load a dataset
data = load_dataset("pendulum")
print(data)
```

## Discovering Datasets

The registry system lets you discover datasets by domain, tags, or name.

### List All Datasets

```python
from dimtensor.datasets import list_datasets

# Get all available datasets
all_datasets = list_datasets()

for ds in all_datasets:
    print(f"{ds.name}: {ds.description}")
    print(f"  Domain: {ds.domain}")
    print(f"  Tags: {', '.join(ds.tags)}")
    print()
```

### Filter by Domain

```python
# Find all mechanics datasets
mechanics = list_datasets(domain="mechanics")
print(f"Found {len(mechanics)} mechanics datasets")

# Find fluid dynamics datasets
fluids = list_datasets(domain="fluid_dynamics")

# Find thermodynamics datasets
thermo = list_datasets(domain="thermodynamics")
```

### Filter by Tags

```python
# Find all PDE datasets
pde_datasets = list_datasets(tags=["pde"])

# Find chaotic systems
chaotic = list_datasets(tags=["chaotic"])

# Find oscillation problems (must have ALL specified tags)
oscillators = list_datasets(tags=["oscillation"])
```

### Get Dataset Metadata

```python
from dimtensor.datasets import get_dataset_info

# Get detailed information about a dataset
info = get_dataset_info("burgers")

print(f"Name: {info.name}")
print(f"Description: {info.description}")
print(f"Domain: {info.domain}")

# View feature dimensions
print("\nFeatures:")
for name, dim in info.features.items():
    print(f"  {name}: {dim}")

# View target dimensions
print("\nTargets:")
for name, dim in info.targets.items():
    print(f"  {name}: {dim}")

# Tags and metadata
print(f"\nTags: {', '.join(info.tags)}")
print(f"Source: {info.source}")
```

## Built-in Synthetic Datasets

dimtensor includes 10 synthetic physics datasets for testing and benchmarking physics-informed neural networks.

### Classical Mechanics

#### Simple Pendulum

Oscillating pendulum with angle and angular velocity.

```python
from dimtensor.datasets import load_dataset

# Load pendulum dataset
data = load_dataset("pendulum", n_samples=100, length=1.0, g=9.8)

# Access features and targets
time = data["time"]
angle = data["angle"]
angular_velocity = data["angular_velocity"]

print(f"Time: {time.unit}")  # seconds
print(f"Angle: dimensionless")  # radians
print(f"Angular velocity: {angular_velocity.unit}")  # rad/s
```

**Features:**
- `time`: Time coordinate (s)
- `length`: Pendulum length (m)
- `initial_angle`: Starting angle (rad)

**Targets:**
- `angle`: Angular position (rad, dimensionless)
- `angular_velocity`: Angular velocity (rad/s)

**Tags:** `oscillation`, `classical`, `ode`

#### Projectile Motion

2D ballistic trajectory under gravity.

```python
# Load projectile motion dataset
data = load_dataset("projectile", n_samples=50, v0=20.0, angle=45.0)

x_pos = data["x_position"]
y_pos = data["y_position"]
x_vel = data["x_velocity"]
y_vel = data["y_velocity"]

print(f"Position: {x_pos.unit}")  # meters
print(f"Velocity: {x_vel.unit}")  # m/s
```

**Features:**
- `time`: Time coordinate (s)
- `initial_velocity`: Launch speed (m/s)
- `launch_angle`: Launch angle (rad, dimensionless)

**Targets:**
- `x_position`, `y_position`: Position coordinates (m)
- `x_velocity`, `y_velocity`: Velocity components (m/s)

**Tags:** `kinematics`, `classical`, `2d`

#### Spring-Mass System

Damped harmonic oscillator with position and velocity.

```python
# Load spring-mass system
data = load_dataset("spring_mass", n_samples=100, mass=1.0, k=10.0, damping=0.5)

position = data["position"]
velocity = data["velocity"]

print(f"Position: {position.unit}")  # meters
print(f"Velocity: {velocity.unit}")  # m/s
```

**Features:**
- `time`: Time coordinate (s)
- `mass`: Mass (kg)
- `spring_constant`: Spring stiffness (kg/s²)
- `damping`: Damping coefficient (kg/s)

**Targets:**
- `position`: Displacement (m)
- `velocity`: Velocity (m/s)

**Tags:** `oscillation`, `ode`, `damped`

#### Three-Body Problem

Gravitational trajectories of three interacting bodies.

```python
# Load three-body problem
data = load_dataset("three_body", n_samples=1000, duration=10.0)

positions = data["positions"]  # Shape: (n_samples, 3, 3) - 3 bodies in 3D
velocities = data["velocities"]

print(f"Positions: {positions.unit}")  # meters
print(f"Shape: {positions.shape}")
```

**Features:**
- `time`: Time coordinate (s)
- `masses`: Body masses (kg)
- `initial_positions`: Starting positions (m)
- `initial_velocities`: Starting velocities (m/s)

**Targets:**
- `positions`: Body positions over time (m)
- `velocities`: Body velocities over time (m/s)

**Tags:** `gravity`, `chaotic`, `n-body`

### Fluid Dynamics

#### Burgers Equation

1D viscous Burgers equation (simplified Navier-Stokes).

```python
# Load Burgers equation dataset
data = load_dataset("burgers", n_samples=100, viscosity=0.01)

position = data["position"]
time = data["time"]
velocity = data["velocity"]

print(f"Velocity field: {velocity.unit}")  # m/s
print(f"Shape: {velocity.shape}")
```

**Features:**
- `position`: Spatial coordinate (m)
- `time`: Time coordinate (s)
- `viscosity`: Kinematic viscosity (m²/s)

**Targets:**
- `velocity`: Velocity field (m/s)

**Tags:** `pde`, `nonlinear`, `cfd`

#### Navier-Stokes 2D

2D incompressible flow with velocity and pressure fields.

```python
# Load Navier-Stokes dataset
data = load_dataset("navier_stokes_2d", n_samples=50, reynolds=100)

u_velocity = data["u_velocity"]
v_velocity = data["v_velocity"]
pressure = data["pressure"]

print(f"Velocity: {u_velocity.unit}")  # m/s
print(f"Pressure: {pressure.unit}")  # Pa
```

**Features:**
- `x`, `y`: Spatial coordinates (m)
- `time`: Time coordinate (s)
- `reynolds_number`: Reynolds number (dimensionless)

**Targets:**
- `u_velocity`: x-component velocity (m/s)
- `v_velocity`: y-component velocity (m/s)
- `pressure`: Pressure field (Pa)

**Tags:** `pde`, `cfd`, `turbulence`

#### Lorenz System

Chaotic attractor from simplified atmospheric convection.

```python
# Load Lorenz system
data = load_dataset("lorenz", n_samples=1000, sigma=10, rho=28, beta=8/3)

x = data["x"]
y = data["y"]
z = data["z"]

print(f"Lorenz variables are dimensionless")
print(f"Shape: {x.shape}")
```

**Features:**
- `time`: Time coordinate (s)
- `sigma`, `rho`, `beta`: Lorenz parameters (dimensionless)

**Targets:**
- `x`, `y`, `z`: State variables (dimensionless)

**Tags:** `chaotic`, `ode`, `attractor`

### Partial Differential Equations

#### Heat Diffusion

1D heat equation on a rod.

```python
# Load heat diffusion dataset
data = load_dataset("heat_diffusion", n_samples=100, diffusivity=0.01)

position = data["position"]
time = data["time"]
temperature = data["temperature"]

print(f"Temperature: {temperature.unit}")  # Kelvin
print(f"Shape: {temperature.shape}")
```

**Features:**
- `position`: Spatial coordinate (m)
- `time`: Time coordinate (s)
- `thermal_diffusivity`: Thermal diffusivity (m²/s)

**Targets:**
- `temperature`: Temperature field (K)

**Tags:** `pde`, `diffusion`, `heat`

#### Wave Equation

1D wave propagation on a string.

```python
# Load wave equation dataset
data = load_dataset("wave_1d", n_samples=100, wave_speed=1.0)

position = data["position"]
time = data["time"]
displacement = data["displacement"]

print(f"Displacement: {displacement.unit}")  # meters
print(f"Wave speed: m/s")
```

**Features:**
- `position`: Spatial coordinate (m)
- `time`: Time coordinate (s)
- `wave_speed`: Wave propagation speed (m/s)

**Targets:**
- `displacement`: String displacement (m)

**Tags:** `pde`, `wave`, `hyperbolic`

### Thermodynamics

#### Ideal Gas

State variables related by PV = nRT.

```python
# Load ideal gas dataset
data = load_dataset("ideal_gas", n_samples=100)

pressure = data["pressure"]
volume = data["volume"]
temperature = data["temperature"]

print(f"Pressure: {pressure.unit}")  # Pa
print(f"Volume: {volume.unit}")  # m³
print(f"Temperature: {temperature.unit}")  # K
```

**Features:**
- `pressure`: Gas pressure (Pa)
- `volume`: Gas volume (m³)
- `moles`: Amount of substance (mol)

**Targets:**
- `temperature`: Gas temperature (K)

**Tags:** `thermodynamics`, `state`

## Real-World Data Loaders

dimtensor provides loaders for real physics datasets from NIST, NASA, and climate archives.

### NIST CODATA Fundamental Constants

Load physical constants from NIST CODATA 2022 with proper units.

```python
from dimtensor.datasets import load_dataset

# Load NIST constants
constants = load_dataset("nist_codata_2022")

# Access specific constants
c = constants["speed of light in vacuum"]
G = constants["Newtonian constant of gravitation"]
h = constants["Planck constant"]
hbar = constants["reduced Planck constant"]

print(f"Speed of light: {c}")
print(f"Gravitational constant: {G}")
print(f"Planck constant: {h}")
print(f"Reduced Planck: {hbar}")

# Use in calculations
from dimtensor import DimArray
from dimtensor.units import meter, second

wavelength = DimArray([500e-9], meter)  # 500 nm
frequency = c / wavelength
print(f"Frequency: {frequency.to(1/second)}")
```

**Features:**
- Constants loaded with proper SI units
- Includes uncertainties (where available)
- Fallback to built-in constants if download fails

**Data Source:** https://physics.nist.gov/cuu/Constants/

### NASA Exoplanet Archive

Load confirmed exoplanet data with masses, radii, and orbital parameters.

```python
# Load NASA exoplanet data
exoplanets = load_dataset("nasa_exoplanets")

# Access planet properties
names = exoplanets["pl_name"]  # List of planet names
masses = exoplanets["pl_masse"]  # Planet mass (Earth masses)
radii = exoplanets["pl_rade"]  # Planet radius (Earth radii)
periods = exoplanets["pl_orbper"]  # Orbital period (days)

print(f"Loaded {len(names)} exoplanets")
print(f"Masses: {masses.unit}")  # kg (Earth masses)
print(f"Radii: {radii.unit}")  # m (Earth radii)
print(f"Periods: {periods.unit}")  # days

# Filter by mass range
import numpy as np
earth_like = masses[(masses > DimArray([0.5], masses.unit)) &
                     (masses < DimArray([2.0], masses.unit))]
print(f"Found {len(earth_like)} Earth-like planets by mass")

# Convert units
from dimtensor.units import kg, meter, day

mass_kg = masses[0].to(kg)
radius_m = radii[0].to(meter)
period_days = periods[0].to(day)

print(f"First planet: {mass_kg}, {radius_m}, {period_days}")
```

**Available Fields:**
- `pl_name`: Planet name (string list)
- `pl_masse`: Planet mass (Earth masses)
- `pl_rade`: Planet radius (Earth radii)
- `pl_orbper`: Orbital period (days)
- `pl_orbsmax`: Semi-major axis (AU)
- `st_mass`: Stellar mass (Solar masses)
- `st_rad`: Stellar radius (Solar radii)

**Data Source:** https://exoplanetarchive.ipac.caltech.edu/

### PRISM Climate Data

Load temperature and precipitation time series (demo/sample data).

```python
# Load PRISM climate data
climate = load_dataset("prism_climate", variable="tmean", start_year=2020, end_year=2023)

# Access climate variables
dates = climate["dates"]  # List of date strings
temperatures = climate["values"]  # Temperature with units
latitude = climate["latitude"]
longitude = climate["longitude"]

print(f"Location: ({latitude}, {longitude})")
print(f"Temperature: {temperatures.unit}")  # °C
print(f"Data points: {len(dates)}")

# Load precipitation
precip = load_dataset("prism_climate", variable="ppt", start_year=2020)
precipitation = precip["values"]
print(f"Precipitation: {precipitation.unit}")  # mm

# Temperature statistics
import numpy as np
mean_temp = np.mean(temperatures.data)
max_temp = np.max(temperatures.data)
min_temp = np.min(temperatures.data)

print(f"Mean: {mean_temp:.1f} {temperatures.unit}")
print(f"Range: {min_temp:.1f} to {max_temp:.1f} {temperatures.unit}")
```

**Available Variables:**
- `tmean`: Mean temperature (°C)
- `tmin`: Minimum temperature (°C)
- `tmax`: Maximum temperature (°C)
- `ppt`: Precipitation (mm)

**Note:** This loader uses synthetic sample data for demonstration. For production use with real PRISM data, update the loader's URL configuration.

**Data Source:** https://prism.oregonstate.edu/

## Working with Loaded Data

All loaders return dictionaries with DimArray values that have proper physical units.

### Data Structure

```python
from dimtensor.datasets import load_dataset

# Load a dataset
data = load_dataset("pendulum", n_samples=100)

# Data is a dictionary
print(type(data))  # <class 'dict'>

# Keys are feature/target names
print(data.keys())  # dict_keys(['time', 'angle', 'angular_velocity', ...])

# Values are DimArrays with units
time = data["time"]
print(type(time))  # <class 'dimtensor.core.dimarray.DimArray'>
print(time.unit)  # s (seconds)
print(time.shape)  # (100,)
```

### Unit Conversions

Convert loaded data to different units.

```python
exoplanets = load_dataset("nasa_exoplanets")

# Planet masses in Earth masses
masses = exoplanets["pl_masse"]
print(f"Original unit: {masses.unit}")

# Convert to kilograms
from dimtensor.units import kg
masses_kg = masses.to(kg)
print(f"In kilograms: {masses_kg[0]}")

# Convert to Jupiter masses
jupiter_mass = kg * 1.898e27
masses_jupiter = masses.to(jupiter_mass)
print(f"In Jupiter masses: {masses_jupiter[0]}")
```

### Filtering and Slicing

Work with datasets using NumPy-style indexing.

```python
import numpy as np
from dimtensor import DimArray

# Load exoplanet data
exoplanets = load_dataset("nasa_exoplanets")
masses = exoplanets["pl_masse"]
radii = exoplanets["pl_rade"]

# Filter by mass range (Earth-like)
mask = (masses.data > 0.5) & (masses.data < 2.0)
earth_like_masses = DimArray(masses.data[mask], unit=masses.unit)
earth_like_radii = DimArray(radii.data[mask], unit=radii.unit)

print(f"Found {len(earth_like_masses)} Earth-like planets")

# Sort by mass
sorted_indices = np.argsort(masses.data)
sorted_masses = DimArray(masses.data[sorted_indices], unit=masses.unit)
```

### Integration with NumPy/PyTorch/JAX

Use loaded datasets with different array backends.

```python
from dimtensor.datasets import load_dataset
import numpy as np

# Load as NumPy DimArray
data = load_dataset("burgers", n_samples=100)
velocity_np = data["velocity"]

print(f"NumPy array: {type(velocity_np.data)}")

# Convert to PyTorch
try:
    from dimtensor.torch import DimTensor
    import torch

    velocity_torch = DimTensor(
        torch.tensor(velocity_np.data),
        unit=velocity_np.unit
    )
    print(f"PyTorch tensor: {type(velocity_torch.data)}")
except ImportError:
    print("PyTorch not available")

# Convert to JAX
try:
    from dimtensor.jax import DimArray as JAXDimArray
    import jax.numpy as jnp

    velocity_jax = JAXDimArray(
        jnp.array(velocity_np.data),
        unit=velocity_np.unit
    )
    print(f"JAX array: {type(velocity_jax.data)}")
except ImportError:
    print("JAX not available")
```

## Caching System

All data loaders use automatic caching to avoid repeated downloads.

### Cache Directory

By default, downloaded datasets are cached in `~/.dimtensor/cache/`.

```python
from dimtensor.datasets.loaders.base import get_cache_dir

# Get cache directory path
cache_dir = get_cache_dir()
print(f"Cache directory: {cache_dir}")

# List cached files
import os
if cache_dir.exists():
    files = list(cache_dir.iterdir())
    print(f"Cached files: {len(files)}")
    for f in files:
        print(f"  {f.name}")
```

### Custom Cache Directory

Override the cache location with an environment variable.

```bash
# Set custom cache directory
export DIMTENSOR_CACHE_DIR=/path/to/cache

# Or in Python
import os
os.environ["DIMTENSOR_CACHE_DIR"] = "/path/to/cache"
```

### Force Re-download

Bypass the cache and force a fresh download.

```python
from dimtensor.datasets import load_dataset

# Force re-download from source
exoplanets = load_dataset("nasa_exoplanets", force_download=True)

# This will download even if cached
constants = load_dataset("nist_codata_2022", force_download=True)
```

### Cache Management

Inspect and clear cached data.

```python
from dimtensor.datasets.loaders.nist import NISTCODATALoader

# Create loader instance
loader = NISTCODATALoader()

# Get cache info for a specific download
info = loader.get_cache_info("nist_codata_2022")
if info:
    print(f"URL: {info['url']}")
    print(f"Size: {info['size']} bytes")
    print(f"Content-Type: {info['content_type']}")

# Clear specific cache entry
loader.clear_cache("nist_codata_2022")
print("Cleared NIST cache")

# Clear all cache (use with caution!)
# loader.clear_cache()  # Clears all cached downloads
```

### Disable Caching

Disable caching entirely for a loader.

```python
from dimtensor.datasets.loaders.nist import NISTCODATALoader

# Create loader with caching disabled
loader = NISTCODATALoader(cache=False)

# Load data (will download every time)
constants = loader.load()
```

## Creating Custom Loaders

Extend the loader system to create your own dataset loaders.

### Extending BaseLoader

Create a custom loader for any data source.

```python
from dimtensor.datasets.loaders.base import BaseLoader
from dimtensor import DimArray
from dimtensor.units import meter, second
import numpy as np

class CustomPhysicsLoader(BaseLoader):
    """Load custom physics dataset."""

    URL = "https://example.com/physics_data.csv"

    def load(self, **kwargs):
        """Load and parse the dataset.

        Returns:
            Dictionary of DimArrays with proper units.
        """
        # Download with automatic caching
        filepath = self.download(
            self.URL,
            cache_key="custom_physics_data"
        )

        # Parse the file (implement your parsing logic)
        data = self._parse_file(filepath)

        return data

    def _parse_file(self, filepath):
        """Parse downloaded file into DimArrays."""
        # Read file
        content = filepath.read_text()
        lines = content.strip().split("\n")

        # Parse data (example: time, position, velocity)
        times = []
        positions = []
        velocities = []

        for line in lines[1:]:  # Skip header
            parts = line.split(",")
            if len(parts) >= 3:
                times.append(float(parts[0]))
                positions.append(float(parts[1]))
                velocities.append(float(parts[2]))

        # Return as DimArrays with units
        return {
            "time": DimArray(np.array(times), unit=second),
            "position": DimArray(np.array(positions), unit=meter),
            "velocity": DimArray(np.array(velocities), unit=meter/second),
        }

# Use the custom loader
loader = CustomPhysicsLoader()
data = loader.load()
print(f"Loaded {len(data['time'])} data points")
```

### Extending CSVLoader

Use the CSV utilities for comma-separated data.

```python
from dimtensor.datasets.loaders.base import CSVLoader
from dimtensor import DimArray
from dimtensor.units import kg, meter, second
import numpy as np

class ParticleDataLoader(CSVLoader):
    """Load particle physics dataset from CSV."""

    URL = "https://example.com/particles.csv"

    def load(self, **kwargs):
        """Load particle data with masses and momenta."""
        # Download with caching
        filepath = self.download(
            self.URL,
            cache_key="particle_data",
            force=kwargs.get("force_download", False)
        )

        # Parse CSV
        rows = self.parse_csv(filepath, skip_rows=1, delimiter=",")

        # Extract columns
        masses = []
        momenta_x = []
        momenta_y = []
        momenta_z = []

        for row in rows:
            if len(row) >= 4:
                masses.append(float(row[0]))
                momenta_x.append(float(row[1]))
                momenta_y.append(float(row[2]))
                momenta_z.append(float(row[3]))

        # Create momentum unit: kg*m/s
        momentum_unit = kg * meter / second

        return {
            "mass": DimArray(np.array(masses), unit=kg),
            "px": DimArray(np.array(momenta_x), unit=momentum_unit),
            "py": DimArray(np.array(momenta_y), unit=momentum_unit),
            "pz": DimArray(np.array(momenta_z), unit=momentum_unit),
        }

# Use the loader
loader = ParticleDataLoader()
data = loader.load()
```

### Registering Custom Datasets

Register your custom dataset in the registry.

```python
from dimtensor.datasets import register_dataset
from dimtensor.datasets.registry import DatasetInfo
from dimtensor.core.dimensions import Dimension

# Define metadata
info = DatasetInfo(
    name="custom_particles",
    description="Custom particle physics dataset with masses and momenta",
    domain="particle_physics",
    features={
        "mass": Dimension(mass=1),
    },
    targets={
        "px": Dimension(mass=1, length=1, time=-1),
        "py": Dimension(mass=1, length=1, time=-1),
        "pz": Dimension(mass=1, length=1, time=-1),
    },
    tags=["particles", "momentum", "custom"],
    source="https://example.com/particles.csv",
    license="MIT",
)

# Create loader function
def load_particle_data(**kwargs):
    loader = ParticleDataLoader()
    return loader.load(**kwargs)

# Register
register_dataset("custom_particles", info=info, loader=load_particle_data)

# Now it's available via load_dataset
from dimtensor.datasets import load_dataset, get_dataset_info

particle_info = get_dataset_info("custom_particles")
print(particle_info.description)

particles = load_dataset("custom_particles")
```

### Using Decorators

Register datasets using the decorator syntax.

```python
from dimtensor.datasets import register_dataset

@register_dataset(
    "my_dataset",
    description="My custom physics dataset",
    domain="custom",
    tags=["experimental"],
)
def load_my_dataset(**kwargs):
    """Load my custom dataset."""
    # Implement loading logic
    return {
        "data": DimArray([1, 2, 3], unit=meter),
    }

# Now available in registry
from dimtensor.datasets import load_dataset
data = load_dataset("my_dataset")
```

## Best Practices

### 1. Always Check Units

Verify units after loading to ensure correctness.

```python
data = load_dataset("nasa_exoplanets")
masses = data["pl_masse"]

# Check unit before calculations
print(f"Unit: {masses.unit}")
assert masses.unit.dimension.mass == 1
```

### 2. Handle Missing Data

Real-world datasets may have NaN values.

```python
import numpy as np

exoplanets = load_dataset("nasa_exoplanets")
masses = exoplanets["pl_masse"]

# Filter out NaN values
valid_mask = ~np.isnan(masses.data)
valid_masses = DimArray(masses.data[valid_mask], unit=masses.unit)

print(f"Valid data: {len(valid_masses)} / {len(masses)}")
```

### 3. Use Metadata for Documentation

Consult dataset metadata before use.

```python
from dimtensor.datasets import get_dataset_info

info = get_dataset_info("burgers")

# Check what features are available
print("Available features:")
for name, dim in info.features.items():
    print(f"  {name}: {dim}")

# Check tags to understand dataset type
if "pde" in info.tags:
    print("This is a PDE dataset")
```

### 4. Combine with Physics Constants

Use datasets with built-in constants for calculations.

```python
from dimtensor.constants.universal import G
from dimtensor.datasets import load_dataset

# Load exoplanet data
exoplanets = load_dataset("nasa_exoplanets")
st_mass = exoplanets["st_mass"]  # Stellar mass
pl_orbsmax = exoplanets["pl_orbsmax"]  # Semi-major axis

# Calculate orbital velocity (circular orbit approximation)
# v = sqrt(G * M / r)
import numpy as np
v_orbit = np.sqrt(G.value * st_mass.data / pl_orbsmax.data)

print(f"Orbital velocities calculated using G = {G}")
```

### 5. Cache Management

Monitor cache size and clean up periodically.

```python
from dimtensor.datasets.loaders.base import get_cache_dir
import os

cache_dir = get_cache_dir()

# Calculate cache size
total_size = sum(
    os.path.getsize(f)
    for f in cache_dir.iterdir()
    if f.is_file()
)

print(f"Cache size: {total_size / 1e6:.2f} MB")

# Clear old cache if too large
if total_size > 100e6:  # 100 MB
    print("Cache is large, consider clearing old files")
```

## Next Steps

- **[PyTorch Integration](pytorch.md)**: Use datasets with PyTorch for neural networks
- **[JAX Integration](jax.md)**: Leverage JAX for high-performance computing
- **[Physics-Informed ML](physics-ml.md)**: Build physics-informed neural networks
- **[Examples](examples.md)**: More physics calculation examples

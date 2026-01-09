---
title: Migration Guide - Moving to dimtensor
description: Comprehensive migration guide from Pint, Astropy units, unyt, or raw NumPy to dimtensor with side-by-side examples and ML integration.
---

# Migration Guide

This comprehensive guide helps you migrate existing code to dimtensor from other unit libraries or raw NumPy. We provide side-by-side code comparisons, common patterns, and show how to leverage dimtensor's unique features like PyTorch/JAX integration and physics-informed machine learning.

## Why Migrate to dimtensor?

dimtensor offers unique advantages for scientific computing and machine learning:

- **Native ML Integration**: Works seamlessly with PyTorch (autograd) and JAX (JIT/vmap/grad)
- **GPU Support**: Run unit-aware computations on CUDA and Apple Silicon
- **Built-in Uncertainty**: Automatic uncertainty propagation through all operations
- **Multiple Backends**: NumPy, PyTorch, and JAX with a unified API
- **Rich I/O**: JSON, HDF5, Parquet, NetCDF, pandas support
- **Physics-Informed ML**: Specialized layers and loss functions for PINNs
- **Dimensional Safety**: Catch physics errors at runtime, not after hours of computation

## From Raw NumPy

If you're adding units to existing NumPy code:

### Before (Raw NumPy)

```python
import numpy as np

# No unit safety - bugs can go unnoticed
velocity = np.array([10, 20, 30])  # m/s (implicit)
time = np.array([1, 2, 3])          # s (implicit)
distance = velocity * time          # Hopefully correct...

# This bug compiles fine but is physically wrong
acceleration = np.array([9.8])      # m/s^2 (implicit)
wrong = velocity + acceleration     # No error! But meaningless.
```

### After (dimtensor)

```python
from dimtensor import DimArray, units

# Units are explicit and checked
velocity = DimArray([10, 20, 30], units.m / units.s)
time = DimArray([1, 2, 3], units.s)
distance = velocity * time  # [10 40 90] m

# Bug caught immediately
acceleration = DimArray([9.8], units.m / units.s**2)
velocity + acceleration  # DimensionError: cannot add m/s to m/s^2
```

### Migration Steps

1. **Import dimtensor**
   ```python
   from dimtensor import DimArray, units
   ```

2. **Wrap arrays with units**
   ```python
   # Before
   data = np.array([1, 2, 3])

   # After
   data = DimArray([1, 2, 3], units.m)
   ```

3. **Access raw data when needed**
   ```python
   raw = data.data  # Get underlying numpy array
   ```

4. **Update function signatures**
   ```python
   # Before
   def calculate_energy(mass, velocity):
       return 0.5 * mass * velocity**2

   # After (works the same, but now type-safe)
   def calculate_energy(mass: DimArray, velocity: DimArray) -> DimArray:
       return 0.5 * mass * velocity**2
   ```

---

## From Pint

[Pint](https://pint.readthedocs.io/) is the most popular Python units library. This section provides detailed migration paths for common Pint patterns.

### Quick Reference

| Pint | dimtensor | Notes |
|------|-----------|-------|
| `ureg = pint.UnitRegistry()` | `from dimtensor import units` | No registry needed |
| `10 * ureg.meter` | `DimArray([10], units.m)` | Explicit array construction |
| `value.magnitude` | `arr.data` | Access underlying array |
| `value.units` | `arr.unit` | Get unit object |
| `value.to('km')` | `arr.to(units.km)` | Type-safe conversion |
| `Q_(10, 'kg')` | `DimArray([10], units.kg)` | Direct construction |

### Side-by-Side: Basic Operations

=== "Pint"

    ```python
    import pint
    import numpy as np

    ureg = pint.UnitRegistry()

    # Create quantities
    velocity = 10 * ureg.meter / ureg.second
    time = 5 * ureg.second

    # Arithmetic
    distance = velocity * time
    print(distance)  # 50 meter

    # Unit conversion
    distance_km = distance.to('kilometer')
    print(distance_km)  # 0.05 kilometer

    # Array operations
    velocities = np.array([10, 20, 30]) * ureg.m / ureg.s
    average = velocities.mean()

    # Get magnitude for plotting
    x_values = velocities.magnitude
    ```

=== "dimtensor"

    ```python
    from dimtensor import DimArray, units
    import numpy as np

    # No registry needed

    # Create quantities
    velocity = DimArray([10], units.m / units.s)
    time = DimArray([5], units.s)

    # Arithmetic
    distance = velocity * time
    print(distance)  # [50.] m

    # Unit conversion
    distance_km = distance.to(units.km)
    print(distance_km)  # [0.05] km

    # Array operations
    velocities = DimArray([10, 20, 30], units.m / units.s)
    average = velocities.mean()

    # Get magnitude for plotting
    x_values = velocities.data
    ```

### Side-by-Side: Advanced Features

=== "Pint"

    ```python
    # Uncertainty propagation (requires uncertainties package)
    from uncertainties import ufloat

    length = ufloat(10.0, 0.1) * ureg.meter
    width = ufloat(5.0, 0.05) * ureg.meter
    area = length * width
    print(area)  # 50.0+/-0.71 meter²

    # Temperature conversions
    temp_c = 25 * ureg.degC
    temp_f = temp_c.to('degF')

    # Context-based conversions
    with ureg.context('spectroscopy'):
        wavelength = 500 * ureg.nm
        frequency = wavelength.to('Hz')

    # Custom unit systems
    ureg.define('smoot = 1.7018 * meter')
    distance = 100 * ureg.smoot
    ```

=== "dimtensor"

    ```python
    # Built-in uncertainty propagation
    from dimtensor import DimArray, units

    length = DimArray([10.0], units.m, uncertainty=[0.1])
    width = DimArray([5.0], units.m, uncertainty=[0.05])
    area = length * width
    print(area)  # [50.] ± [...] m^2

    # Temperature conversions
    temp_c = DimArray([25], units.K)  # Use Kelvin
    # For Celsius: temp_k = temp_c + 273.15

    # Manual conversions with constants
    from dimtensor import constants
    wavelength = DimArray([500e-9], units.m)
    frequency = constants.c / wavelength

    # Domain-specific units
    from dimtensor.domains.astronomy import parsec
    distance = DimArray([100], parsec)
    ```

### Migration Patterns

#### Pattern 1: Creating Quantities

```python
# Pint: Multiplication syntax
mass_pint = 2.5 * ureg.kilogram
velocity_pint = np.array([10, 20]) * ureg.m / ureg.s

# dimtensor: Constructor syntax
mass_dim = DimArray([2.5], units.kg)
velocity_dim = DimArray([10, 20], units.m / units.s)
```

#### Pattern 2: Unit Conversions

```python
# Pint: String-based conversion
distance_pint = 1000 * ureg.meter
distance_km_pint = distance_pint.to('kilometer')

# dimtensor: Type-safe conversion
distance_dim = DimArray([1000], units.m)
distance_km_dim = distance_dim.to(units.km)
```

#### Pattern 3: Accessing Raw Values

```python
# Pint
value_pint = quantity.magnitude
unit_pint = quantity.units

# dimtensor
value_dim = quantity.data  # or .magnitude()
unit_dim = quantity.unit
```

#### Pattern 4: Dimensionless Quantities

```python
# Pint
ratio_pint = (5 * ureg.meter) / (10 * ureg.meter)
# Result: 0.5 dimensionless

# dimtensor
ratio_dim = DimArray([5], units.m) / DimArray([10], units.m)
# Result: [0.5] dimensionless
```

### Migration Script

Automated find-and-replace patterns:

```python
# Import statements
# FROM: import pint
# TO:   from dimtensor import DimArray, units

# Unit registry
# FROM: ureg = pint.UnitRegistry()
# TO:   # No registry needed

# Unit creation
# FROM: quantity * ureg.meter
# TO:   DimArray([quantity], units.m)

# Unit access
# FROM: .magnitude
# TO:   .data

# FROM: .units
# TO:   .unit

# Unit conversion
# FROM: .to('kilometer')
# TO:   .to(units.km)
```

### What's Different?

**Advantages of dimtensor:**
- ✅ Native PyTorch/JAX support for ML workflows
- ✅ Built-in uncertainty propagation (no extra packages)
- ✅ GPU acceleration (CUDA, MPS)
- ✅ Multiple I/O formats (HDF5, Parquet, NetCDF)
- ✅ Type-safe unit conversions

**Advantages of Pint:**
- ✅ Highly customizable unit systems
- ✅ Context-based unit conversions
- ✅ Temperature scales (Celsius, Fahrenheit)
- ✅ Mature ecosystem with extensive documentation
- ✅ Flexible unit definitions

### PyTorch Integration (dimtensor only)

The key advantage - train neural networks with unit checking:

```python
# This is ONLY possible with dimtensor
import torch
from dimtensor.torch import DimTensor, DimLinear
from dimtensor import units, Dimension

# Create unit-aware tensors with gradients
mass = DimTensor(
    torch.tensor([1.0, 2.0], requires_grad=True),
    units.kg
)
velocity = DimTensor(torch.tensor([10.0, 20.0]), units.m / units.s)

# Compute kinetic energy with autograd
energy = 0.5 * mass * velocity**2

# Backpropagation works!
loss = energy.sum()
loss.data.backward()
print(mass.grad)  # Gradients with correct units

# Dimension-aware neural network layers
layer = DimLinear(
    in_features=3,
    out_features=3,
    input_dim=Dimension(L=1),      # Length
    output_dim=Dimension(L=1, T=-1) # Velocity
)

position = DimTensor(torch.randn(32, 3), units.m)
velocity_pred = layer(position)  # Output is m/s
```

**Pint cannot do this** - you must strip units before PyTorch operations and manually track dimensions.

---

## From Astropy

[Astropy](https://docs.astropy.org/en/stable/units/) provides comprehensive units for astronomy and astrophysics. This section shows how to migrate while gaining ML capabilities.

### Quick Reference

| Astropy | dimtensor | Notes |
|---------|-----------|-------|
| `from astropy import units as u` | `from dimtensor import units` | Similar import style |
| `10 * u.m` | `DimArray([10], units.m)` | Explicit construction |
| `value.value` | `arr.data` | Get raw numpy array |
| `value.unit` | `arr.unit` | Get unit object |
| `value.to(u.km)` | `arr.to(units.km)` | Unit conversion |
| `u.Quantity([1,2,3], 'm')` | `DimArray([1,2,3], units.m)` | Array creation |

### Side-by-Side: Basic Astronomy

=== "Astropy"

    ```python
    from astropy import units as u
    from astropy import constants as const
    import numpy as np

    # Stellar physics
    mass = 2.0 * u.solMass
    radius = 1.5 * u.solRad

    # Luminosity
    sigma = const.sigma_sb
    T = 5800 * u.K
    L = 4 * np.pi * radius**2 * sigma * T**4

    # Distance
    distance = 10 * u.pc
    distance_ly = distance.to(u.lyr)

    # Spectroscopy (with equivalencies)
    wavelength = 500 * u.nm
    freq = wavelength.to(u.Hz, equivalencies=u.spectral())
    energy = wavelength.to(u.eV, equivalencies=u.spectral())
    ```

=== "dimtensor"

    ```python
    from dimtensor import DimArray, units, constants
    from dimtensor.domains.astronomy import solar_mass, solar_radius
    import numpy as np

    # Stellar physics
    mass = DimArray([2.0], solar_mass)
    radius = DimArray([1.5], solar_radius)

    # Luminosity
    sigma = constants.stefan_boltzmann
    T = DimArray([5800], units.K)
    L = 4 * np.pi * radius**2 * sigma * T**4

    # Distance
    from dimtensor.domains.astronomy import parsec, light_year
    distance = DimArray([10], parsec)
    distance_ly = distance.to(light_year)

    # Spectroscopy (manual conversion)
    wavelength = DimArray([500e-9], units.m)
    freq = constants.c / wavelength
    energy = constants.h * freq
    ```

### Side-by-Side: Advanced Features

=== "Astropy"

    ```python
    # Coordinate transformations
    from astropy.coordinates import SkyCoord

    coord = SkyCoord(ra=10.68458*u.deg, dec=41.26917*u.deg)

    # Cosmological calculations
    from astropy.cosmology import Planck18

    z = 2.0  # Redshift
    d_L = Planck18.luminosity_distance(z)

    # Equivalencies for spectral units
    with u.set_enabled_equivalencies(u.spectral()):
        wavelength = 21 * u.cm
        frequency = wavelength.to(u.GHz)

    # FITS file I/O
    from astropy.io import fits
    from astropy.nddata import NDDataArray

    data = fits.getdata('image.fits')
    quantity = data * u.Jy
    ```

=== "dimtensor"

    ```python
    # Focus on physical calculations, not coordinates
    # (Use astropy for coordinate transformations)

    # Cosmological calculations with dimtensor
    from dimtensor import DimArray, units, constants

    # Manual cosmology (simplified)
    z = 2.0
    H0 = DimArray([70], units.km / units.s / units.Mpc)
    # Implement distance calculation...

    # Spectral conversions
    wavelength = DimArray([0.21], units.m)  # 21 cm
    frequency = constants.c / wavelength
    # Result in Hz

    # I/O with dimtensor
    from dimtensor.io import save_to_hdf5, load_from_hdf5
    import numpy as np

    data = np.loadtxt('data.txt')
    flux = DimArray(data, units.Jy)
    save_to_hdf5(flux, 'data.h5', 'flux')
    ```

### Migration Patterns

#### Pattern 1: Physical Constants

```python
# Astropy
from astropy import constants as const

c = const.c              # Speed of light
G = const.G              # Gravitational constant
h = const.h              # Planck constant
m_e = const.m_e          # Electron mass

# dimtensor
from dimtensor import constants

c = constants.c
G = constants.G
h = constants.h
m_e = constants.electron_mass
```

#### Pattern 2: Astronomy Units

```python
# Astropy
from astropy import units as u

distance = 100 * u.pc
mass = 10 * u.solMass
luminosity = 1e38 * u.erg / u.s

# dimtensor
from dimtensor import DimArray, units
from dimtensor.domains.astronomy import parsec, solar_mass

distance = DimArray([100], parsec)
mass = DimArray([10], solar_mass)
luminosity = DimArray([1e38], units.erg / units.s)
```

#### Pattern 3: Array Operations

```python
# Astropy
masses = np.array([1, 2, 3, 4, 5]) * u.solMass
total_mass = masses.sum()
avg_mass = masses.mean()

# dimtensor
masses = DimArray([1, 2, 3, 4, 5], solar_mass)
total_mass = masses.sum()
avg_mass = masses.mean()
```

### What's Different?

**Advantages of dimtensor:**
- ✅ PyTorch/JAX integration for ML
- ✅ GPU acceleration
- ✅ Built-in uncertainty propagation
- ✅ Physics-informed neural network layers
- ✅ Multiple I/O formats

**Advantages of Astropy:**
- ✅ Comprehensive astronomy tooling
- ✅ Coordinate transformations
- ✅ Cosmology calculations
- ✅ FITS file I/O
- ✅ Spectral equivalencies
- ✅ Time and WCS handling

### Using Both Libraries

For astronomy projects needing ML, use both:

```python
# Load data with Astropy
from astropy.io import fits
from astropy import units as u

data = fits.getdata('observations.fits')
flux_astropy = data * u.Jy

# Convert to dimtensor for ML
from dimtensor.torch import DimTensor
from dimtensor import units as dt_units
import torch

flux_values = flux_astropy.to(u.W / u.m**2 / u.Hz).value
flux_dimtensor = DimTensor(
    torch.tensor(flux_values, dtype=torch.float32),
    dt_units.W / dt_units.m**2 / dt_units.Hz
)

# Train neural network with unit checking...
model = AstronomyNet()
predictions = model(flux_dimtensor)
```

### Machine Learning with Astronomy Data

Only possible with dimtensor:

```python
import torch
import torch.nn as nn
from dimtensor.torch import DimTensor, DimLinear
from dimtensor import units, Dimension
from dimtensor.domains.astronomy import solar_mass, solar_radius

# Build a stellar parameter estimator
class StellarNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: Temperature [K]
        # Output: Luminosity [L_sun]
        self.net = DimLinear(
            in_features=1,
            out_features=1,
            input_dim=Dimension(Theta=1),     # Temperature
            output_dim=Dimension(M=1, L=2, T=-3)  # Power (Luminosity)
        )

    def forward(self, temperature):
        return self.net(temperature)

# Training data
T_train = DimTensor(torch.randn(100, 1) * 1000 + 5000, units.K)
L_train = DimTensor(torch.randn(100, 1), solar_luminosity)

model = StellarNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    L_pred = model(T_train)
    loss = ((L_pred - L_train) ** 2).sum()

    optimizer.zero_grad()
    loss.data.backward()
    optimizer.step()
```

---

## From unyt

[unyt](https://unyt.readthedocs.io/) is the units library from the yt project, optimized for astrophysical simulations. This section helps you migrate to dimtensor for ML workflows.

### Quick Reference

| unyt | dimtensor | Notes |
|------|-----------|-------|
| `unyt_array([1,2,3], 'm')` | `DimArray([1,2,3], units.m)` | Similar construction |
| `arr.v` or `arr.value` | `arr.data` | Access raw array |
| `arr.units` | `arr.unit` | Get unit object |
| `arr.to('km')` | `arr.to(units.km)` | Unit conversion |
| `unyt_quantity(5, 'kg')` | `DimArray([5], units.kg)` | Scalar quantity |

### Side-by-Side: Basic Operations

=== "unyt"

    ```python
    from unyt import unyt_array
    import numpy as np

    # Create arrays with units
    velocity = unyt_array([10, 20, 30], 'm/s')
    time = unyt_array([1, 2, 3], 's')

    # Arithmetic
    distance = velocity * time
    print(distance)  # [10 40 90] m

    # Unit conversion
    distance_km = distance.to('km')
    print(distance_km)  # [0.01 0.04 0.09] km

    # Array operations
    avg_velocity = velocity.mean()
    total_distance = distance.sum()

    # Get raw values
    values = velocity.v  # or velocity.value
    ```

=== "dimtensor"

    ```python
    from dimtensor import DimArray, units
    import numpy as np

    # Create arrays with units
    velocity = DimArray([10, 20, 30], units.m / units.s)
    time = DimArray([1, 2, 3], units.s)

    # Arithmetic
    distance = velocity * time
    print(distance)  # [10 40 90] m

    # Unit conversion
    distance_km = distance.to(units.km)
    print(distance_km)  # [0.01 0.04 0.09] km

    # Array operations
    avg_velocity = velocity.mean()
    total_distance = distance.sum()

    # Get raw values
    values = velocity.data
    ```

### Side-by-Side: Astrophysical Simulations

=== "unyt"

    ```python
    from unyt import unyt_array
    from unyt.unit_systems import cgs_unit_system

    # Cosmological units
    distance = unyt_array([100], 'Mpc/h')  # Comoving distance

    # CGS units
    density = unyt_array([1e-24], 'g/cm**3')
    density_cgs = density.in_cgs()

    # Integration with yt
    import yt
    ds = yt.load("simulation.hdf5")
    ad = ds.all_data()

    # Units handled automatically
    temp = ad['gas', 'temperature']
    density = ad['gas', 'density']
    ```

=== "dimtensor"

    ```python
    from dimtensor import DimArray, units

    # Standard units (convert from Mpc/h manually)
    h = 0.7
    distance = DimArray([100 * h], units.Mpc)

    # CGS units
    density = DimArray([1e-24], units.g / units.cm**3)
    # Already in CGS, or use .to_base_units() for SI

    # For yt integration, extract then wrap
    import yt
    ds = yt.load("simulation.hdf5")
    ad = ds.all_data()

    # Convert yt data to dimtensor
    temp_values = ad['gas', 'temperature'].to('K').v
    temp = DimArray(temp_values, units.K)

    density_values = ad['gas', 'density'].to('g/cm**3').v
    density = DimArray(density_values, units.g / units.cm**3)
    ```

### Migration Patterns

#### Pattern 1: Array Creation

```python
# unyt: String-based units
velocity_unyt = unyt_array([10, 20, 30], 'm/s')
mass_unyt = unyt_array([1.0], 'kg')

# dimtensor: Object-based units
velocity_dim = DimArray([10, 20, 30], units.m / units.s)
mass_dim = DimArray([1.0], units.kg)
```

#### Pattern 2: Accessing Values

```python
# unyt: .v or .value
values_unyt = array.v
values_unyt_alt = array.value

# dimtensor: .data
values_dim = array.data
```

#### Pattern 3: Unit Systems

```python
# unyt: Built-in unit systems
from unyt.unit_systems import cgs_unit_system, mks_unit_system

quantity_unyt = unyt_array([1.0], 'erg')
quantity_cgs = quantity_unyt.in_cgs()
quantity_mks = quantity_unyt.in_mks()

# dimtensor: Manual conversion to SI base units
quantity_dim = DimArray([1.0], units.erg)
quantity_si = quantity_dim.to_base_units()  # Convert to kg⋅m²/s²
```

### What's Different?

**Advantages of dimtensor:**
- ✅ PyTorch/JAX integration
- ✅ GPU acceleration
- ✅ Built-in uncertainty propagation
- ✅ Physics-informed neural network layers
- ✅ Multiple I/O formats

**Advantages of unyt:**
- ✅ Native yt integration
- ✅ Cosmological unit systems
- ✅ Dask array support
- ✅ Optimized for large simulations
- ✅ CGS unit system

### Using Both Libraries

For yt-based projects needing ML:

```python
import yt
from dimtensor.torch import DimTensor
from dimtensor import units
import torch

# Load simulation with yt (uses unyt internally)
ds = yt.load("simulation.hdf5")
ad = ds.all_data()

# Extract data
density_unyt = ad['gas', 'density']
temperature_unyt = ad['gas', 'temperature']

# Convert to dimtensor for ML
density_values = density_unyt.to('g/cm**3').v
temperature_values = temperature_unyt.to('K').v

density = DimTensor(
    torch.tensor(density_values, dtype=torch.float32),
    units.g / units.cm**3
)
temperature = DimTensor(
    torch.tensor(temperature_values, dtype=torch.float32),
    units.K
)

# Train neural network on simulation data
model = SimulationNet()
predictions = model(density, temperature)
```

### Machine Learning with Simulation Data

Only possible with dimtensor:

```python
import torch
import torch.nn as nn
from dimtensor.torch import DimTensor, DimLinear
from dimtensor import units, Dimension

# Build a network to predict pressure from density and temperature
class EOS_Net(nn.Module):
    """Equation of state neural network."""

    def __init__(self):
        super().__init__()

        # Input: density [g/cm³] and temperature [K]
        # Output: pressure [dyn/cm²]
        self.density_layer = DimLinear(
            1, 32,
            input_dim=Dimension(M=1, L=-3),    # Density
            output_dim=Dimension()             # Dimensionless
        )

        self.temp_layer = DimLinear(
            1, 32,
            input_dim=Dimension(Theta=1),      # Temperature
            output_dim=Dimension()             # Dimensionless
        )

        self.output_layer = DimLinear(
            64, 1,
            input_dim=Dimension(),
            output_dim=Dimension(M=1, L=-1, T=-2)  # Pressure
        )

        self.hidden = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, density, temperature):
        d_features = self.density_layer(density)
        t_features = self.temp_layer(temperature)

        combined = torch.cat([d_features.data, t_features.data], dim=1)
        hidden = self.hidden(combined)

        hidden_dim = DimTensor._from_tensor_and_unit(
            hidden, units.dimensionless
        )
        pressure = self.output_layer(hidden_dim)
        return pressure

# Training loop
model = EOS_Net()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    # Simulation data
    rho = DimTensor(torch.randn(32, 1), units.g / units.cm**3)
    T = DimTensor(torch.randn(32, 1), units.K)
    P_target = DimTensor(torch.randn(32, 1), units.dyne / units.cm**2)

    P_pred = model(rho, T)
    loss = ((P_pred - P_target) ** 2).sum()

    optimizer.zero_grad()
    loss.data.backward()
    optimizer.step()
```

---

## PyTorch Integration

**This is unique to dimtensor** - other unit libraries (Pint, Astropy, unyt) don't support PyTorch autograd. This section shows how to add unit checking to existing PyTorch code.

### Why Add Units to PyTorch?

When training physics models, unit errors can hide for epochs before manifesting as poor predictions:

```python
# WITHOUT units - bug goes unnoticed
import torch

mass = torch.tensor([1.0], requires_grad=True)
velocity = torch.tensor([10.0])
# Oops, velocity is in km/s but model expects m/s!
# Model trains but predictions are off by 1000x
```

With dimtensor, these bugs are caught immediately:

```python
# WITH units - bug caught at training time
from dimtensor.torch import DimTensor
from dimtensor import units

mass = DimTensor(torch.tensor([1.0], requires_grad=True), units.kg)
velocity = DimTensor(torch.tensor([10.0]), units.km / units.s)

# If model expects m/s, conversion is explicit:
velocity_ms = velocity.to(units.m / units.s)
# Or dimension error is raised immediately
```

### Side-by-Side: Basic Training

=== "Raw PyTorch"

    ```python
    import torch
    import torch.nn as nn

    # No unit checking - errors hide in training
    mass = torch.tensor([1.0, 2.0], requires_grad=True)
    velocity = torch.tensor([10.0, 20.0])
    energy = 0.5 * mass * velocity**2

    # Training loop
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(100):
        prediction = model(velocity.reshape(-1, 1))
        loss = ((prediction - energy.reshape(-1, 1)) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ```

=== "dimtensor + PyTorch"

    ```python
    import torch
    import torch.nn as nn
    from dimtensor.torch import DimTensor, DimLinear
    from dimtensor import units, Dimension

    # Unit checking at runtime
    mass = DimTensor(
        torch.tensor([1.0, 2.0], requires_grad=True),
        units.kg
    )
    velocity = DimTensor(torch.tensor([10.0, 20.0]), units.m / units.s)
    energy = 0.5 * mass * velocity**2  # Automatically in Joules

    # Dimension-aware layer
    model = DimLinear(
        1, 1,
        input_dim=Dimension(L=1, T=-1),    # Velocity
        output_dim=Dimension(M=1, L=2, T=-2)  # Energy
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(100):
        prediction = model(velocity.reshape(-1, 1))
        loss = ((prediction - energy.reshape(-1, 1)) ** 2).mean()

        optimizer.zero_grad()
        loss.data.backward()  # .data to get raw tensor
        optimizer.step()

        # Dimension errors would be caught immediately!
    ```

### Common Patterns

#### Pattern 1: Adding Units to Existing Tensors

```python
# Before: Raw tensors
position = torch.randn(32, 3, requires_grad=True)
velocity = torch.randn(32, 3)

# After: Add units
from dimtensor.torch import DimTensor
from dimtensor import units

position = DimTensor(
    torch.randn(32, 3, requires_grad=True),
    units.m
)
velocity = DimTensor(torch.randn(32, 3), units.m / units.s)
```

#### Pattern 2: Dimension-Aware Neural Networks

```python
# Before: Standard PyTorch layers
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(3, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
)

# After: Dimension-aware layers
from dimtensor.torch import DimLinear, DimSequential
from dimtensor import Dimension

L = Dimension(L=1)        # Length
V = Dimension(L=1, T=-1)  # Velocity

model = DimSequential(
    DimLinear(3, 64, input_dim=L, output_dim=V),
    # Standard layers work too (use .data to unwrap)
    DimLinear(64, 3, input_dim=V, output_dim=V)
)
```

#### Pattern 3: Physics-Informed Loss Functions

```python
# Before: Standard MSE loss
import torch.nn.functional as F

def loss_fn(pred, target):
    return F.mse_loss(pred, target)

# After: Dimension-aware + physics constraints
from dimtensor.torch import DimMSELoss, PhysicsLoss, CompositeLoss

composite_loss = CompositeLoss(
    data_loss=DimMSELoss(),
    physics_losses={
        'energy': (PhysicsLoss(), 0.1),    # 10% weight
        'momentum': (PhysicsLoss(), 0.05)  # 5% weight
    }
)

def loss_fn(pred, target, initial_state, final_state):
    # Compute conserved quantities
    E_init = compute_energy(initial_state)
    E_final = compute_energy(final_state)

    return composite_loss(
        pred, target,
        physics_terms={'energy': (E_init, E_final)}
    )
```

#### Pattern 4: GPU Acceleration

```python
# Before: Move to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MyModel().to(device)
x = torch.randn(100, 10).to(device)
y = model(x)

# After: Units preserved on GPU
from dimtensor.torch import DimTensor
from dimtensor import units

model = MyModel().to(device)

x = DimTensor(torch.randn(100, 10), units.m).to(device)
y = model(x)  # Units preserved!

# Verify device and units
print(f"Result device: {y.device}")  # cuda:0
print(f"Result unit: {y.unit}")      # m/s
```

### Complete Example: Physics-Informed Neural Network

Solve the 1D heat equation: ∂T/∂t = α ∂²T/∂x²

```python
import torch
import torch.nn as nn
from dimtensor import units
from dimtensor.torch import DimScaler

class HeatPINN(nn.Module):
    """Predict temperature T(x,t) for 1D heat equation."""

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha  # Thermal diffusivity [m²/s]

        self.net = nn.Sequential(
            nn.Linear(2, 64),   # Input: (x, t)
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)    # Output: T
        )

    def forward(self, x, t):
        """Predict temperature at position x and time t.

        Args:
            x: Position [m] (scaled)
            t: Time [s] (scaled)

        Returns:
            T: Temperature [K] (scaled)
        """
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

    def pde_residual(self, x, t):
        """Compute PDE residual (should be zero).

        residual = ∂T/∂t - α ∂²T/∂x²
        """
        T = self.forward(x, t)

        # Automatic differentiation for PDE
        dT_dt = torch.autograd.grad(
            T, t,
            grad_outputs=torch.ones_like(T),
            create_graph=True
        )[0]

        dT_dx = torch.autograd.grad(
            T, x,
            grad_outputs=torch.ones_like(T),
            create_graph=True
        )[0]

        d2T_dx2 = torch.autograd.grad(
            dT_dx, x,
            grad_outputs=torch.ones_like(dT_dx),
            create_graph=True
        )[0]

        # PDE residual
        return dT_dt - self.alpha * d2T_dx2

# Training setup
model = HeatPINN(alpha=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(5000):
    # Sample collocation points
    x = torch.rand(100, 1, requires_grad=True)
    t = torch.rand(100, 1, requires_grad=True)

    # PDE residual should be zero
    residual = model.pde_residual(x, t)
    pde_loss = (residual ** 2).mean()

    # Boundary conditions
    # T(0, t) = 100 K, T(1, t) = 300 K
    x_left = torch.zeros(20, 1)
    x_right = torch.ones(20, 1)
    t_bc = torch.rand(20, 1)

    T_left = model(x_left, t_bc)
    T_right = model(x_right, t_bc)

    bc_loss = (T_left - 100)**2 + (T_right - 300)**2
    bc_loss = bc_loss.mean()

    # Total loss
    loss = pde_loss + 10.0 * bc_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: PDE={pde_loss.item():.6f}, BC={bc_loss.item():.6f}")
```

---

## JAX Integration

**This is unique to dimtensor** - other unit libraries don't support JAX's JIT compilation, vmap, or grad. This section shows how to add unit checking while keeping JAX's performance benefits.

### Why Add Units to JAX?

JAX is designed for high-performance numerical computing, but dimensional errors can still hide in JIT-compiled code:

```python
# WITHOUT units - bug only appears after JIT compilation
import jax
import jax.numpy as jnp

@jax.jit
def physics_sim(x, v, dt):
    # Oops, dt is in milliseconds but code expects seconds!
    return x + v * dt  # Results off by 1000x

# Bug is hard to trace in compiled code
```

With dimtensor, errors are caught before or during JIT compilation:

```python
# WITH units - dimension errors caught immediately
from dimtensor.jax import DimArray
from dimtensor import units

@jax.jit
def physics_sim(x, v, dt):
    return x + v * dt  # Units checked automatically

x = DimArray(jnp.array([0.0]), units.m)
v = DimArray(jnp.array([10.0]), units.m / units.s)
dt = DimArray(jnp.array([0.001]), units.ms)

# If dt needs to be in seconds, convert explicitly:
dt_s = dt.to(units.s)
```

### Side-by-Side: JIT Compilation

=== "Raw JAX"

    ```python
    import jax
    import jax.numpy as jnp

    @jax.jit
    def kinetic_energy(mass, velocity):
        """KE = 0.5 * m * v²"""
        return 0.5 * mass * velocity**2

    # No unit checking
    m = jnp.array([1.0, 2.0, 3.0])
    v = jnp.array([10.0, 20.0, 30.0])

    # Fast JIT-compiled execution
    E = kinetic_energy(m, v)
    print(E)  # [50. 200. 450.] (what units?)
    ```

=== "dimtensor + JAX"

    ```python
    import jax
    import jax.numpy as jnp
    from dimtensor.jax import DimArray
    from dimtensor import units

    @jax.jit
    def kinetic_energy(mass, velocity):
        """KE = 0.5 * m * v²"""
        return 0.5 * mass * velocity**2

    # Unit checking + JIT compilation
    m = DimArray(jnp.array([1.0, 2.0, 3.0]), units.kg)
    v = DimArray(jnp.array([10.0, 20.0, 30.0]), units.m / units.s)

    # Fast AND dimensionally safe
    E = kinetic_energy(m, v)
    print(E)  # [50. 200. 450.] J (Joules!)
    ```

### Side-by-Side: Vectorization with vmap

=== "Raw JAX"

    ```python
    import jax

    def force(mass, acceleration):
        """F = m * a"""
        return mass * acceleration

    # Vectorize over batch
    masses = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    accels = jnp.array([9.8, 5.0, 2.5, 10.0, 1.0])

    # vmap applies function to each element
    forces = jax.vmap(force)(masses, accels)
    print(forces)  # [9.8, 10.0, 7.5, 40.0, 5.0] (what units?)
    ```

=== "dimtensor + JAX"

    ```python
    import jax
    from dimtensor.jax import DimArray
    from dimtensor import units

    def force(mass, acceleration):
        """F = m * a"""
        return mass * acceleration

    # Vectorize with units
    masses = DimArray(
        jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        units.kg
    )
    accels = DimArray(
        jnp.array([9.8, 5.0, 2.5, 10.0, 1.0]),
        units.m / units.s**2
    )

    # vmap works, units preserved
    forces = jax.vmap(force)(masses, accels)
    print(forces)  # [9.8, 10.0, 7.5, 40.0, 5.0] N (Newtons!)
    ```

### Side-by-Side: Automatic Differentiation

=== "Raw JAX"

    ```python
    import jax

    def energy(velocity):
        """E = 0.5 * v² (assume mass=1)"""
        return 0.5 * velocity ** 2

    # Gradient function
    grad_energy = jax.grad(energy)

    v = jnp.array(10.0)
    dE_dv = grad_energy(v)
    print(f"dE/dv = {dE_dv}")  # 10.0 (what units?)
    ```

=== "dimtensor + JAX"

    ```python
    import jax
    from dimtensor.jax import DimArray
    from dimtensor import units

    def energy(velocity):
        """E = 0.5 * v² (assume mass=1 kg)"""
        mass = DimArray(1.0, units.kg)
        return 0.5 * mass * velocity ** 2

    # Gradient with unit awareness
    grad_energy = jax.grad(
        lambda v: energy(v).magnitude()
    )

    v = DimArray(jnp.array(10.0), units.m / units.s)
    dE_dv = grad_energy(v)

    # Gradient has units [Energy]/[Velocity] = kg⋅m/s
    print(f"dE/dv = {dE_dv}")  # 10.0 (kg⋅m/s, i.e., momentum!)
    ```

### Common Patterns

#### Pattern 1: Adding Units to JAX Arrays

```python
# Before: Raw JAX arrays
import jax.numpy as jnp

position = jnp.array([1.0, 2.0, 3.0])
velocity = jnp.array([10.0, 20.0, 30.0])

# After: Add units
from dimtensor.jax import DimArray
from dimtensor import units

position = DimArray(jnp.array([1.0, 2.0, 3.0]), units.m)
velocity = DimArray(jnp.array([10.0, 20.0, 30.0]), units.m / units.s)
```

#### Pattern 2: JIT + vmap Combined

```python
# Before: Manual vectorization
@jax.jit
def simulate_step(x, v, dt):
    return x + v * dt

# Vectorize manually
x_batch = jnp.array([[0], [1], [2]])
v_batch = jnp.array([[10], [20], [30]])
dt = 0.1

x_next = jax.vmap(lambda x, v: simulate_step(x, v, dt))(x_batch, v_batch)

# After: Units + JIT + vmap
from dimtensor.jax import DimArray
from dimtensor import units

@jax.jit
def simulate_step(x, v, dt):
    return x + v * dt

x_batch = DimArray(jnp.array([[0], [1], [2]]), units.m)
v_batch = DimArray(jnp.array([[10], [20], [30]]), units.m / units.s)
dt = DimArray(0.1, units.s)

# Automatic batching with unit checking
x_next = jax.vmap(simulate_step, in_axes=(0, 0, None))(x_batch, v_batch, dt)
```

#### Pattern 3: Gradient-Based Optimization

```python
# Before: Raw JAX optimization
import jax
import jax.numpy as jnp

def objective(params):
    """Minimize some physics quantity"""
    x, v = params
    return (x ** 2 + v ** 2).sum()

# Gradient descent
params = jnp.array([10.0, 5.0])
lr = 0.1

for step in range(100):
    grad = jax.grad(objective)(params)
    params = params - lr * grad

# After: Units ensure physical correctness
from dimtensor.jax import DimArray
from dimtensor import units

def objective(x, v):
    """Minimize energy"""
    mass = DimArray(1.0, units.kg)
    KE = 0.5 * mass * v ** 2
    PE = mass * DimArray(9.8, units.m / units.s**2) * x
    total = KE + PE
    return total.magnitude()

# Gradient descent with units
x = DimArray(jnp.array(10.0), units.m)
v = DimArray(jnp.array(5.0), units.m / units.s)
lr = 0.1

for step in range(100):
    grad_x = jax.grad(lambda x_val: objective(x_val, v))(x)
    grad_v = jax.grad(lambda v_val: objective(x, v_val))(v)

    # Gradients have correct dimensions for update
    x = x - lr * grad_x  # [m] - [m] = [m] ✓
    v = v - lr * grad_v  # [m/s] - [m/s] = [m/s] ✓
```

### Complete Example: Physics Simulation with JAX

Simulate gravitational motion with JIT + vmap:

```python
import jax
import jax.numpy as jnp
from dimtensor.jax import DimArray
from dimtensor import units

@jax.jit
def update_position(x, v, dt):
    """Euler integration: x(t+dt) = x(t) + v*dt"""
    return x + v * dt

@jax.jit
def update_velocity(v, a, dt):
    """v(t+dt) = v(t) + a*dt"""
    return v + a * dt

# Initial conditions (10 particles)
n_particles = 10

x = DimArray(jnp.zeros(n_particles), units.m)
v = DimArray(jnp.ones(n_particles) * 10, units.m / units.s)
a = DimArray(jnp.ones(n_particles) * (-9.8), units.m / units.s**2)

dt = DimArray(0.01, units.s)
n_steps = 1000

# Simulation loop (JIT-compiled)
@jax.jit
def simulate_step(state, _):
    """One simulation step."""
    x, v = state
    x_new = update_position(x, v, dt)
    v_new = update_velocity(v, a, dt)
    return (x_new, v_new), (x_new, v_new)

# Run simulation (vmap over time steps)
initial_state = (x, v)
final_state, trajectory = jax.lax.scan(
    simulate_step,
    initial_state,
    None,
    length=n_steps
)

# Extract results
x_final, v_final = final_state
x_traj, v_traj = trajectory

print(f"Final positions: {x_final.to(units.m)}")
print(f"Final velocities: {v_final.to(units.m / units.s)}")
```

### Key Differences from PyTorch

| Feature | PyTorch + dimtensor | JAX + dimtensor |
|---------|---------------------|-----------------|
| Compilation | No (eager execution) | Yes (JIT via XLA) |
| Vectorization | Manual batching | Automatic with vmap |
| Differentiation | Autograd (reverse-mode) | Forward & reverse-mode |
| Mutability | Mutable tensors | Immutable arrays |
| Hardware | CUDA (NVIDIA), MPS (Apple) | TPU, CUDA, CPU |

Both support unit-aware computation, but JAX offers functional programming paradigm with JIT compilation.

---

## Physics ML Features (dimtensor only)

These advanced features are unique to dimtensor and don't exist in other unit libraries.

### 1. Dimension-Aware Neural Network Layers

```python
from dimtensor.torch import DimLinear, DimConv1d, DimConv2d
from dimtensor import Dimension

# Linear layer with automatic dimension checking
layer = DimLinear(
    in_features=10,
    out_features=10,
    input_dim=Dimension(L=1),       # Input must be length
    output_dim=Dimension(L=1, T=-1)  # Output is velocity
)

# 1D convolution for time series
conv1d = DimConv1d(
    in_channels=1,
    out_channels=16,
    kernel_size=5,
    input_dim=Dimension(Theta=1),  # Temperature
    output_dim=Dimension(Theta=1)
)

# 2D convolution for spatial fields
conv2d = DimConv2d(
    in_channels=1,
    out_channels=32,
    kernel_size=3,
    input_dim=Dimension(M=1, L=-3),  # Density
    output_dim=Dimension(M=1, L=-3)
)
```

### 2. Physics-Informed Loss Functions

```python
from dimtensor.torch import (
    DimMSELoss,       # MSE with dimension checking
    PhysicsLoss,      # Conservation law enforcement
    CompositeLoss     # Combine data + physics losses
)

# Enforce energy conservation during training
energy_loss = PhysicsLoss(rtol=1e-4)

# Combine losses
composite = CompositeLoss(
    data_loss=DimMSELoss(),
    physics_losses={
        'energy': (energy_loss, 0.1),
        'momentum': (PhysicsLoss(), 0.05)
    }
)
```

### 3. Automatic Non-dimensionalization

```python
from dimtensor.torch import DimScaler, MultiScaler

# Single scaler
scaler = DimScaler(method='characteristic')
scaler.fit(training_data)

# Transform to [-1, 1] for neural network
scaled = scaler.transform(data)

# Inverse transform predictions
predictions = scaler.inverse_transform(output, units.m / units.s)

# Multi-quantity scaler
multi_scaler = MultiScaler()
multi_scaler.add('position', position_train)
multi_scaler.add('velocity', velocity_train)

x_scaled = multi_scaler.transform('position', x)
v_scaled = multi_scaler.transform('velocity', v)
```

### 4. Equation Database Integration

```python
from dimtensor.equations import search_equations, get_equation

# Search for relevant equations
heat_eqs = search_equations("heat")
for eq in heat_eqs:
    print(f"{eq.name}: {eq.formula}")

# Get specific equation for validation
newton = get_equation("Newton's Second Law")

# Validate model outputs match expected dimensions
assert model_output.dimension == newton.variables['F']
```

---

## Common Patterns

### Pattern 1: Wrapping External Data

```python
import numpy as np
from dimtensor import DimArray, units

# Data from file, API, database, etc.
raw_data = np.loadtxt("measurements.csv")

# Add units based on metadata/documentation
temperatures = DimArray(raw_data[:, 0], units.K)
pressures = DimArray(raw_data[:, 1], units.Pa)
velocities = DimArray(raw_data[:, 2:5], units.m / units.s)
```

### Pattern 2: Interfacing with Unit-Unaware Libraries

```python
from dimtensor import DimArray, units
import scipy.optimize

def objective(x_raw):
    """Wrapper for scipy that handles units internally."""
    # Convert raw values to DimArray for physics
    x = DimArray(x_raw, units.m)

    # Do unit-aware calculations
    energy = compute_energy(x)

    # Return raw value for scipy
    return energy.data

# scipy needs raw arrays
x0 = initial_guess.data
result = scipy.optimize.minimize(objective, x0)

# Convert result back to units
x_opt = DimArray(result.x, units.m)
```

### Pattern 3: Gradual Migration

You don't have to convert everything at once:

```python
from dimtensor import DimArray, units
import numpy as np

# Phase 1: Mix old and new code
old_array = np.array([1, 2, 3])  # Legacy code
new_array = DimArray([4, 5, 6], units.m)  # New code

# Convert when needed
old_with_units = DimArray(old_array, units.m)
combined = old_with_units + new_array

# Phase 2: Update functions one at a time
def legacy_function(x):
    """Still uses raw numpy."""
    return x ** 2

def new_function(x: DimArray) -> DimArray:
    """Uses dimtensor."""
    return x ** 2

# Phase 3: Eventually replace all legacy code
```

### Pattern 4: Validation and Testing

```python
from dimtensor import DimArray, units, Dimension

def validate_physics_model(model, test_data):
    """Ensure model maintains dimensional correctness."""

    x_test = DimArray(test_data['position'], units.m)
    v_test = DimArray(test_data['velocity'], units.m / units.s)

    # Predict
    predictions = model(x_test, v_test)

    # Check dimensions
    assert predictions.dimension == Dimension(M=1, L=1, T=-2)  # Force

    # Check magnitudes are reasonable
    assert (predictions.data > 0).all()
    assert (predictions.data < 1e6).all()

    print("Model validation passed!")
```

---

## Summary: Why Migrate?

### From Pint → dimtensor
**Gain:** PyTorch/JAX integration, GPU acceleration, built-in uncertainty
**Keep:** Use both together if you need Pint's flexible unit systems

### From Astropy → dimtensor
**Gain:** ML workflows, GPU support, physics-informed neural networks
**Keep:** Use both together for FITS I/O and coordinate transformations

### From unyt → dimtensor
**Gain:** Neural networks on simulation data, autograd, GPU acceleration
**Keep:** Use both together for yt integration and cosmological units

### From Raw NumPy/PyTorch/JAX → dimtensor
**Gain:** Dimensional safety catches bugs early, physical constants, uncertainty propagation
**Cost:** Minimal overhead, more verbose array creation

---

## Quick Start Checklist

Migrating to dimtensor in 5 steps:

1. **Install**: `pip install dimtensor[torch]` or `dimtensor[all]`

2. **Import**: Replace unit library imports
   ```python
   # Before
   import pint
   ureg = pint.UnitRegistry()

   # After
   from dimtensor import DimArray, units
   ```

3. **Wrap Arrays**: Add units to your data
   ```python
   # Before
   velocity = np.array([10, 20, 30])

   # After
   velocity = DimArray([10, 20, 30], units.m / units.s)
   ```

4. **Update Operations**: Most operations work unchanged!
   ```python
   distance = velocity * time  # Just works
   ```

5. **Add ML Features** (optional): Use DimTensor for PyTorch
   ```python
   from dimtensor.torch import DimTensor

   v_torch = DimTensor(torch.randn(100, 3, requires_grad=True), units.m / units.s)
   ```

---

## Getting Help

If you run into issues during migration:

1. **Documentation**: [Full API reference](../api/dimarray.md)
2. **Examples**: [Example gallery](../guide/examples.md)
3. **Common Errors**: [Error guide](errors.md)
4. **FAQ**: [Frequently asked questions](faq.md)
5. **GitHub Issues**: [Report problems](https://github.com/marcoloco23/dimtensor/issues)

### Useful Resources

- [PyTorch Integration Guide](../guide/pytorch.md)
- [JAX Integration Guide](../guide/jax.md)
- [Physics-Informed ML Guide](../guide/physics-ml.md)
- [Comparison Matrix](../comparisons/feature-matrix.md)

**Questions?** Open a discussion or issue on GitHub!

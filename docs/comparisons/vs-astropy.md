---
title: dimtensor vs Astropy Units - Python Unit Libraries Compared
description: Compare dimtensor and Astropy units for scientific computing. Learn which library is best for astronomy and physics applications.
---

# dimtensor vs Astropy Units

[Astropy](https://docs.astropy.org/en/stable/units/) provides a comprehensive units package designed for astronomy. This page compares it with dimtensor.

## TL;DR

| Feature | dimtensor | Astropy |
|---------|-----------|---------|
| Focus | General physics + ML | Astronomy |
| PyTorch support | Native | None |
| JAX support | Native | None |
| Astronomy units | Domain module | Comprehensive |
| Equivalencies | Basic | Advanced (spectral, etc.) |
| FITS I/O | No | Native |
| Uncertainty | Built-in | Separate NDData |

## When to Choose dimtensor

**Choose dimtensor if you need:**

- Machine learning with units (PyTorch, JAX)
- GPU acceleration
- Built-in uncertainty propagation
- Multi-framework support (NumPy + PyTorch + JAX)
- General physics beyond astronomy

```python
# dimtensor: ML with astronomy units
from dimtensor.torch import DimTensor
from dimtensor.domains.astronomy import solar_mass, parsec
import torch

mass = DimTensor(torch.tensor([1.0], requires_grad=True), solar_mass)
# Use in neural network training...
```

## When to Choose Astropy

**Choose Astropy if you need:**

- Comprehensive astronomy unit system
- Spectral equivalencies (wavelength/frequency/energy)
- FITS file I/O with units
- Coordinate transformations
- Integration with other Astropy tools

```python
# Astropy: Spectral equivalencies
from astropy import units as u

wavelength = 500 * u.nm
frequency = wavelength.to(u.Hz, equivalencies=u.spectral())
energy = wavelength.to(u.eV, equivalencies=u.spectral())
```

## Feature Comparison

### Unit Systems

=== "dimtensor"

    ```python
    from dimtensor import DimArray, units
    from dimtensor.domains.astronomy import parsec, AU, solar_mass, light_year

    # Core astronomy units available
    distance = DimArray([4.24], light_year)
    distance_pc = distance.to(parsec)
    ```

=== "Astropy"

    ```python
    from astropy import units as u

    # Extensive astronomy unit system
    distance = 4.24 * u.lyr
    distance_pc = distance.to(u.pc)

    # Specialized astronomy units
    flux = 1e-23 * u.erg / u.s / u.cm**2 / u.Hz  # Jansky-like
    magnitude = 5 * u.mag
    ```

### Equivalencies

Astropy excels at equivalencies (conversions between physically related but dimensionally different quantities):

=== "dimtensor"

    ```python
    # dimtensor focuses on dimensional correctness
    # Explicit conversions required
    from dimtensor import DimArray, units
    from dimtensor import constants

    wavelength = DimArray([500e-9], units.m)  # 500 nm
    frequency = constants.c / wavelength  # Manual conversion
    ```

=== "Astropy"

    ```python
    from astropy import units as u

    # Automatic equivalencies
    wavelength = 500 * u.nm

    # Convert between wavelength, frequency, and energy
    freq = wavelength.to(u.Hz, equivalencies=u.spectral())
    energy = wavelength.to(u.eV, equivalencies=u.spectral())

    # Temperature equivalencies
    temp = 5000 * u.K
    wavelength_peak = temp.to(u.um, equivalencies=u.temperature_energy())
    ```

### Machine Learning

=== "dimtensor"

    ```python
    import jax
    from dimtensor.jax import DimArray
    from dimtensor.domains.astronomy import solar_mass

    @jax.jit
    def gravitational_energy(m1, m2, r):
        from dimtensor import constants
        return -constants.G * m1 * m2 / r

    # JIT-compiled, units preserved throughout
    ```

=== "Astropy"

    ```python
    # Astropy units don't work with JAX or PyTorch
    # Must strip units before ML operations
    from astropy import units as u

    mass = 1.0 * u.solMass
    mass_value = mass.to(u.kg).value  # Strip units for ML
    ```

### I/O Support

| Format | dimtensor | Astropy |
|--------|-----------|---------|
| JSON | Yes | No |
| HDF5 | Yes | Via Astropy Table |
| Parquet | Yes | No |
| NetCDF | Yes | No |
| FITS | No | Native |
| VOTable | No | Native |

### Uncertainty Handling

=== "dimtensor"

    ```python
    from dimtensor import DimArray
    from dimtensor.domains.astronomy import parsec

    # Built-in uncertainty
    distance = DimArray([10.0], parsec, uncertainty=[0.5])
    luminosity = distance ** 2  # Uncertainty propagates
    ```

=== "Astropy"

    ```python
    from astropy import units as u
    from astropy.nddata import NDDataArray, StdDevUncertainty

    # Separate uncertainty container
    data = NDDataArray([10.0] * u.pc,
                       uncertainty=StdDevUncertainty([0.5] * u.pc))
    ```

## Migration from Astropy

If you're adding ML to astronomy code:

```python
# Astropy code
from astropy import units as u
from astropy import constants as const

mass = 1.0 * u.solMass
velocity = 200 * u.km / u.s
kinetic_energy = 0.5 * mass * velocity**2

# dimtensor equivalent (for ML compatibility)
from dimtensor import DimArray
from dimtensor.domains.astronomy import solar_mass
from dimtensor import units

mass = DimArray([1.0], solar_mass)
velocity = DimArray([200], units.km / units.s)
kinetic_energy = 0.5 * mass * velocity**2
# Now can use with PyTorch/JAX
```

## Using Both Together

For astronomy projects needing ML, consider using both:

```python
from astropy import units as u
from dimtensor import DimArray, units as dt_units
from dimtensor.torch import DimTensor
import torch

# Load data with Astropy (FITS, coordinates, etc.)
from astropy.io import fits
data = fits.getdata('observations.fits')
astropy_flux = data * u.Jy

# Convert to dimtensor for ML
flux_values = astropy_flux.to(u.W / u.m**2 / u.Hz).value
flux = DimTensor(torch.tensor(flux_values), dt_units.W / dt_units.m**2 / dt_units.Hz)

# Train your model with units preserved...
```

## Conclusion

- **Use dimtensor** for ML workflows, GPU computing, or general physics
- **Use Astropy** for astronomy-specific workflows, FITS files, or coordinate transformations
- **Use both** when you need Astropy's astronomy features plus ML capabilities

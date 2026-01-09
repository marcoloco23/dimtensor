---
title: Python Units Library Feature Comparison
description: Comprehensive feature matrix comparing dimtensor, Pint, Astropy units, and unyt for scientific Python computing.
---

# Python Units Library Feature Matrix

A comprehensive comparison of Python units libraries to help you choose the right tool for your project.

## Overview

| Library | Primary Focus | Best For |
|---------|---------------|----------|
| **dimtensor** | Physics + ML | PyTorch/JAX workflows, GPU computing |
| **Pint** | General purpose | Flexible unit systems, data analysis |
| **Astropy** | Astronomy | Astronomical calculations, FITS files |
| **unyt** | Astrophysics | yt project, large simulations |

## Core Features

| Feature | dimtensor | Pint | Astropy | unyt |
|---------|:---------:|:----:|:-------:|:----:|
| NumPy arrays | Yes | Yes | Yes | Yes |
| SI units | Yes | Yes | Yes | Yes |
| CGS units | Yes | Yes | Yes | Yes |
| Imperial units | Yes | Yes | Yes | Yes |
| Custom units | Limited | Extensive | Limited | Limited |
| Unit simplification | Auto | Auto | Auto | Auto |

## Framework Integration

| Feature | dimtensor | Pint | Astropy | unyt |
|---------|:---------:|:----:|:-------:|:----:|
| **PyTorch native** | Yes | No | No | No |
| **JAX native** | Yes | No | No | No |
| **Autograd support** | Yes | No | No | No |
| **GPU (CUDA)** | Yes | No | No | No |
| **GPU (MPS/Apple)** | Yes | No | No | No |
| Dask arrays | No | Yes | No | Yes |
| Sparse arrays | No | Yes | No | No |

## Scientific Computing

| Feature | dimtensor | Pint | Astropy | unyt |
|---------|:---------:|:----:|:-------:|:----:|
| Physical constants | CODATA 2022 | scipy.constants | Built-in | yt constants |
| **Uncertainty propagation** | Built-in | Extension | Separate | No |
| Temperature handling | Yes | Yes | Yes | Yes |
| Logarithmic units (dB) | No | Yes | Yes | No |
| Equivalencies | Basic | Basic | Advanced | Basic |

## Domain-Specific Units

| Domain | dimtensor | Pint | Astropy | unyt |
|--------|:---------:|:----:|:-------:|:----:|
| Astronomy | Module | Basic | Comprehensive | Comprehensive |
| Chemistry | Module | Basic | No | No |
| Engineering | Module | Basic | No | No |
| Cosmology | No | No | Yes | Yes |

## I/O Support

| Format | dimtensor | Pint | Astropy | unyt |
|--------|:---------:|:----:|:-------:|:----:|
| JSON | Yes | Manual | No | Yes |
| **HDF5** | Yes | Manual | Table | Yes |
| **Parquet** | Yes | No | No | No |
| **NetCDF** | Yes | No | No | No |
| pandas | Yes | pint-pandas | No | Limited |
| xarray | Yes | pint-xarray | No | No |
| FITS | No | No | Native | No |
| VOTable | No | No | Native | No |

## Performance

| Metric | dimtensor | Pint | Astropy | unyt |
|--------|:---------:|:----:|:-------:|:----:|
| Overhead vs NumPy | 2-5x | 2-5x | 2-3x | 1.5-3x |
| Memory efficiency | Good | Good | Good | Good |
| GPU acceleration | Yes | No | No | No |
| JIT compilation | JAX | No | No | No |

## Documentation & Community

| Metric | dimtensor | Pint | Astropy | unyt |
|--------|:---------:|:----:|:-------:|:----:|
| Documentation | Good | Excellent | Excellent | Good |
| Tutorials | Growing | Many | Many | Some |
| Stack Overflow | New | Active | Active | Moderate |
| GitHub stars | Growing | ~2k | ~4k (main) | ~400 |

## Installation Size

| Library | Install Size | Dependencies |
|---------|--------------|--------------|
| dimtensor | ~500 KB | numpy |
| Pint | ~1 MB | numpy |
| Astropy | ~50 MB | numpy, scipy, etc. |
| unyt | ~2 MB | numpy, sympy |

## Code Examples

### Creating Arrays

=== "dimtensor"

    ```python
    from dimtensor import DimArray, units
    arr = DimArray([1, 2, 3], units.m / units.s)
    ```

=== "Pint"

    ```python
    import pint
    ureg = pint.UnitRegistry()
    arr = [1, 2, 3] * ureg.meter / ureg.second
    ```

=== "Astropy"

    ```python
    from astropy import units as u
    arr = [1, 2, 3] * u.m / u.s
    ```

=== "unyt"

    ```python
    from unyt import unyt_array
    arr = unyt_array([1, 2, 3], 'm/s')
    ```

### Unit Conversion

=== "dimtensor"

    ```python
    distance = DimArray([1000], units.m)
    in_km = distance.to(units.km)  # 1 km
    ```

=== "Pint"

    ```python
    distance = 1000 * ureg.meter
    in_km = distance.to('km')  # 1 km
    ```

=== "Astropy"

    ```python
    distance = 1000 * u.m
    in_km = distance.to(u.km)  # 1 km
    ```

=== "unyt"

    ```python
    distance = unyt_array([1000], 'm')
    in_km = distance.to('km')  # 1 km
    ```

## Decision Guide

### Choose dimtensor if:

- You need PyTorch or JAX integration
- You're doing machine learning with physical units
- You need GPU acceleration
- You want built-in uncertainty propagation
- You need multiple I/O formats

### Choose Pint if:

- You need maximum flexibility in unit definitions
- You're doing pure data analysis (no ML)
- You have complex custom unit systems
- You need pandas integration

### Choose Astropy if:

- You're doing astronomy/astrophysics
- You need spectral equivalencies
- You work with FITS files
- You use other Astropy tools

### Choose unyt if:

- You use the yt project
- You're doing astrophysical simulations
- You need cosmological units
- You work with very large datasets (Dask)

## Summary

For **machine learning** and **GPU computing**: **dimtensor**

For **general purpose** flexibility: **Pint**

For **astronomy**: **Astropy**

For **yt/simulations**: **unyt**

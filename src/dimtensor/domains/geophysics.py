"""Geophysics units for gravimetry, seismology, and petroleum engineering.

This module provides units commonly used in geophysics, including acceleration
units (gal), gravity gradient units (eotvos), permeability units (darcy), and
magnetic field units (gamma, oersted).

Example:
    >>> from dimtensor import DimArray
    >>> from dimtensor.domains.geophysics import gal, darcy, gamma
    >>> gravity_anomaly = DimArray([50.0], milligal)  # Typical gravity survey
    >>> permeability = DimArray([100.0], millidarcy)  # Typical reservoir rock
    >>> magnetic_field = DimArray([50000.0], gamma)  # Earth's magnetic field

Note on Richter Magnitude:
    Seismic magnitude scales (Richter, moment magnitude) are logarithmic and
    cannot be directly represented as linear unit conversions. Future versions
    may include a dimtensor.geophysics.seismic submodule for magnitude-energy
    conversions.

Reference values from NIST, Society of Petroleum Engineers (SPE), and standard
geophysics literature.
"""

from __future__ import annotations

from ..core.dimensions import Dimension, DIMENSIONLESS
from ..core.units import Unit


# =============================================================================
# Acceleration Units (Gravimetry)
# =============================================================================

# Gal (named after Galileo Galilei)
# 1 Gal = 1 cm/s² = 0.01 m/s²
# Standard unit in gravimetry and gravity surveys
gal = Unit("Gal", Dimension(length=1, time=-2), 0.01)
Gal = gal

# Milligal
# 1 mGal = 0.001 Gal = 1e-5 m/s²
# Common precision in gravity surveys
milligal = Unit("mGal", Dimension(length=1, time=-2), 1e-5)
mGal = milligal


# =============================================================================
# Gravity Gradient Units
# =============================================================================

# Eötvös (named after Loránd Eötvös)
# 1 E = 1e-9 s^-2
# Unit of gravity gradient in geophysical prospecting
# Dimension: T^-2 (inverse time squared)
eotvos = Unit("E", Dimension(time=-2), 1e-9)
E = eotvos


# =============================================================================
# Permeability Units (Petroleum Engineering)
# =============================================================================

# Darcy (named after Henry Darcy)
# 1 darcy = 9.869233e-13 m²
# Standard unit of permeability in petroleum reservoir engineering
# Note: Despite the name, darcy measures area (related to fluid flow through
# porous media), not pressure. It represents the permeability that allows
# 1 cm³/s flow of 1 cP fluid through 1 cm² area under 1 atm/cm pressure gradient.
darcy = Unit("darcy", Dimension(length=2), 9.869233e-13)

# Millidarcy
# 1 mD = 0.001 darcy = 9.869233e-16 m²
# Common unit for typical reservoir rock permeability
millidarcy = Unit("mD", Dimension(length=2), 9.869233e-16)
mD = millidarcy


# =============================================================================
# Magnetic Field Units (Geomagnetism)
# =============================================================================

# Gamma (also called nanotesla)
# 1 γ = 1 nT = 1e-9 T
# Common unit in geomagnetic surveys and exploration
# Dimension: magnetic field = M * T^-2 * I^-1
gamma = Unit("gamma", Dimension(mass=1, time=-2, current=-1), 1e-9)

# Oersted (CGS electromagnetic unit, named after Hans Christian Ørsted)
# 1 Oe = (1000/4π) A/m ≈ 79.5774715 A/m
# Still used in paleomagnetism and rock magnetism
# Dimension: magnetic field intensity = I * L^-1
oersted = Unit("Oe", Dimension(current=1, length=-1), 79.5774715)
Oe = oersted


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Acceleration
    "gal", "Gal",
    "milligal", "mGal",
    # Gravity gradient
    "eotvos", "E",
    # Permeability
    "darcy",
    "millidarcy", "mD",
    # Magnetic field
    "gamma",
    "oersted", "Oe",
]

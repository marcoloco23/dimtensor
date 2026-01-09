"""CGS (Centimeter-Gram-Second) unit system.

This module provides units from the CGS system, including mechanical units
and CGS-Gaussian electromagnetic units. The CGS system was widely used in
physics before SI became the standard, and is still common in some fields
like astrophysics and electromagnetism.

Example:
    >>> from dimtensor import DimArray
    >>> from dimtensor.domains.cgs import dyne, erg, gauss
    >>> force = DimArray([1000], dyne)  # 1000 dyne force
    >>> energy = DimArray([1e7], erg)   # 1e7 erg energy
    >>> field = DimArray([100], gauss)  # 100 gauss magnetic field

Reference values from NIST and physics handbooks.

Note:
    CGS-Gaussian electromagnetic units are defined using their SI equivalents
    with appropriate conversion factors. The CGS-Gaussian system uses different
    dimensional formulas than SI for electromagnetic quantities.
"""

from __future__ import annotations

from ..core.dimensions import Dimension, DIMENSIONLESS
from ..core.units import Unit


# =============================================================================
# CGS Base Units
# =============================================================================
# Note: These are already defined in core.units, but we re-export them here
# for completeness of the CGS system.

# centimeter: Unit("cm", Dimension(length=1), 0.01)
# gram: Unit("g", Dimension(mass=1), 0.001)
# second: Unit("s", Dimension(time=1), 1.0)


# =============================================================================
# CGS Mechanical Derived Units
# =============================================================================

# Dyne - unit of force
# 1 dyne = 1 g·cm/s² = 10^(-5) N
dyne = Unit("dyn", Dimension(mass=1, length=1, time=-2), 1e-5)
dyn = dyne

# Erg - unit of energy
# 1 erg = 1 g·cm²/s² = 10^(-7) J
erg = Unit("erg", Dimension(mass=1, length=2, time=-2), 1e-7)

# Barye (or Barie) - unit of pressure
# 1 Ba = 1 dyn/cm² = 1 g/(cm·s²) = 0.1 Pa
barye = Unit("Ba", Dimension(mass=1, length=-1, time=-2), 0.1)
Ba = barye
barie = barye  # Alternative spelling


# =============================================================================
# CGS Viscosity Units
# =============================================================================

# Poise - unit of dynamic viscosity
# 1 P = 1 g/(cm·s) = 0.1 Pa·s
poise = Unit("P", Dimension(mass=1, length=-1, time=-1), 0.1)
P = poise

# Centipoise - commonly used for liquids (water ≈ 1 cP)
# 1 cP = 0.01 P = 0.001 Pa·s = 1 mPa·s
centipoise = Unit("cP", Dimension(mass=1, length=-1, time=-1), 0.001)
cP = centipoise

# Stokes - unit of kinematic viscosity
# 1 St = 1 cm²/s = 10^(-4) m²/s
stokes = Unit("St", Dimension(length=2, time=-1), 1e-4)
St = stokes

# Centistokes - commonly used (water ≈ 1 cSt)
# 1 cSt = 0.01 St = 10^(-6) m²/s = 1 mm²/s
centistokes = Unit("cSt", Dimension(length=2, time=-1), 1e-6)
cSt = centistokes


# =============================================================================
# CGS-Gaussian Electromagnetic Units
# =============================================================================

# Gauss - unit of magnetic flux density (magnetic induction B)
# 1 G = 10^(-4) T
gauss = Unit("G", Dimension(mass=1, time=-2, current=-1), 1e-4)
G = gauss

# Kilogauss
kilogauss = Unit("kG", Dimension(mass=1, time=-2, current=-1), 0.1)
kG = kilogauss

# Maxwell - unit of magnetic flux
# 1 Mx = 10^(-8) Wb
maxwell = Unit("Mx", Dimension(mass=1, length=2, time=-2, current=-1), 1e-8)
Mx = maxwell

# Oersted - unit of magnetic field strength (H field)
# 1 Oe = 1000/(4π) A/m ≈ 79.5774715459 A/m
oersted = Unit("Oe", Dimension(current=1, length=-1), 79.5774715459)
Oe = oersted

# Statcoulomb (Franklin, ESU charge) - unit of electric charge
# 1 statC = (10/c) C where c is speed of light in m/s
# 1 statC ≈ 3.33564095198152e-10 C
statcoulomb = Unit("statC", Dimension(current=1, time=1), 3.33564095198152e-10)
statC = statcoulomb
franklin = statcoulomb  # Alternative name
esu = statcoulomb  # Electrostatic unit
Fr = franklin

# Statampere (ESU current) - unit of electric current
# 1 statA = 1 statC/s ≈ 3.33564095198152e-10 A
statampere = Unit("statA", Dimension(current=1), 3.33564095198152e-10)
statA = statampere

# Statvolt (ESU potential) - unit of electric potential
# 1 statV = (c/10^6) V ≈ 299.792458 V
statvolt = Unit("statV", Dimension(mass=1, length=2, time=-3, current=-1), 299.792458)
statV = statvolt

# Statohm - unit of electrical resistance
# 1 statΩ = 1 statV/statA ≈ 8.98755178736818e11 Ω
statohm = Unit("statΩ", Dimension(mass=1, length=2, time=-3, current=-2), 8.98755178736818e11)

# Statfarad - unit of capacitance
# 1 statF = 1 statC/statV ≈ 1.11265005605362e-12 F
statfarad = Unit("statF", Dimension(mass=-1, length=-2, time=4, current=2), 1.11265005605362e-12)
statF = statfarad

# Stathenry - unit of inductance
# 1 statH = 1 statV·s/statA ≈ 8.98755178736818e11 H
stathenry = Unit("statH", Dimension(mass=1, length=2, time=-2, current=-2), 8.98755178736818e11)
statH = stathenry


# =============================================================================
# Additional CGS Units
# =============================================================================

# Gal - unit of acceleration (named after Galileo)
# 1 Gal = 1 cm/s² = 0.01 m/s²
# Commonly used in gravimetry and geophysics
gal = Unit("Gal", Dimension(length=1, time=-2), 0.01)
Gal = gal

# Milligal - commonly used in geophysics
milligal = Unit("mGal", Dimension(length=1, time=-2), 1e-5)
mGal = milligal


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Mechanical units
    "dyne", "dyn",
    "erg",
    "barye", "Ba", "barie",
    # Viscosity
    "poise", "P",
    "centipoise", "cP",
    "stokes", "St",
    "centistokes", "cSt",
    # Electromagnetic (CGS-Gaussian)
    "gauss", "G",
    "kilogauss", "kG",
    "maxwell", "Mx",
    "oersted", "Oe",
    "statcoulomb", "statC", "franklin", "Fr", "esu",
    "statampere", "statA",
    "statvolt", "statV",
    "statohm",
    "statfarad", "statF",
    "stathenry", "statH",
    # Additional units
    "gal", "Gal",
    "milligal", "mGal",
]

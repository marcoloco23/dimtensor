"""Nuclear physics units for particle physics, nuclear reactions, and radiation safety.

This module provides units commonly used in nuclear and particle physics,
including energy units (eV, MeV, GeV), cross-sections (barn), radioactivity
(becquerel, curie), and radiation dose units (gray, sievert, rad, rem).

Example:
    >>> from dimtensor import DimArray
    >>> from dimtensor.domains.nuclear import MeV, barn, becquerel
    >>> energy = DimArray([938.3], MeV)  # Proton rest mass energy
    >>> cross_section = DimArray([1.5], barn)  # Neutron capture cross-section
    >>> activity = DimArray([1e6], becquerel)  # 1 MBq source

Reference values from CODATA 2022 and SI definitions.
"""

from __future__ import annotations

from ..core.dimensions import Dimension
from ..core.units import Unit


# =============================================================================
# Energy Units (Electron Volt Family)
# =============================================================================

# Electron volt (CODATA 2022, exact since 2019 SI redefinition)
# 1 eV = 1.602176634e-19 J (exactly)
electronvolt = Unit("eV", Dimension(mass=1, length=2, time=-2), 1.602176634e-19)
eV = electronvolt

# Kiloelectronvolt
kiloelectronvolt = Unit("keV", Dimension(mass=1, length=2, time=-2), 1.602176634e-16)
keV = kiloelectronvolt

# Megaelectronvolt (common in nuclear physics)
megaelectronvolt = Unit("MeV", Dimension(mass=1, length=2, time=-2), 1.602176634e-13)
MeV = megaelectronvolt

# Gigaelectronvolt (common in particle physics)
gigaelectronvolt = Unit("GeV", Dimension(mass=1, length=2, time=-2), 1.602176634e-10)
GeV = gigaelectronvolt


# =============================================================================
# Cross-Section Units
# =============================================================================

# Barn (exactly 1e-28 m² by definition)
# Standard unit for nuclear cross-sections
barn = Unit("b", Dimension(length=2), 1e-28)
b = barn

# Millibarn
millibarn = Unit("mb", Dimension(length=2), 1e-31)
mb = millibarn

# Microbarn (using 'ub' for compatibility, represents µb)
microbarn = Unit("ub", Dimension(length=2), 1e-34)
ub = microbarn


# =============================================================================
# Radioactivity Units
# =============================================================================

# Becquerel (SI unit of radioactivity, 1/s)
# Dimension: T^-1 (frequency/activity)
becquerel = Unit("Bq", Dimension(time=-1), 1.0)
Bq = becquerel

# Curie (exactly 3.7e10 Bq, historical definition based on radium-226)
curie = Unit("Ci", Dimension(time=-1), 3.7e10)
Ci = curie


# =============================================================================
# Absorbed Dose Units
# =============================================================================

# Gray (SI unit of absorbed dose, J/kg = m²/s²)
# Dimension: L²·T⁻² (specific energy)
gray = Unit("Gy", Dimension(length=2, time=-2), 1.0)
Gy = gray

# Rad (exactly 0.01 Gy, from CGS system)
rad = Unit("rad", Dimension(length=2, time=-2), 0.01)


# =============================================================================
# Dose Equivalent Units
# =============================================================================

# Sievert (SI unit of dose equivalent, J/kg = m²/s²)
# Same dimension as gray, but represents dose equivalent (includes quality factor)
# Dimension: L²·T⁻² (specific energy)
sievert = Unit("Sv", Dimension(length=2, time=-2), 1.0)
Sv = sievert

# Rem (exactly 0.01 Sv, from CGS system)
# Lowercase to distinguish from REM (roentgen equivalent man)
rem = Unit("rem", Dimension(length=2, time=-2), 0.01)


# =============================================================================
# Decay Constant (Convenience)
# =============================================================================

# Per second (same dimension as becquerel, but used for decay constants)
# Dimension: T^-1 (frequency)
per_second = Unit("s^-1", Dimension(time=-1), 1.0)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Energy
    "electronvolt", "eV",
    "kiloelectronvolt", "keV",
    "megaelectronvolt", "MeV",
    "gigaelectronvolt", "GeV",
    # Cross-section
    "barn", "b",
    "millibarn", "mb",
    "microbarn", "ub",
    # Radioactivity
    "becquerel", "Bq",
    "curie", "Ci",
    # Absorbed dose
    "gray", "Gy",
    "rad",
    # Dose equivalent
    "sievert", "Sv",
    "rem",
    # Decay constant
    "per_second",
]

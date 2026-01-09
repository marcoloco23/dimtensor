"""Materials science units for stress/strain, hardness, and material properties.

This module provides units commonly used in materials science and engineering,
including strain measurements, hardness scales, fracture toughness, thermal
conductivity, and electrical conductivity/resistivity.

Example:
    >>> from dimtensor import DimArray
    >>> from dimtensor.domains.materials import MPa, strain, microstrain, vickers
    >>> stress = DimArray([500], MPa)  # 500 MPa tensile stress
    >>> elongation = DimArray([1500], microstrain)  # 1500 microstrain
    >>> hardness = DimArray([350], vickers)  # 350 HV hardness
    >>>
    >>> # Fracture toughness calculation
    >>> from dimtensor.domains.materials import MPa_sqrt_m
    >>> K_IC = DimArray([45], MPa_sqrt_m)  # Fracture toughness

Reference values from materials testing standards (ASTM, ISO) and handbooks.
"""

from __future__ import annotations

from fractions import Fraction

from ..core.dimensions import Dimension, DIMENSIONLESS
from ..core.units import Unit

# Import stress/pressure units from engineering module
from .engineering import gigapascal, GPa, megapascal, MPa, kilopascal, kPa


# =============================================================================
# Strain Units (dimensionless)
# =============================================================================

# Base strain unit (dimensionless, ΔL/L)
strain = Unit("strain", DIMENSIONLESS, 1.0)

# Microstrain (με, common in mechanical testing)
# 1 με = 1e-6 strain
microstrain = Unit("με", DIMENSIONLESS, 1e-6)

# Percent strain (engineering strain as percentage)
# 1% = 0.01 strain
percent_strain = Unit("%strain", DIMENSIONLESS, 0.01)


# =============================================================================
# Hardness Units (dimensionless scales)
# =============================================================================

# Vickers hardness (HV) - diamond pyramid indenter
# Most universal hardness scale, can test from soft to very hard materials
vickers = Unit("HV", DIMENSIONLESS, 1.0)
HV = vickers

# Brinell hardness (HB) - ball indenter
# Common for bulk hardness of metals and alloys
brinell = Unit("HB", DIMENSIONLESS, 1.0)
HB = brinell

# Rockwell C hardness (HRC) - conical diamond indenter
# Common for hardened steels, hard alloys, case hardened parts
rockwell_C = Unit("HRC", DIMENSIONLESS, 1.0)
HRC = rockwell_C

# Rockwell B hardness (HRB) - ball indenter
# Common for softer materials (annealed steel, brass, aluminum alloys)
rockwell_B = Unit("HRB", DIMENSIONLESS, 1.0)
HRB = rockwell_B


# =============================================================================
# Fracture Toughness Units
# =============================================================================

# Stress intensity factor in MPa·√m
# K_IC dimension: stress × length^(1/2) = Pa·m^(1/2)
# Mass: 1, Length: -1/2, Time: -2
# Scale: 1e6 (from MPa base)
MPa_sqrt_m = Unit(
    "MPa·√m",
    Dimension(mass=1, length=Fraction(-1, 2), time=-2),
    1e6
)

# Imperial fracture toughness: ksi·√in
# 1 ksi·√in = 1.0988422 MPa·√m
# Scale calculation:
#   1 ksi = 6.894757 MPa
#   1 √in = 0.0254^0.5 √m = 0.1593882 √m
#   1 ksi·√in = 6.894757 × 0.1593882 = 1.0988422 MPa·√m
ksi_sqrt_in = Unit(
    "ksi·√in",
    Dimension(mass=1, length=Fraction(-1, 2), time=-2),
    6.894757e6 * (0.0254 ** 0.5)
)


# =============================================================================
# Thermal Conductivity Units
# =============================================================================

# SI unit: W/(m·K)
# Dimension: power / (length × temperature) = W/(m·K)
# Mass: 1, Length: 1, Time: -3, Temperature: -1
W_per_m_K = Unit(
    "W/(m·K)",
    Dimension(mass=1, length=1, time=-3, temperature=-1),
    1.0
)

# Common in materials handbooks: W/(cm·K)
# 1 W/(cm·K) = 100 W/(m·K)
W_per_cm_K = Unit(
    "W/(cm·K)",
    Dimension(mass=1, length=1, time=-3, temperature=-1),
    100.0
)


# =============================================================================
# Electrical Conductivity Units
# =============================================================================

# SI unit: Siemens per meter (S/m)
# Dimension: conductance / length = (A²·s³)/(kg·m³)
# Mass: -1, Length: -3, Time: 3, Current: 2
S_per_m = Unit(
    "S/m",
    Dimension(mass=-1, length=-3, time=3, current=2),
    1.0
)

# Megasiemens per meter (MS/m) - for highly conductive materials
# 1 MS/m = 1e6 S/m
MS_per_m = Unit(
    "MS/m",
    Dimension(mass=-1, length=-3, time=3, current=2),
    1e6
)


# =============================================================================
# Electrical Resistivity Units
# =============================================================================

# SI unit: Ohm-meter (Ω·m)
# Dimension: resistance × length = (kg·m³)/(A²·s³)
# Mass: 1, Length: 3, Time: -3, Current: -2
ohm_m = Unit(
    "Ω·m",
    Dimension(mass=1, length=3, time=-3, current=-2),
    1.0
)

# Common in materials testing: Ohm-centimeter (Ω·cm)
# 1 Ω·cm = 0.01 Ω·m
ohm_cm = Unit(
    "Ω·cm",
    Dimension(mass=1, length=3, time=-3, current=-2),
    0.01
)

# For low-resistivity materials: microohm-centimeter (μΩ·cm)
# 1 μΩ·cm = 1e-8 Ω·m
microohm_cm = Unit(
    "μΩ·cm",
    Dimension(mass=1, length=3, time=-3, current=-2),
    1e-8
)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Stress (re-exported from engineering)
    "gigapascal", "GPa",
    "megapascal", "MPa",
    "kilopascal", "kPa",
    # Strain
    "strain",
    "microstrain",
    "percent_strain",
    # Hardness
    "vickers", "HV",
    "brinell", "HB",
    "rockwell_C", "HRC",
    "rockwell_B", "HRB",
    # Fracture toughness
    "MPa_sqrt_m",
    "ksi_sqrt_in",
    # Thermal conductivity
    "W_per_m_K",
    "W_per_cm_K",
    # Electrical conductivity
    "S_per_m",
    "MS_per_m",
    # Electrical resistivity
    "ohm_m",
    "ohm_cm",
    "microohm_cm",
]

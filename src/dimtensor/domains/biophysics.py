"""Biophysics units for computational biology and biochemistry.

This module provides units commonly used in biophysics, biochemistry, and
cell biology, including enzyme activity (katal, enzyme unit), cell
concentrations, and electrophysiology units (millivolt for membrane potentials).

Example:
    >>> from dimtensor import DimArray
    >>> from dimtensor.domains.biophysics import enzyme_unit, millivolt, cells_per_mL
    >>> activity = DimArray([100], enzyme_unit)  # 100 U enzyme activity
    >>> membrane_v = DimArray([-70], millivolt)  # -70 mV resting potential
    >>> cell_conc = DimArray([1e6], cells_per_mL)  # 1 million cells/mL

Reference values from IUBMB recommendations, CODATA 2022, and standard
biophysics conventions.
"""

from __future__ import annotations

from ..core.dimensions import Dimension
from ..core.units import Unit

# Import chemistry units for cross-reference
from .chemistry import dalton, Da, molar, M, millimolar, mM


# =============================================================================
# Enzyme Activity Units
# =============================================================================

# Katal (SI unit of catalytic activity)
# 1 kat = 1 mol/s (amount converted per second)
# Dimension: amount / time = N * T^-1
katal = Unit("kat", Dimension(amount=1, time=-1), 1.0)
kat = katal

# Enzyme unit (conventional biochemistry unit)
# 1 U = 1 μmol substrate converted per minute (IUBMB standard)
# 1 U = 1e-6 mol / 60 s = 1.66666667e-8 mol/s
# Dimension: amount / time = N * T^-1
enzyme_unit = Unit("U", Dimension(amount=1, time=-1), 1.66666667e-8)
U = enzyme_unit


# =============================================================================
# Cell Biology - Cell Concentrations
# =============================================================================

# Cells per milliliter (common in cell culture)
# Cells are dimensionless counts, so cell/volume has dimension 1/length^3
# 1 cell/mL = 1 / (1e-6 m^3) = 1e6 / m^3
# Dimension: 1/volume = L^-3
cells_per_mL = Unit("cell/mL", Dimension(length=-3), 1e6)

# Cells per microliter
# 1 cell/μL = 1 / (1e-9 m^3) = 1e9 / m^3
# Dimension: 1/volume = L^-3
cells_per_uL = Unit("cell/uL", Dimension(length=-3), 1e9)


# =============================================================================
# Electrophysiology - Membrane Potential
# =============================================================================

# Millivolt (standard unit for membrane potential measurements)
# 1 mV = 0.001 V = 0.001 kg⋅m²/(s³⋅A)
# Dimension: voltage = M * L^2 * T^-3 * I^-1
millivolt = Unit("mV", Dimension(mass=1, length=2, time=-3, current=-1), 0.001)
mV = millivolt


# =============================================================================
# Cross-References to Chemistry Module
# =============================================================================

# Re-export commonly used chemistry units for biophysics convenience
# These are already defined in chemistry.py and imported above:
# - dalton (Da, u): atomic mass unit for protein/molecule masses
# - molar (M): mol/L for substrate and product concentrations
# - millimolar (mM): mmol/L for typical biochemical concentrations


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enzyme activity
    "katal", "kat",
    "enzyme_unit", "U",
    # Cell biology
    "cells_per_mL",
    "cells_per_uL",
    # Electrophysiology
    "millivolt", "mV",
    # Chemistry cross-references (re-exported for convenience)
    "dalton", "Da",
    "molar", "M",
    "millimolar", "mM",
]

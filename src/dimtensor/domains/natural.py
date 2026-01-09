"""Natural units (c = ℏ = 1) for particle physics and quantum field theory.

This module provides natural unit systems where the speed of light c and reduced
Planck constant ℏ are set to 1. In this system:
- Energy, mass, and momentum all have the same dimension
- Length and time are inverse energy: [L] = [T] = 1/[E]

Common in particle physics, quantum field theory, and high-energy physics.

Examples:
    >>> from dimtensor import DimArray
    >>> from dimtensor.domains.natural import GeV, to_natural, from_natural
    >>> # Electron mass in natural units
    >>> m_e = DimArray([0.511e-3], GeV)  # 0.511 MeV
    >>> # Convert SI mass to natural units
    >>> from dimtensor.core.units import kg
    >>> mass_si = DimArray([9.109e-31], kg)  # electron mass in SI
    >>> mass_nat = to_natural(mass_si, 1e9)  # convert to GeV

Reference:
    Conversion factors from CODATA 2022 and PDG 2024.
    - ℏc = 197.3269804 MeV·fm (exact from ℏ and c definitions)
    - ℏ = 6.582119569e-25 GeV·s (exact from ℏ definition)
    - 1 GeV/c² = 1.782661921e-27 kg
"""

from __future__ import annotations

from typing import Literal

from ..core.dimensions import Dimension, DIMENSIONLESS
from ..core.units import Unit
from ..core.dimarray import DimArray


# =============================================================================
# Natural Unit System Constants
# =============================================================================
# Conversion factors for c = ℏ = 1 system with GeV as base unit

# From CODATA 2022 and PDG 2024:
# c = 299792458 m/s (exact)
# ℏ = 1.054571817e-34 J·s (exact)
# ℏc = 197.3269804 MeV·fm (derived)
# 1 eV = 1.602176634e-19 J (exact)

# Mass-energy conversion: E = mc²
# 1 GeV/c² = (1e9 eV) / (299792458 m/s)² * (1.602176634e-19 J/eV)
_GEV_TO_KG = 1.782661921e-27  # kg

# Length conversion: [L] = ℏc/E
# 1 GeV⁻¹ = ℏc / (1 GeV) = 197.3269804 MeV·fm / 1000 MeV = 0.1973269804 fm
_GEV_INV_TO_METER = 1.973269804e-16  # m (0.1973 fm)

# Time conversion: [T] = ℏ/E
# 1 GeV⁻¹ = ℏ / (1 GeV) = 6.582119569e-25 GeV·s
_GEV_INV_TO_SECOND = 6.582119569e-25  # s


# =============================================================================
# Natural Unit Dimensions
# =============================================================================
# In natural units with c = ℏ = 1:
# - Energy, mass, momentum, temperature: dimension [E] = [M] = [p] = [Θ]
# - Length and time: dimension [L] = [T] = [E]⁻¹
# - Action (angular momentum): dimensionless (ℏ = 1)
# - Velocity: dimensionless (c = 1)

# We represent these with custom dimension objects
# Energy dimension (also used for mass, momentum)
_NATURAL_ENERGY_DIM = Dimension(mass=1, length=2, time=-2)  # SI: M L² T⁻²

# Length dimension in natural units (energy⁻¹)
_NATURAL_LENGTH_DIM = Dimension(mass=-1, length=-2, time=2)  # SI: M⁻¹ L⁻² T²

# Time dimension in natural units (energy⁻¹)
_NATURAL_TIME_DIM = Dimension(mass=-1, length=-2, time=2)  # SI: M⁻¹ L⁻² T²


# =============================================================================
# Energy Units (Base Unit in Natural System)
# =============================================================================

# GeV (gigaelectronvolt) - most common base unit in particle physics
GeV = Unit("GeV", _NATURAL_ENERGY_DIM, 1.602176634e-10)

# MeV (megaelectronvolt)
MeV = Unit("MeV", _NATURAL_ENERGY_DIM, 1.602176634e-13)

# eV (electronvolt)
eV = Unit("eV", _NATURAL_ENERGY_DIM, 1.602176634e-19)

# TeV (teraelectronvolt)
TeV = Unit("TeV", _NATURAL_ENERGY_DIM, 1.602176634e-07)


# =============================================================================
# Mass Units (same dimension as energy in natural units)
# =============================================================================

# In natural units, mass has the same dimension as energy
# 1 GeV/c² in SI, but in natural units it's just GeV
GeV_mass = Unit("GeV/c²", _NATURAL_ENERGY_DIM, 1.602176634e-10)
MeV_mass = Unit("MeV/c²", _NATURAL_ENERGY_DIM, 1.602176634e-13)
eV_mass = Unit("eV/c²", _NATURAL_ENERGY_DIM, 1.602176634e-19)


# =============================================================================
# Momentum Units (same dimension as energy in natural units)
# =============================================================================

# In natural units, momentum has the same dimension as energy
# 1 GeV/c in SI, but in natural units it's just GeV
GeV_momentum = Unit("GeV/c", _NATURAL_ENERGY_DIM, 1.602176634e-10)
MeV_momentum = Unit("MeV/c", _NATURAL_ENERGY_DIM, 1.602176634e-13)


# =============================================================================
# Length Units (inverse energy in natural units)
# =============================================================================

# In natural units, 1/GeV is a unit of length
# 1 GeV⁻¹ = 0.1973 fm = 1.973e-16 m
GeV_inv_length = Unit("GeV⁻¹", _NATURAL_LENGTH_DIM, _GEV_INV_TO_METER)
MeV_inv_length = Unit("MeV⁻¹", _NATURAL_LENGTH_DIM, _GEV_INV_TO_METER * 1000)

# Fermi (femtometer) in natural units
# 1 fm = 1e-15 m = 5.068 GeV⁻¹
fermi_natural = Unit("fm", _NATURAL_LENGTH_DIM, 1e-15)
fm_natural = fermi_natural


# =============================================================================
# Time Units (inverse energy in natural units)
# =============================================================================

# In natural units, 1/GeV is a unit of time
# 1 GeV⁻¹ = 6.582e-25 s
GeV_inv_time = Unit("GeV⁻¹", _NATURAL_TIME_DIM, _GEV_INV_TO_SECOND)
MeV_inv_time = Unit("MeV⁻¹", _NATURAL_TIME_DIM, _GEV_INV_TO_SECOND * 1000)


# =============================================================================
# Conversion Functions
# =============================================================================

def to_natural(
    dimarray: DimArray,
    energy_scale: float = 1e9,
) -> DimArray:
    """Convert a DimArray from SI units to natural units (c = ℏ = 1).

    Args:
        dimarray: DimArray in SI units to convert.
        energy_scale: Energy scale in eV for the natural unit system.
                     Default is 1e9 (GeV). Use 1e6 for MeV, 1e12 for TeV.

    Returns:
        DimArray in natural units.

    Examples:
        >>> from dimtensor import DimArray
        >>> from dimtensor.core.units import kg, meter, second
        >>> from dimtensor.domains.natural import to_natural
        >>> # Convert mass
        >>> m = DimArray([9.109e-31], kg)  # electron mass
        >>> m_nat = to_natural(m, 1e9)  # to GeV
        >>> # Convert length
        >>> r = DimArray([1e-15], meter)  # 1 fm
        >>> r_nat = to_natural(r, 1e9)  # to GeV^-1

    Notes:
        In natural units with c = ℏ = 1:
        - Mass → Energy: m → mc² = m / (1.783e-27 kg/GeV)
        - Length → 1/Energy: L → L / (1.973e-16 m·GeV)
        - Time → 1/Energy: t → t / (6.582e-25 s·GeV)
        - Momentum → Energy: p → pc = p / (5.344e-19 kg·m/s·GeV)
    """
    dim = dimarray.unit.dimension
    si_scale = dimarray.unit.scale

    # Scale factors relative to the chosen energy scale
    # GeV is 1e9 eV, so if energy_scale is different, adjust
    scale_ratio = energy_scale / 1e9  # ratio to GeV

    # Determine the natural unit based on SI dimension
    # Energy: M L² T⁻²
    if dim == Dimension(mass=1, length=2, time=-2):
        # Energy: convert J to eV
        # 1 J = 1 / 1.602176634e-19 eV
        nat_scale = si_scale / (1.602176634e-19 * energy_scale)
        nat_unit = Unit("nat_E", _NATURAL_ENERGY_DIM, si_scale)
        return DimArray._from_data_and_unit(
            dimarray.data * nat_scale,
            nat_unit,
            dimarray.uncertainty * nat_scale if dimarray.uncertainty is not None else None,
        )

    # Mass: M
    elif dim == Dimension(mass=1):
        # Mass: convert kg to GeV/c²
        # 1 kg = 1 / 1.783e-27 GeV (using c² implicitly)
        nat_scale = si_scale / (_GEV_TO_KG * scale_ratio)
        nat_unit = Unit("nat_M", _NATURAL_ENERGY_DIM, si_scale)
        return DimArray._from_data_and_unit(
            dimarray.data * nat_scale,
            nat_unit,
            dimarray.uncertainty * nat_scale if dimarray.uncertainty is not None else None,
        )

    # Momentum: M L T⁻¹
    elif dim == Dimension(mass=1, length=1, time=-1):
        # Momentum: convert kg·m/s to GeV/c
        # 1 kg·m/s = c / (1.783e-27 kg·GeV) = 299792458 / 1.783e-27
        nat_scale = si_scale * 299792458.0 / (_GEV_TO_KG * scale_ratio)
        nat_unit = Unit("nat_p", _NATURAL_ENERGY_DIM, si_scale)
        return DimArray._from_data_and_unit(
            dimarray.data * nat_scale,
            nat_unit,
            dimarray.uncertainty * nat_scale if dimarray.uncertainty is not None else None,
        )

    # Length: L
    elif dim == Dimension(length=1):
        # Length: convert m to GeV⁻¹
        # 1 m = 1 / 1.973e-16 GeV⁻¹
        nat_scale = si_scale / (_GEV_INV_TO_METER / scale_ratio)
        nat_unit = Unit("nat_L", _NATURAL_LENGTH_DIM, si_scale)
        return DimArray._from_data_and_unit(
            dimarray.data * nat_scale,
            nat_unit,
            dimarray.uncertainty * nat_scale if dimarray.uncertainty is not None else None,
        )

    # Time: T
    elif dim == Dimension(time=1):
        # Time: convert s to GeV⁻¹
        # 1 s = 1 / 6.582e-25 GeV⁻¹
        nat_scale = si_scale / (_GEV_INV_TO_SECOND / scale_ratio)
        nat_unit = Unit("nat_T", _NATURAL_TIME_DIM, si_scale)
        return DimArray._from_data_and_unit(
            dimarray.data * nat_scale,
            nat_unit,
            dimarray.uncertainty * nat_scale if dimarray.uncertainty is not None else None,
        )

    # Velocity: L T⁻¹ (dimensionless in natural units)
    elif dim == Dimension(length=1, time=-1):
        # Velocity: convert m/s to c (dimensionless)
        # 1 m/s = 1/299792458 c
        nat_scale = si_scale / 299792458.0
        nat_unit = Unit("nat_v", DIMENSIONLESS, 1.0)
        return DimArray._from_data_and_unit(
            dimarray.data * nat_scale,
            nat_unit,
            dimarray.uncertainty * nat_scale if dimarray.uncertainty is not None else None,
        )

    else:
        raise ValueError(
            f"Unsupported dimension for natural units conversion: {dim}. "
            "Supported: energy, mass, momentum, length, time, velocity."
        )


def from_natural(
    value: float | list | DimArray,
    quantity_type: Literal["energy", "mass", "momentum", "length", "time", "velocity"],
    energy_scale: float = 1e9,
) -> DimArray:
    """Convert a value from natural units (c = ℏ = 1) to SI units.

    Args:
        value: Numerical value in natural units (scalar, list, or DimArray).
        quantity_type: Type of physical quantity.
                      Options: "energy", "mass", "momentum", "length", "time", "velocity"
        energy_scale: Energy scale in eV used in the natural unit system.
                     Default is 1e9 (GeV). Use 1e6 for MeV, 1e12 for TeV.

    Returns:
        DimArray in SI units.

    Examples:
        >>> from dimtensor.domains.natural import from_natural
        >>> # Convert natural mass to SI
        >>> m_nat = 0.511e-3  # electron mass in GeV
        >>> m_si = from_natural(m_nat, "mass", 1e9)
        >>> print(m_si)  # Should be ~9.109e-31 kg
        >>> # Convert natural length to SI
        >>> r_nat = 5.0  # 5 GeV^-1
        >>> r_si = from_natural(r_nat, "length", 1e9)
        >>> print(r_si)  # Should be ~9.87e-16 m

    Notes:
        In natural units with c = ℏ = 1:
        - Energy → Joules: E → E * 1.602e-19 J/eV
        - Mass → kg: m → m * 1.783e-27 kg/GeV
        - Momentum → kg·m/s: p → p * 5.344e-19 kg·m/s/GeV
        - Length → meters: L → L * 1.973e-16 m·GeV
        - Time → seconds: t → t * 6.582e-25 s·GeV
        - Velocity → m/s: v → v * 299792458 m/s
    """
    # Convert input to array
    if isinstance(value, DimArray):
        data = value.data
    else:
        import numpy as np
        data = np.asarray(value)

    # Scale factors relative to the chosen energy scale
    scale_ratio = energy_scale / 1e9  # ratio to GeV

    # Convert based on quantity type
    if quantity_type == "energy":
        # Energy: convert eV to J
        # 1 eV = 1.602176634e-19 J
        si_value = data * (1.602176634e-19 * energy_scale)
        from ..core.units import joule
        return DimArray(si_value, joule)

    elif quantity_type == "mass":
        # Mass: convert natural mass (GeV) to kg
        # 1 GeV/c² = 1.783e-27 kg
        si_value = data * (_GEV_TO_KG * scale_ratio)
        from ..core.units import kilogram
        return DimArray(si_value, kilogram)

    elif quantity_type == "momentum":
        # Momentum: convert natural momentum (GeV) to kg·m/s
        # 1 GeV/c = 5.344e-19 kg·m/s
        si_value = data * (_GEV_TO_KG * scale_ratio * 299792458.0)
        from ..core.units import kilogram, meter, second
        return DimArray(si_value, kilogram * meter / second)

    elif quantity_type == "length":
        # Length: convert GeV⁻¹ to m
        # 1 GeV⁻¹ = 1.973e-16 m
        si_value = data * (_GEV_INV_TO_METER / scale_ratio)
        from ..core.units import meter
        return DimArray(si_value, meter)

    elif quantity_type == "time":
        # Time: convert GeV⁻¹ to s
        # 1 GeV⁻¹ = 6.582e-25 s
        si_value = data * (_GEV_INV_TO_SECOND / scale_ratio)
        from ..core.units import second
        return DimArray(si_value, second)

    elif quantity_type == "velocity":
        # Velocity: dimensionless in natural units, multiply by c
        si_value = data * 299792458.0
        from ..core.units import meter, second
        return DimArray(si_value, meter / second)

    else:
        raise ValueError(
            f"Unknown quantity_type: {quantity_type}. "
            "Supported: 'energy', 'mass', 'momentum', 'length', 'time', 'velocity'"
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Energy units
    "GeV", "MeV", "eV", "TeV",
    # Mass units
    "GeV_mass", "MeV_mass", "eV_mass",
    # Momentum units
    "GeV_momentum", "MeV_momentum",
    # Length units
    "GeV_inv_length", "MeV_inv_length",
    "fermi_natural", "fm_natural",
    # Time units
    "GeV_inv_time", "MeV_inv_time",
    # Conversion functions
    "to_natural", "from_natural",
]

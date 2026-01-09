"""Planck units for natural unit systems and quantum gravity.

This module provides Planck units, a system of natural units where the fundamental
constants c, ℏ, G, and k_B equal 1. These units are particularly important in
quantum gravity, string theory, and theoretical physics.

The Planck scale represents the regime where quantum effects and gravitational
effects become equally important, marking the limit of validity for current
physical theories.

Example:
    >>> from dimtensor import DimArray
    >>> from dimtensor.domains.planck import planck_length, planck_time
    >>> l_p = DimArray([1.0], planck_length)
    >>> t_p = DimArray([1.0], planck_time)
    >>> print(l_p)  # ~1.616e-35 m
    >>> print(t_p)  # ~5.391e-44 s

Reference values derived from CODATA 2022 constants (c, ℏ, G, ε₀, k_B).
"""

from __future__ import annotations

import math

from ..core.dimensions import Dimension
from ..core.units import Unit

# Import fundamental constants (CODATA 2022)
from ..constants.universal import c, hbar, G
from ..constants.electromagnetic import epsilon_0
from ..constants.physico_chemical import k_B


# =============================================================================
# Base Planck Units
# =============================================================================

# Planck length: l_P = sqrt(ℏG/c³)
# The length scale at which quantum gravity effects become significant
_planck_length_value = math.sqrt(
    (hbar.value * G.value) / (c.value ** 3)
)
planck_length = Unit("l_P", Dimension(length=1), _planck_length_value)
l_P = planck_length

# Planck mass: m_P = sqrt(ℏc/G)
# The mass scale at which quantum and gravitational effects are equally important
_planck_mass_value = math.sqrt(
    (hbar.value * c.value) / G.value
)
planck_mass = Unit("m_P", Dimension(mass=1), _planck_mass_value)
m_P = planck_mass

# Planck time: t_P = sqrt(ℏG/c⁵)
# The time scale for quantum gravitational processes
_planck_time_value = math.sqrt(
    (hbar.value * G.value) / (c.value ** 5)
)
planck_time = Unit("t_P", Dimension(time=1), _planck_time_value)
t_P = planck_time

# Planck charge: q_P = sqrt(4πε₀ℏc)
# The natural unit of electric charge in the Planck system
_planck_charge_value = math.sqrt(
    4.0 * math.pi * epsilon_0.value * hbar.value * c.value
)
planck_charge = Unit("q_P", Dimension(current=1, time=1), _planck_charge_value)
q_P = planck_charge

# Planck temperature: T_P = sqrt(ℏc⁵/(Gk_B²))
# The temperature at which thermal energy equals Planck energy
_planck_temperature_value = math.sqrt(
    (hbar.value * c.value ** 5) / (G.value * k_B.value ** 2)
)
planck_temperature = Unit("T_P", Dimension(temperature=1), _planck_temperature_value)
T_P = planck_temperature


# =============================================================================
# Derived Planck Units - Energy and Momentum
# =============================================================================

# Planck energy: E_P = m_P·c² = sqrt(ℏc⁵/G)
# The energy scale at which quantum gravity becomes important
_planck_energy_value = _planck_mass_value * (c.value ** 2)
planck_energy = Unit("E_P", Dimension(mass=1, length=2, time=-2), _planck_energy_value)
E_P = planck_energy

# Planck momentum: p_P = m_P·c = sqrt(ℏc³/G)
_planck_momentum_value = _planck_mass_value * c.value
planck_momentum = Unit("p_P", Dimension(mass=1, length=1, time=-1), _planck_momentum_value)
p_P = planck_momentum


# =============================================================================
# Derived Planck Units - Force and Power
# =============================================================================

# Planck force: F_P = c⁴/G
# The maximum possible force in physics (according to some theories)
_planck_force_value = (c.value ** 4) / G.value
planck_force = Unit("F_P", Dimension(mass=1, length=1, time=-2), _planck_force_value)
F_P = planck_force

# Planck power: P_P = c⁵/G
# The maximum possible power output
_planck_power_value = (c.value ** 5) / G.value
planck_power = Unit("P_P", Dimension(mass=1, length=2, time=-3), _planck_power_value)
P_P = planck_power


# =============================================================================
# Derived Planck Units - Density and Acceleration
# =============================================================================

# Planck density: ρ_P = c⁵/(ℏG²)
# The energy density at the Planck scale
_planck_density_value = (c.value ** 5) / (hbar.value * G.value ** 2)
planck_density = Unit("rho_P", Dimension(mass=1, length=-3), _planck_density_value)
rho_P = planck_density

# Planck acceleration: a_P = c/t_P = sqrt(c⁷/(ℏG))
_planck_acceleration_value = math.sqrt((c.value ** 7) / (hbar.value * G.value))
planck_acceleration = Unit("a_P", Dimension(length=1, time=-2), _planck_acceleration_value)
a_P = planck_acceleration


# =============================================================================
# Derived Planck Units - Spatial Measures
# =============================================================================

# Planck area: A_P = l_P²
_planck_area_value = _planck_length_value ** 2
planck_area = Unit("A_P", Dimension(length=2), _planck_area_value)
A_P = planck_area

# Planck volume: V_P = l_P³
_planck_volume_value = _planck_length_value ** 3
planck_volume = Unit("V_P", Dimension(length=3), _planck_volume_value)
V_P = planck_volume


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base Planck units
    "planck_length", "l_P",
    "planck_mass", "m_P",
    "planck_time", "t_P",
    "planck_charge", "q_P",
    "planck_temperature", "T_P",
    # Energy and momentum
    "planck_energy", "E_P",
    "planck_momentum", "p_P",
    # Force and power
    "planck_force", "F_P",
    "planck_power", "P_P",
    # Density and acceleration
    "planck_density", "rho_P",
    "planck_acceleration", "a_P",
    # Spatial measures
    "planck_area", "A_P",
    "planck_volume", "V_P",
]

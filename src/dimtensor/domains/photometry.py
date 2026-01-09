"""Photometry units for light measurement and optical sciences.

This module provides units for measuring visible light in terms of human
perception, including luminous intensity, flux, illuminance, luminance,
and luminous efficacy. These units are essential for lighting design,
display technology, photography, and vision science.

Example:
    >>> from dimtensor import DimArray
    >>> from dimtensor.domains.photometry import lumen, lux, nit
    >>> light_output = DimArray([800], lumen)  # LED bulb output
    >>> illuminance = DimArray([500], lux)  # Office lighting
    >>> brightness = DimArray([300], nit)  # Monitor brightness

Note:
    The steradian (sr) is dimensionless in SI, so luminous flux (lumen)
    and luminous intensity (candela) have the same dimension.

    Illuminance (lux) and luminance (nit) both have dimension J¹L⁻² but
    represent different concepts:
    - Illuminance: light incident ON a surface (e.g., room lighting)
    - Luminance: light emitted/reflected FROM a surface (e.g., screen brightness)

Reference values from SI Brochure 9th Edition (2019) and NIST SP 811.
"""

from __future__ import annotations

import math

from ..core.dimensions import Dimension, DIMENSIONLESS
from ..core.units import Unit, candela


# =============================================================================
# Luminous Intensity (Base Unit)
# =============================================================================

# Candela (SI base unit) - re-exported for convenience
# The candela is the luminous intensity, in a given direction, of a source
# that emits monochromatic radiation of frequency 540×10¹² hertz and that
# has a radiant intensity in that direction of 1/683 watt per steradian.
cd = candela


# =============================================================================
# Luminous Flux (lumen = candela × steradian)
# =============================================================================

# Lumen (SI derived unit for luminous flux)
# Since steradian is dimensionless, lumen has the same dimension as candela
lumen = Unit("lm", Dimension(luminosity=1), 1.0)
lm = lumen


# =============================================================================
# Illuminance (luminous flux per area)
# =============================================================================

# Lux (SI derived unit: lumen per square meter)
# 1 lx = 1 lm/m²
lux = Unit("lx", Dimension(luminosity=1, length=-2), 1.0)
lx = lux

# Foot-candle (lumen per square foot)
# 1 fc = 1 lm/ft² = 10.76391... lm/m²
# 1 ft = 0.3048 m, so 1 ft² = 0.09290304 m²
foot_candle = Unit("fc", Dimension(luminosity=1, length=-2), 1.0 / 0.09290304)
fc = foot_candle

# Phot (lumen per square centimeter)
# 1 ph = 1 lm/cm² = 10000 lm/m²
phot = Unit("ph", Dimension(luminosity=1, length=-2), 10000.0)
ph = phot


# =============================================================================
# Luminance (luminous intensity per area)
# =============================================================================

# Nit (candela per square meter)
# 1 nit = 1 cd/m²
# This is the standard SI unit for luminance, commonly used for displays
nit = Unit("nit", Dimension(luminosity=1, length=-2), 1.0)
cd_per_m2 = nit  # Alternative name

# Stilb (candela per square centimeter)
# 1 sb = 1 cd/cm² = 10000 cd/m²
stilb = Unit("sb", Dimension(luminosity=1, length=-2), 10000.0)
sb = stilb

# Lambert (candela per square centimeter per pi)
# 1 La = (1/π) cd/cm² = 10000/π cd/m² ≈ 3183.099 cd/m²
# Named after Johann Heinrich Lambert
lambert = Unit("La", Dimension(luminosity=1, length=-2), 10000.0 / math.pi)
La = lambert


# =============================================================================
# Luminous Energy (luminous flux × time)
# =============================================================================

# Lumen-second (luminous energy)
# 1 lm·s = 1 cd·sr·s
lumen_second = Unit("lm·s", Dimension(luminosity=1, time=1), 1.0)

# Talbot (alternative name for lumen-second)
# Named after William Henry Fox Talbot
talbot = lumen_second


# =============================================================================
# Luminous Efficacy (luminous flux per power)
# =============================================================================

# Lumen per watt (luminous efficacy)
# Measures how efficiently a light source converts electrical power to visible light
# 1 lm/W = 1 cd·sr/W = 1 cd/(kg·m²·s⁻³) = 1 cd·s³·kg⁻¹·m⁻²
# Maximum theoretical efficacy for white light is ~683 lm/W at 555 nm (peak photopic vision)
lm_per_W = Unit("lm/W", Dimension(luminosity=1, mass=-1, length=-2, time=3), 1.0)
lumen_per_watt = lm_per_W


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Luminous intensity
    "candela", "cd",
    # Luminous flux
    "lumen", "lm",
    # Illuminance
    "lux", "lx",
    "foot_candle", "fc",
    "phot", "ph",
    # Luminance
    "nit", "cd_per_m2",
    "stilb", "sb",
    "lambert", "La",
    # Luminous energy
    "lumen_second",
    "talbot",
    # Luminous efficacy
    "lm_per_W", "lumen_per_watt",
]

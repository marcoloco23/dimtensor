"""Acoustics units for audio engineering and psychoacoustics.

This module provides units commonly used in acoustics, audio engineering,
room acoustics, and psychoacoustics. This includes acoustic impedance (rayl),
sound pressure levels (micropascal), and psychoacoustic measures (phon, sone).

Example:
    >>> from dimtensor import DimArray
    >>> from dimtensor.domains.acoustics import rayl, uPa, dB
    >>> impedance = DimArray([415.0], rayl)  # Acoustic impedance of air at STP
    >>> pressure = DimArray([20.0], uPa)  # Reference pressure for SPL
    >>> # Note: dB is dimensionless; actual SPL calculation requires logarithm
    >>> # SPL_dB = 20 * log10(p / p_ref) where p_ref = 20 uPa

Warning:
    The decibel (dB) unit is LOGARITHMIC and cannot be directly converted
    using linear scaling. This unit represents dimensionless ratios only.
    For sound pressure level (SPL): SPL_dB = 20·log₁₀(p/p₀) where p₀ = 20 μPa
    For sound power level (SWL): SWL_dB = 10·log₁₀(P/P₀) where P₀ = 1 pW

    Similarly, phon and sone are psychoacoustic scales that require specific
    conversion formulas based on equal-loudness contours (ISO 226) and are
    frequency-dependent. These are provided as dimensionless reference units.

Reference values from ISO 1683, ANSI S1.1, and psychoacoustic standards.
"""

from __future__ import annotations

from ..core.dimensions import Dimension, DIMENSIONLESS
from ..core.units import Unit, hertz, pascal, watt


# =============================================================================
# Frequency Units
# =============================================================================

# Hertz (alias from core units for convenience)
Hz = hertz

# Kilohertz
kilohertz = Unit("kHz", Dimension(time=-1), 1000.0)
kHz = kilohertz


# =============================================================================
# Pressure Units
# =============================================================================

# Pascal (alias from core units for convenience)
Pa = pascal

# Micropascal (reference for sound pressure level in air)
# Reference SPL: 20 μPa (threshold of hearing at 1 kHz)
micropascal = Unit("μPa", Dimension(mass=1, length=-1, time=-2), 1e-6)
uPa = micropascal


# =============================================================================
# Acoustic Impedance
# =============================================================================

# Rayl (acoustic impedance: Pa·s/m = kg/(m²·s))
# Named after Lord Rayleigh
# Dimension: mass / (length² · time) = M L^-2 T^-1
# Example: Acoustic impedance of air at STP ≈ 415 rayl
rayl = Unit("rayl", Dimension(mass=1, length=-2, time=-1), 1.0)


# =============================================================================
# Power Units
# =============================================================================

# Acoustic watt (alias from core units for convenience)
acoustic_watt = watt
W = watt


# =============================================================================
# Dimensionless Logarithmic and Psychoacoustic Units
# =============================================================================

# Decibel (dimensionless logarithmic ratio)
# WARNING: This is a dimensionless reference unit only.
# Actual dB values require logarithmic conversion:
#   - Sound Pressure Level: SPL = 20·log₁₀(p/p₀) where p₀ = 20 μPa
#   - Sound Power Level: SWL = 10·log₁₀(P/P₀) where P₀ = 1 pW
#   - Sound Intensity Level: SIL = 10·log₁₀(I/I₀) where I₀ = 1 pW/m²
decibel = Unit("dB", DIMENSIONLESS, 1.0)
dB = decibel

# Phon (loudness level, dimensionless)
# Unit of loudness level on equal-loudness contours (ISO 226:2003)
# A pure tone at 1 kHz with SPL of X dB has loudness level of X phon
# Requires equal-loudness contour tables for frequency-dependent conversion
phon = Unit("phon", DIMENSIONLESS, 1.0)

# Sone (loudness, dimensionless)
# Linear perceptual loudness scale
# Conversion from phon: L_sone = 2^((L_phon - 40)/10)
# 1 sone is defined as the loudness of a 1 kHz tone at 40 dB SPL
# 2 sones is twice as loud, 0.5 sone is half as loud
sone = Unit("sone", DIMENSIONLESS, 1.0)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Frequency
    "Hz", "hertz",
    "kilohertz", "kHz",
    # Pressure
    "Pa", "pascal",
    "micropascal", "uPa",
    # Acoustic impedance
    "rayl",
    # Power
    "acoustic_watt", "W", "watt",
    # Dimensionless logarithmic/psychoacoustic
    "decibel", "dB",
    "phon",
    "sone",
]

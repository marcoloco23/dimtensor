"""Information theory units for data storage, transmission, and entropy.

This module provides units commonly used in information theory, computer science,
and telecommunications, including information content units (bit, byte, nat),
entropy measures (bit/symbol), data rates (bit/s), and storage densities (bit/m²).

Information content units are mathematically dimensionless but provide important
semantic meaning for calculations involving Shannon entropy, channel capacity,
data storage, and network bandwidth.

Example:
    >>> from dimtensor import DimArray
    >>> from dimtensor.domains.information import bit, byte, kilobyte
    >>> data_size = DimArray([1024], byte)
    >>> data_in_bits = data_size.to(bit)  # Convert to bits
    >>> # Shannon entropy calculation
    >>> from dimtensor.domains.information import bit_per_symbol
    >>> import numpy as np
    >>> p = np.array([0.5, 0.5])  # Binary equiprobable source
    >>> H = -np.sum(p * np.log2(p))
    >>> entropy = DimArray([H], bit_per_symbol)  # 1.0 bit/symbol

Note on binary prefixes:
    This module follows the IEC 60027-2 standard for binary prefixes:
    - 1 kilobyte (KB) = 1024 bytes = 8192 bits
    - 1 megabyte (MB) = 1024² bytes = 1048576 bytes
    - 1 gigabyte (GB) = 1024³ bytes
    - 1 terabyte (TB) = 1024⁴ bytes

References:
    - Shannon, C.E. (1948). "A Mathematical Theory of Communication"
    - IEC 60027-2: Letter symbols for quantities (binary prefixes)
    - NIST Special Publication 811: Guide for the Use of SI Units
"""

from __future__ import annotations

import math

from ..core.dimensions import Dimension, DIMENSIONLESS
from ..core.units import Unit


# =============================================================================
# Information Content Units
# =============================================================================

# Bit (base unit of information)
# Dimensionless, scale = 1.0
bit = Unit("bit", DIMENSIONLESS, 1.0)

# Shannon (alias for bit, common in information theory literature)
shannon = Unit("Sh", DIMENSIONLESS, 1.0)
Sh = shannon

# Nat (natural unit of information, using natural logarithm base e)
# 1 nat = ln(2) bits ≈ 0.693147 bits
nat = Unit("nat", DIMENSIONLESS, math.log(2))

# Byte/Octet (8 bits)
byte = Unit("B", DIMENSIONLESS, 8.0)
B = byte
octet = byte

# Binary prefixes (IEC 60027-2 standard: powers of 1024)
# Kilobyte (1024 bytes = 8192 bits)
kilobyte = Unit("KB", DIMENSIONLESS, 8192.0)
KB = kilobyte
kibibyte = kilobyte  # Alternative IEC name
KiB = kilobyte

# Megabyte (1024² bytes = 1048576 bytes = 8388608 bits)
megabyte = Unit("MB", DIMENSIONLESS, 8388608.0)
MB = megabyte
mebibyte = megabyte
MiB = megabyte

# Gigabyte (1024³ bytes = 1073741824 bytes = 8589934592 bits)
gigabyte = Unit("GB", DIMENSIONLESS, 8589934592.0)
GB = gigabyte
gibibyte = gigabyte
GiB = gigabyte

# Terabyte (1024⁴ bytes = 1099511627776 bytes)
terabyte = Unit("TB", DIMENSIONLESS, 8796093022208.0)
TB = terabyte
tebibyte = terabyte
TiB = terabyte


# =============================================================================
# Entropy Units (information per symbol)
# =============================================================================

# Bit per symbol (dimensionless, scale = 1.0)
# Used for Shannon entropy: H(X) = -Σ p(x) log₂(p(x)) bits/symbol
bit_per_symbol = Unit("bit/symbol", DIMENSIONLESS, 1.0)

# Nat per symbol (natural unit entropy)
nat_per_symbol = Unit("nat/symbol", DIMENSIONLESS, math.log(2))


# =============================================================================
# Data Rate Units (information per time)
# =============================================================================

# Bit per second (dimension: time^-1)
bit_per_second = Unit("bit/s", Dimension(time=-1), 1.0)
bps = bit_per_second

# Kilobit per second (1000 bits/s, using decimal for data rates per convention)
kilobit_per_second = Unit("kbit/s", Dimension(time=-1), 1e3)
kbps = kilobit_per_second

# Megabit per second (10^6 bits/s)
megabit_per_second = Unit("Mbit/s", Dimension(time=-1), 1e6)
Mbps = megabit_per_second

# Gigabit per second (10^9 bits/s)
gigabit_per_second = Unit("Gbit/s", Dimension(time=-1), 1e9)
Gbps = gigabit_per_second

# Byte per second (8 bits/s)
byte_per_second = Unit("B/s", Dimension(time=-1), 8.0)
Bps = byte_per_second

# Kilobyte per second (8192 bits/s, binary prefix)
kilobyte_per_second = Unit("KB/s", Dimension(time=-1), 8192.0)
KBps = kilobyte_per_second

# Megabyte per second (8388608 bits/s, binary prefix)
megabyte_per_second = Unit("MB/s", Dimension(time=-1), 8388608.0)
MBps = megabyte_per_second

# Baud (symbols per second)
# Note: baud and bit/s are distinct when encoding multiple bits per symbol
baud = Unit("Bd", Dimension(time=-1), 1.0)
Bd = baud


# =============================================================================
# Storage Density Units
# =============================================================================

# Bit per square meter (areal density, dimension: length^-2)
# Used for magnetic/optical storage density
bit_per_square_meter = Unit("bit/m²", Dimension(length=-2), 1.0)
bit_per_m2 = bit_per_square_meter

# Bit per cubic meter (volumetric density, dimension: length^-3)
# Used for 3D storage density
bit_per_cubic_meter = Unit("bit/m³", Dimension(length=-3), 1.0)
bit_per_m3 = bit_per_cubic_meter

# Gigabit per square inch (common in hard drive specifications)
# 1 inch = 0.0254 m, 1 in² = 6.4516e-4 m²
# 1 Gbit/in² = 10^9 bit / 6.4516e-4 m² = 1.5500031e12 bit/m²
gigabit_per_square_inch = Unit("Gbit/in²", Dimension(length=-2), 1.5500031e12)
Gbit_per_in2 = gigabit_per_square_inch


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Information content
    "bit",
    "shannon", "Sh",
    "nat",
    "byte", "B", "octet",
    "kilobyte", "KB", "kibibyte", "KiB",
    "megabyte", "MB", "mebibyte", "MiB",
    "gigabyte", "GB", "gibibyte", "GiB",
    "terabyte", "TB", "tebibyte", "TiB",
    # Entropy
    "bit_per_symbol",
    "nat_per_symbol",
    # Data rates
    "bit_per_second", "bps",
    "kilobit_per_second", "kbps",
    "megabit_per_second", "Mbps",
    "gigabit_per_second", "Gbps",
    "byte_per_second", "Bps",
    "kilobyte_per_second", "KBps",
    "megabyte_per_second", "MBps",
    "baud", "Bd",
    # Storage density
    "bit_per_square_meter", "bit_per_m2",
    "bit_per_cubic_meter", "bit_per_m3",
    "gigabit_per_square_inch", "Gbit_per_in2",
]

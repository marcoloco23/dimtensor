"""Domain-specific unit collections.

This module provides specialized units for different scientific domains:

- astronomy: parsec, AU, solar_mass, light_year, etc.
- chemistry: molar, dalton, ppm, angstrom, etc.
- engineering: MPa, ksi, BTU, horsepower, etc.
- nuclear: MeV, barn, becquerel, gray, sievert, etc.
- geophysics: gal, eotvos, darcy, gamma, oersted, etc.
- biophysics: katal, enzyme_unit, cells_per_mL, etc.
- materials: strain, vickers, MPa_sqrt_m, thermal conductivity, etc.
- acoustics: decibel, phon, sone, rayl, etc.
- photometry: lumen, lux, nit, foot_candle, etc.
- information: bit, byte, nat, bit_per_second, etc.
- cgs: dyne, erg, poise, gauss, maxwell, etc.
- imperial: inch, pound, gallon, BTU, psi, etc.
- natural: GeV units for particle physics (c=ℏ=1)
- planck: Planck length, mass, time, energy, etc.

Example:
    >>> from dimtensor.domains.astronomy import parsec, AU
    >>> from dimtensor.domains.chemistry import molar, dalton
    >>> from dimtensor.domains.engineering import MPa, hp
    >>> from dimtensor.domains.nuclear import MeV, barn, gray
    >>> from dimtensor.domains.cgs import dyne, erg, gauss
    >>> from dimtensor.domains.imperial import foot, pound, gallon
"""

from . import astronomy
from . import chemistry
from . import engineering
from . import nuclear
from . import geophysics
from . import biophysics
from . import materials
from . import acoustics
from . import photometry
from . import information
from . import cgs
from . import imperial
from . import natural
from . import planck

__all__ = [
    "astronomy",
    "chemistry",
    "engineering",
    "nuclear",
    "geophysics",
    "biophysics",
    "materials",
    "acoustics",
    "photometry",
    "information",
    "cgs",
    "imperial",
    "natural",
    "planck",
]

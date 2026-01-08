"""Dimension algebra for physical quantities.

Dimensions are represented as a tuple of exponents for the 7 SI base dimensions:
- Length (L): meter
- Mass (M): kilogram
- Time (T): second
- Electric current (I): ampere
- Temperature (Θ): kelvin
- Amount of substance (N): mole
- Luminous intensity (J): candela
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple

# Indices for each base dimension
LENGTH = 0
MASS = 1
TIME = 2
CURRENT = 3
TEMPERATURE = 4
AMOUNT = 5
LUMINOSITY = 6

# Dimension symbols for pretty printing
_DIMENSION_SYMBOLS = ["L", "M", "T", "I", "Θ", "N", "J"]
_DIMENSION_NAMES = [
    "length",
    "mass",
    "time",
    "current",
    "temperature",
    "amount",
    "luminosity",
]


@dataclass(frozen=True, slots=True)
class Dimension:
    """Represents the physical dimension of a quantity.

    A dimension is represented as a tuple of rational exponents for each
    of the 7 SI base dimensions. For example:
    - Velocity has dimension L¹T⁻¹ (length=1, time=-1)
    - Force has dimension M¹L¹T⁻² (mass=1, length=1, time=-2)

    Dimensions support algebraic operations:
    - Multiplication: adds exponents
    - Division: subtracts exponents
    - Power: multiplies exponents by the power
    """

    _exponents: Tuple[Fraction, ...]

    def __init__(
        self,
        length: int | float | Fraction = 0,
        mass: int | float | Fraction = 0,
        time: int | float | Fraction = 0,
        current: int | float | Fraction = 0,
        temperature: int | float | Fraction = 0,
        amount: int | float | Fraction = 0,
        luminosity: int | float | Fraction = 0,
    ) -> None:
        """Create a dimension from base dimension exponents."""
        exponents = tuple(
            Fraction(x).limit_denominator(1000)
            for x in (length, mass, time, current, temperature, amount, luminosity)
        )
        object.__setattr__(self, "_exponents", exponents)

    @classmethod
    def _from_exponents(cls, exponents: Tuple[Fraction, ...]) -> Dimension:
        """Create a dimension from a tuple of exponents (internal use)."""
        dim = object.__new__(cls)
        object.__setattr__(dim, "_exponents", exponents)
        return dim

    @property
    def length(self) -> Fraction:
        """Exponent of length dimension."""
        return self._exponents[LENGTH]

    @property
    def mass(self) -> Fraction:
        """Exponent of mass dimension."""
        return self._exponents[MASS]

    @property
    def time(self) -> Fraction:
        """Exponent of time dimension."""
        return self._exponents[TIME]

    @property
    def current(self) -> Fraction:
        """Exponent of electric current dimension."""
        return self._exponents[CURRENT]

    @property
    def temperature(self) -> Fraction:
        """Exponent of temperature dimension."""
        return self._exponents[TEMPERATURE]

    @property
    def amount(self) -> Fraction:
        """Exponent of amount of substance dimension."""
        return self._exponents[AMOUNT]

    @property
    def luminosity(self) -> Fraction:
        """Exponent of luminous intensity dimension."""
        return self._exponents[LUMINOSITY]

    @property
    def is_dimensionless(self) -> bool:
        """Check if this is a dimensionless quantity."""
        return all(exp == 0 for exp in self._exponents)

    def __mul__(self, other: object) -> Dimension:
        """Multiply dimensions (add exponents)."""
        if not isinstance(other, Dimension):
            return NotImplemented  # type: ignore[return-value]
        new_exponents = tuple(
            a + b for a, b in zip(self._exponents, other._exponents)
        )
        return Dimension._from_exponents(new_exponents)

    def __truediv__(self, other: object) -> Dimension:
        """Divide dimensions (subtract exponents)."""
        if not isinstance(other, Dimension):
            return NotImplemented  # type: ignore[return-value]
        new_exponents = tuple(
            a - b for a, b in zip(self._exponents, other._exponents)
        )
        return Dimension._from_exponents(new_exponents)

    def __pow__(self, power: int | float | Fraction) -> Dimension:
        """Raise dimension to a power (multiply exponents)."""
        p = Fraction(power).limit_denominator(1000)
        new_exponents = tuple(exp * p for exp in self._exponents)
        return Dimension._from_exponents(new_exponents)

    def __eq__(self, other: object) -> bool:
        """Check dimension equality."""
        if not isinstance(other, Dimension):
            return NotImplemented
        return self._exponents == other._exponents

    def __hash__(self) -> int:
        """Hash for use in sets and dicts."""
        return hash(self._exponents)

    def __repr__(self) -> str:
        """Detailed string representation."""
        parts = []
        for name, exp in zip(_DIMENSION_NAMES, self._exponents):
            if exp != 0:
                parts.append(f"{name}={exp}")
        if not parts:
            return "Dimension(dimensionless)"
        return f"Dimension({', '.join(parts)})"

    def __str__(self) -> str:
        """Human-readable dimension string like 'L¹M¹T⁻²'."""
        if self.is_dimensionless:
            return "1"

        # Unicode superscript digits
        superscripts = {
            "0": "⁰",
            "1": "¹",
            "2": "²",
            "3": "³",
            "4": "⁴",
            "5": "⁵",
            "6": "⁶",
            "7": "⁷",
            "8": "⁸",
            "9": "⁹",
            "-": "⁻",
            "/": "ᐟ",
        }

        def to_superscript(n: Fraction) -> str:
            if n == 1:
                return ""
            s = str(n)
            return "".join(superscripts.get(c, c) for c in s)

        parts = []
        for symbol, exp in zip(_DIMENSION_SYMBOLS, self._exponents):
            if exp != 0:
                parts.append(f"{symbol}{to_superscript(exp)}")

        return "".join(parts)


# Common dimensionless constant
DIMENSIONLESS = Dimension()

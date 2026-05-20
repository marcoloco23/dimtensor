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

from fractions import Fraction
from typing import Any, Tuple

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

    # The hash slot is precomputed at construction time. Hashing 7 Fractions
    # is ~14x more expensive than tuple hashing, and the hash is used on every
    # cached arithmetic op, so memoizing it pays back immediately.
    __slots__ = ("_exponents", "_hash")

    _exponents: Tuple[Fraction, ...]
    _hash: int

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
        object.__setattr__(self, "_hash", hash(exponents))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"Dimension is immutable; cannot set {name!r}")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"Dimension is immutable; cannot delete {name!r}")

    @classmethod
    def _from_exponents(cls, exponents: Tuple[Fraction, ...]) -> Dimension:
        """Create a dimension from a tuple of exponents (internal use)."""
        dim = object.__new__(cls)
        object.__setattr__(dim, "_exponents", exponents)
        object.__setattr__(dim, "_hash", hash(exponents))
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
            return NotImplemented
        # Key on the Dimension instances themselves so cache lookups use the
        # precomputed _hash instead of re-hashing 7 Fractions per side.
        cache = _DIM_MUL_CACHE
        key = (self, other)
        cached = cache.get(key)
        if cached is not None:
            return cached
        a = self._exponents
        b = other._exponents
        new_exponents = (
            a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3],
            a[4] + b[4], a[5] + b[5], a[6] + b[6],
        )
        result = Dimension._from_exponents(new_exponents)
        if len(cache) < _DIM_CACHE_MAXSIZE:
            cache[key] = result
        return result

    def __truediv__(self, other: object) -> Dimension:
        """Divide dimensions (subtract exponents)."""
        if not isinstance(other, Dimension):
            return NotImplemented
        cache = _DIM_DIV_CACHE
        key = (self, other)
        cached = cache.get(key)
        if cached is not None:
            return cached
        a = self._exponents
        b = other._exponents
        new_exponents = (
            a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3],
            a[4] - b[4], a[5] - b[5], a[6] - b[6],
        )
        result = Dimension._from_exponents(new_exponents)
        if len(cache) < _DIM_CACHE_MAXSIZE:
            cache[key] = result
        return result

    def __pow__(self, power: int | float | Fraction) -> Dimension:
        """Raise dimension to a power (multiply exponents)."""
        cache = _DIM_POW_CACHE
        key = (self, power)
        cached = cache.get(key)
        if cached is not None:
            return cached
        p = Fraction(power).limit_denominator(1000)
        a = self._exponents
        new_exponents = (
            a[0] * p, a[1] * p, a[2] * p, a[3] * p,
            a[4] * p, a[5] * p, a[6] * p,
        )
        result = Dimension._from_exponents(new_exponents)
        if len(cache) < _DIM_CACHE_MAXSIZE:
            cache[key] = result
        return result

    def __eq__(self, other: object) -> bool:
        """Check dimension equality."""
        if not isinstance(other, Dimension):
            return NotImplemented
        return self._exponents == other._exponents

    def __hash__(self) -> int:
        """Hash for use in sets and dicts (precomputed at construction)."""
        return self._hash

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

# Caches for dimension arithmetic. Dimensions are frozen and hashable, so the
# (exponents-tuple, exponents-tuple) key uniquely identifies an operation. In
# practice physics code reuses a small set of dimensions (m, s, m/s, kg, ...),
# so these caches make repeated operations effectively free. The max size guards
# against unbounded growth in pathological cases (e.g., fuzz tests).
_DIM_CACHE_MAXSIZE = 4096
_DIM_MUL_CACHE: dict[tuple[Any, ...], Dimension] = {}
_DIM_DIV_CACHE: dict[tuple[Any, ...], Dimension] = {}
_DIM_POW_CACHE: dict[tuple[Any, ...], Dimension] = {}

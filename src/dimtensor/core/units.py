"""Unit definitions and algebra.

Units combine a dimension with a scale factor relative to SI base units.
For example:
- meter has dimension L and scale 1.0
- kilometer has dimension L and scale 1000.0
- mile has dimension L and scale 1609.344
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .dimensions import Dimension, DIMENSIONLESS


@dataclass(frozen=True, slots=True)
class Unit:
    """Represents a physical unit.

    A unit combines a dimension with a scale factor. The scale factor
    represents how many SI base units equal one of this unit.

    Examples:
        >>> meter = Unit("m", Dimension(length=1), 1.0)
        >>> kilometer = Unit("km", Dimension(length=1), 1000.0)
        >>> # 1 km = 1000 m, so scale = 1000
    """

    symbol: str
    dimension: Dimension
    scale: float

    def __mul__(self, other: object) -> Unit:
        """Multiply units."""
        if isinstance(other, Unit):
            return Unit(
                symbol=f"{self.symbol}·{other.symbol}",
                dimension=self.dimension * other.dimension,
                scale=self.scale * other.scale,
            )
        elif isinstance(other, (int, float)):
            return Unit(
                symbol=self.symbol,
                dimension=self.dimension,
                scale=self.scale * other,
            )
        return NotImplemented

    def __rmul__(self, other: object) -> Unit:
        """Multiply by scalar from left."""
        if isinstance(other, (int, float)):
            return self * other
        return NotImplemented

    def __truediv__(self, other: object) -> Unit:
        """Divide units."""
        if isinstance(other, Unit):
            return Unit(
                symbol=f"{self.symbol}/{other.symbol}",
                dimension=self.dimension / other.dimension,
                scale=self.scale / other.scale,
            )
        elif isinstance(other, (int, float)):
            return Unit(
                symbol=self.symbol,
                dimension=self.dimension,
                scale=self.scale / other,
            )
        return NotImplemented

    def __rtruediv__(self, other: object) -> Unit:
        """Divide scalar by unit (gives inverse unit)."""
        if isinstance(other, (int, float)):
            return Unit(
                symbol=f"1/{self.symbol}",
                dimension=self.dimension ** -1,
                scale=other / self.scale,
            )
        return NotImplemented

    def __pow__(self, power: int | float | Fraction) -> Unit:
        """Raise unit to a power."""
        if power == 1:
            return self
        elif power == 2:
            new_symbol = f"{self.symbol}²"
        elif power == 3:
            new_symbol = f"{self.symbol}³"
        elif power == -1:
            new_symbol = f"1/{self.symbol}"
        elif power == -2:
            new_symbol = f"1/{self.symbol}²"
        else:
            new_symbol = f"{self.symbol}^{power}"

        return Unit(
            symbol=new_symbol,
            dimension=self.dimension ** power,
            scale=self.scale ** float(power),
        )

    def is_compatible(self, other: Unit) -> bool:
        """Check if two units have the same dimension."""
        return self.dimension == other.dimension

    def conversion_factor(self, other: Unit) -> float:
        """Get the factor to convert from this unit to another.

        Returns the value x such that: value_in_self * x = value_in_other

        Raises:
            ValueError: If units have different dimensions.
        """
        if not self.is_compatible(other):
            raise ValueError(
                f"Cannot convert between {self.symbol} and {other.symbol}: "
                f"dimensions {self.dimension} and {other.dimension} differ"
            )
        return self.scale / other.scale

    def __str__(self) -> str:
        """Return the unit symbol."""
        return self.symbol

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Unit({self.symbol!r}, {self.dimension!r}, {self.scale})"

    def simplified(self) -> Unit:
        """Return a unit with a simplified symbol.

        Attempts to:
        1. Match to a known SI derived unit (e.g., kg·m/s² → N)
        2. Return self if no simplification found
        """
        # Only simplify SI-scale units (scale=1.0)
        if abs(self.scale - 1.0) > 1e-10:
            return self

        # Look up in the registry
        simplified_symbol = _DIMENSION_TO_SYMBOL.get(self.dimension)
        if simplified_symbol:
            return Unit(simplified_symbol, self.dimension, self.scale)

        return self


# =============================================================================
# SI Base Units
# =============================================================================

meter = Unit("m", Dimension(length=1), 1.0)
kilogram = Unit("kg", Dimension(mass=1), 1.0)
second = Unit("s", Dimension(time=1), 1.0)
ampere = Unit("A", Dimension(current=1), 1.0)
kelvin = Unit("K", Dimension(temperature=1), 1.0)
mole = Unit("mol", Dimension(amount=1), 1.0)
candela = Unit("cd", Dimension(luminosity=1), 1.0)

# Dimensionless unit
dimensionless = Unit("1", DIMENSIONLESS, 1.0)

# Short aliases
m = meter
kg = kilogram
s = second
A = ampere
K = kelvin
mol = mole
cd = candela

# =============================================================================
# SI Derived Units
# =============================================================================

# Frequency
hertz = Unit("Hz", Dimension(time=-1), 1.0)
Hz = hertz

# Force
newton = Unit("N", Dimension(mass=1, length=1, time=-2), 1.0)
N = newton

# Energy
joule = Unit("J", Dimension(mass=1, length=2, time=-2), 1.0)
J = joule

# Power
watt = Unit("W", Dimension(mass=1, length=2, time=-3), 1.0)
W = watt

# Pressure
pascal = Unit("Pa", Dimension(mass=1, length=-1, time=-2), 1.0)
Pa = pascal

# Electric charge
coulomb = Unit("C", Dimension(current=1, time=1), 1.0)
C = coulomb

# Electric potential
volt = Unit("V", Dimension(mass=1, length=2, time=-3, current=-1), 1.0)
V = volt

# Electric resistance
ohm = Unit("Ω", Dimension(mass=1, length=2, time=-3, current=-2), 1.0)

# Electric capacitance
farad = Unit("F", Dimension(mass=-1, length=-2, time=4, current=2), 1.0)
F = farad

# Magnetic flux
weber = Unit("Wb", Dimension(mass=1, length=2, time=-2, current=-1), 1.0)
Wb = weber

# Magnetic flux density
tesla = Unit("T", Dimension(mass=1, time=-2, current=-1), 1.0)
T = tesla

# Inductance
henry = Unit("H", Dimension(mass=1, length=2, time=-2, current=-2), 1.0)
H = henry

# =============================================================================
# Common Non-SI Units
# =============================================================================

# Length
centimeter = Unit("cm", Dimension(length=1), 0.01)
millimeter = Unit("mm", Dimension(length=1), 0.001)
kilometer = Unit("km", Dimension(length=1), 1000.0)
inch = Unit("in", Dimension(length=1), 0.0254)
foot = Unit("ft", Dimension(length=1), 0.3048)
yard = Unit("yd", Dimension(length=1), 0.9144)
mile = Unit("mi", Dimension(length=1), 1609.344)

cm = centimeter
mm = millimeter
km = kilometer

# Mass
gram = Unit("g", Dimension(mass=1), 0.001)
milligram = Unit("mg", Dimension(mass=1), 1e-6)
tonne = Unit("t", Dimension(mass=1), 1000.0)
pound = Unit("lb", Dimension(mass=1), 0.45359237)
ounce = Unit("oz", Dimension(mass=1), 0.028349523125)

g = gram
mg = milligram

# Time
minute = Unit("min", Dimension(time=1), 60.0)
hour = Unit("h", Dimension(time=1), 3600.0)
day = Unit("d", Dimension(time=1), 86400.0)
millisecond = Unit("ms", Dimension(time=1), 0.001)
microsecond = Unit("μs", Dimension(time=1), 1e-6)
nanosecond = Unit("ns", Dimension(time=1), 1e-9)

ms = millisecond

# Angle (dimensionless, but useful)
radian = Unit("rad", DIMENSIONLESS, 1.0)
degree = Unit("°", DIMENSIONLESS, 0.017453292519943295)  # pi/180

rad = radian
deg = degree

# Temperature (special handling needed for offset conversions)
# Note: celsius and fahrenheit conversions require offset, not just scaling
# For now, we only support kelvin for strict dimensional analysis

# Energy
electronvolt = Unit("eV", Dimension(mass=1, length=2, time=-2), 1.602176634e-19)
calorie = Unit("cal", Dimension(mass=1, length=2, time=-2), 4.184)
kilocalorie = Unit("kcal", Dimension(mass=1, length=2, time=-2), 4184.0)

eV = electronvolt

# Pressure
bar = Unit("bar", Dimension(mass=1, length=-1, time=-2), 1e5)
atmosphere = Unit("atm", Dimension(mass=1, length=-1, time=-2), 101325.0)
torr = Unit("Torr", Dimension(mass=1, length=-1, time=-2), 133.322)
psi = Unit("psi", Dimension(mass=1, length=-1, time=-2), 6894.757)

atm = atmosphere

# Speed
speed_of_light = Unit("c", Dimension(length=1, time=-1), 299792458.0)
c = speed_of_light

# =============================================================================
# Dimension to Symbol Registry (for unit simplification)
# =============================================================================

_DIMENSION_TO_SYMBOL: dict[Dimension, str] = {
    # Dimensionless
    DIMENSIONLESS: "1",
    # Base dimensions
    Dimension(length=1): "m",
    Dimension(mass=1): "kg",
    Dimension(time=1): "s",
    Dimension(current=1): "A",
    Dimension(temperature=1): "K",
    Dimension(amount=1): "mol",
    Dimension(luminosity=1): "cd",
    # Derived SI units
    Dimension(time=-1): "Hz",  # frequency
    Dimension(mass=1, length=1, time=-2): "N",  # force
    Dimension(mass=1, length=2, time=-2): "J",  # energy
    Dimension(mass=1, length=2, time=-3): "W",  # power
    Dimension(mass=1, length=-1, time=-2): "Pa",  # pressure
    Dimension(current=1, time=1): "C",  # charge
    Dimension(mass=1, length=2, time=-3, current=-1): "V",  # voltage
    Dimension(mass=1, length=2, time=-3, current=-2): "Ω",  # resistance
    Dimension(mass=-1, length=-2, time=4, current=2): "F",  # capacitance
    Dimension(mass=1, length=2, time=-2, current=-1): "Wb",  # magnetic flux
    Dimension(mass=1, time=-2, current=-1): "T",  # magnetic flux density
    Dimension(mass=1, length=2, time=-2, current=-2): "H",  # inductance
    # Common derived quantities
    Dimension(length=1, time=-1): "m/s",  # velocity
    Dimension(length=1, time=-2): "m/s²",  # acceleration
    Dimension(length=2): "m²",  # area
    Dimension(length=3): "m³",  # volume
    Dimension(mass=1, length=-3): "kg/m³",  # density
}

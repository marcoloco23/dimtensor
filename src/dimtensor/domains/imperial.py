"""Imperial and US customary units.

This module provides Imperial and US customary units commonly used in the
United States and historically in the British Commonwealth, including length,
mass, volume, area, temperature, force, energy, pressure, and speed units.

Example:
    >>> from dimtensor import DimArray
    >>> from dimtensor.domains.imperial import foot, pound, gallon
    >>> height = DimArray([6.0], foot)  # 6 feet
    >>> weight = DimArray([150], pound)  # 150 pounds
    >>> volume = DimArray([5.0], gallon)  # 5 US gallons

Note on Fahrenheit:
    Fahrenheit is an offset scale (F = 9/5*C + 32). This module provides the
    degree Fahrenheit as a temperature INTERVAL unit (same as Rankine degree).
    For absolute temperature conversions with offset handling, use dedicated
    temperature conversion functions.

Reference values from NIST Special Publication 811 (2008).
"""

from __future__ import annotations

from ..core.dimensions import Dimension, DIMENSIONLESS
from ..core.units import Unit


# =============================================================================
# Length Units
# =============================================================================

# Inch (defined as exactly 25.4 mm since 1959)
inch = Unit("in", Dimension(length=1), 0.0254)
In = inch  # Alternative symbol

# Foot (12 inches)
foot = Unit("ft", Dimension(length=1), 0.3048)
ft = foot

# Yard (3 feet)
yard = Unit("yd", Dimension(length=1), 0.9144)
yd = yard

# Mile (5280 feet)
mile = Unit("mi", Dimension(length=1), 1609.344)
mi = mile

# Nautical mile (exactly 1852 m by international definition)
nautical_mile = Unit("nmi", Dimension(length=1), 1852.0)
nmi = nautical_mile

# Thou/Mil (thousandth of an inch)
thou = Unit("thou", Dimension(length=1), 2.54e-5)
mil = Unit("mil", Dimension(length=1), 2.54e-5)

# Furlong (660 feet, 1/8 mile)
furlong = Unit("fur", Dimension(length=1), 201.168)
fur = furlong


# =============================================================================
# Mass Units
# =============================================================================

# Grain (exactly 64.79891 mg, base for avoirdupois and troy systems)
grain = Unit("gr", Dimension(mass=1), 6.479891e-5)
gr = grain

# Dram (1/16 ounce avoirdupois)
dram = Unit("dr", Dimension(mass=1), 1.7718451953125e-3)
dr = dram

# Ounce (avoirdupois, 1/16 pound)
ounce = Unit("oz", Dimension(mass=1), 0.028349523125)
oz = ounce

# Pound (avoirdupois, exactly 0.45359237 kg since 1959)
pound = Unit("lb", Dimension(mass=1), 0.45359237)
lb = pound

# Stone (14 pounds, common in UK)
stone = Unit("st", Dimension(mass=1), 6.35029318)
st = stone

# Short ton (US ton, 2000 pounds)
ton = Unit("ton", Dimension(mass=1), 907.18474)
short_ton = ton

# Long ton (Imperial ton, 2240 pounds)
long_ton = Unit("long_ton", Dimension(mass=1), 1016.0469088)


# =============================================================================
# Volume Units (US Customary)
# =============================================================================

# Teaspoon (US, 1/6 fluid ounce)
teaspoon = Unit("tsp", Dimension(length=3), 4.92892159375e-6)
tsp = teaspoon

# Tablespoon (US, 1/2 fluid ounce, 3 teaspoons)
tablespoon = Unit("tbsp", Dimension(length=3), 1.478676478125e-5)
tbsp = tablespoon

# Fluid ounce (US, 1/128 gallon)
fluid_ounce = Unit("fl_oz", Dimension(length=3), 2.95735295625e-5)
fl_oz = fluid_ounce

# Cup (US, 8 fluid ounces)
cup = Unit("cup", Dimension(length=3), 2.365882365e-4)

# Pint (US liquid, 16 fluid ounces)
pint = Unit("pt", Dimension(length=3), 4.73176473e-4)
pt = pint

# Quart (US liquid, 2 pints)
quart = Unit("qt", Dimension(length=3), 9.46352946e-4)
qt = quart

# Gallon (US liquid, 231 cubic inches, exactly 3.785411784 L)
gallon = Unit("gal", Dimension(length=3), 3.785411784e-3)
gal = gallon

# Barrel (oil, 42 US gallons)
barrel = Unit("bbl", Dimension(length=3), 0.158987294928)
bbl = barrel


# =============================================================================
# Area Units
# =============================================================================

# Square foot
square_foot = Unit("ft²", Dimension(length=2), 0.09290304)
sq_ft = square_foot

# Square yard
square_yard = Unit("yd²", Dimension(length=2), 0.83612736)
sq_yd = square_yard

# Acre (43,560 square feet)
acre = Unit("ac", Dimension(length=2), 4046.8564224)
ac = acre

# Square mile (640 acres)
square_mile = Unit("mi²", Dimension(length=2), 2589988.110336)
sq_mi = square_mile


# =============================================================================
# Temperature Units
# =============================================================================

# Rankine (absolute scale, 0 R = absolute zero)
# Temperature interval: 1 R = 5/9 K
rankine = Unit("R", Dimension(temperature=1), 5.0/9.0)
R = rankine

# Fahrenheit degree (temperature INTERVAL, not absolute)
# For intervals: ΔT(F) = 9/5 * ΔT(C)
# Note: This is the temperature interval unit, not the offset scale
# 1 degree F interval = 5/9 K interval
fahrenheit = Unit("°F", Dimension(temperature=1), 5.0/9.0)
degF = fahrenheit


# =============================================================================
# Force Units
# =============================================================================

# Pound-force (force exerted by 1 lb mass under standard gravity)
# 1 lbf = 4.4482216152605 N
pound_force = Unit("lbf", Dimension(mass=1, length=1, time=-2), 4.4482216152605)
lbf = pound_force

# Poundal (force to accelerate 1 lb mass at 1 ft/s²)
# 1 pdl = 0.138254954376 N
poundal = Unit("pdl", Dimension(mass=1, length=1, time=-2), 0.138254954376)
pdl = poundal


# =============================================================================
# Energy Units
# =============================================================================

# British Thermal Unit (International Table definition)
# 1 BTU = 1055.05585262 J
BTU = Unit("BTU", Dimension(mass=1, length=2, time=-2), 1055.05585262)
btu = BTU

# Therm (100,000 BTU, common for natural gas)
# 1 therm = 105,505,585.262 J
therm = Unit("therm", Dimension(mass=1, length=2, time=-2), 1.0550558526e8)

# Foot-pound (energy)
# 1 ft·lb = 1.3558179483314004 J
foot_pound = Unit("ft·lb", Dimension(mass=1, length=2, time=-2), 1.3558179483314004)
ft_lb = foot_pound


# =============================================================================
# Pressure Units
# =============================================================================

# Pound per square inch
# 1 psi = 6894.757293168 Pa
psi = Unit("psi", Dimension(mass=1, length=-1, time=-2), 6894.757293168)

# Inches of mercury (conventional, at 0°C)
# 1 inHg = 3386.389 Pa
inches_of_mercury = Unit("inHg", Dimension(mass=1, length=-1, time=-2), 3386.389)
inHg = inches_of_mercury

# Inches of water (conventional, at 4°C)
# 1 inH2O = 249.082 Pa
inches_of_water = Unit("inH2O", Dimension(mass=1, length=-1, time=-2), 249.082)
inH2O = inches_of_water


# =============================================================================
# Speed/Velocity Units
# =============================================================================

# Mile per hour
# 1 mph = 0.44704 m/s
mile_per_hour = Unit("mph", Dimension(length=1, time=-1), 0.44704)
mph = mile_per_hour

# Knot (nautical mile per hour)
# 1 knot = 0.514444444... m/s
knot = Unit("knot", Dimension(length=1, time=-1), 0.514444444444444)
kt = knot


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Length
    "inch", "In",
    "foot", "ft",
    "yard", "yd",
    "mile", "mi",
    "nautical_mile", "nmi",
    "thou", "mil",
    "furlong", "fur",
    # Mass
    "grain", "gr",
    "dram", "dr",
    "ounce", "oz",
    "pound", "lb",
    "stone", "st",
    "ton", "short_ton",
    "long_ton",
    # Volume
    "teaspoon", "tsp",
    "tablespoon", "tbsp",
    "fluid_ounce", "fl_oz",
    "cup",
    "pint", "pt",
    "quart", "qt",
    "gallon", "gal",
    "barrel", "bbl",
    # Area
    "square_foot", "sq_ft",
    "square_yard", "sq_yd",
    "acre", "ac",
    "square_mile", "sq_mi",
    # Temperature
    "rankine", "R",
    "fahrenheit", "degF",
    # Force
    "pound_force", "lbf",
    "poundal", "pdl",
    # Energy
    "BTU", "btu",
    "therm",
    "foot_pound", "ft_lb",
    # Pressure
    "psi",
    "inches_of_mercury", "inHg",
    "inches_of_water", "inH2O",
    # Speed
    "mile_per_hour", "mph",
    "knot", "kt",
]

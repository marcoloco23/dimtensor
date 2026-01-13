"""Text formatters for accessibility.

Provides specialized formatting for screen readers and high-contrast display modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..core.dimarray import DimArray
    from ..core.units import Unit


# ==============================================================================
# Unit Name Expansion
# ==============================================================================

# Map unit symbols to screen reader-friendly names
UNIT_NAMES = {
    # Base SI units
    "m": "meters",
    "kg": "kilograms",
    "s": "seconds",
    "A": "amperes",
    "K": "kelvin",
    "mol": "moles",
    "cd": "candelas",
    # Common derived units
    "Hz": "hertz",
    "N": "newtons",
    "Pa": "pascals",
    "J": "joules",
    "W": "watts",
    "C": "coulombs",
    "V": "volts",
    "F": "farads",
    "Ω": "ohms",
    "S": "siemens",
    "Wb": "webers",
    "T": "teslas",
    "H": "henrys",
    "lm": "lumens",
    "lx": "lux",
    "Bq": "becquerels",
    "Gy": "grays",
    "Sv": "sieverts",
    "kat": "katals",
    # Common multiples
    "km": "kilometers",
    "cm": "centimeters",
    "mm": "millimeters",
    "μm": "micrometers",
    "nm": "nanometers",
    "g": "grams",
    "mg": "milligrams",
    "μg": "micrograms",
    "ms": "milliseconds",
    "μs": "microseconds",
    "ns": "nanoseconds",
    "mA": "milliamperes",
    "μA": "microamperes",
    "kV": "kilovolts",
    "mV": "millivolts",
    "kW": "kilowatts",
    "MW": "megawatts",
    "GW": "gigawatts",
    "kJ": "kilojoules",
    "MJ": "megajoules",
    "GJ": "gigajoules",
    # Time units
    "min": "minutes",
    "h": "hours",
    "hr": "hours",
    "d": "days",
    "yr": "years",
    "year": "years",
    # Length units
    "mi": "miles",
    "ft": "feet",
    "in": "inches",
    "yd": "yards",
    # Speed units
    "mph": "miles per hour",
    "kph": "kilometers per hour",
    # Angle units
    "rad": "radians",
    "deg": "degrees",
    "°": "degrees",
    # Temperature
    "°C": "degrees Celsius",
    "°F": "degrees Fahrenheit",
    # Dimensionless
    "": "dimensionless",
    "1": "dimensionless",
}


def expand_unit_name(unit_symbol: str) -> str:
    """Expand a unit symbol to its full name for screen readers.

    Args:
        unit_symbol: Unit symbol like 'm', 'kg', 'N'.

    Returns:
        Full unit name like 'meters', 'kilograms', 'newtons'.

    Examples:
        >>> expand_unit_name('m')
        'meters'
        >>> expand_unit_name('kg')
        'kilograms'
        >>> expand_unit_name('N')
        'newtons'
    """
    # Try direct lookup first
    if unit_symbol in UNIT_NAMES:
        return UNIT_NAMES[unit_symbol]

    # Handle compound units like 'm/s' or 'm·s^-1'
    # For screen readers, we'll keep the symbol but add spacing
    # A more sophisticated parser could be added in the future
    return unit_symbol


def format_exponent(exponent: str | int) -> str:
    """Format an exponent for screen reader output.

    Args:
        exponent: Exponent value (e.g., '2', '-1', '3').

    Returns:
        Screen reader-friendly text.

    Examples:
        >>> format_exponent('2')
        'squared'
        >>> format_exponent('3')
        'cubed'
        >>> format_exponent('-1')
        'to the negative first power'
    """
    exp = str(exponent)

    if exp == "1":
        return ""
    elif exp == "2":
        return "squared"
    elif exp == "3":
        return "cubed"
    elif exp.startswith("-"):
        return f"to the negative {exp[1:]} power"
    else:
        return f"to the {exp} power"


# ==============================================================================
# Screen Reader Formatter
# ==============================================================================


class ScreenReaderFormatter:
    """Formatter for screen reader-friendly output.

    Converts scientific notation and unit symbols to spoken text.
    """

    def __init__(self, verbosity: str = "normal"):
        """Initialize the formatter.

        Args:
            verbosity: Verbosity level ('brief', 'normal', 'verbose').
        """
        self.verbosity = verbosity

    def format_value(self, value: float, precision: int = 4) -> str:
        """Format a single numerical value.

        Args:
            value: The numerical value.
            precision: Number of decimal places.

        Returns:
            Screen reader-friendly text.

        Examples:
            >>> formatter = ScreenReaderFormatter()
            >>> formatter.format_value(3.14159)
            '3.1416'
            >>> formatter.format_value(1.23e-6)
            '1.23 times 10 to the negative 6 power'
        """
        # Handle special values
        if np.isnan(value):
            return "not a number"
        if np.isinf(value):
            return "positive infinity" if value > 0 else "negative infinity"

        # For very small or very large numbers, use scientific notation
        abs_val = abs(value)
        if abs_val != 0 and (abs_val < 0.001 or abs_val >= 10000):
            # Scientific notation
            mantissa, exponent = f"{value:.{precision}e}".split("e")
            exp_int = int(exponent)
            return f"{mantissa} times 10 {format_exponent(exp_int)}"
        else:
            # Regular notation
            return f"{value:.{precision}g}"

    def format_dimarray(self, arr: DimArray, max_values: int = 5) -> str:
        """Format a DimArray for screen reader output.

        Args:
            arr: The DimArray to format.
            max_values: Maximum number of values to read (prevents overwhelming output).

        Returns:
            Screen reader-friendly text.

        Examples:
            >>> arr = DimArray([3.14], units.m)
            >>> formatter = ScreenReaderFormatter()
            >>> formatter.format_dimarray(arr)
            '3.14 meters'
        """
        from ..core.dimarray import DimArray

        # Get unit name
        unit_name = expand_unit_name(arr.unit.symbol)

        # Format values
        data = arr._data.flatten()
        n_values = len(data)

        if n_values == 0:
            return f"Empty array with unit {unit_name}"
        elif n_values == 1:
            # Single value
            value_str = self.format_value(float(data[0]))
            if arr.has_uncertainty:
                unc_str = self.format_value(float(arr.uncertainty.flatten()[0]))
                return f"{value_str} plus or minus {unc_str} {unit_name}"
            else:
                return f"{value_str} {unit_name}"
        else:
            # Multiple values
            if n_values <= max_values:
                # Read all values
                values_str = ", ".join(self.format_value(float(v)) for v in data)
            else:
                # Read first few and last few
                n_show = max_values // 2
                first_values = ", ".join(
                    self.format_value(float(v)) for v in data[:n_show]
                )
                last_values = ", ".join(
                    self.format_value(float(v)) for v in data[-n_show:]
                )
                values_str = f"{first_values}, ... {n_values - 2*n_show} more values ..., {last_values}"

            shape_str = f"Array of shape {arr.shape}"
            return f"{shape_str}: {values_str} {unit_name}"

    def format_with_uncertainty(
        self, value: float, uncertainty: float, unit_symbol: str
    ) -> str:
        """Format a value with uncertainty.

        Args:
            value: The central value.
            uncertainty: The uncertainty.
            unit_symbol: Unit symbol.

        Returns:
            Screen reader-friendly text.

        Examples:
            >>> formatter = ScreenReaderFormatter()
            >>> formatter.format_with_uncertainty(3.14, 0.1, 'm')
            '3.14 plus or minus 0.1 meters'
        """
        value_str = self.format_value(value)
        unc_str = self.format_value(uncertainty)
        unit_name = expand_unit_name(unit_symbol)
        return f"{value_str} plus or minus {unc_str} {unit_name}"


# ==============================================================================
# High Contrast Formatter
# ==============================================================================


class HighContrastFormatter:
    """Formatter for high-contrast text output.

    Uses ASCII art and extra spacing for improved readability.
    """

    def __init__(self, use_box_chars: bool = True):
        """Initialize the formatter.

        Args:
            use_box_chars: If True, use Unicode box-drawing characters.
        """
        self.use_box_chars = use_box_chars

    def format_dimarray(self, arr: DimArray) -> str:
        """Format a DimArray with high contrast.

        Args:
            arr: The DimArray to format.

        Returns:
            High-contrast formatted text.

        Examples:
            >>> arr = DimArray([1.0, 2.0, 3.0], units.m)
            >>> formatter = HighContrastFormatter()
            >>> print(formatter.format_dimarray(arr))
            ┌──────────────────────┐
            │  [ 1.0  2.0  3.0 ]   │
            │  Unit: m             │
            └──────────────────────┘
        """
        from ..core.dimarray import DimArray

        # Format the array data
        data_str = np.array2string(
            arr._data,
            precision=4,
            separator="  ",
            suppress_small=True,
        )

        # Format unit
        unit_str = f"Unit: {arr.unit.symbol}"

        # Add uncertainty if present
        if arr.has_uncertainty:
            unc_str = np.array2string(
                arr.uncertainty,
                precision=4,
                separator="  ",
                suppress_small=True,
            )
            unc_line = f"Uncertainty: {unc_str}"
        else:
            unc_line = None

        # Calculate box width
        lines = [data_str, unit_str]
        if unc_line:
            lines.append(unc_line)
        max_width = max(len(line) for line in lines) + 4

        # Build box
        if self.use_box_chars:
            top = "┌" + "─" * (max_width - 2) + "┐"
            bottom = "└" + "─" * (max_width - 2) + "┘"
            side = "│"
        else:
            top = "+" + "-" * (max_width - 2) + "+"
            bottom = "+" + "-" * (max_width - 2) + "+"
            side = "|"

        # Format lines with padding
        result_lines = [top]
        for line in lines:
            padded_line = side + "  " + line.ljust(max_width - 4) + side
            result_lines.append(padded_line)
        result_lines.append(bottom)

        return "\n".join(result_lines)

    def format_section(self, title: str, content: str) -> str:
        """Format a section with a title and content.

        Args:
            title: Section title.
            content: Section content.

        Returns:
            Formatted section.
        """
        if self.use_box_chars:
            separator = "═" * len(title)
        else:
            separator = "=" * len(title)

        return f"\n{title}\n{separator}\n{content}\n"


# ==============================================================================
# Utility Functions
# ==============================================================================


def format_for_screen_reader(arr: Any, **kwargs: Any) -> str:
    """Convenience function to format a DimArray for screen readers.

    Args:
        arr: The DimArray to format.
        **kwargs: Additional arguments passed to ScreenReaderFormatter.

    Returns:
        Screen reader-friendly text.
    """
    formatter = ScreenReaderFormatter(**kwargs)
    return formatter.format_dimarray(arr)


def format_high_contrast(arr: Any, **kwargs: Any) -> str:
    """Convenience function to format a DimArray with high contrast.

    Args:
        arr: The DimArray to format.
        **kwargs: Additional arguments passed to HighContrastFormatter.

    Returns:
        High-contrast formatted text.
    """
    formatter = HighContrastFormatter(**kwargs)
    return formatter.format_dimarray(arr)

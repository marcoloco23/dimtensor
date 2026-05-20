"""Base class for physical constants.

A physical constant combines a numerical value with a unit and optional
uncertainty. Constants support arithmetic operations with other constants
and with DimArrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..core.units import Unit

if TYPE_CHECKING:
    from ..core.dimarray import DimArray
    from ..core.dimensions import Dimension


# Cached DimArray class reference. Populated lazily on first call to avoid a
# potential circular dependency at module-init time (core.dimarray references
# Constant inside its arithmetic methods). Once cached, the 8 call sites
# below avoid an import lookup per invocation.
_DimArray: type[DimArray] | None = None


def _get_dimarray_cls() -> type[DimArray]:
    global _DimArray
    if _DimArray is None:
        from ..core.dimarray import DimArray as _D
        _DimArray = _D
    return _DimArray


@dataclass(frozen=True, slots=True)
class Constant:
    """A physical constant with value, unit, and uncertainty.

    Physical constants are scalar values with associated physical units
    and measurement uncertainties from CODATA. They can be used directly
    in calculations with DimArrays.

    Attributes:
        symbol: Short symbol (e.g., "c", "G", "h").
        name: Full name (e.g., "speed of light in vacuum").
        value: Numerical value in SI units.
        unit: Physical unit.
        uncertainty: Absolute standard uncertainty (0.0 for exact constants).

    Examples:
        >>> from dimtensor.constants import c, G
        >>> print(c)
        c = 299792458 m/s (exact)
        >>> print(c.value)
        299792458.0
        >>> print(c.is_exact)
        True
        >>> print(G.uncertainty)
        1.5e-15
    """

    symbol: str
    name: str
    value: float
    unit: Unit
    uncertainty: float = 0.0

    @property
    def is_exact(self) -> bool:
        """True if this constant has zero uncertainty (exact by definition)."""
        return self.uncertainty == 0.0

    @property
    def relative_uncertainty(self) -> float:
        """Relative standard uncertainty (dimensionless fraction).

        Returns 0.0 if the value is zero or if the constant is exact.
        """
        if self.value == 0 or self.uncertainty == 0:
            return 0.0
        return abs(self.uncertainty / self.value)

    @property
    def dimension(self) -> Dimension:
        """Physical dimension of this constant."""
        return self.unit.dimension

    def to_dimarray(self) -> DimArray:
        """Convert to a scalar DimArray with uncertainty.

        Returns:
            A DimArray containing the constant's value with its unit and uncertainty.
        """
        DimArray = _get_dimarray_cls()

        uncertainty = None
        if self.uncertainty > 0:
            uncertainty = np.array([self.uncertainty])

        return DimArray._from_data_and_unit(
            np.array([self.value]), self.unit, uncertainty
        )

    # =========================================================================
    # Arithmetic operations
    # =========================================================================

    def __mul__(self, other: object) -> DimArray:
        """Multiply constant by another value.

        Args:
            other: Constant, DimArray, or scalar.

        Returns:
            DimArray with the result, with propagated uncertainty.
        """
        DimArray = _get_dimarray_cls()

        if isinstance(other, Constant):
            new_unit = self.unit * other.unit
            new_value = self.value * other.value
            # Propagate uncertainty: sigma_z/|z| = sqrt((sigma_a/a)^2 + (sigma_b/b)^2)
            new_uncertainty = None
            if self.uncertainty > 0 or other.uncertainty > 0:
                rel_a_sq = (self.uncertainty / self.value) ** 2 if self.value != 0 else 0
                rel_b_sq = (other.uncertainty / other.value) ** 2 if other.value != 0 else 0
                rel_combined = np.sqrt(rel_a_sq + rel_b_sq)
                new_uncertainty = np.array([abs(new_value) * rel_combined])
            return DimArray._from_data_and_unit(
                np.array([new_value]), new_unit.simplified(), new_uncertainty
            )
        elif isinstance(other, DimArray):
            return self.to_dimarray() * other
        elif isinstance(other, int | float):
            # Scalar multiplication: uncertainty scales with scalar
            new_uncertainty = None
            if self.uncertainty > 0:
                new_uncertainty = np.array([abs(other) * self.uncertainty])
            return DimArray._from_data_and_unit(
                np.array([self.value * other]), self.unit, new_uncertainty
            )
        return NotImplemented

    def __rmul__(self, other: object) -> DimArray:
        """Right multiply (scalar * Constant or DimArray * Constant)."""
        DimArray = _get_dimarray_cls()

        if isinstance(other, DimArray):
            return other * self.to_dimarray()
        elif isinstance(other, int | float):
            return self.__mul__(other)
        return NotImplemented

    def __truediv__(self, other: object) -> DimArray:
        """Divide constant by another value.

        Args:
            other: Constant, DimArray, or scalar.

        Returns:
            DimArray with the result, with propagated uncertainty.
        """
        DimArray = _get_dimarray_cls()

        if isinstance(other, Constant):
            new_unit = self.unit / other.unit
            new_value = self.value / other.value
            # Propagate uncertainty: sigma_z/|z| = sqrt((sigma_a/a)^2 + (sigma_b/b)^2)
            new_uncertainty = None
            if self.uncertainty > 0 or other.uncertainty > 0:
                rel_a_sq = (self.uncertainty / self.value) ** 2 if self.value != 0 else 0
                rel_b_sq = (other.uncertainty / other.value) ** 2 if other.value != 0 else 0
                rel_combined = np.sqrt(rel_a_sq + rel_b_sq)
                new_uncertainty = np.array([abs(new_value) * rel_combined])
            return DimArray._from_data_and_unit(
                np.array([new_value]), new_unit.simplified(), new_uncertainty
            )
        elif isinstance(other, DimArray):
            return self.to_dimarray() / other
        elif isinstance(other, int | float):
            # Scalar division: uncertainty scales inversely
            new_uncertainty = None
            if self.uncertainty > 0:
                new_uncertainty = np.array([self.uncertainty / abs(other)])
            return DimArray._from_data_and_unit(
                np.array([self.value / other]), self.unit, new_uncertainty
            )
        return NotImplemented

    def __rtruediv__(self, other: object) -> DimArray:
        """Right divide (scalar / Constant or DimArray / Constant)."""
        DimArray = _get_dimarray_cls()

        if isinstance(other, DimArray):
            return other / self.to_dimarray()
        elif isinstance(other, int | float):
            new_unit = Unit(
                f"1/{self.unit.symbol}",
                self.unit.dimension ** -1,
                1.0 / self.unit.scale,
            )
            new_value = other / self.value
            # For scalar/constant: sigma_z/|z| = sigma_x/|x|
            new_uncertainty = None
            if self.uncertainty > 0 and self.value != 0:
                rel_unc = self.uncertainty / abs(self.value)
                new_uncertainty = np.array([abs(new_value) * rel_unc])
            return DimArray._from_data_and_unit(
                np.array([new_value]), new_unit.simplified(), new_uncertainty
            )
        return NotImplemented

    def __pow__(self, power: int | float) -> DimArray:
        """Raise constant to a power.

        Args:
            power: Exponent (int or float).

        Returns:
            DimArray with the result, with propagated uncertainty.
        """
        DimArray = _get_dimarray_cls()

        new_unit = self.unit ** power
        new_value = self.value ** power
        # Propagate uncertainty: sigma_z/|z| = |n| * sigma_x/|x|
        new_uncertainty = None
        if self.uncertainty > 0 and self.value != 0:
            rel_unc = self.uncertainty / abs(self.value)
            new_uncertainty = np.array([abs(new_value) * abs(power) * rel_unc])
        return DimArray._from_data_and_unit(
            np.array([new_value]), new_unit.simplified(), new_uncertainty
        )

    def __neg__(self) -> DimArray:
        """Negate the constant."""
        DimArray = _get_dimarray_cls()

        uncertainty = None
        if self.uncertainty > 0:
            uncertainty = np.array([self.uncertainty])
        return DimArray._from_data_and_unit(
            np.array([-self.value]), self.unit, uncertainty
        )

    def __pos__(self) -> DimArray:
        """Unary positive (returns DimArray copy)."""
        return self.to_dimarray()

    # =========================================================================
    # String representations
    # =========================================================================

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"Constant({self.symbol!r}, {self.name!r}, "
            f"value={self.value}, unit={self.unit.symbol!r}, "
            f"uncertainty={self.uncertainty})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        unit_str = self.unit.simplified().symbol
        if self.is_exact:
            return f"{self.symbol} = {self.value} {unit_str} (exact)"
        else:
            return f"{self.symbol} = {self.value} +/- {self.uncertainty} {unit_str}"

    def __format__(self, format_spec: str) -> str:
        """Support format strings like f'{c:.2e}'.

        The format spec is applied to the numerical value.
        """
        value_str = format(self.value, format_spec)
        unit_str = self.unit.simplified().symbol
        return f"{value_str} {unit_str}"

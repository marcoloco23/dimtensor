"""UnitSchema class for packaging custom units and constants.

A schema packages related units, constants, and custom dimensions into
a shareable, versionable format with metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from ..constants._base import Constant
from ..core.dimensions import Dimension
from ..core.units import Unit


@dataclass
class UnitDefinition:
    """Definition of a unit in a schema.

    Attributes:
        symbol: Unit symbol (e.g., "MeV", "AU").
        dimension: Dimension exponents as dict (e.g., {"M": 1, "L": 2, "T": -2}).
        scale: Scale factor in SI base units.
        description: Human-readable description.
    """

    symbol: str
    dimension: dict[str, int | float]
    scale: float
    description: str = ""

    def to_unit(self) -> Unit:
        """Convert to a Unit object."""
        return Unit(
            symbol=self.symbol,
            dimension=Dimension(
                length=self.dimension.get("L", 0),
                mass=self.dimension.get("M", 0),
                time=self.dimension.get("T", 0),
                current=self.dimension.get("I", 0),
                temperature=self.dimension.get("Theta", 0),
                amount=self.dimension.get("N", 0),
                luminosity=self.dimension.get("J", 0),
            ),
            scale=self.scale,
        )

    @classmethod
    def from_unit(cls, unit: Unit, description: str = "") -> UnitDefinition:
        """Create from a Unit object."""
        dim_dict = {}
        names = ["L", "M", "T", "I", "Theta", "N", "J"]
        for i, name in enumerate(names):
            exp = float(unit.dimension._exponents[i])
            if exp != 0:
                dim_dict[name] = exp

        return cls(
            symbol=unit.symbol,
            dimension=dim_dict,
            scale=unit.scale,
            description=description,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "symbol": self.symbol,
            "dimension": self.dimension,
            "scale": self.scale,
        }
        if self.description:
            result["description"] = self.description
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnitDefinition:
        """Create from dictionary."""
        return cls(
            symbol=data["symbol"],
            dimension=data["dimension"],
            scale=data["scale"],
            description=data.get("description", ""),
        )


@dataclass
class ConstantDefinition:
    """Definition of a physical constant in a schema.

    Attributes:
        symbol: Constant symbol (e.g., "c", "G").
        name: Full name of the constant.
        value: Numerical value in SI units.
        unit: Unit symbol (e.g., "m/s").
        uncertainty: Absolute standard uncertainty (0.0 for exact).
        description: Human-readable description.
    """

    symbol: str
    name: str
    value: float
    unit: str
    uncertainty: float = 0.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "symbol": self.symbol,
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
        }
        if self.uncertainty != 0.0:
            result["uncertainty"] = self.uncertainty
        if self.description:
            result["description"] = self.description
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConstantDefinition:
        """Create from dictionary."""
        return cls(
            symbol=data["symbol"],
            name=data["name"],
            value=data["value"],
            unit=data["unit"],
            uncertainty=data.get("uncertainty", 0.0),
            description=data.get("description", ""),
        )


@dataclass
class UnitSchema:
    """A collection of unit and constant definitions with metadata.

    Schemas provide a way to package and share custom units, constants,
    and dimensions. They support versioning, dependencies, and metadata.

    Attributes:
        name: Unique schema identifier (e.g., "nuclear_physics").
        version: Semantic version string (e.g., "1.0.0").
        description: Human-readable description.
        author: Author name or organization.
        license: License identifier (e.g., "MIT", "CC-BY-4.0").
        dependencies: List of schema dependencies with versions.
        units: List of unit definitions.
        constants: List of constant definitions.
        schema_version: Schema format version (for future compatibility).

    Examples:
        >>> schema = UnitSchema(
        ...     name="nuclear_physics",
        ...     version="1.0.0",
        ...     description="Units for nuclear physics"
        ... )
        >>> schema.add_unit("MeV", {"M": 1, "L": 2, "T": -2}, 1.602176634e-13)
        >>> schema.add_constant("m_proton", "proton mass", 1.67262e-27, "kg")
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    license: str = "MIT"
    dependencies: list[str] = field(default_factory=list)
    units: list[UnitDefinition] = field(default_factory=list)
    constants: list[ConstantDefinition] = field(default_factory=list)
    schema_version: str = "1.0.0"

    def add_unit(
        self,
        symbol: str,
        dimension: dict[str, int | float],
        scale: float,
        description: str = "",
    ) -> None:
        """Add a unit definition to this schema.

        Args:
            symbol: Unit symbol.
            dimension: Dimension exponents (e.g., {"L": 1} for length).
            scale: Scale factor in SI base units.
            description: Optional description.
        """
        self.units.append(
            UnitDefinition(
                symbol=symbol,
                dimension=dimension,
                scale=scale,
                description=description,
            )
        )

    def add_constant(
        self,
        symbol: str,
        name: str,
        value: float,
        unit: str,
        uncertainty: float = 0.0,
        description: str = "",
    ) -> None:
        """Add a constant definition to this schema.

        Args:
            symbol: Constant symbol.
            name: Full name of the constant.
            value: Numerical value.
            unit: Unit symbol (must be defined in schema or SI).
            uncertainty: Absolute uncertainty (default 0.0).
            description: Optional description.
        """
        self.constants.append(
            ConstantDefinition(
                symbol=symbol,
                name=name,
                value=value,
                unit=unit,
                uncertainty=uncertainty,
                description=description,
            )
        )

    def get_unit(self, symbol: str) -> UnitDefinition | None:
        """Get a unit definition by symbol.

        Args:
            symbol: Unit symbol to find.

        Returns:
            UnitDefinition if found, None otherwise.
        """
        for unit in self.units:
            if unit.symbol == symbol:
                return unit
        return None

    def get_constant(self, symbol: str) -> ConstantDefinition | None:
        """Get a constant definition by symbol.

        Args:
            symbol: Constant symbol to find.

        Returns:
            ConstantDefinition if found, None otherwise.
        """
        for const in self.constants:
            if const.symbol == symbol:
                return const
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON/YAML.
        """
        result: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "schema_version": self.schema_version,
        }

        if self.description:
            result["description"] = self.description
        if self.author:
            result["author"] = self.author
        if self.license:
            result["license"] = self.license
        if self.dependencies:
            result["dependencies"] = self.dependencies
        if self.units:
            result["units"] = [u.to_dict() for u in self.units]
        if self.constants:
            result["constants"] = [c.to_dict() for c in self.constants]

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnitSchema:
        """Create from dictionary.

        Args:
            data: Dictionary representation (from JSON/YAML).

        Returns:
            UnitSchema instance.
        """
        return cls(
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            author=data.get("author", ""),
            license=data.get("license", "MIT"),
            dependencies=data.get("dependencies", []),
            units=[
                UnitDefinition.from_dict(u) for u in data.get("units", [])
            ],
            constants=[
                ConstantDefinition.from_dict(c) for c in data.get("constants", [])
            ],
            schema_version=data.get("schema_version", "1.0.0"),
        )

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"UnitSchema(name={self.name!r}, version={self.version!r}, "
            f"{len(self.units)} units, {len(self.constants)} constants)"
        )

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Schema: {self.name} v{self.version}",
        ]
        if self.description:
            lines.append(f"  {self.description}")
        lines.append(f"  Units: {len(self.units)}")
        lines.append(f"  Constants: {len(self.constants)}")
        return "\n".join(lines)

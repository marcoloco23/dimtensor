"""Dataset cards for sharing physics data with unit metadata.

Dataset cards provide structured metadata about datasets, including
per-column unit specifications, coordinate systems, and uncertainties.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..core.dimensions import Dimension
from ..core.units import Unit


class CoordinateSystem(Enum):
    """Coordinate system type."""

    CARTESIAN = "cartesian"
    SPHERICAL = "spherical"
    CYLINDRICAL = "cylindrical"
    CUSTOM = "custom"


@dataclass
class ColumnInfo:
    """Metadata about a single column in a dataset.

    Attributes:
        name: Column name.
        unit: Unit for this column's values.
        description: Human-readable description.
        uncertainty_col: Name of column containing uncertainties (if any).
        coordinate_role: Role in coordinate system ('x', 'y', 'z', 'r', 'theta', etc.).
        required: Whether this column is required.
    """

    name: str
    unit: Unit
    description: str = ""
    uncertainty_col: str | None = None
    coordinate_role: str | None = None
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "unit": {
                "symbol": self.unit.symbol,
                "scale": self.unit.scale,
                "dimension": {
                    "length": float(self.unit.dimension.length),
                    "mass": float(self.unit.dimension.mass),
                    "time": float(self.unit.dimension.time),
                    "current": float(self.unit.dimension.current),
                    "temperature": float(self.unit.dimension.temperature),
                    "amount": float(self.unit.dimension.amount),
                    "luminosity": float(self.unit.dimension.luminosity),
                },
            },
            "description": self.description,
            "uncertainty_col": self.uncertainty_col,
            "coordinate_role": self.coordinate_role,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColumnInfo":
        """Create from dictionary."""
        from fractions import Fraction

        unit_data = data["unit"]
        dim_data = unit_data["dimension"]

        dimension = Dimension(
            length=Fraction(dim_data["length"]).limit_denominator(),
            mass=Fraction(dim_data["mass"]).limit_denominator(),
            time=Fraction(dim_data["time"]).limit_denominator(),
            current=Fraction(dim_data["current"]).limit_denominator(),
            temperature=Fraction(dim_data["temperature"]).limit_denominator(),
            amount=Fraction(dim_data["amount"]).limit_denominator(),
            luminosity=Fraction(dim_data["luminosity"]).limit_denominator(),
        )

        unit = Unit(
            symbol=unit_data["symbol"],
            dimension=dimension,
            scale=unit_data["scale"],
        )

        return cls(
            name=data["name"],
            unit=unit,
            description=data.get("description", ""),
            uncertainty_col=data.get("uncertainty_col"),
            coordinate_role=data.get("coordinate_role"),
            required=data.get("required", True),
        )


@dataclass
class DimDatasetCard:
    """Complete dataset card with all metadata.

    A dataset card contains all information needed to understand,
    use, and validate a physics dataset.

    Attributes:
        name: Dataset name.
        description: Human-readable description.
        domain: Physics domain (e.g., "mechanics", "thermodynamics").
        columns: List of column metadata.
        coordinate_system: Coordinate system type.
        version: Dataset version.
        source: Data source URL or reference.
        license: Dataset license.
        citation: How to cite the dataset.
        tags: List of tags for filtering.
    """

    name: str
    description: str = ""
    domain: str = "general"
    columns: list[ColumnInfo] = field(default_factory=list)
    coordinate_system: CoordinateSystem = CoordinateSystem.CARTESIAN
    version: str = "1.0"
    source: str = ""
    license: str = ""
    citation: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "columns": [col.to_dict() for col in self.columns],
            "coordinate_system": self.coordinate_system.value,
            "version": self.version,
            "source": self.source,
            "license": self.license,
            "citation": self.citation,
            "tags": self.tags,
            "dimtensor_card_version": "1.0",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DimDatasetCard":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            domain=data.get("domain", "general"),
            columns=[ColumnInfo.from_dict(col) for col in data.get("columns", [])],
            coordinate_system=CoordinateSystem(data.get("coordinate_system", "cartesian")),
            version=data.get("version", "1.0"),
            source=data.get("source", ""),
            license=data.get("license", ""),
            citation=data.get("citation", ""),
            tags=data.get("tags", []),
        )

    def get_column(self, name: str) -> ColumnInfo | None:
        """Get column info by name.

        Args:
            name: Column name.

        Returns:
            ColumnInfo if found, None otherwise.
        """
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def add_column(
        self,
        name: str,
        unit: Unit,
        description: str = "",
        uncertainty_col: str | None = None,
        coordinate_role: str | None = None,
        required: bool = True,
    ) -> None:
        """Add a column to the dataset card.

        Args:
            name: Column name.
            unit: Unit for this column.
            description: Column description.
            uncertainty_col: Name of uncertainty column (if any).
            coordinate_role: Role in coordinate system.
            required: Whether column is required.
        """
        col = ColumnInfo(
            name=name,
            unit=unit,
            description=description,
            uncertainty_col=uncertainty_col,
            coordinate_role=coordinate_role,
            required=required,
        )
        self.columns.append(col)

    def to_markdown(self) -> str:
        """Generate markdown representation of the dataset card."""
        lines = [
            f"# {self.name}",
            "",
            f"**Version:** {self.version}",
            f"**Domain:** {self.domain}",
            f"**License:** {self.license}",
            f"**Coordinate System:** {self.coordinate_system.value}",
            "",
            "## Description",
            "",
            self.description or "No description provided.",
            "",
        ]

        # Column information
        if self.columns:
            lines.extend([
                "## Columns",
                "",
                "| Name | Unit | Description | Uncertainty | Coordinate Role | Required |",
                "|------|------|-------------|-------------|----------------|----------|",
            ])
            for col in self.columns:
                unc = col.uncertainty_col or "-"
                role = col.coordinate_role or "-"
                req = "Yes" if col.required else "No"
                lines.append(
                    f"| {col.name} | {col.unit.symbol} | {col.description or '-'} | {unc} | {role} | {req} |"
                )
            lines.append("")

        # Source
        if self.source:
            lines.extend([
                "## Source",
                "",
                self.source,
                "",
            ])

        # Citation
        if self.citation:
            lines.extend([
                "## Citation",
                "",
                "```",
                self.citation,
                "```",
                "",
            ])

        # Tags
        if self.tags:
            tags = ", ".join(f"`{t}`" for t in self.tags)
            lines.extend([
                "## Tags",
                "",
                tags,
                "",
            ])

        return "\n".join(lines)


def save_dataset_card(card: DimDatasetCard, path: str | Path) -> None:
    """Save a dataset card to a file.

    Args:
        card: DimDatasetCard to save.
        path: Path to save to. Extension determines format (.json or .md).
    """
    path = Path(path)

    if path.suffix == ".md":
        # Save as markdown
        with open(path, "w") as f:
            f.write(card.to_markdown())
    else:
        # Save as JSON
        with open(path, "w") as f:
            json.dump(card.to_dict(), f, indent=2)


def load_dataset_card(path: str | Path) -> DimDatasetCard:
    """Load a dataset card from a JSON file.

    Args:
        path: Path to the dataset card file.

    Returns:
        DimDatasetCard loaded from file.

    Raises:
        ValueError: If file format not supported.
    """
    path = Path(path)

    if path.suffix == ".md":
        raise ValueError("Cannot load dataset card from markdown (use JSON)")

    with open(path) as f:
        data = json.load(f)

    return DimDatasetCard.from_dict(data)

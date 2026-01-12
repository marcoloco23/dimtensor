"""Paper metadata for reproduction studies.

This module defines the Paper class for storing physics paper metadata,
including equations, published values, and reproduction metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

from ..core.dimarray import DimArray
from ..core.units import Unit
from ..equations import Equation


@dataclass
class Paper:
    """Metadata for a physics paper to be reproduced.

    Stores all information needed to reproduce computational results
    from a physics paper, including equations used, published values,
    and methodological details.

    Attributes:
        title: Paper title.
        authors: List of author names.
        doi: Digital Object Identifier.
        year: Publication year.
        journal: Journal name.
        abstract: Brief abstract/summary.
        equations: Dict mapping equation names to Equation objects.
        published_values: Dict mapping quantity names to published DimArrays.
        units_used: Dict mapping quantity to original units in paper.
        methods: List of numerical methods used.
        assumptions: List of key assumptions.
        tags: List of tags for categorization.
        url: URL to paper (if available).
        notes: Additional notes about the paper.

    Example:
        >>> from dimtensor.research import Paper
        >>> from dimtensor import DimArray, units
        >>>
        >>> paper = Paper(
        ...     title="Schwarzschild Solution",
        ...     authors=["K. Schwarzschild"],
        ...     doi="10.1002/asna.19160200402",
        ...     year=1916,
        ...     journal="Astronomische Nachrichten",
        ...     published_values={
        ...         "solar_schwarzschild_radius": DimArray(2.95, units.km),
        ...     },
        ...     assumptions=["Spherical symmetry", "Static metric"],
        ...     tags=["general relativity", "black holes"]
        ... )
    """

    title: str
    authors: list[str]
    doi: str = ""
    year: int = 0
    journal: str = ""
    abstract: str = ""
    equations: dict[str, Equation] = field(default_factory=dict)
    published_values: dict[str, DimArray] = field(default_factory=dict)
    units_used: dict[str, str] = field(default_factory=dict)
    methods: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    url: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the paper.
        """
        from ..io.json import to_dict as dimarray_to_dict

        return {
            "title": self.title,
            "authors": self.authors,
            "doi": self.doi,
            "year": self.year,
            "journal": self.journal,
            "abstract": self.abstract,
            "equations": {k: eq.to_dict() for k, eq in self.equations.items()},
            "published_values": {
                k: dimarray_to_dict(v) for k, v in self.published_values.items()
            },
            "units_used": self.units_used,
            "methods": self.methods,
            "assumptions": self.assumptions,
            "tags": self.tags,
            "url": self.url,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Paper:
        """Create from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            Paper object.
        """
        from ..io.json import from_dict as dimarray_from_dict

        # Reconstruct equations
        equations = {}
        for name, eq_dict in data.get("equations", {}).items():
            from ..core.dimensions import Dimension

            variables = {}
            for var_name, dim_str in eq_dict.get("variables", {}).items():
                # Parse dimension string (e.g., "L M T^-2")
                variables[var_name] = Dimension.from_string(dim_str)

            equations[name] = Equation(
                name=eq_dict["name"],
                formula=eq_dict["formula"],
                variables=variables,
                domain=eq_dict.get("domain", "general"),
                tags=eq_dict.get("tags", []),
                description=eq_dict.get("description", ""),
                assumptions=eq_dict.get("assumptions", []),
                latex=eq_dict.get("latex", ""),
                related=eq_dict.get("related", []),
            )

        # Reconstruct published values
        published_values = {
            k: dimarray_from_dict(v) for k, v in data.get("published_values", {}).items()
        }

        return cls(
            title=data["title"],
            authors=data["authors"],
            doi=data.get("doi", ""),
            year=data.get("year", 0),
            journal=data.get("journal", ""),
            abstract=data.get("abstract", ""),
            equations=equations,
            published_values=published_values,
            units_used=data.get("units_used", {}),
            methods=data.get("methods", []),
            assumptions=data.get("assumptions", []),
            tags=data.get("tags", []),
            url=data.get("url", ""),
            notes=data.get("notes", ""),
        )

    def to_json(self, path: str | Path) -> None:
        """Save paper metadata to JSON file.

        Args:
            path: Path to save JSON file.
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> Paper:
        """Load paper metadata from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            Paper object.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def add_equation(self, name: str, equation: Equation) -> None:
        """Add an equation to the paper.

        Args:
            name: Name to use for the equation.
            equation: Equation object.
        """
        self.equations[name] = equation

    def add_published_value(
        self, name: str, value: DimArray, unit_string: str | None = None
    ) -> None:
        """Add a published value to the paper.

        Args:
            name: Name of the quantity.
            value: Published value as DimArray.
            unit_string: String representation of the original units (optional).
        """
        self.published_values[name] = value
        if unit_string is not None:
            self.units_used[name] = unit_string

    def __repr__(self) -> str:
        """String representation."""
        author_str = ", ".join(self.authors[:2])
        if len(self.authors) > 2:
            author_str += " et al."
        return f"Paper('{self.title}' by {author_str}, {self.year})"

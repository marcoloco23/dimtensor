"""Reproduction result tracking for physics papers.

This module defines classes for tracking reproduction attempts,
including computed values, comparison metrics, and reproduction status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
import json

from ..core.dimarray import DimArray
from .paper import Paper
from .comparison import ComparisonResult


class ReproductionStatus(Enum):
    """Status of a reproduction attempt."""

    SUCCESS = "success"  # All values match within tolerance
    PARTIAL = "partial"  # Some values match, some don't
    FAILED = "failed"  # Significant discrepancies or couldn't compute
    PENDING = "pending"  # Reproduction not yet completed


@dataclass
class ReproductionResult:
    """Results from reproducing a paper's computations.

    Tracks all information about a reproduction attempt, including
    computed values, comparison metrics, and reproduction metadata.

    Attributes:
        paper: Reference to Paper object being reproduced.
        computed_values: Dict mapping quantity names to computed DimArrays.
        comparisons: Dict mapping quantity names to ComparisonResult objects.
        status: Overall reproduction status.
        reproduction_date: When reproduction was performed.
        reproducer: Name/email of person reproducing.
        code_repository: Link to reproduction code.
        notes: Additional notes about reproduction process.
        discrepancies: List of significant differences found.
        computation_time: Time taken for computation (seconds).
        software_versions: Dict of software versions used.

    Example:
        >>> from dimtensor.research import Paper, ReproductionResult
        >>> from dimtensor import DimArray, units
        >>>
        >>> paper = Paper(title="Test", authors=["Author"])
        >>> result = ReproductionResult(
        ...     paper=paper,
        ...     computed_values={"e": DimArray(1.602e-19, units.C)},
        ...     status=ReproductionStatus.SUCCESS,
        ...     reproducer="Researcher Name"
        ... )
    """

    paper: Paper
    computed_values: dict[str, DimArray] = field(default_factory=dict)
    comparisons: dict[str, ComparisonResult] = field(default_factory=dict)
    status: ReproductionStatus = ReproductionStatus.PENDING
    reproduction_date: str = field(default_factory=lambda: datetime.now().isoformat())
    reproducer: str = ""
    code_repository: str = ""
    notes: str = ""
    discrepancies: list[str] = field(default_factory=list)
    computation_time: float = 0.0
    software_versions: dict[str, str] = field(default_factory=dict)

    def add_computed_value(self, name: str, value: DimArray) -> None:
        """Add a computed value to the result.

        Args:
            name: Name of the quantity.
            value: Computed value as DimArray.
        """
        self.computed_values[name] = value

    def add_comparison(self, name: str, comparison: ComparisonResult) -> None:
        """Add a comparison result.

        Args:
            name: Name of the quantity compared.
            comparison: ComparisonResult object.
        """
        self.comparisons[name] = comparison

        # Update status based on comparisons
        self._update_status()

    def _update_status(self) -> None:
        """Update reproduction status based on comparison results."""
        if not self.comparisons:
            self.status = ReproductionStatus.PENDING
            return

        matches = [c.matches for c in self.comparisons.values()]
        if all(matches):
            self.status = ReproductionStatus.SUCCESS
        elif any(matches):
            self.status = ReproductionStatus.PARTIAL
        else:
            self.status = ReproductionStatus.FAILED

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics.

        Returns:
            Dictionary with summary statistics.
        """
        if not self.comparisons:
            return {
                "status": self.status.value,
                "num_quantities": len(self.computed_values),
                "num_compared": 0,
            }

        relative_errors = [
            abs(c.relative_error) for c in self.comparisons.values() if c.relative_error is not None
        ]

        return {
            "status": self.status.value,
            "num_quantities": len(self.computed_values),
            "num_compared": len(self.comparisons),
            "num_matches": sum(1 for c in self.comparisons.values() if c.matches),
            "mean_relative_error": sum(relative_errors) / len(relative_errors) if relative_errors else None,
            "max_relative_error": max(relative_errors) if relative_errors else None,
            "reproduction_date": self.reproduction_date,
            "reproducer": self.reproducer,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        from ..io.json import to_dict as dimarray_to_dict

        return {
            "paper": self.paper.to_dict(),
            "computed_values": {
                k: dimarray_to_dict(v) for k, v in self.computed_values.items()
            },
            "comparisons": {k: c.to_dict() for k, c in self.comparisons.items()},
            "status": self.status.value,
            "reproduction_date": self.reproduction_date,
            "reproducer": self.reproducer,
            "code_repository": self.code_repository,
            "notes": self.notes,
            "discrepancies": self.discrepancies,
            "computation_time": self.computation_time,
            "software_versions": self.software_versions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReproductionResult:
        """Create from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            ReproductionResult object.
        """
        from ..io.json import from_dict as dimarray_from_dict

        paper = Paper.from_dict(data["paper"])

        computed_values = {
            k: dimarray_from_dict(v) for k, v in data.get("computed_values", {}).items()
        }

        comparisons = {
            k: ComparisonResult.from_dict(v) for k, v in data.get("comparisons", {}).items()
        }

        return cls(
            paper=paper,
            computed_values=computed_values,
            comparisons=comparisons,
            status=ReproductionStatus(data.get("status", "pending")),
            reproduction_date=data.get("reproduction_date", ""),
            reproducer=data.get("reproducer", ""),
            code_repository=data.get("code_repository", ""),
            notes=data.get("notes", ""),
            discrepancies=data.get("discrepancies", []),
            computation_time=data.get("computation_time", 0.0),
            software_versions=data.get("software_versions", {}),
        )

    def to_json(self, path: str | Path) -> None:
        """Save result to JSON file.

        Args:
            path: Path to save JSON file.
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> ReproductionResult:
        """Load result from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            ReproductionResult object.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ReproductionResult(paper='{self.paper.title}', "
            f"status={self.status.value}, "
            f"comparisons={len(self.comparisons)})"
        )

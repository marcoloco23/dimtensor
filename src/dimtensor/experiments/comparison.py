"""Run comparison utilities for experiment tracking.

Provides RunComparison class for comparing metrics across runs with
automatic unit conversion.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.dimarray import DimArray
from ..core.dimensions import Dimension
from ..core.units import Unit
from fractions import Fraction


class RunComparison:
    """Compare metrics across multiple experiment runs.

    Automatically handles unit conversions when comparing runs that
    used different unit systems (e.g., SI vs CGS).

    Args:
        runs_data: List of run data dictionaries from backend.
        metric_names: List of metric names to compare.

    Examples:
        >>> comparison = RunComparison(runs_data, ["loss", "energy"])
        >>> df = comparison.to_dataframe()
        >>> comparison.plot("loss")
    """

    def __init__(
        self,
        runs_data: list[dict[str, Any]],
        metric_names: list[str],
    ):
        """Initialize run comparison."""
        self.runs_data = runs_data
        self.metric_names = metric_names
        self._comparisons: dict[str, dict[str, Any]] = {}

        # Compute comparisons for each metric
        for metric_name in metric_names:
            self._comparisons[metric_name] = self._compare_metric(metric_name)

    def _parse_unit_from_tags(self, tags: dict[str, Any], metric_name: str) -> Unit | None:
        """Parse unit from run tags.

        Args:
            tags: Run tags dictionary.
            metric_name: Name of the metric.

        Returns:
            Unit object or None if not found/dimensionless.
        """
        unit_key = f"unit.{metric_name}"
        if unit_key not in tags:
            return None

        unit_str = tags[unit_key]
        if unit_str == "dimensionless" or unit_str == "1":
            from ..core.units import dimensionless
            return dimensionless

        # Try to reconstruct unit from symbol
        # This is a simplified approach - for full reconstruction,
        # we'd need dimension info stored as well
        try:
            # Import common units
            from ..core import units
            return getattr(units, unit_str, None)
        except:
            return None

    def _compare_metric(self, metric_name: str) -> dict[str, Any]:
        """Compare a single metric across runs.

        Args:
            metric_name: Name of metric to compare.

        Returns:
            Dictionary with comparison results.
        """
        values = []
        units_list = []
        run_names = []

        for run_data in self.runs_data:
            metrics = run_data.get("metrics", {})
            tags = run_data.get("tags", {})

            if metric_name not in metrics:
                # Skip runs that don't have this metric
                continue

            value = metrics[metric_name]
            unit = self._parse_unit_from_tags(tags, metric_name)

            values.append(value)
            units_list.append(unit)
            run_names.append(run_data.get("run_name", run_data.get("run_id", "unknown")))

        if not values:
            return {
                "metric_name": metric_name,
                "values": [],
                "units": [],
                "run_names": [],
                "compatible": False,
                "warning": f"No runs found with metric '{metric_name}'",
            }

        # Check unit compatibility
        if len(set([u.dimension if u else None for u in units_list])) > 1:
            # Incompatible dimensions
            return {
                "metric_name": metric_name,
                "values": values,
                "units": [u.symbol if u else "dimensionless" for u in units_list],
                "run_names": run_names,
                "compatible": False,
                "warning": "Runs have incompatible dimensions for this metric",
            }

        # Convert all to first unit for comparison
        target_unit = units_list[0]
        converted_values = []

        for value, unit in zip(values, units_list):
            if unit is None or target_unit is None:
                # No conversion needed for dimensionless
                converted_values.append(value)
            elif unit.dimension == target_unit.dimension:
                # Convert to target unit
                conversion_factor = unit.conversion_factor(target_unit)
                converted_values.append(value * conversion_factor)
            else:
                # Should not happen if compatibility check passed
                converted_values.append(value)

        # Compute statistics
        converted_array = np.array(converted_values)
        mean_val = float(np.mean(converted_array))
        std_val = float(np.std(converted_array))
        min_val = float(np.min(converted_array))
        max_val = float(np.max(converted_array))

        # Compute relative differences
        if mean_val != 0:
            relative_diffs = [(v - mean_val) / mean_val * 100 for v in converted_values]
        else:
            relative_diffs = [0.0] * len(converted_values)

        return {
            "metric_name": metric_name,
            "values": values,
            "converted_values": converted_values,
            "units": [u.symbol if u else "dimensionless" for u in units_list],
            "common_unit": target_unit.symbol if target_unit else "dimensionless",
            "run_names": run_names,
            "compatible": True,
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "relative_differences_percent": relative_diffs,
        }

    def to_dataframe(self) -> Any:
        """Convert comparison results to a pandas DataFrame.

        Returns:
            DataFrame with run names as rows and metrics as columns.

        Raises:
            ImportError: If pandas is not installed.

        Examples:
            >>> comparison = exp.compare_runs(["run1", "run2"], ["loss"])
            >>> df = comparison.to_dataframe()
            >>> print(df)
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for DataFrame export. "
                "Install with: pip install pandas"
            )

        # Build DataFrame data
        data_rows = []

        for metric_name, comp_data in self._comparisons.items():
            if not comp_data.get("compatible", False):
                continue

            for i, run_name in enumerate(comp_data["run_names"]):
                data_rows.append({
                    "run_name": run_name,
                    "metric": metric_name,
                    "value": comp_data["converted_values"][i],
                    "unit": comp_data["common_unit"],
                    "original_value": comp_data["values"][i],
                    "original_unit": comp_data["units"][i],
                    "relative_diff_percent": comp_data["relative_differences_percent"][i],
                })

        if not data_rows:
            # Return empty DataFrame
            return pd.DataFrame(columns=[
                "run_name", "metric", "value", "unit",
                "original_value", "original_unit", "relative_diff_percent"
            ])

        return pd.DataFrame(data_rows)

    def plot(self, metric_name: str, show: bool = True) -> Any:
        """Plot comparison for a specific metric.

        Args:
            metric_name: Name of metric to plot.
            show: If True, display the plot immediately.

        Returns:
            Matplotlib figure object.

        Raises:
            ImportError: If matplotlib is not installed.

        Examples:
            >>> comparison = exp.compare_runs(["run1", "run2"], ["loss"])
            >>> comparison.plot("loss")
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            )

        if metric_name not in self._comparisons:
            raise ValueError(f"Metric '{metric_name}' not in comparison")

        comp_data = self._comparisons[metric_name]

        if not comp_data.get("compatible", False):
            raise ValueError(
                f"Cannot plot metric '{metric_name}': {comp_data.get('warning', 'incompatible units')}"
            )

        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))

        run_names = comp_data["run_names"]
        values = comp_data["converted_values"]
        unit = comp_data["common_unit"]

        x_pos = np.arange(len(run_names))
        bars = ax.bar(x_pos, values, align="center", alpha=0.7)

        # Color bars by relative difference
        rel_diffs = np.array(comp_data["relative_differences_percent"])
        for i, (bar, rel_diff) in enumerate(zip(bars, rel_diffs)):
            if abs(rel_diff) < 5:
                bar.set_color("green")
            elif abs(rel_diff) < 20:
                bar.set_color("orange")
            else:
                bar.set_color("red")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(run_names, rotation=45, ha="right")
        ax.set_ylabel(f"{metric_name} [{unit}]")
        ax.set_title(f"Comparison: {metric_name}")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, (x, y) in enumerate(zip(x_pos, values)):
            ax.text(x, y, f"{y:.3g}", ha="center", va="bottom")

        # Add legend for colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="green", alpha=0.7, label="< 5% diff"),
            Patch(facecolor="orange", alpha=0.7, label="5-20% diff"),
            Patch(facecolor="red", alpha=0.7, label="> 20% diff"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def get_best_run(self, metric_name: str, mode: str = "min") -> dict[str, Any]:
        """Get the run with the best value for a metric.

        Args:
            metric_name: Name of metric to optimize.
            mode: "min" for minimization (e.g., loss), "max" for maximization (e.g., accuracy).

        Returns:
            Dictionary with best run information.

        Examples:
            >>> best = comparison.get_best_run("loss", mode="min")
            >>> print(f"Best run: {best['run_name']} with loss={best['value']}")
        """
        if metric_name not in self._comparisons:
            raise ValueError(f"Metric '{metric_name}' not in comparison")

        comp_data = self._comparisons[metric_name]

        if not comp_data.get("compatible", False):
            raise ValueError(
                f"Cannot find best run for '{metric_name}': {comp_data.get('warning', 'incompatible')}"
            )

        values = comp_data["converted_values"]
        run_names = comp_data["run_names"]

        if mode == "min":
            best_idx = int(np.argmin(values))
        elif mode == "max":
            best_idx = int(np.argmax(values))
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'min' or 'max'")

        return {
            "run_name": run_names[best_idx],
            "value": values[best_idx],
            "unit": comp_data["common_unit"],
            "original_value": comp_data["values"][best_idx],
            "original_unit": comp_data["units"][best_idx],
        }

    def summary(self) -> str:
        """Get a text summary of the comparison.

        Returns:
            Multi-line string summarizing comparison results.

        Examples:
            >>> print(comparison.summary())
        """
        lines = []
        lines.append("=" * 60)
        lines.append("Run Comparison Summary")
        lines.append("=" * 60)

        for metric_name, comp_data in self._comparisons.items():
            lines.append(f"\nMetric: {metric_name}")
            lines.append("-" * 40)

            if not comp_data.get("compatible", False):
                lines.append(f"  ⚠️  {comp_data.get('warning', 'Incompatible')}")
                continue

            lines.append(f"  Unit: {comp_data['common_unit']}")
            lines.append(f"  Runs: {len(comp_data['run_names'])}")
            lines.append(f"  Mean: {comp_data['mean']:.4g}")
            lines.append(f"  Std:  {comp_data['std']:.4g}")
            lines.append(f"  Min:  {comp_data['min']:.4g}")
            lines.append(f"  Max:  {comp_data['max']:.4g}")

            lines.append("\n  Run Details:")
            for i, run_name in enumerate(comp_data["run_names"]):
                val = comp_data["converted_values"][i]
                rel_diff = comp_data["relative_differences_percent"][i]
                lines.append(f"    {run_name}: {val:.4g} ({rel_diff:+.1f}%)")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)


__all__ = [
    "RunComparison",
]

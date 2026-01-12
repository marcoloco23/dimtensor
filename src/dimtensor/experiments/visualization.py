"""Visualization utilities for experiment tracking.

Provides plotting functions for experiment histories and comparisons
with automatic unit labels.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def plot_experiment_history(
    experiment: Any,
    metric_name: str,
    run_ids: list[str] | None = None,
    show: bool = True,
) -> Any:
    """Plot metric history across one or more runs.

    Args:
        experiment: DimExperiment instance.
        metric_name: Name of metric to plot.
        run_ids: Optional list of specific run IDs to plot. If None, plots all runs.
        show: If True, display the plot immediately.

    Returns:
        Matplotlib figure object.

    Raises:
        ImportError: If matplotlib is not installed.

    Examples:
        >>> from dimtensor.experiments import DimExperiment
        >>> from dimtensor.experiments.visualization import plot_experiment_history
        >>>
        >>> exp = DimExperiment("my-experiment")
        >>> plot_experiment_history(exp, "loss")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )

    # Get runs
    if run_ids is None:
        runs = experiment.list_runs()
    else:
        runs = [experiment.get_run(run_id) for run_id in run_ids]

    if not runs:
        raise ValueError("No runs found to plot")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each run
    for run in runs:
        run_name = run.get("run_name", run.get("run_id", "unknown"))
        metrics = run.get("metrics", {})
        tags = run.get("tags", {})

        # For now, we only plot final metric values
        # (step-wise plotting would require accessing metric history from backend)
        if metric_name in metrics:
            value = metrics[metric_name]
            unit_key = f"unit.{metric_name}"
            unit = tags.get(unit_key, "dimensionless")

            ax.scatter([run_name], [value], s=100, alpha=0.7, label=run_name)

    # Get unit for y-axis label
    if runs and runs[0].get("tags"):
        unit_key = f"unit.{metric_name}"
        unit = runs[0]["tags"].get(unit_key, "dimensionless")
    else:
        unit = "dimensionless"

    ax.set_ylabel(f"{metric_name} [{unit}]")
    ax.set_xlabel("Run")
    ax.set_title(f"Experiment History: {metric_name}")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_run_comparison(
    comparison: Any,
    metrics: list[str] | None = None,
    show: bool = True,
) -> Any:
    """Plot comparison of multiple metrics across runs.

    Args:
        comparison: RunComparison instance.
        metrics: Optional list of specific metrics to plot. If None, plots all.
        show: If True, display the plot immediately.

    Returns:
        Matplotlib figure object.

    Raises:
        ImportError: If matplotlib is not installed.

    Examples:
        >>> comparison = exp.compare_runs(["run1", "run2"], ["loss", "energy"])
        >>> plot_run_comparison(comparison)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )

    if metrics is None:
        metrics = comparison.metric_names
    else:
        # Validate metrics
        for m in metrics:
            if m not in comparison.metric_names:
                raise ValueError(f"Metric '{m}' not in comparison")

    n_metrics = len(metrics)
    if n_metrics == 0:
        raise ValueError("No metrics to plot")

    # Create subplots
    if n_metrics == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes = [axes]
    else:
        cols = min(2, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 6 * rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]

    for idx, metric_name in enumerate(metrics):
        ax = axes[idx]

        comp_data = comparison._comparisons[metric_name]

        if not comp_data.get("compatible", False):
            ax.text(0.5, 0.5, f"⚠️ {comp_data.get('warning', 'Incompatible')}",
                   ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{metric_name} (incompatible)")
            continue

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
        ax.set_title(f"{metric_name}")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels
        for x, y in zip(x_pos, values):
            ax.text(x, y, f"{y:.3g}", ha="center", va="bottom", fontsize=8)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_parameter_importance(
    experiment: Any,
    target_metric: str,
    param_names: list[str] | None = None,
    show: bool = True,
) -> Any:
    """Plot correlation between parameters and a target metric.

    Shows which parameters have the strongest influence on the target metric
    across all runs.

    Args:
        experiment: DimExperiment instance.
        target_metric: Name of metric to analyze.
        param_names: Optional list of specific parameters to analyze. If None, uses all.
        show: If True, display the plot immediately.

    Returns:
        Matplotlib figure object.

    Raises:
        ImportError: If matplotlib is not installed.

    Examples:
        >>> plot_parameter_importance(exp, "loss", param_names=["learning_rate", "batch_size"])
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )

    # Get all runs
    runs = experiment.list_runs()

    if not runs:
        raise ValueError("No runs found")

    # Extract parameters and target metric
    param_values: dict[str, list[float]] = {}
    metric_values = []

    for run in runs:
        params = run.get("params", {})
        metrics = run.get("metrics", {})

        if target_metric not in metrics:
            continue

        metric_values.append(float(metrics[target_metric]))

        for param_name, param_value in params.items():
            if param_names is not None and param_name not in param_names:
                continue

            try:
                val = float(param_value)
                if param_name not in param_values:
                    param_values[param_name] = []
                param_values[param_name].append(val)
            except (ValueError, TypeError):
                # Skip non-numeric parameters
                pass

    if not param_values or not metric_values:
        raise ValueError("No valid parameter-metric data found")

    # Compute correlations
    correlations = {}
    for param_name, values in param_values.items():
        if len(values) == len(metric_values) and len(values) > 1:
            corr = np.corrcoef(values, metric_values)[0, 1]
            if not np.isnan(corr):
                correlations[param_name] = corr

    if not correlations:
        raise ValueError("Could not compute correlations (insufficient data)")

    # Sort by absolute correlation
    sorted_params = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_params) * 0.5)))

    param_names_sorted = [p[0] for p in sorted_params]
    corr_values = [p[1] for p in sorted_params]

    y_pos = np.arange(len(param_names_sorted))
    bars = ax.barh(y_pos, corr_values, align="center", alpha=0.7)

    # Color bars: positive = green, negative = red
    for bar, corr in zip(bars, corr_values):
        bar.set_color("green" if corr > 0 else "red")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names_sorted)
    ax.set_xlabel(f"Correlation with {target_metric}")
    ax.set_title(f"Parameter Importance for {target_metric}")
    ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)

    # Add correlation values
    for y, corr in zip(y_pos, corr_values):
        ax.text(corr, y, f"{corr:.3f}", ha="left" if corr > 0 else "right", va="center")

    plt.tight_layout()

    if show:
        plt.show()

    return fig


__all__ = [
    "plot_experiment_history",
    "plot_run_comparison",
    "plot_parameter_importance",
]

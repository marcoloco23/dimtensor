"""Experiment tracking system for dimtensor.

Provides high-level experiment tracking API with full unit awareness,
supporting MLflow and Weights & Biases backends.

Examples:
    >>> from dimtensor import DimArray, units
    >>> from dimtensor.experiments import DimExperiment
    >>>
    >>> # Create experiment
    >>> exp = DimExperiment("heat-equation-pinn", backend="mlflow")
    >>>
    >>> # Run 1: SI units
    >>> with exp.start_run("run-si"):
    ...     exp.log_param("learning_rate", DimArray(0.001, 1/units.s))
    ...     exp.log_param("domain_length", DimArray(1.0, units.m))
    ...
    ...     for step in range(100):
    ...         loss = compute_loss()  # returns DimArray with units.J
    ...         exp.log_metric("loss", loss, step=step)
    >>>
    >>> # Run 2: CGS units (for comparison)
    >>> with exp.start_run("run-cgs"):
    ...     exp.log_param("learning_rate", DimArray(0.001, 1/units.s))
    ...     exp.log_param("domain_length", DimArray(100.0, units.cm))
    ...
    ...     for step in range(100):
    ...         loss = compute_loss()  # returns DimArray with units.erg
    ...         exp.log_metric("loss", loss, step=step)
    >>>
    >>> # Compare runs (automatic unit conversion)
    >>> comparison = exp.compare_runs(["run-si", "run-cgs"], ["loss"])
    >>> print(comparison.to_dataframe())
    >>> comparison.plot("loss")  # Shows both in same units
"""

from .backend import ExperimentBackend, MLflowBackend, WandbBackend
from .comparison import RunComparison
from .experiment import DimExperiment
from .visualization import (
    plot_experiment_history,
    plot_parameter_importance,
    plot_run_comparison,
)

__all__ = [
    "DimExperiment",
    "RunComparison",
    "ExperimentBackend",
    "MLflowBackend",
    "WandbBackend",
    "plot_experiment_history",
    "plot_run_comparison",
    "plot_parameter_importance",
]

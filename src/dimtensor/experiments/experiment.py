"""High-level experiment tracking API for dimtensor.

Provides DimExperiment class for organizing and tracking physics experiments
with full unit awareness across runs.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..core.dimarray import DimArray
from .backend import ExperimentBackend, MLflowBackend, WandbBackend

# Forward declaration for type hints
if False:
    from .comparison import RunComparison


class DimExperiment:
    """High-level experiment tracking with unit awareness.

    DimExperiment wraps experiment tracking backends (MLflow, W&B) to provide
    a unified API for logging parameters, metrics, and arrays with their
    physical units preserved.

    Args:
        name: Experiment name.
        backend: Backend type ("mlflow" or "wandb").
        tracking_uri: Optional tracking server URI (MLflow only).
        entity: Optional entity/organization (W&B only).

    Examples:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.experiments import DimExperiment
        >>>
        >>> exp = DimExperiment("heat-equation", backend="mlflow")
        >>>
        >>> with exp.start_run("run-1"):
        ...     exp.log_param("learning_rate", DimArray(0.001, 1/units.s))
        ...     exp.log_metric("loss", DimArray(0.5, units.J), step=0)
    """

    def __init__(
        self,
        name: str,
        backend: str = "mlflow",
        tracking_uri: str | None = None,
        entity: str | None = None,
    ):
        """Initialize experiment with specified backend."""
        self.name = name
        self._backend_type = backend.lower()

        # Initialize backend
        if self._backend_type == "mlflow":
            self._backend: ExperimentBackend = MLflowBackend(
                experiment_name=name,
                tracking_uri=tracking_uri,
            )
        elif self._backend_type == "wandb":
            self._backend = WandbBackend(
                project=name,
                entity=entity,
            )
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                "Supported backends: 'mlflow', 'wandb'"
            )

        self._current_run_id: str | None = None

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Iterator[str]:
        """Start a new experiment run (context manager).

        Args:
            run_name: Optional name for the run.
            tags: Optional tags to attach to the run.

        Yields:
            Run ID.

        Examples:
            >>> exp = DimExperiment("my-experiment")
            >>> with exp.start_run("test-run") as run_id:
            ...     exp.log_param("alpha", DimArray(1.0, units.m))
            ...     exp.log_metric("energy", DimArray(42.0, units.J))
        """
        run_id = self._backend.start_run(run_name=run_name, tags=tags)
        self._current_run_id = run_id
        try:
            yield run_id
        finally:
            self.end_run()

    def end_run(self) -> None:
        """End the current run."""
        self._backend.end_run()
        self._current_run_id = None

    def log_param(self, name: str, value: DimArray | float | int | str) -> None:
        """Log a parameter with optional units.

        Parameters are typically hyperparameters or configuration values
        that remain constant throughout a run.

        Args:
            name: Parameter name.
            value: Parameter value (DimArray or scalar).

        Examples:
            >>> exp.log_param("learning_rate", DimArray(0.001, units.Hz))
            >>> exp.log_param("batch_size", 32)
            >>> exp.log_param("domain_length", DimArray(1.0, units.m))
        """
        self._backend.log_param(name, value)

    def log_metric(
        self,
        name: str,
        value: DimArray | float,
        step: int | None = None,
    ) -> None:
        """Log a metric with optional units.

        Metrics are typically training/validation quantities that change
        over time (e.g., loss, accuracy, error).

        Args:
            name: Metric name.
            value: Metric value (DimArray or scalar).
            step: Optional step number for time series tracking.

        Examples:
            >>> for step in range(100):
            ...     loss = compute_loss()  # Returns DimArray with units.J
            ...     exp.log_metric("loss", loss, step=step)
        """
        self._backend.log_metric(name, value, step=step)

    def log_array(
        self,
        name: str,
        array: DimArray,
        step: int | None = None,
    ) -> None:
        """Log an array with unit metadata.

        For large arrays, logs statistical summaries (mean, std, min, max)
        rather than the full array.

        Args:
            name: Array name.
            array: DimArray to log.
            step: Optional step number.

        Examples:
            >>> predictions = DimArray(pred_values, units.m / units.s)
            >>> exp.log_array("velocity_predictions", predictions, step=0)
        """
        self.log_metric(name, array, step=step)

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Get data for a specific run.

        Args:
            run_id: Run identifier.

        Returns:
            Dictionary containing run data (params, metrics, tags, units).

        Examples:
            >>> run_data = exp.get_run("abc123")
            >>> print(run_data["params"])
            >>> print(run_data["metrics"])
        """
        return self._backend.get_run_data(run_id)

    def list_runs(self, filter_string: str | None = None) -> list[dict[str, Any]]:
        """List all runs for this experiment.

        Args:
            filter_string: Optional filter expression (backend-specific syntax).

        Returns:
            List of run data dictionaries.

        Examples:
            >>> runs = exp.list_runs()
            >>> for run in runs:
            ...     print(f"{run['run_name']}: {run['metrics']}")
        """
        return self._backend.list_runs(filter_string=filter_string)

    def compare_runs(
        self,
        run_ids: list[str],
        metric_names: list[str],
    ) -> "RunComparison":
        """Compare metrics across multiple runs.

        Automatically handles unit conversions when comparing runs that
        used different unit systems.

        Args:
            run_ids: List of run IDs to compare.
            metric_names: List of metric names to compare.

        Returns:
            RunComparison object with comparison results.

        Examples:
            >>> comparison = exp.compare_runs(
            ...     ["run1", "run2"],
            ...     ["loss", "energy"]
            ... )
            >>> df = comparison.to_dataframe()
            >>> comparison.plot("loss")
        """
        from .comparison import RunComparison

        # Fetch run data
        runs_data = [self.get_run(run_id) for run_id in run_ids]

        return RunComparison(
            runs_data=runs_data,
            metric_names=metric_names,
        )

    def export(
        self,
        path: str | Path,
        format: str = "json",
        run_ids: list[str] | None = None,
    ) -> None:
        """Export experiment data to a file.

        Args:
            path: Output file path.
            format: Export format ("json" only for now).
            run_ids: Optional list of specific run IDs to export.
                If None, exports all runs.

        Examples:
            >>> exp.export("experiment_data.json")
            >>> exp.export("runs.json", run_ids=["run1", "run2"])
        """
        if format != "json":
            raise ValueError(f"Unsupported format '{format}'. Only 'json' is supported.")

        # Get runs to export
        if run_ids is None:
            runs_data = self.list_runs()
        else:
            runs_data = [self.get_run(run_id) for run_id in run_ids]

        # Export to JSON
        export_data = {
            "experiment_name": self.name,
            "backend": self._backend_type,
            "runs": runs_data,
        }

        import json
        path = Path(path)
        with open(path, "w") as f:
            json.dump(export_data, f, indent=2)

    @classmethod
    def load(
        cls,
        path: str | Path,
        backend: str | None = None,
    ) -> DimExperiment:
        """Load experiment from exported file.

        Args:
            path: Path to exported experiment file.
            backend: Optional backend to use. If None, uses backend from file.

        Returns:
            DimExperiment instance.

        Examples:
            >>> exp = DimExperiment.load("experiment_data.json")
        """
        import json
        path = Path(path)

        with open(path, "r") as f:
            data = json.load(f)

        experiment_name = data["experiment_name"]
        backend_type = backend or data["backend"]

        # Create experiment instance
        exp = cls(name=experiment_name, backend=backend_type)

        # Note: Runs are not automatically imported to the backend.
        # This just loads the metadata. To re-import runs, user would
        # need to manually log them again.

        return exp


__all__ = [
    "DimExperiment",
]

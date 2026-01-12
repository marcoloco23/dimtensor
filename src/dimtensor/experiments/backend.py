"""Backend abstraction for experiment tracking.

Provides a protocol for experiment tracking backends (MLflow, W&B) that
enables unit-aware experiment tracking with different storage systems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from ..core.dimarray import DimArray


@runtime_checkable
class ExperimentBackend(Protocol):
    """Protocol for experiment tracking backends.

    Defines the interface that all experiment backends must implement
    for logging parameters, metrics, and managing runs.
    """

    def start_run(self, run_name: str | None = None, tags: dict[str, str] | None = None) -> str:
        """Start a new experiment run.

        Args:
            run_name: Optional name for the run.
            tags: Optional tags to attach to the run.

        Returns:
            Run ID (string identifier).
        """
        ...

    def end_run(self) -> None:
        """End the current run."""
        ...

    def log_param(self, name: str, value: DimArray | float | int | str) -> None:
        """Log a parameter with optional units.

        Args:
            name: Parameter name.
            value: Parameter value (DimArray or scalar).
        """
        ...

    def log_metric(
        self,
        name: str,
        value: DimArray | float,
        step: int | None = None,
    ) -> None:
        """Log a metric with optional units.

        Args:
            name: Metric name.
            value: Metric value (DimArray or scalar).
            step: Optional step number for time series.
        """
        ...

    def get_run_data(self, run_id: str) -> dict[str, Any]:
        """Get data for a specific run.

        Args:
            run_id: Run identifier.

        Returns:
            Dictionary containing run data (params, metrics, tags).
        """
        ...

    def list_runs(self, filter_string: str | None = None) -> list[dict[str, Any]]:
        """List all runs for this experiment.

        Args:
            filter_string: Optional filter expression.

        Returns:
            List of run data dictionaries.
        """
        ...


class MLflowBackend:
    """MLflow backend for experiment tracking.

    Uses MLflow tracking server to store experiment data with unit metadata.

    Args:
        experiment_name: Name of the MLflow experiment.
        tracking_uri: Optional MLflow tracking server URI.

    Raises:
        ImportError: If mlflow is not installed.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
    ):
        """Initialize MLflow backend."""
        try:
            import mlflow
            self._mlflow = mlflow
        except ImportError:
            raise ImportError(
                "MLflow is required for experiment tracking. "
                "Install with: pip install mlflow"
            )

        if tracking_uri:
            self._mlflow.set_tracking_uri(tracking_uri)

        self._experiment_name = experiment_name
        self._mlflow.set_experiment(experiment_name)
        self._current_run_id: str | None = None

    def start_run(self, run_name: str | None = None, tags: dict[str, str] | None = None) -> str:
        """Start a new MLflow run."""
        run = self._mlflow.start_run(run_name=run_name, tags=tags)
        self._current_run_id = run.info.run_id
        return self._current_run_id

    def end_run(self) -> None:
        """End the current MLflow run."""
        self._mlflow.end_run()
        self._current_run_id = None

    def log_param(self, name: str, value: DimArray | float | int | str) -> None:
        """Log parameter to MLflow with unit metadata."""
        if isinstance(value, DimArray):
            if value.data.size != 1:
                raise ValueError(
                    f"Expected scalar DimArray for parameter '{name}', "
                    f"got shape {value.shape}"
                )
            scalar_value = float(value.data.item())
            self._mlflow.log_param(name, scalar_value)
            self._mlflow.set_tag(f"unit.{name}", value.unit.symbol)
        else:
            self._mlflow.log_param(name, value)
            if not isinstance(value, str):
                self._mlflow.set_tag(f"unit.{name}", "dimensionless")

    def log_metric(
        self,
        name: str,
        value: DimArray | float,
        step: int | None = None,
    ) -> None:
        """Log metric to MLflow with unit metadata."""
        if isinstance(value, DimArray):
            if value.data.size != 1:
                # Log statistics for arrays
                import numpy as np
                self._mlflow.log_metric(f"{name}_mean", float(np.mean(value.data)), step=step)
                self._mlflow.log_metric(f"{name}_std", float(np.std(value.data)), step=step)
                self._mlflow.log_metric(f"{name}_min", float(np.min(value.data)), step=step)
                self._mlflow.log_metric(f"{name}_max", float(np.max(value.data)), step=step)
                self._mlflow.set_tag(f"shape.{name}", str(value.shape))
            else:
                scalar_value = float(value.data.item())
                self._mlflow.log_metric(name, scalar_value, step=step)

            self._mlflow.set_tag(f"unit.{name}", value.unit.symbol)
        else:
            self._mlflow.log_metric(name, float(value), step=step)
            self._mlflow.set_tag(f"unit.{name}", "dimensionless")

    def get_run_data(self, run_id: str) -> dict[str, Any]:
        """Get data for a specific MLflow run."""
        client = self._mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)

        return {
            "run_id": run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "tags": dict(run.data.tags),
        }

    def list_runs(self, filter_string: str | None = None) -> list[dict[str, Any]]:
        """List all runs in the MLflow experiment."""
        client = self._mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self._experiment_name)

        if experiment is None:
            return []

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string or "",
        )

        return [
            {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
                "tags": dict(run.data.tags),
            }
            for run in runs
        ]


class WandbBackend:
    """Weights & Biases backend for experiment tracking.

    Uses W&B API to store experiment data with unit metadata.

    Args:
        project: W&B project name.
        entity: Optional W&B entity (username or team).

    Raises:
        ImportError: If wandb is not installed.
    """

    def __init__(
        self,
        project: str,
        entity: str | None = None,
    ):
        """Initialize W&B backend."""
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "Weights & Biases is required for experiment tracking. "
                "Install with: pip install wandb"
            )

        self._project = project
        self._entity = entity
        self._current_run: Any = None

    def start_run(self, run_name: str | None = None, tags: dict[str, str] | None = None) -> str:
        """Start a new W&B run."""
        # Convert tags dict to list for W&B
        tag_list = [f"{k}:{v}" for k, v in (tags or {}).items()]

        self._current_run = self._wandb.init(
            project=self._project,
            entity=self._entity,
            name=run_name,
            tags=tag_list,
            reinit=True,
        )
        return self._current_run.id

    def end_run(self) -> None:
        """End the current W&B run."""
        if self._current_run:
            self._current_run.finish()
            self._current_run = None

    def log_param(self, name: str, value: DimArray | float | int | str) -> None:
        """Log parameter to W&B with unit metadata."""
        if not self._current_run:
            raise RuntimeError("No active run. Call start_run() first.")

        if isinstance(value, DimArray):
            if value.data.size != 1:
                raise ValueError(
                    f"Expected scalar DimArray for parameter '{name}', "
                    f"got shape {value.shape}"
                )
            scalar_value = float(value.data.item())
            self._wandb.config.update({name: scalar_value, f"{name}_unit": value.unit.symbol})
        else:
            config_update = {name: value}
            if not isinstance(value, str):
                config_update[f"{name}_unit"] = "dimensionless"
            self._wandb.config.update(config_update)

    def log_metric(
        self,
        name: str,
        value: DimArray | float,
        step: int | None = None,
    ) -> None:
        """Log metric to W&B with unit metadata."""
        if not self._current_run:
            raise RuntimeError("No active run. Call start_run() first.")

        log_data: dict[str, Any] = {}

        if isinstance(value, DimArray):
            if value.data.size != 1:
                # Log statistics for arrays
                import numpy as np
                log_data[f"{name}_mean"] = float(np.mean(value.data))
                log_data[f"{name}_std"] = float(np.std(value.data))
                log_data[f"{name}_min"] = float(np.min(value.data))
                log_data[f"{name}_max"] = float(np.max(value.data))
                # Store unit in run summary
                self._current_run.summary[f"{name}_unit"] = value.unit.symbol
                self._current_run.summary[f"{name}_shape"] = str(value.shape)
            else:
                scalar_value = float(value.data.item())
                log_data[name] = scalar_value
                # Store unit in run summary (only once)
                if f"{name}_unit" not in self._current_run.summary:
                    self._current_run.summary[f"{name}_unit"] = value.unit.symbol
        else:
            log_data[name] = float(value)
            if f"{name}_unit" not in self._current_run.summary:
                self._current_run.summary[f"{name}_unit"] = "dimensionless"

        self._wandb.log(log_data, step=step)

    def get_run_data(self, run_id: str) -> dict[str, Any]:
        """Get data for a specific W&B run."""
        api = self._wandb.Api()
        path = f"{self._entity}/{self._project}/{run_id}" if self._entity else f"{self._project}/{run_id}"
        run = api.run(path)

        return {
            "run_id": run.id,
            "run_name": run.name,
            "status": run.state,
            "start_time": run.created_at,
            "end_time": run.updated_at,
            "params": dict(run.config),
            "metrics": {k: v for k, v in run.summary.items() if not k.endswith("_unit")},
            "tags": dict(run.tags) if hasattr(run, "tags") else {},
        }

    def list_runs(self, filter_string: str | None = None) -> list[dict[str, Any]]:
        """List all runs in the W&B project."""
        api = self._wandb.Api()
        path = f"{self._entity}/{self._project}" if self._entity else self._project
        runs = api.runs(path, filters=filter_string)

        return [
            {
                "run_id": run.id,
                "run_name": run.name,
                "status": run.state,
                "start_time": run.created_at,
                "end_time": run.updated_at,
                "params": dict(run.config),
                "metrics": {k: v for k, v in run.summary.items() if not k.endswith("_unit")},
                "tags": dict(run.tags) if hasattr(run, "tags") else {},
            }
            for run in runs
        ]


__all__ = [
    "ExperimentBackend",
    "MLflowBackend",
    "WandbBackend",
]

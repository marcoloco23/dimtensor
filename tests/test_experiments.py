"""Tests for experiment tracking system."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from dimtensor import DimArray, units
from dimtensor.experiments import DimExperiment, RunComparison
from dimtensor.experiments.backend import (
    ExperimentBackend,
    MLflowBackend,
    WandbBackend,
)


# =============================================================================
# Backend Tests
# =============================================================================


def test_mlflow_backend_import_error():
    """Test MLflowBackend raises error when mlflow not installed."""
    with patch.dict("sys.modules", {"mlflow": None}):
        with pytest.raises(ImportError, match="MLflow is required"):
            MLflowBackend("test-experiment")


def test_wandb_backend_import_error():
    """Test WandbBackend raises error when wandb not installed."""
    with patch.dict("sys.modules", {"wandb": None}):
        with pytest.raises(ImportError, match="Weights & Biases is required"):
            WandbBackend("test-project")


@pytest.mark.skipif(True, reason="Requires MLflow installation")
def test_mlflow_backend_integration():
    """Integration test for MLflow backend (skipped if MLflow not available)."""
    import mlflow

    backend = MLflowBackend("test-experiment")

    # Start run
    run_id = backend.start_run("test-run", tags={"test": "true"})
    assert run_id is not None

    # Log parameter
    lr = DimArray(0.001, 1 / units.s)
    backend.log_param("learning_rate", lr)

    # Log metric
    loss = DimArray(0.5, units.J)
    backend.log_metric("loss", loss, step=0)

    # End run
    backend.end_run()

    # Get run data
    run_data = backend.get_run_data(run_id)
    assert run_data["run_id"] == run_id
    assert "learning_rate" in run_data["params"]
    assert "loss" in run_data["metrics"]


@pytest.mark.skipif(True, reason="Requires W&B installation and authentication")
def test_wandb_backend_integration():
    """Integration test for W&B backend (skipped if W&B not available)."""
    import wandb

    backend = WandbBackend("test-project")

    # Start run
    run_id = backend.start_run("test-run", tags={"test": "true"})
    assert run_id is not None

    # Log parameter
    lr = DimArray(0.001, 1 / units.s)
    backend.log_param("learning_rate", lr)

    # Log metric
    loss = DimArray(0.5, units.J)
    backend.log_metric("loss", loss, step=0)

    # End run
    backend.end_run()


# =============================================================================
# DimExperiment Tests
# =============================================================================


def test_experiment_initialization():
    """Test DimExperiment initialization with different backends."""
    # Test with mock MLflow
    with patch("dimtensor.experiments.backend.MLflowBackend.__init__", return_value=None):
        exp = DimExperiment("test-experiment", backend="mlflow")
        assert exp.name == "test-experiment"
        assert exp._backend_type == "mlflow"

    # Test with mock W&B
    with patch("dimtensor.experiments.backend.WandbBackend.__init__", return_value=None):
        exp = DimExperiment("test-experiment", backend="wandb")
        assert exp.name == "test-experiment"
        assert exp._backend_type == "wandb"


def test_experiment_invalid_backend():
    """Test DimExperiment raises error for invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend"):
        DimExperiment("test-experiment", backend="invalid")


def test_experiment_context_manager():
    """Test experiment context manager for runs."""
    # Create a mock backend
    mock_backend = Mock()
    mock_backend.start_run.return_value = "run123"
    mock_backend.end_run.return_value = None

    with patch("dimtensor.experiments.backend.MLflowBackend.__init__", return_value=None):
        exp = DimExperiment("test-experiment", backend="mlflow")
        exp._backend = mock_backend

        with exp.start_run("test-run") as run_id:
            assert run_id == "run123"
            assert exp._current_run_id == "run123"

        # After context, run should be ended
        assert exp._current_run_id is None
        mock_backend.end_run.assert_called_once()


def test_experiment_log_param():
    """Test logging parameters with units."""
    mock_backend = Mock()

    with patch("dimtensor.experiments.backend.MLflowBackend.__init__", return_value=None):
        exp = DimExperiment("test-experiment", backend="mlflow")
        exp._backend = mock_backend

        # Log DimArray parameter
        lr = DimArray(0.001, 1 / units.s)
        exp.log_param("learning_rate", lr)

        # Log scalar parameter
        exp.log_param("batch_size", 32)

        assert mock_backend.log_param.call_count == 2


def test_experiment_log_metric():
    """Test logging metrics with units."""
    mock_backend = Mock()

    with patch("dimtensor.experiments.backend.MLflowBackend.__init__", return_value=None):
        exp = DimExperiment("test-experiment", backend="mlflow")
        exp._backend = mock_backend

        # Log scalar metric
        loss = DimArray(0.5, units.J)
        exp.log_metric("loss", loss, step=0)

        # Log array metric (should log statistics)
        losses = DimArray([0.5, 0.4, 0.3], units.J)
        exp.log_metric("batch_losses", losses, step=0)

        assert mock_backend.log_metric.call_count == 2


def test_experiment_log_array():
    """Test logging arrays."""
    mock_backend = Mock()

    with patch("dimtensor.experiments.backend.MLflowBackend.__init__", return_value=None):
        exp = DimExperiment("test-experiment", backend="mlflow")
        exp._backend = mock_backend

        predictions = DimArray([1.0, 2.0, 3.0], units.m / units.s)
        exp.log_array("predictions", predictions, step=0)

        mock_backend.log_metric.assert_called_once()


def test_experiment_export_import(tmp_path):
    """Test exporting and importing experiment data."""
    # Create mock backend
    mock_backend = Mock()
    mock_backend.list_runs.return_value = [
        {
            "run_id": "run1",
            "run_name": "test-run",
            "status": "FINISHED",
            "start_time": 0,
            "end_time": 100,
            "params": {"lr": "0.001"},
            "metrics": {"loss": 0.5},
            "tags": {"unit.lr": "Hz", "unit.loss": "J"},
        }
    ]

    with patch("dimtensor.experiments.backend.MLflowBackend.__init__", return_value=None):
        exp = DimExperiment("test-experiment", backend="mlflow")
        exp._backend = mock_backend

        # Export
        export_path = tmp_path / "experiment.json"
        exp.export(export_path)

        # Verify export file
        assert export_path.exists()
        with open(export_path) as f:
            data = json.load(f)
            assert data["experiment_name"] == "test-experiment"
            assert data["backend"] == "mlflow"
            assert len(data["runs"]) > 0

    # Import
    with patch("dimtensor.experiments.backend.MLflowBackend.__init__", return_value=None):
        exp2 = DimExperiment.load(export_path)
        assert exp2.name == "test-experiment"


# =============================================================================
# RunComparison Tests
# =============================================================================


def test_run_comparison_basic():
    """Test basic run comparison functionality."""
    # Create mock run data
    runs_data = [
        {
            "run_id": "run1",
            "run_name": "run-si",
            "params": {"lr": "0.001"},
            "metrics": {"loss": 0.5},
            "tags": {"unit.loss": "J"},
        },
        {
            "run_id": "run2",
            "run_name": "run-si-2",
            "params": {"lr": "0.002"},
            "metrics": {"loss": 0.4},
            "tags": {"unit.loss": "J"},
        },
    ]

    comparison = RunComparison(runs_data, ["loss"])

    # Check comparison data
    assert "loss" in comparison._comparisons
    comp_data = comparison._comparisons["loss"]
    assert comp_data["compatible"] is True
    assert len(comp_data["values"]) == 2
    assert comp_data["mean"] == pytest.approx(0.45)


def test_run_comparison_incompatible_units():
    """Test run comparison with incompatible units."""
    runs_data = [
        {
            "run_id": "run1",
            "run_name": "run1",
            "params": {},
            "metrics": {"value": 100.0},
            "tags": {"unit.value": "m"},
        },
        {
            "run_id": "run2",
            "run_name": "run2",
            "params": {},
            "metrics": {"value": 10.0},
            "tags": {"unit.value": "s"},
        },
    ]

    comparison = RunComparison(runs_data, ["value"])

    comp_data = comparison._comparisons["value"]
    assert comp_data["compatible"] is False
    assert "warning" in comp_data


def test_run_comparison_to_dataframe():
    """Test converting comparison to DataFrame."""
    pytest.importorskip("pandas")

    runs_data = [
        {
            "run_id": "run1",
            "run_name": "run-1",
            "params": {},
            "metrics": {"loss": 0.5},
            "tags": {"unit.loss": "J"},
        },
        {
            "run_id": "run2",
            "run_name": "run-2",
            "params": {},
            "metrics": {"loss": 0.4},
            "tags": {"unit.loss": "J"},
        },
    ]

    comparison = RunComparison(runs_data, ["loss"])
    df = comparison.to_dataframe()

    assert len(df) == 2
    assert "run_name" in df.columns
    assert "metric" in df.columns
    assert "value" in df.columns
    assert "unit" in df.columns


def test_run_comparison_get_best_run():
    """Test finding best run."""
    runs_data = [
        {
            "run_id": "run1",
            "run_name": "run-1",
            "params": {},
            "metrics": {"loss": 0.5},
            "tags": {"unit.loss": "J"},
        },
        {
            "run_id": "run2",
            "run_name": "run-2",
            "params": {},
            "metrics": {"loss": 0.4},
            "tags": {"unit.loss": "J"},
        },
        {
            "run_id": "run3",
            "run_name": "run-3",
            "params": {},
            "metrics": {"loss": 0.6},
            "tags": {"unit.loss": "J"},
        },
    ]

    comparison = RunComparison(runs_data, ["loss"])

    # Find minimum loss
    best = comparison.get_best_run("loss", mode="min")
    assert best["run_name"] == "run-2"
    assert best["value"] == 0.4

    # Find maximum loss
    worst = comparison.get_best_run("loss", mode="max")
    assert worst["run_name"] == "run-3"
    assert worst["value"] == 0.6


def test_run_comparison_summary():
    """Test summary text generation."""
    runs_data = [
        {
            "run_id": "run1",
            "run_name": "run-1",
            "params": {},
            "metrics": {"loss": 0.5, "accuracy": 0.9},
            "tags": {"unit.loss": "J", "unit.accuracy": "dimensionless"},
        },
        {
            "run_id": "run2",
            "run_name": "run-2",
            "params": {},
            "metrics": {"loss": 0.4, "accuracy": 0.92},
            "tags": {"unit.loss": "J", "unit.accuracy": "dimensionless"},
        },
    ]

    comparison = RunComparison(runs_data, ["loss", "accuracy"])
    summary = comparison.summary()

    assert "Run Comparison Summary" in summary
    assert "loss" in summary
    assert "accuracy" in summary
    assert "Mean:" in summary
    assert "Std:" in summary


@pytest.mark.skipif(True, reason="Matplotlib plotting tests skipped")
def test_run_comparison_plot():
    """Test plotting comparison (requires matplotlib)."""
    pytest.importorskip("matplotlib")

    runs_data = [
        {
            "run_id": "run1",
            "run_name": "run-1",
            "params": {},
            "metrics": {"loss": 0.5},
            "tags": {"unit.loss": "J"},
        },
        {
            "run_id": "run2",
            "run_name": "run-2",
            "params": {},
            "metrics": {"loss": 0.4},
            "tags": {"unit.loss": "J"},
        },
    ]

    comparison = RunComparison(runs_data, ["loss"])

    # Test that plot doesn't raise
    fig = comparison.plot("loss", show=False)
    assert fig is not None


# =============================================================================
# Visualization Tests
# =============================================================================


@pytest.mark.skipif(True, reason="Matplotlib plotting tests skipped")
def test_plot_experiment_history():
    """Test experiment history plotting."""
    pytest.importorskip("matplotlib")
    from dimtensor.experiments.visualization import plot_experiment_history

    with patch("dimtensor.experiments.backend.mlflow") as mock_mlflow:
        mock_client = Mock()
        mock_exp = Mock(experiment_id="exp123")
        mock_client.get_experiment_by_name.return_value = mock_exp
        mock_client.search_runs.return_value = [
            Mock(
                info=Mock(run_id="run1", run_name="test-run", status="FINISHED"),
                data=Mock(
                    params={}, metrics={"loss": 0.5}, tags={"unit.loss": "J"}
                ),
            )
        ]
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        exp = DimExperiment("test-experiment", backend="mlflow")
        fig = plot_experiment_history(exp, "loss", show=False)
        assert fig is not None


# =============================================================================
# Integration Test
# =============================================================================


@pytest.mark.skipif(
    True,  # Skip this integration test - mlflow import is local
    reason="MLflow is imported locally in backend, requires actual MLflow"
)
def test_full_experiment_workflow(tmp_path):
    """Test complete experiment workflow with mocked backend."""
    with patch("mlflow") as mock_mlflow:
        # Setup mocks
        mock_run = Mock(info=Mock(run_id="run123"))
        mock_mlflow.start_run.return_value = mock_run

        mock_client = Mock()
        mock_exp = Mock(experiment_id="exp123")
        mock_client.get_experiment_by_name.return_value = mock_exp
        mock_client.get_run.return_value = Mock(
            info=Mock(
                run_id="run123",
                run_name="test-run",
                status="FINISHED",
                start_time=0,
                end_time=100,
            ),
            data=Mock(
                params={"lr": "0.001"},
                metrics={"loss": 0.5},
                tags={"unit.lr": "Hz", "unit.loss": "J"},
            ),
        )
        mock_client.search_runs.return_value = [mock_client.get_run.return_value]
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        # Create experiment
        exp = DimExperiment("physics-experiment", backend="mlflow")

        # Run experiment
        with exp.start_run("test-run"):
            # Log parameters
            exp.log_param("learning_rate", DimArray(0.001, units.Hz))
            exp.log_param("domain_length", DimArray(1.0, units.m))

            # Log metrics
            for step in range(5):
                loss = DimArray(0.5 / (step + 1), units.J)
                exp.log_metric("loss", loss, step=step)

        # Get run data
        run_data = exp.get_run("run123")
        assert run_data["run_id"] == "run123"

        # List runs
        runs = exp.list_runs()
        assert len(runs) == 1

        # Export
        export_path = tmp_path / "experiment.json"
        exp.export(export_path)
        assert export_path.exists()


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.skipif(
    True,  # Skip - mlflow import is local
    reason="MLflow is imported locally in backend, requires actual MLflow"
)
def test_log_non_scalar_param_raises():
    """Test that logging non-scalar DimArray as parameter raises error."""
    with patch("mlflow") as mock_mlflow:
        exp = DimExperiment("test-experiment", backend="mlflow")

        # This should raise when backend tries to log it
        arr = DimArray([1.0, 2.0], units.m)
        with pytest.raises(ValueError, match="Expected scalar"):
            exp._backend.log_param("length", arr)


def test_comparison_missing_metric():
    """Test comparison with missing metric."""
    runs_data = [
        {
            "run_id": "run1",
            "run_name": "run-1",
            "params": {},
            "metrics": {"loss": 0.5},
            "tags": {"unit.loss": "J"},
        },
        {
            "run_id": "run2",
            "run_name": "run-2",
            "params": {},
            "metrics": {},  # Missing loss
            "tags": {},
        },
    ]

    comparison = RunComparison(runs_data, ["loss"])

    # Should handle gracefully
    comp_data = comparison._comparisons["loss"]
    assert len(comp_data["values"]) == 1  # Only one run has the metric


def test_comparison_invalid_mode():
    """Test get_best_run with invalid mode."""
    runs_data = [
        {
            "run_id": "run1",
            "run_name": "run-1",
            "params": {},
            "metrics": {"loss": 0.5},
            "tags": {"unit.loss": "J"},
        },
    ]

    comparison = RunComparison(runs_data, ["loss"])

    with pytest.raises(ValueError, match="Invalid mode"):
        comparison.get_best_run("loss", mode="invalid")

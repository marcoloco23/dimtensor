"""Simplified tests for model package functionality."""

import json
import tempfile
from pathlib import Path

import pytest

# Import only what we need to avoid dependency issues
from dimtensor.core.dimarray import DimArray
from dimtensor.core.dimensions import Dimension
from dimtensor.core.units import Unit, meter, second, kilogram
from dimtensor.hub.package import DimModelPackage
from dimtensor.hub.cards import ModelCard, ModelInfo, TrainingInfo
from dimtensor.hub.validators import DimModelWrapper, validate_model_io
from dimtensor.errors import DimensionError


# Skip tests if torch not available
torch = pytest.importorskip("torch", reason="PyTorch not installed")
import torch.nn as nn


class SimpleDimModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def simple_model():
    """Create a simple PyTorch model."""
    return SimpleDimModel()


@pytest.fixture
def model_card():
    """Create a model card for testing."""
    info = ModelInfo(
        name="test-model",
        version="1.0.0",
        description="Test model for unit tests",
        input_dims={
            "velocity": Dimension(length=1, time=-1),
        },
        output_dims={
            "force": Dimension(length=1, mass=1, time=-2),
        },
        domain="mechanics",
        architecture="SimpleDimModel",
    )
    training = TrainingInfo(
        dataset="test-dataset",
        epochs=10,
        batch_size=32,
    )
    return ModelCard(info=info, training=training)


def test_package_init(simple_model, model_card):
    """Test package initialization."""
    package = DimModelPackage(simple_model, model_card)

    assert package.model is simple_model
    assert package.card is model_card
    assert package.framework == "pytorch"
    assert package.auto_convert is False


def test_save_pytorch(simple_model, model_card):
    """Test saving PyTorch model."""
    package = DimModelPackage(simple_model, model_card, framework="pytorch")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test-model"
        package.save(save_path)

        # Check files created
        assert (save_path / "model_card.json").exists()
        assert (save_path / "weights.pt").exists()
        assert (save_path / "config.json").exists()
        assert (save_path / "architecture.json").exists()

        # Check config contents
        with open(save_path / "config.json") as f:
            config = json.load(f)
        assert config["framework"] == "pytorch"
        assert config["auto_convert"] is False


def test_load_pytorch(simple_model, model_card):
    """Test loading PyTorch model."""
    package = DimModelPackage(simple_model, model_card, framework="pytorch")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test-model"
        package.save(save_path)

        # Load package
        loaded = DimModelPackage.load(save_path)

        assert loaded.framework == "pytorch"
        assert loaded.card.info.name == "test-model"
        assert loaded.card.info.version == "1.0.0"


def test_validate_inputs_correct(simple_model, model_card):
    """Test input validation with correct dimensions."""
    package = DimModelPackage(simple_model, model_card)

    # Create velocity with correct dimension (L/T)
    velocity = DimArray([1.0, 2.0], meter / second)
    inputs = {"velocity": velocity}

    # Should not raise
    package.validate_inputs(inputs)


def test_validate_inputs_wrong_dimension(simple_model, model_card):
    """Test input validation with wrong dimensions."""
    package = DimModelPackage(simple_model, model_card)

    # Wrong dimension (length instead of velocity)
    position = DimArray([1.0, 2.0], meter)
    inputs = {"velocity": position}

    with pytest.raises(DimensionError) as exc_info:
        package.validate_inputs(inputs)

    assert "wrong dimension" in str(exc_info.value).lower()
    assert "velocity" in str(exc_info.value)


def test_validate_inputs_missing(simple_model, model_card):
    """Test input validation with missing input."""
    package = DimModelPackage(simple_model, model_card)

    inputs = {}

    with pytest.raises(DimensionError) as exc_info:
        package.validate_inputs(inputs)

    assert "missing" in str(exc_info.value).lower()


def test_validate_outputs_correct(simple_model, model_card):
    """Test output validation with correct dimensions."""
    package = DimModelPackage(simple_model, model_card)

    # Force has dimension L*M/T^2
    force = DimArray([1.0], kilogram * meter / (second**2))
    outputs = {"force": force}

    # Should not raise
    package.validate_outputs(outputs)


def test_validate_outputs_wrong_dimension(simple_model, model_card):
    """Test output validation with wrong dimensions."""
    package = DimModelPackage(simple_model, model_card)

    # Wrong dimension (energy L^2*M/T^2 instead of force L*M/T^2)
    energy = DimArray([1.0], kilogram * (meter**2) / (second**2))
    outputs = {"force": energy}

    with pytest.raises(DimensionError) as exc_info:
        package.validate_outputs(outputs)

    assert "wrong dimension" in str(exc_info.value).lower()


def test_wrapper_init(simple_model, model_card):
    """Test wrapper initialization."""
    wrapper = DimModelWrapper(simple_model, model_card)

    assert wrapper.model is simple_model
    assert wrapper.card is model_card
    assert wrapper.auto_convert is False
    assert wrapper.strict is True


def test_wrapper_call(model_card):
    """Test calling wrapper with validation."""

    def mock_model(velocity):
        force = DimArray([1.0], kilogram * meter / (second**2))
        return {"force": force}

    wrapper = DimModelWrapper(mock_model, model_card)

    velocity = DimArray([1.0, 2.0], meter / second)
    output = wrapper(velocity=velocity)

    assert "force" in output


def test_validate_model_io_valid(simple_model, model_card):
    """Test validate_model_io with correct inputs."""
    velocity = DimArray([1.0, 2.0], meter / second)
    inputs = {"velocity": velocity}

    is_valid, errors = validate_model_io(simple_model, model_card, inputs)

    assert is_valid
    assert len(errors) == 0


def test_validate_model_io_invalid(simple_model, model_card):
    """Test validate_model_io with incorrect inputs."""
    position = DimArray([1.0, 2.0], meter)
    inputs = {"velocity": position}

    is_valid, errors = validate_model_io(simple_model, model_card, inputs)

    assert not is_valid
    assert len(errors) > 0


def test_save_load_roundtrip(simple_model, model_card):
    """Test save/load preserves metadata."""
    package = DimModelPackage(simple_model, model_card)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test-model"
        package.save(save_path)
        loaded = DimModelPackage.load(save_path)

        # Check model card preserved
        assert loaded.card.info.name == model_card.info.name
        assert loaded.card.info.description == model_card.info.description
        assert loaded.card.training.dataset == model_card.training.dataset


def test_numpy_package(model_card):
    """Test save/load with numpy model."""
    import numpy as np

    # Use a simple numpy array instead of a function (functions can't be pickled)
    model_data = {"weights": np.array([1.0, 2.0]), "bias": np.array(0.5)}

    package = DimModelPackage(model_data, model_card, framework="numpy")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "numpy-model"
        package.save(save_path)

        assert (save_path / "model.pkl").exists()

        loaded = DimModelPackage.load(save_path)
        assert loaded.framework == "numpy"
        assert "weights" in loaded.model
        assert "bias" in loaded.model


def test_load_nonexistent_package():
    """Test loading non-existent package."""
    with pytest.raises(FileNotFoundError):
        DimModelPackage.load("/nonexistent/path")


def test_unsupported_framework(model_card):
    """Test saving with unsupported framework."""
    package = DimModelPackage(None, model_card, framework="tensorflow")  # type: ignore

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test-model"

        with pytest.raises(ValueError) as exc_info:
            package.save(save_path)

        assert "unknown framework" in str(exc_info.value).lower()

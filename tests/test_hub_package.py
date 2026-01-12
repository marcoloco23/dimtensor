"""Tests for model package save/load functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from dimtensor import DimArray
from dimtensor.core.dimensions import Dimension
from dimtensor.core.units import Unit, meter, second
from dimtensor.hub import (
    DimModelPackage,
    DimModelWrapper,
    ModelCard,
    ModelInfo,
    TrainingInfo,
    validate_model_io,
    create_validator,
)
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


class TestDimModelPackage:
    """Tests for DimModelPackage."""

    def test_init(self, simple_model, model_card):
        """Test package initialization."""
        package = DimModelPackage(simple_model, model_card)

        assert package.model is simple_model
        assert package.card is model_card
        assert package.framework == "pytorch"
        assert package.auto_convert is False

    def test_save_pytorch(self, simple_model, model_card):
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
            assert "version" in config

    def test_load_pytorch(self, simple_model, model_card):
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
            assert "state_dict" in loaded.model

    def test_save_load_roundtrip(self, simple_model, model_card):
        """Test save/load preserves model card."""
        package = DimModelPackage(simple_model, model_card)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test-model"
            package.save(save_path)
            loaded = DimModelPackage.load(save_path)

            # Check model card preserved
            assert loaded.card.info.name == model_card.info.name
            assert loaded.card.info.description == model_card.info.description
            assert loaded.card.training.dataset == model_card.training.dataset
            assert loaded.card.training.epochs == model_card.training.epochs

    def test_validate_inputs_correct(self, simple_model, model_card):
        """Test input validation with correct dimensions."""
        package = DimModelPackage(simple_model, model_card)

        velocity = DimArray([1.0, 2.0], meter / second)
        inputs = {"velocity": velocity}

        # Should not raise
        package.validate_inputs(inputs)

    def test_validate_inputs_wrong_dimension(self, simple_model, model_card):
        """Test input validation with wrong dimensions."""
        package = DimModelPackage(simple_model, model_card)

        # Wrong dimension (length instead of velocity)
        position = DimArray([1.0, 2.0], meter)
        inputs = {"velocity": position}

        with pytest.raises(DimensionError) as exc_info:
            package.validate_inputs(inputs)

        assert "wrong dimension" in str(exc_info.value).lower()
        assert "velocity" in str(exc_info.value)

    def test_validate_inputs_missing(self, simple_model, model_card):
        """Test input validation with missing input."""
        package = DimModelPackage(simple_model, model_card)

        inputs = {}

        with pytest.raises(DimensionError) as exc_info:
            package.validate_inputs(inputs)

        assert "missing" in str(exc_info.value).lower()
        assert "velocity" in str(exc_info.value)

    def test_validate_inputs_non_strict(self, simple_model, model_card):
        """Test input validation in non-strict mode."""
        package = DimModelPackage(simple_model, model_card)

        # Wrong dimension
        position = DimArray([1.0, 2.0], meter)
        inputs = {"velocity": position}

        # Should not raise in non-strict mode
        package.validate_inputs(inputs, strict=False)

    def test_validate_outputs_correct(self, simple_model, model_card):
        """Test output validation with correct dimensions."""
        package = DimModelPackage(simple_model, model_card)

        force_unit = Unit(Dimension(length=1, mass=1, time=-2), 1.0)
        force = DimArray([1.0], force_unit)
        outputs = {"force": force}

        # Should not raise
        package.validate_outputs(outputs)

    def test_validate_outputs_wrong_dimension(self, simple_model, model_card):
        """Test output validation with wrong dimensions."""
        package = DimModelPackage(simple_model, model_card)

        # Wrong dimension
        energy = DimArray([1.0], Unit(Dimension(length=2, mass=1, time=-2), 1.0))
        outputs = {"force": energy}

        with pytest.raises(DimensionError) as exc_info:
            package.validate_outputs(outputs)

        assert "wrong dimension" in str(exc_info.value).lower()
        assert "force" in str(exc_info.value)

    def test_repr(self, simple_model, model_card):
        """Test string representation."""
        package = DimModelPackage(simple_model, model_card)
        repr_str = repr(package)

        assert "DimModelPackage" in repr_str
        assert "test-model" in repr_str
        assert "1.0.0" in repr_str
        assert "pytorch" in repr_str


class TestDimModelWrapper:
    """Tests for DimModelWrapper."""

    def test_init(self, simple_model, model_card):
        """Test wrapper initialization."""
        wrapper = DimModelWrapper(simple_model, model_card)

        assert wrapper.model is simple_model
        assert wrapper.card is model_card
        assert wrapper.auto_convert is False
        assert wrapper.strict is True

    def test_call_with_validation(self, model_card):
        """Test calling wrapper with validation."""

        def mock_model(velocity):
            # Return dict with force output
            force_unit = Unit(Dimension(length=1, mass=1, time=-2), 1.0)
            return {"force": DimArray([1.0], force_unit)}

        wrapper = DimModelWrapper(mock_model, model_card)

        velocity = DimArray([1.0, 2.0], meter / second)
        output = wrapper(velocity=velocity)

        assert "force" in output

    def test_call_validation_fails(self, model_card):
        """Test calling wrapper with invalid input."""

        def mock_model(velocity):
            return {"force": DimArray([1.0], meter)}

        wrapper = DimModelWrapper(mock_model, model_card)

        # Wrong dimension
        position = DimArray([1.0, 2.0], meter)

        with pytest.raises(DimensionError):
            wrapper(velocity=position)

    def test_forward_alias(self, model_card):
        """Test forward() alias."""

        def mock_model(velocity):
            force_unit = Unit(Dimension(length=1, mass=1, time=-2), 1.0)
            return {"force": DimArray([1.0], force_unit)}

        wrapper = DimModelWrapper(mock_model, model_card)

        velocity = DimArray([1.0, 2.0], meter / second)
        output = wrapper.forward(velocity=velocity)

        assert "force" in output

    def test_non_strict_mode(self, model_card):
        """Test wrapper in non-strict mode."""

        def mock_model(velocity):
            return {"force": DimArray([1.0], meter)}

        wrapper = DimModelWrapper(mock_model, model_card, strict=False)

        # Wrong dimension - should warn but not raise
        position = DimArray([1.0, 2.0], meter)
        wrapper(velocity=position)

    def test_repr(self, simple_model, model_card):
        """Test string representation."""
        wrapper = DimModelWrapper(simple_model, model_card)
        repr_str = repr(wrapper)

        assert "DimModelWrapper" in repr_str
        assert "test-model" in repr_str
        assert "strict=True" in repr_str


class TestValidateModelIO:
    """Tests for validate_model_io function."""

    def test_valid_inputs(self, simple_model, model_card):
        """Test validation with correct inputs."""
        velocity = DimArray([1.0, 2.0], meter / second)
        inputs = {"velocity": velocity}

        is_valid, errors = validate_model_io(simple_model, model_card, inputs)

        assert is_valid
        assert len(errors) == 0

    def test_invalid_inputs(self, simple_model, model_card):
        """Test validation with incorrect inputs."""
        position = DimArray([1.0, 2.0], meter)
        inputs = {"velocity": position}

        is_valid, errors = validate_model_io(simple_model, model_card, inputs)

        assert not is_valid
        assert len(errors) > 0
        assert "velocity" in errors[0]

    def test_missing_inputs(self, simple_model, model_card):
        """Test validation with missing inputs."""
        inputs = {}

        is_valid, errors = validate_model_io(simple_model, model_card, inputs)

        assert not is_valid
        assert len(errors) > 0
        assert "missing" in errors[0].lower()

    def test_valid_outputs(self, simple_model, model_card):
        """Test validation with correct outputs."""
        velocity = DimArray([1.0, 2.0], meter / second)
        inputs = {"velocity": velocity}

        force_unit = Unit(Dimension(length=1, mass=1, time=-2), 1.0)
        force = DimArray([1.0], force_unit)
        outputs = {"force": force}

        is_valid, errors = validate_model_io(
            simple_model, model_card, inputs, outputs
        )

        assert is_valid
        assert len(errors) == 0

    def test_invalid_outputs(self, simple_model, model_card):
        """Test validation with incorrect outputs."""
        velocity = DimArray([1.0, 2.0], meter / second)
        inputs = {"velocity": velocity}

        # Wrong dimension
        energy = DimArray([1.0], Unit(Dimension(length=2, mass=1, time=-2), 1.0))
        outputs = {"force": energy}

        is_valid, errors = validate_model_io(
            simple_model, model_card, inputs, outputs
        )

        assert not is_valid
        assert len(errors) > 0
        assert "force" in errors[0]


class TestCreateValidator:
    """Tests for create_validator decorator."""

    def test_decorator(self, model_card):
        """Test validator decorator."""

        @create_validator(model_card)
        def my_model(velocity):
            force_unit = Unit(Dimension(length=1, mass=1, time=-2), 1.0)
            return {"force": DimArray([1.0], force_unit)}

        velocity = DimArray([1.0, 2.0], meter / second)
        output = my_model(velocity=velocity)

        assert "force" in output

    def test_decorator_validation_fails(self, model_card):
        """Test decorator with invalid input."""

        @create_validator(model_card)
        def my_model(velocity):
            return {"force": DimArray([1.0], meter)}

        # Wrong dimension
        position = DimArray([1.0, 2.0], meter)

        with pytest.raises(DimensionError):
            my_model(velocity=position)

    def test_decorator_non_strict(self, model_card):
        """Test decorator in non-strict mode."""

        @create_validator(model_card, strict=False)
        def my_model(velocity):
            return {"force": DimArray([1.0], meter)}

        # Wrong dimension - should warn but not raise
        position = DimArray([1.0, 2.0], meter)
        my_model(velocity=position)


class TestNumpyPackage:
    """Tests for numpy model packages."""

    def test_save_load_numpy(self, model_card):
        """Test save/load with numpy model."""

        def numpy_model(x):
            return x * 2

        package = DimModelPackage(numpy_model, model_card, framework="numpy")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "numpy-model"
            package.save(save_path)

            # Check files created
            assert (save_path / "model.pkl").exists()
            assert (save_path / "config.json").exists()

            # Load
            loaded = DimModelPackage.load(save_path)
            assert loaded.framework == "numpy"


class TestJAXPackage:
    """Tests for JAX model packages."""

    def test_save_load_jax(self, model_card):
        """Test save/load with JAX model."""
        jax = pytest.importorskip("jax", reason="JAX not installed")
        pytest.importorskip("flax", reason="Flax not installed")
        import jax.numpy as jnp

        # Simple JAX pytree
        params = {"weights": jnp.array([1.0, 2.0]), "bias": jnp.array(0.5)}

        package = DimModelPackage(params, model_card, framework="jax")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "jax-model"
            package.save(save_path)

            # Check files created
            assert (save_path / "weights.msgpack").exists()
            assert (save_path / "config.json").exists()

            # Load
            loaded = DimModelPackage.load(save_path)
            assert loaded.framework == "jax"


class TestPackageErrors:
    """Tests for error handling."""

    def test_load_nonexistent_package(self):
        """Test loading non-existent package."""
        with pytest.raises(FileNotFoundError):
            DimModelPackage.load("/nonexistent/path")

    def test_load_missing_config(self):
        """Test loading package with missing config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            package_path = Path(tmpdir) / "broken-package"
            package_path.mkdir()

            # Create model card but no config
            card = ModelCard(
                info=ModelInfo(name="test", version="1.0.0")
            )
            from dimtensor.hub.cards import save_model_card

            save_model_card(card, package_path / "model_card.json")

            with pytest.raises(ValueError) as exc_info:
                DimModelPackage.load(package_path)

            assert "missing config.json" in str(exc_info.value).lower()

    def test_unsupported_framework(self, model_card):
        """Test saving with unsupported framework."""
        package = DimModelPackage(None, model_card, framework="tensorflow")  # type: ignore

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test-model"

            with pytest.raises(ValueError) as exc_info:
                package.save(save_path)

            assert "unknown framework" in str(exc_info.value).lower()

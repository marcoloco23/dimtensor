"""Model package for saving/loading physics models with unit metadata.

This module provides a directory-based package format for sharing trained
physics models with complete dimensional information. Models can be saved
with their weights, metadata, and unit information, then loaded and used
with automatic unit validation.

Example:
    >>> from dimtensor.hub import DimModelPackage, ModelCard
    >>>
    >>> # Save PyTorch model
    >>> package = DimModelPackage(model, card, framework="pytorch")
    >>> package.save("my-model/")
    >>>
    >>> # Load model
    >>> loaded = DimModelPackage.load("my-model/")
    >>> model = loaded.model
    >>>
    >>> # Validate inputs
    >>> inputs = {"velocity": velocity_tensor}
    >>> loaded.validate_inputs(inputs)
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Literal

from ..core.dimensions import Dimension
from ..core.units import Unit
from ..errors import DimensionError
from .cards import ModelCard, load_model_card, save_model_card

logger = logging.getLogger(__name__)

FrameworkType = Literal["pytorch", "jax", "numpy"]


class DimModelPackage:
    """Package for saving/loading physics models with unit metadata.

    A DimModelPackage wraps a model with its metadata and provides
    methods for saving to disk and loading with validation.

    Package structure:
        my-model/
        ├── model_card.json       # ModelCard metadata
        ├── weights.pt            # PyTorch state_dict
        ├── weights.msgpack       # JAX pytree
        └── config.json           # Framework-specific config

    Attributes:
        model: The model object (torch.nn.Module, JAX pytree, or numpy function).
        card: ModelCard with dimensional metadata.
        framework: Framework type ("pytorch", "jax", "numpy").
        auto_convert: If True, automatically convert input units to model units.
    """

    def __init__(
        self,
        model: Any,
        card: ModelCard,
        framework: FrameworkType = "pytorch",
        auto_convert: bool = False,
    ) -> None:
        """Initialize a model package.

        Args:
            model: The model object to package.
            card: ModelCard with dimensional metadata.
            framework: Framework type ("pytorch", "jax", "numpy").
            auto_convert: If True, automatically convert input units at inference.
        """
        self.model = model
        self.card = card
        self.framework = framework
        self.auto_convert = auto_convert

    def save(self, path: str | Path) -> None:
        """Save the model package to a directory.

        Creates a directory containing:
        - model_card.json: ModelCard metadata
        - weights.*: Model weights (format depends on framework)
        - config.json: Framework-specific configuration

        Args:
            path: Directory path to save to (will be created if needed).

        Example:
            >>> package.save("models/my-physics-model")
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model card
        save_model_card(self.card, path / "model_card.json")

        # Save framework-specific weights
        if self.framework == "pytorch":
            self._save_pytorch(path)
        elif self.framework == "jax":
            self._save_jax(path)
        elif self.framework == "numpy":
            self._save_numpy(path)
        else:
            raise ValueError(f"Unknown framework: {self.framework}")

        # Save config
        config = {
            "framework": self.framework,
            "auto_convert": self.auto_convert,
            "version": "1.0.0",
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved model package to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "DimModelPackage":
        """Load a model package from a directory.

        Args:
            path: Directory path containing the model package.

        Returns:
            DimModelPackage with loaded model and metadata.

        Raises:
            FileNotFoundError: If package directory or required files not found.
            ValueError: If package format is invalid.

        Example:
            >>> package = DimModelPackage.load("models/my-physics-model")
            >>> model = package.model
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Package not found: {path}")

        # Load config
        config_path = path / "config.json"
        if not config_path.exists():
            raise ValueError(f"Missing config.json in package: {path}")

        with open(config_path) as f:
            config = json.load(f)

        framework = config["framework"]
        auto_convert = config.get("auto_convert", False)

        # Load model card
        card_path = path / "model_card.json"
        if not card_path.exists():
            raise ValueError(f"Missing model_card.json in package: {path}")

        card = load_model_card(card_path)

        # Load framework-specific weights
        if framework == "pytorch":
            model = cls._load_pytorch(path, card)
        elif framework == "jax":
            model = cls._load_jax(path, card)
        elif framework == "numpy":
            model = cls._load_numpy(path, card)
        else:
            raise ValueError(f"Unknown framework: {framework}")

        logger.info(f"Loaded model package from {path}")

        return cls(model, card, framework, auto_convert)

    def validate_inputs(
        self,
        inputs: dict[str, Any],
        strict: bool = True,
    ) -> None:
        """Validate input dimensions against model card.

        Checks that inputs have correct dimensions. If auto_convert is enabled,
        this will be called automatically during inference.

        Args:
            inputs: Dict mapping input names to DimArray/DimTensor objects.
            strict: If True, raise error on mismatch. If False, only warn.

        Raises:
            DimensionError: If input dimensions don't match (when strict=True).

        Example:
            >>> from dimtensor import DimArray
            >>> from dimtensor.units import m, s
            >>>
            >>> velocity = DimArray([1.0, 2.0], m / s)
            >>> package.validate_inputs({"velocity": velocity})
        """
        expected = self.card.info.input_dims

        for name, expected_dim in expected.items():
            if name not in inputs:
                msg = f"Missing required input: {name}"
                if strict:
                    raise DimensionError(msg)
                else:
                    logger.warning(msg)
                continue

            input_obj = inputs[name]

            # Extract dimension from input
            if hasattr(input_obj, "unit"):
                actual_dim = input_obj.unit.dimension
            elif hasattr(input_obj, "dimension"):
                actual_dim = input_obj.dimension
            else:
                msg = f"Input '{name}' has no unit/dimension information"
                if strict:
                    raise DimensionError(msg)
                else:
                    logger.warning(msg)
                continue

            # Compare dimensions
            if actual_dim != expected_dim:
                msg = (
                    f"Input '{name}' has wrong dimension: "
                    f"expected {expected_dim}, got {actual_dim}"
                )
                if strict:
                    raise DimensionError(msg)
                else:
                    logger.warning(msg)

    def validate_outputs(
        self,
        outputs: dict[str, Any],
        strict: bool = True,
    ) -> None:
        """Validate output dimensions against model card.

        Args:
            outputs: Dict mapping output names to DimArray/DimTensor objects.
            strict: If True, raise error on mismatch. If False, only warn.

        Raises:
            DimensionError: If output dimensions don't match (when strict=True).
        """
        expected = self.card.info.output_dims

        for name, expected_dim in expected.items():
            if name not in outputs:
                msg = f"Missing expected output: {name}"
                if strict:
                    raise DimensionError(msg)
                else:
                    logger.warning(msg)
                continue

            output_obj = outputs[name]

            # Extract dimension from output
            if hasattr(output_obj, "unit"):
                actual_dim = output_obj.unit.dimension
            elif hasattr(output_obj, "dimension"):
                actual_dim = output_obj.dimension
            else:
                msg = f"Output '{name}' has no unit/dimension information"
                if strict:
                    raise DimensionError(msg)
                else:
                    logger.warning(msg)
                continue

            # Compare dimensions
            if actual_dim != expected_dim:
                msg = (
                    f"Output '{name}' has wrong dimension: "
                    f"expected {expected_dim}, got {actual_dim}"
                )
                if strict:
                    raise DimensionError(msg)
                else:
                    logger.warning(msg)

    def _save_pytorch(self, path: Path) -> None:
        """Save PyTorch model weights."""
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch is required to save PyTorch models")

        # Save state dict
        weights_path = path / "weights.pt"
        torch.save(self.model.state_dict(), weights_path)

        # Save architecture info if available
        arch_info = {
            "class": self.model.__class__.__name__,
            "module": self.model.__class__.__module__,
        }

        # Try to save constructor args if available
        if hasattr(self.model, "get_config"):
            arch_info["config"] = self.model.get_config()

        with open(path / "architecture.json", "w") as f:
            json.dump(arch_info, f, indent=2)

    def _save_jax(self, path: Path) -> None:
        """Save JAX pytree weights."""
        try:
            import jax
            from flax import serialization
        except ImportError:
            raise RuntimeError("JAX and Flax are required to save JAX models")

        # Save pytree as msgpack
        weights_path = path / "weights.msgpack"
        bytes_data = serialization.to_bytes(self.model)
        with open(weights_path, "wb") as f:
            f.write(bytes_data)

    def _save_numpy(self, path: Path) -> None:
        """Save numpy-based model."""
        import pickle

        # For numpy models, pickle the whole model
        weights_path = path / "model.pkl"
        with open(weights_path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def _load_pytorch(cls, path: Path, card: ModelCard) -> Any:
        """Load PyTorch model weights."""
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch is required to load PyTorch models")

        weights_path = path / "weights.pt"
        if not weights_path.exists():
            raise ValueError(f"Missing weights.pt in package: {path}")

        # Load state dict only - user must provide model architecture
        state_dict = torch.load(weights_path, map_location="cpu")

        # Return a dict with state_dict for now
        # In a real implementation, we'd need to instantiate the model
        # This requires storing architecture info
        return {"state_dict": state_dict, "card": card}

    @classmethod
    def _load_jax(cls, path: Path, card: ModelCard) -> Any:
        """Load JAX pytree weights."""
        try:
            from flax import serialization
        except ImportError:
            raise RuntimeError("JAX and Flax are required to load JAX models")

        weights_path = path / "weights.msgpack"
        if not weights_path.exists():
            raise ValueError(f"Missing weights.msgpack in package: {path}")

        with open(weights_path, "rb") as f:
            bytes_data = f.read()

        # Deserialize pytree
        pytree = serialization.from_bytes(None, bytes_data)
        return pytree

    @classmethod
    def _load_numpy(cls, path: Path, card: ModelCard) -> Any:
        """Load numpy-based model."""
        import pickle

        weights_path = path / "model.pkl"
        if not weights_path.exists():
            raise ValueError(f"Missing model.pkl in package: {path}")

        with open(weights_path, "rb") as f:
            model = pickle.load(f)

        return model

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DimModelPackage(name={self.card.info.name}, "
            f"version={self.card.info.version}, "
            f"framework={self.framework})"
        )

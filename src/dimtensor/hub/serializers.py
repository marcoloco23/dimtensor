"""Framework-specific serialization for model packages.

This module provides serialization utilities for different frameworks
(PyTorch, JAX) with proper handling of dimensional metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


class ModelSerializer(Protocol):
    """Protocol for model serializers."""

    def save(self, model: Any, path: Path) -> dict[str, Any]:
        """Save model to path, return metadata dict."""
        ...

    def load(self, path: Path, metadata: dict[str, Any]) -> Any:
        """Load model from path using metadata."""
        ...


class PyTorchSerializer:
    """Serializer for PyTorch models."""

    def save(self, model: Any, path: Path) -> dict[str, Any]:
        """Save PyTorch model state dict.

        Args:
            model: PyTorch nn.Module.
            path: Directory to save to.

        Returns:
            Metadata dict with architecture information.
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch is required")

        # Save state dict
        state_dict_path = path / "weights.pt"
        torch.save(model.state_dict(), state_dict_path)

        # Extract metadata
        metadata = {
            "class_name": model.__class__.__name__,
            "module": model.__class__.__module__,
        }

        # Try to get config if available
        if hasattr(model, "get_config"):
            metadata["config"] = model.get_config()

        # Save architecture metadata
        with open(path / "architecture.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def load(self, path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        """Load PyTorch model state dict.

        Args:
            path: Directory to load from.
            metadata: Architecture metadata.

        Returns:
            Dict with 'state_dict' key containing loaded weights.
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch is required")

        state_dict_path = path / "weights.pt"
        state_dict = torch.load(state_dict_path, map_location="cpu")

        return {
            "state_dict": state_dict,
            "metadata": metadata,
        }

    def load_into_model(self, path: Path, model: Any) -> None:
        """Load weights directly into a model instance.

        Args:
            path: Directory to load from.
            model: PyTorch nn.Module to load weights into.
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch is required")

        state_dict_path = path / "weights.pt"
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)


class JAXSerializer:
    """Serializer for JAX models."""

    def save(self, model: Any, path: Path) -> dict[str, Any]:
        """Save JAX pytree.

        Args:
            model: JAX pytree (typically Flax parameters).
            path: Directory to save to.

        Returns:
            Metadata dict.
        """
        try:
            from flax import serialization
        except ImportError:
            raise RuntimeError("Flax is required for JAX serialization")

        # Serialize pytree to msgpack
        weights_path = path / "weights.msgpack"
        bytes_data = serialization.to_bytes(model)
        with open(weights_path, "wb") as f:
            f.write(bytes_data)

        metadata = {
            "format": "flax_msgpack",
        }

        return metadata

    def load(self, path: Path, metadata: dict[str, Any]) -> Any:
        """Load JAX pytree.

        Args:
            path: Directory to load from.
            metadata: Serialization metadata.

        Returns:
            Deserialized pytree.
        """
        try:
            from flax import serialization
        except ImportError:
            raise RuntimeError("Flax is required for JAX serialization")

        weights_path = path / "weights.msgpack"
        with open(weights_path, "rb") as f:
            bytes_data = f.read()

        # Deserialize - need target pytree structure
        # For now, return raw bytes and let user handle restoration
        pytree = serialization.from_bytes(None, bytes_data)
        return pytree


class NumPySerializer:
    """Serializer for NumPy-based models."""

    def save(self, model: Any, path: Path) -> dict[str, Any]:
        """Save numpy model using pickle.

        Args:
            model: Model object to save.
            path: Directory to save to.

        Returns:
            Metadata dict.
        """
        import pickle

        model_path = path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        metadata = {
            "format": "pickle",
            "type": type(model).__name__,
        }

        return metadata

    def load(self, path: Path, metadata: dict[str, Any]) -> Any:
        """Load numpy model.

        Args:
            path: Directory to load from.
            metadata: Serialization metadata.

        Returns:
            Loaded model object.
        """
        import pickle

        model_path = path / "model.pkl"
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        return model


def get_serializer(framework: str) -> ModelSerializer:
    """Get serializer for a framework.

    Args:
        framework: Framework name ("pytorch", "jax", "numpy").

    Returns:
        Appropriate serializer instance.

    Raises:
        ValueError: If framework not supported.
    """
    serializers = {
        "pytorch": PyTorchSerializer(),
        "jax": JAXSerializer(),
        "numpy": NumPySerializer(),
    }

    if framework not in serializers:
        raise ValueError(
            f"Unknown framework: {framework}. "
            f"Supported: {list(serializers.keys())}"
        )

    return serializers[framework]

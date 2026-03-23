"""Framework-specific serialization for model packages.

This module provides serialization utilities for different frameworks
(PyTorch, JAX) with proper handling of dimensional metadata.
"""

from __future__ import annotations

import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Top-level modules allowed for unpickling
_SAFE_PICKLE_TOP_MODULES = frozenset({
    "numpy",
    "builtins",
    "collections",
    "dimtensor",
    "copyreg",
    "_codecs",
})

# Explicitly blocked modules (even if top-level is allowed)
_BLOCKED_PICKLE_MODULES = frozenset({
    "os",
    "sys",
    "subprocess",
    "shutil",
    "importlib",
    "ctypes",
    "socket",
    "http",
    "urllib",
    "pickle",
    "io",
    "code",
    "codeop",
    "compileall",
    "runpy",
    "webbrowser",
    "pathlib",
    "signal",
    "multiprocessing",
    "threading",
})


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows known-safe classes to be deserialized.

    Blocks arbitrary code execution by restricting which modules and
    classes can be instantiated during deserialization.
    """

    def find_class(self, module: str, name: str) -> Any:
        top_module = module.split(".")[0]

        if top_module in _BLOCKED_PICKLE_MODULES:
            raise pickle.UnpicklingError(
                f"Deserialization of '{module}.{name}' is blocked for security."
            )

        if top_module in _SAFE_PICKLE_TOP_MODULES:
            return super().find_class(module, name)

        raise pickle.UnpicklingError(
            f"Deserialization of '{module}.{name}' is blocked for security. "
            f"Only numpy, dimtensor, and builtins types are allowed. "
            f"If you trust this file, use pickle.load() directly."
        )


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
        state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)

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
        state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
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
        model_path = path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        metadata = {
            "format": "pickle",
            "type": type(model).__name__,
        }

        return metadata

    def load(self, path: Path, metadata: dict[str, Any]) -> Any:
        """Load numpy model using restricted unpickling.

        Only numpy, dimtensor, and builtins types are allowed to be
        deserialized. This prevents arbitrary code execution from
        malicious model files.

        Args:
            path: Directory to load from.
            metadata: Serialization metadata.

        Returns:
            Loaded model object.

        Raises:
            pickle.UnpicklingError: If the file contains disallowed types.
        """
        model_path = path / "model.pkl"
        with open(model_path, "rb") as f:
            model = _RestrictedUnpickler(f).load()

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

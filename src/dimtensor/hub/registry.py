"""Model registry for physics-aware neural networks.

Provides a registry system for discovering, registering, and loading
pre-trained physics models with dimensional metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.dimensions import Dimension

# Default cache directory
_CACHE_DIR = Path.home() / ".cache" / "dimtensor" / "models"

# Global registry
_REGISTRY: dict[str, "ModelInfo"] = {}

# Model factory functions
_FACTORIES: dict[str, Callable[..., Any]] = {}


@dataclass
class ModelInfo:
    """Metadata about a registered physics model.

    Attributes:
        name: Unique model identifier (e.g., "fluid-velocity-predictor").
        version: Model version string.
        description: Human-readable description.
        input_dims: Dict mapping input names to their dimensions.
        output_dims: Dict mapping output names to their dimensions.
        domain: Physics domain (e.g., "mechanics", "thermodynamics").
        characteristic_scales: Dict of characteristic values used in training.
        tags: List of tags for search/filtering.
        source: URL or local path to model weights.
        architecture: Model architecture name.
        author: Model author/organization.
        license: License for the model.
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    input_dims: dict[str, Dimension] = field(default_factory=dict)
    output_dims: dict[str, Dimension] = field(default_factory=dict)
    domain: str = "general"
    characteristic_scales: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    source: str = ""
    architecture: str = ""
    author: str = ""
    license: str = "MIT"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "input_dims": {
                k: self._dim_to_dict(v) for k, v in self.input_dims.items()
            },
            "output_dims": {
                k: self._dim_to_dict(v) for k, v in self.output_dims.items()
            },
            "domain": self.domain,
            "characteristic_scales": self.characteristic_scales,
            "tags": self.tags,
            "source": self.source,
            "architecture": self.architecture,
            "author": self.author,
            "license": self.license,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelInfo":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            input_dims={
                k: cls._dict_to_dim(v)
                for k, v in data.get("input_dims", {}).items()
            },
            output_dims={
                k: cls._dict_to_dim(v)
                for k, v in data.get("output_dims", {}).items()
            },
            domain=data.get("domain", "general"),
            characteristic_scales=data.get("characteristic_scales", {}),
            tags=data.get("tags", []),
            source=data.get("source", ""),
            architecture=data.get("architecture", ""),
            author=data.get("author", ""),
            license=data.get("license", "MIT"),
        )

    @staticmethod
    def _dim_to_dict(dim: Dimension) -> dict[str, float]:
        """Convert Dimension to dict."""
        result = {}
        names = ["L", "M", "T", "I", "Theta", "N", "J"]
        for i, name in enumerate(names):
            exp = float(dim._exponents[i])
            if exp != 0:
                result[name] = exp
        return result

    @staticmethod
    def _dict_to_dim(data: dict[str, float]) -> Dimension:
        """Convert dict to Dimension."""
        return Dimension(
            length=data.get("L", 0),
            mass=data.get("M", 0),
            time=data.get("T", 0),
            current=data.get("I", 0),
            temperature=data.get("Theta", 0),
            amount=data.get("N", 0),
            luminosity=data.get("J", 0),
        )


def register_model(
    name: str,
    info: ModelInfo | None = None,
    factory: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]] | None:
    """Register a model in the hub.

    Can be used as a decorator or called directly.

    Args:
        name: Unique model identifier.
        info: ModelInfo with metadata. If None, created from kwargs.
        factory: Factory function to create the model.
        **kwargs: Additional arguments passed to ModelInfo.

    Returns:
        Decorator function if used as decorator, None otherwise.

    Examples:
        # As decorator
        @register_model("my-model", domain="mechanics")
        def create_my_model():
            return MyModel()

        # Direct call
        register_model("my-model", info=my_info, factory=create_fn)
    """
    if info is None:
        info = ModelInfo(name=name, **kwargs)

    _REGISTRY[name] = info

    if factory is not None:
        _FACTORIES[name] = factory
        return None

    # Return decorator
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _FACTORIES[name] = fn
        return fn

    return decorator


def get_model_info(name: str) -> ModelInfo:
    """Get metadata for a registered model.

    Args:
        name: Model identifier.

    Returns:
        ModelInfo with model metadata.

    Raises:
        KeyError: If model not found.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Model '{name}' not found in registry")
    return _REGISTRY[name]


def list_models(
    domain: str | None = None,
    tags: list[str] | None = None,
) -> list[ModelInfo]:
    """List available models in the registry.

    Args:
        domain: Filter by physics domain.
        tags: Filter by tags (models must have ALL specified tags).

    Returns:
        List of ModelInfo objects matching the filters.

    Examples:
        >>> # List all models
        >>> models = list_models()
        >>>
        >>> # Filter by domain
        >>> mechanics_models = list_models(domain="mechanics")
        >>>
        >>> # Filter by tags
        >>> cfd_models = list_models(tags=["fluid", "cfd"])
    """
    results = list(_REGISTRY.values())

    if domain is not None:
        results = [m for m in results if m.domain == domain]

    if tags is not None:
        results = [m for m in results if all(t in m.tags for t in tags)]

    return results


def load_model(
    name: str,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    **kwargs: Any,
) -> Any:
    """Load a model from the registry.

    Args:
        name: Model identifier.
        cache_dir: Directory for cached models. Defaults to ~/.cache/dimtensor/models.
        force_download: If True, re-download even if cached.
        **kwargs: Additional arguments passed to the model factory.

    Returns:
        The loaded model (typically a torch.nn.Module).

    Raises:
        KeyError: If model not found.
        RuntimeError: If model cannot be loaded.

    Examples:
        >>> model = load_model("fluid-velocity-predictor")
        >>> model = load_model("my-model", device="cuda")
    """
    if name not in _REGISTRY:
        raise KeyError(f"Model '{name}' not found in registry")

    info = _REGISTRY[name]

    # Check if we have a factory function
    if name in _FACTORIES:
        return _FACTORIES[name](**kwargs)

    # Try to load from source
    if info.source:
        return _load_from_source(info, cache_dir, force_download, **kwargs)

    raise RuntimeError(
        f"Model '{name}' has no factory or source URL. "
        "Register a factory function or provide a source URL."
    )


def _load_from_source(
    info: ModelInfo,
    cache_dir: str | Path | None,
    force_download: bool,
    **kwargs: Any,
) -> Any:
    """Load model from source URL or path."""
    cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    source = info.source

    # Local file
    if os.path.exists(source):
        return _load_weights(source, **kwargs)

    # Remote URL - check cache. MD5 is used as a cache filename suffix,
    # not for security; a collision just causes a cache miss.
    url_hash = hashlib.md5(source.encode(), usedforsecurity=False).hexdigest()[:12]
    cache_path = cache_dir / f"{info.name}-{info.version}-{url_hash}.pt"

    if cache_path.exists() and not force_download:
        return _load_weights(str(cache_path), **kwargs)

    # Download
    try:
        import urllib.request

        urllib.request.urlretrieve(source, cache_path)
        return _load_weights(str(cache_path), **kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to download model from {source}: {e}") from e


def _load_weights(path: str, **kwargs: Any) -> Any:
    """Load model weights from a file."""
    try:
        import torch

        return torch.load(path, **kwargs)
    except ImportError:
        raise RuntimeError("PyTorch is required to load model weights")


def save_registry(path: str | Path) -> None:
    """Save the registry to a JSON file.

    Args:
        path: Path to save the registry.
    """
    data = {name: info.to_dict() for name, info in _REGISTRY.items()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_registry(path: str | Path) -> None:
    """Load registry from a JSON file.

    Args:
        path: Path to the registry file.
    """
    with open(path) as f:
        data = json.load(f)

    for name, info_dict in data.items():
        _REGISTRY[name] = ModelInfo.from_dict(info_dict)


def clear_registry() -> None:
    """Clear all registered models (mainly for testing)."""
    _REGISTRY.clear()
    _FACTORIES.clear()

"""Model hub for physics-aware neural networks.

Provides a registry for pre-trained physics models with dimensional
metadata, enabling discovery and loading of models with proper units.

Example:
    >>> from dimtensor.hub import list_models, load_model
    >>>
    >>> # Find models for fluid dynamics
    >>> models = list_models(domain="fluid_dynamics")
    >>> for m in models:
    ...     print(f"{m.name}: {m.description}")
    >>>
    >>> # Load a model
    >>> model = load_model("fluid-velocity-predictor")
"""

from .registry import (
    ModelInfo,
    list_models,
    load_model,
    register_model,
    get_model_info,
)
from .cards import ModelCard, load_model_card, save_model_card, TrainingInfo
from .package import DimModelPackage
from .validators import DimModelWrapper, validate_model_io, create_validator
from .serializers import get_serializer

__all__ = [
    "ModelInfo",
    "ModelCard",
    "TrainingInfo",
    "list_models",
    "load_model",
    "register_model",
    "get_model_info",
    "load_model_card",
    "save_model_card",
    "DimModelPackage",
    "DimModelWrapper",
    "validate_model_io",
    "create_validator",
    "get_serializer",
]

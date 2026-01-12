"""Unit schema sharing system for dimtensor.

This module provides tools for creating, sharing, and managing unit schemas.
Schemas package custom units, constants, and dimensions into shareable formats
with versioning and conflict resolution.

Examples:
    >>> from dimtensor.schema import UnitSchema, load_schema, merge_schemas
    >>>
    >>> # Load a built-in schema
    >>> astro = load_schema("astronomy")
    >>>
    >>> # Create a custom schema
    >>> schema = UnitSchema(
    ...     name="nuclear_physics",
    ...     version="1.0.0",
    ...     description="Units for nuclear physics"
    ... )
    >>> schema.add_unit("MeV", {"M": 1, "L": 2, "T": -2}, 1.602176634e-13)
    >>>
    >>> # Save and share
    >>> schema.save("nuclear_physics.yaml")
    >>>
    >>> # Merge schemas
    >>> combined = merge_schemas([astro, schema], strategy="namespace")
"""

from .merge import merge_schemas, MergeStrategy
from .registry import SchemaRegistry, get_registry
from .schema import UnitSchema
from .serialization import load_schema, save_schema
from .validation import validate_schema

__all__ = [
    "UnitSchema",
    "load_schema",
    "save_schema",
    "merge_schemas",
    "MergeStrategy",
    "SchemaRegistry",
    "get_registry",
    "validate_schema",
]

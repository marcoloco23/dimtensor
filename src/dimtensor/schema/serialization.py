"""Serialization functions for unit schemas.

Provides YAML and JSON serialization with automatic format detection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .schema import UnitSchema


def save_schema(schema: UnitSchema, path: str | Path, format: str = "auto") -> None:
    """Save a schema to a file.

    Args:
        schema: UnitSchema to save.
        path: Output file path.
        format: Format to use ("yaml", "json", or "auto" to detect from extension).

    Raises:
        ImportError: If YAML format requested but PyYAML not installed.
        ValueError: If format is invalid.

    Examples:
        >>> schema = UnitSchema(name="test", version="1.0.0")
        >>> save_schema(schema, "test.yaml")
        >>> save_schema(schema, "test.json", format="json")
    """
    path = Path(path)

    # Detect format from extension
    if format == "auto":
        ext = path.suffix.lower()
        if ext in (".yaml", ".yml"):
            format = "yaml"
        elif ext == ".json":
            format = "json"
        else:
            # Default to YAML if available
            format = "yaml" if HAS_YAML else "json"

    # Convert to dict
    data = schema.to_dict()

    # Save based on format
    if format == "yaml":
        if not HAS_YAML:
            raise ImportError(
                "PyYAML is required for YAML serialization. "
                "Install with: pip install pyyaml"
            )
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    elif format == "json":
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Invalid format: {format}. Use 'yaml', 'json', or 'auto'.")


def load_schema(path: str | Path) -> UnitSchema:
    """Load a schema from a file.

    Automatically detects format from file extension or content.

    Args:
        path: Path to schema file (YAML or JSON).

    Returns:
        Loaded UnitSchema.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ImportError: If YAML file but PyYAML not installed.
        ValueError: If file format is invalid or content is malformed.

    Examples:
        >>> schema = load_schema("astronomy.yaml")
        >>> schema = load_schema("chemistry.json")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")

    # Try to detect format from extension
    ext = path.suffix.lower()

    try:
        if ext in (".yaml", ".yml"):
            if not HAS_YAML:
                raise ImportError(
                    "PyYAML is required to load YAML files. "
                    "Install with: pip install pyyaml"
                )
            with open(path) as f:
                data = yaml.safe_load(f)
        elif ext == ".json":
            with open(path) as f:
                data = json.load(f)
        else:
            # Try to auto-detect by attempting JSON first, then YAML
            with open(path) as f:
                content = f.read()
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                if not HAS_YAML:
                    raise ImportError(
                        "PyYAML is required for YAML format. "
                        "Install with: pip install pyyaml"
                    )
                data = yaml.safe_load(content)
    except Exception as e:
        raise ValueError(f"Failed to parse schema file {path}: {e}") from e

    # Validate basic structure
    if not isinstance(data, dict):
        raise ValueError(f"Schema file must contain a dictionary, got {type(data)}")

    if "name" not in data:
        raise ValueError("Schema must have a 'name' field")

    if "version" not in data:
        raise ValueError("Schema must have a 'version' field")

    # Create schema from dict
    return UnitSchema.from_dict(data)


def to_yaml(schema: UnitSchema) -> str:
    """Convert schema to YAML string.

    Args:
        schema: UnitSchema to serialize.

    Returns:
        YAML string representation.

    Raises:
        ImportError: If PyYAML is not installed.
    """
    if not HAS_YAML:
        raise ImportError(
            "PyYAML is required for YAML serialization. "
            "Install with: pip install pyyaml"
        )
    data = schema.to_dict()
    return yaml.dump(data, default_flow_style=False, sort_keys=False)


def from_yaml(yaml_str: str) -> UnitSchema:
    """Load schema from YAML string.

    Args:
        yaml_str: YAML string to parse.

    Returns:
        Loaded UnitSchema.

    Raises:
        ImportError: If PyYAML is not installed.
        ValueError: If YAML is malformed.
    """
    if not HAS_YAML:
        raise ImportError(
            "PyYAML is required for YAML deserialization. "
            "Install with: pip install pyyaml"
        )

    try:
        data = yaml.safe_load(yaml_str)
    except Exception as e:
        raise ValueError(f"Failed to parse YAML: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"YAML must contain a dictionary, got {type(data)}")

    return UnitSchema.from_dict(data)


def to_json(schema: UnitSchema, indent: int = 2) -> str:
    """Convert schema to JSON string.

    Args:
        schema: UnitSchema to serialize.
        indent: Indentation level (default 2).

    Returns:
        JSON string representation.
    """
    data = schema.to_dict()
    return json.dumps(data, indent=indent)


def from_json(json_str: str) -> UnitSchema:
    """Load schema from JSON string.

    Args:
        json_str: JSON string to parse.

    Returns:
        Loaded UnitSchema.

    Raises:
        ValueError: If JSON is malformed.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"JSON must contain a dictionary, got {type(data)}")

    return UnitSchema.from_dict(data)

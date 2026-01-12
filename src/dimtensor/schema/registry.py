"""Schema registry for managing installed schemas.

Provides local storage and discovery of schemas at ~/.cache/dimtensor/schemas/.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .schema import UnitSchema
from .serialization import load_schema as load_schema_file, save_schema
from .validation import validate_schema, validate_version, ValidationError


# Default cache directory
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "dimtensor" / "schemas"

# Global registry instance
_GLOBAL_REGISTRY: SchemaRegistry | None = None


class SchemaRegistry:
    """Registry for managing installed schemas.

    Stores schemas in a local directory and provides discovery,
    installation, and management functions.

    Attributes:
        cache_dir: Directory where schemas are stored.

    Examples:
        >>> registry = SchemaRegistry()
        >>> registry.install("astronomy.yaml")
        >>> schemas = registry.list_schemas()
        >>> schema = registry.load("astronomy")
    """

    def __init__(self, cache_dir: str | Path | None = None):
        """Initialize registry.

        Args:
            cache_dir: Directory for schema storage (default: ~/.cache/dimtensor/schemas).
        """
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def install(
        self,
        source: str | Path,
        name: str | None = None,
        validate: bool = True,
    ) -> str:
        """Install a schema from a file or URL.

        Args:
            source: Path to schema file or URL.
            name: Optional custom name (default: use schema's name).
            validate: Whether to validate schema before installing (default: True).

        Returns:
            Installed schema identifier (name@version).

        Raises:
            FileNotFoundError: If source file doesn't exist.
            ValidationError: If schema is invalid (when validate=True).

        Examples:
            >>> registry.install("nuclear_physics.yaml")
            'nuclear_physics@1.0.0'
            >>> registry.install("https://example.com/schema.yaml")
            'chemistry@2.0.0'
        """
        # Load schema from source
        source_path = Path(source)

        if not source_path.exists():
            # TODO: Support URLs in future version
            raise FileNotFoundError(f"Schema file not found: {source}")

        schema = load_schema_file(source_path)

        # Validate if requested
        if validate:
            warnings = validate_schema(schema)
            if warnings:
                print(f"Schema validation warnings for {schema.name}:")
                for warning in warnings:
                    print(f"  - {warning}")

        # Use custom name if provided
        if name:
            schema.name = name

        # Save to registry
        schema_id = f"{schema.name}@{schema.version}"
        dest_path = self._get_schema_path(schema.name, schema.version)
        save_schema(schema, dest_path)

        return schema_id

    def list_schemas(self) -> list[dict[str, str]]:
        """List all installed schemas.

        Returns:
            List of dicts with schema metadata (name, version, description).

        Examples:
            >>> schemas = registry.list_schemas()
            >>> for schema in schemas:
            ...     print(f"{schema['name']} v{schema['version']}")
        """
        result = []

        # Scan cache directory for schema files
        for path in self.cache_dir.glob("*.yaml"):
            try:
                schema = load_schema_file(path)
                result.append(
                    {
                        "name": schema.name,
                        "version": schema.version,
                        "description": schema.description,
                        "path": str(path),
                    }
                )
            except Exception:
                # Skip invalid files
                pass

        for path in self.cache_dir.glob("*.json"):
            try:
                schema = load_schema_file(path)
                result.append(
                    {
                        "name": schema.name,
                        "version": schema.version,
                        "description": schema.description,
                        "path": str(path),
                    }
                )
            except Exception:
                # Skip invalid files
                pass

        return result

    def load(self, identifier: str) -> UnitSchema:
        """Load a schema by name or name@version.

        Args:
            identifier: Schema name or name@version (e.g., "astronomy" or "astronomy@1.0.0").

        Returns:
            Loaded UnitSchema.

        Raises:
            KeyError: If schema not found.
            ValueError: If multiple versions found and no version specified.

        Examples:
            >>> schema = registry.load("astronomy")
            >>> schema = registry.load("astronomy@1.0.0")
        """
        # Parse identifier
        if "@" in identifier:
            name, version = identifier.split("@", 1)
            path = self._get_schema_path(name, version)
            if not path.exists():
                raise KeyError(f"Schema not found: {identifier}")
            return load_schema_file(path)
        else:
            # Find latest version
            name = identifier
            matching = [
                s for s in self.list_schemas() if s["name"] == name
            ]
            if not matching:
                raise KeyError(f"Schema not found: {name}")

            if len(matching) > 1:
                versions = [s["version"] for s in matching]
                raise ValueError(
                    f"Multiple versions of '{name}' found: {versions}. "
                    "Please specify version like 'name@version'"
                )

            return load_schema_file(matching[0]["path"])

    def remove(self, identifier: str) -> None:
        """Remove an installed schema.

        Args:
            identifier: Schema name or name@version.

        Raises:
            KeyError: If schema not found.

        Examples:
            >>> registry.remove("astronomy@1.0.0")
            >>> registry.remove("chemistry")  # Removes all versions
        """
        if "@" in identifier:
            # Remove specific version
            name, version = identifier.split("@", 1)
            path = self._get_schema_path(name, version)
            if path.exists():
                path.unlink()
            else:
                raise KeyError(f"Schema not found: {identifier}")
        else:
            # Remove all versions
            name = identifier
            matching = [
                s for s in self.list_schemas() if s["name"] == name
            ]
            if not matching:
                raise KeyError(f"Schema not found: {name}")

            for schema_info in matching:
                Path(schema_info["path"]).unlink()

    def _get_schema_path(self, name: str, version: str) -> Path:
        """Get path for a schema file.

        Args:
            name: Schema name.
            version: Schema version.

        Returns:
            Path to schema file.
        """
        # Use YAML by default
        return self.cache_dir / f"{name}@{version}.yaml"

    def export_builtin_schemas(self) -> None:
        """Export built-in domain modules as schemas.

        Converts existing domain modules (astronomy, chemistry, etc.)
        to schema format and installs them.
        """
        # This will be implemented after built-in schemas are created
        pass


def get_registry(cache_dir: str | Path | None = None) -> SchemaRegistry:
    """Get the global schema registry instance.

    Args:
        cache_dir: Optional custom cache directory.

    Returns:
        SchemaRegistry instance.

    Examples:
        >>> registry = get_registry()
        >>> registry.install("my_schema.yaml")
    """
    global _GLOBAL_REGISTRY

    if _GLOBAL_REGISTRY is None or cache_dir is not None:
        _GLOBAL_REGISTRY = SchemaRegistry(cache_dir)

    return _GLOBAL_REGISTRY

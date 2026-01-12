"""Validation functions for unit schemas.

Validates schema structure, version strings, unit definitions,
and dependency graphs.
"""

from __future__ import annotations

import re
from typing import Any

from .schema import UnitSchema


class ValidationError(Exception):
    """Raised when schema validation fails."""

    pass


def validate_schema(schema: UnitSchema) -> list[str]:
    """Validate a schema and return list of warnings.

    Args:
        schema: UnitSchema to validate.

    Returns:
        List of warning messages (empty if no warnings).

    Raises:
        ValidationError: If schema has critical errors.

    Examples:
        >>> schema = UnitSchema(name="test", version="1.0.0")
        >>> warnings = validate_schema(schema)
        >>> if warnings:
        ...     print("Warnings:", warnings)
    """
    warnings = []

    # Check required fields
    if not schema.name:
        raise ValidationError("Schema must have a name")

    if not schema.version:
        raise ValidationError("Schema must have a version")

    # Validate name format (alphanumeric, underscores, hyphens)
    if not re.match(r"^[a-zA-Z0-9_-]+$", schema.name):
        raise ValidationError(
            f"Schema name '{schema.name}' must contain only letters, "
            "numbers, underscores, and hyphens"
        )

    # Validate version string (semantic versioning)
    try:
        validate_version(schema.version)
    except ValidationError as e:
        raise ValidationError(f"Invalid version: {e}") from e

    # Validate schema_version
    try:
        validate_version(schema.schema_version)
    except ValidationError as e:
        raise ValidationError(f"Invalid schema_version: {e}") from e

    # Check for duplicate unit symbols
    unit_symbols = [u.symbol for u in schema.units]
    duplicates = [s for s in set(unit_symbols) if unit_symbols.count(s) > 1]
    if duplicates:
        raise ValidationError(
            f"Duplicate unit symbols found: {', '.join(duplicates)}"
        )

    # Check for duplicate constant symbols
    const_symbols = [c.symbol for c in schema.constants]
    duplicates = [s for s in set(const_symbols) if const_symbols.count(s) > 1]
    if duplicates:
        raise ValidationError(
            f"Duplicate constant symbols found: {', '.join(duplicates)}"
        )

    # Validate unit definitions
    for unit in schema.units:
        try:
            validate_unit_definition(unit, schema)
        except ValidationError as e:
            raise ValidationError(f"Invalid unit '{unit.symbol}': {e}") from e

    # Validate constant definitions
    for const in schema.constants:
        try:
            validate_constant_definition(const, schema)
        except ValidationError as e:
            raise ValidationError(
                f"Invalid constant '{const.symbol}': {e}"
            ) from e

    # Validate dependencies
    for dep in schema.dependencies:
        try:
            validate_dependency(dep)
        except ValidationError as e:
            raise ValidationError(f"Invalid dependency '{dep}': {e}") from e

    # Check for potential issues (warnings only)
    if not schema.description:
        warnings.append("Schema has no description")

    if not schema.author:
        warnings.append("Schema has no author")

    if not schema.units and not schema.constants:
        warnings.append("Schema contains no units or constants")

    return warnings


def validate_version(version: str) -> None:
    """Validate semantic version string.

    Args:
        version: Version string to validate (e.g., "1.0.0", "2.1.3-beta").

    Raises:
        ValidationError: If version format is invalid.
    """
    # Semantic versioning pattern: major.minor.patch[-prerelease][+build]
    pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?(?:\+([a-zA-Z0-9.-]+))?$"
    if not re.match(pattern, version):
        raise ValidationError(
            f"Version '{version}' must follow semantic versioning (e.g., '1.0.0')"
        )


def validate_unit_definition(unit: Any, schema: UnitSchema) -> None:
    """Validate a unit definition.

    Args:
        unit: UnitDefinition to validate.
        schema: Parent schema (for context).

    Raises:
        ValidationError: If unit definition is invalid.
    """
    # Check required fields
    if not unit.symbol:
        raise ValidationError("Unit must have a symbol")

    if unit.dimension is None:
        raise ValidationError("Unit must have a dimension")

    if unit.scale is None:
        raise ValidationError("Unit must have a scale")

    # Validate dimension dict
    if not isinstance(unit.dimension, dict):
        raise ValidationError(
            f"Dimension must be a dict, got {type(unit.dimension)}"
        )

    # Check dimension keys are valid
    valid_keys = {"L", "M", "T", "I", "Theta", "N", "J"}
    for key in unit.dimension.keys():
        if key not in valid_keys:
            raise ValidationError(
                f"Invalid dimension key '{key}'. "
                f"Valid keys are: {', '.join(sorted(valid_keys))}"
            )

    # Check dimension values are numeric
    for key, value in unit.dimension.items():
        if not isinstance(value, (int, float)):
            raise ValidationError(
                f"Dimension exponent for '{key}' must be numeric, got {type(value)}"
            )

    # Check scale is positive
    if unit.scale <= 0:
        raise ValidationError(f"Scale must be positive, got {unit.scale}")


def validate_constant_definition(const: Any, schema: UnitSchema) -> None:
    """Validate a constant definition.

    Args:
        const: ConstantDefinition to validate.
        schema: Parent schema (for context).

    Raises:
        ValidationError: If constant definition is invalid.
    """
    # Check required fields
    if not const.symbol:
        raise ValidationError("Constant must have a symbol")

    if not const.name:
        raise ValidationError("Constant must have a name")

    if const.value is None:
        raise ValidationError("Constant must have a value")

    if not const.unit:
        raise ValidationError("Constant must have a unit")

    # Check uncertainty is non-negative
    if const.uncertainty < 0:
        raise ValidationError(
            f"Uncertainty must be non-negative, got {const.uncertainty}"
        )


def validate_dependency(dep: str) -> None:
    """Validate a dependency string.

    Args:
        dep: Dependency string (e.g., "chemistry@1.0.0" or "chemistry@>=1.0.0").

    Raises:
        ValidationError: If dependency format is invalid.
    """
    # Pattern: name[@version_constraint]
    # Version constraint can be: 1.0.0, >=1.0.0, ^1.0.0, ~1.0.0
    pattern = r"^[a-zA-Z0-9_-]+(?:@(?:[>=^~])?[\d.]+)?$"
    if not re.match(pattern, dep):
        raise ValidationError(
            f"Dependency '{dep}' must be in format 'name@version' "
            "or 'name@>=version' (e.g., 'chemistry@1.0.0')"
        )


def check_circular_dependencies(schemas: list[UnitSchema]) -> list[list[str]]:
    """Check for circular dependencies in a list of schemas.

    Args:
        schemas: List of schemas to check.

    Returns:
        List of circular dependency chains found (empty if none).

    Examples:
        >>> cycles = check_circular_dependencies([schema1, schema2, schema3])
        >>> if cycles:
        ...     print("Circular dependencies found:", cycles)
    """
    # Build dependency graph
    graph: dict[str, list[str]] = {}
    for schema in schemas:
        deps = []
        for dep in schema.dependencies:
            # Extract schema name (before @)
            dep_name = dep.split("@")[0] if "@" in dep else dep
            deps.append(dep_name)
        graph[schema.name] = deps

    # Find cycles using DFS
    cycles = []
    visited = set()
    rec_stack = set()

    def dfs(node: str, path: list[str]) -> None:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, path[:])
            elif neighbor in rec_stack:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)

        rec_stack.remove(node)

    for node in graph:
        if node not in visited:
            dfs(node, [])

    return cycles

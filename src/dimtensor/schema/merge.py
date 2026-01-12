"""Schema merging with conflict resolution strategies.

Provides functions to merge multiple schemas with different strategies
for handling conflicts (duplicate unit or constant names).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from .schema import UnitSchema, UnitDefinition, ConstantDefinition
from .validation import ValidationError


class MergeStrategy(str, Enum):
    """Strategy for resolving conflicts when merging schemas.

    Attributes:
        STRICT: Raise error on any conflict.
        OVERRIDE: Later schemas override earlier ones.
        NAMESPACE: Prefix conflicting names with schema name.
    """

    STRICT = "strict"
    OVERRIDE = "override"
    NAMESPACE = "namespace"


class MergeConflict(Exception):
    """Raised when schema merge conflict occurs in strict mode."""

    pass


def merge_schemas(
    schemas: list[UnitSchema],
    strategy: MergeStrategy | str = MergeStrategy.STRICT,
    name: str | None = None,
    version: str = "1.0.0",
) -> UnitSchema:
    """Merge multiple schemas into one.

    Args:
        schemas: List of schemas to merge (order matters for OVERRIDE strategy).
        strategy: Conflict resolution strategy (STRICT, OVERRIDE, or NAMESPACE).
        name: Name for merged schema (default: first schema's name + "_merged").
        version: Version for merged schema (default: "1.0.0").

    Returns:
        Merged UnitSchema.

    Raises:
        MergeConflict: If conflicts detected in STRICT mode.
        ValueError: If schemas list is empty or strategy is invalid.

    Examples:
        >>> astro = load_schema("astronomy.yaml")
        >>> chem = load_schema("chemistry.yaml")
        >>> merged = merge_schemas([astro, chem], strategy=MergeStrategy.NAMESPACE)
    """
    if not schemas:
        raise ValueError("Cannot merge empty list of schemas")

    # Convert string strategy to enum
    if isinstance(strategy, str):
        try:
            strategy = MergeStrategy(strategy.lower())
        except ValueError:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Valid options: {[s.value for s in MergeStrategy]}"
            )

    # Default name
    if name is None:
        name = f"{schemas[0].name}_merged"

    # Collect metadata
    descriptions = [s.description for s in schemas if s.description]
    authors = [s.author for s in schemas if s.author]
    licenses = list(set(s.license for s in schemas if s.license))
    all_deps = []
    for s in schemas:
        all_deps.extend(s.dependencies)

    # Create base schema
    merged = UnitSchema(
        name=name,
        version=version,
        description=" + ".join(descriptions) if descriptions else "",
        author=", ".join(authors) if authors else "",
        license=", ".join(licenses) if licenses else "MIT",
        dependencies=list(set(all_deps)),
    )

    # Merge units
    if strategy == MergeStrategy.STRICT:
        merged.units = _merge_units_strict(schemas)
    elif strategy == MergeStrategy.OVERRIDE:
        merged.units = _merge_units_override(schemas)
    elif strategy == MergeStrategy.NAMESPACE:
        merged.units = _merge_units_namespace(schemas)

    # Merge constants
    if strategy == MergeStrategy.STRICT:
        merged.constants = _merge_constants_strict(schemas)
    elif strategy == MergeStrategy.OVERRIDE:
        merged.constants = _merge_constants_override(schemas)
    elif strategy == MergeStrategy.NAMESPACE:
        merged.constants = _merge_constants_namespace(schemas)

    return merged


def _merge_units_strict(schemas: list[UnitSchema]) -> list[UnitDefinition]:
    """Merge units in STRICT mode (error on conflicts)."""
    units_by_symbol: dict[str, tuple[UnitDefinition, str]] = {}

    for schema in schemas:
        for unit in schema.units:
            if unit.symbol in units_by_symbol:
                existing_unit, existing_schema = units_by_symbol[unit.symbol]
                # Check if definitions are identical
                if not _units_equal(unit, existing_unit):
                    raise MergeConflict(
                        f"Conflict: unit '{unit.symbol}' defined differently in "
                        f"'{schema.name}' and '{existing_schema}'"
                    )
            else:
                units_by_symbol[unit.symbol] = (unit, schema.name)

    return [u for u, _ in units_by_symbol.values()]


def _merge_units_override(schemas: list[UnitSchema]) -> list[UnitDefinition]:
    """Merge units in OVERRIDE mode (later schemas win)."""
    units_by_symbol: dict[str, UnitDefinition] = {}

    for schema in schemas:
        for unit in schema.units:
            units_by_symbol[unit.symbol] = unit

    return list(units_by_symbol.values())


def _merge_units_namespace(schemas: list[UnitSchema]) -> list[UnitDefinition]:
    """Merge units in NAMESPACE mode (prefix with schema name on conflict)."""
    units_by_symbol: dict[str, tuple[UnitDefinition, str]] = {}
    result = []

    for schema in schemas:
        for unit in schema.units:
            if unit.symbol in units_by_symbol:
                existing_unit, existing_schema = units_by_symbol[unit.symbol]

                # Check if definitions are identical
                if _units_equal(unit, existing_unit):
                    # Same definition, no need to namespace
                    continue

                # Different definitions - need to namespace both
                # First time we see conflict, namespace the existing unit
                if existing_unit in result:
                    result.remove(existing_unit)
                    namespaced_existing = UnitDefinition(
                        symbol=f"{existing_schema}_{existing_unit.symbol}",
                        dimension=existing_unit.dimension,
                        scale=existing_unit.scale,
                        description=existing_unit.description,
                    )
                    result.append(namespaced_existing)
                    units_by_symbol[unit.symbol] = (
                        namespaced_existing,
                        existing_schema,
                    )

                # Namespace the new unit
                namespaced_unit = UnitDefinition(
                    symbol=f"{schema.name}_{unit.symbol}",
                    dimension=unit.dimension,
                    scale=unit.scale,
                    description=unit.description,
                )
                result.append(namespaced_unit)
            else:
                units_by_symbol[unit.symbol] = (unit, schema.name)
                result.append(unit)

    return result


def _merge_constants_strict(schemas: list[UnitSchema]) -> list[ConstantDefinition]:
    """Merge constants in STRICT mode (error on conflicts)."""
    constants_by_symbol: dict[str, tuple[ConstantDefinition, str]] = {}

    for schema in schemas:
        for const in schema.constants:
            if const.symbol in constants_by_symbol:
                existing_const, existing_schema = constants_by_symbol[const.symbol]
                # Check if definitions are identical
                if not _constants_equal(const, existing_const):
                    raise MergeConflict(
                        f"Conflict: constant '{const.symbol}' defined differently in "
                        f"'{schema.name}' and '{existing_schema}'"
                    )
            else:
                constants_by_symbol[const.symbol] = (const, schema.name)

    return [c for c, _ in constants_by_symbol.values()]


def _merge_constants_override(schemas: list[UnitSchema]) -> list[ConstantDefinition]:
    """Merge constants in OVERRIDE mode (later schemas win)."""
    constants_by_symbol: dict[str, ConstantDefinition] = {}

    for schema in schemas:
        for const in schema.constants:
            constants_by_symbol[const.symbol] = const

    return list(constants_by_symbol.values())


def _merge_constants_namespace(
    schemas: list[UnitSchema],
) -> list[ConstantDefinition]:
    """Merge constants in NAMESPACE mode (prefix with schema name on conflict)."""
    constants_by_symbol: dict[str, tuple[ConstantDefinition, str]] = {}
    result = []

    for schema in schemas:
        for const in schema.constants:
            if const.symbol in constants_by_symbol:
                existing_const, existing_schema = constants_by_symbol[const.symbol]

                # Check if definitions are identical
                if _constants_equal(const, existing_const):
                    # Same definition, no need to namespace
                    continue

                # Different definitions - need to namespace both
                # First time we see conflict, namespace the existing constant
                if existing_const in result:
                    result.remove(existing_const)
                    namespaced_existing = ConstantDefinition(
                        symbol=f"{existing_schema}_{existing_const.symbol}",
                        name=existing_const.name,
                        value=existing_const.value,
                        unit=existing_const.unit,
                        uncertainty=existing_const.uncertainty,
                        description=existing_const.description,
                    )
                    result.append(namespaced_existing)
                    constants_by_symbol[const.symbol] = (
                        namespaced_existing,
                        existing_schema,
                    )

                # Namespace the new constant
                namespaced_const = ConstantDefinition(
                    symbol=f"{schema.name}_{const.symbol}",
                    name=const.name,
                    value=const.value,
                    unit=const.unit,
                    uncertainty=const.uncertainty,
                    description=const.description,
                )
                result.append(namespaced_const)
            else:
                constants_by_symbol[const.symbol] = (const, schema.name)
                result.append(const)

    return result


def _units_equal(a: UnitDefinition, b: UnitDefinition) -> bool:
    """Check if two unit definitions are equal (ignoring description)."""
    return (
        a.symbol == b.symbol
        and a.dimension == b.dimension
        and abs(a.scale - b.scale) < 1e-10
    )


def _constants_equal(a: ConstantDefinition, b: ConstantDefinition) -> bool:
    """Check if two constant definitions are equal (ignoring description)."""
    return (
        a.symbol == b.symbol
        and a.name == b.name
        and abs(a.value - b.value) < 1e-15
        and a.unit == b.unit
        and abs(a.uncertainty - b.uncertainty) < 1e-15
    )

"""Tests for schema module."""

import json
import tempfile
from pathlib import Path

import pytest

from dimtensor.schema import (
    UnitSchema,
    load_schema,
    save_schema,
    merge_schemas,
    MergeStrategy,
    SchemaRegistry,
    validate_schema,
)
from dimtensor.schema.merge import MergeConflict
from dimtensor.schema.schema import UnitDefinition, ConstantDefinition
from dimtensor.schema.validation import ValidationError


# =============================================================================
# UnitSchema Tests
# =============================================================================


def test_unit_schema_creation():
    """Test creating a basic unit schema."""
    schema = UnitSchema(
        name="test_schema",
        version="1.0.0",
        description="Test schema",
        author="Test Author",
    )

    assert schema.name == "test_schema"
    assert schema.version == "1.0.0"
    assert schema.description == "Test schema"
    assert schema.author == "Test Author"
    assert len(schema.units) == 0
    assert len(schema.constants) == 0


def test_add_unit():
    """Test adding units to a schema."""
    schema = UnitSchema(name="test", version="1.0.0")
    schema.add_unit("AU", {"L": 1}, 1.496e11, "Astronomical unit")

    assert len(schema.units) == 1
    assert schema.units[0].symbol == "AU"
    assert schema.units[0].dimension == {"L": 1}
    assert schema.units[0].scale == 1.496e11
    assert schema.units[0].description == "Astronomical unit"


def test_add_constant():
    """Test adding constants to a schema."""
    schema = UnitSchema(name="test", version="1.0.0")
    schema.add_constant(
        "c", "speed of light", 299792458.0, "m/s", uncertainty=0.0
    )

    assert len(schema.constants) == 1
    assert schema.constants[0].symbol == "c"
    assert schema.constants[0].name == "speed of light"
    assert schema.constants[0].value == 299792458.0
    assert schema.constants[0].unit == "m/s"
    assert schema.constants[0].uncertainty == 0.0


def test_get_unit():
    """Test retrieving units from schema."""
    schema = UnitSchema(name="test", version="1.0.0")
    schema.add_unit("AU", {"L": 1}, 1.496e11)
    schema.add_unit("pc", {"L": 1}, 3.086e16)

    au = schema.get_unit("AU")
    assert au is not None
    assert au.symbol == "AU"

    missing = schema.get_unit("nonexistent")
    assert missing is None


def test_get_constant():
    """Test retrieving constants from schema."""
    schema = UnitSchema(name="test", version="1.0.0")
    schema.add_constant("c", "speed of light", 299792458.0, "m/s")
    schema.add_constant("G", "gravitational constant", 6.674e-11, "m^3/(kg s^2)")

    c = schema.get_constant("c")
    assert c is not None
    assert c.symbol == "c"

    missing = schema.get_constant("nonexistent")
    assert missing is None


def test_schema_to_dict():
    """Test converting schema to dictionary."""
    schema = UnitSchema(
        name="test",
        version="1.0.0",
        description="Test schema",
        author="Test",
    )
    schema.add_unit("AU", {"L": 1}, 1.496e11)

    data = schema.to_dict()
    assert data["name"] == "test"
    assert data["version"] == "1.0.0"
    assert data["description"] == "Test schema"
    assert len(data["units"]) == 1
    assert data["units"][0]["symbol"] == "AU"


def test_schema_from_dict():
    """Test creating schema from dictionary."""
    data = {
        "name": "test",
        "version": "1.0.0",
        "description": "Test schema",
        "author": "Test",
        "units": [
            {
                "symbol": "AU",
                "dimension": {"L": 1},
                "scale": 1.496e11,
                "description": "Astronomical unit",
            }
        ],
        "constants": [
            {
                "symbol": "c",
                "name": "speed of light",
                "value": 299792458.0,
                "unit": "m/s",
            }
        ],
    }

    schema = UnitSchema.from_dict(data)
    assert schema.name == "test"
    assert schema.version == "1.0.0"
    assert len(schema.units) == 1
    assert len(schema.constants) == 1


# =============================================================================
# Serialization Tests
# =============================================================================


def test_save_and_load_json():
    """Test saving and loading schema as JSON."""
    schema = UnitSchema(name="test", version="1.0.0")
    schema.add_unit("AU", {"L": 1}, 1.496e11)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = Path(f.name)

    try:
        save_schema(schema, temp_path, format="json")
        loaded = load_schema(temp_path)

        assert loaded.name == schema.name
        assert loaded.version == schema.version
        assert len(loaded.units) == 1
        assert loaded.units[0].symbol == "AU"
    finally:
        temp_path.unlink()


def test_save_and_load_yaml():
    """Test saving and loading schema as YAML."""
    pytest.importorskip("yaml")  # Skip if PyYAML not installed

    schema = UnitSchema(name="test", version="1.0.0")
    schema.add_unit("AU", {"L": 1}, 1.496e11)

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        temp_path = Path(f.name)

    try:
        save_schema(schema, temp_path, format="yaml")
        loaded = load_schema(temp_path)

        assert loaded.name == schema.name
        assert loaded.version == schema.version
        assert len(loaded.units) == 1
        assert loaded.units[0].symbol == "AU"
    finally:
        temp_path.unlink()


def test_load_nonexistent_file():
    """Test loading from nonexistent file raises error."""
    with pytest.raises(FileNotFoundError):
        load_schema("/nonexistent/path/schema.yaml")


def test_save_auto_format_detection():
    """Test automatic format detection from extension."""
    pytest.importorskip("yaml")

    schema = UnitSchema(name="test", version="1.0.0")

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        temp_path = Path(f.name)

    try:
        save_schema(schema, temp_path)  # Auto-detect YAML
        loaded = load_schema(temp_path)
        assert loaded.name == "test"
    finally:
        temp_path.unlink()


# =============================================================================
# Validation Tests
# =============================================================================


def test_validate_valid_schema():
    """Test validation passes for valid schema."""
    schema = UnitSchema(
        name="test",
        version="1.0.0",
        description="Test schema",
        author="Test",
    )
    schema.add_unit("AU", {"L": 1}, 1.496e11)

    warnings = validate_schema(schema)
    assert isinstance(warnings, list)


def test_validate_missing_name():
    """Test validation fails for schema without name."""
    schema = UnitSchema(name="", version="1.0.0")

    with pytest.raises(ValidationError, match="must have a name"):
        validate_schema(schema)


def test_validate_invalid_version():
    """Test validation fails for invalid version string."""
    schema = UnitSchema(name="test", version="invalid")

    with pytest.raises(ValidationError, match="semantic versioning"):
        validate_schema(schema)


def test_validate_duplicate_units():
    """Test validation fails for duplicate unit symbols."""
    schema = UnitSchema(name="test", version="1.0.0")
    schema.add_unit("AU", {"L": 1}, 1.496e11)
    schema.add_unit("AU", {"L": 1}, 1.5e11)  # Duplicate

    with pytest.raises(ValidationError, match="Duplicate unit symbols"):
        validate_schema(schema)


def test_validate_invalid_dimension_key():
    """Test validation fails for invalid dimension key."""
    schema = UnitSchema(name="test", version="1.0.0")
    schema.add_unit("bad", {"X": 1}, 1.0)  # X is not valid

    with pytest.raises(ValidationError, match="Invalid dimension key"):
        validate_schema(schema)


def test_validate_negative_scale():
    """Test validation fails for negative scale."""
    schema = UnitSchema(name="test", version="1.0.0")
    schema.add_unit("bad", {"L": 1}, -1.0)

    with pytest.raises(ValidationError, match="Scale must be positive"):
        validate_schema(schema)


def test_validate_negative_uncertainty():
    """Test validation fails for negative uncertainty."""
    schema = UnitSchema(name="test", version="1.0.0")
    schema.add_constant("bad", "bad constant", 1.0, "m", uncertainty=-0.1)

    with pytest.raises(ValidationError, match="Uncertainty must be non-negative"):
        validate_schema(schema)


# =============================================================================
# Merge Tests
# =============================================================================


def test_merge_schemas_no_conflict():
    """Test merging schemas with no conflicts."""
    schema1 = UnitSchema(name="schema1", version="1.0.0")
    schema1.add_unit("AU", {"L": 1}, 1.496e11)

    schema2 = UnitSchema(name="schema2", version="1.0.0")
    schema2.add_unit("pc", {"L": 1}, 3.086e16)

    merged = merge_schemas([schema1, schema2], strategy=MergeStrategy.STRICT)

    assert len(merged.units) == 2
    symbols = [u.symbol for u in merged.units]
    assert "AU" in symbols
    assert "pc" in symbols


def test_merge_strict_conflict():
    """Test strict merge raises error on conflict."""
    schema1 = UnitSchema(name="schema1", version="1.0.0")
    schema1.add_unit("AU", {"L": 1}, 1.496e11)

    schema2 = UnitSchema(name="schema2", version="1.0.0")
    schema2.add_unit("AU", {"L": 1}, 1.5e11)  # Different scale

    with pytest.raises(MergeConflict, match="defined differently"):
        merge_schemas([schema1, schema2], strategy=MergeStrategy.STRICT)


def test_merge_override_strategy():
    """Test override merge (later schema wins)."""
    schema1 = UnitSchema(name="schema1", version="1.0.0")
    schema1.add_unit("AU", {"L": 1}, 1.496e11)

    schema2 = UnitSchema(name="schema2", version="1.0.0")
    schema2.add_unit("AU", {"L": 1}, 1.5e11)  # Different scale

    merged = merge_schemas([schema1, schema2], strategy=MergeStrategy.OVERRIDE)

    assert len(merged.units) == 1
    assert merged.units[0].scale == 1.5e11  # schema2's value


def test_merge_namespace_strategy():
    """Test namespace merge (prefix with schema name)."""
    schema1 = UnitSchema(name="schema1", version="1.0.0")
    schema1.add_unit("AU", {"L": 1}, 1.496e11)

    schema2 = UnitSchema(name="schema2", version="1.0.0")
    schema2.add_unit("AU", {"L": 1}, 1.5e11)  # Different scale

    merged = merge_schemas([schema1, schema2], strategy=MergeStrategy.NAMESPACE)

    assert len(merged.units) == 2
    symbols = [u.symbol for u in merged.units]
    assert "schema1_AU" in symbols
    assert "schema2_AU" in symbols


def test_merge_empty_list():
    """Test merging empty list raises error."""
    with pytest.raises(ValueError, match="Cannot merge empty list"):
        merge_schemas([])


def test_merge_with_constants():
    """Test merging schemas with constants."""
    schema1 = UnitSchema(name="schema1", version="1.0.0")
    schema1.add_constant("c", "speed of light", 299792458.0, "m/s")

    schema2 = UnitSchema(name="schema2", version="1.0.0")
    schema2.add_constant("G", "gravitational constant", 6.674e-11, "m^3/(kg s^2)")

    merged = merge_schemas([schema1, schema2], strategy=MergeStrategy.STRICT)

    assert len(merged.constants) == 2
    symbols = [c.symbol for c in merged.constants]
    assert "c" in symbols
    assert "G" in symbols


# =============================================================================
# Registry Tests
# =============================================================================


def test_registry_install_and_load():
    """Test installing and loading schemas from registry."""
    pytest.importorskip("yaml")

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = SchemaRegistry(cache_dir=tmpdir)

        # Create and save a test schema
        schema = UnitSchema(name="test", version="1.0.0")
        schema.add_unit("AU", {"L": 1}, 1.496e11)

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_schema(schema, temp_path)

            # Install to registry
            schema_id = registry.install(temp_path)
            assert schema_id == "test@1.0.0"

            # Load from registry
            loaded = registry.load("test")
            assert loaded.name == "test"
            assert loaded.version == "1.0.0"
        finally:
            temp_path.unlink()


def test_registry_list_schemas():
    """Test listing installed schemas."""
    pytest.importorskip("yaml")

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = SchemaRegistry(cache_dir=tmpdir)

        # Install two schemas
        schema1 = UnitSchema(name="test1", version="1.0.0")
        schema2 = UnitSchema(name="test2", version="2.0.0")

        for schema in [schema1, schema2]:
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
                temp_path = Path(f.name)
            try:
                save_schema(schema, temp_path)
                registry.install(temp_path)
            finally:
                temp_path.unlink()

        # List schemas
        schemas = registry.list_schemas()
        assert len(schemas) >= 2
        names = [s["name"] for s in schemas]
        assert "test1" in names
        assert "test2" in names


def test_registry_remove_schema():
    """Test removing installed schema."""
    pytest.importorskip("yaml")

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = SchemaRegistry(cache_dir=tmpdir)

        # Install schema
        schema = UnitSchema(name="test", version="1.0.0")
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_schema(schema, temp_path)
            registry.install(temp_path)

            # Verify it's installed
            loaded = registry.load("test")
            assert loaded.name == "test"

            # Remove it
            registry.remove("test@1.0.0")

            # Verify it's gone
            with pytest.raises(KeyError):
                registry.load("test")
        finally:
            temp_path.unlink()


def test_registry_load_missing_schema():
    """Test loading nonexistent schema raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = SchemaRegistry(cache_dir=tmpdir)

        with pytest.raises(KeyError, match="Schema not found"):
            registry.load("nonexistent")


# =============================================================================
# Built-in Schema Tests
# =============================================================================


def test_load_builtin_astronomy_schema():
    """Test loading built-in astronomy schema."""
    pytest.importorskip("yaml")

    # Get path to built-in schema
    from dimtensor.schema import serialization

    schema_path = (
        Path(serialization.__file__).parent / "builtin" / "astronomy.yaml"
    )

    if schema_path.exists():
        schema = load_schema(schema_path)
        assert schema.name == "astronomy"
        assert len(schema.units) > 0

        # Check for some expected units
        au = schema.get_unit("AU")
        assert au is not None
        assert au.dimension == {"L": 1}


def test_load_builtin_chemistry_schema():
    """Test loading built-in chemistry schema."""
    pytest.importorskip("yaml")

    from dimtensor.schema import serialization

    schema_path = (
        Path(serialization.__file__).parent / "builtin" / "chemistry.yaml"
    )

    if schema_path.exists():
        schema = load_schema(schema_path)
        assert schema.name == "chemistry"
        assert len(schema.units) > 0


def test_load_builtin_engineering_schema():
    """Test loading built-in engineering schema."""
    pytest.importorskip("yaml")

    from dimtensor.schema import serialization

    schema_path = (
        Path(serialization.__file__).parent / "builtin" / "engineering.yaml"
    )

    if schema_path.exists():
        schema = load_schema(schema_path)
        assert schema.name == "engineering"
        assert len(schema.units) > 0


# =============================================================================
# Unit/Constant Definition Tests
# =============================================================================


def test_unit_definition_to_unit():
    """Test converting UnitDefinition to Unit."""
    unit_def = UnitDefinition(
        symbol="AU",
        dimension={"L": 1},
        scale=1.496e11,
        description="Astronomical unit",
    )

    unit = unit_def.to_unit()
    assert unit.symbol == "AU"
    assert unit.scale == 1.496e11
    assert unit.dimension.length == 1


def test_unit_definition_from_unit():
    """Test creating UnitDefinition from Unit."""
    from dimtensor.core.dimensions import Dimension
    from dimtensor.core.units import Unit

    unit = Unit("AU", Dimension(length=1), 1.496e11)
    unit_def = UnitDefinition.from_unit(unit, "Astronomical unit")

    assert unit_def.symbol == "AU"
    assert unit_def.dimension == {"L": 1.0}
    assert unit_def.scale == 1.496e11
    assert unit_def.description == "Astronomical unit"


def test_unit_definition_round_trip():
    """Test round-trip conversion: Unit -> UnitDefinition -> Unit."""
    from dimtensor.core.dimensions import Dimension
    from dimtensor.core.units import Unit

    original = Unit("AU", Dimension(length=1), 1.496e11)
    unit_def = UnitDefinition.from_unit(original)
    converted = unit_def.to_unit()

    assert converted.symbol == original.symbol
    assert converted.scale == original.scale
    assert converted.dimension == original.dimension

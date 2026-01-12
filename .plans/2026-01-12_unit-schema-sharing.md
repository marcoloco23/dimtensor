# Plan: Unit Schema Sharing for dimtensor

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Allow researchers to share and reuse custom unit definitions across projects by creating a standardized UnitSchema system with YAML/JSON serialization, versioning, and conflict resolution.

---

## Background

dimtensor currently has:
- Built-in SI units in `core/units.py` (meter, kilogram, etc.)
- Domain-specific units in `domains/` (astronomy, chemistry, engineering, etc.)
- Plugin system in `plugins/` for loading custom units
- Existing registry pattern in `hub/registry.py` for models and `datasets/registry.py` for datasets

However, there's no way for users to:
1. Package their custom units into a shareable format
2. Version control unit definitions
3. Merge unit schemas from multiple sources
4. Discover and import community-contributed unit sets

This feature will enable researchers to share domain-specific units (e.g., "nuclear_physics", "meteorology") just like they share datasets or models.

---

## Approach

### Option A: Extend Plugin System
- Description: Build on existing `plugins/` infrastructure, add schema export/import
- Pros:
  - Leverages existing plugin discovery mechanism
  - Already has validation framework
  - Python entry points work well for installed packages
- Cons:
  - Requires packaging as Python package for discovery
  - Not ideal for quick sharing (e.g., GitHub gist)
  - Harder to version control just the schema

### Option B: Standalone Schema System
- Description: Create new `schema/` module with YAML/JSON formats, independent registry
- Pros:
  - Lightweight: just a file, no packaging needed
  - Easy to version control (git-friendly)
  - Simple sharing via URLs, GitHub, etc.
  - Can still integrate with plugins later
- Cons:
  - Need new registry mechanism
  - More code to maintain

### Option C: Hybrid Approach
- Description: Create standalone schema system, but integrate with plugin system for discovery
- Pros:
  - Best of both worlds: simple file format + plugin discovery
  - Schemas can be distributed as files OR packages
  - Backward compatible with existing plugins
- Cons:
  - More complex implementation
  - Need clear documentation on when to use which approach

### Decision: Option B (Standalone Schema System)

Rationale:
1. Simplicity wins for sharing: YAML/JSON files are universally accessible
2. Git-friendly: schemas can be versioned in any repo
3. No packaging required: users can share via gist, URL, or copy-paste
4. Can add plugin integration later if needed (v2)
5. Follows model in `hub/registry.py` which uses JSON serialization

---

## Implementation Steps

1. [ ] Create `src/dimtensor/schema/` module structure
   - `__init__.py` - Public API
   - `schema.py` - UnitSchema class
   - `serialization.py` - YAML/JSON I/O
   - `registry.py` - Schema discovery and management
   - `validation.py` - Schema validation
   - `merge.py` - Conflict resolution when merging schemas

2. [ ] Implement UnitSchema class
   - Store units, constants, and custom dimensions
   - Include metadata (name, version, author, description)
   - Support semantic versioning (major.minor.patch)
   - Track dependencies on other schemas

3. [ ] Implement YAML/JSON serialization
   - Serialize Unit objects (symbol, dimension, scale)
   - Serialize Constant objects (name, value, unit, uncertainty)
   - Serialize custom Dimension definitions
   - Support both formats with automatic detection

4. [ ] Create SchemaRegistry class
   - Local registry at `~/.cache/dimtensor/schemas/`
   - Install schemas from local files, URLs, or built-in collection
   - List installed schemas
   - Load schema by name or name@version
   - Update/remove schemas

5. [ ] Implement schema validation
   - Check required fields (name, version)
   - Validate version string format (semver)
   - Validate unit definitions (no circular dependencies)
   - Check dimension consistency
   - Warn on potential conflicts

6. [ ] Implement schema merging with conflict resolution
   - Strategies: "strict" (error on conflict), "override" (last wins), "namespace" (prefix with schema name)
   - Detect conflicts: same unit name, different definition
   - Merge multiple schemas into composite schema
   - Track provenance of each unit

7. [ ] Add convenience functions
   - `load_schema(name_or_path)` - Load from registry or file
   - `save_schema(schema, path)` - Export to file
   - `merge_schemas([schema1, schema2], strategy)` - Combine multiple
   - `list_schemas()` - Show installed schemas
   - `install_schema(source)` - Add to registry

8. [ ] Create built-in schema collection
   - Extract existing domain modules as schemas
   - `astronomy.yaml` - From domains/astronomy.py
   - `chemistry.yaml` - From domains/chemistry.py
   - `engineering.yaml` - From domains/engineering.py
   - Ship in `src/dimtensor/schema/builtin/`

9. [ ] Add CLI support
   - `dimtensor schema list` - List installed schemas
   - `dimtensor schema install <source>` - Install schema
   - `dimtensor schema export <domain>` - Export domain to schema file
   - `dimtensor schema validate <file>` - Validate schema file
   - `dimtensor schema merge <file1> <file2> -o output.yaml`

10. [ ] Update documentation
    - Tutorial: Creating a custom schema
    - Tutorial: Sharing schemas via GitHub
    - API reference for schema module
    - Migration guide: plugins vs schemas

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/schema/__init__.py` | New: Public API exports |
| `src/dimtensor/schema/schema.py` | New: UnitSchema dataclass |
| `src/dimtensor/schema/serialization.py` | New: to_yaml/from_yaml, to_json/from_json |
| `src/dimtensor/schema/registry.py` | New: SchemaRegistry class |
| `src/dimtensor/schema/validation.py` | New: Schema validation functions |
| `src/dimtensor/schema/merge.py` | New: Merge strategies and conflict resolution |
| `src/dimtensor/schema/builtin/*.yaml` | New: Built-in schema collection |
| `src/dimtensor/__init__.py` | Add schema module exports |
| `src/dimtensor/cli/__main__.py` | Add schema subcommands |
| `pyproject.toml` | Add PyYAML dependency |

---

## Testing Strategy

- [ ] Unit tests for UnitSchema
  - Create schema with units, constants, dimensions
  - Validate metadata fields
  - Test version comparisons

- [ ] Unit tests for serialization
  - Round-trip: schema -> YAML -> schema
  - Round-trip: schema -> JSON -> schema
  - Test with complex units (composite, derived)
  - Test with constants (with/without uncertainty)

- [ ] Unit tests for registry
  - Install schema from file
  - Install schema from URL (mock HTTP)
  - List installed schemas
  - Load by name, load by name@version
  - Remove schema

- [ ] Unit tests for validation
  - Valid schema passes
  - Invalid version string rejected
  - Circular unit dependencies detected
  - Missing required fields caught

- [ ] Unit tests for merging
  - Merge compatible schemas (no conflicts)
  - Detect conflicts (same name, different definition)
  - Test "strict" strategy (errors on conflict)
  - Test "override" strategy (last wins)
  - Test "namespace" strategy (prefixed names)

- [ ] Integration tests
  - Create astronomy schema from domains/astronomy.py
  - Load schema and use units in DimArray
  - Merge astronomy + chemistry schemas
  - Export custom schema and reload

- [ ] CLI tests
  - Test all schema subcommands
  - Validate output format
  - Test error handling

---

## Risks / Edge Cases

- **Risk 1: Version conflicts**
  - Scenario: User has schemas depending on different versions of same schema
  - Mitigation: Registry supports multiple versions simultaneously, explicit version pinning

- **Risk 2: Circular dependencies**
  - Scenario: Schema A depends on B, B depends on C, C depends on A
  - Mitigation: Validation detects cycles, rejects schemas with circular deps

- **Risk 3: Unit name collisions**
  - Scenario: Multiple schemas define "bar" (unit of pressure vs. music measure)
  - Mitigation: Merge strategies handle this; namespace strategy prefixes with schema name

- **Risk 4: Scale factor precision**
  - Scenario: YAML/JSON may lose floating-point precision for scale factors
  - Mitigation: Use scientific notation in YAML, test round-trip precision

- **Edge case: Empty schema**
  - Handling: Allow empty schema (just metadata), useful for namespacing

- **Edge case: Schema with only constants**
  - Handling: Support schemas with units=[], constants=[...] for physical constant libraries

- **Edge case: URL schemas behind authentication**
  - Handling: Support local download first, then install from file (document workaround)

- **Edge case: Schema format evolution**
  - Handling: Include schema_version field (default 1.0), plan for future format changes

- **Risk 5: Large schemas**
  - Scenario: Schema with thousands of units (e.g., currency conversion)
  - Mitigation: Lazy loading, don't import all units upfront

- **Risk 6: Backward compatibility**
  - Scenario: Existing code uses domains.astronomy, we deprecate for schemas
  - Mitigation: Keep domain modules for now, mark as "legacy", schemas are additive

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] All tests pass (unit + integration + CLI)
- [ ] Test coverage > 85% for schema module
- [ ] Documentation written and reviewed
- [ ] Can create, save, load, and merge schemas via API
- [ ] Can install and use schemas via CLI
- [ ] Built-in schemas created for existing domains
- [ ] CONTINUITY.md updated with completion status

---

## Notes / Log

**2026-01-12 15:30** - Research complete. Key findings:
- Existing registry pattern in hub/registry.py provides good template
- Plugin system is more heavyweight than needed for sharing
- JSON serialization already used in io/json.py for DimArray
- Domain modules (domains/astronomy.py etc.) contain ~15 units each
- Model registry uses to_dict/from_dict pattern - we should follow same

**2026-01-12 15:35** - Decision made: Standalone schema system (Option B)
- Simpler for end users
- Git-friendly
- No packaging required
- Can add plugin integration later

**2026-01-12 15:40** - Schema format design:
```yaml
# Example: nuclear_physics.yaml
name: nuclear_physics
version: 1.0.0
author: J. Researcher
description: Units for nuclear and particle physics
license: MIT
dependencies:
  - chemistry@1.0.0  # Optional: depends on other schemas

units:
  - symbol: MeV
    dimension: {M: 1, L: 2, T: -2}
    scale: 1.602176634e-13
    description: Megaelectronvolt

  - symbol: barn
    dimension: {L: 2}
    scale: 1e-28
    description: Nuclear cross-section unit

constants:
  - symbol: m_proton
    name: proton mass
    value: 1.67262192369e-27
    unit: kg
    uncertainty: 5.1e-37
```

**2026-01-12 15:45** - CLI design:
- `dimtensor schema list` - Show installed
- `dimtensor schema install nuclear_physics.yaml`
- `dimtensor schema install https://example.com/schema.yaml`
- `dimtensor schema validate my_schema.yaml`
- `dimtensor schema merge astro.yaml chem.yaml -o combined.yaml --strategy namespace`

---

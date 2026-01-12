# Plan: Dataset Sharing Protocol for dimtensor

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create a standardized protocol for sharing physics datasets with complete unit metadata, enabling researchers to publish, discover, and validate dimensional datasets across multiple file formats (CSV, Parquet, HDF5, NetCDF).

---

## Background

The existing `datasets/registry.py` provides a Python-based registry for synthetic physics datasets with `DatasetInfo` metadata. However, there's no standard way to:

1. Export real datasets with dimensional metadata to common file formats
2. Validate that a loaded dataset matches its declared units
3. Share dataset metadata in a portable format (separate from code)
4. Coordinate systems and measurement uncertainties are not captured

This creates a gap for researchers who want to:
- Publish experimental data with proper dimensional annotations
- Share datasets across institutions using standard formats
- Validate data integrity before using in physics simulations
- Document measurement uncertainties and coordinate systems

---

## Approach

### Option A: JSON Sidecar Files
- Store metadata in `.json` files alongside data files (e.g., `data.csv` + `data.json`)
- Pros:
  - Simple to implement
  - Works with any data format
  - Human-readable metadata
  - Easy to version control
- Cons:
  - Two files to manage
  - No format-specific optimizations
  - Could get out of sync with data

### Option B: Embedded Metadata
- Store metadata within file formats that support it (HDF5 attributes, NetCDF attributes, Parquet schema)
- Pros:
  - Single file to manage
  - Atomic updates
  - Format-native metadata access
- Cons:
  - Not all formats support rich metadata (CSV)
  - Requires format-specific implementations
  - Less human-readable

### Option C: Hybrid Approach (SELECTED)
- Use embedded metadata where supported (HDF5, NetCDF, Parquet)
- Fallback to JSON sidecar for CSV and other formats
- Provide unified `DimDatasetCard` abstraction
- Pros:
  - Best of both worlds
  - Format-agnostic API
  - Handles all required formats
  - Extensible to new formats
- Cons:
  - More implementation complexity
  - Two code paths to maintain

### Decision: Option C (Hybrid Approach)

Rationale:
- The existing I/O modules (hdf5.py, netcdf.py, parquet.py) already store dimensional metadata in format-specific ways
- CSV files need external metadata (no native metadata support)
- A unified `DimDatasetCard` provides consistent API regardless of storage format
- Validation can be standardized across all formats

---

## Implementation Steps

1. [x] Research existing patterns (datasets/registry.py, hub/cards.py, io/*.py)
2. [ ] Create `datasets/card.py` with `DimDatasetCard` dataclass
3. [ ] Add column-level metadata (units, uncertainties, coordinate systems)
4. [ ] Implement validation functions (validate_units, validate_schema)
5. [ ] Extend existing I/O modules to save/load dataset cards:
   - `io/csv.py` - New module with JSON sidecar support
   - `io/hdf5.py` - Extend with dataset card attributes
   - `io/netcdf.py` - Extend with dataset card attributes
   - `io/parquet.py` - Extend with dataset card in schema metadata
6. [ ] Add coordinate system support (Cartesian, spherical, cylindrical)
7. [ ] Add measurement uncertainty tracking per column
8. [ ] Create CLI command: `dimtensor dataset create-card <data_file>`
9. [ ] Create CLI command: `dimtensor dataset validate <data_file>`
10. [ ] Add comprehensive tests (validation, format round-trips)
11. [ ] Update documentation with dataset sharing workflow

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/datasets/card.py` | NEW: DimDatasetCard, ColumnInfo, CoordinateSystem |
| `src/dimtensor/datasets/validation.py` | NEW: validate_dataset, validate_schema, check_units |
| `src/dimtensor/io/csv.py` | NEW: save_csv_with_card, load_csv_with_card (JSON sidecar) |
| `src/dimtensor/io/hdf5.py` | EXTEND: save_dataset_with_card, load_dataset_with_card |
| `src/dimtensor/io/netcdf.py` | EXTEND: save_dataset_with_card, load_dataset_with_card |
| `src/dimtensor/io/parquet.py` | EXTEND: save_dataset_with_card, load_dataset_with_card |
| `src/dimtensor/io/__init__.py` | UPDATE: Export new functions |
| `src/dimtensor/datasets/__init__.py` | UPDATE: Export DimDatasetCard |
| `src/dimtensor/__main__.py` | EXTEND: Add dataset card CLI commands |
| `tests/test_dataset_card.py` | NEW: Test dataset card creation, validation |
| `tests/test_dataset_io.py` | NEW: Test format-specific card persistence |

---

## Testing Strategy

How will we verify this works?

- [ ] Unit tests for `DimDatasetCard` creation and serialization
- [ ] Unit tests for validation (correct units pass, incorrect units fail)
- [ ] Integration tests for each format (CSV, HDF5, NetCDF, Parquet):
  - Create dataset with card
  - Save to file
  - Load from file
  - Validate metadata matches
- [ ] Test coordinate system transformations
- [ ] Test uncertainty propagation through loading
- [ ] Test validation error messages are helpful
- [ ] Manual verification: Create sample physics dataset, share it, load it

---

## Risks / Edge Cases

### Risk 1: Metadata size limits in file formats
- Mitigation: Keep card metadata compact, use references to external docs for large descriptions

### Risk 2: Breaking changes to existing I/O functions
- Mitigation: Add new functions (save_dataset_with_card) instead of modifying existing save_* functions

### Risk 3: Version compatibility of dataset cards
- Mitigation: Include version field in card, support migration from older versions

### Edge case: Mixed units in multi-column datasets
- Handling: ColumnInfo per column, validate each independently

### Edge case: Time series with irregular sampling
- Handling: Support flexible coordinate specifications (not just uniform grids)

### Edge case: Large datasets (>1GB)
- Handling: Card metadata is separate from data, can be lightweight

### Edge case: Nested/hierarchical data structures
- Handling: Support HDF5 groups, NetCDF groups with nested cards

---

## Definition of Done

- [ ] DimDatasetCard implemented with full metadata support
- [ ] Validation functions correctly identify unit mismatches
- [ ] All 4 formats (CSV, HDF5, NetCDF, Parquet) support dataset cards
- [ ] CLI commands work for create-card and validate
- [ ] Tests pass (50+ new tests expected)
- [ ] Documentation updated with examples
- [ ] CONTINUITY.md updated with task completion

---

## Notes / Log

**2026-01-12 15:00** - Started planning
- Reviewed existing registry.py (DatasetInfo) and cards.py (ModelCard) patterns
- Reviewed I/O modules (hdf5, netcdf, parquet) for metadata storage patterns
- Key insight: Each format stores metadata differently, need unified abstraction

**Design decisions**:
1. `DimDatasetCard` is richer than `DatasetInfo`:
   - Per-column units (not just features/targets)
   - Coordinate systems (spatial, temporal)
   - Measurement uncertainties per column
   - Validation rules

2. Validation is separate from loading:
   - `load_csv_with_card()` loads data + metadata
   - `validate_dataset(card, data)` checks units match

3. Coordinate systems support:
   - Cartesian: (x, y, z)
   - Spherical: (r, theta, phi)
   - Cylindrical: (r, theta, z)
   - Custom: user-defined with unit specifications

4. Build on existing patterns:
   - Similar to ModelCard structure (cards.py)
   - Use existing dimension storage patterns (io/hdf5.py lines 56-65)
   - Leverage existing BaseLoader for cache (loaders/base.py)

---

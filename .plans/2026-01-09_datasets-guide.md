# Plan: Datasets Guide Documentation

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner-agent

---

## Goal

Create comprehensive documentation (docs/guide/datasets.md) that teaches users how to work with physics datasets in dimtensor, including built-in synthetic datasets, real-world data loaders (NIST, NASA, PRISM), custom dataset creation, and caching mechanisms.

---

## Background

The dimtensor library includes a powerful datasets module for physics-aware machine learning:
- **Registry system**: Metadata-driven dataset discovery with dimensional annotations
- **Built-in datasets**: 10 synthetic physics datasets (pendulum, projectile, Navier-Stokes, etc.)
- **Real-world loaders**: NIST CODATA constants, NASA exoplanets, PRISM climate data
- **Caching infrastructure**: BaseLoader with automatic download caching
- **Custom loader API**: BaseLoader and CSVLoader base classes for extension

Currently, there is no user-facing guide for this functionality. Users need documentation that:
1. Shows how to discover and use built-in datasets
2. Demonstrates loading real physics data with proper units
3. Explains the caching system and cache management
4. Provides patterns for creating custom dataset loaders

---

## Approach

### Option A: API-First Documentation
- Focus on API reference with minimal examples
- Pros: Complete coverage, easy to maintain
- Cons: Not beginner-friendly, lacks real-world context

### Option B: Tutorial-Style with Progressive Examples
- Start with simple use cases, build to advanced topics
- Multiple complete examples showing different dataset types
- Dedicated sections for each loader type (NIST, NASA, PRISM)
- Advanced section on custom loaders
- Pros: Easy to follow, practical, matches existing guide style
- Cons: Longer document, more examples to maintain

### Option C: Hybrid Approach (RECOMMENDED)
- Quick start with common use cases
- Detailed sections organized by dataset category
- Separate advanced topics (custom loaders, caching)
- Many runnable examples throughout
- Pros: Balances accessibility with completeness
- Cons: Requires careful organization

### Decision: Option C (Hybrid Approach)

This matches the style of existing guides (units.md, examples.md) and provides both quick wins for beginners and depth for advanced users.

---

## Implementation Steps

1. [x] Research existing datasets code (registry, loaders, caching)
2. [x] Analyze existing documentation style
3. [ ] Create docs/guide/datasets.md with structure:
   - Introduction and imports
   - Section 1: Discovering Datasets (list_datasets, get_dataset_info)
   - Section 2: Built-in Synthetic Datasets
     - Physics simulations (pendulum, spring_mass, projectile)
     - PDE datasets (heat_diffusion, burgers, wave_1d)
     - Multi-body systems (three_body, lorenz)
   - Section 3: Real-World Data Loaders
     - NIST CODATA fundamental constants
     - NASA Exoplanet Archive
     - PRISM climate data
   - Section 4: Working with Loaded Data
     - Data structures returned by loaders
     - Unit conversions
     - Integration with NumPy/PyTorch/JAX
   - Section 5: Caching System
     - Cache directory location
     - Force re-download
     - Cache management (clearing, inspection)
   - Section 6: Creating Custom Loaders
     - Extending BaseLoader
     - Extending CSVLoader
     - Registering custom datasets
     - Example: Custom CSV loader with units
4. [ ] Write 12-15 complete, runnable code examples
5. [ ] Add cross-references to related guides (PyTorch, JAX, units)
6. [ ] Test all code examples for correctness
7. [ ] Update docs/index.md or navigation to link to new guide

---

## Files to Modify

| File | Change |
|------|--------|
| docs/guide/datasets.md | **CREATE** - Main datasets guide (new file, ~400-600 lines) |
| docs/index.md | **UPDATE** - Add link to datasets guide in navigation/TOC |

---

## Testing Strategy

How will we verify this works?

- [ ] All code examples must be syntactically valid Python
- [ ] Examples should use realistic, working code (may use synthetic data if loaders unavailable)
- [ ] Run Python interpreter check on all code blocks
- [ ] Verify cross-references point to existing documentation
- [ ] Manual review: Read through as a new user would
- [ ] Verify examples match actual API from tests/test_datasets.py and tests/test_dataset_loaders.py

---

## Risks / Edge Cases

- **Risk 1**: Real-world loaders (NASA, PRISM) may fail if network unavailable or APIs change
  - **Mitigation**: Document fallback behavior, show cache usage, note that loaders return synthetic data when download fails

- **Risk 2**: Examples may become outdated as API evolves
  - **Mitigation**: Keep examples simple and focused on stable API (list_datasets, load_dataset, register_dataset)

- **Risk 3**: Cache management examples might encourage users to delete cache incorrectly
  - **Mitigation**: Add clear warnings, show both targeted and bulk cache clearing

- **Edge Case**: Users on Windows with different cache paths
  - **Handling**: Document DIMTENSOR_CACHE_DIR environment variable, show Path examples

- **Edge Case**: Users without requests library installed
  - **Handling**: Document the ImportError and show pip install command

---

## Definition of Done

- [ ] docs/guide/datasets.md created with all sections complete
- [ ] 12-15 working code examples included
- [ ] Cross-references to other guides added
- [ ] Navigation/TOC updated
- [ ] All code examples tested (syntax check minimum)
- [ ] CONTINUITY.md updated with completion status

---

## Notes / Log

**[Initial Research]** - Dataset module has excellent structure:
- 10 built-in datasets with metadata (features, targets, dimensions, tags)
- 3 real-world loaders (NIST, NASA, PRISM) with fallback data
- BaseLoader provides download() with MD5 caching
- CSVLoader extends BaseLoader for CSV parsing
- Registry supports filtering by domain and tags

**[Documentation Style]** - Existing guides use:
- Code-first approach with working examples
- Clear section headings
- Markdown admonitions (!!! warning)
- Import statements at top of each section
- Print statements to show expected output

**[Key Examples Needed]**:
1. List all datasets and filter by domain
2. Get metadata for a specific dataset
3. Load synthetic pendulum dataset
4. Load NIST constants and use in calculation
5. Load NASA exoplanets and analyze distribution
6. Load PRISM climate data and plot trend
7. Check cache location and contents
8. Force re-download with cache=False
9. Create custom CSV loader
10. Register custom dataset with metadata

---

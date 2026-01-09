# Plan: Basics Tutorial Notebook (01_basics.ipynb)

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create an interactive Jupyter notebook that teaches new users the fundamentals of dimtensor's DimArray class, including creation, unit conversions, arithmetic operations, dimensional safety, physical constants, and uncertainty propagation.

---

## Background

dimtensor currently has comprehensive test coverage and documentation, but lacks hands-on tutorial materials for new users. A basics notebook will:
- Lower the barrier to entry for new users
- Demonstrate core features with runnable examples
- Show common patterns and use cases
- Serve as a reference for typical workflows

This is the first in a planned series of example notebooks (task #151 from ROADMAP).

---

## Approach

### Option A: Comprehensive Tutorial (30+ cells)
- Cover all core features in depth
- Include extensive explanations and theory
- Multiple examples per concept
- Pros: Thorough, educational, complete reference
- Cons: May be overwhelming for quick start, longer to complete

### Option B: Quick Start Guide (15-20 cells)
- Focus on essential features only
- Minimal explanations, more code
- One example per concept
- Pros: Fast to complete, easy to follow
- Cons: Users may need to refer to docs frequently

### Option C: Progressive Tutorial (20-30 cells)
- Start simple, gradually increase complexity
- Balance explanations with code examples
- Build on previous concepts
- Pros: Natural learning progression, covers essentials without overwhelming
- Cons: Middle ground may not satisfy either extreme preference

### Decision: Option C - Progressive Tutorial

A 20-30 cell progressive tutorial strikes the right balance:
- New users can follow along without feeling overwhelmed
- Covers all essential features (creation, operations, safety, constants, uncertainty)
- Each section builds naturally on previous concepts
- Provides practical, runnable examples users can modify
- Length is appropriate for a "basics" tutorial (15-30 minutes to work through)

---

## Implementation Steps

1. [ ] Create examples/ directory
2. [ ] Create 01_basics.ipynb with initial metadata and structure
3. [ ] Section 1: Installation and imports (2 cells)
   - Installation instructions
   - Import statements
4. [ ] Section 2: Creating DimArrays (4-5 cells)
   - From lists and numpy arrays
   - With different units
   - Scalar and multidimensional arrays
   - Inspecting properties (shape, dtype, unit, dimension)
5. [ ] Section 3: Available units (2-3 cells)
   - SI base units
   - Derived units (N, J, W, Pa, etc.)
   - Common non-SI units (km, hour, eV, etc.)
6. [ ] Section 4: Unit conversions (3-4 cells)
   - Using .to() method
   - Converting between compatible units
   - to_base_units() method
   - Show UnitConversionError when attempting incompatible conversion
7. [ ] Section 5: Arithmetic operations (4-5 cells)
   - Addition/subtraction (same dimension required)
   - Multiplication/division (dimensions combine)
   - Power operations
   - Scalar operations
8. [ ] Section 6: Dimensional safety (3-4 cells)
   - Show DimensionError examples (velocity + acceleration)
   - Show DimensionError for incompatible operations
   - Demonstrate how this catches physics errors early
   - Show numpy ufunc errors (np.sin on non-dimensionless)
9. [ ] Section 7: Physical constants (3-4 cells)
   - Import and use constants (c, G, h, e, k_B, etc.)
   - Show constant properties (value, uncertainty, is_exact)
   - Use constants in calculations
   - Example: Calculate Schwarzschild radius or de Broglie wavelength
10. [ ] Section 8: Uncertainty propagation (3-4 cells)
    - Creating DimArrays with uncertainty
    - Uncertainty through arithmetic operations
    - Relative uncertainty
    - Using constants with uncertainty
11. [ ] Section 9: Array operations (2-3 cells)
    - Indexing and slicing
    - Reductions (sum, mean, min, max)
    - Reshaping operations
12. [ ] Section 10: Working with numpy (2-3 cells)
    - numpy ufuncs (np.sqrt, np.abs, etc.)
    - Unit-preserving operations
    - Dimensionless-requiring functions (np.sin, np.exp)
13. [ ] Section 11: Summary and next steps (1 cell)
    - Recap of key concepts
    - Links to other examples (when available)
    - Links to documentation
14. [ ] Test notebook execution from start to finish
15. [ ] Verify all cells run without errors

---

## Files to Modify

| File | Change |
|------|--------|
| examples/01_basics.ipynb | Create new tutorial notebook |
| examples/README.md | Create index of example notebooks (optional, can be added later) |

---

## Testing Strategy

How will we verify this works?

- [ ] Execute notebook from clean state (Restart & Run All)
- [ ] Verify all code cells execute without errors
- [ ] Verify all intentional errors (DimensionError examples) are caught properly
- [ ] Test with both numpy 1.x and 2.x (if needed)
- [ ] Verify output formatting is clear and readable
- [ ] Have someone unfamiliar with dimtensor follow the tutorial (if possible)
- [ ] Check that all imports work correctly

---

## Risks / Edge Cases

- **Risk**: Notebook becomes too long or complex
  - Mitigation: Stick to 20-30 cells, focus on essentials, save advanced topics for other notebooks

- **Risk**: Examples fail due to version incompatibilities
  - Mitigation: Test with current dependencies, avoid features from very recent Python/NumPy versions

- **Risk**: Users might not have Jupyter installed
  - Mitigation: Include installation instructions at the top, mention alternatives (VS Code, Google Colab)

- **Edge case**: Constants module has many constants - which to showcase?
  - Handling: Pick well-known constants (c, G, h, e, k_B) that users recognize, mention others are available

- **Edge case**: Uncertainty propagation might be confusing for beginners
  - Handling: Keep examples simple (1-2 operations), explain the sigma_z formula briefly

- **Risk**: Examples directory doesn't exist yet
  - Mitigation: Create it as first step, verify with ls

---

## Definition of Done

- [x] All implementation steps complete (notebook created with all sections)
- [x] Notebook executes successfully from start to finish
- [x] All code cells produce expected output
- [x] Intentional errors are demonstrated properly with try/except blocks
- [x] Markdown cells provide clear explanations
- [x] Length is appropriate (20-30 cells)
- [x] CONTINUITY.md updated with completion status

---

## Notes / Log

**Target audience**: New users with basic Python and NumPy knowledge, but no experience with unit-aware libraries.

**Learning objectives**:
- Understand what DimArray is and why it's useful
- Know how to create DimArrays with different units
- Perform unit conversions
- Use arithmetic operations correctly
- Understand dimensional safety and error messages
- Work with physical constants
- Track measurement uncertainties
- Apply array operations while preserving units

**Narrative flow**:
1. Start with "why" - show the problem dimtensor solves
2. Show simple creation and inspection
3. Introduce operations gradually (conversion → arithmetic → safety)
4. Add constants as real-world examples
5. Show uncertainty as advanced feature
6. Demonstrate array operations
7. End with numpy integration

**Style guidelines**:
- Use clear, concise markdown explanations
- Show output for every code cell
- Use print() statements to make output explicit
- Add comments in code where helpful
- Use real physics examples (projectile motion, energy calculations, etc.)
- Keep variable names readable (velocity, distance, not just v, d)

---

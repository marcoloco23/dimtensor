# Plan: Geophysics Units Module

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create a geophysics domain module providing units commonly used in geophysics, seismology, petroleum engineering, and geomagnetism including acceleration (gal), gravity gradient (eotvos), permeability (darcy), magnetic field (gamma, oersted), and seismic magnitude helpers.

---

## Background

Geophysicists work with specialized units that differ from standard SI units, particularly in gravimetry, magnetometry, and petroleum engineering. Many of these units (gal, eotvos, gamma) are CGS-derived but remain widely used in modern geophysics literature. The darcy is the standard unit in petroleum reservoir engineering. This module will enable dimensional correctness for geophysical calculations similar to the existing astronomy, chemistry, and engineering domain modules.

---

## Approach

### Option A: Units Only (No Helper Functions)
- Define all geophysical units as Unit objects
- Exclude Richter magnitude helpers (as they require logarithmic transformations)
- Keep the module simple and consistent with existing domain modules
- Pros:
  - Consistent with existing domain module pattern
  - Simple implementation
  - Easy to test
- Cons:
  - Richter magnitude requested by user cannot be fully supported
  - May need future expansion

### Option B: Units + Helper Functions
- Define geophysical units as Unit objects
- Add helper functions module for Richter magnitude conversions
- Include `magnitude_to_energy()` and `energy_to_magnitude()` functions
- Pros:
  - Complete feature set matching user requirements
  - More useful for seismology applications
- Cons:
  - Deviates from simple unit-only pattern
  - Helper functions may belong in separate utilities module
  - More complex testing

### Decision: Option A with documentation note

Start with units-only approach for consistency with existing domain modules. Add a docstring note about Richter magnitude being logarithmic and suggesting a future `dimtensor.geophysics.seismic` submodule for magnitude-energy conversions. This keeps the initial implementation clean while acknowledging the limitation.

---

## Implementation Steps

1. [ ] Create `/home/user/dimtensor/src/dimtensor/domains/geophysics.py`
2. [ ] Add module docstring with examples and references
3. [ ] Import required dependencies (Dimension, DIMENSIONLESS, Unit)
4. [ ] Define acceleration units section:
   - gal (1 cm/s² = 0.01 m/s²)
   - milligal (0.001 gal = 1e-5 m/s²)
5. [ ] Define gravity gradient section:
   - eotvos (1 E = 1e-9 s^-2, dimension T^-2)
6. [ ] Define permeability section:
   - darcy (9.869233e-13 m², dimension L^2)
   - millidarcy (9.869233e-16 m²)
7. [ ] Define magnetic field section:
   - gamma (1 nT = 1e-9 T, dimension M*T^-2*I^-1)
   - oersted (79.5774715 A/m, dimension I*L^-1)
8. [ ] Add note about seismic magnitude in docstring
9. [ ] Create comprehensive `__all__` export list
10. [ ] Add tests to `/home/user/dimtensor/tests/test_domains.py`:
    - TestGeophysicsUnits class
    - Dimension checks for all units
    - Scale factor verification
    - Conversion tests (gal to m/s², darcy to m², etc.)
    - Import tests
11. [ ] Run pytest to verify all tests pass
12. [ ] Update `/home/user/dimtensor/src/dimtensor/domains/__init__.py` if needed

---

## Files to Modify

| File | Change |
|------|--------|
| /home/user/dimtensor/src/dimtensor/domains/geophysics.py | Create new module with geophysics units |
| /home/user/dimtensor/tests/test_domains.py | Add TestGeophysicsUnits class with comprehensive tests |
| /home/user/dimtensor/src/dimtensor/domains/__init__.py | Add geophysics module to imports (if not already auto-imported) |

---

## Testing Strategy

How will we verify this works?

- [ ] Unit dimension tests: Verify each unit has correct SI dimension
  - gal, milligal: Dimension(length=1, time=-2)
  - eotvos: Dimension(time=-2)
  - darcy, millidarcy: Dimension(length=2)
  - gamma: Dimension(mass=1, time=-2, current=-1)
  - oersted: Dimension(current=1, length=-1)
- [ ] Scale factor tests: Verify conversion factors are correct
  - gal = 0.01 m/s²
  - milligal = 1e-5 m/s²
  - eotvos = 1e-9 s^-2
  - darcy ≈ 9.869233e-13 m²
  - millidarcy ≈ 9.869233e-16 m²
  - gamma = 1e-9 T
  - oersted ≈ 79.5774715 A/m
- [ ] Conversion tests: Test unit conversions using DimArray
  - gal to m/s² and back
  - darcy to m² and back
  - gamma to tesla and back
- [ ] Alias tests: Verify short aliases work correctly
- [ ] Import tests: Verify module is importable from dimtensor.domains
- [ ] Integration test: Create realistic geophysics calculation (e.g., permeability calculation, gravity anomaly)

---

## Risks / Edge Cases

- **Risk 1**: CGS unit conversion accuracy
  - Mitigation: Use high-precision constants from authoritative references (NIST, SPE)

- **Risk 2**: Oersted conversion complexity (CGS electromagnetic unit)
  - Mitigation: Use standard conversion factor 1 Oe = (1000/4π) A/m ≈ 79.5774715 A/m

- **Risk 3**: User expectation for Richter magnitude functionality
  - Mitigation: Add clear docstring note explaining logarithmic nature and suggest future enhancement

- **Edge case**: Eotvos unit has unusual dimension (T^-2)
  - Handling: Document clearly in comments, ensure Dimension(time=-2) is used correctly

- **Edge case**: Darcy represents permeability (area), not pressure
  - Handling: Document this clearly as it's counterintuitive; include reference to petroleum engineering context

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Module created with all requested units (except Richter helpers)
- [ ] All tests pass (pytest)
- [ ] Type checking passes (mypy)
- [ ] Documentation includes:
  - Module docstring with usage examples
  - Reference sources for conversion factors
  - Note about Richter magnitude limitation
- [ ] Module is importable from dimtensor.domains.geophysics
- [ ] CONTINUITY.md updated with task completion

---

## Notes / Log

**2026-01-09** - Plan created by planner agent

Key references for implementation:
- Gal: Named after Galileo, standard unit in gravimetry (1 Gal = 1 cm/s²)
- Eötvös: Named after Loránd Eötvös, used in gravity gradiometry
- Darcy: Named after Henry Darcy, SPE standard for permeability
- Gamma: Common in geomagnetic surveys (1 γ = 1 nT)
- Oersted: CGS electromagnetic unit, still used in paleomagnetism

Pattern follows existing domain modules (astronomy.py, chemistry.py, engineering.py):
- Organized by measurement type with section headers
- Include both long names and short aliases
- Provide detailed docstrings with examples
- Include authoritative references
- Comprehensive __all__ list for clean imports

---

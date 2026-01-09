# Plan: Materials Science Units Module

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create a materials science domain module providing units for stress/strain, hardness, fracture mechanics, thermal conductivity, and electrical conductivity to enable materials characterization and property calculations.

---

## Background

Materials scientists and engineers need specialized units for characterizing material properties. While some pressure units (GPa, MPa, kPa) exist in engineering.py, materials science requires additional units for:
- Strain measurements (dimensionless, microstrain)
- Hardness scales (Vickers, Rockwell, Brinell)
- Fracture toughness (stress intensity factor: MPa√m)
- Thermal and electrical conductivity

These units are essential for materials testing, quality control, and research.

---

## Approach

### Option A: Standalone materials.py module
- Create `domains/materials.py` with all materials-specific units
- Import pressure units from engineering.py for stress
- Pros: Clear separation of materials science domain, avoids duplication
- Cons: Requires importing from engineering module

### Option B: Self-contained materials.py
- Define all units within materials.py, including stress units
- Pros: Module is independent, easier to use standalone
- Cons: Duplicates GPa, MPa, kPa from engineering.py

### Decision: Option A (Import from engineering)
Use imports from engineering.py to avoid duplication of pressure units. Materials science builds on engineering fundamentals, so this dependency is natural. We'll re-export stress units in materials.py __all__ for convenience.

---

## Implementation Steps

1. [ ] Create `src/dimtensor/domains/materials.py` with sections:

   **Stress Units (re-exported from engineering):**
   - Import GPa, MPa, kPa from engineering module
   - Re-export in __all__ for convenience

   **Strain Units (dimensionless):**
   - strain: dimensionless (scale=1.0) - base unit
   - microstrain: dimensionless (scale=1e-6) - με, common in testing
   - percent_strain: dimensionless (scale=0.01) - for reporting

   **Hardness Units (dimensionless scales):**
   - vickers: HV (dimensionless, scale=1.0) - diamond pyramid indenter
   - brinell: HB (dimensionless, scale=1.0) - ball indenter
   - rockwell_C: HRC (dimensionless, scale=1.0) - conical indenter, hardened steel
   - rockwell_B: HRB (dimensionless, scale=1.0) - ball indenter, softer materials
   - Note: Hardness is technically force/area but reported as dimensionless scales

   **Fracture Toughness:**
   - MPa_sqrt_m: MPa·m^(1/2) (stress intensity factor K_IC)
     - Dimension: mass=1, length=-1/2, time=-2
     - Scale: 1e6 (from MPa base)
   - ksi_sqrt_in: ksi·in^(1/2) (imperial equivalent)
     - Conversion: 1 ksi√in = 1.099 MPa√m

   **Thermal Conductivity:**
   - W_per_m_K: W/(m·K) - SI unit for thermal conductivity
     - Dimension: mass=1, length=1, time=-3, temperature=-1
     - Scale: 1.0
   - W_per_cm_K: W/(cm·K) - common in materials handbooks
     - Scale: 100.0 (100 W/(m·K))

   **Electrical Conductivity:**
   - S_per_m: siemens per meter (S/m) - SI unit
     - Dimension: mass=-1, length=-3, time=3, current=2
     - Scale: 1.0
   - MS_per_m: megasiemens per meter (MS/m) - for highly conductive materials
     - Scale: 1e6

   **Electrical Resistivity (inverse of conductivity):**
   - ohm_m: Ω·m - SI unit for resistivity
     - Dimension: mass=1, length=3, time=-3, current=-2
     - Scale: 1.0
   - ohm_cm: Ω·cm - common in materials testing
     - Scale: 0.01 (1e-2 Ω·m)
   - microohm_cm: μΩ·cm - for low-resistivity materials
     - Scale: 1e-8

2. [ ] Update `src/dimtensor/domains/__init__.py` to expose materials module

3. [ ] Add comprehensive tests in `tests/test_domains.py`:
   - TestMaterialsScienceUnits class
   - Dimension checks for all units
   - Scale factor verification
   - Cross-unit conversions (MPa√m ↔ ksi√in)
   - Integration with engineering units

4. [ ] Add docstring examples showing typical use cases

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/domains/materials.py | CREATE - materials science units module |
| src/dimtensor/domains/__init__.py | UPDATE - add materials to imports and __all__ |
| tests/test_domains.py | UPDATE - add TestMaterialsScienceUnits class |

---

## Testing Strategy

- [ ] Test dimension correctness for each unit type:
  - Strain: dimensionless
  - Hardness: dimensionless
  - Fracture toughness: MPa·m^(1/2) dimension
  - Thermal conductivity: W/(m·K) dimension
  - Electrical conductivity: S/m dimension

- [ ] Test scale factors:
  - microstrain = 1e-6 of base strain
  - W_per_cm_K = 100 × W_per_m_K
  - MS_per_m = 1e6 × S_per_m

- [ ] Test unit conversions:
  - MPa√m to ksi√in (fracture toughness)
  - S/m to Ω·m (conductivity ↔ resistivity relationship)
  - Strain to microstrain to percent_strain

- [ ] Test cross-domain usage:
  - Import stress from materials, verify same as engineering.MPa
  - Calculate fracture stress from K_IC and crack length
  - Combine thermal and electrical conductivity (Wiedemann-Franz law)

- [ ] Test imports:
  - `from dimtensor.domains.materials import MPa, strain, vickers`
  - `from dimtensor.domains import materials`

---

## Risks / Edge Cases

- **Risk**: Hardness scales (HV, HRC, HRB) have different measurement procedures and are not directly convertible.
  - **Mitigation**: Define each as separate dimensionless units with scale=1.0. Document that conversions between hardness scales are empirical and domain-specific, not dimensional.

- **Risk**: Fracture toughness with fractional exponents (m^(1/2)) may cause issues with Fraction-based Dimension class.
  - **Mitigation**: Verify that Dimension supports Fraction(1, 2) for length exponent. Test this early.

- **Risk**: Conductivity and resistivity are inverses (σ = 1/ρ), but have different dimensions.
  - **Mitigation**: Document relationship clearly. Users must use `1/value` for conversion, not unit conversion.

- **Risk**: Importing from engineering.py creates module dependency.
  - **Mitigation**: This is acceptable - materials science builds on engineering. Alternative would be duplicate definitions.

- **Edge case**: Thermal conductivity in cryogenics uses mW/(m·K) instead of W/(m·K).
  - **Handling**: Define if needed, or users can use standard SI prefixes.

- **Edge case**: Some materials property databases use BTU/(hr·ft·°F) for thermal conductivity.
  - **Handling**: Can be added later if needed; focus on SI units first.

---

## Definition of Done

- [ ] materials.py module created with all specified units
- [ ] Module properly imports and re-exports stress units from engineering
- [ ] Fractional dimension (m^1/2) works correctly for fracture toughness
- [ ] All units have correct dimensions and scale factors
- [ ] Comprehensive tests pass (dimension, scale, conversion)
- [ ] Module importable: `from dimtensor.domains.materials import MPa, strain, vickers, MPa_sqrt_m`
- [ ] domains/__init__.py updated to expose materials
- [ ] CONTINUITY.md updated with completion status

---

## Notes / Log

### Key Design Decisions

1. **Hardness as dimensionless**: While hardness technically has units of pressure (force/area), in practice the various hardness scales (Vickers, Rockwell, Brinell) are reported as dimensionless numbers. This matches real-world usage.

2. **Fracture toughness dimension**: K_IC has dimension of stress × length^(1/2). This tests the Dimension class's ability to handle Fraction(1,2) as an exponent.

3. **Conductivity vs Resistivity**: Both are provided as separate units rather than inverses because they have different dimensions and different common usage contexts.

4. **Stress unit reuse**: Re-exporting from engineering.py maintains consistency and avoids "which MPa should I use?" confusion.

---

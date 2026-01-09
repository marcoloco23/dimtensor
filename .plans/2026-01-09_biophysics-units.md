# Plan: Biophysics Units Module

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Add biophysics-specific units (enzyme activity, membrane potentials, cell concentrations) to enable computational biophysics and cell biology applications with dimtensor.

---

## Background

Biophysics and computational biology require specialized units that bridge chemistry, physics, and cell biology. These include enzyme kinetics units (katal, enzyme unit), electrophysiology units (millivolt for membrane potentials), and cell biology units (cells per volume). While some concentration units exist in the chemistry module, biophysics applications need additional units specific to biological systems.

---

## Approach

### Decision: Create domains/biophysics.py

Following the same pattern as astronomy.py, chemistry.py, and engineering.py in the domains/ folder. We'll leverage existing units from chemistry (dalton, molar concentrations) by referencing them and add biophysics-specific units.

**Key Design Decisions:**
- **Enzyme activity**: Include both SI (katal) and conventional (enzyme unit U) for compatibility with biochemistry literature
- **Cell concentrations**: Treat cells as dimensionless counts, so cell/volume has dimension L^-3
- **Membrane potential**: millivolt is standard in electrophysiology, include for convenience
- **Cross-references**: Import/reference dalton and molar from chemistry module to maintain consistency

---

## Implementation Steps

1. [ ] Create `src/dimtensor/domains/biophysics.py` with units:
   - **Mass** (reference from chemistry):
     - dalton (Da, u) - atomic mass unit (reference chemistry.dalton)
   - **Enzyme activity**:
     - katal (kat) - SI unit: 1 mol/s, scale=1.0, Dimension(amount=1, time=-1)
     - enzyme_unit (U) - conventional: 1 μmol/min = 1.66667e-8 mol/s, Dimension(amount=1, time=-1)
   - **Concentration** (reference from chemistry):
     - molar (M) - mol/L (reference chemistry.molar)
     - millimolar (mM) - mmol/L (reference chemistry.millimolar)
   - **Cell biology**:
     - cells_per_mL - cell concentration: 1e6 cells/m³, Dimension(length=-3)
     - cells_per_uL - cell concentration: 1e9 cells/m³, Dimension(length=-3)
   - **Electrophysiology**:
     - millivolt (mV) - membrane potential: 0.001 V, Dimension(mass=1, length=2, time=-3, current=-1)
2. [ ] Update `src/dimtensor/domains/__init__.py` to expose biophysics module
3. [ ] Add tests in `tests/test_domains_biophysics.py`

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/domains/biophysics.py | CREATE - biophysics units module |
| src/dimtensor/domains/__init__.py | UPDATE - add biophysics import and to __all__ |
| tests/test_domains_biophysics.py | CREATE - comprehensive unit tests |

---

## Testing Strategy

- [ ] Test enzyme activity units:
  - Verify katal has correct dimension (amount=1, time=-1)
  - Verify enzyme_unit converts correctly to katal
  - Test that 1 U = 60 μmol/min
- [ ] Test cell concentration units:
  - Verify dimension is length^-3
  - Test conversions between cells/mL and cells/μL
  - Verify scale factors (1e6 and 1e9 per m³)
- [ ] Test electrophysiology units:
  - Verify millivolt has correct voltage dimension
  - Test conversion to base SI volt
- [ ] Test imports and cross-references:
  - Verify dalton reference from chemistry works
  - Verify molar/millimolar references work
  - Test that all units are importable

---

## Risks / Edge Cases

- **Risk**: Cell concentration treats cells as dimensionless counts, which differs from molar concentration (amount in moles).
  - **Mitigation**: Document clearly that cells are counted objects, not chemical amounts. Dimension is 1/length³, not amount/length³.

- **Risk**: Enzyme unit (U) definition varies slightly in literature (some use μmol/min, others use specific activity).
  - **Mitigation**: Use IUBMB standard definition: 1 U = 1 μmol substrate converted per minute. Document the reference.

- **Edge case**: Membrane potential is actually a voltage difference, not absolute potential.
  - **Handling**: Document as "membrane potential difference" in docstring for clarity.

- **Edge case**: References to chemistry units may create circular import issues.
  - **Mitigation**: Import at module level, not at package level. Test imports carefully.

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Tests pass (pytest)
- [ ] Module importable: `from dimtensor.domains.biophysics import katal, enzyme_unit, millivolt`
- [ ] Documentation includes usage examples
- [ ] Reference sources documented (IUBMB, CODATA 2022)
- [ ] CONTINUITY.md updated with completion status

---

## Notes / Log

**Scientific References:**
- Enzyme activity: IUBMB recommendations (1 U = 1 μmol substrate/min)
- Katal: SI brochure (1 kat = 1 mol/s)
- Dalton: CODATA 2022 (via chemistry module)
- Voltage: SI base units (V = kg⋅m²/(s³⋅A))

**Scale Factor Calculations:**
- 1 U = 1 μmol/min = 1e-6 mol / 60 s = 1.66666667e-8 mol/s
- 1 mV = 0.001 V = 0.001 kg⋅m²/(s³⋅A)
- 1 cell/mL = 1 / (1e-6 m³) = 1e6 / m³
- 1 cell/μL = 1 / (1e-9 m³) = 1e9 / m³

---

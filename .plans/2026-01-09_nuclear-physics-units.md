# Plan: Nuclear Physics Units Module

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Add nuclear physics units (MeV, barn, becquerel, gray, sievert) to enable dimensional analysis for nuclear physics, particle physics, and radiation safety applications.

---

## Background

Nuclear physics uses specialized units that are distinct from general physics or chemistry. These include energy units in the electron volt family (MeV, GeV), cross-sections in barns, radioactivity in becquerels/curies, and radiation dose in grays and sieverts. Adding these units will make dimtensor useful for nuclear reactor physics, particle physics experiments, radiation protection, and medical physics.

The electron volt (eV) is already defined in core units, but the higher-energy variants (keV, MeV, GeV) commonly used in nuclear physics should be in this domain-specific module.

---

## Approach

### Decision: Create domains/nuclear.py

Following the same pattern as astronomy.py, chemistry.py, and engineering.py in the domains/ folder. Group units by category (energy, cross-section, radioactivity, dose, decay).

**Key design decisions:**
- Include keV, MeV, GeV even though eV exists in core (these are standard in nuclear physics)
- Gray and sievert have the same dimension (length²/time²) but represent different physical quantities (absorbed dose vs dose equivalent). Both should be included as they're used differently in practice.
- Decay constant is just 1/time, same dimension as becquerel and hertz, but provide for convenience

---

## Implementation Steps

1. [ ] Create `src/dimtensor/domains/nuclear.py` with units organized by category:
   - **Energy units (electron volt family):**
     - electronvolt (eV) - 1.602176634e-19 J (CODATA 2022, exact)
     - kiloelectronvolt (keV) - 1e3 eV
     - megaelectronvolt (MeV) - 1e6 eV
     - gigaelectronvolt (GeV) - 1e9 eV
   - **Cross-section units:**
     - barn (b) - 1e-28 m² (exactly, by definition)
     - millibarn (mb) - 1e-31 m²
     - microbarn (µb) - 1e-34 m²
   - **Radioactivity units:**
     - becquerel (Bq) - 1/s (SI unit, time^-1)
     - curie (Ci) - 3.7e10 Bq (exactly, historical definition)
   - **Absorbed dose units:**
     - gray (Gy) - J/kg = m²/s² (SI unit)
     - rad - 0.01 Gy (exactly)
   - **Dose equivalent units:**
     - sievert (Sv) - J/kg = m²/s² (SI unit)
     - rem - 0.01 Sv (exactly)
   - **Decay constant (convenience):**
     - per_second - 1/s (same as Bq, for decay constants)

2. [ ] Update `src/dimtensor/domains/__init__.py` to import and expose nuclear module

3. [ ] Add comprehensive tests in `tests/test_domains.py` (add TestNuclearUnits class):
   - Test dimensions for each unit type
   - Test scale factors and conversions
   - Test cross-domain conversions (MeV to joules, barn to m²)
   - Test that Gy and Sv have same dimension but different symbols

4. [ ] Update documentation strings with proper references (CODATA 2022, SI definitions)

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/domains/nuclear.py | CREATE - nuclear physics units module |
| src/dimtensor/domains/__init__.py | UPDATE - add nuclear to imports and __all__ |
| tests/test_domains.py | UPDATE - add TestNuclearUnits class with comprehensive tests |

---

## Testing Strategy

### Unit dimension tests:
- [ ] Test energy units (keV, MeV, GeV) have energy dimension (M·L²·T⁻²)
- [ ] Test cross-section units (barn, mb, µb) have area dimension (L²)
- [ ] Test radioactivity units (Bq, Ci) have frequency dimension (T⁻¹)
- [ ] Test absorbed dose units (Gy, rad) have specific energy dimension (L²·T⁻²)
- [ ] Test dose equivalent units (Sv, rem) have same dimension as Gy
- [ ] Test decay constant has frequency dimension (T⁻¹)

### Scale factor tests:
- [ ] Test MeV = 1e6 eV (exact)
- [ ] Test GeV = 1e9 eV (exact)
- [ ] Test barn = 1e-28 m² (exact)
- [ ] Test curie = 3.7e10 Bq (exact)
- [ ] Test rad = 0.01 Gy (exact)
- [ ] Test rem = 0.01 Sv (exact)

### Conversion tests:
- [ ] Test converting MeV to joules (use core eV)
- [ ] Test converting barn to m²
- [ ] Test converting curie to becquerel
- [ ] Test converting rad to gray
- [ ] Test rem to sievert

### Cross-domain tests:
- [ ] Test nuclear energy units work with core joule
- [ ] Test that energy from core units can convert to MeV
- [ ] Test cross-section calculations with barn

### Import tests:
- [ ] Test importing nuclear module: `from dimtensor.domains import nuclear`
- [ ] Test direct imports: `from dimtensor.domains.nuclear import MeV, barn`

---

## Risks / Edge Cases

**Risk 1:** Gray and sievert have identical dimensions but represent different physical quantities.
- **Mitigation:** Keep both units with different symbols. Document that they're dimensionally equivalent but represent absorbed dose vs dose equivalent (which includes quality factor).

**Risk 2:** Decay constant has same dimension as becquerel (both are 1/time).
- **Mitigation:** Provide `per_second` as a convenience alias, document that it's dimensionally equivalent to Bq but semantically different (rate constant vs activity).

**Edge case:** Electron volt (eV) already exists in core units.
- **Handling:** Re-define eV in nuclear module for completeness, or import from core. Decision: re-define for consistency with module pattern (users can import from one place).

**Edge case:** Microbarn symbol should be µb but may have encoding issues.
- **Handling:** Use "ub" in symbol string for compatibility, provide µb as alias if needed.

**Edge case:** Users might confuse rem (dose) with REM (roentgen equivalent man).
- **Mitigation:** Use lowercase "rem" consistently, document clearly in docstring.

---

## Definition of Done

- [ ] src/dimtensor/domains/nuclear.py created with all units
- [ ] All units have correct dimensions and scale factors
- [ ] Module properly structured with sections and docstrings
- [ ] tests/test_domains.py updated with TestNuclearUnits class
- [ ] All tests pass
- [ ] Module importable: `from dimtensor.domains.nuclear import MeV, barn, becquerel`
- [ ] __all__ export list complete
- [ ] CODATA 2022 and SI references documented
- [ ] CONTINUITY.md updated with task completion

---

## Notes / Log

**Physical Constants Reference:**
- eV = 1.602176634e-19 J (CODATA 2022, exact since 2019 SI redefinition)
- barn = 1e-28 m² (exactly, by definition)
- curie = 3.7e10 Bq (exactly, historical definition based on radium-226)
- rad = 0.01 Gy (exactly, from CGS system)
- rem = 0.01 Sv (exactly, from CGS system)

**Dimensional Analysis:**
- Energy: [M·L²·T⁻²]
- Cross-section: [L²] (area)
- Radioactivity: [T⁻¹] (frequency/activity)
- Absorbed dose: [L²·T⁻²] (energy per unit mass)
- Dose equivalent: [L²·T⁻²] (same as absorbed dose, but includes quality factor in usage)

---

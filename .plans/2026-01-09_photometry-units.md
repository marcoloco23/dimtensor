# Plan: Photometry Units Module

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner-agent

---

## Goal

Create a photometry domain module providing units for light measurement including luminous intensity, flux, illuminance, luminance, and luminous efficacy.

---

## Background

Photometry is the science of measuring visible light in terms of human perception. It's essential for:
- Lighting design and engineering
- Display technology (monitors, TVs)
- Photography and cinematography
- Astronomy (stellar photometry)
- Vision science

The SI base unit candela (cd) is already defined in core/units.py. This module will provide derived photometric units commonly used in lighting, displays, and optical sciences.

---

## Approach

### Option A: Include candela in photometry module
- Description: Re-export candela from core.units and define all photometric units
- Pros: Complete photometry module, one-stop import
- Cons: Duplication (candela already in core)

### Option B: Only derived photometric units
- Description: Define only lumen, lux, nit, etc. Users import cd from core.units
- Pros: No duplication, clear separation
- Cons: Users need two imports for complete photometry work

### Decision: Option A (Re-export candela)

We'll re-export candela for convenience, similar to how engineering module includes newton_meter even though it's derivable. This provides a complete photometry namespace for users working with light measurements.

---

## Implementation Steps

1. [ ] Create `/home/user/dimtensor/src/dimtensor/domains/photometry.py`
2. [ ] Add module docstring with examples and references
3. [ ] Import Dimension, DIMENSIONLESS, Unit from core
4. [ ] Re-export candela from core.units for convenience
5. [ ] Define luminous flux units (lumen)
6. [ ] Define illuminance units (lux, foot-candle)
7. [ ] Define luminance units (nit, lambert, stilb)
8. [ ] Define luminous energy units (lumen-second, talbot)
9. [ ] Define luminous efficacy units (lm/W)
10. [ ] Create comprehensive __all__ export list
11. [ ] Add photometry tests to test_domains.py
12. [ ] Update domains/__init__.py to include photometry
13. [ ] Verify all tests pass

---

## Files to Modify

| File | Change |
|------|--------|
| `/home/user/dimtensor/src/dimtensor/domains/photometry.py` | Create new module with photometric units |
| `/home/user/dimtensor/src/dimtensor/domains/__init__.py` | Add `from . import photometry` |
| `/home/user/dimtensor/tests/test_domains.py` | Add TestPhotometryUnits class with unit tests |

---

## Unit Definitions

### Luminous Intensity (base)
- **candela (cd)**: Re-export from core.units - Dimension(luminosity=1), scale=1.0

### Luminous Flux
- **lumen (lm)**: cd·sr (steradian is dimensionless) - Dimension(luminosity=1), scale=1.0
- **Symbol**: "lm"

### Illuminance (lux = lm/m²)
- **lux (lx)**: lm/m² - Dimension(luminosity=1, length=-2), scale=1.0
- **foot-candle (fc)**: lm/ft² - Dimension(luminosity=1, length=-2), scale=10.76391... (1/0.3048²)
- **phot (ph)**: lm/cm² - Dimension(luminosity=1, length=-2), scale=10000.0

### Luminance (cd/m²)
- **nit**: cd/m² - Dimension(luminosity=1, length=-2), scale=1.0
- **candela per square meter (cd_per_m2)**: alias for nit
- **stilb (sb)**: cd/cm² - Dimension(luminosity=1, length=-2), scale=10000.0
- **lambert (La)**: cd/(cm²·π) - Dimension(luminosity=1, length=-2), scale=3183.0988... (10000/π)

### Luminous Energy
- **lumen-second (lm·s)**: Dimension(luminosity=1, time=1), scale=1.0
- **talbot**: Alternative name for lumen-second

### Luminous Efficacy (lm/W)
- **lumen per watt (lm_per_W)**: Dimension(luminosity=1, mass=-1, length=-2, time=3), scale=1.0

---

## Testing Strategy

- [ ] Test dimension correctness for each unit
- [ ] Test scale factors (lux=1.0, foot_candle≈10.76, etc.)
- [ ] Test unit conversions (lux to foot-candle, nit to lambert)
- [ ] Test candela re-export works correctly
- [ ] Test illuminance vs luminance (both have same dimension but different meanings)
- [ ] Test luminous efficacy dimension (lm/W)
- [ ] Test imports from dimtensor.domains.photometry
- [ ] Test practical calculations (light bulb lumens, display brightness in nits)
- [ ] Verify all units in __all__ are defined

---

## Risks / Edge Cases

- **Risk 1**: Confusion between illuminance (lux) and luminance (nit) - both have dimension J¹L⁻²
  - Mitigation: Clear docstring explaining the difference (incident vs emitted/reflected light)

- **Risk 2**: Steradian handling in lumen definition
  - Mitigation: Note in docstring that steradian is dimensionless, so lm = cd in dimensional analysis

- **Edge case**: Lambert uses π in its definition (cd/cm²/π)
  - Handling: Use math.pi for accurate conversion factor

- **Edge case**: Users may import candela from both core.units and domains.photometry
  - Handling: Both refer to same object, no issue. Document this is intentional.

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Tests pass (pytest tests/test_domains.py::TestPhotometryUnits)
- [ ] Module properly integrated into domains package
- [ ] CONTINUITY.md updated

---

## Notes / Log

**Key References**:
- SI Brochure 9th Edition (2019) - defines candela
- NIST SP 811 - Guide for the Use of SI Units
- CIE (International Commission on Illumination) standards

**Dimensional Analysis Notes**:
- Luminous intensity (cd): J¹
- Luminous flux (lm): J¹ (since sr is dimensionless)
- Illuminance (lx, fc): J¹L⁻² (flux per area)
- Luminance (nit, La): J¹L⁻² (intensity per area)
- Luminous energy (lm·s): J¹T¹
- Luminous efficacy (lm/W): J¹M⁻¹L⁻²T³

**Note**: Illuminance and luminance have the same dimension but represent different physical concepts:
- Illuminance: light falling ON a surface (incident)
- Luminance: light emitted/reflected FROM a surface

---

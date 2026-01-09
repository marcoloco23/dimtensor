# Plan: Acoustics Units Module

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: agent (planner)

---

## Goal

Add acoustics-specific units (decibel, rayl, phon, sone, etc.) to enable scientific computing for acoustics, audio engineering, and psychoacoustics applications.

---

## Background

Acoustics and audio engineering use specialized units that differ from standard SI. The field includes both physical quantities (acoustic impedance, sound pressure) and psychoacoustic measures (loudness in phon/sone). Logarithmic scales (decibel) are ubiquitous. Adding these units will make dimtensor useful for audio processing, room acoustics, noise analysis, and psychoacoustic research.

---

## Approach

### Option A: Follow existing domains/ pattern
- Create `src/dimtensor/domains/acoustics.py` alongside astronomy, chemistry, engineering
- Pros: Consistent with existing architecture, discoverable
- Cons: None significant

### Option B: Add to engineering.py
- Pros: Could argue acoustics is engineering subdomain
- Cons: Mixing different domains, less organized, engineering.py already comprehensive

### Decision: Option A - Create dedicated acoustics.py module
This maintains consistency with the domains/ folder structure and keeps acoustics units cleanly separated. Users can `from dimtensor.domains.acoustics import rayl, dB`.

**Special consideration**: Decibel is logarithmic and requires conversion formulas (not simple linear scaling). We'll document this limitation in the docstring and provide dimensionless unit for reference ratios.

---

## Implementation Steps

1. [ ] Create `src/dimtensor/domains/acoustics.py` with comprehensive docstring
2. [ ] Implement frequency units (as aliases/convenience):
   - hertz (Hz) - alias to standard frequency
   - kilohertz (kHz) - 1000 Hz
3. [ ] Implement pressure units:
   - micropascal (uPa) - 1e-6 Pa (reference for sound pressure level)
   - pascal (Pa) - alias for convenience
4. [ ] Implement acoustic impedance:
   - rayl - Pa·s/m = kg/(m²·s) = Dimension(mass=1, length=-2, time=-1)
5. [ ] Implement dimensionless logarithmic/psychoacoustic units:
   - decibel (dB) - dimensionless, with documentation about logarithmic nature
   - phon - dimensionless loudness level (equal-loudness contour)
   - sone - dimensionless loudness (linear perceptual scale)
6. [ ] Implement power/energy units (as aliases):
   - acoustic_watt (W) - sound power
7. [ ] Add comprehensive docstring with:
   - Example usage for typical acoustics calculations
   - Warning about decibel logarithmic conversion
   - Reference to standards (ISO 1683, ANSI, etc.)
8. [ ] Create `__all__` export list
9. [ ] Add tests in `tests/test_domains_acoustics.py`

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/domains/acoustics.py | CREATE - acoustics units module |
| tests/test_domains_acoustics.py | CREATE - unit tests for acoustics |
| src/dimtensor/domains/__init__.py | UPDATE - export acoustics module |

---

## Testing Strategy

- [ ] Test that rayl has correct dimension (M L^-2 T^-1)
- [ ] Test that rayl has correct scale factor (1.0 for Pa·s/m)
- [ ] Test micropascal scale factor (1e-6)
- [ ] Test frequency unit conversions (Hz to kHz)
- [ ] Test dimensionless units (dB, phon, sone) are DIMENSIONLESS
- [ ] Test unit arithmetic:
  - Pressure / (velocity) = acoustic impedance
  - Sound pressure conversions
- [ ] Test that decibel warnings/documentation are clear
- [ ] Test imports: `from dimtensor.domains.acoustics import rayl, uPa`

---

## Risks / Edge Cases

- **Risk**: Decibel is logarithmic, not linear - users might expect automatic conversion.
  - **Mitigation**: Clear documentation in module docstring and dB unit docstring explaining that dB requires logarithmic conversion (10*log10(ratio) or 20*log10(ratio)). Provide dimensionless unit for ratios only.

- **Edge case**: Acoustic impedance dimension confusion (Pa·s/m vs Pa/v where v is velocity).
  - **Handling**: Document clearly that rayl = kg/(m²·s) and provide example showing pressure/velocity calculation.

- **Edge case**: Frequency units (Hz, kHz) might already exist in base units.
  - **Handling**: Check core units.py first. Create aliases if they exist, or define if missing.

- **Risk**: Phon and sone are psychoacoustic and context-dependent (require equal-loudness curves).
  - **Mitigation**: Document that these are dimensionless reference scales and require external conversion tables/formulas for meaningful use.

- **Edge case**: Reference pressure (20 micropascals) for SPL calculations.
  - **Handling**: Provide micropascal unit and document standard reference in docstring.

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Tests pass (pytest tests/test_domains_acoustics.py)
- [ ] Module importable: `from dimtensor.domains.acoustics import rayl, dB, phon, sone`
- [ ] All units have correct dimensions and scale factors
- [ ] Documentation clearly explains logarithmic unit limitations
- [ ] CONTINUITY.md updated with completion status

---

## Notes / Log

**[Planning Phase]** - Initial research complete. Key decision: decibel will be dimensionless with documentation warnings about logarithmic nature. Rayl dimension confirmed as M L^-2 T^-1 (kg/(m²·s)).

---

## Technical Reference

### Unit Definitions

**Rayl** (acoustic impedance):
- Symbol: rayl (sometimes Pa·s/m)
- Dimension: M L^-2 T^-1
- Definition: 1 rayl = 1 Pa·s/m = 1 kg/(m²·s)
- Scale to SI: 1.0 (already in SI)

**Micropascal** (sound pressure):
- Symbol: μPa or uPa
- Dimension: M L^-1 T^-2 (pressure)
- Definition: 1 μPa = 10^-6 Pa
- Reference: 20 μPa is standard reference for SPL in air

**Decibel** (dimensionless ratio):
- Symbol: dB
- Dimension: DIMENSIONLESS
- Note: Logarithmic scale, requires conversion formula
- SPL: 20·log₁₀(p/p₀) where p₀ = 20 μPa
- Power: 10·log₁₀(P/P₀)

**Phon** (loudness level):
- Symbol: phon
- Dimension: DIMENSIONLESS
- Note: Equal-loudness contour level, requires ISO 226 curves

**Sone** (loudness):
- Symbol: sone
- Dimension: DIMENSIONLESS
- Note: Linear perceptual loudness scale
- Conversion: L_sone = 2^((L_phon - 40)/10)

**Frequency**:
- Hz: time^-1, scale = 1.0
- kHz: time^-1, scale = 1000.0

---

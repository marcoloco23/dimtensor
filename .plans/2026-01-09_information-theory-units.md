# Plan: Information Theory Units Module

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create a domain-specific unit module for information theory, providing units for information content (bit, byte, nat), entropy (bit/symbol), data rates (bit/s), and storage densities (bit/m²).

---

## Background

Information theory units are dimensionless in SI (pure numbers) but tracking them explicitly provides important semantic clarity in:
- Data storage and transmission applications
- Entropy and information content calculations
- Shannon's information theory
- Network bandwidth and data rate specifications
- Storage density calculations for memory devices

Similar to angular units in astronomy (arcsecond, radian), these units are mathematically dimensionless but domain-meaningful.

---

## Approach

### Option A: All units as DIMENSIONLESS with scale factors

- Description: Treat all information units as dimensionless with scale factors relative to a base unit (bit = 1.0)
- Pros:
  - Consistent with physics (information is dimensionless)
  - Allows conversion between bit, byte, nat, shannon naturally
  - Simple implementation matching angular units pattern
  - Data rate and entropy units naturally combine with time/symbol dimensions
- Cons:
  - Cannot prevent mixing bit and nat without explicit conversions
  - Less type safety than custom dimension

### Option B: Custom information dimension (8th dimension)

- Description: Add an 8th dimension to Dimension tuple for information
- Pros:
  - Stronger type safety (bit vs nat are different dimensions)
  - Prevents accidental mixing
- Cons:
  - Major architectural change (Dimension is 7-tuple for SI)
  - Breaks SI foundation of the library
  - Inconsistent with treatment of other dimensionless quantities

### Decision: Option A - DIMENSIONLESS with scale factors

Rationale:
1. Maintains SI consistency (information is mathematically dimensionless)
2. Matches existing pattern for angular units in astronomy module
3. Allows natural composition (bit/s = bit * s^-1, bit/m² = bit * m^-2)
4. Users can track semantic meaning through unit symbols
5. No architectural changes needed

---

## Implementation Steps

1. [ ] Create `/home/user/dimtensor/src/dimtensor/domains/information.py` module file
2. [ ] Add module docstring with information theory context and examples
3. [ ] Import necessary components (Dimension, DIMENSIONLESS, Unit)
4. [ ] Define information content units section:
   - bit (base unit, scale=1.0)
   - byte/octet (scale=8.0)
   - kilobyte, megabyte, gigabyte, terabyte (using 1024-based binary prefixes: kibibyte = 8192 bits)
   - nat (natural unit, scale=ln(2) ≈ 0.693147)
   - shannon (alias for bit)
5. [ ] Define entropy units section:
   - bit_per_symbol (dimensionless, scale=1.0)
   - nat_per_symbol (dimensionless, scale=ln(2))
6. [ ] Define data rate units section:
   - bit_per_second (Dimension(time=-1), scale=1.0)
   - kilobit_per_second, megabit_per_second, gigabit_per_second
   - byte_per_second (Dimension(time=-1), scale=8.0)
   - baud (symbol/s, Dimension(time=-1), scale=1.0)
7. [ ] Define storage density units section:
   - bit_per_square_meter (Dimension(length=-2), scale=1.0)
   - bit_per_cubic_meter (Dimension(length=-3), scale=1.0)
8. [ ] Add comprehensive __all__ exports list
9. [ ] Update `/home/user/dimtensor/src/dimtensor/domains/__init__.py` to include information module
10. [ ] Create test class `TestInformationUnits` in `/home/user/dimtensor/tests/test_domains.py`
11. [ ] Write unit tests for dimensions, scales, and conversions
12. [ ] Write practical calculation tests (e.g., Shannon entropy, channel capacity)
13. [ ] Run full test suite to ensure no regressions

---

## Files to Modify

| File | Change |
|------|--------|
| `/home/user/dimtensor/src/dimtensor/domains/information.py` | Create new module with information theory units |
| `/home/user/dimtensor/src/dimtensor/domains/__init__.py` | Add `from . import information` and update __all__ |
| `/home/user/dimtensor/tests/test_domains.py` | Add TestInformationUnits class with comprehensive tests |

---

## Testing Strategy

### Unit Tests
- [ ] Test dimension correctness:
  - Information units (bit, byte, nat) are DIMENSIONLESS
  - Data rate units have time^-1 dimension
  - Storage density units have length^-2 or length^-3 dimensions
- [ ] Test scale factors:
  - byte = 8 bits
  - kilobyte = 8192 bits (1024 bytes)
  - nat = ln(2) bits ≈ 0.693147
  - shannon = 1 bit
- [ ] Test conversions:
  - bit ↔ byte ↔ kilobyte ↔ megabyte
  - bit ↔ nat ↔ shannon
  - bit/s ↔ byte/s ↔ Mbit/s
- [ ] Test practical calculations:
  - Shannon entropy: H = -Σ p_i log₂(p_i) bits
  - Channel capacity: C = B log₂(1 + SNR) bit/s
  - Storage density conversions

### Edge Cases
- [ ] Mixing bit and nat (should work but require explicit conversion)
- [ ] Data rate arithmetic (10 Mbit/s * 60 s = 600 Mbit)
- [ ] Volume density to area density (requires volume dimensions)

---

## Risks / Edge Cases

**Risk 1: Binary vs decimal prefixes confusion**
- Issue: kilobyte can mean 1000 bytes (SI) or 1024 bytes (IEC binary)
- Mitigation: Use IEC binary prefixes (kibibyte=1024, mebibyte=1024²) and document clearly
- Note: Follow NIST/IEC standards for binary data (KiB, MiB, GiB, TiB)

**Risk 2: Nat vs bit conversion precision**
- Issue: ln(2) = 0.69314718055994530942... is irrational
- Mitigation: Use high-precision constant (math.log(2)) for scale factor
- Testing: Verify round-trip conversions are accurate

**Risk 3: Dimensionless units mixing**
- Issue: Users might accidentally mix bit and nat without realizing
- Mitigation: Clear documentation that conversion is needed; unit symbols make intent clear
- Alternative: Consider whether nat should be a separate "natural information" category

**Risk 4: Data rate vs frequency confusion**
- Issue: bit/s and Hz both have time^-1 dimension
- Mitigation: Symbol tracking makes semantic difference clear
- Note: baud (symbols/s) is distinct from bit/s when encoding multiple bits per symbol

**Edge Case 1: Entropy per symbol**
- Handling: Treat as dimensionless (information content per discrete event)
- Example: 3.5 bit/symbol means average of 3.5 bits of information per symbol

**Edge Case 2: Compound units**
- Handling: Bit/s, bit/m² naturally compose with time and length dimensions
- Example: DimArray([100], bit_per_second) * DimArray([60], second) = 6000 bits

---

## Definition of Done

- [x] Plan document created with detailed approach
- [ ] Implementation steps are clear and actionable
- [ ] All files to modify are identified with absolute paths
- [ ] Testing strategy covers dimensions, scales, and practical use cases
- [ ] Risks and edge cases are documented with mitigations
- [ ] Plan reviewed for consistency with existing domain module patterns

---

## Notes / Log

**2026-01-09** - Plan created by planner agent
- Researched existing domain modules (astronomy, chemistry, engineering)
- Identified DIMENSIONLESS pattern from angular units in astronomy
- Decided on IEC binary prefixes for byte multiples (KiB, MiB, GiB, TiB)
- Key decision: nat scale = ln(2) ≈ 0.693147 for natural logarithm base
- Noted that baud and bit/s are distinct (symbols vs bits when multi-bit encoding)

**Design Rationale:**
- Information content units: dimensionless with scale factors
- Natural logarithm base: 1 nat = ln(2) bits ≈ 0.693147 bits
- Shannon is alias for bit (common in literature)
- Binary prefixes follow IEC 60027-2 standard (power of 1024)
- Data rates combine information with time^-1 dimension
- Storage densities combine information with length^-2 or length^-3 dimensions

**References:**
- Shannon, C.E. (1948). "A Mathematical Theory of Communication"
- IEC 60027-2: Letter symbols for quantities (binary prefixes)
- NIST Special Publication 811: Guide for the Use of SI Units

---

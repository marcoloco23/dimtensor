# Plan: Conservation Law Tracking

**Date**: 2026-01-09
**Status**: IN PROGRESS
**Author**: agent

---

## Goal

Provide a simple utility for tracking conserved quantities (energy, momentum, mass) across computations, alerting users when conservation is violated beyond a tolerance.

---

## Background

In physics simulations, conservation laws are fundamental:
- Energy conservation: Total energy remains constant
- Momentum conservation: Total momentum remains constant
- Mass conservation: Total mass remains constant

Users want to verify these hold throughout their computations. Currently, they must manually track and compare values.

---

## Approach

### Option A: Full Framework
- Automatic tracking integrated into DimArray operations
- Complex, intrusive, performance overhead
- Cons: Over-engineered for v1.4.0

### Option B: Lightweight Tracker (CHOSEN)
- Simple class that records values and checks conservation
- User explicitly records checkpoints
- Non-intrusive, no performance overhead
- Pros: Simple, useful, easy to implement

### Decision: Option B

A lightweight `ConservationTracker` that:
1. User creates tracker with a conserved quantity
2. User records checkpoints with `.record(value)`
3. User checks conservation with `.is_conserved(rtol=1e-9)`
4. Tracker compares current value to initial value

---

## Implementation

```python
from dimtensor import DimArray, units
from dimtensor.validation import ConservationTracker

# Track energy conservation
tracker = ConservationTracker("Total Energy")

# Initial state
KE = 0.5 * m * v**2
PE = m * g * h
total_E = KE + PE
tracker.record(total_E)

# After some computation...
KE2 = 0.5 * m * v2**2
PE2 = m * g * h2
total_E2 = KE2 + PE2
tracker.record(total_E2)

# Check conservation
if not tracker.is_conserved(rtol=1e-6):
    print(f"Energy not conserved! Drift: {tracker.drift()}")
```

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/validation/conservation.py | NEW: ConservationTracker class |
| src/dimtensor/validation/__init__.py | Add ConservationTracker export |
| tests/test_constraints.py | Add conservation tests |

---

## Testing Strategy

- [ ] Test recording values
- [ ] Test is_conserved with matching values
- [ ] Test is_conserved with drifting values
- [ ] Test drift calculation
- [ ] Test with different tolerances

---

## Definition of Done

- [ ] ConservationTracker implemented
- [ ] Tests pass
- [ ] CONTINUITY.md updated

---

## Notes

**2026-01-09** - Chose lightweight approach for v1.4.0. Full automatic tracking deferred.

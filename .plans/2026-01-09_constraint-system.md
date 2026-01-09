# Plan: Constraint System

**Date**: 2026-01-09
**Status**: IN PROGRESS
**Author**: agent

---

## Goal

Add value constraints to DimArray for enforcing physical validity (e.g., mass must be positive, probability must be 0-1, temperature in Kelvin must be >= 0).

---

## Background

Physical quantities often have validity constraints:
- Mass, energy, time intervals must be positive
- Probabilities must be [0, 1]
- Temperature in Kelvin must be >= 0
- Divisors must be non-zero

Currently, dimtensor only validates dimensions, not values. Adding constraint validation catches physics errors earlier (e.g., negative mass from a bug).

---

## Approach

### Option A: Instance-based constraints
- Attach constraints to individual DimArray instances
- Flexible: same unit can have different constraints in different contexts
- Pros: Flexibility, explicit
- Cons: More verbose, constraints don't propagate automatically

### Option B: Subclasses (PositiveArray, etc.)
- Create constrained subclasses
- Pros: Clean type hierarchy
- Cons: Combinatorial explosion, inflexible

### Option C: Decorator/wrapper pattern
- Wrap DimArray with constraint checking
- Pros: Non-intrusive
- Cons: Adds complexity

### Decision: Option A (Instance-based)

Most Pythonic and flexible. Constraints are explicit, can be mixed, and don't require subclassing. Constraints are checked:
1. On creation (if constraints provided)
2. On `.validate()` call
3. Optionally after operations (configurable)

---

## Implementation Steps

1. [x] Design constraint system (this plan)
2. [ ] Create validation/constraints.py:
   - Base `Constraint` class with `check(arr)` method
   - `PositiveConstraint` - values > 0
   - `NonNegativeConstraint` - values >= 0
   - `NonZeroConstraint` - values != 0
   - `BoundedConstraint(min, max)` - min <= values <= max
3. [ ] Create `ConstraintError` exception
4. [ ] Add optional `constraints` parameter to DimArray.__init__
5. [ ] Add `.validate()` method to DimArray
6. [ ] Add `.with_constraints()` method for adding constraints to existing array
7. [ ] Add tests
8. [ ] Update exports

---

## API Design

```python
from dimtensor import DimArray, units
from dimtensor.validation import Positive, NonNegative, Bounded, NonZero

# Create with constraints
mass = DimArray([1.0, 2.0], units.kg, constraints=[Positive()])
prob = DimArray([0.5, 0.3], units.dimensionless, constraints=[Bounded(0, 1)])

# Validate manually
mass.validate()  # Raises ConstraintError if violated

# Add constraints to existing array
arr = DimArray([1.0, 2.0], units.m)
constrained_arr = arr.with_constraints([NonNegative()])

# Operations preserve constraints (optional, configurable)
mass2 = mass * 2  # Still has Positive constraint if propagate_constraints=True

# Check if constraint is satisfied
if Positive().is_satisfied(mass):
    print("Valid!")
```

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/validation/__init__.py | NEW: Export constraints |
| src/dimtensor/validation/constraints.py | NEW: Constraint classes |
| src/dimtensor/errors.py | Add ConstraintError |
| src/dimtensor/core/dimarray.py | Add constraints param, validate(), with_constraints() |
| src/dimtensor/__init__.py | Export validation module |
| tests/test_constraints.py | NEW: Tests for constraints |

---

## Testing Strategy

- [ ] Test each constraint type (Positive, NonNegative, NonZero, Bounded)
- [ ] Test constraint violation raises ConstraintError
- [ ] Test `.validate()` method
- [ ] Test `.with_constraints()` method
- [ ] Test constraints with edge cases (NaN, inf, empty arrays)
- [ ] Test constraint propagation (optional feature)

---

## Risks / Edge Cases

- Edge case: NaN values - should always fail constraints (NaN is not positive, not negative, not zero)
- Edge case: Infinite values - Positive should accept inf, Bounded should reject
- Edge case: Empty arrays - should pass all constraints (vacuously true)
- Risk: Performance overhead - Mitigation: Validation is opt-in, not automatic on every op

---

## Definition of Done

- [ ] All constraint types implemented
- [ ] Tests pass
- [ ] DimArray supports constraints
- [ ] CONTINUITY.md updated

---

## Notes / Log

**2026-01-09** - Plan created. Chose instance-based approach for flexibility.

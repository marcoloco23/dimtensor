# Plan: Validation Guide Documentation

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create comprehensive user guide (docs/guide/validation.md) that teaches users how to validate physics constraints on DimArray values, track conservation laws, and create custom constraints for domain-specific validation.

---

## Background

The validation module (src/dimtensor/validation/) provides tools for enforcing physical constraints on array values:
- Value constraints (Positive, Bounded, etc.) catch physics errors like negative mass
- ConservationTracker monitors conserved quantities in simulations
- Custom constraints can be created for domain-specific validation

Currently there's working code and tests but no user-facing documentation. Users need a guide to learn how to use these features effectively in their physics simulations.

---

## Approach

### Option A: Single comprehensive guide covering all validation features
- Description: Create one guide with sections for constraints, conservation, and custom validation
- Pros:
  - All validation features in one place
  - Easier to maintain and update
  - Natural flow from basic constraints to advanced custom validation
  - Follows existing guide structure (units.md, operations.md, examples.md)
- Cons:
  - Could become lengthy
  - Mixing basic and advanced topics

### Option B: Split into multiple guides (basic constraints, conservation, advanced)
- Description: Create separate guides for different validation topics
- Pros:
  - Each guide stays focused
  - Easier to find specific topics
- Cons:
  - More files to maintain
  - Breaks up related content
  - Not consistent with existing docs structure

### Decision: Option A - Single comprehensive guide

Rationale: The validation features are closely related and benefit from being presented together. The guide can use clear sections to organize basic vs advanced topics. This matches the style of existing guides like examples.md which covers many topics in one file.

---

## Implementation Steps

1. [ ] Create docs/guide/validation.md with initial structure
2. [ ] Write Introduction section
   - What is validation and why it matters in physics
   - Overview of constraint types and conservation tracking
   - When to use validation
3. [ ] Write Built-in Constraints section
   - Positive (mass, temperature, time intervals)
   - NonNegative (counts, magnitudes)
   - NonZero (divisors)
   - Bounded (probabilities, efficiencies, angles)
   - Finite (measurements, computed values)
   - NotNaN (error detection)
   - Code examples for each
4. [ ] Write Using Constraints with DimArray section
   - DimArray.validate() method
   - Chaining constraints
   - Multiple constraints on one array
   - Error handling with ConstraintError
5. [ ] Write Conservation Tracking section
   - ConservationTracker basics
   - Recording values during simulation
   - Checking conservation with tolerances
   - Calculating drift
   - Physics examples (energy, momentum, mass)
6. [ ] Write Custom Constraints section
   - Extending the Constraint base class
   - Implementing check() method
   - Domain-specific examples (temperature ranges for materials, valid quantum numbers)
7. [ ] Write Physics Examples section
   - Temperature validation (absolute zero, material limits)
   - Mass constraints (positive, realistic bounds)
   - Probability validation (bounded [0,1])
   - Energy conservation in orbital mechanics
   - Momentum conservation in collisions
8. [ ] Write Best Practices section
   - When to validate (constructor vs operation)
   - Performance considerations
   - Choosing appropriate tolerances for conservation
   - Combining constraints for complex validation
9. [ ] Add cross-references to other guides
   - Link to operations.md for dimension checking
   - Link to examples.md for full physics simulations
10. [ ] Review and test all code examples
11. [ ] Add to main documentation index/table of contents

---

## Files to Modify

| File | Change |
|------|--------|
| docs/guide/validation.md | CREATE - New comprehensive validation guide |
| docs/README.md or docs/index.md | UPDATE - Add link to validation guide (if exists) |

---

## Testing Strategy

How will we verify this works?

- [ ] All code examples in the guide execute without errors
- [ ] Code examples demonstrate key features (each constraint type, conservation tracking, custom constraints)
- [ ] Examples use realistic physics scenarios
- [ ] Links to other documentation work correctly
- [ ] Markdown renders correctly (check headings, code blocks, formatting)
- [ ] Manual review: Guide is clear and teachable for new users

---

## Risks / Edge Cases

- Risk 1: Code examples may not run if validation module has changed since planning
  - Mitigation: Test all examples against current codebase before finalizing

- Risk 2: Examples may be too basic and not demonstrate real-world use cases
  - Mitigation: Include physics simulation examples from test suite, add realistic scenarios

- Risk 3: Custom constraint examples may be too complex for beginners
  - Mitigation: Start with simple custom constraint (e.g., temperature > absolute zero), then progress to complex ones

- Risk 4: Conservation tracking examples may not clearly show when/why to use it
  - Mitigation: Include full simulation example showing energy drift detection

- Edge case: Users may not understand relationship between validation and dimension checking
  - Handling: Add clear explanation that validation checks VALUES while dimensions check UNITS

---

## Definition of Done

- [ ] validation.md file created with all sections
- [ ] All code examples tested and working
- [ ] Guide covers all constraint types (Positive, NonNegative, NonZero, Bounded, Finite, NotNaN)
- [ ] ConservationTracker fully documented with examples
- [ ] Custom constraint creation explained with examples
- [ ] At least 3 complete physics examples (temperature, probability, energy conservation)
- [ ] Cross-references to other guides added
- [ ] Guide reviewed for clarity and completeness
- [ ] CONTINUITY.md updated with task completion

---

## Notes / Log

**2026-01-09** - Plan created by planner agent

### Key Content Decisions:
- Focus on practical physics examples rather than abstract demonstrations
- Include both inline validation (manual) and DimArray.validate() method usage
- Show conservation tracking in realistic simulation scenarios (free fall, collisions)
- Demonstrate custom constraints for domain-specific cases (material properties, quantum numbers)

### Example Structure for Each Constraint:
1. What it validates
2. When to use it (physics use cases)
3. Code example with DimArray
4. Error handling example

### Physics Examples to Include:
1. **Temperature Validation**: Absolute zero bounds, material melting points
2. **Mass Validation**: Positive mass, realistic astronomical bounds
3. **Probability**: Bounded [0,1] for quantum mechanics
4. **Energy Conservation**: Free fall, orbital mechanics
5. **Momentum Conservation**: Elastic/inelastic collisions
6. **Custom**: Valid quantum numbers (n >= 1, l < n, m in [-l, l])

---

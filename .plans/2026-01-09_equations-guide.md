# Plan: Equations Guide Documentation

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create comprehensive user documentation (docs/guide/equations.md) that teaches users how to discover, browse, search, validate against, and extend the physics equation database.

---

## Background

dimtensor includes a rich database of 50+ physics equations across multiple domains (mechanics, thermodynamics, electromagnetism, fluid dynamics, relativity, quantum, optics, acoustics). Each equation contains:
- Symbolic formula and LaTeX representation
- Variable names mapped to their dimensions
- Domain categorization and tags
- Descriptions, assumptions, and related equations

However, there's no user-facing guide explaining how to use this powerful feature. Users need to learn how to:
1. Browse and search the equation database
2. Use equations to validate their calculations
3. Get equation metadata (LaTeX, assumptions, related equations)
4. Register custom equations for their domain
5. Integrate equations with the inference system

---

## Approach

### Option A: Simple Reference Guide
- Description: Flat documentation listing all API functions with minimal examples
- Pros: Quick to write, comprehensive API coverage
- Cons: Not user-friendly, doesn't teach workflows, lacks motivation

### Option B: Task-Oriented Guide
- Description: Organize by user tasks (browsing, searching, validating, extending) with motivating examples
- Pros: More engaging, teaches practical workflows, shows real use cases
- Cons: Takes longer to write, may duplicate some API reference material

### Option C: Hybrid Approach
- Description: Start with motivating overview and common workflows, include comprehensive reference tables, end with advanced examples
- Pros: Best of both worlds - accessible for beginners, comprehensive for advanced users
- Cons: Longer document, requires careful organization

### Decision: Option C - Hybrid Approach

This provides the best user experience by:
1. Starting with motivation (why use the equation database?)
2. Showing common workflows with examples
3. Providing reference tables for quick lookup
4. Including advanced integration examples
5. Following the established doc style from units.md and inference.md

---

## Implementation Steps

1. [ ] Create docs/guide/equations.md with basic structure
2. [ ] Write "Overview" section explaining the equation database concept
3. [ ] Write "Browsing Equations" section (list domains, get by domain, get all)
4. [ ] Write "Searching Equations" section (search by name, tag, variable)
5. [ ] Write "Equation Metadata" section (formula, LaTeX, assumptions, related)
6. [ ] Write "Validating Calculations" section (check against known equations)
7. [ ] Write "Custom Equations" section (register_equation with examples)
8. [ ] Create reference tables:
   - Available domains
   - Example equations by domain
   - Equation metadata fields
9. [ ] Write advanced examples:
   - Integration with inference system
   - Building equation validators
   - Creating domain-specific equation sets
10. [ ] Add cross-references to API docs (api/validation.md, api/inference.md)
11. [ ] Test all code examples
12. [ ] Update docs/index.md to link to equations guide

---

## Files to Modify

| File | Change |
|------|--------|
| docs/guide/equations.md | CREATE - Main equations guide (new file) |
| docs/index.md | UPDATE - Add link to equations guide in table of contents |
| docs/getting-started.md | MAYBE UPDATE - Mention equation database in "What's Next?" section |

---

## Testing Strategy

How will we verify this works?

- [ ] Run all code examples in the guide to ensure they execute correctly
- [ ] Verify all imports work (dimtensor.equations.database)
- [ ] Check cross-references link to correct API docs
- [ ] Validate LaTeX rendering in example output
- [ ] Test with both NumPy backend (primary) and ensure examples are backend-agnostic
- [ ] Manual review for clarity, tone consistency, and completeness

---

## Risks / Edge Cases

- **Risk 1**: Code examples may not work if equation database API changes
  - **Mitigation**: Use Read tool to verify current API before writing examples; add tests

- **Risk 2**: Users may confuse equation database with inference system
  - **Mitigation**: Clearly distinguish the two in overview; show how they integrate

- **Risk 3**: Document may become outdated as new domains/equations are added
  - **Mitigation**: Use dynamic examples that list domains rather than hardcoding; note equation count is approximate

- **Edge Case**: User tries to register equation with same name as existing
  - **Handling**: Show warning in custom equations section about name conflicts

- **Edge Case**: Search returns no results
  - **Handling**: Include example showing empty result handling

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] All code examples tested and working
- [ ] Documentation follows established style (consistent with units.md, operations.md)
- [ ] Cross-references added to related docs
- [ ] Table of contents updated
- [ ] No broken links or imports
- [ ] CONTINUITY.md updated with completion

---

## Key Examples to Include

### 1. Browsing by Domain
```python
from dimtensor.equations.database import list_domains, get_equations

# List all available domains
domains = list_domains()
print(domains)  # ['acoustics', 'electromagnetism', 'fluid_dynamics', ...]

# Get all mechanics equations
mechanics_eqs = get_equations(domain="mechanics")
for eq in mechanics_eqs:
    print(f"{eq.name}: {eq.formula}")
```

### 2. Searching Equations
```python
from dimtensor.equations.database import search_equations, get_equations

# Search by name/description/variable
energy_eqs = search_equations("energy")
for eq in energy_eqs:
    print(f"{eq.name}: {eq.formula}")

# Filter by tags
fundamental = get_equations(tags=["fundamental"])
```

### 3. Getting Equation Metadata
```python
from dimtensor.equations.database import get_equation

eq = get_equation("Newton's Second Law")
print(f"Formula: {eq.formula}")
print(f"LaTeX: {eq.latex}")
print(f"Variables: {eq.variables}")
print(f"Assumptions: {eq.assumptions}")
print(f"Related: {eq.related}")
```

### 4. Validating Calculations
```python
from dimtensor import DimArray, units
from dimtensor.equations.database import get_equation

# Get equation
eq = get_equation("Newton's Second Law")

# Calculate force
mass = DimArray([2.0], units.kg)
accel = DimArray([10.0], units.m / units.s**2)
force = mass * accel

# Validate dimensions match F = ma
F_dim = eq.variables["F"]
assert force.dimension == F_dim
print(f"Dimensional validation passed: {force}")
```

### 5. Registering Custom Equations
```python
from dimtensor.equations.database import Equation, register_equation
from dimtensor.core.dimensions import Dimension

# Define custom equation for your domain
custom_eq = Equation(
    name="Rocket Equation",
    formula="delta_v = v_e * ln(m0/mf)",
    variables={
        "delta_v": Dimension(length=1, time=-1),
        "v_e": Dimension(length=1, time=-1),
        "m0": Dimension(mass=1),
        "mf": Dimension(mass=1),
    },
    domain="aerospace",
    tags=["rocket", "propulsion"],
    description="Tsiolkovsky rocket equation",
    latex=r"\Delta v = v_e \ln\frac{m_0}{m_f}",
)

register_equation(custom_eq)
```

### 6. Integration with Inference
```python
from dimtensor.inference import infer_units
from dimtensor.equations.database import get_equation
from dimtensor import units

# Use equation to guide inference
eq = get_equation("Ideal Gas Law")
print(f"Equation: {eq.formula}")
print(f"Variables needed: {list(eq.variables.keys())}")

# Infer missing variable
result = infer_units(
    "P * V = n * R * T",
    known_units={
        "P": units.Pa,
        "V": units.m**3,
        "n": units.mol,
        "T": units.K
    }
)
print(f"Inferred R: {result['inferred']['R']}")
```

---

## Document Structure

```
# Working with Equations

## Overview
- What is the equation database?
- Why use it?
- What domains are covered?

## Browsing Equations
- Listing all domains
- Getting equations by domain
- Getting all equations

## Searching Equations
- Searching by name/description
- Filtering by tags
- Finding equations with specific variables

## Equation Metadata
- Accessing formula and LaTeX
- Viewing variable dimensions
- Checking assumptions
- Finding related equations

## Validating Calculations
- Checking dimensional correctness against equations
- Using equations as templates
- Validation workflow examples

## Custom Equations
- Creating custom equations
- Registering equations
- Organizing domain-specific equation sets
- Best practices

## Reference Tables
- Available Domains
- Example Equations by Domain
- Equation Metadata Fields

## Advanced Examples
- Integration with inference system
- Building equation validators
- Creating interactive equation browsers
- Multi-equation validation

## See Also
- [Dimensional Inference](../api/inference.md)
- [Validation](../api/validation.md)
```

---

## Notes / Log

**2026-01-09** - Plan created after reviewing:
- .plans/_TEMPLATE.md
- src/dimtensor/equations/database.py (main database with 50+ equations)
- src/dimtensor/inference/equations.py (inference integration)
- docs/guide/units.md (style reference)
- docs/api/inference.md (related API docs)

Key insights:
- Database has rich metadata (LaTeX, assumptions, related equations)
- 8 domains covered: mechanics, thermodynamics, electromagnetism, fluid_dynamics, relativity, quantum, optics, acoustics
- Search/filter capabilities are powerful but undocumented
- Integration with inference system is a key use case
- Custom equations enable domain-specific extensions

---

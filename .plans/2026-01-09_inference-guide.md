# Plan: Inference Guide Documentation

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner-agent

---

## Goal

Create comprehensive user documentation for dimtensor's automatic unit inference system and linting CLI. Teach users how to leverage variable name patterns, equation patterns, and static analysis to catch dimensional bugs before runtime.

---

## Background

Dimtensor v3.3.0 introduced a sophisticated inference system that can:
1. Infer physical dimensions from variable names (e.g., "velocity" → L·T⁻¹)
2. Match equation patterns against a physics database (e.g., F = ma)
3. Solve constraint systems to infer unknown units from equations
4. Perform static analysis of Python code to detect dimensional mismatches

This feature is unique among unit libraries and needs comprehensive documentation. The target audience includes:
- Physics researchers who want to catch bugs early
- Students learning physics who want IDE-like guidance
- Library maintainers who want to enforce dimensional correctness in CI

Currently no user-facing documentation exists for this feature set.

---

## Approach

### Option A: Single comprehensive guide
- Description: One large inference.md covering all aspects
- Pros:
  - Complete reference in one place
  - Easy to search
  - Natural flow from basic to advanced
- Cons:
  - Could be overwhelming
  - Harder to maintain

### Option B: Split into multiple guides
- Description: Separate guides for heuristics, equations, and linting
- Pros:
  - Focused content
  - Users can find specific topics
- Cons:
  - Information fragmented
  - More files to maintain
  - Navigation complexity

### Decision: Option A - Single comprehensive guide

Reasoning:
- The inference system is cohesive - all components work together
- Users need to understand the full picture to use it effectively
- A single page with clear sections and TOC makes it easy to navigate
- Follows the pattern of existing docs (e.g., examples.md is comprehensive)
- We can add cross-references to API docs for details

---

## Implementation Steps

1. [ ] Create docs/guide/inference.md structure with sections:
   - Overview and motivation
   - Variable name patterns
   - Equation pattern database
   - Unit inference from equations
   - CLI linting tool
   - IDE integration
   - Best practices

2. [ ] Write "Overview" section:
   - What is inference?
   - Why is it useful?
   - When to use it vs explicit units
   - Quick example showing inference catching a bug

3. [ ] Write "Variable Name Patterns" section:
   - Explain heuristic system
   - Show confidence levels (0.9 = exact, 0.7 = prefix/suffix, 0.5 = partial)
   - Examples: velocity, initial_velocity_x, distance_m
   - Table of common patterns from VARIABLE_PATTERNS
   - Code examples using infer_dimension()

4. [ ] Write "Equation Pattern Database" section:
   - Explain the 50+ physics equation database
   - Show domains: mechanics, electromagnetics, thermodynamics, etc.
   - Examples: finding equations by domain/tag
   - Example: using suggest_dimension_from_equations()
   - Show how to explore EQUATION_DATABASE

5. [ ] Write "Unit Inference from Equations" section:
   - Explain constraint solver
   - Show infer_units() API
   - Examples:
     - F = ma with known m, a → infer F
     - E = 0.5 * m * v**2 with known E, m → infer v
     - Detecting dimensional inconsistencies
   - Show confidence scores
   - Discuss limitations (nonlinear equations, multiple solutions)

6. [ ] Write "CLI Linting" section:
   - Introduce dimtensor lint command
   - Show basic usage: dimtensor lint myfile.py
   - Explain severity levels: ERROR, WARNING, INFO
   - Show examples of caught bugs:
     - velocity + acceleration (dimension mismatch)
     - Suggested explicit units in --strict mode
   - CLI options: --format json, --strict, --no-recursive
   - Directory linting

7. [ ] Write "IDE Integration" section:
   - How to use in pre-commit hooks
   - Integration with flake8/pylint (future work?)
   - VS Code/PyCharm integration suggestions
   - CI/CD pipeline examples

8. [ ] Write "Best Practices" section:
   - When to use inference vs explicit units
   - Naming conventions for maximum inference accuracy
   - Using inference during prototyping, explicit units in production
   - Combining linting with tests
   - Performance considerations

9. [ ] Add cross-references:
   - Link to API docs for detailed signatures
   - Link to examples.md for physics use cases
   - Link to getting-started.md for installation

10. [ ] Test all code examples:
    - Verify every code snippet runs
    - Ensure output matches documented behavior
    - Test on representative Python 3.9-3.12

11. [ ] Add to documentation index:
    - Update docs/index.md to reference inference guide
    - Add to navigation/TOC if MkDocs is used

---

## Files to Modify

| File | Change |
|------|--------|
| docs/guide/inference.md | CREATE - Main guide document |
| docs/index.md | UPDATE - Add inference guide to table of contents |
| docs/getting-started.md | UPDATE (optional) - Add pointer to inference features |

---

## Testing Strategy

How will we verify this works?

- [ ] Manual testing: Run all code examples in docs
- [ ] Verify code examples produce expected output
- [ ] Test lint examples with actual Python files
- [ ] Proofread for technical accuracy against source code
- [ ] Get feedback from 1-2 test readers if possible
- [ ] Verify all links work (internal and to API docs)
- [ ] Check rendering in documentation viewer (MkDocs/Sphinx)

---

## Risks / Edge Cases

- Risk: Examples become outdated as inference system evolves
  - Mitigation: Include version markers, test examples in CI

- Risk: Users expect inference to be perfect, get false confidence
  - Mitigation: Clearly document confidence levels and limitations

- Risk: Documentation too technical for students
  - Mitigation: Start with simple examples, gradually increase complexity

- Edge case: Ambiguous variable names (e.g., "v" = velocity or voltage?)
  - Handling: Document ambiguity explicitly, show how to resolve

- Edge case: Custom physics domains not in equation database
  - Handling: Document database extensibility (future feature?)

- Risk: CLI examples don't work on Windows
  - Mitigation: Test cross-platform, provide Windows-specific commands if needed

---

## Definition of Done

- [x] Plan document created and reviewed
- [ ] All implementation steps complete
- [ ] All code examples tested and working
- [ ] Documentation renders correctly
- [ ] Cross-references verified
- [ ] CONTINUITY.md updated with task completion
- [ ] Guide accessible from main documentation index

---

## Notes / Log

**2026-01-09 - Initial Planning**
- Analyzed inference codebase:
  - heuristics.py: ~160 variable patterns, confidence scoring
  - equations.py: 50+ physics equations across 8 domains
  - solver.py: Constraint propagation system
  - parser.py: Expression tree builder
  - lint.py: AST-based static analysis, 3 severity levels
- Reviewed test files to understand usage patterns
- Examined existing documentation structure (examples.md style)
- Decision: Single comprehensive guide with clear sections
- Key insight: Emphasize practical bug-catching over theoretical aspects
- Target length: ~1500-2000 lines (similar to examples.md)

---

## Key Examples to Include

### Example 1: Catching a Bug (Opening motivator)
```python
# Without inference - bug goes undetected
velocity = 10  # forgot units!
time = 5
distance = velocity + time  # BUG: adding different dimensions

# With inference linting
$ dimtensor lint buggy.py
buggy.py:3:10: W002 Potential dimension mismatch: L·T⁻¹ + T
```

### Example 2: Variable Name Inference
```python
from dimtensor.inference import infer_dimension

result = infer_dimension("initial_velocity_x")
print(result.dimension)  # L·T⁻¹
print(result.confidence)  # 0.7
print(result.source)  # "prefix_stripped"
```

### Example 3: Equation Database
```python
from dimtensor.inference import get_equations_by_domain

mechanics = get_equations_by_domain("mechanics")
for eq in mechanics[:3]:
    print(f"{eq.name}: {eq.formula}")
# Newton's Second Law: F = ma
# Kinetic Energy: KE = ½mv²
# ...
```

### Example 4: Unit Inference
```python
from dimtensor.inference import infer_units
from dimtensor.core.units import kg, m, s

result = infer_units(
    "F = m * a",
    known_units={"m": kg, "a": m / s**2}
)
print(result['inferred']['F'])  # N (kg·m/s²)
print(result['is_consistent'])  # True
```

### Example 5: CLI Usage
```bash
# Lint a single file
dimtensor lint physics_sim.py

# Lint a directory recursively
dimtensor lint src/ --format json

# Strict mode with suggestions
dimtensor lint . --strict
```

---

## Documentation Structure Outline

```markdown
# Unit Inference and Linting

> Automatic dimension inference and static analysis for dimensional correctness

## Table of Contents
- Overview
- Variable Name Patterns
- Equation Pattern Database
- Unit Inference from Equations
- CLI Linting Tool
- IDE Integration
- Best Practices

## Overview
[What, why, when to use inference]

## Variable Name Patterns
### How It Works
### Confidence Levels
### Common Patterns
### Examples
### Pattern Tables

## Equation Pattern Database
### Physics Domains
### Searching Equations
### Examples
### Database Contents

## Unit Inference from Equations
### Constraint Solving
### API Usage
### Examples
### Limitations

## CLI Linting Tool
### Installation
### Basic Usage
### Severity Levels
### Output Formats
### Examples

## IDE Integration
### Pre-commit Hooks
### CI/CD Integration
### Editor Plugins

## Best Practices
### When to Use Inference
### Naming Conventions
### Combining with Tests
### Performance

## API Reference
[Links to detailed API docs]
```

---

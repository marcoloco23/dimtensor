# Plan: Automatic Unit Inference for Equations

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner-agent

---

## Goal

Implement a constraint-solving system that automatically infers unknown units in equations given some known variable units. For example, given "F = m * a" with m=kg and a=m/s², the system should infer F=N (or kg·m/s²).

---

## Background

### Current State

dimtensor v3.0.0 has:
1. **Variable name heuristics** (src/dimtensor/inference/heuristics.py) - infers units from variable names like "velocity" → m/s
2. **Equation database** (src/dimtensor/equations/database.py) - contains known physics equations with their dimensional relationships

### What's Missing

A **constraint solver** that can:
- Parse an equation (e.g., "F = m * a")
- Accept known variable units (e.g., {m: kg, a: m/s²})
- Propagate dimensional constraints through the equation
- Infer unknown variable units (e.g., F: kg·m/s²)

### Use Cases

1. **Interactive physics coding**: User writes `F = m * a`, system suggests F should be in Newtons
2. **Unit verification**: Check if equation is dimensionally consistent
3. **Educational tool**: Help students understand dimensional analysis
4. **Code completion**: IDE suggests appropriate units based on context

---

## Approach

### Constraint-Based Inference Algorithm

The system will work by treating equations as dimensional constraints:

**Step 1: Parse Equation Structure**
- Identify variables and operations (+, -, *, /, **)
- Build expression tree

**Step 2: Generate Dimensional Constraints**
- For `a = b * c`: `dim(a) = dim(b) * dim(c)`
- For `a = b + c`: `dim(a) = dim(b) = dim(c)` (must match)
- For `a = b / c`: `dim(a) = dim(b) / dim(c)`
- For `a = b ** n`: `dim(a) = dim(b) ** n` (n must be dimensionless)

**Step 3: Constraint Propagation**
- Start with known variable dimensions
- Propagate through constraint graph
- Identify contradictions (dimensional errors)
- Infer unknown dimensions

**Step 4: Simplify & Suggest**
- Simplify inferred dimensions (kg·m/s² → N)
- Return all inferred dimensions with confidence

### Architecture

```
inference/
├── solver.py          # NEW: Core constraint solver
├── parser.py          # NEW: Expression parsing
├── constraints.py     # NEW: Constraint system
└── simplify.py        # NEW: Unit simplification
```

### API Design

```python
from dimtensor.inference import infer_units_from_equation

# Basic usage
result = infer_units_from_equation(
    equation="F = m * a",
    known_units={"m": units.kg, "a": units.m / units.s**2}
)
print(result.inferred)  # {"F": <Unit: kg·m/s² (N)>}
print(result.is_consistent)  # True

# With multiple unknowns
result = infer_units_from_equation(
    equation="KE = 0.5 * m * v**2",
    known_units={"m": units.kg}  # Only mass known
)
print(result.inferred)  # {"v": unknown, "KE": unknown}
print(result.constraints)  # ["KE = m * v²", "v must be L/T", ...]

# Detect inconsistencies
result = infer_units_from_equation(
    equation="F = m + a",  # WRONG!
    known_units={"m": units.kg, "a": units.m / units.s**2}
)
print(result.is_consistent)  # False
print(result.errors)  # ["Cannot add kg and m/s²"]
```

---

## Implementation Steps

### Phase 1: Expression Parser (Task #137a)
1. [ ] Create inference/parser.py
2. [ ] Parse basic arithmetic: +, -, *, /, **
3. [ ] Build expression tree (AST-like)
4. [ ] Handle parentheses
5. [ ] Support constants (numeric literals are dimensionless)
6. [ ] Support function calls: sin(), cos(), exp() (require dimensionless args)

### Phase 2: Constraint System (Task #137b)
1. [ ] Create inference/constraints.py
2. [ ] Define Constraint base class
3. [ ] Implement EqualityConstraint (a + b → dims must match)
4. [ ] Implement MultiplicationConstraint (a * b → dim_a * dim_b)
5. [ ] Implement DivisionConstraint (a / b → dim_a / dim_b)
6. [ ] Implement PowerConstraint (a ** n → dim_a ** n, n dimensionless)
7. [ ] Constraint validation & error reporting

### Phase 3: Constraint Solver (Task #137c)
1. [ ] Create inference/solver.py
2. [ ] Build constraint graph from expression tree
3. [ ] Implement forward propagation (known → unknown)
4. [ ] Implement backward propagation (infer from both sides)
5. [ ] Detect over-constrained systems (contradictions)
6. [ ] Detect under-constrained systems (multiple solutions)
7. [ ] Assign confidence scores

### Phase 4: Unit Simplification (Task #137d)
1. [ ] Create inference/simplify.py
2. [ ] Map complex dimensions to named units (kg·m/s² → N)
3. [ ] Use SI derived unit database
4. [ ] Suggest most common unit for dimension
5. [ ] Optional: dimensional prefix matching (km, cm, etc.)

### Phase 5: Integration & API (Task #137e)
1. [ ] Add infer_units_from_equation() to inference/__init__.py
2. [ ] Integration with equation database (match known equations)
3. [ ] Combine with variable name heuristics
4. [ ] CLI tool: `dimtensor infer "F = m * a" --known m=kg a=m/s²`
5. [ ] Documentation and examples

### Phase 6: Testing (Task #137f)
1. [ ] Unit tests for parser
2. [ ] Unit tests for constraints
3. [ ] Unit tests for solver
4. [ ] Integration tests with common equations
5. [ ] Test edge cases and error handling

---

## Files to Create/Modify

| File | Change |
|------|--------|
| src/dimtensor/inference/parser.py | NEW: Parse equation expressions |
| src/dimtensor/inference/constraints.py | NEW: Constraint classes |
| src/dimtensor/inference/solver.py | NEW: Constraint solver |
| src/dimtensor/inference/simplify.py | NEW: Unit simplification |
| src/dimtensor/inference/__init__.py | MOD: Export infer_units_from_equation |
| tests/inference/test_solver.py | NEW: Solver tests |
| tests/inference/test_parser.py | NEW: Parser tests |
| tests/inference/test_integration.py | MOD: Add equation inference tests |
| docs/inference.md | MOD: Document automatic inference |

---

## Algorithm Examples

### Example 1: Newton's Second Law

**Input:**
- Equation: `F = m * a`
- Known: `m = kg`, `a = m/s²`

**Process:**
1. Parse: `F` = `m` × `a`
2. Constraint: `dim(F) = dim(m) × dim(a)`
3. Substitute: `dim(F) = [M] × [L·T⁻²]`
4. Simplify: `dim(F) = [M·L·T⁻²]`
5. Match: `M·L·T⁻² = N (Newton)`

**Output:**
```python
{
    "F": Unit(Dimension(mass=1, length=1, time=-2), scale=1.0, name="N"),
    "is_consistent": True,
    "confidence": 1.0
}
```

### Example 2: Kinetic Energy

**Input:**
- Equation: `KE = 0.5 * m * v**2`
- Known: `m = kg`, `v = m/s`

**Process:**
1. Parse: `KE` = `0.5` × `m` × (`v` ^ `2`)
2. Constraint: `dim(0.5) = dimensionless`
3. Constraint: `dim(v**2) = dim(v) ** 2 = [L·T⁻¹]² = [L²·T⁻²]`
4. Constraint: `dim(KE) = dim(m) × dim(v**2) = [M] × [L²·T⁻²]`
5. Simplify: `dim(KE) = [M·L²·T⁻²]`
6. Match: `M·L²·T⁻² = J (Joule)`

**Output:**
```python
{
    "KE": Unit(Dimension(mass=1, length=2, time=-2), scale=1.0, name="J"),
    "is_consistent": True,
    "confidence": 1.0
}
```

### Example 3: Ideal Gas Law (Multiple Unknowns)

**Input:**
- Equation: `P * V = n * R * T`
- Known: `P = Pa`, `V = m³`, `T = K`

**Process:**
1. Parse: (`P` × `V`) = (`n` × `R` × `T`)
2. Constraint: `dim(P × V) = dim(n × R × T)`
3. Substitute LHS: `[M·L⁻¹·T⁻²] × [L³] = [M·L²·T⁻²]`
4. Rearrange: `dim(n × R) = [M·L²·T⁻²] / dim(T) = [M·L²·T⁻²·Θ⁻¹]`
5. **Problem**: Can't separate `n` and `R` without more info

**Output:**
```python
{
    "inferred": {},  # Can't fully determine n and R separately
    "constraints": [
        "dim(n) × dim(R) = M·L²·T⁻²·Θ⁻¹",
        "Need one of: n, R to infer the other"
    ],
    "is_consistent": True,
    "confidence": 0.5
}
```

### Example 4: Dimensional Error

**Input:**
- Equation: `F = m + a`  # WRONG!
- Known: `m = kg`, `a = m/s²`

**Process:**
1. Parse: `F` = `m` + `a`
2. Constraint: `dim(F) = dim(m) = dim(a)` (addition requires same dimension)
3. Check: `[M] ≠ [L·T⁻²]`
4. **ERROR**: Incompatible dimensions

**Output:**
```python
{
    "inferred": {},
    "is_consistent": False,
    "errors": [
        "Addition requires matching dimensions",
        "Cannot add [M] (kg) and [L·T⁻²] (m/s²)"
    ],
    "confidence": 0.0
}
```

---

## Integration with Existing Code

### Combine with Variable Name Heuristics

```python
# User writes incomplete equation
result = infer_units_from_equation(
    equation="kinetic_energy = 0.5 * mass * velocity**2",
    known_units={}  # Nothing explicitly known
)

# System uses heuristics:
# 1. "mass" → probably kg (from heuristics.py)
# 2. "velocity" → probably m/s (from heuristics.py)
# 3. Infer "kinetic_energy" from constraint: m * v² → J

print(result.inferred)
# {
#   "mass": kg (confidence: 0.9, source: heuristic),
#   "velocity": m/s (confidence: 0.9, source: heuristic),
#   "kinetic_energy": J (confidence: 0.8, source: inferred)
# }
```

### Match Against Equation Database

```python
# Check if equation matches known pattern
result = infer_units_from_equation(
    equation="F = m * a",
    known_units={"m": units.kg, "a": units.m/units.s**2}
)

# System recognizes this as Newton's Second Law
print(result.matched_equation)  # "Newton's Second Law"
print(result.reference)  # Link to equation database entry
```

---

## Testing Strategy

### Unit Tests

- [ ] **Parser tests**: Parse "a + b", "a * b / c", "a**2", "(a + b) * c"
- [ ] **Constraint tests**: Each constraint type validates correctly
- [ ] **Solver tests**: Known → unknown propagation
- [ ] **Error detection**: Catch dimensional mismatches
- [ ] **Simplification**: kg·m/s² → N, J/s → W

### Integration Tests

- [ ] Test on all equations in database.py
- [ ] F = ma, KE = ½mv², W = Fd, P = F/A
- [ ] PV = nRT, Q = mcΔT, V = IR, P = IV
- [ ] Complex equations: Bernoulli, Navier-Stokes (partial)

### Edge Cases

- [ ] Constants in equations (c = 3e8 m/s)
- [ ] Multiple unknowns
- [ ] Over-constrained systems
- [ ] Under-constrained systems
- [ ] Transcendental functions (sin, cos require dimensionless)
- [ ] Implicit equations (F - ma = 0)

---

## Risks / Edge Cases

### Risk 1: Expression Parsing Complexity
- **Problem**: Parsing arbitrary Python expressions is complex
- **Mitigation**: Start with basic arithmetic, extend gradually. Use AST if needed.

### Risk 2: Under-Constrained Systems
- **Problem**: Can't always infer all unknowns (e.g., PV = nRT with only 3 knowns)
- **Mitigation**: Report partial results, suggest what's needed

### Risk 3: Non-Linear Constraints
- **Problem**: Some equations have multiple valid solutions
- **Mitigation**: Report ambiguity, use heuristics to suggest most likely

### Edge Case 1: Dimensionless Constants
- **Problem**: How to handle π, e, 0.5 in equations?
- **Handling**: Treat all numeric literals as dimensionless

### Edge Case 2: Transcendental Functions
- **Problem**: sin(x), exp(x) require dimensionless input
- **Handling**: Add constraint that argument must be dimensionless

### Edge Case 3: Vector Operations
- **Problem**: Dot products, cross products, norms
- **Handling**: Future work - v3.2.0 or later

---

## Definition of Done

- [ ] Parser can handle arithmetic expressions with precedence
- [ ] Constraint system supports +, -, *, /, **
- [ ] Solver propagates known → unknown dimensions
- [ ] Solver detects dimensional errors
- [ ] Unit simplification suggests named units (N, J, W, etc.)
- [ ] API: infer_units_from_equation() works as specified
- [ ] Tests pass (>90% coverage for new code)
- [ ] Documentation with 5+ examples
- [ ] Integration with equation database
- [ ] CLI tool: `dimtensor infer "equation" --known x=unit`

---

## Notes / Log

**2026-01-09 (PLANNING)** - Created plan for automatic unit inference via constraint solving. This is orthogonal to existing variable name heuristics - it's about propagating dimensional constraints through equations. Key insight: treat equations as constraint satisfaction problems where dimensions must be consistent.

**Design Decision**: Use explicit constraint propagation rather than full symbolic math. This is simpler, more predictable, and sufficient for most use cases. Future versions could integrate with SymPy for more complex symbolic analysis.

**Estimated Complexity**: Medium-High
- Parser: Medium (can use Python's AST or simple recursive descent)
- Constraints: Low (straightforward dimensional arithmetic)
- Solver: Medium (graph traversal + propagation)
- Integration: Low

---

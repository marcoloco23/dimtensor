# Plan: Numba Integration for JIT-Compiled Unit Operations

**Date**: 2026-01-10
**Status**: PLANNING
**Author**: planner agent (task #204)

---

## Goal

Enable Numba JIT compilation for unit-aware operations in dimtensor, allowing users to write `@jit`-decorated functions that accept `DimArray` inputs with efficient dimension checking and parallel loop support via `prange`.

---

## Background

### Why Numba?
- Numba provides LLVM-based JIT compilation for NumPy-heavy Python code
- Can achieve near-C performance with minimal code changes
- Supports parallel loops via `prange` for embarrassingly parallel operations
- Native NumPy array support makes it ideal for scientific computing

### Current State in dimtensor
- `DimArray` wraps NumPy arrays with `Unit` and `Dimension` metadata
- `JAX DimArray` uses pytree registration for JIT compatibility
- Rust backend (`_rust.py`) provides optional accelerated operations
- No current Numba support

### Key Challenge
Numba's `nopython` mode requires all types to be known at compile time. `DimArray` contains:
1. `_data`: NumPy array (Numba-compatible)
2. `_unit`: `Unit` dataclass with `Dimension` (not Numba-compatible by default)
3. `_uncertainty`: Optional NumPy array (Numba-compatible)

The `Dimension` class uses `Fraction` exponents which are not natively supported in Numba.

---

## Approach

### Option A: StructRef-Based Custom Type
- Register `DimArray` as a Numba StructRef with explicit type model
- Define lowering for dimension algebra operations
- Pros: Full integration, clean API, dimension checking at compile time
- Cons: Complex implementation, significant Numba internals knowledge needed

### Option B: Extraction Pattern (like JAX pytree)
- Extract raw arrays before JIT, validate dimensions outside JIT
- Pass raw arrays to compiled functions, reconstruct DimArray after
- Pros: Simple, works today, minimal Numba complexity
- Cons: Dimension checking happens outside JIT, less elegant API

### Option C: Hybrid with NumbaInt64 Dimension Encoding
- Encode dimension as int64 tuple (numerator/denominator pairs for 7 dimensions = 14 int64s)
- Create lightweight `DimArrayData` structref for JIT containing (data, dim_encoding, scale)
- Validate/decode dimensions outside JIT, perform pure numeric operations inside
- Pros: Good balance of performance and safety, moderate complexity
- Cons: Some overhead for encoding/decoding

### Decision: Option C (Hybrid Approach)

Rationale:
1. Matches the existing pattern used in `_rust.py` for Rust integration
2. Allows dimension checking to be hoisted outside loops
3. Supports both eager (single-call) and lazy (loop-fused) modes
4. Integer-encoded dimensions can be compared efficiently in compiled code
5. Follows dimtensor's philosophy of catching errors early while enabling performance

---

## Architecture Design

### Module Structure

```
src/dimtensor/numba/
    __init__.py          # Public API, exports dim_jit, prange, etc.
    _types.py            # DimArrayData structref and type definitions
    _lowering.py         # Numba lowering implementations
    _decorators.py       # @dim_jit decorator and helpers
    _encoding.py         # Dimension <-> int64 tuple encoding
    _operations.py       # JIT-compiled operation kernels
```

### Core Types

```python
# _encoding.py
def encode_dimension(dim: Dimension) -> tuple[int, ...]:
    """Encode Dimension as 14 int64s (7 numerator/denominator pairs)."""
    result = []
    for exp in dim._exponents:
        result.extend([exp.numerator, exp.denominator])
    return tuple(result)

def decode_dimension(encoded: tuple[int, ...]) -> Dimension:
    """Decode 14 int64s back to Dimension."""
    from fractions import Fraction
    exponents = []
    for i in range(0, 14, 2):
        exponents.append(Fraction(encoded[i], encoded[i+1]))
    return Dimension._from_exponents(tuple(exponents))

# _types.py - Numba StructRef
from numba.experimental import structref
from numba import types

class DimArrayDataType(structref.StructRef):
    """Numba type for dimension-aware array data."""
    pass

@structref.register_structref_proxy(DimArrayDataType)
class DimArrayDataProxy(structref.StructRefProxy):
    @property
    def data(self):
        return structref_getattr(self, 'data')

    @property
    def dim_encoded(self):
        return structref_getattr(self, 'dim_encoded')

    @property
    def scale(self):
        return structref_getattr(self, 'scale')

# Fields: data (array), dim_encoded (int64[14]), scale (float64)
DimArrayData = structref.define_type(
    'DimArrayData',
    [
        ('data', types.float64[:]),  # Or generic array type
        ('dim_encoded', types.int64[14]),
        ('scale', types.float64),
    ]
)
```

### Decorator Design

```python
# _decorators.py
from numba import njit, prange
from functools import wraps

def dim_jit(fn=None, *, parallel=False, cache=True, fastmath=False):
    """JIT compile a function with dimension-aware inputs.

    Dimension checking occurs before the JIT-compiled code runs.
    Inside the compiled region, only raw arrays are manipulated.

    Usage:
        @dim_jit
        def kinetic_energy(mass, velocity):
            return 0.5 * mass * velocity**2

        @dim_jit(parallel=True)
        def vector_sum(arr):
            total = 0.0
            for i in prange(arr.shape[0]):
                total += arr[i]
            return total

    Args:
        parallel: Enable automatic parallelization (default: False)
        cache: Cache compiled function to disk (default: True)
        fastmath: Enable fast math optimizations (default: False)

    Returns:
        Decorated function that accepts DimArray inputs
    """
    def decorator(func):
        # Analyze function signature for dimension constraints
        # Generate dimension-checking wrapper
        # Create inner JIT-compiled function on raw arrays
        # Wrap result in DimArray with computed output dimension
        ...

    if fn is not None:
        return decorator(fn)
    return decorator
```

### Two Execution Modes

#### Mode 1: Eager (Simple Operations)
```python
@dim_jit
def add_arrays(a, b):
    return a + b

# Usage:
result = add_arrays(distance1, distance2)
# 1. Extract dimensions, verify compatible
# 2. Extract raw arrays
# 3. Call JIT-compiled kernel
# 4. Wrap result with computed dimension
```

#### Mode 2: Fused (Parallel Loops)
```python
@dim_jit(parallel=True)
def parallel_norm(positions):
    n = positions.shape[0]
    norms = np.empty(n)
    for i in prange(n):
        norms[i] = np.sqrt(positions[i, 0]**2 + positions[i, 1]**2 + positions[i, 2]**2)
    return norms

# Dimension checking happens once before the loop
# All iterations execute in parallel without dimension overhead
```

### Dimension Algebra in Compiled Code

For operations inside JIT, we provide compiled dimension helpers:

```python
# _operations.py
from numba import njit

@njit(inline='always')
def dims_equal(dim_a: types.int64[14], dim_b: types.int64[14]) -> bool:
    """Check if two encoded dimensions are equal."""
    for i in range(14):
        if dim_a[i] != dim_b[i]:
            return False
    return True

@njit(inline='always')
def dims_multiply(dim_a: types.int64[14], dim_b: types.int64[14]) -> types.int64[14]:
    """Multiply two encoded dimensions (add exponents)."""
    result = np.empty(14, dtype=np.int64)
    for i in range(0, 14, 2):
        # (a_num/a_den) + (b_num/b_den) = (a_num*b_den + b_num*a_den) / (a_den*b_den)
        a_num, a_den = dim_a[i], dim_a[i+1]
        b_num, b_den = dim_b[i], dim_b[i+1]
        result[i] = a_num * b_den + b_num * a_den
        result[i+1] = a_den * b_den
        # Simplify by GCD
        g = gcd(abs(result[i]), result[i+1])
        result[i] //= g
        result[i+1] //= g
    return result

@njit(inline='always')
def gcd(a: types.int64, b: types.int64) -> types.int64:
    """Greatest common divisor (Euclidean algorithm)."""
    while b:
        a, b = b, a % b
    return a
```

---

## Implementation Steps

### Phase 1: Core Infrastructure
1. [ ] Create `src/dimtensor/numba/` package structure
2. [ ] Implement `_encoding.py` with dimension encoding/decoding
3. [ ] Add unit tests for encoding round-trip correctness
4. [ ] Implement `_types.py` with DimArrayData structref
5. [ ] Add type inference and model registration

### Phase 2: Basic JIT Support
6. [ ] Implement `_decorators.py` with basic `@dim_jit`
7. [ ] Support scalar arithmetic operations (+, -, *, /)
8. [ ] Support power operations (**)
9. [ ] Implement dimension tracking through operations
10. [ ] Add error handling for dimension mismatches

### Phase 3: Parallel Support
11. [ ] Implement `prange` integration for parallel loops
12. [ ] Add reduction variable detection
13. [ ] Support `parallel=True` in `@dim_jit`
14. [ ] Add `fastmath=True` option

### Phase 4: NumPy Functions
15. [ ] Implement JIT-compatible `np.sum`, `np.mean`, etc.
16. [ ] Implement `np.sqrt`, `np.sin`, etc. with dimension checks
17. [ ] Support `np.dot`, `np.matmul` with dimension multiplication

### Phase 5: Integration & Polish
18. [ ] Add to `src/dimtensor/__init__.py` exports
19. [ ] Write comprehensive documentation
20. [ ] Add performance benchmarks
21. [ ] Integration tests with existing dimtensor features

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/numba/__init__.py` | New: Public API exports |
| `src/dimtensor/numba/_types.py` | New: StructRef type definitions |
| `src/dimtensor/numba/_encoding.py` | New: Dimension encoding utilities |
| `src/dimtensor/numba/_decorators.py` | New: @dim_jit decorator |
| `src/dimtensor/numba/_lowering.py` | New: Numba lowering implementations |
| `src/dimtensor/numba/_operations.py` | New: JIT-compiled kernels |
| `src/dimtensor/__init__.py` | Add numba exports |
| `pyproject.toml` | Add numba to optional dependencies |
| `tests/test_numba.py` | New: Test suite for Numba integration |

---

## Testing Strategy

### Unit Tests
- [ ] Dimension encoding/decoding round-trip
- [ ] DimArrayData structref creation and access
- [ ] `@dim_jit` basic function compilation
- [ ] Dimension checking for +/- (requires same dimension)
- [ ] Dimension multiplication for * /
- [ ] Power operation dimension handling
- [ ] Error messages for dimension mismatches

### Performance Tests
- [ ] Compare `@dim_jit` vs pure NumPy vs plain `@njit`
- [ ] Parallel speedup with `prange`
- [ ] JIT compilation overhead (first call vs subsequent)
- [ ] Large array scaling tests

### Integration Tests
- [ ] Interoperability with existing DimArray
- [ ] Round-trip: DimArray -> dim_jit -> DimArray
- [ ] Mixed operations (DimArray + raw arrays)
- [ ] Uncertainty propagation through JIT

---

## Risks / Edge Cases

### Risk 1: Fraction Overflow
- Dimension exponents use `Fraction` which can have arbitrary precision
- Encoding as int64 may overflow for pathological cases
- **Mitigation**: Use `limit_denominator(1000)` as already done in Dimension.__init__

### Risk 2: Type Inference Complexity
- Numba's type system is complex and may not handle all edge cases
- **Mitigation**: Start with float64 arrays only, extend later

### Risk 3: Memory Layout
- DimArrayData structref has specific memory requirements
- **Mitigation**: Use contiguous arrays, document requirements

### Edge Case: Dimensionless Operations
- `np.sin`, `np.exp` require dimensionless input
- **Handling**: Check dimension before JIT, raise DimensionError

### Edge Case: Mixed DimArray/scalar Operations
- `DimArray * 2.0` should preserve dimension
- **Handling**: Type dispatch in decorator

### Edge Case: Broadcasting
- NumPy broadcasting with different shapes
- **Handling**: Rely on NumPy's broadcasting, dimension check once

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Tests pass (unit, performance, integration)
- [ ] Documentation in docstrings and README
- [ ] Performance at least 10x faster than pure Python for parallel workloads
- [ ] Works with Python 3.10, 3.11, 3.12
- [ ] Numba optional dependency (graceful fallback if not installed)
- [ ] CONTINUITY.md updated
- [ ] Example notebook demonstrating usage

---

## API Examples

### Basic Usage

```python
from dimtensor import DimArray, units
from dimtensor.numba import dim_jit, prange

# Simple JIT compilation
@dim_jit
def gravitational_force(m1, m2, r):
    """F = G * m1 * m2 / r^2"""
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    return G * m1 * m2 / r**2

mass1 = DimArray(5.972e24, units.kg)  # Earth mass
mass2 = DimArray(7.348e22, units.kg)  # Moon mass
distance = DimArray(3.844e8, units.m)  # Earth-Moon distance

force = gravitational_force(mass1, mass2, distance)
print(force)  # ~1.98e20 N
```

### Parallel Computation

```python
@dim_jit(parallel=True)
def particle_distances(positions, reference):
    """Compute distance from each particle to reference point."""
    n = positions.shape[0]
    distances = np.empty(n)
    for i in prange(n):
        dx = positions[i, 0] - reference[0]
        dy = positions[i, 1] - reference[1]
        dz = positions[i, 2] - reference[2]
        distances[i] = np.sqrt(dx*dx + dy*dy + dz*dz)
    return distances

# positions: (N, 3) array in meters
# reference: (3,) array in meters
# result: (N,) array in meters
```

### Error Handling

```python
@dim_jit
def bad_addition(length, time):
    return length + time  # DimensionError before JIT runs

length = DimArray([1.0, 2.0], units.m)
time = DimArray([3.0, 4.0], units.s)

bad_addition(length, time)
# Raises: DimensionError: Cannot add quantities with dimensions L and T
```

---

## Notes / Log

**2026-01-10 13:30** - Initial plan created by planner agent

**Key Research Findings:**
1. Numba StructRef API is stable and suitable for custom types
2. `nopython=True` with `parallel=True` enables prange parallelization
3. JAX pytree pattern provides good template for metadata handling
4. Rust backend pattern shows dimension encoding approach

**Open Questions:**
- Q: Should we support GPU via numba.cuda?
  - A: Defer to separate plan, start with CPU-only
- Q: Support for complex arrays?
  - A: Start with float64, extend in Phase 5

---

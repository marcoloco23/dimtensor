# Plan: Rust Backend for dimtensor

**Date**: 2026-01-09
**Status**: IN PROGRESS
**Author**: agent

---

## Goal

Add an optional Rust backend to dimtensor for performance-critical operations, targeting <10% overhead compared to raw NumPy while maintaining full Python API compatibility.

---

## Background

Current pure-Python implementation has overhead from:
1. Dimension checking on every operation
2. Unit conversion calculations
3. Python object creation for each result

For large-scale physics simulations and ML training loops, this overhead can be significant. A Rust backend can:
- Perform dimension checks at compile-time where possible
- Use SIMD operations for array math
- Minimize Python object allocations
- Enable lazy evaluation and operator fusion

---

## Research Summary (Task #63)

### PyO3 (Python-Rust bindings)
- Primary library for Python extension modules in Rust
- Build with `maturin` for easy distribution
- `#[pyclass]` for Python-visible classes, `#[pymethods]` for methods
- Automatic GIL management

### rust-numpy (NumPy integration)
- Zero-copy array access via `PyReadonlyArray` and `PyReadwriteArray`
- Direct access to NumPy array data as Rust slices
- Support for all NumPy dtypes
- Uses Rust's `ndarray` crate internally

### Build System: maturin
- Handles cross-platform compilation
- Builds wheels for PyPI distribution
- `maturin develop` for local development
- `maturin build` for release wheels

---

## Approach

### Option A: Full Rust Core
- Rewrite Dimension, Unit, and DimArray entirely in Rust
- Maximum performance
- Cons: Large effort, complex FFI for all methods, harder to maintain

### Option B: Accelerated Operations (CHOSEN)
- Keep Python types for Dimension, Unit, DimArray
- Accelerate specific hot paths in Rust:
  - Array arithmetic operations (+, -, *, /, **)
  - Dimension checking (validate compatible, multiply dimensions)
  - Common ufuncs (sin, cos, sqrt, exp, log)
- Pure Python fallback when Rust unavailable
- Pros: Incremental, maintains compatibility, measurable ROI
- Cons: Still some Python overhead

### Option C: Hybrid Types
- Rust types with Python wrappers
- Middle ground complexity and performance
- Cons: Complex to maintain two type systems

### Decision: Option B

Start with accelerated operations. This gives us:
1. Immediate measurable performance gains
2. Maintains 100% API compatibility
3. Pure Python fallback for platforms without Rust
4. Can evolve toward Option C later if needed

---

## Implementation Steps

### Phase 1: Project Setup (Task #65)
1. [ ] Create rust/ folder with Cargo workspace
2. [ ] Set up maturin build in rust/Cargo.toml
3. [ ] Add rust-numpy and pyo3 dependencies
4. [ ] Configure pyproject.toml for optional Rust build
5. [ ] Create basic _dimtensor_core module structure

### Phase 2: Core Types in Rust (Tasks #66-67)
1. [ ] Implement RustDimension struct (7 rational exponents)
2. [ ] Implement dimension arithmetic (multiply, divide, power)
3. [ ] Implement dimension compatibility checking
4. [ ] Implement RustUnit struct (dimension + scale)
5. [ ] Add Python bindings for dimension operations

### Phase 3: Array Operations (Task #68)
1. [ ] Implement add_arrays(a, b, dim_a, dim_b) -> result, dim_result
2. [ ] Implement mul_arrays(a, b, dim_a, dim_b) -> result, dim_result
3. [ ] Implement div_arrays(a, b, dim_a, dim_b) -> result, dim_result
4. [ ] Implement pow_array(a, exp, dim) -> result, dim_result
5. [ ] All with zero-copy via PyReadonlyArray

### Phase 4: Python Integration (Task #69)
1. [ ] Create dimtensor._rust module
2. [ ] Add feature detection: `HAS_RUST_BACKEND`
3. [ ] Create fallback mechanism in DimArray operations
4. [ ] Hook Rust operations into existing DimArray class

### Phase 5: Advanced Features (Tasks #70-72)
1. [ ] Implement lazy evaluation graph
2. [ ] Implement operator fusion for common patterns
3. [ ] Optimize memory: minimize allocations
4. [ ] Add SIMD hints where beneficial

### Phase 6: Benchmarking & Polish (Tasks #73-76)
1. [ ] Benchmark vs pure Python implementation
2. [ ] Benchmark vs raw NumPy
3. [ ] Target: <10% overhead vs NumPy
4. [ ] Add tests for Rust backend
5. [ ] Update pyproject.toml build config
6. [ ] Document installation with Rust

---

## Files to Create/Modify

| File | Change |
|------|--------|
| rust/Cargo.toml | NEW: Rust workspace configuration |
| rust/src/lib.rs | NEW: Main Rust library entry point |
| rust/src/dimension.rs | NEW: Dimension type in Rust |
| rust/src/unit.rs | NEW: Unit type in Rust |
| rust/src/ops.rs | NEW: Array operations |
| rust/src/lazy.rs | NEW: Lazy evaluation (Phase 5) |
| src/dimtensor/_rust.py | NEW: Rust backend Python wrapper |
| src/dimtensor/core/dimarray.py | MOD: Add Rust backend hooks |
| pyproject.toml | MOD: Add maturin build config |
| tests/test_rust_backend.py | NEW: Tests for Rust backend |

---

## Testing Strategy

- [ ] Unit tests for Rust dimension operations
- [ ] Unit tests for Rust array operations
- [ ] Integration tests: same results as pure Python
- [ ] Benchmark tests: verify performance targets
- [ ] Test fallback behavior when Rust unavailable
- [ ] Test all dtypes (float32, float64, int32, int64)

---

## Risks / Edge Cases

- **Risk**: Cross-platform compilation complexity
  - Mitigation: Use maturin's cross-compile support, test on CI

- **Risk**: Memory safety at FFI boundary
  - Mitigation: Use PyReadonlyArray for safe borrowing

- **Risk**: Version compatibility PyO3/numpy/Python
  - Mitigation: Pin versions, test against multiple Python versions

- **Edge case**: Non-contiguous arrays
  - Handling: Fall back to Python for non-contiguous

- **Edge case**: Custom dtypes
  - Handling: Fall back to Python for unsupported dtypes

---

## Definition of Done

- [ ] Rust workspace compiles
- [ ] Python can import _dimtensor_core
- [ ] All DimArray operations work with Rust backend
- [ ] Pure Python fallback works when Rust unavailable
- [ ] Benchmarks show <10% overhead vs raw NumPy
- [ ] All existing tests pass
- [ ] CI builds wheels for major platforms
- [ ] README documents Rust installation

---

## Notes / Log

**2026-01-09** - Completed PyO3 research. Key findings:
- PyO3 + rust-numpy is mature and well-documented
- maturin is the recommended build tool
- PyReadonlyArray enables zero-copy array access
- Start with accelerated operations, evolve later

---

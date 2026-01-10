# Plan: CuPy Integration for GPU Arrays with Units

**Date**: 2026-01-10
**Status**: PLANNING
**Author**: planner agent
**Task**: #201

---

## Goal

Create a CuPy-based DimArray that wraps CuPy GPU arrays with physical unit tracking, mirroring the NumPy DimArray API while supporting GPU memory management and CUDA-specific features.

---

## Background

### Why CuPy Integration?

CuPy is a NumPy-compatible GPU array library that enables seamless transition from CPU to GPU computation. Unlike PyTorch (which we already support via DimTensor), CuPy:
- Provides NumPy-compatible API (drop-in replacement)
- Supports raw CUDA kernel writing
- Has excellent integration with existing NumPy codebases
- Is commonly used in scientific computing workflows

### Current State

dimtensor already has:
- **NumPy DimArray** (`core/dimarray.py`): 1019 lines, full-featured with uncertainty propagation
- **JAX DimArray** (`jax/dimarray.py`): 509 lines, pytree-registered for JIT/vmap/grad
- **PyTorch DimTensor** (`torch/dimtensor.py`): 595 lines, autograd-compatible

### Design Principles from Existing Backends

All backends share common patterns:
1. `__slots__` for memory efficiency: `("_data", "_unit")` or `("_data", "_unit", "_uncertainty")`
2. Internal constructor `_from_data_and_unit()` that bypasses copying
3. Same property interface: `data`, `unit`, `dimension`, `shape`, `ndim`, `size`, `dtype`, `is_dimensionless`
4. Consistent error handling via `DimensionError` and `UnitConversionError`
5. Unit arithmetic: `+/-` require same dimension, `*//` combine dimensions, `**` requires scalar exponent

### CuPy-Specific Considerations

- GPU memory management (explicit transfers between CPU/GPU)
- CUDA stream support for async operations
- Memory pool management
- Device context management (multi-GPU)
- Interoperability with NumPy (easy CPU fallback)

---

## Approach

### Option A: Minimal Port (JAX-style)

**Description**: Create a lean DimArray similar to JAX version (~500 lines), without uncertainty propagation.

**Pros**:
- Quick to implement
- Smaller codebase to maintain
- Matches JAX complexity level

**Cons**:
- Missing uncertainty propagation (important for scientific computing)
- Users may expect NumPy DimArray feature parity
- Would need separate implementation later for uncertainty

### Option B: Full Port (NumPy-style)

**Description**: Create a feature-complete DimArray with uncertainty propagation, matching NumPy DimArray functionality.

**Pros**:
- Full feature parity with NumPy DimArray
- Uncertainty propagation on GPU
- Users can migrate NumPy code directly

**Cons**:
- More complex implementation (~1000 lines)
- Uncertainty on GPU may have performance implications
- More testing required

### Option C: Full Port with GPU-Specific Enhancements

**Description**: Full NumPy parity plus CuPy-specific features (memory pools, streams, multi-GPU).

**Pros**:
- Best GPU utilization
- Production-ready for large-scale scientific computing
- Showcases dimtensor's GPU capabilities

**Cons**:
- Most complex implementation
- Requires deep CuPy knowledge
- May be over-engineered for initial release

### Decision: Option B - Full Port (NumPy-style)

**Rationale**:
1. Scientific computing users expect uncertainty propagation
2. CuPy's NumPy compatibility makes porting straightforward
3. GPU-specific features can be added incrementally in v4.3.0
4. Maintains consistency with NumPy DimArray (users learn one API)

**GPU memory features (deferred to v4.3.0)**:
- Custom memory pools
- CUDA stream support
- Multi-GPU device management

---

## Implementation Steps

### Phase 1: Module Setup (1-4)
1. [ ] Create `src/dimtensor/cupy/` directory
2. [ ] Create `src/dimtensor/cupy/__init__.py` with lazy import pattern
3. [ ] Create `src/dimtensor/cupy/dimarray.py` skeleton
4. [ ] Add CuPy availability detection function

### Phase 2: Core DimArray Class (5-11)
5. [ ] Implement class with `__slots__ = ("_data", "_unit", "_uncertainty")`
6. [ ] Implement `__init__` constructor with CuPy array handling
7. [ ] Implement `_from_data_and_unit()` internal constructor
8. [ ] Implement properties: `data`, `unit`, `dimension`, `shape`, `ndim`, `size`, `dtype`
9. [ ] Implement `is_dimensionless` property
10. [ ] Implement uncertainty properties: `uncertainty`, `has_uncertainty`, `relative_uncertainty`
11. [ ] Implement `device` property (return current GPU device)

### Phase 3: Unit Conversion (12-14)
12. [ ] Implement `to(unit)` method with uncertainty scaling
13. [ ] Implement `to_base_units()` method
14. [ ] Implement `magnitude()` method

### Phase 4: Uncertainty Propagation Helpers (15-17)
15. [ ] Implement `_propagate_add_sub()` static method
16. [ ] Implement `_propagate_mul_div()` static method
17. [ ] Implement `_propagate_power()` static method

### Phase 5: Arithmetic Operations (18-27)
18. [ ] Implement `__add__` / `__radd__` with uncertainty
19. [ ] Implement `__sub__` / `__rsub__` with uncertainty
20. [ ] Implement `__mul__` / `__rmul__` with uncertainty
21. [ ] Implement `__truediv__` / `__rtruediv__` with uncertainty
22. [ ] Implement `__pow__` with uncertainty
23. [ ] Implement `__neg__` / `__pos__` / `__abs__`
24. [ ] Implement `sqrt()` method
25. [ ] Handle Constant type (late import pattern)
26. [ ] Verify broadcasting works correctly with CuPy
27. [ ] Test GPU array operations don't trigger CPU transfers

### Phase 6: Comparison Operations (28-32)
28. [ ] Implement `__eq__` / `__ne__`
29. [ ] Implement `__lt__` / `__le__`
30. [ ] Implement `__gt__` / `__ge__`
31. [ ] Ensure comparisons return CuPy arrays (not NumPy)
32. [ ] Test dimension checking for comparisons

### Phase 7: Indexing (33-35)
33. [ ] Implement `__getitem__` preserving units and uncertainty
34. [ ] Implement `__len__`
35. [ ] Implement `__iter__`

### Phase 8: Reduction Operations (36-42)
36. [ ] Implement `sum()` with uncertainty propagation
37. [ ] Implement `mean()` with uncertainty propagation
38. [ ] Implement `std()` (uncertainty propagation complex, return None)
39. [ ] Implement `var()` with squared units
40. [ ] Implement `min()` / `max()` with uncertainty selection
41. [ ] Implement `argmin()` / `argmax()` (return indices)
42. [ ] Verify reductions work on correct axis

### Phase 9: Reshaping Operations (43-46)
43. [ ] Implement `reshape()`
44. [ ] Implement `transpose()`
45. [ ] Implement `flatten()`
46. [ ] Implement `squeeze()` / `expand_dims()` (from JAX pattern)

### Phase 10: Linear Algebra (47-50)
47. [ ] Implement `dot()` with dimension multiplication
48. [ ] Implement `matmul()` / `__matmul__`
49. [ ] Implement `norm()` (preserves units for L1/L2/Linf)
50. [ ] Use CuPy's cuBLAS bindings for performance

### Phase 11: String Representations (51-53)
51. [ ] Implement `__repr__`
52. [ ] Implement `__str__` with display options
53. [ ] Implement `__format__`

### Phase 12: NumPy/CuPy Interop (54-58)
54. [ ] Implement `__array__` protocol for NumPy conversion
55. [ ] Implement `numpy()` method (explicit CPU transfer)
56. [ ] Implement `cupy()` method (return underlying array)
57. [ ] Implement `get()` method (CuPy convention for CPU transfer)
58. [ ] Implement `set()` method (update from NumPy array)

### Phase 13: GPU Memory Management (59-63)
59. [ ] Implement `device` property
60. [ ] Implement `to_device(device_id)` method for multi-GPU
61. [ ] Implement `cpu()` method (convenience wrapper)
62. [ ] Implement `gpu()` method (convenience wrapper)
63. [ ] Add context manager for device selection

### Phase 14: Module Exports (64-66)
64. [ ] Update `__init__.py` with public API
65. [ ] Add type annotations throughout
66. [ ] Add module docstring with examples

### Phase 15: Testing (67-76)
67. [ ] Create `tests/test_cupy_dimarray.py`
68. [ ] Test basic construction and properties
69. [ ] Test arithmetic operations with dimension checking
70. [ ] Test uncertainty propagation
71. [ ] Test unit conversion
72. [ ] Test reductions preserve units
73. [ ] Test GPU memory doesn't leak
74. [ ] Test CPU/GPU interoperability
75. [ ] Test edge cases (empty arrays, scalar arrays)
76. [ ] Add skip decorators for systems without CuPy/GPU

### Phase 16: Documentation (77-79)
77. [ ] Add docstrings to all public methods
78. [ ] Add usage examples in module docstring
79. [ ] Update package exports if needed

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/cupy/__init__.py` | Create: Module initialization, public exports, availability check |
| `src/dimtensor/cupy/dimarray.py` | Create: CuPy DimArray implementation (~1000 lines) |
| `src/dimtensor/__init__.py` | Modify: Add cupy submodule to `__all__` if desired |
| `tests/test_cupy_dimarray.py` | Create: Comprehensive test suite |
| `pyproject.toml` | Modify: Add cupy as optional dependency `cupy = {version = ">=12.0.0", optional = true}` |

---

## Testing Strategy

### Unit Tests
- [ ] Test each arithmetic operation independently
- [ ] Test dimension checking (errors on mismatch)
- [ ] Test unit conversion with compatible/incompatible units
- [ ] Test uncertainty propagation formulas
- [ ] Test all reduction operations
- [ ] Test reshaping preserves units/uncertainty

### Integration Tests
- [ ] Test with physical constants
- [ ] Test in physics equations (F=ma, E=mc^2)
- [ ] Test mixed DimArray/scalar operations
- [ ] Test CPU to GPU roundtrip (NumPy -> CuPy -> NumPy)

### GPU-Specific Tests
- [ ] Test memory doesn't leak after operations
- [ ] Test operations stay on GPU (no accidental transfers)
- [ ] Test multi-GPU scenarios (if hardware available)
- [ ] Test large arrays (>1M elements)

### Skip Conditions
```python
import pytest

cupy_available = pytest.importorskip("cupy")

@pytest.mark.skipif(not cupy_available, reason="CuPy not installed")
def test_cupy_dimarray():
    ...
```

---

## Risks / Edge Cases

### Risk 1: CuPy Version Compatibility
**Problem**: CuPy API may differ across versions.
**Mitigation**:
- Target CuPy >= 12.0.0 (stable API)
- Test against multiple CuPy versions in CI
- Document minimum required version

### Risk 2: Memory Transfer Overhead
**Problem**: Operations may inadvertently trigger CPU-GPU transfers.
**Mitigation**:
- Use CuPy arrays throughout (avoid np.asarray calls)
- Profile with `cupy.cuda.memory_hook`
- Add explicit transfer methods (`cpu()`, `gpu()`)

### Risk 3: Uncertainty Propagation Performance
**Problem**: Uncertainty math doubles memory usage and computation.
**Mitigation**:
- Make uncertainty optional (only allocate if provided)
- Consider lazy evaluation for uncertainty
- Benchmark with and without uncertainty

### Risk 4: CUDA Not Available
**Problem**: Users may import dimtensor.cupy without GPU.
**Mitigation**:
- Lazy import with clear error message
- Provide `is_available()` function
- Document GPU requirements

### Risk 5: Numerical Precision
**Problem**: GPU float32 may differ from CPU float64 defaults.
**Mitigation**:
- Default to float64 like NumPy DimArray
- Provide dtype parameter
- Document precision differences

### Edge Case 1: Empty Arrays
**Scenario**: DimArray with shape (0,) or (0, 3).
**Handling**: Support empty arrays, return empty results for reductions.

### Edge Case 2: Scalar Arrays
**Scenario**: 0-dimensional CuPy array.
**Handling**: Wrap scalars in 1-element array like NumPy DimArray.

### Edge Case 3: Non-contiguous Arrays
**Scenario**: Array from slicing is non-contiguous.
**Handling**: CuPy handles this transparently, but profile for performance.

### Edge Case 4: Mixed Device Arrays
**Scenario**: Operations between arrays on different GPUs.
**Handling**: Raise clear error, require explicit device transfer.

---

## Definition of Done

- [ ] All implementation steps complete (1-79)
- [ ] DimArray mirrors NumPy DimArray API
- [ ] Uncertainty propagation works correctly on GPU
- [ ] All tests pass (skip gracefully without CuPy)
- [ ] No memory leaks in GPU operations
- [ ] Operations stay on GPU (no accidental transfers)
- [ ] Type annotations complete
- [ ] Docstrings with examples
- [ ] Optional dependency properly configured
- [ ] CONTINUITY.md updated (task #201 complete)

---

## Code Sketch

### Module Structure
```
src/dimtensor/cupy/
├── __init__.py      # Public API, availability check
└── dimarray.py      # DimArray implementation
```

### `__init__.py` Pattern
```python
"""CuPy DimArray: GPU arrays with dimensional awareness."""

def is_available() -> bool:
    """Check if CuPy and CUDA are available."""
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False

# Lazy import to avoid import errors without CuPy
def __getattr__(name: str):
    if name == "DimArray":
        if not is_available():
            raise ImportError(
                "CuPy is not available. Install with: pip install cupy-cuda12x"
            )
        from .dimarray import DimArray
        return DimArray
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["DimArray", "is_available"]
```

### DimArray Skeleton
```python
"""CuPy DimArray: GPU arrays with dimensional awareness and uncertainty."""

from __future__ import annotations

from typing import Any, Iterator, overload

import cupy as cp
from cupy import ndarray as CuPyArray

from ..core.dimensions import Dimension
from ..core.units import Unit, dimensionless
from ..errors import DimensionError, UnitConversionError


class DimArray:
    """A CuPy GPU array with attached physical units.

    Mirrors the NumPy DimArray API while keeping data on GPU.
    Supports uncertainty propagation for scientific computing.

    Examples:
        >>> import cupy as cp
        >>> from dimtensor.cupy import DimArray
        >>> from dimtensor import units
        >>>
        >>> # Create GPU arrays with units
        >>> v = DimArray(cp.array([1.0, 2.0, 3.0]), units.m / units.s)
        >>> t = DimArray(cp.array([0.5, 1.0, 1.5]), units.s)
        >>> d = v * t  # distance in meters, computed on GPU
        >>> print(d)
        [0.5 2.0 4.5] m
        >>>
        >>> # Transfer to CPU
        >>> d_numpy = d.numpy()
    """

    __slots__ = ("_data", "_unit", "_uncertainty")

    def __init__(
        self,
        data: CuPyArray | Any,
        unit: Unit | None = None,
        dtype: Any = None,
        copy: bool = False,
        uncertainty: CuPyArray | Any | None = None,
    ) -> None:
        """Create a CuPy DimArray.

        Args:
            data: Array data (CuPy array, NumPy array, list, or scalar).
            unit: Physical unit. If None, assumes dimensionless.
            dtype: CuPy dtype for the underlying array.
            copy: If True, always copy the data.
            uncertainty: Optional absolute uncertainty (same shape as data).
        """
        # Handle DimArray input
        if isinstance(data, DimArray):
            if copy:
                arr = cp.array(data._data, dtype=dtype, copy=True)
            else:
                arr = cp.asarray(data._data, dtype=dtype)
            unit = unit if unit is not None else data._unit
            if uncertainty is None and data._uncertainty is not None:
                uncertainty = data._uncertainty
        else:
            # Convert to CuPy array (handles NumPy arrays, lists, scalars)
            if copy:
                arr = cp.array(data, dtype=dtype, copy=True)
            else:
                arr = cp.asarray(data, dtype=dtype)

        self._data: CuPyArray = arr
        self._unit: Unit = unit if unit is not None else dimensionless

        # Handle uncertainty
        if uncertainty is not None:
            if copy:
                unc_arr = cp.array(uncertainty, dtype=dtype, copy=True)
            else:
                unc_arr = cp.asarray(uncertainty, dtype=dtype)
            if unc_arr.shape != arr.shape:
                raise ValueError(
                    f"Uncertainty shape {unc_arr.shape} must match data shape {arr.shape}"
                )
            self._uncertainty: CuPyArray | None = unc_arr
        else:
            self._uncertainty = None

    @classmethod
    def _from_data_and_unit(
        cls,
        data: CuPyArray,
        unit: Unit,
        uncertainty: CuPyArray | None = None,
    ) -> DimArray:
        """Internal constructor that doesn't copy data."""
        result = object.__new__(cls)
        result._data = data
        result._unit = unit
        result._uncertainty = uncertainty
        return result

    # ... (full implementation follows NumPy DimArray pattern)
```

---

## Performance Considerations

### Memory Bandwidth
- GPU operations are memory-bound for element-wise ops
- Uncertainty doubles memory traffic
- Consider fused kernels in future (v4.3.0)

### Kernel Launch Overhead
- Small arrays may be slower on GPU due to launch overhead
- Consider CPU fallback for arrays < 1000 elements

### Benchmarking Targets
- Overhead vs raw CuPy: <15% for arrays > 10K elements
- CPU-GPU transfer: explicit only (no hidden transfers)
- Memory usage: ~2x raw CuPy (with uncertainty)

---

## Notes / Log

**Design Decision: Uncertainty on GPU**
Uncertainty propagation formulas use sqrt, division, and multiplication - all well-suited for GPU parallelism. Keeping uncertainty on GPU avoids costly CPU-GPU transfers during computation.

**Design Decision: CuPy Version**
Target CuPy >= 12.0.0:
- Stable API
- Good NumPy 2.0 compatibility
- CUDA 11.x and 12.x support

**Design Decision: Optional Dependency**
CuPy is large (~500MB) and requires CUDA. Make it optional:
```toml
[project.optional-dependencies]
cupy = ["cupy-cuda12x>=12.0.0"]
```
Users install with: `pip install dimtensor[cupy]`

**Relationship to CUDA Kernels Plan**
The CUDA kernels plan (2026-01-09_cuda-kernels.md) targets PyTorch DimTensor optimization. This CuPy integration is separate - it provides a NumPy-compatible GPU backend. Future work could share optimized kernels between both.

---

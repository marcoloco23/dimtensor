# Plan: Memory Profiling Tools

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Add memory profiling tools to measure and report the overhead of unit tracking in dimtensor, enabling users to optimize memory usage and understand the cost of dimensional safety.

---

## Background

dimtensor adds metadata (Unit, Dimension, uncertainty arrays) to numerical arrays. For v3.6.0 "Production-ready speed", we need tools to:
1. Measure memory overhead per array
2. Identify optimization opportunities (shared units, lazy dimensions)
3. Compare memory usage to baseline numpy/torch
4. Track GPU memory for DimTensor
5. Help users make informed trade-offs

Current situation:
- No existing memory profiling utilities
- benchmarks.py only measures execution time, not memory
- Users can't easily quantify metadata overhead

---

## Approach

### Option A: Built-in sys.getsizeof + Manual Tracking
- Description: Use sys.getsizeof for basic size, manually traverse object graph
- Pros: No dependencies, lightweight, full control
- Cons: Doesn't account for shared objects, incomplete for complex structures

### Option B: tracemalloc Integration
- Description: Python's built-in memory profiler, track allocations over time
- Pros: Built-in, accurate, tracks allocations, no dependencies
- Cons: Runtime overhead, harder to get per-object stats

### Option C: Pympler asizeof + Custom Analysis
- Description: Use pympler.asizeof for deep size calculation, custom reporting
- Pros: Accurate deep size, handles shared objects, rich analysis
- Cons: External dependency (pympler), slower for large object graphs

### Decision: Hybrid Approach

Use **sys.getsizeof** with custom traversal for per-object stats (fast, no deps), plus **optional tracemalloc** for session-level profiling. GPU memory via torch.cuda if available.

Rationale:
- Keep core lightweight (no external deps)
- Provide both micro (per-array) and macro (session) views
- Handle object sharing explicitly (units/dimensions)
- Special-case GPU memory

---

## Implementation Steps

1. [ ] Create `src/dimtensor/profiling.py` module
2. [ ] Implement `calculate_metadata_size()` for DimArray/DimTensor
3. [ ] Implement `memory_overhead()` to compare vs raw array
4. [ ] Implement `memory_stats()` for structured per-array statistics
5. [ ] Implement `memory_report()` for human-readable output
6. [ ] Add GPU memory tracking for DimTensor (torch.cuda.memory_*)
7. [ ] Implement optional `MemoryProfiler` context manager with tracemalloc
8. [ ] Add shared object tracking (unit/dimension reuse analysis)
9. [ ] Create benchmark comparison vs numpy/torch baselines
10. [ ] Add optimization recommendations function
11. [ ] Write comprehensive tests
12. [ ] Add documentation and usage examples

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/profiling.py | NEW: Core memory profiling functions |
| src/dimtensor/__init__.py | Export profiling utilities |
| src/dimtensor/core/dimarray.py | Optional: Add .memory_stats() method |
| src/dimtensor/torch/dimtensor.py | Optional: Add .memory_stats() method |
| tests/test_profiling.py | NEW: Tests for profiling tools |
| docs/guide/performance.md | NEW or UPDATE: Memory profiling guide |
| examples/memory_profiling.py | NEW: Example usage |

---

## API Design

### Core Functions

```python
# Per-object memory statistics
def memory_stats(obj: DimArray | DimTensor) -> MemoryStats:
    """Get detailed memory statistics for a dimtensor object.

    Returns:
        MemoryStats with:
        - data_bytes: Size of numerical data
        - metadata_bytes: Size of unit/dimension metadata
        - uncertainty_bytes: Size of uncertainty array (if present)
        - total_bytes: Total memory footprint
        - overhead_ratio: metadata / data ratio
        - shared_metadata: Whether unit/dimension are shared
    """

# Calculate just metadata overhead
def metadata_overhead(obj: DimArray | DimTensor) -> int:
    """Return bytes of metadata overhead (unit + dimension + uncertainty)."""

# Human-readable report
def memory_report(
    obj: DimArray | DimTensor | list[DimArray | DimTensor],
    detailed: bool = True
) -> str:
    """Generate formatted memory report.

    Args:
        obj: Single array or list of arrays
        detailed: Include per-component breakdown

    Returns:
        Formatted string with memory usage and recommendations
    """

# Baseline comparison
def compare_to_baseline(obj: DimArray | DimTensor) -> ComparisonStats:
    """Compare memory usage to raw numpy/torch equivalent.

    Returns:
        ComparisonStats with:
        - baseline_bytes: Size of raw array
        - dimtensor_bytes: Size with metadata
        - overhead_bytes: Difference
        - overhead_percent: Percentage overhead
    """

# Optimization analysis
def analyze_shared_metadata(arrays: list[DimArray]) -> SharedMetadataReport:
    """Analyze unit/dimension sharing across arrays.

    Identifies opportunities for memory savings through object reuse.

    Returns:
        Report with:
        - unique_units: Number of unique Unit objects
        - unique_dimensions: Number of unique Dimension objects
        - sharing_potential: Bytes that could be saved
        - recommendations: List of optimization suggestions
    """

# GPU memory (PyTorch only)
def gpu_memory_stats(device: str | int = 0) -> GPUMemoryStats:
    """Get GPU memory statistics for DimTensor objects.

    Returns:
        GPUMemoryStats with:
        - allocated_bytes: Currently allocated GPU memory
        - reserved_bytes: Reserved by PyTorch
        - peak_bytes: Peak memory usage since last reset
        - dimtensor_estimate: Estimated memory used by DimTensor
    """
```

### Context Manager for Session Profiling

```python
class MemoryProfiler:
    """Context manager for tracking memory allocations.

    Uses tracemalloc to track all allocations during a session.

    Example:
        >>> with MemoryProfiler() as prof:
        ...     x = DimArray([1, 2, 3], units.m)
        ...     y = x * 2
        >>> print(prof.report())
    """

    def __enter__(self) -> MemoryProfiler: ...
    def __exit__(self, *args) -> None: ...
    def get_stats(self) -> ProfileStats: ...
    def report(self) -> str: ...
    def reset_peak(self) -> None: ...
```

### Data Classes

```python
@dataclass
class MemoryStats:
    """Memory statistics for a dimtensor object."""
    data_bytes: int
    metadata_bytes: int
    uncertainty_bytes: int
    total_bytes: int
    overhead_ratio: float
    shared_metadata: bool
    device: str  # 'cpu' or 'cuda:0', etc.

@dataclass
class ComparisonStats:
    """Comparison to baseline numpy/torch array."""
    baseline_bytes: int
    dimtensor_bytes: int
    overhead_bytes: int
    overhead_percent: float

@dataclass
class SharedMetadataReport:
    """Analysis of metadata sharing across arrays."""
    num_arrays: int
    unique_units: int
    unique_dimensions: int
    total_metadata_bytes: int
    shared_savings_bytes: int
    recommendations: list[str]

@dataclass
class GPUMemoryStats:
    """GPU memory statistics."""
    device: str
    allocated_bytes: int
    reserved_bytes: int
    peak_bytes: int
    dimtensor_estimate: int
```

---

## Memory Calculation Details

### DimArray Memory Components

```python
# 1. Data array (baseline)
data_size = array.data.nbytes

# 2. Unit object (typically shared)
#    - symbol: str (24 + len bytes on CPython)
#    - dimension: reference (8 bytes)
#    - scale: float (24 bytes)
#    Total: ~56 + len(symbol) bytes

# 3. Dimension object (typically shared)
#    - _exponents: Tuple of 7 Fraction objects
#    - Each Fraction: numerator (28 bytes) + denominator (28 bytes)
#    Total: ~200-300 bytes

# 4. Uncertainty array (optional, same size as data)
uncertainty_size = array.uncertainty.nbytes if array.has_uncertainty else 0

# 5. DimArray object overhead
#    - Python object header: 16 bytes
#    - __slots__: 3 references * 8 bytes = 24 bytes
#    Total: 40 bytes

total_overhead = unit_size + dimension_size + uncertainty_size + 40
```

### DimTensor Memory Components

Similar to DimArray but:
- Uses torch.Tensor instead of numpy.ndarray
- tensor.element_size() * tensor.nelement() for data size
- GPU memory tracked via torch.cuda.memory_allocated()
- No uncertainty tracking (yet)

---

## Testing Strategy

### Unit Tests

- [ ] Test memory_stats() on various array sizes
- [ ] Test with/without uncertainty
- [ ] Test GPU memory tracking (requires CUDA)
- [ ] Test shared vs non-shared units/dimensions
- [ ] Test compare_to_baseline() accuracy
- [ ] Test memory_report() formatting
- [ ] Test MemoryProfiler context manager
- [ ] Test analyze_shared_metadata()
- [ ] Verify calculations against manual sizeof

### Integration Tests

- [ ] Profile realistic physics simulation
- [ ] Track memory growth over many operations
- [ ] Verify GPU memory tracking matches torch stats
- [ ] Test with large arrays (>1GB)
- [ ] Test with many small arrays (>1000)

### Benchmarks

- [ ] Compare overhead across array sizes (10, 1K, 1M, 1B elements)
- [ ] Measure profiler overhead (should be <1%)
- [ ] Benchmark against raw numpy/torch
- [ ] GPU memory overhead vs raw torch.Tensor

---

## Risks / Edge Cases

### Risk 1: Shared Object Accounting
**Issue**: Units and Dimensions are shared across arrays via flyweight pattern. Naively summing sizes will overcount.

**Mitigation**:
- Track object IDs to detect sharing
- Report both "total if shared" and "total if duplicated"
- Provide sharing analysis function

### Risk 2: sys.getsizeof Limitations
**Issue**: sys.getsizeof doesn't traverse references, may miss nested objects.

**Mitigation**:
- Manually traverse known structure (Unit -> Dimension -> Fractions)
- Document limitations in docstrings
- Provide "deep size" option using object graph traversal

### Risk 3: GPU Memory Attribution
**Issue**: torch.cuda.memory_allocated() shows all GPU memory, not just DimTensor.

**Mitigation**:
- Clearly document that GPU stats are estimates
- Provide delta measurement (before/after snapshot)
- Recommend using torch.cuda.reset_peak_memory_stats()

### Risk 4: Uncertainty Array Impact
**Issue**: Uncertainty arrays double memory usage but are optional and not always used.

**Mitigation**:
- Clearly separate uncertainty from metadata in reports
- Show "with/without uncertainty" scenarios
- Recommend uncertainty=None when not needed

### Edge Case 1: Empty Arrays
**Handling**: Handle 0-element arrays gracefully, metadata overhead is 100%

### Edge Case 2: Scalar Values
**Handling**: DimArray always wraps in array, report appropriately

### Edge Case 3: Multi-GPU Systems
**Handling**: gpu_memory_stats() takes device parameter, report per-device

### Edge Case 4: NumPy View vs Copy
**Handling**: Views share data memory, account for this in reporting

---

## Optimization Opportunities to Surface

The profiling tools should identify and report:

1. **Unit/Dimension Reuse**:
   - "10 arrays share same unit 'm' - good!"
   - "50 dimension objects created with same exponents - consider caching"

2. **Unnecessary Uncertainty**:
   - "Uncertainty arrays add 2GB memory overhead"
   - "Consider uncertainty=None if not needed"

3. **Scale Factor Waste**:
   - "Using non-SI scale factors (km, mm) adds conversion overhead"
   - "Consider standardizing to SI base units"

4. **GPU Memory Pressure**:
   - "GPU memory 85% full, consider cpu offloading"
   - "Peak memory exceeded allocation by 500MB"

5. **Small Array Overhead**:
   - "Metadata is 90% of memory for arrays <100 elements"
   - "Consider batching small arrays"

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Tests pass (>95% coverage for profiling.py)
- [ ] Documentation with examples
- [ ] CONTINUITY.md updated with task status
- [ ] Memory profiling integrated into benchmarks.py
- [ ] Works on CPU and GPU (if CUDA available)
- [ ] No external dependencies beyond stdlib + numpy/torch

---

## Notes / Log

**Planning Phase** - Researched memory profiling approaches

Key insights:
- DimArray uses __slots__ which helps minimize overhead
- Fractions in Dimension use Python ints (variable size based on value)
- Unit sharing via flyweight is already implemented (units.py globals)
- Need to distinguish between "shallow" and "deep" memory accounting

Example overhead calculation for 1M element array:
- Data: 8MB (float64)
- Metadata: ~500 bytes (Unit + Dimension)
- Overhead: 0.006% (negligible for large arrays)

For 10 element array:
- Data: 80 bytes
- Metadata: ~500 bytes
- Overhead: 625% (significant!)

This shows why size-dependent profiling is important.

Memory sources identified:
1. DimArray: _data (numpy), _unit (Unit ref), _uncertainty (optional numpy)
2. Unit: symbol (str), dimension (Dimension ref), scale (float) - ~60-100 bytes
3. Dimension: 7 Fractions in tuple - ~200-300 bytes
4. DimTensor: _data (torch.Tensor), _unit (Unit ref) - no uncertainty yet

---

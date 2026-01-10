# Plan: Dask Integration for Distributed/Chunked DimArrays

**Date**: 2026-01-10
**Status**: PLANNING
**Author**: planner agent (task #202)

---

## Goal

Implement a DimArray class for Dask that enables distributed/chunked arrays with physical units, supporting lazy evaluation and parallel computation while maintaining dimensional safety.

---

## Background

### Why Dask Integration?

Dask provides:
- **Lazy evaluation**: Build computation graphs without executing, optimize before running
- **Chunked arrays**: Work with datasets larger than memory by processing in chunks
- **Parallel computation**: Distribute work across cores/machines
- **NumPy API compatibility**: Familiar interface for scientific Python users

Scientific computing often involves:
- Terabyte-scale simulation outputs (climate, CFD, particle physics)
- Parameter sweeps with millions of combinations
- Multi-node HPC cluster computations

Adding unit tracking to Dask arrays catches dimension errors before expensive distributed computations.

### Prior Art: Existing Patterns in dimtensor

| Module | Backend | Key Patterns |
|--------|---------|--------------|
| `core/dimarray.py` | NumPy | `_from_data_and_unit()`, `__slots__`, numpy ufunc protocol |
| `jax/dimarray.py` | JAX | Pytree registration, immutable design |
| `torch/dimtensor.py` | PyTorch | Autograd support, device handling |

All follow the same core pattern:
1. Wrap backend array in `_data` attribute
2. Store unit in `_unit` attribute
3. Internal constructor `_from_data_and_unit()` for efficiency
4. All arithmetic ops validate dimensions then delegate to backend

### Dask Collection Protocol

Dask collections implement these methods:
- `__dask_graph__()` - Returns task graph (dict or HighLevelGraph)
- `__dask_keys__()` - Returns output keys for this collection
- `__dask_scheduler__()` - Returns default scheduler
- `__dask_postcompute__()` - Returns (finalize_func, extra_args) for after compute
- `__dask_postpersist__()` - Returns (rebuild_func, extra_args) for after persist
- `__dask_tokenize__()` - Returns unique hash input for caching

Key insight: Unit information is **static metadata** - it doesn't change during computation and should NOT be part of the task graph.

---

## Approach

### Option A: Wrapper around dask.array.Array

Wrap a `dask.array.Array` similar to how `DimArray` wraps `numpy.ndarray`.

```python
class DaskDimArray:
    __slots__ = ("_data", "_unit")

    def __init__(self, data, unit=None, chunks="auto"):
        if isinstance(data, da.Array):
            self._data = data
        else:
            self._data = da.from_array(data, chunks=chunks)
        self._unit = unit or dimensionless
```

**Pros:**
- Simple implementation following existing patterns
- Leverages all dask.array functionality (rechunk, persist, etc.)
- Unit metadata stays outside task graph (efficient)
- Consistent with JAX/PyTorch approaches

**Cons:**
- May need custom handling for some dask.array operations
- Operations that change dtype/shape need careful unit handling

### Option B: Full Dask Collection Protocol Implementation

Implement `DaskCollection` protocol from scratch, managing our own task graph.

```python
class DaskDimArray(DaskMethodsMixin):
    def __dask_graph__(self):
        # Custom graph including unit checks
        ...
```

**Pros:**
- Complete control over task graph
- Could embed unit validation in graph

**Cons:**
- Much more complex
- Reinvents dask.array functionality
- Unit checks in graph add overhead to every worker
- Harder to maintain

### Option C: Hybrid with map_blocks

Use `dask.array.Array` wrapper but use `map_blocks` for custom operations.

```python
class DaskDimArray:
    def custom_op(self, func, *args):
        result = self._data.map_blocks(func, *args, dtype=self._data.dtype)
        return DaskDimArray._from_data_and_unit(result, result_unit)
```

**Pros:**
- Simple wrapper with extensibility
- `map_blocks` handles chunking automatically
- Can verify units before graph execution

**Cons:**
- `map_blocks` overhead for simple operations
- Some operations don't map naturally to blocks

### Decision: Option A (Wrapper around dask.array.Array)

Rationale:
1. **Consistency**: Matches existing JAX/PyTorch patterns exactly
2. **Simplicity**: Dask handles all parallelism, we handle units
3. **Efficiency**: Units are metadata, not computed - no graph overhead
4. **Maintainability**: Leverages tested dask.array code
5. **Flexibility**: Can add map_blocks for custom operations later

---

## Implementation Steps

### Phase 1: Core DaskDimArray Class

1. [ ] Create `src/dimtensor/dask/__init__.py`
2. [ ] Create `src/dimtensor/dask/dimarray.py` with basic structure:
   ```python
   class DaskDimArray:
       __slots__ = ("_data", "_unit")

       def __init__(self, data, unit=None, chunks="auto")

       @classmethod
       def _from_data_and_unit(cls, data, unit) -> DaskDimArray

       # Properties: data, unit, dimension, shape, ndim, size, dtype, chunks
       # is_dimensionless, chunksize
   ```

### Phase 2: Array Creation Functions

3. [ ] Implement constructors:
   ```python
   @classmethod
   def from_array(cls, data, unit, chunks="auto")

   @classmethod
   def from_delayed(cls, delayed_arrays, unit, shape, dtype)

   @classmethod
   def zeros(cls, shape, unit, chunks="auto", dtype=float)

   @classmethod
   def ones(cls, shape, unit, chunks="auto", dtype=float)

   @classmethod
   def arange(cls, start, stop, step, unit, chunks="auto")
   ```

### Phase 3: Lazy Evaluation Support

4. [ ] Implement Dask collection protocol:
   ```python
   def __dask_graph__(self) -> HighLevelGraph
   def __dask_keys__(self) -> list
   __dask_scheduler__ = staticmethod(dask.threaded.get)
   def __dask_postcompute__(self) -> tuple
   def __dask_postpersist__(self) -> tuple
   def __dask_tokenize__(self) -> tuple
   ```

5. [ ] Implement compute/persist methods:
   ```python
   def compute(self, **kwargs) -> DimArray  # Returns NumPy DimArray
   def persist(self, **kwargs) -> DaskDimArray  # Returns in-memory DaskDimArray
   def visualize(self, **kwargs)  # Task graph visualization
   ```

### Phase 4: Arithmetic Operations

6. [ ] Implement arithmetic with dimension checking:
   ```python
   def __add__(self, other)  # Requires same dimension
   def __sub__(self, other)  # Requires same dimension
   def __mul__(self, other)  # Dimensions multiply
   def __truediv__(self, other)  # Dimensions divide
   def __pow__(self, power)  # Dimension raised to power
   def __neg__(self)
   def __abs__(self)
   def sqrt(self)
   ```

7. [ ] Implement comparison ops:
   ```python
   def __eq__(self, other)
   def __lt__(self, other)
   def __le__(self, other)
   def __gt__(self, other)
   def __ge__(self, other)
   ```

### Phase 5: Reduction Operations

8. [ ] Implement reductions (lazy, return DaskDimArray):
   ```python
   def sum(self, axis=None, keepdims=False)
   def mean(self, axis=None, keepdims=False)
   def std(self, axis=None, keepdims=False)
   def var(self, axis=None, keepdims=False)  # Returns squared unit
   def min(self, axis=None, keepdims=False)
   def max(self, axis=None, keepdims=False)
   ```

### Phase 6: Reshaping & Indexing

9. [ ] Implement reshaping:
   ```python
   def reshape(self, shape)
   def transpose(self, axes=None)
   def flatten(self)
   def rechunk(self, chunks)
   ```

10. [ ] Implement indexing:
    ```python
    def __getitem__(self, key)  # Preserves unit
    def __len__(self)
    ```

### Phase 7: Unit Conversion

11. [ ] Implement unit operations:
    ```python
    def to(self, unit) -> DaskDimArray  # Convert units
    def to_base_units(self) -> DaskDimArray
    def magnitude(self) -> da.Array  # Strip units
    ```

### Phase 8: Linear Algebra

12. [ ] Implement basic linalg:
    ```python
    def dot(self, other)  # Dimensions multiply
    def matmul(self, other)  # Dimensions multiply
    def __matmul__(self, other)
    def norm(self, ord=None, axis=None)
    ```

### Phase 9: Integration with Core DimArray

13. [ ] Add conversion methods:
    ```python
    def to_dimarray(self) -> DimArray  # Compute and return NumPy DimArray

    @classmethod
    def from_dimarray(cls, dimarray, chunks="auto") -> DaskDimArray
    ```

14. [ ] Add interoperability with other backends:
    ```python
    def to_jax(self)  # Returns JAX DimArray
    def to_torch(self)  # Returns DimTensor
    ```

### Phase 10: Testing

15. [ ] Create `tests/test_dask_dimarray.py`:
    - Basic creation and properties
    - Lazy evaluation (operations don't compute immediately)
    - Compute and persist
    - Arithmetic with dimension checking
    - Reductions
    - Reshaping and indexing
    - Unit conversion
    - Integration with core DimArray
    - Skip if dask not installed

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/dask/__init__.py` | NEW: Module exports |
| `src/dimtensor/dask/dimarray.py` | NEW: DaskDimArray class (~500-600 lines) |
| `src/dimtensor/__init__.py` | Add dask to optional imports |
| `pyproject.toml` | Add dask optional dependency |
| `tests/test_dask_dimarray.py` | NEW: Test suite (~300 lines) |

---

## API Design

### Basic Usage

```python
import dask.array as da
from dimtensor.dask import DaskDimArray
from dimtensor import units

# Create from NumPy array with chunks
data = np.random.randn(10000, 10000)
velocity = DaskDimArray(data, unit=units.m/units.s, chunks=(1000, 1000))

# Lazy operations - no computation yet
speed = (velocity**2).sum(axis=1).sqrt()

# Trigger computation
result = speed.compute()  # Returns numpy DimArray
print(result)  # [127.4, 131.2, ...] m/s

# Persist in distributed memory
speed_persisted = speed.persist()

# Dimension checking works lazily
time = DaskDimArray(np.ones((10000, 10000)), unit=units.s, chunks=(1000, 1000))
distance = velocity * time  # OK, units multiply
# velocity + time  # DimensionError raised BEFORE computation
```

### Distributed Example

```python
from dask.distributed import Client

# Connect to cluster
client = Client("scheduler:8786")

# Large simulation output
temperature = DaskDimArray.from_zarr(
    "s3://bucket/simulation/temperature.zarr",
    unit=units.K
)
pressure = DaskDimArray.from_zarr(
    "s3://bucket/simulation/pressure.zarr",
    unit=units.Pa
)

# Ideal gas law: PV = nRT
# Solve for n/V = P/(RT)
R = 8.314 * units.J / (units.mol * units.K)
density = pressure / (R * temperature)  # mol/m^3

# Compute on cluster
result = density.compute()  # Distributed computation
```

---

## Testing Strategy

- [ ] Unit tests for each operation class (20-30 tests):
  - Creation and properties
  - Arithmetic ops with dimension validation
  - Reduction ops
  - Reshaping and indexing

- [ ] Lazy evaluation tests (5-10 tests):
  - Operations don't trigger computation
  - compute() returns numpy DimArray
  - persist() returns DaskDimArray

- [ ] Error handling tests (5-10 tests):
  - DimensionError raised before computation for mismatched units
  - UnitConversionError for incompatible conversions

- [ ] Integration tests (5 tests):
  - Round-trip: DimArray -> DaskDimArray -> DimArray
  - Large array computation (verify chunking works)

- [ ] Skip tests if dask not installed:
  ```python
  pytest.importorskip("dask")
  pytest.importorskip("dask.array")
  ```

---

## Risks / Edge Cases

### Risk 1: Unit Validation Timing
**Issue**: When should dimension errors be raised?
**Decision**: Raise immediately when building graph, not during compute()
**Rationale**: Fail fast - don't waste cluster resources on invalid graphs

### Risk 2: Chunks with Different Units
**Issue**: What if user tries to concatenate arrays with different units?
**Handling**: Validate units match, convert if compatible

### Risk 3: Scalar Operations
**Issue**: Dask handles scalars differently than NumPy
**Handling**: Wrap scalar results appropriately, test edge cases

### Risk 4: Memory for Large Unit Metadata
**Issue**: Unit stored per-array, not per-chunk (correct behavior)
**Non-issue**: Unit is small constant overhead, chunks share unit reference

### Edge Cases:
- Empty arrays
- 0-d arrays (scalars)
- Single chunk (should still work)
- Very small chunks (performance warning?)
- Mixed dask/numpy operations

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] 40+ tests pass (including skip markers for missing dask)
- [ ] mypy clean
- [ ] Documentation in docstrings
- [ ] CONTINUITY.md updated with completion
- [ ] Example usage in docstrings/comments

---

## Notes / Log

**2026-01-10** - Initial plan created based on research:
- Reviewed existing patterns in dimarray.py, jax/dimarray.py, torch/dimtensor.py
- Studied Dask collection protocol and custom collections
- Decided on wrapper approach for consistency and simplicity
- Key insight: Units are metadata, stay outside task graph

---

## Appendix: Dask Array Key Concepts

### Task Graph
```python
# Example task graph for x + y
{
    ('add-xyz', 0, 0): (add, ('x', 0, 0), ('y', 0, 0)),
    ('add-xyz', 0, 1): (add, ('x', 0, 1), ('y', 0, 1)),
    ...
}
```

### Chunk Specification
```python
# 10000x10000 array with 1000x1000 chunks
x = da.random.random((10000, 10000), chunks=(1000, 1000))
x.chunks  # ((1000, 1000, ..., 1000), (1000, 1000, ..., 1000))
```

### Lazy Evaluation Flow
```
Build graph -> (no compute) -> compute() -> Execute tasks -> Return result
                                     ^
                                     |
                     persist() -> Execute -> Keep in memory
```

### DaskMethodsMixin
Provides `compute()`, `persist()`, `visualize()` when you implement:
- `__dask_graph__()`
- `__dask_keys__()`
- `__dask_scheduler__`
- `__dask_postcompute__()`
- `__dask_postpersist__()`

---

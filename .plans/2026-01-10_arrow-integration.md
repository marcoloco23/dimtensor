# Plan: Apache Arrow Integration for Zero-Copy Unit Arrays

**Date**: 2026-01-10
**Status**: PLANNING
**Author**: planner agent
**Task**: #205

---

## Goal

Integrate Apache Arrow's memory model with dimtensor to enable zero-copy operations on unit-aware arrays, including efficient IPC (inter-process communication) with preserved unit metadata.

---

## Background

dimtensor currently uses:
- NumPy arrays as the backing store for `DimArray` in `core/dimarray.py`
- PyArrow for Parquet serialization in `io/parquet.py` (copies data via `.flatten()` and `.to_numpy()`)
- Polars integration in `io/polars.py` (also copies data)

Apache Arrow provides:
- **Zero-copy memory buffers**: `pa.Buffer` wraps memory without copying
- **Extension types**: Custom types with metadata that persist through IPC
- **Memory-mapped I/O**: Read large datasets without loading into RAM
- **IPC protocols**: Stream and file formats for cross-process data sharing

The current Parquet implementation flattens arrays and reconstructs them, losing zero-copy benefits. A proper Arrow integration would allow:
1. Arrow arrays to be the backing store for DimArray (zero-copy from Arrow sources)
2. Unit metadata to persist through Arrow's IPC mechanisms
3. Efficient data sharing between processes without serialization overhead

---

## Research Summary

### Apache Arrow Memory Model

From [Arrow Memory Documentation](https://arrow.apache.org/docs/python/memory.html):
- `pa.Buffer` provides standard interface: data pointer + length
- Zero-copy slicing preserves memory lifetime
- `MemoryMappedFile` enables disk access without copying to RAM
- `NativeFile` allows C++ code to bypass Python GIL

### Extension Types

From [Extending PyArrow](https://arrow.apache.org/docs/python/extending_types.html):
- Subclass `pa.ExtensionType` to create custom types
- Implement `__arrow_ext_serialize__()` and `__arrow_ext_deserialize__()` for IPC
- Register types before deserializing: `pa.register_extension_type()`
- Extension arrays persist through IPC with full metadata

### IPC Capabilities

From [Arrow IPC](https://arrow.apache.org/docs/python/ipc.html):
- Streaming format: arbitrary-length sequence of record batches
- File format: fixed batches with random access
- Zero-copy reads from memory-mapped sources
- Schema metadata preserved through serialization

---

## Approach

### Option A: Arrow-Backed DimArray (New Class)

Create a new `ArrowDimArray` class that uses Arrow arrays as backing store.

**Pros:**
- Zero-copy from Arrow sources
- Clean separation of concerns
- Can coexist with existing NumPy-backed DimArray

**Cons:**
- Code duplication (arithmetic, properties, etc.)
- Users must choose between two array types
- Harder to maintain

### Option B: Unified DimArray with Pluggable Backend

Modify `DimArray` to accept either NumPy or Arrow arrays.

**Pros:**
- Single API for users
- Automatic backend selection

**Cons:**
- Increased complexity in DimArray
- Risk of breaking existing code
- Performance overhead from backend detection

### Option C: Arrow Extension Type + Conversion Functions (Recommended)

Create a PyArrow extension type `DimArrayType` that stores unit metadata, plus functions to convert between DimArray and Arrow arrays.

**Pros:**
- Leverages Arrow's native extension system
- Unit metadata persists through IPC without custom code
- Clean integration with existing dimtensor patterns
- Matches patterns in parquet.py, polars.py
- Can later add zero-copy backing store (Option A) if needed

**Cons:**
- Initial implementation still requires conversion
- Zero-copy only for Arrow-to-Arrow operations initially

### Decision: Option C

Start with Arrow extension type and conversion functions. This:
1. Follows existing io/ module patterns
2. Provides immediate value (IPC with units)
3. Sets foundation for future zero-copy backing store
4. Lower risk than modifying core DimArray

---

## Implementation Steps

### Phase 1: Arrow Extension Type (Core)

1. [ ] Create `src/dimtensor/io/arrow.py` module
2. [ ] Implement `DimArrayType(pa.ExtensionType)` for scalar unit metadata
3. [ ] Implement `DimArrayColumnType(pa.ExtensionType)` for array unit metadata
4. [ ] Add serialization/deserialization for unit metadata
5. [ ] Register extension types on module import

### Phase 2: Conversion Functions

6. [ ] Implement `to_arrow_array(arr: DimArray) -> pa.ExtensionArray`
7. [ ] Implement `from_arrow_array(arr: pa.ExtensionArray) -> DimArray`
8. [ ] Implement `to_arrow_table(arrays: dict[str, DimArray]) -> pa.Table`
9. [ ] Implement `from_arrow_table(table: pa.Table) -> dict[str, DimArray]`

### Phase 3: IPC Functions

10. [ ] Implement `write_ipc_stream(arrays, sink)` for streaming IPC
11. [ ] Implement `read_ipc_stream(source) -> dict[str, DimArray]`
12. [ ] Implement `write_ipc_file(arrays, path)` for random-access IPC
13. [ ] Implement `read_ipc_file(path) -> dict[str, DimArray]`

### Phase 4: Record Batch Support

14. [ ] Implement `to_record_batch(arrays: dict[str, DimArray]) -> pa.RecordBatch`
15. [ ] Implement `from_record_batch(batch: pa.RecordBatch) -> dict[str, DimArray]`
16. [ ] Add batched reading/writing for large datasets

### Phase 5: Zero-Copy Enhancements

17. [ ] Implement `from_arrow_buffer(buffer, dtype, unit) -> DimArray` (zero-copy)
18. [ ] Add memory-mapped file support via Arrow
19. [ ] Create `ArrowDimArray` class for native Arrow backing (optional)

### Phase 6: Integration

20. [ ] Update `io/__init__.py` exports
21. [ ] Update existing parquet.py to use new Arrow primitives
22. [ ] Add Arrow support to Polars integration
23. [ ] Document Arrow integration

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/dimtensor/io/arrow.py` | Arrow extension types and conversion functions |
| `tests/test_arrow.py` | Test suite for Arrow integration |
| `docs/user-guide/arrow.md` | Documentation for Arrow features |

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/io/__init__.py` | Add Arrow exports |
| `src/dimtensor/io/parquet.py` | Refactor to use Arrow primitives |
| `pyproject.toml` | Ensure pyarrow is in optional deps |

---

## Design Details

### DimArrayType Extension Type

```python
class DimArrayType(pa.ExtensionType):
    """Arrow extension type for unit-aware arrays.

    Stores unit metadata (symbol, dimension, scale) as serialized bytes.
    The storage type is the underlying numeric Arrow type.
    """

    def __init__(self, storage_type: pa.DataType, unit: Unit):
        self._unit = unit
        super().__init__(storage_type, "dimtensor.dimarray")

    def __arrow_ext_serialize__(self) -> bytes:
        """Serialize unit metadata for IPC."""
        metadata = {
            "symbol": self._unit.symbol,
            "scale": self._unit.scale,
            "dim_length": float(self._unit.dimension.length),
            "dim_mass": float(self._unit.dimension.mass),
            "dim_time": float(self._unit.dimension.time),
            "dim_current": float(self._unit.dimension.current),
            "dim_temperature": float(self._unit.dimension.temperature),
            "dim_amount": float(self._unit.dimension.amount),
            "dim_luminosity": float(self._unit.dimension.luminosity),
        }
        return json.dumps(metadata).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        """Deserialize unit metadata from IPC."""
        metadata = json.loads(serialized.decode())
        dimension = Dimension(
            length=Fraction(metadata["dim_length"]).limit_denominator(),
            mass=Fraction(metadata["dim_mass"]).limit_denominator(),
            time=Fraction(metadata["dim_time"]).limit_denominator(),
            current=Fraction(metadata["dim_current"]).limit_denominator(),
            temperature=Fraction(metadata["dim_temperature"]).limit_denominator(),
            amount=Fraction(metadata["dim_amount"]).limit_denominator(),
            luminosity=Fraction(metadata["dim_luminosity"]).limit_denominator(),
        )
        unit = Unit(metadata["symbol"], dimension, metadata["scale"])
        return cls(storage_type, unit)

    @property
    def unit(self) -> Unit:
        return self._unit
```

### Zero-Copy Conversion Pattern

```python
def from_arrow_array(arr: pa.Array, unit: Unit | None = None) -> DimArray:
    """Convert Arrow array to DimArray.

    If the array is a DimArrayType extension array, extracts the unit.
    Otherwise, uses the provided unit or dimensionless.

    Uses zero-copy when possible via Arrow's buffer protocol.
    """
    if isinstance(arr.type, DimArrayType):
        unit = arr.type.unit
        storage = arr.cast(arr.type.storage_type)
    else:
        storage = arr
        unit = unit or dimensionless

    # Zero-copy via numpy view of Arrow buffer
    if pa.types.is_floating(storage.type) or pa.types.is_integer(storage.type):
        # For primitive types, zero-copy is possible
        np_arr = storage.to_numpy(zero_copy_only=True)
        return DimArray._from_data_and_unit(np_arr, unit)
    else:
        # Fall back to copy for complex types
        np_arr = storage.to_numpy()
        return DimArray._from_data_and_unit(np_arr, unit)
```

### IPC Usage Example

```python
from dimtensor.io.arrow import write_ipc_stream, read_ipc_stream

# Write DimArrays to IPC stream
distances = DimArray([1.0, 2.0, 3.0], units.m)
times = DimArray([0.5, 1.0, 1.5], units.s)

with open("data.arrow", "wb") as f:
    write_ipc_stream({"distance": distances, "time": times}, f)

# Read in another process - units preserved!
with open("data.arrow", "rb") as f:
    data = read_ipc_stream(f)
    print(data["distance"].unit)  # m
```

### Memory-Mapped Reading

```python
from dimtensor.io.arrow import read_ipc_file_mmap

# Zero-copy read from memory-mapped file
data = read_ipc_file_mmap("large_dataset.arrow")
# Data is accessed directly from disk, not copied to RAM
```

---

## Testing Strategy

### Unit Tests (`tests/test_arrow.py`)

- [ ] `test_extension_type_serialize_deserialize`: Round-trip unit metadata
- [ ] `test_to_from_arrow_array_basic`: Convert DimArray <-> Arrow Array
- [ ] `test_to_from_arrow_array_with_units`: Verify units preserved
- [ ] `test_to_from_arrow_table`: Multiple arrays in table
- [ ] `test_record_batch_roundtrip`: Record batch conversion
- [ ] `test_ipc_stream_roundtrip`: Stream IPC preserves units
- [ ] `test_ipc_file_roundtrip`: File IPC preserves units
- [ ] `test_zero_copy_from_arrow`: Verify no copy on conversion
- [ ] `test_extension_type_registration`: Type recognized after registration
- [ ] `test_multidimensional_arrays`: Shape preservation
- [ ] `test_with_uncertainty`: Uncertainty column handling
- [ ] `test_derived_units`: Complex unit expressions

### Integration Tests

- [ ] `test_arrow_parquet_interop`: Arrow arrays to Parquet and back
- [ ] `test_cross_process_ipc`: Send units via IPC to subprocess
- [ ] `test_memory_mapped_large_file`: Large dataset memory efficiency

### Performance Benchmarks

- [ ] Compare copy vs zero-copy array creation
- [ ] IPC throughput with unit metadata
- [ ] Memory usage for large datasets

---

## Risks and Mitigations

### Risk 1: Extension Type Registration Conflicts
**Risk**: Other libraries might register conflicting extension types.
**Mitigation**: Use namespaced type name `dimtensor.dimarray`. Check if already registered before registering.

### Risk 2: Zero-Copy Limitations
**Risk**: Zero-copy may not work for all dtypes or when data isn't contiguous.
**Mitigation**: Implement `zero_copy_only=False` fallback. Document when zero-copy works.

### Risk 3: Arrow Version Compatibility
**Risk**: API changes between PyArrow versions.
**Mitigation**: Test against multiple PyArrow versions. Document minimum version.

### Risk 4: Uncertainty Handling Complexity
**Risk**: Storing uncertainty alongside data complicates Arrow schema.
**Mitigation**: Store as separate column with naming convention `{name}_uncertainty`. Handle in to/from functions.

### Edge Cases

- **Masked arrays**: Arrow supports null/mask; DimArray doesn't currently
- **String/object columns**: Not applicable to DimArray (numeric only)
- **Complex numbers**: Test Arrow support for complex128
- **Empty arrays**: Ensure unit metadata preserved for zero-length arrays

---

## Definition of Done

- [ ] `DimArrayType` extension type implemented and tested
- [ ] Conversion functions: `to_arrow_array`, `from_arrow_array`, `to_arrow_table`, `from_arrow_table`
- [ ] IPC functions: `write_ipc_stream`, `read_ipc_stream`, `write_ipc_file`, `read_ipc_file`
- [ ] Record batch functions: `to_record_batch`, `from_record_batch`
- [ ] Zero-copy conversion where possible
- [ ] All tests pass
- [ ] Documentation written
- [ ] `io/__init__.py` updated with exports
- [ ] CONTINUITY.md updated with completion status

---

## API Summary

### New Functions in `dimtensor.io.arrow`

```python
# Extension Type
class DimArrayType(pa.ExtensionType): ...

# Array Conversion (zero-copy where possible)
def to_arrow_array(arr: DimArray) -> pa.ExtensionArray
def from_arrow_array(arr: pa.Array, unit: Unit | None = None) -> DimArray

# Table Conversion
def to_arrow_table(arrays: dict[str, DimArray]) -> pa.Table
def from_arrow_table(table: pa.Table) -> dict[str, DimArray]

# Record Batch
def to_record_batch(arrays: dict[str, DimArray]) -> pa.RecordBatch
def from_record_batch(batch: pa.RecordBatch) -> dict[str, DimArray]

# IPC Streaming
def write_ipc_stream(arrays: dict[str, DimArray], sink: BinaryIO | str)
def read_ipc_stream(source: BinaryIO | str) -> dict[str, DimArray]

# IPC File (random access)
def write_ipc_file(arrays: dict[str, DimArray], path: str | Path)
def read_ipc_file(path: str | Path) -> dict[str, DimArray]

# Memory-mapped (zero-copy large files)
def read_ipc_file_mmap(path: str | Path) -> dict[str, DimArray]
```

---

## References

- [Apache Arrow Memory and IO](https://arrow.apache.org/docs/python/memory.html)
- [Extending PyArrow](https://arrow.apache.org/docs/python/extending_types.html)
- [Arrow IPC](https://arrow.apache.org/docs/python/ipc.html)
- [Arrow Data Types](https://arrow.apache.org/docs/python/data.html)
- Existing dimtensor: `io/parquet.py`, `io/polars.py`, `jax/dimarray.py`

---

## Notes / Log

**2026-01-10** - Initial plan created. Researched Arrow memory model, extension types, and IPC. Decided on Option C (extension type + conversion functions) as the approach. This balances immediate utility with clean integration patterns.

---

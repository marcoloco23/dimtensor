"""Apache Arrow integration for DimArray.

Provides zero-copy conversion between DimArrays and Arrow arrays,
with unit metadata preserved through Arrow's extension type system.

This enables efficient IPC (inter-process communication) with preserved
unit metadata, and can serve as a foundation for zero-copy operations
on unit-aware arrays.

Requires pyarrow.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

import numpy as np

from ..core.dimarray import DimArray
from ..core.dimensions import Dimension
from ..core.units import Unit, dimensionless


# Module-level cache for the extension type class
_DimArrayType = None
_EXTENSION_TYPE_REGISTERED = False


def _ensure_pyarrow():
    """Import and return pyarrow, raising ImportError if not available."""
    try:
        import pyarrow as pa
        return pa
    except ImportError:
        raise ImportError(
            "pyarrow is required for Arrow support. "
            "Install with: pip install pyarrow"
        )


def _get_extension_type_class():
    """Get the DimArrayType class, creating it once and caching."""
    global _DimArrayType

    if _DimArrayType is not None:
        return _DimArrayType

    pa = _ensure_pyarrow()

    class DimArrayType(pa.ExtensionType):
        """Arrow extension type for unit-aware arrays.

        Stores unit metadata (symbol, dimension, scale) as serialized JSON bytes.
        The storage type is the underlying numeric Arrow type.
        """

        def __init__(self, storage_type, unit: Unit):
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
            """The unit associated with this type."""
            return self._unit

        def __eq__(self, other):
            if not isinstance(other, DimArrayType):
                return False
            return (
                self.storage_type == other.storage_type
                and self._unit.symbol == other._unit.symbol
                and self._unit.scale == other._unit.scale
                and self._unit.dimension == other._unit.dimension
            )

        def __hash__(self):
            return hash((self.storage_type, self._unit.symbol, self._unit.scale))

    _DimArrayType = DimArrayType
    return DimArrayType


def _register_extension_type():
    """Register the DimArrayType extension type with PyArrow."""
    global _EXTENSION_TYPE_REGISTERED
    if _EXTENSION_TYPE_REGISTERED:
        return

    pa = _ensure_pyarrow()
    DimArrayType = _get_extension_type_class()

    try:
        # Create a dummy instance to register
        dummy_type = DimArrayType(pa.float64(), dimensionless)
        pa.register_extension_type(dummy_type)
        _EXTENSION_TYPE_REGISTERED = True
    except pa.ArrowKeyError:
        # Already registered (can happen if module is reimported)
        _EXTENSION_TYPE_REGISTERED = True


def _numpy_dtype_to_arrow_type(dtype: np.dtype):
    """Convert numpy dtype to pyarrow type."""
    pa = _ensure_pyarrow()

    dtype_map = {
        np.dtype("float16"): pa.float16(),
        np.dtype("float32"): pa.float32(),
        np.dtype("float64"): pa.float64(),
        np.dtype("int8"): pa.int8(),
        np.dtype("int16"): pa.int16(),
        np.dtype("int32"): pa.int32(),
        np.dtype("int64"): pa.int64(),
        np.dtype("uint8"): pa.uint8(),
        np.dtype("uint16"): pa.uint16(),
        np.dtype("uint32"): pa.uint32(),
        np.dtype("uint64"): pa.uint64(),
        np.dtype("bool"): pa.bool_(),
    }

    # Handle complex types
    if dtype == np.dtype("complex64"):
        # Arrow doesn't have native complex support, use struct
        return pa.struct([("real", pa.float32()), ("imag", pa.float32())])
    elif dtype == np.dtype("complex128"):
        return pa.struct([("real", pa.float64()), ("imag", pa.float64())])

    return dtype_map.get(dtype, pa.float64())


def _is_dimarray_extension_type(arrow_type) -> bool:
    """Check if an Arrow type is our DimArrayType extension by name."""
    pa = _ensure_pyarrow()

    if not isinstance(arrow_type, pa.ExtensionType):
        return False

    # Check by extension name since class identity may differ after IPC
    return arrow_type.extension_name == "dimtensor.dimarray"


def _extract_unit_from_extension_type(arrow_type) -> Unit | None:
    """Extract unit from a DimArrayType extension type by deserializing metadata."""
    pa = _ensure_pyarrow()

    if not _is_dimarray_extension_type(arrow_type):
        return None

    # The type has a unit attribute if it's our type
    if hasattr(arrow_type, "unit"):
        return arrow_type.unit

    # Fallback: deserialize from serialized form
    try:
        serialized = arrow_type.__arrow_ext_serialize__()
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
        return Unit(metadata["symbol"], dimension, metadata["scale"])
    except Exception:
        return None


# =============================================================================
# Array Conversion Functions
# =============================================================================


def to_arrow_array(arr: DimArray):
    """Convert a DimArray to an Arrow ExtensionArray.

    The resulting array uses the DimArrayType extension type which
    preserves unit metadata through Arrow's IPC mechanisms.

    Args:
        arr: DimArray to convert.

    Returns:
        PyArrow ExtensionArray with unit metadata.

    Raises:
        ImportError: If pyarrow is not installed.

    Example:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.io.arrow import to_arrow_array
        >>> arr = DimArray([1.0, 2.0, 3.0], units.m)
        >>> pa_arr = to_arrow_array(arr)
        >>> pa_arr.type.unit.symbol
        'm'
    """
    pa = _ensure_pyarrow()
    _register_extension_type()
    DimArrayType = _get_extension_type_class()

    # Flatten multidimensional arrays for Arrow storage
    flat_data = arr._data.flatten()

    # Create storage array
    storage_type = _numpy_dtype_to_arrow_type(arr.dtype)

    # Handle complex numbers specially
    if np.iscomplexobj(flat_data):
        # Convert to struct array
        real_part = flat_data.real
        imag_part = flat_data.imag
        storage = pa.StructArray.from_arrays(
            [pa.array(real_part), pa.array(imag_part)],
            names=["real", "imag"]
        )
    else:
        storage = pa.array(flat_data, type=storage_type)

    # Create extension type and array
    ext_type = DimArrayType(storage_type, arr.unit)
    ext_array = pa.ExtensionArray.from_storage(ext_type, storage)

    return ext_array


def from_arrow_array(
    arr,
    unit: Unit | None = None,
    shape: tuple[int, ...] | None = None,
) -> DimArray:
    """Convert an Arrow array to a DimArray.

    If the array is a DimArrayType extension array, extracts the unit
    from the type metadata. Otherwise, uses the provided unit or dimensionless.

    Attempts zero-copy conversion when possible via Arrow's buffer protocol.

    Args:
        arr: PyArrow Array (can be ExtensionArray with DimArrayType).
        unit: Unit to use if arr is not a DimArrayType. Ignored if arr has unit.
        shape: Shape to reshape the resulting array. If None, returns 1D.

    Returns:
        DimArray with appropriate unit.

    Raises:
        ImportError: If pyarrow is not installed.

    Example:
        >>> from dimtensor.io.arrow import to_arrow_array, from_arrow_array
        >>> pa_arr = to_arrow_array(DimArray([1.0, 2.0], units.m))
        >>> restored = from_arrow_array(pa_arr)
        >>> restored.unit.symbol
        'm'
    """
    pa = _ensure_pyarrow()
    _register_extension_type()

    # Extract unit from extension type if present
    extracted_unit = _extract_unit_from_extension_type(arr.type)
    if extracted_unit is not None:
        unit = extracted_unit
        # Get storage array for extension arrays
        storage = arr.storage if hasattr(arr, "storage") else arr
    else:
        storage = arr
        unit = unit or dimensionless

    # Convert to numpy
    if pa.types.is_struct(storage.type):
        # Handle complex numbers
        real_arr = storage.field("real").to_numpy()
        imag_arr = storage.field("imag").to_numpy()
        np_arr = real_arr + 1j * imag_arr
    elif pa.types.is_primitive(storage.type):
        # Try zero-copy first
        try:
            np_arr = storage.to_numpy(zero_copy_only=True)
        except pa.ArrowInvalid:
            # Fall back to copy if zero-copy not possible
            np_arr = storage.to_numpy(zero_copy_only=False)
    else:
        np_arr = storage.to_numpy()

    # Reshape if requested
    if shape is not None:
        np_arr = np_arr.reshape(shape)

    return DimArray._from_data_and_unit(np_arr, unit)


# =============================================================================
# Table Conversion Functions
# =============================================================================


def to_arrow_table(arrays: dict[str, DimArray]):
    """Convert multiple DimArrays to an Arrow Table.

    Each array is stored as a column with its unit preserved in the
    extension type metadata. Shape information is stored in schema metadata.

    Args:
        arrays: Dictionary mapping column names to DimArrays.

    Returns:
        PyArrow Table with unit-aware columns.

    Raises:
        ImportError: If pyarrow is not installed.

    Example:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.io.arrow import to_arrow_table
        >>> table = to_arrow_table({
        ...     "distance": DimArray([1.0, 2.0], units.m),
        ...     "time": DimArray([0.5, 1.0], units.s)
        ... })
    """
    pa = _ensure_pyarrow()
    _register_extension_type()

    columns = {}
    metadata = {}

    for name, arr in arrays.items():
        # Store shape metadata
        metadata[f"{name}_shape"] = json.dumps(list(arr.shape))
        metadata[f"{name}_dtype"] = str(arr.dtype)

        # Store uncertainty if present
        if arr.has_uncertainty and arr._uncertainty is not None:
            columns[f"{name}_uncertainty"] = pa.array(arr._uncertainty.flatten())
            metadata[f"{name}_has_uncertainty"] = "True"
        else:
            metadata[f"{name}_has_uncertainty"] = "False"

        # Convert main data
        columns[name] = to_arrow_array(arr)

    table = pa.table(columns)

    # Add metadata to schema
    existing_metadata = table.schema.metadata or {}
    combined_metadata = {
        **existing_metadata,
        **{k.encode(): v.encode() for k, v in metadata.items()}
    }
    table = table.replace_schema_metadata(combined_metadata)

    return table


def from_arrow_table(table) -> dict[str, DimArray]:
    """Convert an Arrow Table to a dictionary of DimArrays.

    Extracts unit metadata from extension types and reconstructs
    the original array shapes.

    Args:
        table: PyArrow Table created by to_arrow_table.

    Returns:
        Dictionary mapping column names to DimArrays.

    Raises:
        ImportError: If pyarrow is not installed.

    Example:
        >>> from dimtensor.io.arrow import to_arrow_table, from_arrow_table
        >>> table = to_arrow_table({"x": DimArray([1.0], units.m)})
        >>> arrays = from_arrow_table(table)
        >>> arrays["x"].unit.symbol
        'm'
    """
    pa = _ensure_pyarrow()
    _register_extension_type()

    metadata = {}
    if table.schema.metadata:
        metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}

    result = {}

    for name in table.column_names:
        # Skip uncertainty columns (handled with main column)
        if name.endswith("_uncertainty"):
            continue

        column = table.column(name)
        # Combine chunks if necessary
        if column.num_chunks > 1:
            column = column.combine_chunks()
        else:
            column = column.chunk(0)

        # Get shape from metadata
        shape_key = f"{name}_shape"
        if shape_key in metadata:
            shape = tuple(json.loads(metadata[shape_key]))
        else:
            shape = (len(column),)

        # Convert to DimArray
        arr = from_arrow_array(column, shape=shape)

        # Load uncertainty if present
        unc_key = f"{name}_has_uncertainty"
        if metadata.get(unc_key) == "True":
            unc_col = f"{name}_uncertainty"
            if unc_col in table.column_names:
                unc_column = table.column(unc_col)
                if unc_column.num_chunks > 1:
                    unc_column = unc_column.combine_chunks()
                else:
                    unc_column = unc_column.chunk(0)
                uncertainty = unc_column.to_numpy().reshape(shape)
                arr = DimArray._from_data_and_unit(arr._data, arr.unit, uncertainty)

        result[name] = arr

    return result


# =============================================================================
# Record Batch Functions
# =============================================================================


def to_record_batch(arrays: dict[str, DimArray]):
    """Convert multiple DimArrays to an Arrow RecordBatch.

    Similar to to_arrow_table but returns a RecordBatch instead,
    which is more efficient for streaming IPC.

    Args:
        arrays: Dictionary mapping column names to DimArrays.

    Returns:
        PyArrow RecordBatch with unit-aware columns.

    Raises:
        ImportError: If pyarrow is not installed.
    """
    pa = _ensure_pyarrow()
    _register_extension_type()

    columns = {}
    metadata = {}

    for name, arr in arrays.items():
        metadata[f"{name}_shape"] = json.dumps(list(arr.shape))
        metadata[f"{name}_dtype"] = str(arr.dtype)

        if arr.has_uncertainty and arr._uncertainty is not None:
            columns[f"{name}_uncertainty"] = pa.array(arr._uncertainty.flatten())
            metadata[f"{name}_has_uncertainty"] = "True"
        else:
            metadata[f"{name}_has_uncertainty"] = "False"

        columns[name] = to_arrow_array(arr)

    # Create record batch with metadata
    arrays_list = list(columns.values())
    names = list(columns.keys())

    # Build schema with metadata
    fields = [pa.field(n, a.type) for n, a in zip(names, arrays_list)]
    schema = pa.schema(
        fields,
        metadata={k.encode(): v.encode() for k, v in metadata.items()}
    )

    batch = pa.record_batch(arrays_list, schema=schema)
    return batch


def from_record_batch(batch) -> dict[str, DimArray]:
    """Convert an Arrow RecordBatch to a dictionary of DimArrays.

    Args:
        batch: PyArrow RecordBatch created by to_record_batch.

    Returns:
        Dictionary mapping column names to DimArrays.

    Raises:
        ImportError: If pyarrow is not installed.
    """
    pa = _ensure_pyarrow()
    _register_extension_type()

    metadata = {}
    if batch.schema.metadata:
        metadata = {k.decode(): v.decode() for k, v in batch.schema.metadata.items()}

    result = {}

    for i, name in enumerate(batch.schema.names):
        if name.endswith("_uncertainty"):
            continue

        column = batch.column(i)

        shape_key = f"{name}_shape"
        if shape_key in metadata:
            shape = tuple(json.loads(metadata[shape_key]))
        else:
            shape = (len(column),)

        arr = from_arrow_array(column, shape=shape)

        unc_key = f"{name}_has_uncertainty"
        if metadata.get(unc_key) == "True":
            unc_name = f"{name}_uncertainty"
            unc_idx = batch.schema.get_field_index(unc_name)
            if unc_idx >= 0:
                unc_column = batch.column(unc_idx)
                uncertainty = unc_column.to_numpy().reshape(shape)
                arr = DimArray._from_data_and_unit(arr._data, arr.unit, uncertainty)

        result[name] = arr

    return result


# =============================================================================
# IPC Streaming Functions
# =============================================================================


def write_ipc_stream(arrays: dict[str, DimArray], sink: BinaryIO | str | Path) -> None:
    """Write DimArrays to an Arrow IPC stream.

    The stream format is suitable for streaming data between processes.
    Unit metadata is preserved via the DimArrayType extension type.

    Args:
        arrays: Dictionary mapping names to DimArrays.
        sink: File path or binary file object to write to.

    Raises:
        ImportError: If pyarrow is not installed.

    Example:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.io.arrow import write_ipc_stream
        >>> with open("data.arrow", "wb") as f:
        ...     write_ipc_stream({"x": DimArray([1.0], units.m)}, f)
    """
    pa = _ensure_pyarrow()
    _register_extension_type()

    batch = to_record_batch(arrays)

    # Handle path or file object
    if isinstance(sink, (str, Path)):
        with open(sink, "wb") as f:
            writer = pa.ipc.new_stream(f, batch.schema)
            writer.write_batch(batch)
            writer.close()
    else:
        writer = pa.ipc.new_stream(sink, batch.schema)
        writer.write_batch(batch)
        writer.close()


def read_ipc_stream(source: BinaryIO | str | Path) -> dict[str, DimArray]:
    """Read DimArrays from an Arrow IPC stream.

    Args:
        source: File path or binary file object to read from.

    Returns:
        Dictionary mapping names to DimArrays with preserved units.

    Raises:
        ImportError: If pyarrow is not installed.

    Example:
        >>> from dimtensor.io.arrow import read_ipc_stream
        >>> with open("data.arrow", "rb") as f:
        ...     arrays = read_ipc_stream(f)
    """
    pa = _ensure_pyarrow()
    _register_extension_type()

    # Handle path or file object
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            reader = pa.ipc.open_stream(f)
            batches = [batch for batch in reader]
    else:
        reader = pa.ipc.open_stream(source)
        batches = [batch for batch in reader]

    if not batches:
        return {}

    # Concatenate all batches
    if len(batches) == 1:
        return from_record_batch(batches[0])
    else:
        table = pa.Table.from_batches(batches)
        return from_arrow_table(table)


# =============================================================================
# IPC File Functions (Random Access)
# =============================================================================


def write_ipc_file(arrays: dict[str, DimArray], path: str | Path) -> None:
    """Write DimArrays to an Arrow IPC file.

    The file format supports random access to record batches,
    making it suitable for larger datasets.

    Args:
        arrays: Dictionary mapping names to DimArrays.
        path: File path to write to.

    Raises:
        ImportError: If pyarrow is not installed.

    Example:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.io.arrow import write_ipc_file
        >>> write_ipc_file({"x": DimArray([1.0], units.m)}, "data.arrow")
    """
    pa = _ensure_pyarrow()
    _register_extension_type()

    batch = to_record_batch(arrays)
    path = Path(path)

    with pa.OSFile(str(path), "wb") as sink:
        writer = pa.ipc.new_file(sink, batch.schema)
        writer.write_batch(batch)
        writer.close()


def read_ipc_file(path: str | Path) -> dict[str, DimArray]:
    """Read DimArrays from an Arrow IPC file.

    Args:
        path: File path to read from.

    Returns:
        Dictionary mapping names to DimArrays with preserved units.

    Raises:
        ImportError: If pyarrow is not installed.

    Example:
        >>> from dimtensor.io.arrow import read_ipc_file
        >>> arrays = read_ipc_file("data.arrow")
    """
    pa = _ensure_pyarrow()
    _register_extension_type()

    path = Path(path)

    with pa.OSFile(str(path), "rb") as source:
        reader = pa.ipc.open_file(source)
        batches = [reader.get_batch(i) for i in range(reader.num_record_batches)]

    if not batches:
        return {}

    if len(batches) == 1:
        return from_record_batch(batches[0])
    else:
        table = pa.Table.from_batches(batches)
        return from_arrow_table(table)


def read_ipc_file_mmap(path: str | Path) -> dict[str, DimArray]:
    """Read DimArrays from an Arrow IPC file using memory mapping.

    This provides zero-copy access for large files by memory-mapping
    the file instead of loading it entirely into RAM.

    Args:
        path: File path to read from.

    Returns:
        Dictionary mapping names to DimArrays. The underlying data
        may be backed by memory-mapped buffers.

    Raises:
        ImportError: If pyarrow is not installed.

    Example:
        >>> from dimtensor.io.arrow import read_ipc_file_mmap
        >>> arrays = read_ipc_file_mmap("large_data.arrow")
    """
    pa = _ensure_pyarrow()
    _register_extension_type()

    path = Path(path)

    # Use memory-mapped file
    mmap = pa.memory_map(str(path), "r")
    reader = pa.ipc.open_file(mmap)

    batches = [reader.get_batch(i) for i in range(reader.num_record_batches)]

    if not batches:
        mmap.close()
        return {}

    if len(batches) == 1:
        result = from_record_batch(batches[0])
    else:
        table = pa.Table.from_batches(batches)
        result = from_arrow_table(table)

    # Note: memory map stays open until arrays are garbage collected
    # This is intentional for zero-copy access
    return result


# =============================================================================
# Zero-Copy Buffer Conversion
# =============================================================================


def from_arrow_buffer(
    buffer,
    dtype: np.dtype,
    unit: Unit,
    shape: tuple[int, ...] | None = None,
) -> DimArray:
    """Create a DimArray from an Arrow buffer with zero-copy.

    This provides the most direct zero-copy path from Arrow to DimArray.
    The resulting array shares memory with the Arrow buffer.

    Args:
        buffer: PyArrow Buffer containing the raw data.
        dtype: NumPy dtype for interpreting the buffer.
        unit: Unit to attach to the array.
        shape: Shape for the resulting array. If None, inferred from buffer size.

    Returns:
        DimArray backed by the Arrow buffer's memory.

    Raises:
        ImportError: If pyarrow is not installed.
        ValueError: If shape is incompatible with buffer size.

    Example:
        >>> import pyarrow as pa
        >>> import numpy as np
        >>> from dimtensor import units
        >>> from dimtensor.io.arrow import from_arrow_buffer
        >>> buf = pa.py_buffer(np.array([1.0, 2.0, 3.0]).tobytes())
        >>> arr = from_arrow_buffer(buf, np.float64, units.m)
    """
    pa = _ensure_pyarrow()

    # Create numpy array from buffer (zero-copy view)
    np_arr = np.frombuffer(buffer, dtype=dtype)

    if shape is not None:
        expected_size = np.prod(shape)
        if np_arr.size != expected_size:
            raise ValueError(
                f"Buffer size {np_arr.size} incompatible with shape {shape}"
            )
        np_arr = np_arr.reshape(shape)

    return DimArray._from_data_and_unit(np_arr, unit)


# =============================================================================
# Utility Functions
# =============================================================================


def get_arrow_schema_with_units(arrays: dict[str, DimArray]):
    """Get an Arrow schema for a dictionary of DimArrays.

    Useful for creating empty tables or validating schemas.

    Args:
        arrays: Dictionary mapping names to DimArrays.

    Returns:
        PyArrow Schema with unit metadata in extension types.

    Raises:
        ImportError: If pyarrow is not installed.
    """
    pa = _ensure_pyarrow()
    _register_extension_type()
    DimArrayType = _get_extension_type_class()

    fields = []
    for name, arr in arrays.items():
        storage_type = _numpy_dtype_to_arrow_type(arr.dtype)
        ext_type = DimArrayType(storage_type, arr.unit)
        fields.append(pa.field(name, ext_type))

        if arr.has_uncertainty:
            fields.append(pa.field(f"{name}_uncertainty", storage_type))

    return pa.schema(fields)


def is_dimarray_type(arrow_type) -> bool:
    """Check if an Arrow type is a DimArrayType extension type.

    Args:
        arrow_type: PyArrow DataType to check.

    Returns:
        True if the type is a DimArrayType.

    Raises:
        ImportError: If pyarrow is not installed.
    """
    _ensure_pyarrow()
    _register_extension_type()

    return _is_dimarray_extension_type(arrow_type)


def extract_unit_from_type(arrow_type) -> Unit | None:
    """Extract unit from a DimArrayType extension type.

    Args:
        arrow_type: PyArrow DataType (possibly DimArrayType).

    Returns:
        Unit if the type is a DimArrayType, None otherwise.

    Raises:
        ImportError: If pyarrow is not installed.
    """
    _ensure_pyarrow()
    _register_extension_type()

    return _extract_unit_from_extension_type(arrow_type)

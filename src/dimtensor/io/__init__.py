"""Serialization and I/O for dimtensor.

Provides functions to save and load DimArrays to various formats
while preserving unit metadata.

Supported formats:
- JSON: Simple text format, no dependencies
- Pandas: DataFrame integration (requires pandas)
- HDF5: Binary format for large data (requires h5py)
- NetCDF: Scientific data format (requires netCDF4)
- Parquet: Columnar format (requires pyarrow)
- xarray: Labelled arrays (requires xarray)
- Arrow: Zero-copy IPC with unit metadata (requires pyarrow)

Example:
    >>> from dimtensor import DimArray, units
    >>> from dimtensor.io import to_json, from_json, to_dataframe
    >>>
    >>> arr = DimArray([1.0, 2.0, 3.0], units.m)
    >>> json_str = to_json(arr)
    >>> loaded = from_json(json_str)
    >>>
    >>> # Pandas integration
    >>> df = to_dataframe({"distance": arr, "time": times})
    >>>
    >>> # Arrow IPC (for zero-copy inter-process communication)
    >>> from dimtensor.io.arrow import write_ipc_file, read_ipc_file
    >>> write_ipc_file({"distance": arr}, "data.arrow")
"""

from .json import to_json, from_json, save_json, load_json
from .pandas import to_dataframe, from_dataframe, to_series, from_series
from .hdf5 import save_hdf5, load_hdf5
from .netcdf import save_netcdf, load_netcdf
from .parquet import save_parquet, load_parquet
from .xarray import to_xarray, from_xarray, to_dataset, from_dataset
from .arrow import (
    to_arrow_array,
    from_arrow_array,
    to_arrow_table,
    from_arrow_table,
    to_record_batch,
    from_record_batch,
    write_ipc_stream,
    read_ipc_stream,
    write_ipc_file,
    read_ipc_file,
    read_ipc_file_mmap,
    from_arrow_buffer,
)

__all__ = [
    # JSON
    "to_json",
    "from_json",
    "save_json",
    "load_json",
    # Pandas
    "to_dataframe",
    "from_dataframe",
    "to_series",
    "from_series",
    # HDF5
    "save_hdf5",
    "load_hdf5",
    # NetCDF
    "save_netcdf",
    "load_netcdf",
    # Parquet
    "save_parquet",
    "load_parquet",
    # xarray
    "to_xarray",
    "from_xarray",
    "to_dataset",
    "from_dataset",
    # Arrow
    "to_arrow_array",
    "from_arrow_array",
    "to_arrow_table",
    "from_arrow_table",
    "to_record_batch",
    "from_record_batch",
    "write_ipc_stream",
    "read_ipc_stream",
    "write_ipc_file",
    "read_ipc_file",
    "read_ipc_file_mmap",
    "from_arrow_buffer",
]

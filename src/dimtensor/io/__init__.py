"""Serialization and I/O for dimtensor.

Provides functions to save and load DimArrays to various formats
while preserving unit metadata.

Supported formats:
- JSON: Simple text format, no dependencies
- Pandas: DataFrame integration (requires pandas)
- HDF5: Binary format for large data (requires h5py)

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
"""

from .json import to_json, from_json, save_json, load_json
from .pandas import to_dataframe, from_dataframe, to_series, from_series
from .hdf5 import save_hdf5, load_hdf5

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
]

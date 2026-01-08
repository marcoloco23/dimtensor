"""HDF5 serialization for DimArray.

Save and load DimArrays in HDF5 format with unit metadata.
Requires h5py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core.dimarray import DimArray
from ..core.units import Unit, dimensionless


def save_hdf5(
    arr: DimArray,
    path: str | Path,
    dataset_name: str = "data",
    compression: str | None = "gzip",
) -> None:
    """Save DimArray to HDF5 file.

    The data is stored in a dataset, and unit metadata is stored
    as attributes on the dataset.

    Args:
        arr: DimArray to save.
        path: File path.
        dataset_name: Name for the dataset within the HDF5 file.
        compression: Compression algorithm ('gzip', 'lzf', None).

    Raises:
        ImportError: If h5py is not installed.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 support. "
            "Install with: pip install h5py"
        )

    path = Path(path)

    with h5py.File(path, "w") as f:
        # Create dataset with compression
        ds = f.create_dataset(
            dataset_name,
            data=arr._data,
            compression=compression,
        )

        # Store unit metadata as attributes
        ds.attrs["unit_symbol"] = arr.unit.symbol
        ds.attrs["unit_scale"] = arr.unit.scale
        ds.attrs["dim_length"] = float(arr.dimension.length)
        ds.attrs["dim_mass"] = float(arr.dimension.mass)
        ds.attrs["dim_time"] = float(arr.dimension.time)
        ds.attrs["dim_current"] = float(arr.dimension.current)
        ds.attrs["dim_temperature"] = float(arr.dimension.temperature)
        ds.attrs["dim_amount"] = float(arr.dimension.amount)
        ds.attrs["dim_luminosity"] = float(arr.dimension.luminosity)

        # Store uncertainty if present
        if arr.has_uncertainty:
            unc_ds = f.create_dataset(
                f"{dataset_name}_uncertainty",
                data=arr._uncertainty,
                compression=compression,
            )


def load_hdf5(
    path: str | Path,
    dataset_name: str = "data",
) -> DimArray:
    """Load DimArray from HDF5 file.

    Args:
        path: File path.
        dataset_name: Name of the dataset to load.

    Returns:
        DimArray.

    Raises:
        ImportError: If h5py is not installed.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 support. "
            "Install with: pip install h5py"
        )

    from ..core.dimensions import Dimension
    from fractions import Fraction

    path = Path(path)

    with h5py.File(path, "r") as f:
        ds = f[dataset_name]
        data = ds[:]

        # Reconstruct dimension
        dimension = Dimension(
            length=Fraction(ds.attrs["dim_length"]).limit_denominator(),
            mass=Fraction(ds.attrs["dim_mass"]).limit_denominator(),
            time=Fraction(ds.attrs["dim_time"]).limit_denominator(),
            current=Fraction(ds.attrs["dim_current"]).limit_denominator(),
            temperature=Fraction(ds.attrs["dim_temperature"]).limit_denominator(),
            amount=Fraction(ds.attrs["dim_amount"]).limit_denominator(),
            luminosity=Fraction(ds.attrs["dim_luminosity"]).limit_denominator(),
        )

        # Reconstruct unit
        unit = Unit(
            symbol=ds.attrs["unit_symbol"],
            dimension=dimension,
            scale=ds.attrs["unit_scale"],
        )

        # Load uncertainty if present
        uncertainty = None
        unc_name = f"{dataset_name}_uncertainty"
        if unc_name in f:
            uncertainty = f[unc_name][:]

    return DimArray(data, unit, uncertainty=uncertainty)


def save_multiple_hdf5(
    arrays: dict[str, DimArray],
    path: str | Path,
    compression: str | None = "gzip",
) -> None:
    """Save multiple DimArrays to a single HDF5 file.

    Each DimArray is stored as a separate dataset.

    Args:
        arrays: Dictionary mapping names to DimArrays.
        path: File path.
        compression: Compression algorithm.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 support. "
            "Install with: pip install h5py"
        )

    path = Path(path)

    with h5py.File(path, "w") as f:
        for name, arr in arrays.items():
            # Create dataset
            ds = f.create_dataset(
                name,
                data=arr._data,
                compression=compression,
            )

            # Store unit metadata
            ds.attrs["unit_symbol"] = arr.unit.symbol
            ds.attrs["unit_scale"] = arr.unit.scale
            ds.attrs["dim_length"] = float(arr.dimension.length)
            ds.attrs["dim_mass"] = float(arr.dimension.mass)
            ds.attrs["dim_time"] = float(arr.dimension.time)
            ds.attrs["dim_current"] = float(arr.dimension.current)
            ds.attrs["dim_temperature"] = float(arr.dimension.temperature)
            ds.attrs["dim_amount"] = float(arr.dimension.amount)
            ds.attrs["dim_luminosity"] = float(arr.dimension.luminosity)

            if arr.has_uncertainty:
                f.create_dataset(
                    f"{name}_uncertainty",
                    data=arr._uncertainty,
                    compression=compression,
                )


def load_multiple_hdf5(
    path: str | Path,
    dataset_names: list[str] | None = None,
) -> dict[str, DimArray]:
    """Load multiple DimArrays from HDF5 file.

    Args:
        path: File path.
        dataset_names: Names of datasets to load. If None, loads all.

    Returns:
        Dictionary mapping names to DimArrays.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 support. "
            "Install with: pip install h5py"
        )

    from ..core.dimensions import Dimension
    from fractions import Fraction

    path = Path(path)
    result = {}

    with h5py.File(path, "r") as f:
        if dataset_names is None:
            # Get all datasets that aren't uncertainty arrays
            dataset_names = [
                name for name in f.keys()
                if not name.endswith("_uncertainty")
            ]

        for name in dataset_names:
            ds = f[name]
            data = ds[:]

            dimension = Dimension(
                length=Fraction(ds.attrs["dim_length"]).limit_denominator(),
                mass=Fraction(ds.attrs["dim_mass"]).limit_denominator(),
                time=Fraction(ds.attrs["dim_time"]).limit_denominator(),
                current=Fraction(ds.attrs["dim_current"]).limit_denominator(),
                temperature=Fraction(ds.attrs["dim_temperature"]).limit_denominator(),
                amount=Fraction(ds.attrs["dim_amount"]).limit_denominator(),
                luminosity=Fraction(ds.attrs["dim_luminosity"]).limit_denominator(),
            )

            unit = Unit(
                symbol=ds.attrs["unit_symbol"],
                dimension=dimension,
                scale=ds.attrs["unit_scale"],
            )

            uncertainty = None
            unc_name = f"{name}_uncertainty"
            if unc_name in f:
                uncertainty = f[unc_name][:]

            result[name] = DimArray(data, unit, uncertainty=uncertainty)

    return result

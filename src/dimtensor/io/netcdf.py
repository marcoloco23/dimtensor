"""NetCDF serialization for DimArray.

Save and load DimArrays in NetCDF format with unit metadata.
Requires netCDF4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core.dimarray import DimArray
from ..core.units import Unit, dimensionless


def save_netcdf(
    arr: DimArray,
    path: str | Path,
    variable_name: str = "data",
    compression: bool = True,
) -> None:
    """Save DimArray to NetCDF file.

    The data is stored as a variable, and unit metadata is stored
    as attributes on the variable.

    Args:
        arr: DimArray to save.
        path: File path.
        variable_name: Name for the variable within the NetCDF file.
        compression: Whether to enable zlib compression.

    Raises:
        ImportError: If netCDF4 is not installed.
    """
    try:
        import netCDF4
    except ImportError:
        raise ImportError(
            "netCDF4 is required for NetCDF support. "
            "Install with: pip install netCDF4"
        )

    path = Path(path)

    with netCDF4.Dataset(path, "w", format="NETCDF4") as nc:
        # Create dimensions for each axis of the array
        for i, size in enumerate(arr.shape):
            nc.createDimension(f"dim_{i}", size)

        # Create variable with compression
        dims = tuple(f"dim_{i}" for i in range(arr.ndim))
        var = nc.createVariable(
            variable_name,
            arr.dtype,
            dims,
            zlib=compression,
        )
        var[:] = arr._data

        # Store unit metadata as attributes
        var.unit_symbol = arr.unit.symbol
        var.unit_scale = arr.unit.scale
        var.dim_length = float(arr.dimension.length)
        var.dim_mass = float(arr.dimension.mass)
        var.dim_time = float(arr.dimension.time)
        var.dim_current = float(arr.dimension.current)
        var.dim_temperature = float(arr.dimension.temperature)
        var.dim_amount = float(arr.dimension.amount)
        var.dim_luminosity = float(arr.dimension.luminosity)

        # Store uncertainty if present
        if arr.has_uncertainty and arr._uncertainty is not None:
            unc_var = nc.createVariable(
                f"{variable_name}_uncertainty",
                arr._uncertainty.dtype,
                dims,
                zlib=compression,
            )
            unc_var[:] = arr._uncertainty


def load_netcdf(
    path: str | Path,
    variable_name: str = "data",
) -> DimArray:
    """Load DimArray from NetCDF file.

    Args:
        path: File path.
        variable_name: Name of the variable to load.

    Returns:
        DimArray.

    Raises:
        ImportError: If netCDF4 is not installed.
    """
    try:
        import netCDF4
    except ImportError:
        raise ImportError(
            "netCDF4 is required for NetCDF support. "
            "Install with: pip install netCDF4"
        )

    from ..core.dimensions import Dimension
    from fractions import Fraction

    path = Path(path)

    with netCDF4.Dataset(path, "r") as nc:
        var = nc.variables[variable_name]
        data = var[:]

        # Reconstruct dimension
        dimension = Dimension(
            length=Fraction(float(var.dim_length)).limit_denominator(),
            mass=Fraction(float(var.dim_mass)).limit_denominator(),
            time=Fraction(float(var.dim_time)).limit_denominator(),
            current=Fraction(float(var.dim_current)).limit_denominator(),
            temperature=Fraction(float(var.dim_temperature)).limit_denominator(),
            amount=Fraction(float(var.dim_amount)).limit_denominator(),
            luminosity=Fraction(float(var.dim_luminosity)).limit_denominator(),
        )

        # Reconstruct unit
        unit = Unit(
            symbol=str(var.unit_symbol),
            dimension=dimension,
            scale=float(var.unit_scale),
        )

        # Load uncertainty if present
        uncertainty = None
        unc_name = f"{variable_name}_uncertainty"
        if unc_name in nc.variables:
            uncertainty = nc.variables[unc_name][:]

    return DimArray(data, unit, uncertainty=uncertainty)


def save_multiple_netcdf(
    arrays: dict[str, DimArray],
    path: str | Path,
    compression: bool = True,
) -> None:
    """Save multiple DimArrays to a single NetCDF file.

    Each DimArray is stored as a separate variable.

    Args:
        arrays: Dictionary mapping names to DimArrays.
        path: File path.
        compression: Whether to enable compression.

    Raises:
        ImportError: If netCDF4 is not installed.
    """
    try:
        import netCDF4
    except ImportError:
        raise ImportError(
            "netCDF4 is required for NetCDF support. "
            "Install with: pip install netCDF4"
        )

    path = Path(path)

    with netCDF4.Dataset(path, "w", format="NETCDF4") as nc:
        # Track created dimensions to reuse them
        dim_sizes: dict[int, str] = {}

        for name, arr in arrays.items():
            # Create dimensions for this array's shape
            dims = []
            for i, size in enumerate(arr.shape):
                if size not in dim_sizes:
                    dim_name = f"{name}_dim_{i}"
                    nc.createDimension(dim_name, size)
                    dim_sizes[size] = dim_name
                dims.append(dim_sizes[size])

            # Create variable
            var = nc.createVariable(
                name,
                arr.dtype,
                tuple(dims),
                zlib=compression,
            )
            var[:] = arr._data

            # Store unit metadata
            var.unit_symbol = arr.unit.symbol
            var.unit_scale = arr.unit.scale
            var.dim_length = float(arr.dimension.length)
            var.dim_mass = float(arr.dimension.mass)
            var.dim_time = float(arr.dimension.time)
            var.dim_current = float(arr.dimension.current)
            var.dim_temperature = float(arr.dimension.temperature)
            var.dim_amount = float(arr.dimension.amount)
            var.dim_luminosity = float(arr.dimension.luminosity)

            if arr.has_uncertainty and arr._uncertainty is not None:
                unc_var = nc.createVariable(
                    f"{name}_uncertainty",
                    arr._uncertainty.dtype,
                    tuple(dims),
                    zlib=compression,
                )
                unc_var[:] = arr._uncertainty


def load_multiple_netcdf(
    path: str | Path,
    variable_names: list[str] | None = None,
) -> dict[str, DimArray]:
    """Load multiple DimArrays from NetCDF file.

    Args:
        path: File path.
        variable_names: Names of variables to load. If None, loads all.

    Returns:
        Dictionary mapping names to DimArrays.

    Raises:
        ImportError: If netCDF4 is not installed.
    """
    try:
        import netCDF4
    except ImportError:
        raise ImportError(
            "netCDF4 is required for NetCDF support. "
            "Install with: pip install netCDF4"
        )

    from ..core.dimensions import Dimension
    from fractions import Fraction

    path = Path(path)
    result = {}

    with netCDF4.Dataset(path, "r") as nc:
        if variable_names is None:
            # Get all variables that aren't uncertainty arrays
            variable_names = [
                name for name in nc.variables.keys()
                if not name.endswith("_uncertainty")
            ]

        for name in variable_names:
            var = nc.variables[name]
            data = var[:]

            dimension = Dimension(
                length=Fraction(float(var.dim_length)).limit_denominator(),
                mass=Fraction(float(var.dim_mass)).limit_denominator(),
                time=Fraction(float(var.dim_time)).limit_denominator(),
                current=Fraction(float(var.dim_current)).limit_denominator(),
                temperature=Fraction(float(var.dim_temperature)).limit_denominator(),
                amount=Fraction(float(var.dim_amount)).limit_denominator(),
                luminosity=Fraction(float(var.dim_luminosity)).limit_denominator(),
            )

            unit = Unit(
                symbol=str(var.unit_symbol),
                dimension=dimension,
                scale=float(var.unit_scale),
            )

            uncertainty = None
            unc_name = f"{name}_uncertainty"
            if unc_name in nc.variables:
                uncertainty = nc.variables[unc_name][:]

            result[name] = DimArray(data, unit, uncertainty=uncertainty)

    return result


def save_netcdf_with_card(
    data: dict[str, DimArray],
    path: str | Path,
    card: "DimDatasetCard",
    compression: bool = True,
) -> None:
    """Save dataset to NetCDF with dataset card metadata.

    The dataset card is stored as JSON in the global attributes.

    Args:
        data: Dictionary mapping column names to DimArrays.
        path: File path.
        card: Dataset card with metadata.
        compression: Whether to enable compression.

    Raises:
        ImportError: If netCDF4 is not installed.
    """
    try:
        import netCDF4
    except ImportError:
        raise ImportError(
            "netCDF4 is required for NetCDF support. "
            "Install with: pip install netCDF4"
        )

    import json

    path = Path(path)

    with netCDF4.Dataset(path, "w", format="NETCDF4") as nc:
        # Store dataset card as JSON in global attributes
        nc.dataset_card = json.dumps(card.to_dict())

        # Track created dimensions to reuse them
        dim_sizes: dict[int, str] = {}

        for name, arr in data.items():
            # Create dimensions for this array's shape
            dims = []
            for i, size in enumerate(arr.shape):
                if size not in dim_sizes:
                    dim_name = f"{name}_dim_{i}"
                    nc.createDimension(dim_name, size)
                    dim_sizes[size] = dim_name
                dims.append(dim_sizes[size])

            # Create variable
            var = nc.createVariable(
                name,
                arr.dtype,
                tuple(dims),
                zlib=compression,
            )
            var[:] = arr._data

            # Store unit metadata
            var.unit_symbol = arr.unit.symbol
            var.unit_scale = arr.unit.scale
            var.dim_length = float(arr.dimension.length)
            var.dim_mass = float(arr.dimension.mass)
            var.dim_time = float(arr.dimension.time)
            var.dim_current = float(arr.dimension.current)
            var.dim_temperature = float(arr.dimension.temperature)
            var.dim_amount = float(arr.dimension.amount)
            var.dim_luminosity = float(arr.dimension.luminosity)

            if arr.has_uncertainty and arr._uncertainty is not None:
                unc_var = nc.createVariable(
                    f"{name}_uncertainty",
                    arr._uncertainty.dtype,
                    tuple(dims),
                    zlib=compression,
                )
                unc_var[:] = arr._uncertainty


def load_netcdf_with_card(
    path: str | Path,
) -> tuple[dict[str, DimArray], "DimDatasetCard"]:
    """Load dataset from NetCDF with dataset card metadata.

    Args:
        path: File path.

    Returns:
        Tuple of (data dictionary, dataset card).

    Raises:
        ImportError: If netCDF4 is not installed.
        AttributeError: If dataset card not found in file.
    """
    try:
        import netCDF4
    except ImportError:
        raise ImportError(
            "netCDF4 is required for NetCDF support. "
            "Install with: pip install netCDF4"
        )

    import json
    from ..datasets.card import DimDatasetCard

    path = Path(path)
    data = load_multiple_netcdf(path)

    with netCDF4.Dataset(path, "r") as nc:
        if not hasattr(nc, "dataset_card"):
            raise AttributeError(
                "No dataset card found in NetCDF file. "
                "Use load_multiple_netcdf() for files without cards."
            )

        card_json = nc.dataset_card
        card = DimDatasetCard.from_dict(json.loads(card_json))

    return data, card

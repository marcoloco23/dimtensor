"""xarray integration for DimArray.

Convert between DimArrays and xarray DataArrays with unit metadata.
Requires xarray.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.dimarray import DimArray
from ..core.units import Unit, dimensionless


def to_xarray(
    arr: DimArray,
    name: str | None = None,
    dims: tuple[str, ...] | None = None,
    coords: dict[str, Any] | None = None,
    attrs: dict[str, Any] | None = None,
) -> Any:  # xarray.DataArray
    """Convert DimArray to xarray DataArray.

    Unit metadata is stored in the DataArray's attrs.

    Args:
        arr: DimArray to convert.
        name: Name for the DataArray.
        dims: Dimension names. Defaults to ("dim_0", "dim_1", ...).
        coords: Coordinate arrays for each dimension.
        attrs: Additional attributes to include.

    Returns:
        xarray.DataArray with unit metadata in attrs.

    Raises:
        ImportError: If xarray is not installed.
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError(
            "xarray is required for xarray integration. "
            "Install with: pip install xarray"
        )

    # Default dimension names
    if dims is None:
        dims = tuple(f"dim_{i}" for i in range(arr.ndim))

    # Build attrs with unit metadata
    unit_attrs = {
        "dimtensor_unit_symbol": arr.unit.symbol,
        "dimtensor_unit_scale": arr.unit.scale,
        "dimtensor_dim_length": float(arr.dimension.length),
        "dimtensor_dim_mass": float(arr.dimension.mass),
        "dimtensor_dim_time": float(arr.dimension.time),
        "dimtensor_dim_current": float(arr.dimension.current),
        "dimtensor_dim_temperature": float(arr.dimension.temperature),
        "dimtensor_dim_amount": float(arr.dimension.amount),
        "dimtensor_dim_luminosity": float(arr.dimension.luminosity),
        "units": arr.unit.symbol,  # Common xarray convention
    }

    if arr.has_uncertainty and arr._uncertainty is not None:
        unit_attrs["dimtensor_has_uncertainty"] = True

    if attrs:
        unit_attrs.update(attrs)

    da = xr.DataArray(
        data=arr._data,
        dims=dims,
        coords=coords,
        name=name,
        attrs=unit_attrs,
    )

    # Store uncertainty as a separate coordinate if present
    if arr.has_uncertainty and arr._uncertainty is not None:
        da = da.assign_coords({
            f"{name or 'data'}_uncertainty": (dims, arr._uncertainty)
        })

    return da


def from_xarray(da: Any) -> DimArray:  # da: xarray.DataArray
    """Convert xarray DataArray to DimArray.

    Reads unit metadata from the DataArray's attrs.

    Args:
        da: xarray DataArray with dimtensor unit metadata.

    Returns:
        DimArray.

    Raises:
        ImportError: If xarray is not installed.
        ValueError: If required unit metadata is missing.
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError(
            "xarray is required for xarray integration. "
            "Install with: pip install xarray"
        )

    from ..core.dimensions import Dimension
    from fractions import Fraction

    attrs = da.attrs

    # Check for dimtensor metadata
    if "dimtensor_unit_symbol" not in attrs:
        # Fall back to dimensionless if no metadata
        return DimArray(da.values)

    # Reconstruct dimension
    dimension = Dimension(
        length=Fraction(float(attrs["dimtensor_dim_length"])).limit_denominator(),
        mass=Fraction(float(attrs["dimtensor_dim_mass"])).limit_denominator(),
        time=Fraction(float(attrs["dimtensor_dim_time"])).limit_denominator(),
        current=Fraction(float(attrs["dimtensor_dim_current"])).limit_denominator(),
        temperature=Fraction(float(attrs["dimtensor_dim_temperature"])).limit_denominator(),
        amount=Fraction(float(attrs["dimtensor_dim_amount"])).limit_denominator(),
        luminosity=Fraction(float(attrs["dimtensor_dim_luminosity"])).limit_denominator(),
    )

    # Reconstruct unit
    unit = Unit(
        symbol=attrs["dimtensor_unit_symbol"],
        dimension=dimension,
        scale=float(attrs["dimtensor_unit_scale"]),
    )

    # Check for uncertainty in coordinates
    uncertainty = None
    if attrs.get("dimtensor_has_uncertainty"):
        unc_name = f"{da.name or 'data'}_uncertainty"
        if unc_name in da.coords:
            uncertainty = da.coords[unc_name].values

    return DimArray(da.values, unit, uncertainty=uncertainty)


def to_dataset(
    arrays: dict[str, DimArray],
    coords: dict[str, Any] | None = None,
    attrs: dict[str, Any] | None = None,
) -> Any:  # xarray.Dataset
    """Convert multiple DimArrays to xarray Dataset.

    Each DimArray becomes a variable in the Dataset with unit metadata.

    Args:
        arrays: Dictionary mapping names to DimArrays.
        coords: Shared coordinates.
        attrs: Global attributes.

    Returns:
        xarray.Dataset with unit metadata.

    Raises:
        ImportError: If xarray is not installed.
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError(
            "xarray is required for xarray integration. "
            "Install with: pip install xarray"
        )

    data_vars: dict[str, Any] = {}
    for name, arr in arrays.items():
        dims = tuple(f"{name}_dim_{i}" for i in range(arr.ndim))

        var_attrs = {
            "dimtensor_unit_symbol": arr.unit.symbol,
            "dimtensor_unit_scale": arr.unit.scale,
            "dimtensor_dim_length": float(arr.dimension.length),
            "dimtensor_dim_mass": float(arr.dimension.mass),
            "dimtensor_dim_time": float(arr.dimension.time),
            "dimtensor_dim_current": float(arr.dimension.current),
            "dimtensor_dim_temperature": float(arr.dimension.temperature),
            "dimtensor_dim_amount": float(arr.dimension.amount),
            "dimtensor_dim_luminosity": float(arr.dimension.luminosity),
            "units": arr.unit.symbol,
        }

        if arr.has_uncertainty and arr._uncertainty is not None:
            var_attrs["dimtensor_has_uncertainty"] = True
            # Store uncertainty as separate variable
            data_vars[f"{name}_uncertainty"] = (dims, arr._uncertainty, {})

        data_vars[name] = (dims, arr._data, var_attrs)

    return xr.Dataset(data_vars, coords=coords, attrs=attrs or {})


def from_dataset(
    ds: Any,  # xarray.Dataset
    variable_names: list[str] | None = None,
) -> dict[str, DimArray]:
    """Convert xarray Dataset to DimArrays.

    Args:
        ds: xarray Dataset with dimtensor unit metadata.
        variable_names: Variables to convert. If None, converts all.

    Returns:
        Dictionary mapping names to DimArrays.

    Raises:
        ImportError: If xarray is not installed.
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError(
            "xarray is required for xarray integration. "
            "Install with: pip install xarray"
        )

    from ..core.dimensions import Dimension
    from fractions import Fraction

    if variable_names is None:
        # Get all variables that aren't uncertainty arrays
        variable_names = [
            name for name in ds.data_vars
            if not name.endswith("_uncertainty")
        ]

    result = {}
    for name in variable_names:
        var = ds[name]
        attrs = var.attrs

        if "dimtensor_unit_symbol" not in attrs:
            # Fall back to dimensionless
            result[name] = DimArray(var.values)
            continue

        dimension = Dimension(
            length=Fraction(float(attrs["dimtensor_dim_length"])).limit_denominator(),
            mass=Fraction(float(attrs["dimtensor_dim_mass"])).limit_denominator(),
            time=Fraction(float(attrs["dimtensor_dim_time"])).limit_denominator(),
            current=Fraction(float(attrs["dimtensor_dim_current"])).limit_denominator(),
            temperature=Fraction(float(attrs["dimtensor_dim_temperature"])).limit_denominator(),
            amount=Fraction(float(attrs["dimtensor_dim_amount"])).limit_denominator(),
            luminosity=Fraction(float(attrs["dimtensor_dim_luminosity"])).limit_denominator(),
        )

        unit = Unit(
            symbol=attrs["dimtensor_unit_symbol"],
            dimension=dimension,
            scale=float(attrs["dimtensor_unit_scale"]),
        )

        uncertainty = None
        if attrs.get("dimtensor_has_uncertainty"):
            unc_name = f"{name}_uncertainty"
            if unc_name in ds.data_vars:
                uncertainty = ds[unc_name].values

        result[name] = DimArray(var.values, unit, uncertainty=uncertainty)

    return result

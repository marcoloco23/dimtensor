"""Pandas integration for DimArray.

Provides functions to convert between DimArray and pandas DataFrame/Series.
Unit information is stored in the column/index metadata.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.dimarray import DimArray
from ..core.units import Unit, dimensionless


def to_series(arr: DimArray, name: str | None = None) -> Any:
    """Convert DimArray to pandas Series with unit metadata.

    The unit is stored in the Series attrs as 'unit_symbol'.

    Args:
        arr: DimArray to convert.
        name: Optional name for the Series.

    Returns:
        pandas Series with unit metadata.

    Raises:
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for DataFrame integration. "
            "Install with: pip install pandas"
        )

    series = pd.Series(arr._data, name=name)
    series.attrs["unit_symbol"] = arr.unit.symbol
    series.attrs["unit_scale"] = arr.unit.scale
    series.attrs["dimension"] = {
        "length": float(arr.dimension.length),
        "mass": float(arr.dimension.mass),
        "time": float(arr.dimension.time),
        "current": float(arr.dimension.current),
        "temperature": float(arr.dimension.temperature),
        "amount": float(arr.dimension.amount),
        "luminosity": float(arr.dimension.luminosity),
    }

    if arr.has_uncertainty and arr._uncertainty is not None:
        series.attrs["uncertainty"] = arr._uncertainty.tolist()

    return series


def from_series(series: Any) -> DimArray:
    """Create DimArray from pandas Series with unit metadata.

    Args:
        series: pandas Series with unit attrs.

    Returns:
        DimArray.

    Raises:
        ValueError: If series lacks unit metadata.
    """
    from ..core.dimensions import Dimension
    from fractions import Fraction

    if "unit_symbol" not in series.attrs:
        # No unit metadata, return dimensionless
        return DimArray(series.values, dimensionless)

    # Reconstruct dimension
    dim_data = series.attrs.get("dimension", {})
    if dim_data:
        dimension = Dimension(
            length=Fraction(dim_data.get("length", 0)).limit_denominator(),
            mass=Fraction(dim_data.get("mass", 0)).limit_denominator(),
            time=Fraction(dim_data.get("time", 0)).limit_denominator(),
            current=Fraction(dim_data.get("current", 0)).limit_denominator(),
            temperature=Fraction(dim_data.get("temperature", 0)).limit_denominator(),
            amount=Fraction(dim_data.get("amount", 0)).limit_denominator(),
            luminosity=Fraction(dim_data.get("luminosity", 0)).limit_denominator(),
        )
    else:
        dimension = dimensionless.dimension

    # Reconstruct unit
    unit = Unit(
        symbol=series.attrs["unit_symbol"],
        dimension=dimension,
        scale=series.attrs.get("unit_scale", 1.0),
    )

    # Handle uncertainty
    uncertainty = None
    if "uncertainty" in series.attrs:
        uncertainty = np.array(series.attrs["uncertainty"])

    return DimArray(series.values, unit, uncertainty=uncertainty)


def to_dataframe(
    arrays: dict[str, DimArray],
    index: Any = None,
) -> Any:
    """Convert multiple DimArrays to a pandas DataFrame.

    Each DimArray becomes a column. Unit metadata is stored in
    the DataFrame attrs as a dictionary mapping column names to units.

    Args:
        arrays: Dictionary mapping column names to DimArrays.
        index: Optional index for the DataFrame.

    Returns:
        pandas DataFrame with unit metadata.

    Raises:
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for DataFrame integration. "
            "Install with: pip install pandas"
        )

    data = {}
    units_meta = {}

    for name, arr in arrays.items():
        data[name] = arr._data
        units_meta[name] = {
            "unit_symbol": arr.unit.symbol,
            "unit_scale": arr.unit.scale,
            "dimension": {
                "length": float(arr.dimension.length),
                "mass": float(arr.dimension.mass),
                "time": float(arr.dimension.time),
                "current": float(arr.dimension.current),
                "temperature": float(arr.dimension.temperature),
                "amount": float(arr.dimension.amount),
                "luminosity": float(arr.dimension.luminosity),
            },
        }
        if arr.has_uncertainty and arr._uncertainty is not None:
            units_meta[name]["uncertainty"] = arr._uncertainty.tolist()

    df = pd.DataFrame(data, index=index)
    df.attrs["dimtensor_units"] = units_meta

    return df


def from_dataframe(df: Any, columns: list[str] | None = None) -> dict[str, DimArray]:
    """Create DimArrays from a pandas DataFrame.

    Args:
        df: pandas DataFrame with unit metadata.
        columns: Optional list of columns to extract. If None, extracts all.

    Returns:
        Dictionary mapping column names to DimArrays.
    """
    from ..core.dimensions import Dimension
    from fractions import Fraction

    if columns is None:
        columns = list(df.columns)

    units_meta = df.attrs.get("dimtensor_units", {})
    result = {}

    for col in columns:
        series_data = df[col].values

        if col in units_meta:
            meta = units_meta[col]
            dim_data = meta.get("dimension", {})
            dimension = Dimension(
                length=Fraction(dim_data.get("length", 0)).limit_denominator(),
                mass=Fraction(dim_data.get("mass", 0)).limit_denominator(),
                time=Fraction(dim_data.get("time", 0)).limit_denominator(),
                current=Fraction(dim_data.get("current", 0)).limit_denominator(),
                temperature=Fraction(dim_data.get("temperature", 0)).limit_denominator(),
                amount=Fraction(dim_data.get("amount", 0)).limit_denominator(),
                luminosity=Fraction(dim_data.get("luminosity", 0)).limit_denominator(),
            )
            unit = Unit(
                symbol=meta["unit_symbol"],
                dimension=dimension,
                scale=meta.get("unit_scale", 1.0),
            )
            uncertainty = None
            if "uncertainty" in meta:
                uncertainty = np.array(meta["uncertainty"])
            result[col] = DimArray(series_data, unit, uncertainty=uncertainty)
        else:
            result[col] = DimArray(series_data, dimensionless)

    return result

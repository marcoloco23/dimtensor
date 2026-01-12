"""Parquet serialization for DimArray.

Save and load DimArrays in Parquet format with unit metadata.
Requires pyarrow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..core.dimarray import DimArray
from ..core.units import Unit, dimensionless


def save_parquet(
    arr: DimArray,
    path: str | Path,
    compression: str = "snappy",
) -> None:
    """Save DimArray to Parquet file.

    The data is stored in a table with unit metadata in the schema.

    Args:
        arr: DimArray to save.
        path: File path.
        compression: Compression codec ('snappy', 'gzip', 'lz4', 'zstd', None).

    Raises:
        ImportError: If pyarrow is not installed.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet support. "
            "Install with: pip install pyarrow"
        )

    path = Path(path)

    # Flatten multidimensional arrays for storage
    flat_data = arr._data.flatten()

    # Create metadata dict
    metadata = {
        "dimtensor_version": "1.1",
        "shape": json.dumps(list(arr.shape)),
        "dtype": str(arr.dtype),
        "unit_symbol": arr.unit.symbol,
        "unit_scale": str(arr.unit.scale),
        "dim_length": str(float(arr.dimension.length)),
        "dim_mass": str(float(arr.dimension.mass)),
        "dim_time": str(float(arr.dimension.time)),
        "dim_current": str(float(arr.dimension.current)),
        "dim_temperature": str(float(arr.dimension.temperature)),
        "dim_amount": str(float(arr.dimension.amount)),
        "dim_luminosity": str(float(arr.dimension.luminosity)),
        "has_uncertainty": str(arr.has_uncertainty),
    }

    # Create table with data column
    table_data = {"data": flat_data}

    if arr.has_uncertainty and arr._uncertainty is not None:
        table_data["uncertainty"] = arr._uncertainty.flatten()

    table = pa.Table.from_pydict(table_data)

    # Add metadata to schema
    existing_metadata = table.schema.metadata or {}
    combined_metadata = {
        **existing_metadata,
        **{k.encode(): v.encode() for k, v in metadata.items()}
    }
    table = table.replace_schema_metadata(combined_metadata)

    # Write with compression
    pq.write_table(table, path, compression=compression)


def load_parquet(path: str | Path) -> DimArray:
    """Load DimArray from Parquet file.

    Args:
        path: File path.

    Returns:
        DimArray.

    Raises:
        ImportError: If pyarrow is not installed.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet support. "
            "Install with: pip install pyarrow"
        )

    from ..core.dimensions import Dimension
    from fractions import Fraction

    path = Path(path)
    table = pq.read_table(path)

    # Extract metadata
    metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}

    # Get data and reshape
    flat_data = table.column("data").to_numpy()
    shape = tuple(json.loads(metadata["shape"]))
    data = flat_data.reshape(shape)

    # Reconstruct dimension
    dimension = Dimension(
        length=Fraction(float(metadata["dim_length"])).limit_denominator(),
        mass=Fraction(float(metadata["dim_mass"])).limit_denominator(),
        time=Fraction(float(metadata["dim_time"])).limit_denominator(),
        current=Fraction(float(metadata["dim_current"])).limit_denominator(),
        temperature=Fraction(float(metadata["dim_temperature"])).limit_denominator(),
        amount=Fraction(float(metadata["dim_amount"])).limit_denominator(),
        luminosity=Fraction(float(metadata["dim_luminosity"])).limit_denominator(),
    )

    # Reconstruct unit
    unit = Unit(
        symbol=metadata["unit_symbol"],
        dimension=dimension,
        scale=float(metadata["unit_scale"]),
    )

    # Load uncertainty if present
    uncertainty = None
    if metadata.get("has_uncertainty") == "True" and "uncertainty" in table.column_names:
        uncertainty = table.column("uncertainty").to_numpy().reshape(shape)

    return DimArray(data, unit, uncertainty=uncertainty)


def save_multiple_parquet(
    arrays: dict[str, DimArray],
    path: str | Path,
    compression: str = "snappy",
) -> None:
    """Save multiple DimArrays to a single Parquet file.

    Each DimArray is stored with a prefix in the column names.

    Args:
        arrays: Dictionary mapping names to DimArrays.
        path: File path.
        compression: Compression codec.

    Raises:
        ImportError: If pyarrow is not installed.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet support. "
            "Install with: pip install pyarrow"
        )

    path = Path(path)

    # Build combined table data and metadata
    table_data = {}
    metadata = {"dimtensor_version": "1.1", "array_names": json.dumps(list(arrays.keys()))}

    for name, arr in arrays.items():
        # Flatten and store data
        table_data[f"{name}_data"] = arr._data.flatten()

        if arr.has_uncertainty and arr._uncertainty is not None:
            table_data[f"{name}_uncertainty"] = arr._uncertainty.flatten()

        # Store metadata for this array
        metadata[f"{name}_shape"] = json.dumps(list(arr.shape))
        metadata[f"{name}_dtype"] = str(arr.dtype)
        metadata[f"{name}_unit_symbol"] = arr.unit.symbol
        metadata[f"{name}_unit_scale"] = str(arr.unit.scale)
        metadata[f"{name}_dim_length"] = str(float(arr.dimension.length))
        metadata[f"{name}_dim_mass"] = str(float(arr.dimension.mass))
        metadata[f"{name}_dim_time"] = str(float(arr.dimension.time))
        metadata[f"{name}_dim_current"] = str(float(arr.dimension.current))
        metadata[f"{name}_dim_temperature"] = str(float(arr.dimension.temperature))
        metadata[f"{name}_dim_amount"] = str(float(arr.dimension.amount))
        metadata[f"{name}_dim_luminosity"] = str(float(arr.dimension.luminosity))
        metadata[f"{name}_has_uncertainty"] = str(arr.has_uncertainty)

    table = pa.Table.from_pydict(table_data)
    combined_metadata = {k.encode(): v.encode() for k, v in metadata.items()}
    table = table.replace_schema_metadata(combined_metadata)

    pq.write_table(table, path, compression=compression)


def load_multiple_parquet(
    path: str | Path,
    array_names: list[str] | None = None,
) -> dict[str, DimArray]:
    """Load multiple DimArrays from Parquet file.

    Args:
        path: File path.
        array_names: Names of arrays to load. If None, loads all.

    Returns:
        Dictionary mapping names to DimArrays.

    Raises:
        ImportError: If pyarrow is not installed.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet support. "
            "Install with: pip install pyarrow"
        )

    from ..core.dimensions import Dimension
    from fractions import Fraction

    path = Path(path)
    table = pq.read_table(path)
    metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}

    if array_names is None:
        array_names = json.loads(metadata["array_names"])

    result = {}
    for name in array_names:
        # Get data and reshape
        flat_data = table.column(f"{name}_data").to_numpy()
        shape = tuple(json.loads(metadata[f"{name}_shape"]))
        data = flat_data.reshape(shape)

        # Reconstruct dimension
        dimension = Dimension(
            length=Fraction(float(metadata[f"{name}_dim_length"])).limit_denominator(),
            mass=Fraction(float(metadata[f"{name}_dim_mass"])).limit_denominator(),
            time=Fraction(float(metadata[f"{name}_dim_time"])).limit_denominator(),
            current=Fraction(float(metadata[f"{name}_dim_current"])).limit_denominator(),
            temperature=Fraction(float(metadata[f"{name}_dim_temperature"])).limit_denominator(),
            amount=Fraction(float(metadata[f"{name}_dim_amount"])).limit_denominator(),
            luminosity=Fraction(float(metadata[f"{name}_dim_luminosity"])).limit_denominator(),
        )

        # Reconstruct unit
        unit = Unit(
            symbol=metadata[f"{name}_unit_symbol"],
            dimension=dimension,
            scale=float(metadata[f"{name}_unit_scale"]),
        )

        # Load uncertainty if present
        uncertainty = None
        unc_col = f"{name}_uncertainty"
        if metadata.get(f"{name}_has_uncertainty") == "True" and unc_col in table.column_names:
            uncertainty = table.column(unc_col).to_numpy().reshape(shape)

        result[name] = DimArray(data, unit, uncertainty=uncertainty)

    return result


def save_parquet_with_card(
    data: dict[str, DimArray],
    path: str | Path,
    card: "DimDatasetCard",
    compression: str = "snappy",
) -> None:
    """Save dataset to Parquet with dataset card metadata.

    The dataset card is stored as JSON in the schema metadata.

    Args:
        data: Dictionary mapping column names to DimArrays.
        path: File path.
        card: Dataset card with metadata.
        compression: Compression codec.

    Raises:
        ImportError: If pyarrow is not installed.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet support. "
            "Install with: pip install pyarrow"
        )

    import json

    path = Path(path)

    # Build combined table data and metadata
    table_data = {}
    metadata = {
        "dimtensor_version": "1.1",
        "array_names": json.dumps(list(data.keys())),
        "dataset_card": json.dumps(card.to_dict()),
    }

    for name, arr in data.items():
        # Flatten and store data
        table_data[f"{name}_data"] = arr._data.flatten()

        if arr.has_uncertainty and arr._uncertainty is not None:
            table_data[f"{name}_uncertainty"] = arr._uncertainty.flatten()

        # Store metadata for this array
        metadata[f"{name}_shape"] = json.dumps(list(arr.shape))
        metadata[f"{name}_dtype"] = str(arr.dtype)
        metadata[f"{name}_unit_symbol"] = arr.unit.symbol
        metadata[f"{name}_unit_scale"] = str(arr.unit.scale)
        metadata[f"{name}_dim_length"] = str(float(arr.dimension.length))
        metadata[f"{name}_dim_mass"] = str(float(arr.dimension.mass))
        metadata[f"{name}_dim_time"] = str(float(arr.dimension.time))
        metadata[f"{name}_dim_current"] = str(float(arr.dimension.current))
        metadata[f"{name}_dim_temperature"] = str(float(arr.dimension.temperature))
        metadata[f"{name}_dim_amount"] = str(float(arr.dimension.amount))
        metadata[f"{name}_dim_luminosity"] = str(float(arr.dimension.luminosity))
        metadata[f"{name}_has_uncertainty"] = str(arr.has_uncertainty)

    table = pa.Table.from_pydict(table_data)
    combined_metadata = {k.encode(): v.encode() for k, v in metadata.items()}
    table = table.replace_schema_metadata(combined_metadata)

    pq.write_table(table, path, compression=compression)


def load_parquet_with_card(
    path: str | Path,
) -> tuple[dict[str, DimArray], "DimDatasetCard"]:
    """Load dataset from Parquet with dataset card metadata.

    Args:
        path: File path.

    Returns:
        Tuple of (data dictionary, dataset card).

    Raises:
        ImportError: If pyarrow is not installed.
        KeyError: If dataset card not found in file.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet support. "
            "Install with: pip install pyarrow"
        )

    import json
    from ..datasets.card import DimDatasetCard

    path = Path(path)
    table = pq.read_table(path)
    metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}

    if "dataset_card" not in metadata:
        raise KeyError(
            "No dataset card found in Parquet file. "
            "Use load_multiple_parquet() for files without cards."
        )

    card = DimDatasetCard.from_dict(json.loads(metadata["dataset_card"]))
    data = load_multiple_parquet(path)

    return data, card

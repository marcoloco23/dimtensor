"""CSV serialization for DimArray with JSON sidecar metadata.

Save and load datasets with dimensional metadata using CSV for data
and JSON sidecar files for unit metadata.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.dimarray import DimArray

if TYPE_CHECKING:
    from ..datasets.card import DimDatasetCard


def save_csv_with_card(
    data: dict[str, DimArray | np.ndarray],
    csv_path: str | Path,
    card: DimDatasetCard | None = None,
    delimiter: str = ",",
) -> None:
    """Save dataset to CSV with metadata in JSON sidecar file.

    Args:
        data: Dictionary mapping column names to data arrays.
        csv_path: Path to CSV file.
        card: Dataset card with metadata. If None, minimal card is created.
        delimiter: CSV delimiter character.

    Raises:
        ValueError: If arrays have incompatible shapes.
    """
    csv_path = Path(csv_path)

    # Check all arrays have same length
    lengths = {name: len(arr) for name, arr in data.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"All arrays must have same length. Got: {lengths}"
        )

    # Create minimal card if not provided
    if card is None:
        from ..core.units import dimensionless
        from ..datasets.card import DimDatasetCard

        card = DimDatasetCard(name=csv_path.stem)
        for name, arr in data.items():
            if isinstance(arr, DimArray):
                unit = arr.unit
            else:
                unit = dimensionless
            card.add_column(name=name, unit=unit)

    # Write CSV data
    column_names = list(data.keys())
    num_rows = len(next(iter(data.values())))

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(column_names)

        for i in range(num_rows):
            row = []
            for name in column_names:
                arr = data[name]
                if isinstance(arr, DimArray):
                    value = arr._data[i]
                else:
                    value = arr[i]
                row.append(value)
            writer.writerow(row)

    # Write JSON sidecar
    from ..datasets.card import save_dataset_card

    json_path = csv_path.with_suffix(".json")
    save_dataset_card(card, json_path)


def load_csv_with_card(
    csv_path: str | Path,
    delimiter: str = ",",
) -> tuple[dict[str, DimArray], "DimDatasetCard"]:
    """Load dataset from CSV with metadata from JSON sidecar file.

    Args:
        csv_path: Path to CSV file.
        delimiter: CSV delimiter character.

    Returns:
        Tuple of (data dictionary, dataset card).

    Raises:
        FileNotFoundError: If CSV or JSON sidecar not found.
    """
    from ..datasets.card import load_dataset_card

    csv_path = Path(csv_path)
    json_path = csv_path.with_suffix(".json")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not json_path.exists():
        raise FileNotFoundError(
            f"JSON sidecar not found: {json_path}. "
            "Use load_csv() for plain CSV without metadata."
        )

    # Load card
    card = load_dataset_card(json_path)

    # Load CSV data
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        return {}, card

    # Convert to arrays
    data = {}
    for col in card.columns:
        if col.name not in rows[0]:
            if col.required:
                raise ValueError(f"Required column '{col.name}' not in CSV")
            continue

        # Extract column data
        values = [float(row[col.name]) for row in rows]
        arr = np.array(values)

        # Create DimArray with unit
        data[col.name] = DimArray(arr, col.unit)

    return data, card


def save_csv(
    data: dict[str, np.ndarray],
    path: str | Path,
    delimiter: str = ",",
) -> None:
    """Save data to plain CSV without metadata.

    This is a simple CSV export without unit information.
    For datasets with units, use save_csv_with_card() instead.

    Args:
        data: Dictionary mapping column names to data arrays.
        path: Path to CSV file.
        delimiter: CSV delimiter character.

    Raises:
        ValueError: If arrays have incompatible shapes.
    """
    path = Path(path)

    # Check all arrays have same length
    lengths = {name: len(arr) for name, arr in data.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"All arrays must have same length. Got: {lengths}"
        )

    # Write CSV
    column_names = list(data.keys())
    num_rows = len(next(iter(data.values())))

    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(column_names)

        for i in range(num_rows):
            row = []
            for name in column_names:
                arr = data[name]
                if isinstance(arr, DimArray):
                    value = arr._data[i]
                else:
                    value = arr[i]
                row.append(value)
            writer.writerow(row)


def load_csv(
    path: str | Path,
    delimiter: str = ",",
) -> dict[str, np.ndarray]:
    """Load data from plain CSV without metadata.

    This loads raw numeric data without unit information.
    For datasets with units, use load_csv_with_card() instead.

    Args:
        path: Path to CSV file.
        delimiter: CSV delimiter character.

    Returns:
        Dictionary mapping column names to numpy arrays.

    Raises:
        FileNotFoundError: If CSV file not found.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    # Load CSV data
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        return {}

    # Convert to arrays
    data = {}
    for col_name in rows[0].keys():
        try:
            values = [float(row[col_name]) for row in rows]
            data[col_name] = np.array(values)
        except ValueError:
            # Skip non-numeric columns
            pass

    return data

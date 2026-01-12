"""Dataset registry for physics-aware machine learning.

Provides a registry of datasets with dimensional metadata for
training physics-informed neural networks.

Example:
    >>> from dimtensor.datasets import list_datasets, get_dataset_info, load_dataset
    >>>
    >>> # List all datasets
    >>> datasets = list_datasets()
    >>> for ds in datasets[:3]:
    ...     print(f"{ds.name}: {ds.description}")
    >>>
    >>> # Get specific dataset
    >>> info = get_dataset_info("pendulum")
    >>>
    >>> # Load a real physics dataset
    >>> constants = load_dataset("nist_codata_2022")
    >>>
    >>> # Create and use dataset cards
    >>> from dimtensor.datasets import DimDatasetCard, save_dataset_card
    >>> card = DimDatasetCard(name="my_data", domain="mechanics")
    >>> card.add_column("position", units.m, "Position in meters")
"""

from .registry import (
    DatasetInfo,
    get_dataset_info,
    list_datasets,
    load_dataset,
    register_dataset,
)
from .card import (
    DimDatasetCard,
    ColumnInfo,
    CoordinateSystem,
    save_dataset_card,
    load_dataset_card,
)
from .validation import (
    ValidationError,
    validate_dataset,
    validate_column_units,
    validate_schema,
    check_units_convertible,
    validate_coordinate_system,
    validate_uncertainties,
)

# Import loaders submodule for direct access if needed
from . import loaders

__all__ = [
    # Registry
    "DatasetInfo",
    "get_dataset_info",
    "list_datasets",
    "load_dataset",
    "register_dataset",
    # Cards
    "DimDatasetCard",
    "ColumnInfo",
    "CoordinateSystem",
    "save_dataset_card",
    "load_dataset_card",
    # Validation
    "ValidationError",
    "validate_dataset",
    "validate_column_units",
    "validate_schema",
    "check_units_convertible",
    "validate_coordinate_system",
    "validate_uncertainties",
    # Loaders
    "loaders",
]

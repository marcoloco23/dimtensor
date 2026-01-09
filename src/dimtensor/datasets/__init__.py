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
"""

from .registry import (
    DatasetInfo,
    get_dataset_info,
    list_datasets,
    load_dataset,
    register_dataset,
)

# Import loaders submodule for direct access if needed
from . import loaders

__all__ = [
    "DatasetInfo",
    "get_dataset_info",
    "list_datasets",
    "load_dataset",
    "register_dataset",
    "loaders",
]

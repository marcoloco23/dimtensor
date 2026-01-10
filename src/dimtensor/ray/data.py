"""Ray Data integration for unit-aware data pipelines.

Provides DimDataset, a wrapper around Ray Dataset that tracks
physical units through distributed data transformations.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, TYPE_CHECKING

import numpy as np

from ..core.dimarray import DimArray
from ..core.units import Unit

if TYPE_CHECKING:
    import ray.data


class DimDataset:
    """Unit-aware wrapper for Ray Dataset.

    Tracks physical units through Ray Data transformations, ensuring
    dimensional correctness in distributed data pipelines.

    Example:
        >>> import numpy as np
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.ray import DimDataset
        >>>
        >>> # Create from DimArrays
        >>> positions = DimArray(np.random.randn(1000, 3), units.m)
        >>> velocities = DimArray(np.random.randn(1000, 3), units.m / units.s)
        >>>
        >>> ds = DimDataset.from_dimarrays({
        ...     "position": positions,
        ...     "velocity": velocities,
        ... })
        >>>
        >>> # Transform (units preserved)
        >>> ds = ds.map_batches(lambda batch: {
        ...     "position": batch["position"] * 2,
        ...     "velocity": batch["velocity"],
        ... })
        >>>
        >>> # Iterate with units
        >>> for batch in ds.iter_batches(batch_size=256):
        ...     print(batch["position"].unit)  # m
        ...     print(batch["velocity"].unit)  # m/s
    """

    def __init__(
        self,
        dataset: "ray.data.Dataset",
        units: dict[str, Unit],
    ) -> None:
        """Initialize DimDataset.

        Args:
            dataset: Underlying Ray Dataset.
            units: Mapping of column names to their units.
        """
        self._dataset = dataset
        self._units = units.copy()

    @classmethod
    def from_dimarrays(
        cls,
        arrays: dict[str, DimArray],
        *,
        parallelism: int = -1,
    ) -> "DimDataset":
        """Create DimDataset from dictionary of DimArrays.

        Args:
            arrays: Dictionary mapping column names to DimArrays.
            parallelism: Number of blocks for the dataset. -1 for auto.

        Returns:
            DimDataset with unit tracking.

        Example:
            >>> positions = DimArray(np.random.randn(1000, 3), units.m)
            >>> ds = DimDataset.from_dimarrays({"position": positions})
        """
        import ray.data

        # Extract numpy arrays and units
        data = {name: arr._data for name, arr in arrays.items()}
        units_map = {name: arr.unit for name, arr in arrays.items()}

        # Get number of rows from first array
        first_arr = next(iter(data.values()))
        n_rows = len(first_arr)

        # Convert dict of arrays to list of dicts (rows)
        rows = []
        for i in range(n_rows):
            row = {}
            for name, arr in data.items():
                # Handle multi-dimensional arrays
                if arr.ndim == 1:
                    row[name] = arr[i]
                else:
                    row[name] = arr[i]  # Keep as array for each row
            rows.append(row)

        # Create Ray Dataset
        dataset = ray.data.from_items(rows, parallelism=parallelism)

        return cls(dataset, units_map)

    @classmethod
    def from_numpy(
        cls,
        arrays: dict[str, np.ndarray],
        units: dict[str, Unit],
        *,
        parallelism: int = -1,
    ) -> "DimDataset":
        """Create DimDataset from numpy arrays with explicit units.

        Args:
            arrays: Dictionary mapping column names to numpy arrays.
            units: Dictionary mapping column names to their units.
            parallelism: Number of blocks for the dataset. -1 for auto.

        Returns:
            DimDataset with unit tracking.
        """
        import ray.data

        # Validate units match arrays
        if set(arrays.keys()) != set(units.keys()):
            raise ValueError(
                f"Keys mismatch: arrays has {set(arrays.keys())}, "
                f"units has {set(units.keys())}"
            )

        # Get number of rows from first array
        first_arr = next(iter(arrays.values()))
        n_rows = len(first_arr)

        # Convert dict of arrays to list of dicts (rows)
        rows = []
        for i in range(n_rows):
            row = {}
            for name, arr in arrays.items():
                if arr.ndim == 1:
                    row[name] = arr[i]
                else:
                    row[name] = arr[i]
            rows.append(row)

        # Create Ray Dataset
        dataset = ray.data.from_items(rows, parallelism=parallelism)

        return cls(dataset, units.copy())

    def map(self, fn: Callable[[dict], dict]) -> "DimDataset":
        """Apply function to each row, preserving units.

        Args:
            fn: Function that takes a row dict and returns a row dict.
                The function should preserve the column structure.

        Returns:
            Transformed DimDataset.

        Note:
            The function receives and should return raw numpy values.
            Units are tracked separately and preserved automatically.
        """
        new_dataset = self._dataset.map(fn)
        return DimDataset(new_dataset, self._units.copy())

    def map_batches(
        self,
        fn: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]],
        *,
        batch_size: int | None = None,
        batch_format: str = "numpy",
        **kwargs: Any,
    ) -> "DimDataset":
        """Apply function to batches, preserving units.

        Args:
            fn: Function that takes a batch dict and returns a batch dict.
            batch_size: Batch size for processing. None for full batch.
            batch_format: Batch format, typically "numpy".
            **kwargs: Additional arguments to ray.data.Dataset.map_batches.

        Returns:
            Transformed DimDataset.

        Example:
            >>> ds = ds.map_batches(lambda batch: {
            ...     "position": batch["position"] * 2,
            ...     "velocity": batch["velocity"],
            ... })
        """
        if batch_size is not None:
            new_dataset = self._dataset.map_batches(
                fn, batch_size=batch_size, batch_format=batch_format, **kwargs
            )
        else:
            new_dataset = self._dataset.map_batches(
                fn, batch_format=batch_format, **kwargs
            )
        return DimDataset(new_dataset, self._units.copy())

    def filter(self, fn: Callable[[dict], bool]) -> "DimDataset":
        """Filter rows, preserving units.

        Args:
            fn: Predicate function that takes a row and returns True to keep.

        Returns:
            Filtered DimDataset.
        """
        new_dataset = self._dataset.filter(fn)
        return DimDataset(new_dataset, self._units.copy())

    def select_columns(self, cols: list[str]) -> "DimDataset":
        """Select specific columns, preserving their units.

        Args:
            cols: Column names to select.

        Returns:
            DimDataset with only selected columns.
        """
        new_dataset = self._dataset.select_columns(cols)
        new_units = {k: v for k, v in self._units.items() if k in cols}
        return DimDataset(new_dataset, new_units)

    def to_dimarrays(self) -> dict[str, DimArray]:
        """Convert back to dictionary of DimArrays.

        Collects all data to the driver node.

        Returns:
            Dictionary mapping column names to DimArrays.

        Warning:
            This materializes the entire dataset in memory.
        """
        # Collect all rows
        rows = self._dataset.take_all()

        if not rows:
            # Empty dataset
            return {
                name: DimArray(np.array([]), unit)
                for name, unit in self._units.items()
            }

        # Convert to column format
        result = {}
        for name, unit in self._units.items():
            data = np.array([row[name] for row in rows])
            result[name] = DimArray._from_data_and_unit(data, unit)

        return result

    def iter_batches(
        self,
        *,
        batch_size: int = 256,
        batch_format: str = "numpy",
        drop_last: bool = False,
    ) -> Iterator[dict[str, DimArray]]:
        """Iterate over batches as DimArrays.

        Memory-efficient iteration for large datasets.

        Args:
            batch_size: Number of rows per batch.
            batch_format: Batch format (typically "numpy").
            drop_last: Whether to drop the last incomplete batch.

        Yields:
            Dictionary mapping column names to DimArrays.

        Example:
            >>> for batch in ds.iter_batches(batch_size=256):
            ...     position = batch["position"]  # DimArray with units.m
            ...     velocity = batch["velocity"]  # DimArray with units.m/s
        """
        for batch in self._dataset.iter_batches(
            batch_size=batch_size,
            batch_format=batch_format,
            drop_last=drop_last,
        ):
            yield {
                name: DimArray._from_data_and_unit(batch[name], self._units[name])
                for name in self._units
                if name in batch
            }

    def iter_torch_batches(
        self,
        *,
        batch_size: int = 256,
        drop_last: bool = False,
        device: str | None = None,
    ) -> Iterator[dict[str, "DimTensor"]]:
        """Iterate over batches as DimTensors.

        Args:
            batch_size: Number of rows per batch.
            drop_last: Whether to drop the last incomplete batch.
            device: Device to place tensors on ('cpu', 'cuda', 'mps').

        Yields:
            Dictionary mapping column names to DimTensors.
        """
        import torch
        from ..torch.dimtensor import DimTensor

        for batch in self._dataset.iter_batches(
            batch_size=batch_size,
            batch_format="numpy",
            drop_last=drop_last,
        ):
            result = {}
            for name, unit in self._units.items():
                if name in batch:
                    tensor = torch.tensor(batch[name])
                    if device is not None:
                        tensor = tensor.to(device=device)
                    result[name] = DimTensor._from_tensor_and_unit(tensor, unit)
            yield result

    def add_column(
        self,
        name: str,
        fn: Callable[[dict], Any],
        unit: Unit,
    ) -> "DimDataset":
        """Add a new column with specified unit.

        Args:
            name: Name for the new column.
            fn: Function to compute the column value from each row.
            unit: Unit for the new column.

        Returns:
            DimDataset with the new column.
        """
        new_dataset = self._dataset.add_column(name, fn)
        new_units = self._units.copy()
        new_units[name] = unit
        return DimDataset(new_dataset, new_units)

    def drop_columns(self, cols: list[str]) -> "DimDataset":
        """Drop specified columns.

        Args:
            cols: Column names to drop.

        Returns:
            DimDataset without the specified columns.
        """
        new_dataset = self._dataset.drop_columns(cols)
        new_units = {k: v for k, v in self._units.items() if k not in cols}
        return DimDataset(new_dataset, new_units)

    def random_shuffle(self, *, seed: int | None = None) -> "DimDataset":
        """Randomly shuffle the dataset.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Shuffled DimDataset.
        """
        new_dataset = self._dataset.random_shuffle(seed=seed)
        return DimDataset(new_dataset, self._units.copy())

    def split(
        self,
        n: int,
        *,
        equal: bool = False,
    ) -> list["DimDataset"]:
        """Split dataset into n parts.

        Args:
            n: Number of splits.
            equal: If True, splits have equal size (may drop rows).

        Returns:
            List of DimDatasets.
        """
        splits = self._dataset.split(n, equal=equal)
        return [DimDataset(s, self._units.copy()) for s in splits]

    def train_test_split(
        self,
        test_size: float,
        *,
        seed: int | None = None,
        shuffle: bool = True,
    ) -> tuple["DimDataset", "DimDataset"]:
        """Split dataset into train and test sets.

        Args:
            test_size: Fraction of data for test set (0.0 to 1.0).
            seed: Random seed for reproducibility.
            shuffle: Whether to shuffle before splitting.

        Returns:
            Tuple of (train_dataset, test_dataset).
        """
        if shuffle:
            ds = self._dataset.random_shuffle(seed=seed)
        else:
            ds = self._dataset

        n_test = int(len(self) * test_size)
        splits = ds.split_at_indices([len(self) - n_test])

        return (
            DimDataset(splits[0], self._units.copy()),
            DimDataset(splits[1], self._units.copy()),
        )

    @property
    def units(self) -> dict[str, Unit]:
        """Get unit mapping.

        Returns:
            Copy of the column-to-unit mapping.
        """
        return self._units.copy()

    @property
    def columns(self) -> list[str]:
        """Get column names.

        Returns:
            List of column names.
        """
        return list(self._units.keys())

    def __len__(self) -> int:
        """Number of rows in dataset."""
        return self._dataset.count()

    def count(self) -> int:
        """Number of rows in dataset."""
        return self._dataset.count()

    def __repr__(self) -> str:
        units_str = ", ".join(f"{k}: {v.symbol}" for k, v in self._units.items())
        return f"DimDataset(count={len(self)}, units={{{units_str}}})"

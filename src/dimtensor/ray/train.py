"""Ray Train integration for distributed unit-aware training.

Provides utilities for distributed training with physical units,
including DimTrainContext for automatic scaling/descaling.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor
    from ray.train.torch import TorchTrainer

from ..core.units import Unit

# Try to import DimTensor-related types
try:
    from ..torch.dimtensor import DimTensor
    from ..torch.scaler import DimScaler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DimTrainContext:
    """Context manager for unit-aware distributed training.

    Handles non-dimensionalization and re-dimensionalization
    in distributed training context, making it easy to train
    neural networks on scaled data while preserving physical units.

    Example:
        >>> def train_func(config):
        ...     from dimtensor.ray import DimTrainContext
        ...     from ray import train
        ...
        ...     ctx = DimTrainContext(
        ...         scaler=config["scaler"],
        ...         input_units={"position": units.m, "velocity": units.m/units.s},
        ...         output_units={"energy": units.J},
        ...     )
        ...
        ...     model = create_model()
        ...     model = train.torch.prepare_model(model)
        ...
        ...     shard = train.get_dataset_shard("train")
        ...     for batch in ctx.iter_batches(shard):
        ...         # batch contains scaled (dimensionless) tensors
        ...         pred = model(batch["position"], batch["velocity"])
        ...         loss = criterion(pred, batch["energy"])
        ...         ...
    """

    def __init__(
        self,
        scaler: "DimScaler | None" = None,
        input_units: dict[str, Unit] | None = None,
        output_units: dict[str, Unit] | None = None,
    ) -> None:
        """Initialize training context.

        Args:
            scaler: DimScaler for non-dimensionalization. If None,
                no scaling is applied.
            input_units: Mapping of input names to their units.
            output_units: Mapping of output names to their units.
        """
        self.scaler = scaler
        self.input_units = input_units or {}
        self.output_units = output_units or {}

    def iter_batches(
        self,
        dataset_shard: Any,
        *,
        batch_size: int = 32,
        dtype: "torch.dtype | None" = None,
        device: str | None = None,
    ) -> Iterator[dict[str, "Tensor"]]:
        """Iterate over dataset shard with automatic scaling.

        Args:
            dataset_shard: Ray Train dataset shard.
            batch_size: Batch size.
            dtype: Torch dtype for tensors. If None, uses float32.
            device: Device for tensors. If None, uses default.

        Yields:
            Dict of scaled (dimensionless) torch.Tensors.
        """
        import torch

        if dtype is None:
            dtype = torch.float32

        for batch in dataset_shard.iter_torch_batches(
            batch_size=batch_size,
            dtypes=dtype,
        ):
            scaled_batch = {}
            for name, tensor in batch.items():
                if device is not None:
                    tensor = tensor.to(device=device)

                if self.scaler is not None and name in self.input_units:
                    unit = self.input_units[name]
                    # Create DimTensor, scale, extract raw tensor
                    dim_tensor = DimTensor._from_tensor_and_unit(tensor, unit)
                    scaled_batch[name] = self.scaler.transform(dim_tensor)
                else:
                    scaled_batch[name] = tensor

            yield scaled_batch

    def scale_input(
        self,
        tensor: "Tensor",
        name: str,
    ) -> "Tensor":
        """Scale an input tensor to dimensionless values.

        Args:
            tensor: Raw tensor with physical values.
            name: Name of the input (to look up unit and scale).

        Returns:
            Scaled dimensionless tensor.
        """
        if self.scaler is None:
            return tensor

        if name not in self.input_units:
            return tensor

        unit = self.input_units[name]
        dim_tensor = DimTensor._from_tensor_and_unit(tensor, unit)
        return self.scaler.transform(dim_tensor)

    def inverse_transform(
        self,
        tensor: "Tensor",
        name: str,
    ) -> "DimTensor":
        """Convert model output back to physical units.

        Args:
            tensor: Raw model output (dimensionless).
            name: Output name (to look up unit).

        Returns:
            DimTensor with physical units.

        Raises:
            KeyError: If name not in output_units.
        """
        if name not in self.output_units:
            raise KeyError(f"Unknown output: {name}. Known: {list(self.output_units.keys())}")

        unit = self.output_units[name]

        if self.scaler is not None:
            return self.scaler.inverse_transform(tensor, unit)

        # No scaling, just assign unit
        return DimTensor._from_tensor_and_unit(tensor.clone(), unit)

    def scale_target(
        self,
        tensor: "Tensor",
        name: str,
    ) -> "Tensor":
        """Scale a target tensor for loss computation.

        Args:
            tensor: Target tensor with physical values.
            name: Name of the target (to look up unit and scale).

        Returns:
            Scaled dimensionless tensor.
        """
        if self.scaler is None:
            return tensor

        if name not in self.output_units:
            return tensor

        unit = self.output_units[name]
        dim_tensor = DimTensor._from_tensor_and_unit(tensor, unit)
        return self.scaler.transform(dim_tensor)


def create_dim_trainer(
    train_func: Callable[[dict], None],
    *,
    datasets: dict[str, "DimDataset"] | None = None,
    scaler: "DimScaler | None" = None,
    num_workers: int = 2,
    use_gpu: bool = False,
    resources_per_worker: dict[str, float] | None = None,
    **kwargs: Any,
) -> "TorchTrainer":
    """Create a TorchTrainer with unit-aware datasets.

    Convenience function that sets up Ray Train with DimDatasets
    and passes scaler and unit information to workers.

    Args:
        train_func: Training function that receives config dict.
        datasets: Dict of DimDatasets for training/validation.
        scaler: DimScaler for non-dimensionalization.
        num_workers: Number of training workers.
        use_gpu: Whether to use GPUs.
        resources_per_worker: Resource requirements per worker.
        **kwargs: Additional TorchTrainer arguments.

    Returns:
        Configured TorchTrainer.

    Example:
        >>> from dimtensor.ray import DimDataset, create_dim_trainer
        >>> from dimtensor.torch import DimScaler
        >>>
        >>> scaler = DimScaler(method="characteristic")
        >>> scaler.fit(train_positions, train_velocities)
        >>>
        >>> train_ds = DimDataset.from_dimarrays({
        ...     "position": positions,
        ...     "velocity": velocities,
        ... })
        >>>
        >>> def train_func(config):
        ...     # Training logic here
        ...     pass
        >>>
        >>> trainer = create_dim_trainer(
        ...     train_func,
        ...     datasets={"train": train_ds},
        ...     scaler=scaler,
        ...     num_workers=4,
        ... )
        >>> result = trainer.fit()
    """
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer

    # Convert DimDatasets to Ray Datasets
    ray_datasets = {}
    units_info = {}

    if datasets is not None:
        for name, ds in datasets.items():
            ray_datasets[name] = ds._dataset
            units_info[name] = ds.units

    # Build config with dimtensor metadata
    config = kwargs.pop("train_loop_config", {})
    config["_dimtensor_scaler"] = scaler
    config["_dimtensor_units"] = units_info

    # Build scaling config
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker=resources_per_worker,
    )

    return TorchTrainer(
        train_func,
        datasets=ray_datasets if ray_datasets else None,
        train_loop_config=config,
        scaling_config=scaling_config,
        **kwargs,
    )


def get_dim_context_from_config(config: dict[str, Any]) -> DimTrainContext:
    """Extract DimTrainContext from training config.

    Helper function to reconstruct context in training workers.

    Args:
        config: Training config dict passed to train_func.

    Returns:
        DimTrainContext configured from the config.

    Example:
        >>> def train_func(config):
        ...     ctx = get_dim_context_from_config(config)
        ...     # Use ctx.iter_batches(), ctx.inverse_transform(), etc.
    """
    scaler = config.get("_dimtensor_scaler")
    units_info = config.get("_dimtensor_units", {})

    # Flatten units from all datasets for input_units
    input_units = {}
    for ds_units in units_info.values():
        input_units.update(ds_units)

    return DimTrainContext(
        scaler=scaler,
        input_units=input_units,
        output_units={},  # Must be set by user in train_func
    )

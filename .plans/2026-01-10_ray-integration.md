# Plan: Ray Integration for Distributed ML with Units

**Date**: 2026-01-10
**Status**: PLANNING
**Task ID**: #203
**Author**: agent (planner)

---

## Goal

Integrate dimtensor with Ray to enable unit-aware distributed computing for machine learning workflows, including Ray remote functions, Ray Train for distributed training, and Ray Data for data pipelines.

---

## Background

Ray is a leading distributed computing framework for scaling AI/ML workloads. dimtensor users working on large-scale physics simulations and scientific ML need:

1. **Distributed computation** - Run unit-aware computations across clusters
2. **Distributed training** - Scale PyTorch/JAX training with physical units
3. **Data pipelines** - Process large physics datasets with unit preservation

Currently, dimtensor supports:
- NumPy `DimArray` with unit tracking
- PyTorch `DimTensor` with autograd
- JAX `DimArray` with pytree registration
- Serialization via JSON, HDF5, Parquet

**Gap**: No native support for Ray's distributed primitives.

---

## Research Summary

### Ray Core Patterns

1. **Remote Functions (`@ray.remote`)**: Execute functions on workers
   - Objects passed via `ray.put()` and `ray.get()`
   - Need custom serialization for DimArray/DimTensor

2. **Actors (`@ray.remote` on classes)**: Stateful distributed objects
   - Can maintain unit context across method calls

3. **Object Store**: Efficient data transfer via references
   - NumPy arrays serialized efficiently
   - Custom objects need serialization handlers

### Ray Train Patterns

1. **TorchTrainer**: Distributed PyTorch training
   - `prepare_model()` wraps model for DDP
   - `prepare_data_loader()` shards data
   - `train.report()` for metrics/checkpoints

2. **Scaling**: `ScalingConfig(num_workers=N, use_gpu=True)`

### Ray Data Patterns

1. **Dataset**: Distributed data abstraction
   - `map()` / `map_batches()` for transformations
   - `filter()` for conditional selection
   - `from_items()`, `from_numpy()`, `from_numpy_refs()`

2. **Integration with Train**: `train.get_dataset_shard()`

### dimtensor Existing Patterns

1. **Serialization**: `io/json.py` - `to_dict()` / `from_dict()` pattern
2. **sklearn Integration**: `sklearn/transformers.py` - extract/wrap pattern
3. **Scaler Pattern**: `torch/scaler.py` - transform/inverse_transform
4. **JAX pytree**: `jax/dimarray.py` - flatten/unflatten registration

---

## Implementation Approach

### Option A: Wrapper Functions (Recommended)

Create thin wrappers that handle unit serialization transparently.

**Pros:**
- Non-invasive, works with existing Ray code
- Users can opt-in selectively
- Easy to test and maintain

**Cons:**
- Some overhead from serialization
- Users must use dimtensor-specific wrappers

### Option B: Custom Serializers Only

Register custom serializers with Ray for automatic handling.

**Pros:**
- Transparent to user code
- No wrapper functions needed

**Cons:**
- Global state (serializer registration)
- May conflict with other libraries
- Less explicit control

### Decision: Option A (Wrapper Functions)

Provides explicit control while remaining compatible with existing Ray patterns.

---

## Design

### Module Structure

```
src/dimtensor/ray/
├── __init__.py         # Public exports
├── serialization.py    # DimArray/DimTensor serialization for Ray
├── remote.py           # Unit-aware remote function decorators
├── data.py             # Ray Data integration (DimDataset)
├── train.py            # Ray Train integration utilities
└── actors.py           # Unit-aware actor patterns
```

### Phase 1: Serialization (Tasks #204-205)

Custom serialization for Ray object store.

```python
# ray/serialization.py

import ray
from ..core.dimarray import DimArray
from ..io.json import to_dict, from_dict

def serialize_dimarray(arr: DimArray) -> dict:
    """Serialize DimArray for Ray object store."""
    return {
        "__dimarray__": True,
        "data": arr._data,  # numpy array - Ray handles efficiently
        "unit": {
            "symbol": arr.unit.symbol,
            "dimension": arr.dimension.as_tuple(),
            "scale": arr.unit.scale,
        },
        "uncertainty": arr._uncertainty,
    }

def deserialize_dimarray(d: dict) -> DimArray:
    """Deserialize DimArray from Ray object store."""
    from ..core.dimensions import Dimension
    from ..core.units import Unit
    from fractions import Fraction

    dim = Dimension(*[Fraction(x) for x in d["unit"]["dimension"]])
    unit = Unit(d["unit"]["symbol"], dim, d["unit"]["scale"])
    return DimArray._from_data_and_unit(d["data"], unit, d["uncertainty"])

def register_serializers():
    """Register dimtensor serializers with Ray.

    Call this once at startup to enable automatic serialization.
    """
    # Register custom serializer
    ray.util.register_serializer(
        DimArray,
        serializer=serialize_dimarray,
        deserializer=deserialize_dimarray,
    )
```

### Phase 2: Remote Functions (Tasks #206-208)

Unit-aware remote function decorator.

```python
# ray/remote.py

import functools
from typing import Callable, Any
import ray
from ..core.dimarray import DimArray
from .serialization import serialize_dimarray, deserialize_dimarray

def dim_remote(*args, **kwargs):
    """Decorator for unit-aware remote functions.

    Wraps @ray.remote to automatically handle DimArray serialization.

    Example:
        @dim_remote(num_cpus=2)
        def compute_energy(mass: DimArray, velocity: DimArray) -> DimArray:
            return 0.5 * mass * velocity**2

        # Usage
        result_ref = compute_energy.remote(mass, velocity)
        energy = ray.get(result_ref)  # Returns DimArray with units
    """
    def decorator(fn: Callable) -> ray.remote_function.RemoteFunction:
        @functools.wraps(fn)
        def wrapped(*fn_args, **fn_kwargs):
            # Deserialize DimArray arguments
            args_converted = [
                deserialize_dimarray(a) if isinstance(a, dict) and a.get("__dimarray__") else a
                for a in fn_args
            ]
            kwargs_converted = {
                k: deserialize_dimarray(v) if isinstance(v, dict) and v.get("__dimarray__") else v
                for k, v in fn_kwargs.items()
            }

            # Call original function
            result = fn(*args_converted, **kwargs_converted)

            # Serialize DimArray results
            if isinstance(result, DimArray):
                return serialize_dimarray(result)
            elif isinstance(result, tuple):
                return tuple(
                    serialize_dimarray(r) if isinstance(r, DimArray) else r
                    for r in result
                )
            return result

        # Apply ray.remote
        remote_fn = ray.remote(*args, **kwargs)(wrapped)

        # Add convenience method to automatically deserialize results
        original_remote = remote_fn.remote
        def smart_remote(*fn_args, **fn_kwargs):
            # Serialize DimArray arguments before sending
            args_serialized = [
                serialize_dimarray(a) if isinstance(a, DimArray) else a
                for a in fn_args
            ]
            kwargs_serialized = {
                k: serialize_dimarray(v) if isinstance(v, DimArray) else v
                for k, v in fn_kwargs.items()
            }
            return original_remote(*args_serialized, **kwargs_serialized)

        remote_fn.remote = smart_remote
        return remote_fn

    # Handle @dim_remote and @dim_remote() syntax
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return decorator(args[0])
    return decorator


def dim_get(object_ref):
    """Get result from remote function, deserializing DimArrays.

    Example:
        result = dim_get(compute_energy.remote(mass, velocity))
    """
    result = ray.get(object_ref)

    if isinstance(result, dict) and result.get("__dimarray__"):
        return deserialize_dimarray(result)
    elif isinstance(result, (list, tuple)):
        converted = [
            deserialize_dimarray(r) if isinstance(r, dict) and r.get("__dimarray__") else r
            for r in result
        ]
        return type(result)(converted)
    return result
```

### Phase 3: Ray Data Integration (Tasks #209-212)

Unit-aware data pipelines.

```python
# ray/data.py

from typing import Callable, Any, Iterator
import numpy as np
import ray.data
from ..core.dimarray import DimArray
from ..core.units import Unit

class DimDataset:
    """Unit-aware wrapper for Ray Dataset.

    Tracks physical units through Ray Data transformations.

    Example:
        # Create from DimArrays
        ds = DimDataset.from_dimarrays({
            "position": positions,  # DimArray with units.m
            "velocity": velocities,  # DimArray with units.m / units.s
        })

        # Transform (preserves units)
        ds = ds.map_batches(lambda batch: {
            "position": batch["position"] * 2,
            "velocity": batch["velocity"],
        })

        # Convert back to DimArrays
        result = ds.to_dimarrays()
    """

    def __init__(
        self,
        dataset: ray.data.Dataset,
        units: dict[str, Unit],
    ):
        self._dataset = dataset
        self._units = units

    @classmethod
    def from_dimarrays(cls, arrays: dict[str, DimArray]) -> "DimDataset":
        """Create DimDataset from dictionary of DimArrays.

        Args:
            arrays: Dictionary mapping column names to DimArrays.

        Returns:
            DimDataset with unit tracking.
        """
        # Extract numpy arrays and units
        data = {name: arr.data for name, arr in arrays.items()}
        units = {name: arr.unit for name, arr in arrays.items()}

        # Create Ray Dataset
        # Convert dict of arrays to list of dicts (rows)
        n_rows = len(next(iter(data.values())))
        rows = [
            {name: data[name][i] for name in data}
            for i in range(n_rows)
        ]
        dataset = ray.data.from_items(rows)

        return cls(dataset, units)

    @classmethod
    def from_numpy_refs(
        cls,
        refs: dict[str, ray.ObjectRef],
        units: dict[str, Unit],
    ) -> "DimDataset":
        """Create from Ray object references to numpy arrays.

        Efficient for large arrays already in Ray object store.
        """
        # Use Ray Data's from_numpy_refs for each column
        # This is a simplified version; full implementation would be more complex
        raise NotImplementedError("Use from_dimarrays for now")

    def map(self, fn: Callable) -> "DimDataset":
        """Apply function to each row, preserving units."""
        new_dataset = self._dataset.map(fn)
        return DimDataset(new_dataset, self._units.copy())

    def map_batches(
        self,
        fn: Callable,
        batch_format: str = "numpy",
        **kwargs,
    ) -> "DimDataset":
        """Apply function to batches, preserving units.

        Args:
            fn: Function to apply to each batch.
            batch_format: "numpy" or "pandas".
            **kwargs: Additional arguments to Ray Data map_batches.

        Returns:
            Transformed DimDataset.
        """
        new_dataset = self._dataset.map_batches(fn, batch_format=batch_format, **kwargs)
        return DimDataset(new_dataset, self._units.copy())

    def filter(self, fn: Callable) -> "DimDataset":
        """Filter rows, preserving units."""
        new_dataset = self._dataset.filter(fn)
        return DimDataset(new_dataset, self._units.copy())

    def to_dimarrays(self) -> dict[str, DimArray]:
        """Convert back to dictionary of DimArrays.

        Collects all data to driver node.
        """
        # Collect all rows
        rows = self._dataset.take_all()

        # Convert to column format
        if not rows:
            return {name: DimArray([], unit) for name, unit in self._units.items()}

        result = {}
        for name, unit in self._units.items():
            data = np.array([row[name] for row in rows])
            result[name] = DimArray._from_data_and_unit(data, unit)

        return result

    def iter_batches(
        self,
        batch_size: int = 256,
    ) -> Iterator[dict[str, DimArray]]:
        """Iterate over batches as DimArrays.

        Memory-efficient iteration for large datasets.
        """
        for batch in self._dataset.iter_batches(batch_size=batch_size, batch_format="numpy"):
            yield {
                name: DimArray._from_data_and_unit(batch[name], self._units[name])
                for name in self._units
            }

    @property
    def units(self) -> dict[str, Unit]:
        """Get unit mapping."""
        return self._units.copy()

    @property
    def count(self) -> int:
        """Number of rows."""
        return self._dataset.count()

    def __repr__(self) -> str:
        units_str = ", ".join(f"{k}: {v.symbol}" for k, v in self._units.items())
        return f"DimDataset(count={self.count()}, units={{{units_str}}})"
```

### Phase 4: Ray Train Integration (Tasks #213-216)

Unit-aware distributed training utilities.

```python
# ray/train.py

from typing import Callable, Any
import torch
import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

from ..core.dimarray import DimArray
from ..torch.dimtensor import DimTensor
from ..torch.scaler import DimScaler
from .data import DimDataset

class DimTrainContext:
    """Context manager for unit-aware distributed training.

    Handles non-dimensionalization and re-dimensionalization
    in distributed training context.

    Example:
        def train_func(config):
            ctx = DimTrainContext(
                scaler=config["scaler"],
                input_units={"x": units.m, "y": units.m/units.s},
                output_units={"pred": units.J},
            )

            shard = train.get_dataset_shard("train")
            for batch in ctx.iter_batches(shard):
                # batch contains scaled torch.Tensors
                pred = model(batch["x"])
                loss = criterion(pred, batch["y"])
                ...
    """

    def __init__(
        self,
        scaler: DimScaler,
        input_units: dict[str, "Unit"] | None = None,
        output_units: dict[str, "Unit"] | None = None,
    ):
        self.scaler = scaler
        self.input_units = input_units or {}
        self.output_units = output_units or {}

    def iter_batches(
        self,
        dataset_shard,
        batch_size: int = 32,
        dtype: torch.dtype = torch.float32,
    ):
        """Iterate over dataset shard with automatic scaling.

        Args:
            dataset_shard: Ray Train dataset shard.
            batch_size: Batch size.
            dtype: Torch dtype for tensors.

        Yields:
            Dict of scaled torch.Tensors (dimensionless).
        """
        for batch in dataset_shard.iter_torch_batches(batch_size=batch_size, dtypes=dtype):
            # Scale each column
            scaled_batch = {}
            for name, tensor in batch.items():
                if name in self.input_units:
                    unit = self.input_units[name]
                    # Create DimTensor, scale, extract raw tensor
                    dim_tensor = DimTensor._from_tensor_and_unit(tensor, unit)
                    scaled_batch[name] = self.scaler.transform(dim_tensor)
                else:
                    scaled_batch[name] = tensor
            yield scaled_batch

    def inverse_transform(
        self,
        tensor: torch.Tensor,
        name: str,
    ) -> DimTensor:
        """Convert model output back to physical units.

        Args:
            tensor: Raw model output.
            name: Output name (to look up unit).

        Returns:
            DimTensor with physical units.
        """
        if name not in self.output_units:
            raise KeyError(f"Unknown output: {name}")
        return self.scaler.inverse_transform(tensor, self.output_units[name])


def create_dim_trainer(
    train_func: Callable,
    datasets: dict[str, DimDataset],
    scaler: DimScaler,
    num_workers: int = 2,
    use_gpu: bool = False,
    **kwargs,
) -> TorchTrainer:
    """Create a TorchTrainer with unit-aware datasets.

    Args:
        train_func: Training function.
        datasets: Dict of DimDatasets.
        scaler: DimScaler for non-dimensionalization.
        num_workers: Number of training workers.
        use_gpu: Whether to use GPUs.
        **kwargs: Additional TorchTrainer arguments.

    Returns:
        Configured TorchTrainer.
    """
    # Convert DimDatasets to Ray Datasets, pass units via config
    ray_datasets = {name: ds._dataset for name, ds in datasets.items()}
    units = {name: ds.units for name, ds in datasets.items()}

    # Serialize scaler for workers
    # (DimScaler is simple enough to pickle)

    config = kwargs.pop("train_loop_config", {})
    config["_dimtensor_scaler"] = scaler
    config["_dimtensor_units"] = units

    return TorchTrainer(
        train_func,
        datasets=ray_datasets,
        train_loop_config=config,
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        **kwargs,
    )
```

### Phase 5: Actors (Tasks #217-218)

Unit-aware actor patterns for stateful computations.

```python
# ray/actors.py

import ray
from ..core.dimarray import DimArray
from .serialization import serialize_dimarray, deserialize_dimarray

def dim_actor(*args, **kwargs):
    """Decorator for unit-aware Ray actors.

    Example:
        @dim_actor
        class PhysicsSimulator:
            def __init__(self, dt: DimArray):
                self.dt = dt  # time step with units

            def step(self, position: DimArray, velocity: DimArray) -> DimArray:
                return position + velocity * self.dt
    """
    def decorator(cls):
        # Wrap methods to handle DimArray serialization
        original_init = cls.__init__

        def wrapped_init(self, *init_args, **init_kwargs):
            # Deserialize DimArray arguments
            args_converted = [
                deserialize_dimarray(a) if isinstance(a, dict) and a.get("__dimarray__") else a
                for a in init_args
            ]
            kwargs_converted = {
                k: deserialize_dimarray(v) if isinstance(v, dict) and v.get("__dimarray__") else v
                for k, v in init_kwargs.items()
            }
            original_init(self, *args_converted, **kwargs_converted)

        cls.__init__ = wrapped_init

        # Wrap other methods similarly
        for name, method in list(cls.__dict__.items()):
            if callable(method) and not name.startswith("_"):
                cls.__dict__[name] = _wrap_actor_method(method)

        return ray.remote(*args, **kwargs)(cls)

    if len(args) == 1 and isinstance(args[0], type) and not kwargs:
        return decorator(args[0])
    return decorator


def _wrap_actor_method(method):
    """Wrap actor method for DimArray handling."""
    import functools

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        args_converted = [
            deserialize_dimarray(a) if isinstance(a, dict) and a.get("__dimarray__") else a
            for a in args
        ]
        kwargs_converted = {
            k: deserialize_dimarray(v) if isinstance(v, dict) and v.get("__dimarray__") else v
            for k, v in kwargs.items()
        }

        result = method(self, *args_converted, **kwargs_converted)

        if isinstance(result, DimArray):
            return serialize_dimarray(result)
        return result

    return wrapped
```

---

## Implementation Steps

### Phase 1: Serialization Foundation
1. [ ] **Task #204**: Create `ray/serialization.py` with DimArray serializers
2. [ ] **Task #205**: Add DimTensor serialization support

### Phase 2: Remote Functions
3. [ ] **Task #206**: Create `ray/remote.py` with `@dim_remote` decorator
4. [ ] **Task #207**: Implement `dim_get()` and `dim_put()` helpers
5. [ ] **Task #208**: Add support for nested DimArray structures

### Phase 3: Ray Data Integration
6. [ ] **Task #209**: Create `ray/data.py` with `DimDataset` class
7. [ ] **Task #210**: Implement `from_dimarrays()` factory
8. [ ] **Task #211**: Implement `map()`, `map_batches()`, `filter()` with unit preservation
9. [ ] **Task #212**: Implement `iter_batches()` for memory-efficient iteration

### Phase 4: Ray Train Integration
10. [ ] **Task #213**: Create `ray/train.py` with `DimTrainContext`
11. [ ] **Task #214**: Implement `create_dim_trainer()` factory
12. [ ] **Task #215**: Add checkpoint utilities with unit metadata
13. [ ] **Task #216**: Add metric reporting with physical units

### Phase 5: Actors
14. [ ] **Task #217**: Create `ray/actors.py` with `@dim_actor` decorator
15. [ ] **Task #218**: Add actor pool utilities

### Phase 6: Testing and Documentation
16. [ ] **Task #219**: Unit tests for serialization
17. [ ] **Task #220**: Integration tests for remote functions
18. [ ] **Task #221**: Integration tests for Ray Data
19. [ ] **Task #222**: Integration tests for Ray Train
20. [ ] **Task #223**: Documentation and examples

---

## Files to Create

| File | Purpose | Lines (est) |
|------|---------|-------------|
| `ray/__init__.py` | Public exports | ~30 |
| `ray/serialization.py` | DimArray/DimTensor serialization | ~100 |
| `ray/remote.py` | Unit-aware remote functions | ~150 |
| `ray/data.py` | Ray Data integration | ~200 |
| `ray/train.py` | Ray Train integration | ~200 |
| `ray/actors.py` | Unit-aware actors | ~100 |
| `tests/test_ray.py` | All Ray integration tests | ~300 |

---

## Testing Strategy

### Unit Tests
- [ ] Serialization round-trip for DimArray
- [ ] Serialization round-trip for DimTensor
- [ ] Unit preservation through remote calls
- [ ] Unit tracking through data transformations

### Integration Tests
- [ ] Multi-worker remote function execution
- [ ] Distributed training with physical units
- [ ] Large dataset pipeline with units
- [ ] Actor state with units

### Manual Verification
- [ ] Performance overhead measurement
- [ ] Memory usage with large DimArrays
- [ ] Cluster deployment testing

---

## Risks / Edge Cases

| Risk | Mitigation |
|------|------------|
| **Serialization overhead** | Use efficient numpy serialization; units are small metadata |
| **Version compatibility** | Test with Ray 2.x; document minimum version |
| **Unit mismatch across workers** | Validate units in serialization/deserialization |
| **Large array performance** | Use `ray.put()` for large arrays; pass refs |
| **GPU tensor serialization** | Convert to CPU before serialization |
| **Checkpoint format changes** | Version checkpoint format; add migration |

---

## API Examples

### Remote Functions

```python
import ray
from dimtensor import DimArray, units
from dimtensor.ray import dim_remote, dim_get

ray.init()

@dim_remote(num_cpus=2)
def compute_kinetic_energy(mass: DimArray, velocity: DimArray) -> DimArray:
    """Compute kinetic energy = 0.5 * m * v^2"""
    return 0.5 * mass * velocity**2

# Usage
mass = DimArray([1.0, 2.0, 3.0], units.kg)
velocity = DimArray([10.0, 20.0, 30.0], units.m / units.s)

# Submit remote computation
ref = compute_kinetic_energy.remote(mass, velocity)

# Get result with units
energy = dim_get(ref)
print(energy)  # [50.0, 400.0, 1350.0] J
```

### Ray Data Pipeline

```python
from dimtensor import DimArray, units
from dimtensor.ray import DimDataset

# Create dataset from DimArrays
positions = DimArray(np.random.randn(10000, 3), units.m)
velocities = DimArray(np.random.randn(10000, 3), units.m / units.s)

ds = DimDataset.from_dimarrays({
    "position": positions,
    "velocity": velocities,
})

# Transform (units preserved)
ds = ds.map_batches(lambda batch: {
    "position": batch["position"] * 2,  # Still in meters
    "velocity": batch["velocity"],
})

# Iterate with units
for batch in ds.iter_batches(batch_size=256):
    print(batch["position"].unit)  # m
    print(batch["velocity"].unit)  # m/s
```

### Distributed Training

```python
from dimtensor import DimArray, units
from dimtensor.torch import DimScaler
from dimtensor.ray import DimDataset, DimTrainContext, create_dim_trainer

# Prepare data
scaler = DimScaler(method="characteristic")
scaler.fit(train_positions, train_velocities, train_energies)

train_ds = DimDataset.from_dimarrays({
    "position": train_positions,
    "velocity": train_velocities,
    "energy": train_energies,
})

def train_func(config):
    ctx = DimTrainContext(
        scaler=config["_dimtensor_scaler"],
        input_units={"position": units.m, "velocity": units.m/units.s},
        output_units={"energy": units.J},
    )

    model = create_model()
    model = ray.train.torch.prepare_model(model)

    shard = ray.train.get_dataset_shard("train")
    for batch in ctx.iter_batches(shard):
        # batch contains scaled (dimensionless) tensors
        pred = model(batch["position"], batch["velocity"])
        loss = criterion(pred, batch["energy"])
        ...

trainer = create_dim_trainer(
    train_func,
    datasets={"train": train_ds},
    scaler=scaler,
    num_workers=4,
    use_gpu=True,
)
result = trainer.fit()
```

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Documentation created in `docs/guide/ray.md`
- [ ] Example notebook in `examples/`
- [ ] CONTINUITY.md updated
- [ ] API reference documented

---

## Dependencies

```toml
# In pyproject.toml [project.optional-dependencies]
ray = [
    "ray[default]>=2.0.0",
    "ray[train]>=2.0.0",
    "ray[data]>=2.0.0",
]
```

---

## References

- [Ray Documentation](https://docs.ray.io/en/latest/)
- [Ray Core API](https://docs.ray.io/en/latest/ray-core/walkthrough.html)
- [Ray Train](https://docs.ray.io/en/latest/train/getting-started-pytorch.html)
- [Ray Data](https://docs.ray.io/en/latest/data/data.html)
- [Custom Serialization](https://docs.ray.io/en/latest/ray-core/objects/serialization.html)

---

## Notes / Log

**[2026-01-10]** - Initial plan created based on research of Ray patterns and existing dimtensor architecture. Recommended phased approach starting with serialization foundation.

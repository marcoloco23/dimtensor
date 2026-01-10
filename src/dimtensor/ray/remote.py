"""Unit-aware remote function decorators for Ray.

Provides @dim_remote decorator and dim_get/dim_put helpers for
distributed computing with physical units.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar, overload

from ..core.dimarray import DimArray
from .serialization import (
    serialize_dimarray,
    deserialize_dimarray,
    serialize_dimtensor,
    deserialize_dimtensor,
    is_serialized_dimarray,
    is_serialized_dimtensor,
)

F = TypeVar("F", bound=Callable[..., Any])


def _serialize_value(value: Any) -> Any:
    """Serialize DimArray/DimTensor values for remote transfer."""
    if isinstance(value, DimArray):
        return serialize_dimarray(value)

    # Check for DimTensor without importing torch at module level
    try:
        from ..torch.dimtensor import DimTensor

        if isinstance(value, DimTensor):
            return serialize_dimtensor(value)
    except ImportError:
        pass

    return value


def _deserialize_value(value: Any) -> Any:
    """Deserialize DimArray/DimTensor values from remote transfer."""
    if is_serialized_dimarray(value):
        return deserialize_dimarray(value)
    if is_serialized_dimtensor(value):
        return deserialize_dimtensor(value)
    return value


def _serialize_args(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Serialize DimArray/DimTensor arguments."""
    serialized_args = tuple(_serialize_value(a) for a in args)
    serialized_kwargs = {k: _serialize_value(v) for k, v in kwargs.items()}
    return serialized_args, serialized_kwargs


def _deserialize_args(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Deserialize DimArray/DimTensor arguments."""
    deserialized_args = tuple(_deserialize_value(a) for a in args)
    deserialized_kwargs = {k: _deserialize_value(v) for k, v in kwargs.items()}
    return deserialized_args, deserialized_kwargs


def _serialize_result(result: Any) -> Any:
    """Serialize DimArray/DimTensor results."""
    if isinstance(result, (list, tuple)):
        return type(result)(_serialize_value(r) for r in result)
    if isinstance(result, dict):
        return {k: _serialize_value(v) for k, v in result.items()}
    return _serialize_value(result)


def _deserialize_result(result: Any) -> Any:
    """Deserialize DimArray/DimTensor results."""
    # First check if this dict IS a serialized DimArray/DimTensor
    if is_serialized_dimarray(result):
        return deserialize_dimarray(result)
    if is_serialized_dimtensor(result):
        return deserialize_dimtensor(result)
    # Then handle collections
    if isinstance(result, (list, tuple)):
        return type(result)(_deserialize_result(r) for r in result)
    if isinstance(result, dict):
        return {k: _deserialize_result(v) for k, v in result.items()}
    return _deserialize_value(result)


@overload
def dim_remote(fn: F) -> Any: ...


@overload
def dim_remote(
    *,
    num_cpus: float | None = None,
    num_gpus: float | None = None,
    memory: int | None = None,
    max_retries: int = 0,
    **kwargs: Any,
) -> Callable[[F], Any]: ...


def dim_remote(
    fn: F | None = None,
    *,
    num_cpus: float | None = None,
    num_gpus: float | None = None,
    memory: int | None = None,
    max_retries: int = 0,
    **kwargs: Any,
) -> Any:
    """Decorator for unit-aware remote functions.

    Wraps @ray.remote to automatically handle DimArray and DimTensor
    serialization. Decorated functions can accept and return dimensional
    types directly.

    Args:
        fn: Function to decorate.
        num_cpus: Number of CPUs for the task.
        num_gpus: Number of GPUs for the task.
        memory: Memory requirement in bytes.
        max_retries: Number of retries on failure.
        **kwargs: Additional Ray remote options.

    Returns:
        Ray remote function with unit-aware serialization.

    Example:
        >>> import ray
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.ray import dim_remote, dim_get
        >>>
        >>> @dim_remote(num_cpus=2)
        ... def compute_energy(mass: DimArray, velocity: DimArray) -> DimArray:
        ...     return 0.5 * mass * velocity**2
        >>>
        >>> mass = DimArray([1.0, 2.0], units.kg)
        >>> velocity = DimArray([10.0, 20.0], units.m / units.s)
        >>> ref = compute_energy.remote(mass, velocity)
        >>> energy = dim_get(ref)
        >>> print(energy.unit)  # J

        >>> # Also supports @dim_remote without parentheses
        >>> @dim_remote
        ... def simple_func(x: DimArray) -> DimArray:
        ...     return x * 2
    """
    import ray

    def decorator(func: Callable[..., Any]) -> Any:
        @functools.wraps(func)
        def wrapped(*args: Any, **kw: Any) -> Any:
            # Deserialize DimArray/DimTensor arguments
            deserialized_args, deserialized_kwargs = _deserialize_args(args, kw)

            # Call original function
            result = func(*deserialized_args, **deserialized_kwargs)

            # Serialize DimArray/DimTensor results
            return _serialize_result(result)

        # Build Ray remote options
        remote_options: dict[str, Any] = {}
        if num_cpus is not None:
            remote_options["num_cpus"] = num_cpus
        if num_gpus is not None:
            remote_options["num_gpus"] = num_gpus
        if memory is not None:
            remote_options["memory"] = memory
        if max_retries > 0:
            remote_options["max_retries"] = max_retries
        remote_options.update(kwargs)

        # Apply ray.remote
        if remote_options:
            remote_fn = ray.remote(**remote_options)(wrapped)
        else:
            remote_fn = ray.remote(wrapped)

        # Wrap .remote() to auto-serialize arguments
        original_remote = remote_fn.remote

        def smart_remote(*fn_args: Any, **fn_kwargs: Any) -> Any:
            serialized_args, serialized_kwargs = _serialize_args(fn_args, fn_kwargs)
            return original_remote(*serialized_args, **serialized_kwargs)

        remote_fn.remote = smart_remote

        return remote_fn

    # Handle @dim_remote and @dim_remote() syntax
    if fn is not None:
        return decorator(fn)
    return decorator


def dim_get(
    object_refs: Any,
    *,
    timeout: float | None = None,
) -> Any:
    """Get results from remote functions, deserializing DimArrays/DimTensors.

    This is a unit-aware wrapper around ray.get() that automatically
    deserializes dimensional types in the results.

    Args:
        object_refs: Ray ObjectRef or list of ObjectRefs.
        timeout: Timeout in seconds.

    Returns:
        Result(s) with DimArray/DimTensor types reconstructed.

    Example:
        >>> ref = compute_energy.remote(mass, velocity)
        >>> energy = dim_get(ref)
        >>> print(energy.unit)  # J

        >>> # Multiple results
        >>> refs = [compute.remote(x) for x in data]
        >>> results = dim_get(refs)
    """
    import ray

    # Get raw results from Ray
    if timeout is not None:
        raw_results = ray.get(object_refs, timeout=timeout)
    else:
        raw_results = ray.get(object_refs)

    # Handle list of results
    if isinstance(raw_results, list):
        return [_deserialize_result(r) for r in raw_results]

    return _deserialize_result(raw_results)


def dim_put(
    value: Any,
    *,
    _owner: Any = None,
) -> Any:
    """Put a value into the Ray object store with DimArray/DimTensor support.

    This is a unit-aware wrapper around ray.put() that handles
    serialization of dimensional types.

    Args:
        value: Value to put in object store (can be DimArray/DimTensor).
        _owner: Actor to own the object (passed to ray.put).

    Returns:
        Ray ObjectRef.

    Example:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.ray import dim_put, dim_get
        >>>
        >>> data = DimArray([1.0, 2.0, 3.0], units.m)
        >>> ref = dim_put(data)
        >>> retrieved = dim_get(ref)
    """
    import ray

    serialized = _serialize_value(value)

    if _owner is not None:
        return ray.put(serialized, _owner=_owner)
    return ray.put(serialized)

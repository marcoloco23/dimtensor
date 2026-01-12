"""Input validation utilities for serverless handlers.

Provides validation decorators and utilities for checking DimArray inputs
in serverless functions.
"""

from __future__ import annotations

import functools
from typing import Callable, Any

from ..core.dimarray import DimArray
from ..core.dimensions import Dimension
from ..errors import DimensionError


def require_dimension(
    param_name: str, required_dimension: Dimension
) -> Callable:
    """Decorator to validate that a parameter has a specific dimension.

    Args:
        param_name: Name of the parameter to validate.
        required_dimension: Expected dimension.

    Returns:
        Decorated function that validates dimension before execution.

    Example:
        >>> from dimtensor import units
        >>> from dimtensor.core.dimensions import Dimension
        >>>
        >>> @require_dimension("velocity", units.m.dimension / units.s.dimension)
        ... def compute_momentum(event, context):
        ...     mass = event["mass"]
        ...     velocity = event["velocity"]  # Must have [L T^-1] dimension
        ...     return mass * velocity
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(event: dict[str, Any], context: Any = None) -> Any:
            if param_name in event:
                value = event[param_name]
                if isinstance(value, DimArray):
                    if value.dimension != required_dimension:
                        raise DimensionError(
                            f"Parameter '{param_name}' has dimension {value.dimension}, "
                            f"but expected {required_dimension}"
                        )
            return fn(event, context)

        return wrapper

    return decorator


def require_positive(param_name: str) -> Callable:
    """Decorator to validate that a parameter is positive.

    Args:
        param_name: Name of the parameter to validate.

    Returns:
        Decorated function that validates positivity.

    Example:
        >>> @require_positive("mass")
        ... def compute_energy(event, context):
        ...     mass = event["mass"]  # Must be positive
        ...     return mass * 9e16  # E = mc^2
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(event: dict[str, Any], context: Any = None) -> Any:
            if param_name in event:
                value = event[param_name]
                if isinstance(value, DimArray):
                    if (value._data <= 0).any():
                        raise ValueError(
                            f"Parameter '{param_name}' must be positive, "
                            f"but contains non-positive values"
                        )
            return fn(event, context)

        return wrapper

    return decorator


def require_dimensionless(param_name: str) -> Callable:
    """Decorator to validate that a parameter is dimensionless.

    Args:
        param_name: Name of the parameter to validate.

    Returns:
        Decorated function that validates dimensionlessness.

    Example:
        >>> @require_dimensionless("angle")
        ... def compute_sin(event, context):
        ...     angle = event["angle"]  # Must be dimensionless (radians)
        ...     import numpy as np
        ...     return np.sin(angle)
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(event: dict[str, Any], context: Any = None) -> Any:
            if param_name in event:
                value = event[param_name]
                if isinstance(value, DimArray):
                    from ..core.units import dimensionless

                    if value.dimension != dimensionless.dimension:
                        raise DimensionError(
                            f"Parameter '{param_name}' must be dimensionless, "
                            f"but has dimension {value.dimension}"
                        )
            return fn(event, context)

        return wrapper

    return decorator


def require_params(*param_names: str) -> Callable:
    """Decorator to validate that required parameters are present.

    Args:
        param_names: Names of required parameters.

    Returns:
        Decorated function that validates parameter presence.

    Example:
        >>> @require_params("mass", "velocity")
        ... def compute_momentum(event, context):
        ...     # Both mass and velocity must be present
        ...     return event["mass"] * event["velocity"]
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(event: dict[str, Any], context: Any = None) -> Any:
            missing = [name for name in param_names if name not in event]
            if missing:
                raise ValueError(
                    f"Missing required parameters: {', '.join(missing)}"
                )
            return fn(event, context)

        return wrapper

    return decorator


def validate_shape(param_name: str, expected_shape: tuple[int, ...]) -> Callable:
    """Decorator to validate array shape.

    Args:
        param_name: Name of the parameter to validate.
        expected_shape: Expected shape tuple. Use -1 for any dimension.

    Returns:
        Decorated function that validates shape.

    Example:
        >>> @validate_shape("position", (3,))  # Must be 3D vector
        ... def compute_distance(event, context):
        ...     position = event["position"]
        ...     import numpy as np
        ...     return np.linalg.norm(position._data)
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(event: dict[str, Any], context: Any = None) -> Any:
            if param_name in event:
                value = event[param_name]
                if isinstance(value, DimArray):
                    actual_shape = value.shape
                    if len(actual_shape) != len(expected_shape):
                        raise ValueError(
                            f"Parameter '{param_name}' has {len(actual_shape)} dimensions, "
                            f"but expected {len(expected_shape)}"
                        )
                    for i, (actual, expected) in enumerate(
                        zip(actual_shape, expected_shape)
                    ):
                        if expected != -1 and actual != expected:
                            raise ValueError(
                                f"Parameter '{param_name}' has shape {actual_shape}, "
                                f"but expected {expected_shape}"
                            )
            return fn(event, context)

        return wrapper

    return decorator

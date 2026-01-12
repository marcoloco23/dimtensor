"""Serverless deployment support for AWS Lambda and Google Cloud Functions.

This module provides decorators and utilities for deploying dimtensor-based
physics computations to serverless platforms.

Key Features:
- Automatic DimArray serialization/deserialization
- Dimensional error handling
- Cold start optimization (lazy imports)
- Memory configuration warnings
- Input validation decorators

Example (AWS Lambda):
    >>> from dimtensor.serverless import lambda_physics
    >>> from dimtensor import DimArray, units
    >>>
    >>> @lambda_physics
    ... def compute_energy(event, context):
    ...     mass = event["mass"]  # Automatically deserialized DimArray
    ...     velocity = event["velocity"]
    ...     return 0.5 * mass * velocity**2  # Automatically serialized

Example (Google Cloud Functions):
    >>> from dimtensor.serverless import cf_physics
    >>> import functions_framework
    >>>
    >>> @functions_framework.http
    >>> @cf_physics
    ... def compute_force(request):
    ...     data = request.get_json()
    ...     mass = data["mass"]
    ...     acceleration = data["acceleration"]
    ...     return mass * acceleration
"""

from .aws import (
    lambda_handler,
    lambda_physics,
    lambda_ml,
    lambda_batch,
)

from .gcp import (
    cloud_function,
    cf_physics,
    cf_ml,
    cloud_event_function,
)

from .serialization import (
    serialize_for_http,
    deserialize_from_http,
    simple_response,
    error_response,
)

from .validation import (
    require_dimension,
    require_positive,
    require_dimensionless,
    require_params,
    validate_shape,
)

__all__ = [
    # AWS Lambda decorators
    "lambda_handler",
    "lambda_physics",
    "lambda_ml",
    "lambda_batch",
    # Google Cloud Functions decorators
    "cloud_function",
    "cf_physics",
    "cf_ml",
    "cloud_event_function",
    # Serialization utilities
    "serialize_for_http",
    "deserialize_from_http",
    "simple_response",
    "error_response",
    # Validation decorators
    "require_dimension",
    "require_positive",
    "require_dimensionless",
    "require_params",
    "validate_shape",
]

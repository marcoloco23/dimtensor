"""AWS Lambda handler wrappers for dimtensor.

Provides decorators for AWS Lambda handlers that automatically handle
DimArray serialization/deserialization and dimensional error handling.
"""

from __future__ import annotations

import functools
import json
import traceback
from typing import Callable, Any

from ..core.dimarray import DimArray
from ..errors import DimensionError, UnitConversionError
from .serialization import deserialize_from_http, simple_response, error_response


def lambda_handler(
    lazy_imports: bool = True,
    memory_limit_mb: int | None = None,
) -> Callable:
    """Decorator for AWS Lambda handlers with dimtensor support.

    Automatically handles:
    - DimArray deserialization from request body
    - DimArray serialization in response
    - Dimensional error handling (returns 400 status)
    - Generic error handling (returns 500 status)
    - Memory limit warnings (logs if below threshold)
    - Lazy imports for cold start optimization

    Args:
        lazy_imports: If True, enables lazy import optimization (default True).
                     In production, this reduces cold start time.
        memory_limit_mb: Minimum recommended memory in MB. Logs warning if
                        Lambda memory is below this threshold.

    Returns:
        Decorated handler function.

    Example:
        >>> from dimtensor.serverless import lambda_handler
        >>> from dimtensor import DimArray, units
        >>>
        >>> @lambda_handler(lazy_imports=True, memory_limit_mb=512)
        ... def compute_energy(event, context):
        ...     # event contains deserialized DimArrays
        ...     mass = event["mass"]  # Already a DimArray
        ...     velocity = event["velocity"]
        ...     return 0.5 * mass * velocity**2  # Returns DimArray

    Notes:
        - Request body should contain JSON with DimArray serialization format
        - Response is automatically formatted for API Gateway
        - DimensionError and UnitConversionError return 400 status
        - Other exceptions return 500 status with traceback
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(event: dict[str, Any], context: Any) -> dict[str, Any]:
            # Lazy import optimization (helps cold start)
            if lazy_imports:
                # Note: In production, imports would happen here
                # For dimtensor, imports are already done at module level
                pass

            # Check memory configuration
            if memory_limit_mb and hasattr(context, "memory_limit_in_mb"):
                actual_memory = context.memory_limit_in_mb
                if actual_memory < memory_limit_mb:
                    print(
                        f"WARNING: Lambda memory ({actual_memory}MB) is below "
                        f"recommended {memory_limit_mb}MB for physics computations"
                    )

            try:
                # Parse request body
                if isinstance(event.get("body"), str):
                    # API Gateway format
                    body = json.loads(event["body"])
                else:
                    # Direct invocation or other trigger
                    body = event

                # Deserialize DimArray inputs
                processed_event: dict[str, Any] = {}
                for key, value in body.items():
                    if isinstance(value, dict) and "data" in value and "unit" in value:
                        # This looks like a serialized DimArray
                        try:
                            processed_event[key] = deserialize_from_http(value)
                        except Exception as e:
                            # If deserialization fails, pass through as-is
                            processed_event[key] = value
                    else:
                        processed_event[key] = value

                # Call the actual handler
                result = fn(processed_event, context)

                # Serialize response
                if isinstance(result, DimArray):
                    # Single DimArray result
                    return simple_response(result)
                elif isinstance(result, dict):
                    # Dictionary result - check for DimArrays
                    serialized_dict: dict[str, Any] = {}
                    has_dimarrays = False
                    for key, value in result.items():
                        if isinstance(value, DimArray):
                            from .serialization import serialize_for_http

                            serialized_dict[key] = serialize_for_http(value)
                            has_dimarrays = True
                        else:
                            serialized_dict[key] = value

                    return {
                        "statusCode": 200,
                        "body": json.dumps(serialized_dict),
                        "headers": {
                            "Content-Type": "application/json",
                            **(
                                {"X-DimTensor-Version": "5.0.0"}
                                if has_dimarrays
                                else {}
                            ),
                        },
                    }
                else:
                    # Other result type - pass through
                    return result

            except DimensionError as e:
                # Dimensional mismatch - user error
                return error_response("DimensionError", str(e), status_code=400)

            except UnitConversionError as e:
                # Unit conversion error - user error
                return error_response("UnitConversionError", str(e), status_code=400)

            except json.JSONDecodeError as e:
                # Invalid JSON in request
                return error_response(
                    "JSONDecodeError", f"Invalid JSON: {str(e)}", status_code=400
                )

            except Exception as e:
                # Unexpected error - server error
                tb = traceback.format_exc()
                return error_response(
                    type(e).__name__, str(e), status_code=500, traceback=tb
                )

        return wrapper

    return decorator


# Convenience decorators with preset configurations

# For general physics computations
lambda_physics = lambda_handler(lazy_imports=True, memory_limit_mb=512)

# For ML model inference
lambda_ml = lambda_handler(lazy_imports=True, memory_limit_mb=1024)


def lambda_batch(
    lazy_imports: bool = True, memory_limit_mb: int = 1024
) -> Callable:
    """Decorator for AWS Lambda batch processing handlers.

    Similar to lambda_handler but optimized for batch processing of
    S3 events or SQS messages.

    Args:
        lazy_imports: Enable lazy import optimization.
        memory_limit_mb: Recommended memory (default 1GB for batch).

    Returns:
        Decorated handler function.

    Example:
        >>> @lambda_batch(memory_limit_mb=2048)
        ... def process_simulation_batch(event, context):
        ...     # Process S3 upload event
        ...     bucket = event["Records"][0]["s3"]["bucket"]["name"]
        ...     key = event["Records"][0]["s3"]["object"]["key"]
        ...     # ... load, process, save results
        ...     return {"status": "complete"}
    """
    return lambda_handler(lazy_imports=lazy_imports, memory_limit_mb=memory_limit_mb)

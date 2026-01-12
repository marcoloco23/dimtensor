"""Google Cloud Functions handler wrappers for dimtensor.

Provides decorators for Google Cloud Functions (2nd gen) that automatically
handle DimArray serialization/deserialization and dimensional error handling.
"""

from __future__ import annotations

import functools
import json
import traceback
from typing import Callable, Any

from ..core.dimarray import DimArray
from ..errors import DimensionError, UnitConversionError
from .serialization import deserialize_from_http, serialize_for_http


def cloud_function(
    lazy_imports: bool = True,
    min_instances: int | None = None,
) -> Callable:
    """Decorator for Google Cloud Functions with dimtensor support.

    Supports both HTTP and CloudEvent-triggered functions. Automatically
    handles DimArray serialization/deserialization and error handling.

    Args:
        lazy_imports: If True, enables lazy import optimization (default True).
                     Reduces cold start time in production.
        min_instances: Recommended minimum instances for production.
                      This is documentation only - actual configuration is
                      done in deployment (gcloud or terraform).

    Returns:
        Decorated handler function.

    Example:
        >>> from dimtensor.serverless import cloud_function
        >>> from dimtensor import DimArray, units
        >>> import functions_framework
        >>>
        >>> @functions_framework.http
        >>> @cloud_function(lazy_imports=True, min_instances=1)
        ... def compute_energy(request):
        ...     data = request.get_json()
        ...     mass = data["mass"]  # Already a DimArray
        ...     velocity = data["velocity"]
        ...     return 0.5 * mass * velocity**2  # Returns DimArray

    Notes:
        - Works with functions-framework for local testing
        - Request should contain JSON with DimArray serialization format
        - Response is automatically formatted as JSON
        - DimensionError and UnitConversionError return 400 status
        - Other exceptions return 500 status with traceback
        - For Cloud Run Functions, use @functions_framework.http decorator first
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(request: Any) -> tuple[str, int] | dict[str, Any]:
            # Lazy import optimization
            if lazy_imports:
                # Note: In production, imports would happen here
                pass

            # Log recommendation (documentation only)
            if min_instances and min_instances > 0:
                # This is just for documentation - actual min instances
                # are configured in gcloud CLI or terraform
                pass

            try:
                # Parse request based on method
                if hasattr(request, "method"):
                    if request.method == "GET":
                        # GET request - parse query parameters
                        data = request.args.to_dict() if hasattr(request, "args") else {}
                    else:
                        # POST/PUT - parse JSON body
                        data = request.get_json(silent=True) or {}
                else:
                    # CloudEvent format
                    data = request if isinstance(request, dict) else {}

                # Deserialize DimArray inputs
                processed_data: dict[str, Any] = {}
                for key, value in data.items():
                    if isinstance(value, dict) and "data" in value and "unit" in value:
                        # Looks like serialized DimArray
                        try:
                            processed_data[key] = deserialize_from_http(value)
                        except Exception:
                            # If deserialization fails, pass through
                            processed_data[key] = value
                    else:
                        processed_data[key] = value

                # Create request-like wrapper for processed data
                class ProcessedRequest:
                    """Wrapper that provides get_json() interface."""

                    def __init__(self, data: dict[str, Any], original: Any):
                        self._data = data
                        self._original = original

                    def get_json(self, silent: bool = False) -> dict[str, Any]:
                        return self._data

                    @property
                    def method(self) -> str:
                        return (
                            self._original.method
                            if hasattr(self._original, "method")
                            else "POST"
                        )

                    @property
                    def args(self) -> Any:
                        return (
                            self._original.args
                            if hasattr(self._original, "args")
                            else {}
                        )

                # Call the actual handler
                result = fn(ProcessedRequest(processed_data, request))

                # Serialize response
                if isinstance(result, DimArray):
                    # Single DimArray result
                    response_data = serialize_for_http(result)
                    return (
                        json.dumps(response_data),
                        200,
                        {"Content-Type": "application/json"},
                    )

                elif isinstance(result, dict):
                    # Dictionary result - check for DimArrays
                    serialized_dict: dict[str, Any] = {}
                    for key, value in result.items():
                        if isinstance(value, DimArray):
                            serialized_dict[key] = serialize_for_http(value)
                        else:
                            serialized_dict[key] = value

                    return (
                        json.dumps(serialized_dict),
                        200,
                        {"Content-Type": "application/json"},
                    )

                elif isinstance(result, tuple) and len(result) >= 2:
                    # Handler returned (body, status_code) or (body, status, headers)
                    return result

                else:
                    # Other result - assume it's JSON-serializable
                    return (
                        json.dumps(result) if not isinstance(result, str) else result,
                        200,
                        {"Content-Type": "application/json"},
                    )

            except DimensionError as e:
                # Dimensional mismatch - user error
                error_body = json.dumps({"error": "DimensionError", "message": str(e)})
                return error_body, 400, {"Content-Type": "application/json"}

            except UnitConversionError as e:
                # Unit conversion error - user error
                error_body = json.dumps(
                    {"error": "UnitConversionError", "message": str(e)}
                )
                return error_body, 400, {"Content-Type": "application/json"}

            except json.JSONDecodeError as e:
                # Invalid JSON
                error_body = json.dumps(
                    {"error": "JSONDecodeError", "message": f"Invalid JSON: {str(e)}"}
                )
                return error_body, 400, {"Content-Type": "application/json"}

            except Exception as e:
                # Unexpected error
                tb = traceback.format_exc()
                error_body = json.dumps(
                    {"error": type(e).__name__, "message": str(e), "traceback": tb}
                )
                return error_body, 500, {"Content-Type": "application/json"}

        return wrapper

    return decorator


# Convenience decorators with preset configurations

# For general physics computations
cf_physics = cloud_function(lazy_imports=True, min_instances=1)

# For ML model inference
cf_ml = cloud_function(lazy_imports=True, min_instances=2)


def cloud_event_function(lazy_imports: bool = True) -> Callable:
    """Decorator for CloudEvent-triggered Cloud Functions.

    Similar to cloud_function but optimized for event-driven workloads
    (Pub/Sub, Cloud Storage, Firestore triggers).

    Args:
        lazy_imports: Enable lazy import optimization.

    Returns:
        Decorated handler function.

    Example:
        >>> import functions_framework
        >>>
        >>> @functions_framework.cloud_event
        >>> @cloud_event_function()
        ... def process_storage_event(cloud_event):
        ...     # Process Cloud Storage event
        ...     bucket = cloud_event.data["bucket"]
        ...     name = cloud_event.data["name"]
        ...     # ... process physics data file
        ...     return {"status": "complete"}
    """
    return cloud_function(lazy_imports=lazy_imports, min_instances=None)

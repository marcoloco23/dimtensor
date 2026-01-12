"""Example: Physics calculation API endpoint.

Demonstrates:
- Unit validation
- Common physics computations
- Error handling
- Multiple operation types

Deploy to AWS Lambda or Google Cloud Functions.
"""

from dimtensor import DimArray, units
from dimtensor.serverless import lambda_physics, require_params
import numpy as np


@lambda_physics
@require_params("operation")
def lambda_handler(event, context):
    """Physics computation API handler.

    Supported operations:
    - kinetic_energy: mass, velocity -> energy
    - gravitational_force: mass1, mass2, distance -> force
    - projectile_range: velocity, angle, gravity (optional) -> range
    - momentum: mass, velocity -> momentum
    - work: force, distance -> energy

    Example request (AWS Lambda):
    {
        "operation": "kinetic_energy",
        "mass": {
            "data": "AACAPw==",  # base64 [1.0]
            "unit": "kg",
            "dimension": [0, 1, 0, 0, 0, 0, 0],
            "shape": [1],
            "dtype": "float64",
            "scale": 1.0,
            "uncertainty": null
        },
        "velocity": {
            "data": "AAAAAAAAJEA=",  # base64 [10.0]
            "unit": "m/s",
            "dimension": [1, 0, -1, 0, 0, 0, 0],
            "shape": [1],
            "dtype": "float64",
            "scale": 1.0,
            "uncertainty": null
        }
    }

    Response:
    {
        "statusCode": 200,
        "body": "{...DimArray serialization...}",
        "headers": {"Content-Type": "application/json"}
    }
    """
    operation = event.get("operation")

    if operation == "kinetic_energy":
        # KE = 0.5 * m * v^2
        mass = event["mass"]
        velocity = event["velocity"]
        energy = 0.5 * mass * velocity**2
        return energy

    elif operation == "gravitational_force":
        # F = G * m1 * m2 / r^2
        from dimtensor.constants import G

        m1 = event["mass1"]
        m2 = event["mass2"]
        r = event["distance"]
        force = G * m1 * m2 / r**2
        return force

    elif operation == "projectile_range":
        # Range = v0^2 * sin(2*theta) / g
        v0 = event["velocity"]
        angle = event["angle"]  # radians (dimensionless)

        # Get gravity (use default if not provided)
        if "gravity" in event:
            g = event["gravity"]
        else:
            g = DimArray([9.81], units.m / units.s**2)

        # Calculate range
        range_val = v0**2 * np.sin(2 * angle) / g
        return range_val

    elif operation == "momentum":
        # p = m * v
        mass = event["mass"]
        velocity = event["velocity"]
        momentum = mass * velocity
        return momentum

    elif operation == "work":
        # W = F · d
        force = event["force"]
        distance = event["distance"]
        work = force * distance
        return work

    else:
        raise ValueError(
            f"Unknown operation: {operation}. "
            "Supported: kinetic_energy, gravitational_force, projectile_range, "
            "momentum, work"
        )


# For Google Cloud Functions deployment
def gcp_handler(request):
    """GCP Cloud Functions entry point."""
    from dimtensor.serverless import cf_physics

    @cf_physics
    def handler_impl(req):
        data = req.get_json()
        # Convert to Lambda-style event
        event = data
        context = None
        return lambda_handler(event, context)

    return handler_impl(request)

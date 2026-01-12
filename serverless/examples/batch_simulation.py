"""Example: Batch physics simulation processor.

Demonstrates:
- Processing S3/GCS uploaded data
- Time integration
- Results storage
- Event-driven architecture

Triggered by S3 ObjectCreated event or GCS Cloud Storage trigger.
"""

from dimtensor import DimArray, units
from dimtensor.serverless import lambda_batch
import numpy as np


@lambda_batch(memory_limit_mb=2048)
def lambda_handler(event, context):
    """Run physics simulation on uploaded data.

    Triggered by S3 upload event. Runs simulation and saves results back to S3.

    S3 Event format:
    {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": "my-bucket"},
                    "object": {"key": "input/simulation_001.json"}
                }
            }
        ]
    }

    Input file format (JSON):
    {
        "position": {...DimArray...},
        "velocity": {...DimArray...},
        "dt": {...DimArray...},
        "n_steps": 1000,
        "simulation_type": "projectile" | "orbital" | "pendulum"
    }
    """
    # Parse S3 event
    record = event["Records"][0]
    bucket = record["s3"]["bucket"]["name"]
    key = record["s3"]["object"]["key"]

    print(f"Processing simulation: s3://{bucket}/{key}")

    # Load initial conditions from S3
    import boto3
    import json

    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = json.loads(obj["Body"].read().decode())

    # Deserialize input data
    from dimtensor.serverless import deserialize_from_http

    position = deserialize_from_http(data["position"])
    velocity = deserialize_from_http(data["velocity"])
    dt = deserialize_from_http(data["dt"])
    n_steps = data["n_steps"]
    simulation_type = data.get("simulation_type", "projectile")

    # Run appropriate simulation
    if simulation_type == "projectile":
        results = simulate_projectile(position, velocity, dt, n_steps)
    elif simulation_type == "orbital":
        results = simulate_orbital(position, velocity, dt, n_steps, data)
    elif simulation_type == "pendulum":
        results = simulate_pendulum(position, velocity, dt, n_steps, data)
    else:
        raise ValueError(f"Unknown simulation type: {simulation_type}")

    # Save results back to S3
    from dimtensor.serverless import serialize_for_http

    output_data = {
        "positions": [serialize_for_http(p) for p in results["positions"]],
        "velocities": [serialize_for_http(v) for v in results["velocities"]],
        "times": [serialize_for_http(t) for t in results["times"]],
        "energies": [serialize_for_http(e) for e in results["energies"]],
    }

    result_key = key.replace("input/", "output/")
    s3.put_object(
        Bucket=bucket, Key=result_key, Body=json.dumps(output_data)
    )

    print(f"Results saved to: s3://{bucket}/{result_key}")

    return {"status": "complete", "output_key": result_key, "n_steps": n_steps}


def simulate_projectile(position, velocity, dt, n_steps):
    """Simulate projectile motion with constant gravity."""
    positions = [position]
    velocities = [velocity]
    times = [DimArray([0.0], units.s)]

    # Constant acceleration (gravity)
    g = DimArray([0.0, 0.0, -9.81], units.m / units.s**2)

    for i in range(n_steps):
        # Euler integration
        velocity_new = velocities[-1] + g * dt
        position_new = positions[-1] + velocities[-1] * dt
        time_new = times[-1] + dt

        velocities.append(velocity_new)
        positions.append(position_new)
        times.append(time_new)

        # Stop if hit ground
        if position_new._data[2] < 0:
            break

    # Calculate energies (kinetic only for projectile)
    energies = []
    mass = DimArray([1.0], units.kg)  # Assume unit mass
    for v in velocities:
        ke = 0.5 * mass * np.sum(v._data**2)
        energies.append(ke)

    return {
        "positions": positions,
        "velocities": velocities,
        "times": times,
        "energies": energies,
    }


def simulate_orbital(position, velocity, dt, n_steps, params):
    """Simulate orbital motion under gravity."""
    from dimtensor.constants import G

    positions = [position]
    velocities = [velocity]
    times = [DimArray([0.0], units.s)]

    # Central mass (e.g., Earth)
    M = params.get("central_mass", 5.972e24)  # kg
    central_mass = DimArray([M], units.kg)

    for i in range(n_steps):
        r = positions[-1]
        v = velocities[-1]

        # Distance from origin
        r_mag = np.sqrt(np.sum(r._data**2))

        # Gravitational acceleration: a = -G*M/r^2 * r_hat
        r_hat = r._data / r_mag
        a_mag = G * central_mass / DimArray([r_mag**2], units.m**2)
        acceleration = DimArray(-a_mag._data[0] * r_hat, units.m / units.s**2)

        # Update using Euler method (use Verlet for better accuracy)
        velocity_new = v + acceleration * dt
        position_new = r + v * dt
        time_new = times[-1] + dt

        velocities.append(velocity_new)
        positions.append(position_new)
        times.append(time_new)

    # Calculate total energy (kinetic + potential)
    energies = []
    mass = DimArray([1.0], units.kg)
    for r, v in zip(positions, velocities):
        r_mag = np.sqrt(np.sum(r._data**2))
        ke = 0.5 * mass * np.sum(v._data**2)
        pe = -G * central_mass * mass / DimArray([r_mag], units.m)
        energies.append(ke + pe)

    return {
        "positions": positions,
        "velocities": velocities,
        "times": times,
        "energies": energies,
    }


def simulate_pendulum(position, velocity, dt, n_steps, params):
    """Simulate simple pendulum."""
    # Extract angle from position (assume position is [theta])
    angles = [position]
    angular_velocities = [velocity]
    times = [DimArray([0.0], units.s)]

    # Pendulum parameters
    L = params.get("length", 1.0)  # meters
    g = params.get("gravity", 9.81)  # m/s^2

    length = DimArray([L], units.m)
    gravity = DimArray([g], units.m / units.s**2)

    for i in range(n_steps):
        theta = angles[-1]
        omega = angular_velocities[-1]

        # Angular acceleration: alpha = -(g/L) * sin(theta)
        alpha = -(gravity / length) * np.sin(theta._data)

        # Update
        omega_new = omega + alpha * dt
        theta_new = theta + omega * dt
        time_new = times[-1] + dt

        angular_velocities.append(omega_new)
        angles.append(theta_new)
        times.append(time_new)

    # Calculate energy
    energies = []
    mass = DimArray([1.0], units.kg)
    for theta, omega in zip(angles, angular_velocities):
        # Kinetic energy: 0.5 * m * L^2 * omega^2
        ke = 0.5 * mass * length**2 * omega**2
        # Potential energy: m * g * L * (1 - cos(theta))
        pe = mass * gravity * length * (1 - np.cos(theta._data))
        energies.append(ke + pe)

    return {
        "positions": angles,
        "velocities": angular_velocities,
        "times": times,
        "energies": energies,
    }


# For Google Cloud Functions deployment
def gcp_handler(event, context):
    """GCP Cloud Storage trigger entry point.

    Event format:
    {
        "bucket": "my-bucket",
        "name": "input/simulation_001.json",
        "metageneration": "1",
        "timeCreated": "...",
        "updated": "..."
    }
    """
    # Convert GCS event to Lambda S3 format
    lambda_event = {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": event["bucket"]},
                    "object": {"key": event["name"]},
                }
            }
        ]
    }

    # Use GCS client instead of S3
    # (Similar logic but with google.cloud.storage)
    return lambda_handler(lambda_event, context)

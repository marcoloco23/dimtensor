#!/usr/bin/env python3
"""
Batch Analysis Workload for Kubernetes

This script demonstrates a batch processing workload using dimtensor
for parameter sweep analysis with Buckingham Pi theorem.

Usage:
    python batch-analysis.py --task-id 0 --total-tasks 100 --output-dir /data/output
"""

import argparse
import os
import json
import numpy as np
from dimtensor import DimArray, units
from dimtensor.analysis import buckingham_pi


def analyze_drag_force(velocity, length, density, viscosity):
    """
    Analyze drag force dimensionless groups using Buckingham Pi theorem.

    Returns Reynolds number and drag coefficient relationships.
    """
    # Calculate Reynolds number
    Re = (density * velocity * length / viscosity).to(units.dimensionless)

    # Simplified drag coefficient (turbulent flow approximation)
    if Re.data[0] > 1000:
        Cd = 0.001 * (1 + 10 / Re.data[0] ** (1/3))
    else:
        Cd = 24 / Re.data[0]  # Stokes flow

    # Calculate drag force
    area = length ** 2
    drag = 0.5 * Cd * density * velocity ** 2 * area

    return {
        "reynolds": float(Re.data[0]),
        "drag_coefficient": Cd,
        "drag_force": {
            "value": float(drag.data[0]),
            "unit": str(drag.unit),
            "dimension": str(drag.dimension)
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Batch parameter sweep analysis")
    parser.add_argument("--task-id", type=int, default=0, help="Task index")
    parser.add_argument("--total-tasks", type=int, default=100, help="Total tasks")
    parser.add_argument("--output-dir", type=str, default="/data/output", help="Output directory")
    args = parser.parse_args()

    # Try to get task ID from Kubernetes environment
    task_id = int(os.environ.get("JOB_COMPLETION_INDEX", args.task_id))

    print(f"Starting batch analysis task {task_id}/{args.total_tasks}")

    # Define parameter space
    velocities = np.linspace(1, 100, args.total_tasks)

    # Process this task's parameter
    v = velocities[task_id]
    velocity = DimArray([v], units.m / units.s)
    length = DimArray([1.0], units.m)
    density = DimArray([1.225], units.kg / units.m**3)  # Air at sea level
    viscosity = DimArray([1.81e-5], units.kg / (units.m * units.s))  # Air at 15°C

    print(f"Analyzing velocity: {velocity}")

    # Perform analysis
    result = analyze_drag_force(velocity, length, density, viscosity)

    print(f"Reynolds number: {result['reynolds']:.2f}")
    print(f"Drag coefficient: {result['drag_coefficient']:.6f}")
    print(f"Drag force: {result['drag_force']['value']:.4f} {result['drag_force']['unit']}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"result_{task_id}.json")

    with open(output_file, 'w') as f:
        json.dump({
            "task_id": task_id,
            "velocity": v,
            "analysis": result
        }, f, indent=2)

    print(f"Results saved to {output_file}")
    print(f"Task {task_id} complete!")


if __name__ == "__main__":
    main()

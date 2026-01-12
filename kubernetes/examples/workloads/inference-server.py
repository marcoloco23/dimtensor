#!/usr/bin/env python3
"""
Inference Server Workload for Kubernetes

This script provides a RESTful API for physics predictions with
dimensional validation using dimtensor.

Endpoints:
    GET  /health           - Health check
    GET  /ready            - Readiness check
    POST /api/v1/validate  - Validate equation dimensions
    POST /api/v1/convert   - Convert between units
    POST /api/v1/compute   - Compute physics quantities
    GET  /metrics          - Prometheus metrics

Usage:
    python inference-server.py --port 8080
"""

import argparse
from flask import Flask, request, jsonify
from dimtensor import DimArray, units
from dimtensor.errors import DimensionError, UnitConversionError
import time


app = Flask(__name__)

# Simple metrics tracking
request_count = 0
error_count = 0
start_time = time.time()


@app.route('/health')
def health():
    """Health check endpoint for liveness probe"""
    return jsonify({"status": "healthy", "service": "dimtensor-inference"}), 200


@app.route('/ready')
def ready():
    """Readiness check endpoint"""
    try:
        # Verify dimtensor works
        test = DimArray([1], units.m)
        return jsonify({
            "status": "ready",
            "service": "dimtensor-inference",
            "uptime": time.time() - start_time
        }), 200
    except Exception as e:
        return jsonify({
            "status": "not ready",
            "error": str(e)
        }), 503


@app.route('/api/v1/validate', methods=['POST'])
def validate_dimensions():
    """
    Validate dimensional consistency of equation.

    Request body:
        {
            "mass": 10,           # kg
            "acceleration": 9.81  # m/s^2
        }

    Response:
        {
            "valid": true,
            "force": {
                "value": 98.1,
                "unit": "N",
                "dimension": "L M T^-2"
            }
        }
    """
    global request_count, error_count
    request_count += 1

    try:
        data = request.json

        # Extract values
        mass = DimArray([data['mass']], units.kg)
        acceleration = DimArray([data['acceleration']], units.m / units.s**2)

        # Compute force: F = m * a
        force = mass * acceleration

        return jsonify({
            "valid": True,
            "force": {
                "value": float(force.data[0]),
                "unit": str(force.unit),
                "dimension": str(force.dimension)
            }
        }), 200

    except DimensionError as e:
        error_count += 1
        return jsonify({
            "valid": False,
            "error": "Dimensional mismatch",
            "details": str(e)
        }), 400

    except Exception as e:
        error_count += 1
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/api/v1/convert', methods=['POST'])
def convert_units():
    """
    Convert between compatible units.

    Request body:
        {
            "value": 1000,
            "from_unit": "m",
            "to_unit": "km"
        }

    Response:
        {
            "original": {"value": 1000, "unit": "m"},
            "converted": {"value": 1.0, "unit": "km"}
        }
    """
    global request_count, error_count
    request_count += 1

    try:
        data = request.json
        value = data['value']
        from_unit_name = data['from_unit']
        to_unit_name = data['to_unit']

        # Parse units
        from_unit = getattr(units, from_unit_name)
        to_unit = getattr(units, to_unit_name)

        # Convert
        arr = DimArray([value], from_unit)
        converted = arr.to(to_unit)

        return jsonify({
            "original": {
                "value": value,
                "unit": from_unit_name
            },
            "converted": {
                "value": float(converted.data[0]),
                "unit": to_unit_name
            }
        }), 200

    except UnitConversionError as e:
        error_count += 1
        return jsonify({
            "error": "Unit conversion failed",
            "details": str(e)
        }), 400

    except AttributeError:
        error_count += 1
        return jsonify({
            "error": "Invalid unit name"
        }), 400

    except Exception as e:
        error_count += 1
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/api/v1/compute', methods=['POST'])
def compute_physics():
    """
    Compute physics quantities with dimensional analysis.

    Request body:
        {
            "operation": "kinetic_energy",
            "mass": 10,      # kg
            "velocity": 5    # m/s
        }

    Response:
        {
            "result": {
                "value": 125.0,
                "unit": "J",
                "dimension": "L^2 M T^-2"
            }
        }
    """
    global request_count, error_count
    request_count += 1

    try:
        data = request.json
        operation = data['operation']

        if operation == "kinetic_energy":
            mass = DimArray([data['mass']], units.kg)
            velocity = DimArray([data['velocity']], units.m / units.s)
            result = 0.5 * mass * velocity ** 2

        elif operation == "potential_energy":
            mass = DimArray([data['mass']], units.kg)
            height = DimArray([data['height']], units.m)
            g = DimArray([9.81], units.m / units.s**2)
            result = mass * g * height

        elif operation == "momentum":
            mass = DimArray([data['mass']], units.kg)
            velocity = DimArray([data['velocity']], units.m / units.s)
            result = mass * velocity

        elif operation == "pressure":
            force = DimArray([data['force']], units.N)
            area = DimArray([data['area']], units.m**2)
            result = force / area

        else:
            error_count += 1
            return jsonify({
                "error": f"Unknown operation: {operation}"
            }), 400

        return jsonify({
            "operation": operation,
            "result": {
                "value": float(result.data[0]),
                "unit": str(result.unit),
                "dimension": str(result.dimension)
            }
        }), 200

    except Exception as e:
        error_count += 1
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    uptime = time.time() - start_time

    metrics_text = f"""# HELP dimtensor_requests_total Total number of requests
# TYPE dimtensor_requests_total counter
dimtensor_requests_total {request_count}

# HELP dimtensor_errors_total Total number of errors
# TYPE dimtensor_errors_total counter
dimtensor_errors_total {error_count}

# HELP dimtensor_uptime_seconds Service uptime in seconds
# TYPE dimtensor_uptime_seconds gauge
dimtensor_uptime_seconds {uptime:.2f}
"""
    return metrics_text, 200, {'Content-Type': 'text/plain; charset=utf-8'}


def main():
    parser = argparse.ArgumentParser(description="dimtensor inference API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    args = parser.parse_args()

    print(f"Starting dimtensor inference server on {args.host}:{args.port}")
    print("Endpoints:")
    print("  GET  /health           - Health check")
    print("  GET  /ready            - Readiness check")
    print("  POST /api/v1/validate  - Validate dimensions")
    print("  POST /api/v1/convert   - Convert units")
    print("  POST /api/v1/compute   - Compute physics quantities")
    print("  GET  /metrics          - Prometheus metrics")

    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

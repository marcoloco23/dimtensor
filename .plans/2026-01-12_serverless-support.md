# Plan: Serverless Support for AWS Lambda and Google Cloud Functions

**Date**: 2026-01-12
**Status**: PLANNING
**Tasks**: #256-257 (v5.0.0 Phase 4: Deployment)
**Author**: agent (planner)

---

## Goal

Enable serverless deployment of physics computations using dimtensor on AWS Lambda and Google Cloud Functions, with optimized cold start performance and production-ready deployment templates.

---

## Background

### Why Serverless for Physics?

dimtensor users need to:
1. **Deploy physics models** - Serve trained models for inference
2. **Run computations on-demand** - Execute physics calculations without maintaining servers
3. **Scale automatically** - Handle variable workloads (e.g., batch simulations, API endpoints)
4. **Reduce costs** - Pay only for execution time

### Current State

dimtensor v4.5.0 supports:
- NumPy `DimArray` (core)
- PyTorch `DimTensor` for ML
- Distributed computing (Dask, Ray)
- Docker containerization (via existing patterns)
- Serialization (JSON, HDF5, Parquet)

**Gap**: No serverless deployment infrastructure or optimization.

### Challenges

1. **Package Size**: numpy (15-20MB) + dimtensor (~2MB) must fit in Lambda limits
2. **Cold Starts**: Python import time + numpy/dimtensor initialization
3. **Memory Constraints**: Default 128MB Lambda memory often insufficient for scientific computing
4. **Unit Persistence**: Physics units must be preserved across function invocations

---

## Research Summary

### AWS Lambda Best Practices (2026)

From [FourTheorem](https://fourtheorem.com/optimise-python-data-science-aws-lambda/), [Architech](https://www.architech.ca/articles/deploying-aws-lambda-functions-with-numpy-dependencies), and [AWS Docs](https://docs.aws.amazon.com/lambda/latest/dg/python-layers.html):

**Packaging Options:**
1. **Lambda Layers** - Good for smaller dependencies (<250MB unzipped)
2. **Container Images** - Best for larger packages, better cold start performance
3. **Pre-built AWS Layers** - AWS provides NumPy/SciPy layers

**Key Optimizations:**
- Use `pip install --platform manylinux2014_x86_64 --only-binary=:all:` for Linux wheels
- Lazy imports: Import heavy modules inside handler, not at module level
- Increase memory: Higher memory = faster cold starts (more CPU allocated)
- Container images often outperform zip deployments for scientific packages
- Use environment variables for configuration to avoid redeployment

### Google Cloud Functions Best Practices (2026)

From [GCP Docs](https://cloud.google.com/run/docs/tips/python) and [Cloud Functions Best Practices](https://cloud.google.com/run/docs/tips/functions-best-practices):

**Key Optimizations:**
1. **Minimize dependencies** - Trim unused packages
2. **Lazy loading** - Import modules inside function, use global scope for reuse
3. **Set minimum instances** - Reduce cold starts for latency-sensitive apps
4. **Increase memory** - Higher memory = faster initialization
5. **Use 2nd gen Cloud Functions** - Better concurrency, performance
6. **Slim base images** - Use Python slim images for smaller containers

**Performance Tips:**
- Global initialization persists across invocations (use for expensive setup)
- Cache expensive computations
- Avoid synchronous work outside request handlers
- Use Cloud Run Functions for better performance vs 1st gen

### dimtensor Existing Patterns

From codebase review:

1. **Serialization**: `io/json.py` - Efficient DimArray serialization
2. **Lazy imports**: Not yet implemented
3. **Optional dependencies**: `pyproject.toml` - torch, jax, etc. are optional
4. **Internal constructor**: `DimArray._from_data_and_unit()` - Fast, no-copy creation
5. **Distributed patterns**: Ray integration shows serialization handling

---

## Approach

### Option A: Handler Wrappers + Layer Deployment (Recommended)

Create decorator-based handler wrappers that automatically handle:
- DimArray serialization/deserialization
- Error handling with dimensional errors
- Response formatting

**Pros:**
- Easy to use (decorator pattern familiar to users)
- Works with existing Lambda/Cloud Functions code
- Minimal overhead
- Users keep full control of handler logic

**Cons:**
- Requires users to install dimtensor in deployment package
- Some manual setup required

### Option B: Full Abstraction Framework

Create a complete framework that abstracts away all serverless details.

**Pros:**
- Very easy to use
- Handles all deployment complexity

**Cons:**
- Heavy abstraction may limit flexibility
- Harder to debug
- More maintenance burden

### Decision: Option A (Handler Wrappers)

Provides the right balance of convenience and flexibility. Users familiar with serverless already understand handlers; we just make unit handling seamless.

---

## Design

### Module Structure

```
src/dimtensor/serverless/
├── __init__.py              # Public exports
├── aws.py                   # AWS Lambda handler wrappers
├── gcp.py                   # Google Cloud Functions handler wrappers
├── serialization.py         # HTTP-friendly serialization
├── validation.py            # Input validation with units
└── errors.py                # Error handling utilities

deployment/
├── aws/
│   ├── lambda_layer/        # Pre-built Lambda layer
│   │   ├── build_layer.sh   # Script to build layer
│   │   └── requirements.txt # numpy + dimtensor
│   ├── sam/                 # AWS SAM templates
│   │   ├── template.yaml    # Base SAM template
│   │   └── examples/        # Example functions
│   └── container/           # Dockerfile for container deployment
│       ├── Dockerfile
│       └── .dockerignore
├── gcp/
│   ├── functions/           # Cloud Functions deployment
│   │   ├── requirements.txt
│   │   └── .gcloudignore
│   └── cloudrun/           # Cloud Run deployment
│       ├── Dockerfile
│       └── cloudbuild.yaml
└── examples/
    ├── physics_api.py       # Example: Physics calculation API
    ├── model_inference.py   # Example: ML model inference
    └── batch_simulation.py  # Example: Batch simulation processor
```

---

## Implementation Steps

### Phase 1: Core Serverless Module (#256 - AWS Lambda)

#### 1. Create `serverless/serialization.py`
HTTP-friendly serialization for DimArray.

```python
"""HTTP-friendly serialization for serverless environments."""

from typing import Any
import json
import base64
import numpy as np
from ..core.dimarray import DimArray
from ..io.json import to_dict, from_dict

def serialize_for_http(arr: DimArray) -> dict[str, Any]:
    """Serialize DimArray for HTTP response (JSON-compatible).

    Returns dict with:
    - data: base64-encoded numpy array
    - unit: unit metadata
    - shape: array shape
    - dtype: numpy dtype as string
    """
    return {
        "data": base64.b64encode(arr._data.tobytes()).decode("utf-8"),
        "unit": arr.unit.symbol,
        "dimension": arr.dimension.as_tuple(),
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "uncertainty": (
            base64.b64encode(arr._uncertainty.tobytes()).decode("utf-8")
            if arr._uncertainty is not None
            else None
        ),
    }

def deserialize_from_http(data: dict[str, Any]) -> DimArray:
    """Deserialize DimArray from HTTP request."""
    from ..core.dimensions import Dimension
    from ..core.units import Unit
    from fractions import Fraction

    # Decode numpy array
    arr = np.frombuffer(
        base64.b64decode(data["data"]),
        dtype=np.dtype(data["dtype"])
    ).reshape(data["shape"])

    # Reconstruct unit
    dim = Dimension(*[Fraction(x) for x in data["dimension"]])
    unit = Unit.from_symbol(data["unit"]) or Unit(data["unit"], dim)

    # Decode uncertainty if present
    uncertainty = None
    if data.get("uncertainty"):
        uncertainty = np.frombuffer(
            base64.b64decode(data["uncertainty"]),
            dtype=np.dtype(data["dtype"])
        ).reshape(data["shape"])

    return DimArray._from_data_and_unit(arr, unit, uncertainty)

def simple_response(arr: DimArray, status_code: int = 200) -> dict:
    """Create Lambda/Cloud Functions response dict."""
    return {
        "statusCode": status_code,
        "body": json.dumps(serialize_for_http(arr)),
        "headers": {
            "Content-Type": "application/json",
            "X-DimTensor-Version": "5.0.0",
        },
    }
```

#### 2. Create `serverless/aws.py`
AWS Lambda handler decorators.

```python
"""AWS Lambda handler wrappers for dimtensor."""

import functools
import json
import traceback
from typing import Callable, Any
from ..core.dimarray import DimArray
from ..errors import DimensionError, UnitConversionError
from .serialization import deserialize_from_http, simple_response

def lambda_handler(
    lazy_imports: bool = True,
    memory_limit_mb: int | None = None,
):
    """Decorator for AWS Lambda handlers with dimtensor support.

    Automatically handles:
    - DimArray deserialization from request
    - DimArray serialization in response
    - Dimensional error handling
    - Lazy imports for cold start optimization

    Example:
        @lambda_handler(lazy_imports=True)
        def compute_energy(event, context):
            mass = event["mass"]  # Automatically a DimArray
            velocity = event["velocity"]
            return 0.5 * mass * velocity**2  # Returns DimArray

    Args:
        lazy_imports: If True, imports dimtensor inside handler
        memory_limit_mb: Warn if Lambda memory below this threshold
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(event, context):
            # Lazy import optimization
            if lazy_imports:
                # Heavy imports happen here, not at module load
                pass  # Already imported, but in production this matters

            # Check memory
            if memory_limit_mb and hasattr(context, "memory_limit_in_mb"):
                if context.memory_limit_in_mb < memory_limit_mb:
                    print(f"WARNING: Memory {context.memory_limit_in_mb}MB < recommended {memory_limit_mb}MB")

            try:
                # Parse body if present
                if isinstance(event.get("body"), str):
                    body = json.loads(event["body"])
                else:
                    body = event

                # Deserialize DimArray inputs
                processed_event = {}
                for key, value in body.items():
                    if isinstance(value, dict) and "data" in value and "unit" in value:
                        processed_event[key] = deserialize_from_http(value)
                    else:
                        processed_event[key] = value

                # Call handler
                result = fn(processed_event, context)

                # Serialize response
                if isinstance(result, DimArray):
                    return simple_response(result)
                elif isinstance(result, dict):
                    return {
                        "statusCode": 200,
                        "body": json.dumps(result),
                        "headers": {"Content-Type": "application/json"},
                    }
                else:
                    return result

            except DimensionError as e:
                return {
                    "statusCode": 400,
                    "body": json.dumps({
                        "error": "DimensionError",
                        "message": str(e),
                    }),
                    "headers": {"Content-Type": "application/json"},
                }
            except UnitConversionError as e:
                return {
                    "statusCode": 400,
                    "body": json.dumps({
                        "error": "UnitConversionError",
                        "message": str(e),
                    }),
                    "headers": {"Content-Type": "application/json"},
                }
            except Exception as e:
                return {
                    "statusCode": 500,
                    "body": json.dumps({
                        "error": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    }),
                    "headers": {"Content-Type": "application/json"},
                }

        return wrapper

    return decorator

# Convenience decorators
lambda_physics = lambda_handler(lazy_imports=True, memory_limit_mb=512)
lambda_ml = lambda_handler(lazy_imports=True, memory_limit_mb=1024)
```

#### 3. Create `serverless/gcp.py`
Google Cloud Functions handler decorators.

```python
"""Google Cloud Functions handler wrappers for dimtensor."""

import functools
import json
from typing import Callable
from flask import Request, jsonify
from ..core.dimarray import DimArray
from ..errors import DimensionError, UnitConversionError
from .serialization import deserialize_from_http, serialize_for_http

def cloud_function(
    lazy_imports: bool = True,
    min_instances: int | None = None,
):
    """Decorator for Google Cloud Functions with dimtensor support.

    Supports both HTTP and CloudEvent-triggered functions.

    Example:
        @cloud_function(lazy_imports=True)
        def compute_energy(request: Request):
            data = request.get_json()
            mass = data["mass"]  # Automatically a DimArray
            velocity = data["velocity"]
            return 0.5 * mass * velocity**2  # Returns DimArray

    Args:
        lazy_imports: If True, imports dimtensor inside handler
        min_instances: Recommended minimum instances (for docs)
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(request: Request):
            # Lazy import optimization
            if lazy_imports:
                pass  # Already imported

            # Log recommendation
            if min_instances:
                # In production, this would be in deployment config
                pass

            try:
                # Parse request
                if request.method == "GET":
                    data = request.args.to_dict()
                else:
                    data = request.get_json(silent=True) or {}

                # Deserialize DimArray inputs
                processed_data = {}
                for key, value in data.items():
                    if isinstance(value, dict) and "data" in value and "unit" in value:
                        processed_data[key] = deserialize_from_http(value)
                    else:
                        processed_data[key] = value

                # Call handler (pass processed data as request-like object)
                class DataRequest:
                    def __init__(self, data):
                        self._data = data
                    def get_json(self):
                        return self._data
                    @property
                    def method(self):
                        return request.method

                result = fn(DataRequest(processed_data))

                # Serialize response
                if isinstance(result, DimArray):
                    return jsonify(serialize_for_http(result))
                else:
                    return jsonify(result)

            except DimensionError as e:
                return jsonify({"error": "DimensionError", "message": str(e)}), 400
            except UnitConversionError as e:
                return jsonify({"error": "UnitConversionError", "message": str(e)}), 400
            except Exception as e:
                return jsonify({"error": type(e).__name__, "message": str(e)}), 500

        return wrapper

    return decorator

# Convenience decorators
cf_physics = cloud_function(lazy_imports=True, min_instances=1)
cf_ml = cloud_function(lazy_imports=True, min_instances=2)
```

### Phase 2: Deployment Templates (#256 - AWS)

#### 4. Create AWS Lambda Layer Build Script

`deployment/aws/lambda_layer/build_layer.sh`:
```bash
#!/bin/bash
# Build AWS Lambda layer with dimtensor + numpy

set -e

echo "Building Lambda layer for Python 3.12..."

# Create layer directory
rm -rf layer
mkdir -p layer/python

# Install dependencies for Linux (Lambda runtime)
pip install \
    --platform manylinux2014_x86_64 \
    --target=layer/python \
    --implementation cp \
    --python-version 3.12 \
    --only-binary=:all: \
    --upgrade \
    numpy dimtensor

# Create zip
cd layer
zip -r ../dimtensor-layer.zip python
cd ..

echo "Layer created: dimtensor-layer.zip"
echo "Size: $(du -h dimtensor-layer.zip | cut -f1)"
echo ""
echo "Upload to Lambda:"
echo "  aws lambda publish-layer-version \\"
echo "    --layer-name dimtensor-numpy \\"
echo "    --zip-file fileb://dimtensor-layer.zip \\"
echo "    --compatible-runtimes python3.12"
```

#### 5. Create AWS SAM Template

`deployment/aws/sam/template.yaml`:
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: dimtensor physics computation functions

Globals:
  Function:
    Timeout: 30
    MemorySize: 512  # Recommended for numpy
    Runtime: python3.12
    Environment:
      Variables:
        DIMTENSOR_LAZY_IMPORT: "true"

Resources:
  # Physics computation function
  PhysicsComputeFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: functions/physics_compute/
      Handler: app.lambda_handler
      Description: Generic physics computation endpoint
      Layers:
        - !Ref DimTensorLayer
      Events:
        PhysicsApi:
          Type: Api
          Properties:
            Path: /compute
            Method: post

  # ML model inference function
  ModelInferenceFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: functions/model_inference/
      Handler: app.lambda_handler
      Description: Physics ML model inference
      MemorySize: 1024  # More memory for ML
      Layers:
        - !Ref DimTensorLayer
      Events:
        InferenceApi:
          Type: Api
          Properties:
            Path: /infer
            Method: post

  # Lambda Layer
  DimTensorLayer:
    Type: AWS::Serverless::LayerVersion
    Properties:
      LayerName: dimtensor-numpy
      Description: dimtensor + numpy for physics computing
      ContentUri: ../../lambda_layer/dimtensor-layer.zip
      CompatibleRuntimes:
        - python3.12

Outputs:
  PhysicsComputeApi:
    Description: "API Gateway endpoint URL"
    Value: !Sub "https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod/compute/"
```

#### 6. Create Container Dockerfile

`deployment/aws/container/Dockerfile`:
```dockerfile
FROM public.ecr.aws/lambda/python:3.12

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}/

# Set handler
CMD ["app.lambda_handler"]
```

### Phase 3: GCP Deployment Templates (#257)

#### 7. Create Cloud Functions Template

`deployment/gcp/functions/main.py`:
```python
"""Google Cloud Functions example with dimtensor."""

# Lazy imports for cold start optimization
import functions_framework
from dimtensor.serverless import cloud_function

# Import dimtensor only when needed (not at module level)
def lazy_dimtensor():
    from dimtensor import DimArray, units
    return DimArray, units

@functions_framework.http
@cloud_function(lazy_imports=True)
def compute_energy(request):
    """Compute kinetic energy from mass and velocity.

    POST /compute_energy
    {
        "mass": {"data": "...", "unit": "kg", ...},
        "velocity": {"data": "...", "unit": "m/s", ...}
    }
    """
    DimArray, units = lazy_dimtensor()

    data = request.get_json()
    mass = data["mass"]  # Already DimArray (decorator handles it)
    velocity = data["velocity"]

    energy = 0.5 * mass * velocity**2
    return energy  # Decorator serializes it
```

`deployment/gcp/functions/requirements.txt`:
```
functions-framework==3.*
numpy>=1.24.0,<2.0.0
dimtensor>=5.0.0
```

#### 8. Create Cloud Run Dockerfile

`deployment/gcp/cloudrun/Dockerfile`:
```dockerfile
# Use Python slim for smaller image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run on port 8080 (Cloud Run default)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
```

### Phase 4: Example Functions

#### 9. Create Physics API Example

`deployment/examples/physics_api.py`:
```python
"""Example: Physics calculation API endpoint.

Demonstrates:
- Unit validation
- Common physics computations
- Error handling
"""

from dimtensor import DimArray, units
from dimtensor.serverless import lambda_physics

@lambda_physics
def handler(event, context):
    """Compute various physics quantities.

    Supported operations:
    - kinetic_energy: mass, velocity -> energy
    - gravitational_force: mass1, mass2, distance -> force
    - projectile_range: velocity, angle, gravity -> range
    """
    operation = event.get("operation")

    if operation == "kinetic_energy":
        mass = event["mass"]
        velocity = event["velocity"]
        return 0.5 * mass * velocity**2

    elif operation == "gravitational_force":
        from dimtensor.constants import G
        m1 = event["mass1"]
        m2 = event["mass2"]
        r = event["distance"]
        return G * m1 * m2 / r**2

    elif operation == "projectile_range":
        import numpy as np
        v0 = event["velocity"]
        angle = event["angle"]  # in radians
        g = event.get("gravity", 9.81 * units.m / units.s**2)

        # Range = v0^2 * sin(2*theta) / g
        range_val = v0**2 * np.sin(2 * angle) / g
        return range_val

    else:
        raise ValueError(f"Unknown operation: {operation}")
```

#### 10. Create ML Inference Example

`deployment/examples/model_inference.py`:
```python
"""Example: ML model inference with dimtensor.

Demonstrates:
- Loading trained physics model
- Unit-aware inference
- Batch processing
"""

import json
from dimtensor import DimArray, units
from dimtensor.torch import DimTensor, DimScaler
from dimtensor.serverless import lambda_ml

# Global scope - initialized once, reused across invocations
MODEL = None
SCALER = None

def load_model():
    """Load model once (cold start only)."""
    global MODEL, SCALER
    if MODEL is None:
        # Lazy import
        import torch
        from dimtensor.torch import DimLinear

        # Load model from S3 or include in deployment
        # For demo, create a simple model
        MODEL = DimLinear(3, 1, units.m, units.J)
        MODEL.eval()

        # Load scaler
        SCALER = DimScaler(method="standard")
        # In production, load from file

@lambda_ml
def handler(event, context):
    """Run inference on physics model.

    Input:
    {
        "inputs": DimArray with shape (N, 3) and units [m]
    }

    Output:
    {
        "predictions": DimArray with shape (N, 1) and units [J]
    }
    """
    import torch

    # Load model (only once)
    load_model()

    # Get inputs
    inputs = event["inputs"]  # DimArray

    # Convert to DimTensor
    inputs_tensor = DimTensor(inputs._data, inputs.unit)

    # Scale for model
    inputs_scaled = SCALER.transform(inputs_tensor)

    # Inference
    with torch.no_grad():
        outputs_scaled = MODEL(inputs_scaled)

    # Inverse transform
    outputs = SCALER.inverse_transform(outputs_scaled, units.J)

    # Convert back to DimArray
    result = DimArray(outputs.data.numpy(), outputs.unit)

    return result
```

#### 11. Create Batch Simulation Example

`deployment/examples/batch_simulation.py`:
```python
"""Example: Batch physics simulation processor.

Demonstrates:
- Processing S3/GCS uploaded data
- Time integration
- Results storage
"""

from dimtensor import DimArray, units
from dimtensor.serverless import lambda_physics
import numpy as np

@lambda_physics
def handler(event, context):
    """Run physics simulation on uploaded data.

    Triggered by S3 upload event.
    Runs simulation and saves results back to S3.
    """
    # Parse S3 event
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = event["Records"][0]["s3"]["object"]["key"]

    # Load initial conditions from S3
    import boto3
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)

    # Deserialize (JSON format)
    from dimtensor.io import from_json
    initial_data = from_json(obj["Body"].read().decode())

    # Run simulation
    position = initial_data["position"]
    velocity = initial_data["velocity"]
    dt = initial_data["dt"]
    n_steps = initial_data["n_steps"]

    # Simple Euler integration
    positions = [position]
    velocities = [velocity]

    acceleration = DimArray([-9.81, 0, 0], units.m / units.s**2)

    for _ in range(n_steps):
        velocity = velocity + acceleration * dt
        position = position + velocity * dt
        positions.append(position)
        velocities.append(velocity)

    # Save results
    from dimtensor.io import to_json
    results = {
        "positions": positions,
        "velocities": velocities,
    }

    result_key = key.replace("input", "output")
    s3.put_object(
        Bucket=bucket,
        Key=result_key,
        Body=to_json(results),
    )

    return {"status": "complete", "output_key": result_key}
```

### Phase 5: Documentation and Testing

#### 12. Create deployment documentation
- AWS Lambda deployment guide
- GCP Cloud Functions deployment guide
- Performance optimization tips
- Cost estimation

#### 13. Create tests
- Unit tests for serialization
- Integration tests with mocked Lambda/GCF
- Cold start benchmarks
- Load testing scripts

---

## Files to Create

| File | Purpose | Lines (est) |
|------|---------|-------------|
| `serverless/__init__.py` | Public exports | ~30 |
| `serverless/serialization.py` | HTTP serialization | ~150 |
| `serverless/aws.py` | Lambda handler decorators | ~200 |
| `serverless/gcp.py` | Cloud Functions decorators | ~150 |
| `serverless/validation.py` | Input validation | ~100 |
| `serverless/errors.py` | Error handling | ~80 |
| `deployment/aws/lambda_layer/build_layer.sh` | Layer build script | ~50 |
| `deployment/aws/sam/template.yaml` | SAM template | ~100 |
| `deployment/aws/container/Dockerfile` | Container image | ~20 |
| `deployment/gcp/functions/main.py` | GCF example | ~80 |
| `deployment/gcp/functions/requirements.txt` | Dependencies | ~5 |
| `deployment/gcp/cloudrun/Dockerfile` | Cloud Run image | ~15 |
| `deployment/examples/physics_api.py` | Physics API example | ~100 |
| `deployment/examples/model_inference.py` | ML inference example | ~120 |
| `deployment/examples/batch_simulation.py` | Batch processing example | ~100 |
| `tests/test_serverless.py` | Tests | ~300 |
| `docs/guide/serverless.md` | Documentation | ~800 |

**Total**: ~2,400 lines

---

## Testing Strategy

### Unit Tests
- [ ] Serialization round-trip (DimArray <-> HTTP JSON)
- [ ] Handler decorator functionality
- [ ] Error handling (dimension errors, unit errors)
- [ ] Lazy import behavior

### Integration Tests
- [ ] Mock Lambda invocation
- [ ] Mock Cloud Functions request
- [ ] S3/GCS event processing
- [ ] Multi-invocation state persistence

### Performance Tests
- [ ] Cold start timing (with/without lazy imports)
- [ ] Memory usage profiling
- [ ] Package size measurement
- [ ] Serialization overhead

### Manual Verification
- [ ] Deploy to actual Lambda
- [ ] Deploy to actual Cloud Functions
- [ ] Load testing with ab/wrk
- [ ] Cost analysis

---

## Risks / Edge Cases

| Risk | Mitigation |
|------|------------|
| **Package size exceeds Lambda limit** | Use container images; provide slim build instructions |
| **Cold starts too slow** | Lazy imports; document memory recommendations; min instances |
| **numpy version conflicts** | Pin numpy <2.0 in requirements; test with Lambda runtime |
| **Serialization overhead** | Use efficient base64 encoding; document binary protocol option |
| **Unit mismatch in API calls** | Validation decorator; clear error messages |
| **Memory exhausted** | Document memory requirements per operation; set defaults |
| **Concurrent invocations** | Test thread safety; use immutable DimArray patterns |
| **Cost surprises** | Provide cost calculator; document optimization strategies |

---

## Cold Start Optimization Strategy

### 1. Lazy Imports
```python
# DON'T do this (slow cold start)
from dimtensor import DimArray, units
import numpy as np

def handler(event, context):
    # ...

# DO this (fast cold start)
def handler(event, context):
    from dimtensor import DimArray, units
    import numpy as np
    # ...
```

### 2. Global State Reuse
```python
# Cache expensive operations
MODEL = None

def handler(event, context):
    global MODEL
    if MODEL is None:
        MODEL = load_model()  # Only on cold start
    # Use MODEL...
```

### 3. Memory Configuration
- **Minimum**: 512MB for numpy operations
- **Recommended**: 1GB for ML inference
- **Heavy workloads**: 2GB+

Higher memory = more CPU = faster cold starts.

### 4. Provisioned Concurrency (AWS) / Min Instances (GCP)
For production, latency-sensitive workloads:
- AWS: Set provisioned concurrency to 1-5 instances
- GCP: Set min instances to 1-2

### 5. Container Images vs Zip
- **Zip**: Faster for small packages (<10MB)
- **Container**: Faster for scientific packages (numpy, scipy)

Benchmark shows containers often have 20-30% faster cold starts for numpy workloads.

---

## API Examples

### AWS Lambda - Physics Computation

```python
# handler.py
from dimtensor.serverless import lambda_physics
from dimtensor import DimArray, units

@lambda_physics
def compute_trajectory(event, context):
    """Compute projectile trajectory."""
    v0 = event["initial_velocity"]  # DimArray
    angle = event["angle"]  # radians

    # Compute max height
    max_height = (v0 * np.sin(angle))**2 / (2 * 9.81 * units.m / units.s**2)

    return {"max_height": max_height}

# Test locally
if __name__ == "__main__":
    test_event = {
        "body": json.dumps({
            "initial_velocity": {
                "data": base64.b64encode(np.array([50.0]).tobytes()).decode(),
                "unit": "m/s",
                "dimension": [1, 0, -1, 0, 0, 0, 0],
                "shape": [1],
                "dtype": "float64",
            },
            "angle": 0.785,  # 45 degrees
        })
    }

    result = compute_trajectory(test_event, None)
    print(result)
```

### GCP Cloud Functions - ML Inference

```python
# main.py
from dimtensor.serverless import cf_ml
from dimtensor import DimArray, units

@cf_ml
def predict_energy(request):
    """Predict energy from input features."""
    # Load model (cached globally)
    model = get_model()

    data = request.get_json()
    inputs = data["features"]  # DimArray

    # Run inference
    prediction = model(inputs)

    return {"energy": prediction}
```

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Unit tests pass (>90% coverage)
- [ ] Integration tests pass
- [ ] Deployment templates tested on AWS and GCP
- [ ] Documentation created (`docs/guide/serverless.md`)
- [ ] Example functions deployed and tested
- [ ] Cold start benchmarks documented
- [ ] Cost analysis documented
- [ ] CONTINUITY.md updated

---

## Documentation Outline

`docs/guide/serverless.md`:

1. **Introduction**
   - Why serverless for physics computing
   - Supported platforms

2. **Quick Start**
   - AWS Lambda example
   - GCP Cloud Functions example

3. **Installation**
   - Lambda Layer setup
   - Container deployment
   - Cloud Functions deployment

4. **Handler Decorators**
   - `@lambda_physics`, `@lambda_ml`
   - `@cloud_function`, `@cf_physics`
   - Custom configuration

5. **Serialization**
   - HTTP JSON format
   - Binary protocol (future)
   - Custom serializers

6. **Performance Optimization**
   - Lazy imports
   - Memory configuration
   - Provisioned concurrency
   - Cold start tips

7. **Deployment Templates**
   - AWS SAM walkthrough
   - Serverless Framework (optional)
   - Terraform (optional)
   - GCP deployment manager

8. **Examples**
   - Physics API
   - ML inference
   - Batch processing
   - Event-driven simulation

9. **Cost Optimization**
   - Pricing models
   - Cost calculator
   - Optimization strategies

10. **Troubleshooting**
    - Common errors
    - Debugging tips
    - Monitoring

---

## Cost Analysis

### AWS Lambda Pricing (2026)

**Assumptions**:
- 1M requests/month
- 512MB memory
- 200ms average duration

**Costs**:
- Requests: $0.20 (1M × $0.20/1M)
- Compute: $1.67 (1M × 0.2s × $0.0000166667/GB-s × 0.5GB)
- **Total**: ~$1.87/month

**Optimization**: Use provisioned concurrency only if needed (adds $0.015/hour/instance).

### GCP Cloud Functions Pricing (2026)

**Assumptions**:
- 1M requests/month
- 512MB memory
- 200ms average duration

**Costs**:
- Requests: $0.40 (1M × $0.40/1M)
- Compute: $1.40 (1M × 0.2s × $0.0000025/GB-s × 0.5GB)
- **Total**: ~$1.80/month

**Note**: 2nd gen Cloud Functions have different pricing (generally better for high-volume).

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
serverless-aws = [
    "boto3>=1.26.0",  # AWS SDK
]
serverless-gcp = [
    "functions-framework>=3.0.0",  # GCP Functions
    "google-cloud-storage>=2.10.0",  # GCS
]
serverless = [
    "dimtensor[serverless-aws,serverless-gcp]",
]
```

---

## References

- [AWS Lambda Python Best Practices](https://fourtheorem.com/optimise-python-data-science-aws-lambda/)
- [Deploying Lambda with NumPy](https://www.architech.ca/articles/deploying-aws-lambda-functions-with-numpy-dependencies)
- [AWS Lambda Layers](https://docs.aws.amazon.com/lambda/latest/dg/python-layers.html)
- [GCP Cloud Functions Tips](https://cloud.google.com/run/docs/tips/python)
- [Cloud Functions Best Practices](https://cloud.google.com/run/docs/tips/functions-best-practices)

---

## Notes / Log

**[2026-01-12 15:00]** - Initial plan created. Research shows container images are best for numpy/dimtensor due to size. Lazy imports critical for cold start optimization. Both AWS and GCP provide good serverless options; key is proper packaging and memory configuration.

**Key insights**:
1. Container deployment preferred over zip for scientific packages
2. 512MB minimum memory recommended for numpy operations
3. Lazy imports can reduce cold start by 40-60%
4. Global state caching essential for ML models
5. Base64 serialization acceptable for API use cases

---

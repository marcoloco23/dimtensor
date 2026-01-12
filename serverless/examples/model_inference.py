"""Example: ML model inference with dimtensor.

Demonstrates:
- Loading trained physics model
- Unit-aware inference
- Batch processing
- Model state caching (cold start optimization)

Deploy to AWS Lambda or Google Cloud Functions.
"""

import json
from typing import Any
from dimtensor import DimArray, units
from dimtensor.serverless import lambda_ml

# Global scope - initialized once, reused across invocations
# This is the key to efficient serverless ML inference
MODEL = None
SCALER = None


def load_model():
    """Load model once (cold start only).

    In production, this would load from S3/GCS or include model in
    deployment package. For this demo, we create a simple model.
    """
    global MODEL, SCALER

    if MODEL is not None:
        return  # Already loaded

    # Lazy imports (only on cold start)
    try:
        import torch
        from dimtensor.torch import DimLinear, DimScaler

        print("Loading physics model (cold start)...")

        # Create simple model
        # In production: MODEL = torch.load("model.pt")
        MODEL = DimLinear(3, 1, units.m, units.J)
        MODEL.eval()

        # Create scaler
        # In production: load from saved state
        SCALER = DimScaler(method="standard")

        print("Model loaded successfully")

    except ImportError:
        # Fallback if torch not available
        print("PyTorch not available, using dummy model")
        MODEL = "dummy"
        SCALER = "dummy"


@lambda_ml
def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Run inference on physics model.

    Input:
    {
        "inputs": DimArray with shape (N, 3) and units [m]
    }

    Output:
    {
        "predictions": DimArray with shape (N, 1) and units [J]
    }

    Example request:
    {
        "inputs": {
            "data": "...",  # base64 encoded array
            "unit": "m",
            "dimension": [1, 0, 0, 0, 0, 0, 0],
            "shape": [10, 3],
            "dtype": "float64",
            "scale": 1.0,
            "uncertainty": null
        }
    }
    """
    # Load model (only once)
    load_model()

    # Get inputs
    inputs = event["inputs"]  # Already a DimArray (decorator handles it)

    # Validate input shape
    if len(inputs.shape) != 2 or inputs.shape[1] != 3:
        raise ValueError(
            f"Expected input shape (N, 3), got {inputs.shape}"
        )

    # Check if using real PyTorch model
    if MODEL == "dummy":
        # Dummy prediction (for testing without torch)
        import numpy as np
        predictions_data = np.sum(inputs._data**2, axis=1, keepdims=True)
        result = DimArray(predictions_data, units.J)
        return {"predictions": result}

    # Real PyTorch inference
    try:
        import torch
        from dimtensor.torch import DimTensor

        # Convert to DimTensor
        inputs_tensor = DimTensor(
            torch.from_numpy(inputs._data).float(), inputs.unit
        )

        # Scale for model (normalize inputs)
        # Note: In production, fit scaler on training data
        inputs_scaled = SCALER.transform(inputs_tensor)

        # Run inference
        with torch.no_grad():
            outputs_scaled = MODEL(inputs_scaled)

        # Inverse transform
        # Note: Need output unit for inverse transform
        outputs = SCALER.inverse_transform(outputs_scaled, units.J)

        # Convert back to DimArray for response
        result = DimArray(outputs.data.numpy(), outputs.unit)

        return {"predictions": result}

    except Exception as e:
        # If inference fails, return error with details
        raise RuntimeError(f"Inference failed: {str(e)}")


# For Google Cloud Functions deployment
def gcp_handler(request):
    """GCP Cloud Functions entry point."""
    from dimtensor.serverless import cf_ml

    @cf_ml
    def handler_impl(req):
        data = req.get_json()
        # Convert to Lambda-style event
        event = data
        context = None
        return lambda_handler(event, context)

    return handler_impl(request)

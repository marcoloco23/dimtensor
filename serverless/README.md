# Serverless Deployment for dimtensor

Deploy physics computations to AWS Lambda and Google Cloud Functions.

## Quick Start

### AWS Lambda

1. Install AWS SAM CLI:
   ```bash
   pip install aws-sam-cli
   ```

2. Deploy:
   ```bash
   cd serverless/aws
   sam build
   sam deploy --guided
   ```

### Google Cloud Functions

1. Install gcloud CLI and authenticate:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. Deploy:
   ```bash
   cd serverless/gcp/cloudfunctions
   gcloud functions deploy compute-energy \
     --runtime python312 \
     --trigger-http \
     --allow-unauthenticated \
     --entry-point compute_energy \
     --memory 512MB
   ```

## Performance Tips

### Cold Start Optimization

1. **Use Container Images** (AWS Lambda):
   - Better performance for numpy/scientific packages
   - See `aws/Dockerfile`

2. **Lazy Imports**:
   - Import heavy modules inside handlers
   - Already enabled in decorators

3. **Memory Configuration**:
   - Minimum 512MB for physics computations
   - 1GB+ for ML inference
   - Higher memory = more CPU = faster cold starts

4. **Provisioned Concurrency** (AWS):
   ```bash
   aws lambda put-provisioned-concurrency-config \
     --function-name physics-compute \
     --provisioned-concurrent-executions 2
   ```

5. **Minimum Instances** (GCP):
   ```bash
   gcloud functions deploy compute-energy \
     --min-instances 1
   ```

## Example Functions

See `examples/` for complete implementations:

- `physics_api.py` - Generic physics calculations
- `model_inference.py` - ML model inference
- `batch_simulation.py` - S3/GCS-triggered batch processing

## Documentation

For full documentation, see: https://dimtensor.readthedocs.io/en/latest/guide/serverless.html

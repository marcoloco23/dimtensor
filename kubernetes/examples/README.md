# Kubernetes Examples for dimtensor

This directory contains simple, ready-to-use Kubernetes YAML manifests for common dimtensor workloads.

## Quick Start

```bash
# Deploy batch processing job
kubectl apply -f batch-job.yaml

# Deploy training job (requires GPU nodes)
kubectl apply -f training-job.yaml

# Deploy inference API
kubectl apply -f inference-deployment.yaml
kubectl apply -f service.yaml

# Enable autoscaling
kubectl apply -f hpa.yaml
```

## Examples

### 1. Batch Processing (`batch-job.yaml`)

Runs parameter sweeps or Monte Carlo simulations in parallel.

**Use cases:**
- Buckingham Pi theorem analysis across parameter space
- Dimensional analysis of large datasets
- Physics parameter optimization

**Features:**
- Parallel execution (10 workers by default)
- ConfigMap for physics parameters
- EmptyDir for temporary data storage
- Automatic cleanup after 24 hours

**Deploy:**
```bash
kubectl apply -f batch-job.yaml

# Monitor progress
kubectl get jobs
kubectl logs -l component=batch --tail=50

# Get results (if using persistent storage)
kubectl exec -it dimtensor-batch-job-<pod-id> -- cat /data/output/result_0.txt
```

### 2. Training Job (`training-job.yaml`)

Trains Physics-Informed Neural Networks (PINNs) with GPU acceleration.

**Use cases:**
- PINN training for differential equations
- GNN optimization for molecular dynamics
- Transformer training for physics data

**Features:**
- GPU support (NVIDIA CUDA)
- Init container for data preparation
- Checkpoint saving
- MLflow/W&B integration (optional)

**Prerequisites:**
- GPU nodes with NVIDIA device plugin
- `nvidia.com/gpu` resource available

**Deploy:**
```bash
# Verify GPU availability
kubectl get nodes -o json | jq '.items[].status.allocatable."nvidia.com/gpu"'

# Deploy training job
kubectl apply -f training-job.yaml

# Monitor training
kubectl logs -f dimtensor-training-job-<pod-id>

# Check GPU usage
kubectl exec -it dimtensor-training-job-<pod-id> -- nvidia-smi
```

### 3. Inference API (`inference-deployment.yaml`)

RESTful API for physics predictions with dimensional validation.

**Use cases:**
- Real-time unit conversion
- Dimensional consistency validation
- Physics equation evaluation

**Features:**
- 3 replicas for high availability
- Health/readiness/startup probes
- Prometheus metrics endpoint
- Rolling updates with zero downtime
- Pod anti-affinity for node distribution

**Endpoints:**
- `GET /health` - Health check
- `GET /ready` - Readiness check
- `POST /api/v1/validate` - Validate equation dimensions
- `POST /api/v1/convert` - Convert units
- `GET /metrics` - Prometheus metrics

**Deploy:**
```bash
kubectl apply -f inference-deployment.yaml
kubectl apply -f service.yaml

# Wait for rollout
kubectl rollout status deployment/dimtensor-inference

# Test locally
kubectl port-forward svc/dimtensor-inference 8080:80

# Validate dimensions
curl -X POST http://localhost:8080/api/v1/validate \
  -H "Content-Type: application/json" \
  -d '{"mass": 10, "acceleration": 9.81}'

# Convert units
curl -X POST http://localhost:8080/api/v1/convert \
  -H "Content-Type: application/json" \
  -d '{"value": 1000, "from_unit": "m", "to_unit": "km"}'
```

### 4. Service (`service.yaml`)

Exposes inference API within cluster or externally.

**Options:**
- `ClusterIP` (default) - Internal cluster access
- `LoadBalancer` - External access with cloud load balancer
- `Ingress` - External access with domain name and TLS

**Deploy ClusterIP:**
```bash
kubectl apply -f service.yaml
```

**Deploy LoadBalancer:**
```bash
# Edit service.yaml: change type to LoadBalancer
kubectl apply -f service.yaml

# Get external IP
kubectl get svc dimtensor-inference-external
```

**Deploy Ingress:**
```bash
# Edit service.yaml: set your domain in Ingress spec
kubectl apply -f service.yaml

# Wait for Ingress
kubectl get ingress dimtensor-inference
```

### 5. Horizontal Pod Autoscaler (`hpa.yaml`)

Automatically scales inference API based on load.

**Metrics:**
- CPU utilization (70% threshold)
- Memory utilization (80% threshold, optional)

**Scaling:**
- Min replicas: 2
- Max replicas: 10
- Scale up: immediate, add up to 4 pods
- Scale down: gradual (5 min wait), remove up to 2 pods

**Prerequisites:**
- Metrics Server installed

**Deploy:**
```bash
# Verify metrics-server
kubectl get deployment metrics-server -n kube-system

# Deploy HPA
kubectl apply -f hpa.yaml

# Monitor autoscaling
kubectl get hpa
kubectl describe hpa dimtensor-inference-hpa

# Load test to trigger scaling
kubectl run -it --rm load-generator --image=busybox -- /bin/sh
# Inside the pod:
while true; do wget -q -O- http://dimtensor-inference/health; done
```

## Configuration

### Image Variants

Choose the appropriate image for your workload:

- `dimtensor:latest` - Minimal (NumPy only) - 400MB
- `dimtensor:cpu-full` - All dependencies (PyTorch, JAX, etc.) - 2GB
- `dimtensor:ml` - GPU-enabled (PyTorch CUDA, JAX CUDA) - 3GB
- `dimtensor:jupyter` - Interactive notebooks - 2.5GB

### Resource Limits

Default resource requests/limits:

| Workload | CPU Request | CPU Limit | Memory Request | Memory Limit | GPU |
|----------|-------------|-----------|----------------|--------------|-----|
| Batch    | 500m        | 2000m     | 1Gi            | 4Gi          | 0   |
| Training | 2000m       | 4000m     | 8Gi            | 16Gi         | 1   |
| Inference| 500m        | 2000m     | 2Gi            | 8Gi          | 0   |

Adjust based on your workload:

```yaml
resources:
  requests:
    cpu: "1000m"      # 1 CPU core
    memory: "4Gi"     # 4 GB RAM
  limits:
    cpu: "4000m"      # 4 CPU cores
    memory: "16Gi"    # 16 GB RAM
```

### Security

All examples follow security best practices:

- Non-root user (UID 1000)
- No privilege escalation
- Dropped capabilities
- Read-only root filesystem (where possible)

## Troubleshooting

### Pod stuck in Pending

```bash
kubectl describe pod <pod-name>

# Common causes:
# - Insufficient resources: Check node capacity
# - GPU not available: Verify nvidia.com/gpu resource
# - Image pull error: Check image name and registry
```

### Pod OOMKilled

```bash
kubectl describe pod <pod-name> | grep -i oom

# Solution: Increase memory limits
resources:
  limits:
    memory: "16Gi"  # Increase from default
```

### Training job fails on GPU

```bash
# Check GPU availability
kubectl exec -it <pod-name> -- nvidia-smi

# Verify CUDA
kubectl exec -it <pod-name> -- python -c "import torch; print(torch.cuda.is_available())"

# Check node labels
kubectl get nodes --show-labels | grep gpu
```

### Inference API not responding

```bash
# Check pod status
kubectl get pods -l component=inference

# Check logs
kubectl logs -l component=inference --tail=100

# Test readiness probe
kubectl port-forward <pod-name> 8080:8080
curl http://localhost:8080/ready
```

## Next Steps

For production deployments with advanced features:

- Use the [Helm chart](../helm/dimtensor/) for parameterized deployments
- Set up persistent storage for checkpoints and results
- Configure Prometheus/Grafana for monitoring
- Integrate with MLflow or Weights & Biases for experiment tracking
- Enable network policies for security
- Set up CI/CD with GitOps (ArgoCD, Flux)

## See Also

- [Kubernetes README](../README.md) - Overview and architecture
- [Helm Chart](../helm/dimtensor/) - Flexible production deployment
- [Docker Images](../../docker/) - Image build instructions
- [dimtensor Documentation](https://github.com/marcoloco23/dimtensor)

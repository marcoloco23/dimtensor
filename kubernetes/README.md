# Kubernetes Deployment Guide for dimtensor

Production-ready Kubernetes manifests and Helm charts for deploying dimtensor-based physics ML workloads.

## Overview

This directory provides two deployment options:

1. **Simple YAML Examples** (`examples/`) - Quick start with ready-to-use manifests
2. **Helm Chart** (`helm/dimtensor/`) - Flexible, parameterized deployments for production

## Quick Start

### Option 1: Simple YAML Deployment

Deploy inference API in 30 seconds:

```bash
# Deploy inference API
kubectl apply -f examples/inference-deployment.yaml
kubectl apply -f examples/service.yaml

# Wait for rollout
kubectl rollout status deployment/dimtensor-inference

# Test locally
kubectl port-forward svc/dimtensor-inference 8080:80
curl http://localhost:8080/health
```

### Option 2: Helm Chart Deployment

Install with default settings:

```bash
# Add dimtensor Helm repository (future)
# helm repo add dimtensor https://charts.dimtensor.org
# helm repo update

# Install from local chart
cd helm/dimtensor
helm install dimtensor . --values values-inference.yaml

# Get status
helm status dimtensor
kubectl get pods -l app.kubernetes.io/name=dimtensor
```

## Use Cases

### 1. Batch Processing

Run parameter sweeps or Monte Carlo simulations:

```bash
# Simple YAML
kubectl apply -f examples/batch-job.yaml

# Helm chart
helm install dimtensor-batch ./helm/dimtensor -f helm/dimtensor/values-batch.yaml
```

**Use cases:**
- Buckingham Pi theorem analysis across parameter space
- Dimensional analysis of large datasets
- Physics parameter optimization

### 2. ML Training (GPU)

Train Physics-Informed Neural Networks:

```bash
# Prerequisites: GPU nodes with NVIDIA device plugin
kubectl get nodes -o json | jq '.items[].status.allocatable."nvidia.com/gpu"'

# Deploy training job
kubectl apply -f examples/training-job.yaml

# Or with Helm
helm install dimtensor-training ./helm/dimtensor -f helm/dimtensor/values-training.yaml

# Monitor training
kubectl logs -f job/dimtensor-training-job
```

**Use cases:**
- PINN training for differential equations
- GNN optimization for molecular dynamics
- Transformer training for physics data

### 3. Inference API

RESTful API for physics predictions:

```bash
# Deploy with autoscaling
kubectl apply -f examples/inference-deployment.yaml
kubectl apply -f examples/service.yaml
kubectl apply -f examples/hpa.yaml

# Or with Helm (recommended)
helm install dimtensor-api ./helm/dimtensor -f helm/dimtensor/values-inference.yaml

# Test API
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

## Architecture

### Image Variants

Choose the appropriate image for your workload:

| Variant | Size | Use Case | PyTorch | JAX | GPU |
|---------|------|----------|---------|-----|-----|
| `latest` | 400MB | CPU batch jobs | ✗ | ✗ | ✗ |
| `cpu-full` | 2GB | Inference API | ✓ | ✓ | ✗ |
| `ml` | 3GB | GPU training | ✓ | ✓ | ✓ |
| `jupyter` | 2.5GB | Interactive notebooks | ✓ | ✓ | ✗ |

### Resource Recommendations

| Workload | CPU Request | Memory Request | GPU | Replicas |
|----------|-------------|----------------|-----|----------|
| Batch Processing | 500m | 1Gi | 0 | N (parallel) |
| ML Training | 2000m | 8Gi | 1 | 1 |
| Inference API | 500m | 2Gi | 0 | 2-10 (HPA) |

### Network Architecture

```
┌─────────────┐
│   Ingress   │  (Optional: TLS, domain routing)
└──────┬──────┘
       │
┌──────▼──────┐
│   Service   │  (ClusterIP/LoadBalancer)
└──────┬──────┘
       │
┌──────▼──────────────────┐
│   Deployment (3 pods)   │  (Inference API)
│   - Pod 1 (Node A)      │
│   - Pod 2 (Node B)      │
│   - Pod 3 (Node C)      │
└─────────────────────────┘
       │
┌──────▼──────┐
│     HPA     │  (Autoscaling 2-10 replicas)
└─────────────┘
```

## Prerequisites

### Required

- Kubernetes cluster (1.21+)
- `kubectl` configured
- Container runtime (Docker, containerd, CRI-O)

### Optional

- Helm 3.0+ (for Helm chart deployment)
- NVIDIA device plugin (for GPU workloads)
- Metrics Server (for autoscaling)
- Ingress controller (for external access)
- cert-manager (for TLS certificates)

### GPU Setup

For GPU workloads, install NVIDIA device plugin:

```bash
# Install NVIDIA device plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify GPU availability
kubectl get nodes -o json | jq '.items[].status.allocatable."nvidia.com/gpu"'

# Label GPU nodes
kubectl label nodes <node-name> accelerator=nvidia-gpu
```

## Configuration

### Image Configuration

Specify image in deployment:

```yaml
containers:
- name: dimtensor
  image: dimtensor:cpu-full  # Change variant here
  imagePullPolicy: IfNotPresent
```

### Resource Limits

Adjust based on workload:

```yaml
resources:
  requests:
    cpu: "1000m"      # 1 CPU core
    memory: "4Gi"     # 4 GB RAM
  limits:
    cpu: "4000m"      # 4 CPU cores
    memory: "16Gi"    # 16 GB RAM
```

For GPU:

```yaml
resources:
  requests:
    nvidia.com/gpu: "1"
  limits:
    nvidia.com/gpu: "1"
```

### Autoscaling

Configure HPA thresholds:

```yaml
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70  # Scale at 70% CPU
```

### Persistence

Enable persistent storage:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dimtensor-data
spec:
  accessModes:
  - ReadWriteOnce
  storageClassName: standard  # Adjust for your cluster
  resources:
    requests:
      storage: 50Gi
```

## Security

All examples follow security best practices:

- **Non-root user**: Containers run as UID 1000
- **No privilege escalation**: `allowPrivilegeEscalation: false`
- **Dropped capabilities**: `capabilities.drop: [ALL]`
- **Security context**: Applied to all pods

### Network Policies

Restrict traffic to inference API:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: dimtensor-inference-policy
spec:
  podSelector:
    matchLabels:
      app: dimtensor
      component: inference
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          role: api-client
    ports:
    - protocol: TCP
      port: 8080
```

## Monitoring

### Prometheus Integration

Inference API exposes metrics at `/metrics`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: dimtensor-inference
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8080"
    prometheus.io/path: "/metrics"
```

Available metrics:
- `dimtensor_requests_total` - Total requests
- `dimtensor_errors_total` - Total errors
- `dimtensor_uptime_seconds` - Service uptime

### Logging

View logs:

```bash
# Deployment logs
kubectl logs -l app=dimtensor,component=inference --tail=100 -f

# Job logs
kubectl logs job/dimtensor-batch-job --tail=100

# Previous container logs (if pod restarted)
kubectl logs <pod-name> --previous
```

## Troubleshooting

### Pod Stuck in Pending

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

### GPU Not Available

```bash
# Check GPU nodes
kubectl get nodes -o json | jq '.items[].status.allocatable."nvidia.com/gpu"'

# Check NVIDIA device plugin
kubectl get pods -n kube-system | grep nvidia

# Verify from inside pod
kubectl exec -it <pod-name> -- nvidia-smi
```

### Training Job Fails

```bash
# Check logs
kubectl logs job/dimtensor-training-job

# Check events
kubectl describe job/dimtensor-training-job

# Common issues:
# - Out of memory: Reduce batch size or increase memory limit
# - CUDA out of memory: Reduce model size or batch size
# - Data not available: Check init container logs
```

### HPA Not Scaling

```bash
# Check HPA status
kubectl describe hpa dimtensor-inference-hpa

# Verify metrics-server
kubectl get deployment metrics-server -n kube-system

# Check metrics
kubectl top pods -l app=dimtensor
```

## Advanced Topics

### Multi-Cluster Deployment

Deploy across multiple clusters for high availability:

```bash
# Cluster 1 (us-west)
kubectl --context=us-west apply -f examples/inference-deployment.yaml

# Cluster 2 (us-east)
kubectl --context=us-east apply -f examples/inference-deployment.yaml

# Global load balancer (cloud provider specific)
# - GCP: Global Load Balancer
# - AWS: Route53 + ALB
# - Azure: Traffic Manager
```

### CI/CD Integration

Example GitOps workflow with ArgoCD:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: dimtensor-inference
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/marcoloco23/dimtensor
    targetRevision: main
    path: kubernetes/helm/dimtensor
    helm:
      valueFiles:
      - values-inference.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

### Experiment Tracking

Integrate with MLflow:

```yaml
env:
- name: MLFLOW_TRACKING_URI
  value: "http://mlflow-server.mlflow.svc.cluster.local:5000"
- name: MLFLOW_EXPERIMENT_NAME
  value: "pinn-training"
```

Or Weights & Biases:

```yaml
env:
- name: WANDB_API_KEY
  valueFrom:
    secretKeyRef:
      name: wandb-secret
      key: api-key
- name: WANDB_PROJECT
  value: "dimtensor-physics"
```

## Examples

See the following for complete examples:

- [Simple YAML Examples](examples/README.md) - Quick start guides
- [Helm Chart](helm/dimtensor/README.md) - Advanced configuration
- [Example Workloads](examples/workloads/) - Python scripts for common use cases

## FAQ

**Q: Which image variant should I use?**

A: Use `latest` for CPU-only batch jobs, `cpu-full` for inference APIs with all features, `ml` for GPU training, and `jupyter` for interactive exploration.

**Q: How do I persist training checkpoints?**

A: Use a PersistentVolumeClaim mounted to `/checkpoints`. See `examples/training-job.yaml` for an example.

**Q: Can I use multiple GPUs?**

A: Yes, set `resources.gpu.count: 2` in Helm values or request multiple GPUs in the YAML manifest.

**Q: How do I enable HTTPS for the inference API?**

A: Use an Ingress with cert-manager for TLS. See `examples/service.yaml` for configuration.

**Q: How do I scale to zero?**

A: Use Knative Serving or KEDA for scale-to-zero functionality. Set `autoscaling.minReplicas: 0` requires these tools.

## Support

- GitHub Issues: https://github.com/marcoloco23/dimtensor/issues
- Documentation: https://github.com/marcoloco23/dimtensor
- Examples: https://github.com/marcoloco23/dimtensor/tree/main/kubernetes/examples

## License

MIT License - see LICENSE file for details

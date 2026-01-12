# Plan: Kubernetes Templates for dimtensor

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Provide production-ready Kubernetes manifests and Helm charts for deploying dimtensor-based physics ML workloads, enabling batch processing, one-off computations, and API services with GPU support and horizontal autoscaling.

---

## Background

dimtensor is increasingly used for physics ML workloads that require:
- Large-scale batch processing (parameter sweeps, Monte Carlo simulations)
- One-off expensive computations (GNN training, transformer inference)
- API endpoints for real-time physics predictions
- GPU acceleration for PyTorch/JAX operations
- Horizontal scaling to handle variable loads

Currently, users must create their own deployment configurations. This creates barriers to production deployment and prevents standardization of best practices. The goal is to provide reference templates that work out-of-the-box for common dimtensor use cases.

Key use cases:
1. **Batch Processing**: Physics simulations, dataset preprocessing, analysis pipelines
2. **Training Jobs**: One-off PINN training, GNN optimization, transformer fine-tuning
3. **Inference API**: RESTful endpoints for physics predictions with units
4. **Experiment Tracking**: Integration with MLflow/W&B in cluster
5. **Data Pipeline**: Process physics data from CERN, LIGO, NOAA loaders

---

## Approach

### Option A: Raw YAML Manifests
- Description: Provide plain Kubernetes YAML files for each use case
- Pros:
  - Simple, no dependencies
  - Easy to understand and customize
  - Works with any K8s cluster
- Cons:
  - Hard to parameterize (must edit YAML directly)
  - No templating for different environments (dev/staging/prod)
  - Duplication across similar configs

### Option B: Helm Charts Only
- Description: Provide Helm charts with values.yaml for configuration
- Pros:
  - Highly parameterizable
  - Easy multi-environment deployment
  - Package management (versioning, rollback)
- Cons:
  - Requires Helm installation
  - Steeper learning curve for K8s beginners
  - Overkill for simple deployments

### Option C: Both Raw YAML + Helm Chart (Hybrid)
- Description: Provide both simple YAML examples AND a comprehensive Helm chart
- Pros:
  - Beginners can use YAML examples
  - Production users get Helm flexibility
  - YAML serves as documentation for Helm chart
  - Best of both worlds
- Cons:
  - More maintenance (keep both in sync)
  - More files to manage

### Decision: Option C - Hybrid Approach

**Rationale**: Different users have different needs. Physics researchers want simple YAML to get started quickly. ML engineers deploying to production need Helm's power. By providing both, we serve the full spectrum while using YAML as living documentation.

**Structure**:
```
k8s/
├── examples/           # Simple YAML for quick start
│   ├── batch-job.yaml
│   ├── training-job.yaml
│   ├── inference-deployment.yaml
│   └── README.md
├── helm/
│   └── dimtensor/      # Full Helm chart
│       ├── Chart.yaml
│       ├── values.yaml
│       ├── templates/
│       │   ├── deployment.yaml
│       │   ├── job.yaml
│       │   ├── service.yaml
│       │   ├── hpa.yaml
│       │   ├── configmap.yaml
│       │   └── _helpers.tpl
│       └── README.md
└── README.md           # Top-level guide
```

---

## Implementation Steps

### Phase 1: Docker Foundation (Prerequisite)
1. [ ] Create base Dockerfile for dimtensor (if task #254 incomplete)
   - Python 3.10+ base image
   - NumPy <2.0 constraint
   - Optional: PyTorch, JAX, all deps
   - Multi-stage build for smaller images
2. [ ] Create GPU-enabled Dockerfile variant
   - CUDA base image (nvidia/cuda:12.1-runtime-ubuntu22.04)
   - PyTorch with CUDA support
   - cuDNN libraries
3. [ ] Test Docker images locally

### Phase 2: Simple YAML Examples
4. [ ] Create `k8s/examples/batch-job.yaml`
   - Job resource for parameter sweep
   - ConfigMap for physics parameters
   - Persistent volume for results
   - Example: Buckingham Pi analysis batch
5. [ ] Create `k8s/examples/training-job.yaml`
   - Job with GPU resources
   - Init container for data download
   - EmptyDir for checkpoints
   - Example: PINN training from equation database
6. [ ] Create `k8s/examples/inference-deployment.yaml`
   - Deployment with 3 replicas
   - Service (ClusterIP)
   - Liveness/readiness probes
   - Example: Physics prediction REST API
7. [ ] Create `k8s/examples/inference-service.yaml`
   - Service (LoadBalancer) for external access
   - Optional: Ingress configuration
8. [ ] Create `k8s/examples/hpa.yaml`
   - HorizontalPodAutoscaler
   - CPU-based scaling (70% threshold)
   - Min 2, max 10 replicas
9. [ ] Document all examples in `k8s/examples/README.md`

### Phase 3: Helm Chart Structure
10. [ ] Create Helm chart skeleton: `helm create k8s/helm/dimtensor`
11. [ ] Define `Chart.yaml` metadata
    - name: dimtensor
    - version: 1.0.0 (chart version)
    - appVersion: 4.5.0 (dimtensor version)
    - description, keywords, maintainers
12. [ ] Design `values.yaml` with sensible defaults
    - Image configuration (repo, tag, pull policy)
    - Resource limits (CPU, memory, GPU)
    - Autoscaling settings
    - Service configuration
    - Job/CronJob settings
    - Environment variables

### Phase 4: Helm Templates - Core Resources
13. [ ] Create `templates/deployment.yaml`
    - Parameterized replicas, image, resources
    - GPU node affinity when GPU requested
    - Volume mounts for data/checkpoints
    - Environment variable injection
    - Security context (non-root)
14. [ ] Create `templates/job.yaml`
    - Batch/training job template
    - Conditional GPU resources
    - Parallelism/completions
    - Backoff limit, TTL
15. [ ] Create `templates/cronjob.yaml`
    - Scheduled physics analysis
    - Suspend flag
    - Concurrency policy
16. [ ] Create `templates/service.yaml`
    - ClusterIP/LoadBalancer/NodePort
    - Port configuration
    - Selector from deployment
17. [ ] Create `templates/hpa.yaml`
    - Conditional (enabled flag)
    - CPU/memory metrics
    - Min/max replicas
    - Behavior policies

### Phase 5: Helm Templates - Configuration & Storage
18. [ ] Create `templates/configmap.yaml`
    - Physics constants
    - Equation database snippets
    - Application config
19. [ ] Create `templates/secret.yaml`
    - API keys (Materials Project, CERN, etc.)
    - MLflow/W&B tokens
20. [ ] Create `templates/pvc.yaml`
    - Persistent volume claims
    - Storage class selection
    - Size configuration
21. [ ] Create `templates/_helpers.tpl`
    - Chart name helpers
    - Label generators
    - Image name builder
    - Resource name functions

### Phase 6: GPU Support
22. [ ] Add GPU resource configuration to values.yaml
    - nvidia.com/gpu count
    - GPU memory requirements
23. [ ] Add node affinity for GPU nodes in deployment template
    - nodeSelector for GPU nodes
    - Tolerations for GPU taints
24. [ ] Document NVIDIA device plugin requirement
    - Installation instructions
    - Verification steps
25. [ ] Add GPU example values file: `values-gpu.yaml`

### Phase 7: Testing & Documentation
26. [ ] Create `k8s/README.md` with:
    - Overview of K8s deployment options
    - Prerequisites (kubectl, helm, GPU plugin)
    - Quick start guide
    - Architecture diagrams
27. [ ] Create `k8s/helm/dimtensor/README.md` with:
    - Helm installation instructions
    - values.yaml parameter reference
    - Example configurations
    - Upgrade/rollback procedures
28. [ ] Create test script: `k8s/test-deployment.sh`
    - Deploy to local kind/minikube cluster
    - Verify resources created
    - Test health checks
    - Clean up
29. [ ] Add examples for each use case:
    - `values-batch.yaml` - Batch processing
    - `values-training.yaml` - ML training
    - `values-inference.yaml` - API service
    - `values-gpu-training.yaml` - GPU training

### Phase 8: Integration Examples
30. [ ] Create example physics workloads:
    - `examples/workloads/pinn-training.py` - PINN job
    - `examples/workloads/batch-analysis.py` - Buckingham Pi batch
    - `examples/workloads/inference-server.py` - Flask/FastAPI server
31. [ ] Add monitoring integration examples:
    - Prometheus metrics endpoint
    - Grafana dashboard config
32. [ ] Add MLflow/W&B integration:
    - Environment variables for tracking
    - Volume mounts for artifacts

---

## Files to Modify

| File | Change |
|------|--------|
| **NEW** k8s/README.md | Top-level K8s deployment guide |
| **NEW** k8s/examples/batch-job.yaml | Simple batch processing job |
| **NEW** k8s/examples/training-job.yaml | Simple training job with GPU |
| **NEW** k8s/examples/inference-deployment.yaml | Simple deployment example |
| **NEW** k8s/examples/inference-service.yaml | Service + optional Ingress |
| **NEW** k8s/examples/hpa.yaml | Horizontal Pod Autoscaler |
| **NEW** k8s/examples/README.md | Examples documentation |
| **NEW** k8s/helm/dimtensor/Chart.yaml | Helm chart metadata |
| **NEW** k8s/helm/dimtensor/values.yaml | Default configuration |
| **NEW** k8s/helm/dimtensor/values-batch.yaml | Batch workload preset |
| **NEW** k8s/helm/dimtensor/values-training.yaml | Training workload preset |
| **NEW** k8s/helm/dimtensor/values-inference.yaml | Inference API preset |
| **NEW** k8s/helm/dimtensor/values-gpu.yaml | GPU-enabled preset |
| **NEW** k8s/helm/dimtensor/templates/deployment.yaml | Deployment template |
| **NEW** k8s/helm/dimtensor/templates/job.yaml | Job template |
| **NEW** k8s/helm/dimtensor/templates/cronjob.yaml | CronJob template |
| **NEW** k8s/helm/dimtensor/templates/service.yaml | Service template |
| **NEW** k8s/helm/dimtensor/templates/hpa.yaml | HPA template |
| **NEW** k8s/helm/dimtensor/templates/configmap.yaml | ConfigMap template |
| **NEW** k8s/helm/dimtensor/templates/secret.yaml | Secret template |
| **NEW** k8s/helm/dimtensor/templates/pvc.yaml | PVC template |
| **NEW** k8s/helm/dimtensor/templates/_helpers.tpl | Helper functions |
| **NEW** k8s/helm/dimtensor/templates/NOTES.txt | Post-install notes |
| **NEW** k8s/helm/dimtensor/README.md | Helm chart documentation |
| **NEW** k8s/test-deployment.sh | Testing script |
| **NEW** k8s/examples/workloads/pinn-training.py | PINN training example |
| **NEW** k8s/examples/workloads/batch-analysis.py | Batch job example |
| **NEW** k8s/examples/workloads/inference-server.py | API server example |
| **NEW** Dockerfile | Base dimtensor image (if not exists) |
| **NEW** Dockerfile.gpu | GPU-enabled image |
| **NEW** .dockerignore | Docker build exclusions |
| README.md | Add Kubernetes deployment section |
| docs/guide/deployment.md | **NEW** or update with K8s guide |

---

## Testing Strategy

### Local Testing
- [ ] Test Docker images build successfully
  - Base image: `docker build -t dimtensor:latest .`
  - GPU image: `docker build -f Dockerfile.gpu -t dimtensor:gpu .`
  - Verify dimtensor imports and basic ops
- [ ] Test YAML examples on kind cluster
  - `kind create cluster --config k8s/kind-config.yaml`
  - Apply batch-job.yaml, verify completion
  - Apply inference-deployment.yaml, verify pods running
  - Apply hpa.yaml, verify autoscaler created

### Helm Testing
- [ ] Test Helm chart installation
  - `helm install dimtensor-test k8s/helm/dimtensor`
  - Verify all resources created
  - Check pod logs for startup
- [ ] Test Helm upgrades
  - Change values, run `helm upgrade`
  - Verify rollout
- [ ] Test each preset values file
  - Install with `helm install -f values-batch.yaml`
  - Verify correct resources created

### GPU Testing
- [ ] Verify GPU jobs on GPU-enabled cluster
  - GKE with GPU node pool OR
  - Local cluster with GPU passthrough
  - Check `nvidia-smi` in pod
  - Run simple PyTorch GPU operation

### Integration Testing
- [ ] Run PINN training job to completion
  - Verify checkpoints saved
  - Check logs for training progress
- [ ] Deploy inference API
  - Send HTTP requests
  - Verify unit handling in responses
- [ ] Test autoscaling
  - Load test inference API
  - Verify HPA scales up
  - Verify scale down after load

### Documentation Testing
- [ ] Follow README.md from scratch
  - Clean cluster
  - Execute all commands
  - Verify success
- [ ] Test all example commands
  - Copy-paste from docs
  - Ensure no errors

---

## Risks / Edge Cases

### Risk 1: Docker Image Size
- **Issue**: dimtensor[all] is very large (~3GB with all deps)
- **Mitigation**:
  - Multi-stage build to reduce final image
  - Separate images for different use cases (cpu, gpu, minimal)
  - Document image variants clearly

### Risk 2: NumPy Version Conflicts
- **Issue**: dimtensor requires numpy<2.0, some deps may pull 2.x
- **Mitigation**:
  - Pin numpy version in Dockerfile
  - Test image imports before pushing
  - Document version constraints

### Risk 3: GPU Node Affinity
- **Issue**: GPU pods scheduled on CPU nodes fail
- **Mitigation**:
  - Strong nodeSelector for GPU resources
  - Tolerations for GPU node taints
  - Clear error messages in pod events

### Risk 4: Persistent Storage
- **Issue**: Different clusters have different storage classes
- **Mitigation**:
  - Make storage class configurable
  - Default to standard/default
  - Document storage requirements

### Risk 5: Resource Limits
- **Issue**: Physics ML jobs can be memory-intensive
- **Mitigation**:
  - Sensible default limits in values.yaml
  - Document how to profile and set limits
  - Examples for different workload sizes

### Risk 6: API Authentication
- **Issue**: Inference API has no built-in auth
- **Mitigation**:
  - Document that user must add auth layer
  - Provide example with API gateway
  - Note this is just a template

### Edge Case: Out-of-Memory Kills
- **Handling**: Set appropriate memory limits, document OOMKilled debugging

### Edge Case: GPU Out-of-Memory
- **Handling**: Document batch size tuning, gradient checkpointing

### Edge Case: Job Failures
- **Handling**: Set backoffLimit in job template, document retry strategies

### Edge Case: Network Policies
- **Handling**: Templates work with default network policy, document custom policies

### Edge Case: Namespace Isolation
- **Handling**: All resources namespaced, document multi-tenant setup

---

## Definition of Done

- [ ] All YAML examples created and documented
- [ ] Helm chart fully templated and parameterized
- [ ] Docker images build successfully
- [ ] Tested on local kind cluster (CPU)
- [ ] GPU configuration documented (even if not tested on real GPU cluster)
- [ ] All preset values files created
- [ ] Example workload Python scripts included
- [ ] README.md updated with K8s section
- [ ] k8s/README.md comprehensive guide completed
- [ ] helm/dimtensor/README.md with full parameter reference
- [ ] Test script (test-deployment.sh) runs successfully
- [ ] CONTINUITY.md updated

---

## Notes / Log

**Design Decisions**:

1. **Hybrid Approach**: Provide both simple YAML and Helm chart to serve beginners and production users.

2. **Image Variants**: Three Docker images:
   - `dimtensor:latest` - CPU-only, minimal deps (numpy only)
   - `dimtensor:cpu-full` - CPU with all optional deps
   - `dimtensor:gpu` - GPU-enabled with PyTorch CUDA

3. **Resource Defaults**: Conservative defaults to avoid OOM:
   - CPU: request 500m, limit 2000m
   - Memory: request 2Gi, limit 8Gi
   - GPU: request/limit 1 (when enabled)

4. **Storage Strategy**: EmptyDir for temporary data, PVC for persistent data, ConfigMap for config

5. **Autoscaling**: CPU-based by default (70% threshold), custom metrics optional

6. **Security**: Non-root containers, read-only root filesystem where possible, security context in templates

7. **Monitoring**: Expose metrics endpoint in inference API for Prometheus scraping

8. **Example Workloads**: Real physics use cases from dimtensor capabilities:
   - PINN training (equations database + DimTensor)
   - Batch analysis (Buckingham Pi solver)
   - Inference API (dimensional validation)

**Integration Points**:
- MLflow/W&B via environment variables
- Prometheus metrics (future)
- External data sources (CERN, LIGO loaders run in init containers)

**Future Enhancements** (not in v5.0.0):
- Distributed training (PyTorch DDP, Horovod)
- Kubeflow Pipelines integration
- ArgoCD app manifests
- Kustomize overlays
- Istio service mesh configs

---

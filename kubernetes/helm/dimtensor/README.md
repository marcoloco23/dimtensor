# dimtensor Helm Chart

Official Helm chart for deploying dimtensor workloads on Kubernetes.

## TL;DR

```bash
# Install inference API
helm install dimtensor . -f values-inference.yaml

# Install training job
helm install dimtensor-training . -f values-training.yaml

# Install batch processing
helm install dimtensor-batch . -f values-batch.yaml
```

## Introduction

This chart bootstraps dimtensor deployments on Kubernetes using the Helm package manager.

It supports three workload types:
- **Deployment**: Long-running inference APIs
- **Job**: One-off batch processing or training
- **CronJob**: Scheduled periodic tasks

## Prerequisites

- Kubernetes 1.21+
- Helm 3.0+
- (Optional) NVIDIA device plugin for GPU workloads
- (Optional) Metrics Server for autoscaling

## Installing the Chart

### Install with default values (inference API)

```bash
helm install dimtensor .
```

### Install with preset configurations

```bash
# Inference API with autoscaling
helm install dimtensor-api . -f values-inference.yaml

# ML training job with GPU
helm install dimtensor-training . -f values-training.yaml

# Batch processing job
helm install dimtensor-batch . -f values-batch.yaml
```

### Install with custom values

```bash
# Create custom values file
cat > my-values.yaml <<EOF
replicaCount: 5
resources:
  requests:
    cpu: 1000m
    memory: 4Gi
EOF

# Install
helm install dimtensor . -f my-values.yaml
```

## Uninstalling the Chart

```bash
helm uninstall dimtensor
```

This removes all Kubernetes resources associated with the chart.

## Configuration

### Parameters

#### Global Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global.imageRegistry` | Global image registry override | `""` |
| `workloadType` | Workload type: `deployment`, `job`, or `cronjob` | `deployment` |

#### Image Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.registry` | Image registry | `docker.io` |
| `image.repository` | Image repository | `dimtensor` |
| `image.tag` | Image tag (defaults to chart appVersion) | `""` |
| `image.variant` | Image variant: `latest`, `cpu-full`, `ml`, `jupyter` | `latest` |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |
| `image.pullSecrets` | Image pull secrets | `[]` |

#### Deployment Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas (deployment only) | `3` |
| `strategy.type` | Deployment strategy | `RollingUpdate` |
| `strategy.rollingUpdate.maxSurge` | Max surge during rolling update | `1` |
| `strategy.rollingUpdate.maxUnavailable` | Max unavailable during rolling update | `0` |

#### Container Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `container.port` | Container port | `8080` |
| `container.command` | Command override | `[]` |
| `container.args` | Args override | `[]` |
| `container.env` | Environment variables | `[]` |
| `container.envFrom` | Environment from ConfigMap/Secret | `[]` |

#### Resource Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `resources.requests.cpu` | CPU request | `500m` |
| `resources.requests.memory` | Memory request | `2Gi` |
| `resources.limits.cpu` | CPU limit | `2000m` |
| `resources.limits.memory` | Memory limit | `8Gi` |
| `resources.gpu.enabled` | Enable GPU | `false` |
| `resources.gpu.count` | Number of GPUs | `1` |

#### Health Probes

| Parameter | Description | Default |
|-----------|-------------|---------|
| `livenessProbe.enabled` | Enable liveness probe | `true` |
| `livenessProbe.httpGet.path` | Liveness probe path | `/health` |
| `livenessProbe.initialDelaySeconds` | Initial delay | `10` |
| `livenessProbe.periodSeconds` | Period | `30` |
| `readinessProbe.enabled` | Enable readiness probe | `true` |
| `readinessProbe.httpGet.path` | Readiness probe path | `/ready` |
| `readinessProbe.initialDelaySeconds` | Initial delay | `5` |
| `startupProbe.enabled` | Enable startup probe | `true` |
| `startupProbe.failureThreshold` | Failure threshold | `12` |

#### Service Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `service.enabled` | Enable service (deployment only) | `true` |
| `service.type` | Service type | `ClusterIP` |
| `service.port` | Service port | `80` |
| `service.targetPort` | Target port | `http` |
| `service.nodePort` | Node port (if type is NodePort) | `""` |
| `service.loadBalancerIP` | Load balancer IP | `""` |
| `service.annotations` | Service annotations | `{}` |

#### Ingress Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ingress.enabled` | Enable ingress | `false` |
| `ingress.className` | Ingress class name | `nginx` |
| `ingress.annotations` | Ingress annotations | `{}` |
| `ingress.hosts` | Ingress hosts | See values.yaml |
| `ingress.tls` | Ingress TLS configuration | `[]` |

#### Autoscaling Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `autoscaling.enabled` | Enable HPA | `false` |
| `autoscaling.minReplicas` | Minimum replicas | `2` |
| `autoscaling.maxReplicas` | Maximum replicas | `10` |
| `autoscaling.targetCPUUtilizationPercentage` | Target CPU utilization | `70` |
| `autoscaling.targetMemoryUtilizationPercentage` | Target memory utilization | `80` |

#### Job Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `job.backoffLimit` | Backoff limit | `3` |
| `job.ttlSecondsAfterFinished` | TTL after job completion | `86400` |
| `job.parallelism` | Parallel executions | `1` |
| `job.completions` | Required completions | `1` |
| `job.restartPolicy` | Restart policy: `Never` or `OnFailure` | `Never` |

#### CronJob Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `cronjob.schedule` | Cron schedule | `"0 0 * * *"` |
| `cronjob.concurrencyPolicy` | Concurrency policy | `Forbid` |
| `cronjob.failedJobsHistoryLimit` | Failed jobs history | `1` |
| `cronjob.successfulJobsHistoryLimit` | Successful jobs history | `3` |
| `cronjob.suspend` | Suspend cron jobs | `false` |

#### Storage Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `persistence.enabled` | Enable persistent storage | `false` |
| `persistence.storageClass` | Storage class | `""` |
| `persistence.accessMode` | Access mode | `ReadWriteOnce` |
| `persistence.size` | Storage size | `10Gi` |
| `persistence.mountPath` | Mount path | `/data` |
| `persistence.existingClaim` | Existing PVC name | `""` |

#### ConfigMap/Secret Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `configMap.enabled` | Enable ConfigMap | `false` |
| `configMap.data` | ConfigMap data | `{}` |
| `secret.enabled` | Enable Secret | `false` |
| `secret.data` | Secret data (base64 encoded) | `{}` |

#### Pod Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `nodeSelector` | Node selector | `{}` |
| `tolerations` | Tolerations | `[]` |
| `affinity` | Affinity rules | `{}` |
| `podAnnotations` | Pod annotations | `{}` |
| `podSecurityContext` | Pod security context | See values.yaml |
| `securityContext` | Container security context | See values.yaml |
| `terminationGracePeriodSeconds` | Termination grace period | `30` |
| `priorityClassName` | Priority class name | `""` |

#### Service Account Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `serviceAccount.create` | Create service account | `true` |
| `serviceAccount.annotations` | Service account annotations | `{}` |
| `serviceAccount.name` | Service account name | `""` |

## Examples

### Inference API with Autoscaling

```yaml
# values-production.yaml
workloadType: deployment
replicaCount: 5

image:
  variant: cpu-full

resources:
  requests:
    cpu: 1000m
    memory: 4Gi
  limits:
    cpu: 4000m
    memory: 16Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 60

service:
  type: LoadBalancer

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: api.dimtensor.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: dimtensor-tls
      hosts:
        - api.dimtensor.com

podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8080"
```

Deploy:

```bash
helm install dimtensor-prod . -f values-production.yaml
```

### GPU Training Job

```yaml
# values-gpu-training.yaml
workloadType: job

image:
  variant: ml

resources:
  requests:
    cpu: 4000m
    memory: 16Gi
  limits:
    cpu: 8000m
    memory: 32Gi
  gpu:
    enabled: true
    count: 2

job:
  backoffLimit: 2
  ttlSecondsAfterFinished: 172800
  restartPolicy: Never

persistence:
  enabled: true
  size: 100Gi
  mountPath: /checkpoints

secret:
  enabled: true
  data:
    mlflow-uri: aHR0cDovL21sZmxvdy1zZXJ2ZXI6NTAwMA==

container:
  env:
    - name: MLFLOW_TRACKING_URI
      valueFrom:
        secretKeyRef:
          name: dimtensor
          key: mlflow-uri
    - name: CUDA_VISIBLE_DEVICES
      value: "0,1"
```

Deploy:

```bash
helm install training-job . -f values-gpu-training.yaml
```

### Batch Processing with Parallelism

```yaml
# values-batch.yaml
workloadType: job

image:
  variant: latest

resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi

job:
  parallelism: 20
  completions: 1000
  backoffLimit: 5
  ttlSecondsAfterFinished: 3600
  restartPolicy: OnFailure

configMap:
  enabled: true
  data:
    config.json: |
      {
        "parameter_range": [0, 100],
        "num_samples": 1000
      }
```

Deploy:

```bash
helm install batch-job . -f values-batch.yaml
```

### Scheduled Analysis (CronJob)

```yaml
# values-cronjob.yaml
workloadType: cronjob

image:
  variant: cpu-full

cronjob:
  schedule: "0 2 * * *"  # 2 AM daily
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 5

resources:
  requests:
    cpu: 1000m
    memory: 2Gi

persistence:
  enabled: true
  size: 50Gi
  mountPath: /data
```

Deploy:

```bash
helm install scheduled-analysis . -f values-cronjob.yaml
```

## Upgrading

### Upgrade with new values

```bash
helm upgrade dimtensor . -f values-inference.yaml
```

### Upgrade with inline values

```bash
helm upgrade dimtensor . --set replicaCount=5 --set resources.limits.memory=16Gi
```

### Rollback

```bash
# View history
helm history dimtensor

# Rollback to previous version
helm rollback dimtensor

# Rollback to specific revision
helm rollback dimtensor 2
```

## Best Practices

### Production Deployments

1. **Use specific image tags**: Avoid `latest` tag in production
2. **Set resource limits**: Prevent resource exhaustion
3. **Enable autoscaling**: Handle variable load
4. **Use persistent storage**: For checkpoints and results
5. **Enable monitoring**: Prometheus metrics and logging
6. **Secure ingress**: Use TLS with cert-manager
7. **Network policies**: Restrict pod communication
8. **Pod disruption budgets**: Ensure availability during updates

Example production values:

```yaml
image:
  tag: "4.2.0"  # Specific version
  variant: cpu-full

replicaCount: 3

resources:
  requests:
    cpu: 1000m
    memory: 4Gi
  limits:
    cpu: 4000m
    memory: 16Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20

affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchLabels:
          app.kubernetes.io/name: dimtensor
      topologyKey: kubernetes.io/hostname
```

### GPU Workloads

1. **Label GPU nodes**: `kubectl label nodes <node> accelerator=nvidia-gpu`
2. **Install device plugin**: NVIDIA device plugin for Kubernetes
3. **Set resource limits**: Request exact number of GPUs
4. **Add tolerations**: For GPU node taints
5. **Monitor GPU usage**: Use nvidia-dcgm-exporter

### Security

1. **Non-root user**: Always run as non-root (default: UID 1000)
2. **Read-only root filesystem**: Where possible
3. **Drop capabilities**: Remove unnecessary Linux capabilities
4. **Network policies**: Restrict ingress/egress
5. **Secret management**: Use external secret managers (Vault, Sealed Secrets)

## Troubleshooting

### View Chart Values

```bash
# View current values
helm get values dimtensor

# View all values (including defaults)
helm get values dimtensor --all
```

### Debug Template Rendering

```bash
# Dry run to see rendered templates
helm install dimtensor . --dry-run --debug

# Template with specific values
helm template dimtensor . -f values-inference.yaml
```

### Common Issues

**Issue: Image pull errors**

```bash
# Check image pull secrets
kubectl get pods -o jsonpath='{.items[*].spec.imagePullSecrets}'

# Add pull secret
helm upgrade dimtensor . --set image.pullSecrets[0].name=my-registry-secret
```

**Issue: GPU not available**

```bash
# Verify GPU nodes
kubectl get nodes -o json | jq '.items[].status.allocatable."nvidia.com/gpu"'

# Check device plugin
kubectl get daemonset nvidia-device-plugin-daemonset -n kube-system
```

**Issue: HPA not working**

```bash
# Check metrics-server
kubectl get deployment metrics-server -n kube-system

# View HPA status
kubectl describe hpa dimtensor
```

## Contributing

Contributions are welcome! Please submit issues and pull requests to:
https://github.com/marcoloco23/dimtensor

## License

MIT License - see LICENSE file for details

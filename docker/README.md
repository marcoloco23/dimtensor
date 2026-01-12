# Docker Images for dimtensor

Pre-built Docker images for running dimtensor in production environments with various dependency configurations.

## Available Images

| Image | Size | Description | Use Case |
|-------|------|-------------|----------|
| `base` | ~300 MB | Core dimtensor + NumPy | Lightweight production deployments |
| `ml` | ~8 GB | PyTorch + JAX + CUDA 12.1 | GPU-accelerated ML training |
| `full` | ~2.5 GB | All optional dependencies | Research & development |
| `jupyter` | ~3.5 GB | JupyterLab + all dependencies | Interactive exploration |

## Quick Start

### Pull from GitHub Container Registry

```bash
# Base image
docker pull ghcr.io/marcoloco23/dimtensor:base

# ML image with GPU support
docker pull ghcr.io/marcoloco23/dimtensor:ml

# Full image with all dependencies
docker pull ghcr.io/marcoloco23/dimtensor:full

# Jupyter image
docker pull ghcr.io/marcoloco23/dimtensor:jupyter
```

### Run Interactive Session

```bash
# Base image
docker run -it ghcr.io/marcoloco23/dimtensor:base python

# With local directory mounted
docker run -it -v $(pwd)/data:/home/dimtensor/data ghcr.io/marcoloco23/dimtensor:base python
```

### Run Jupyter Notebook

```bash
# Start JupyterLab
docker run -it -p 8888:8888 ghcr.io/marcoloco23/dimtensor:jupyter

# With local notebooks mounted
docker run -it -p 8888:8888 \
  -v $(pwd)/notebooks:/home/dimtensor/notebooks \
  ghcr.io/marcoloco23/dimtensor:jupyter

# Open browser to: http://localhost:8888
```

### Run with GPU Support

```bash
# Requires NVIDIA Docker runtime
docker run -it --gpus all ghcr.io/marcoloco23/dimtensor:ml python

# Verify GPU access
docker run --rm --gpus all ghcr.io/marcoloco23/dimtensor:ml \
  python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Building Images Locally

### Build Specific Target

```bash
# Base image
docker build --target=base -t dimtensor:base -f docker/Dockerfile .

# ML image
docker build --target=ml -t dimtensor:ml -f docker/Dockerfile .

# Full image
docker build --target=full -t dimtensor:full -f docker/Dockerfile .

# Jupyter image
docker build --target=jupyter -t dimtensor:jupyter -f docker/Dockerfile .
```

### Multi-platform Builds

```bash
# Build for multiple architectures (amd64 + arm64)
docker buildx build --platform linux/amd64,linux/arm64 \
  --target=base -t dimtensor:base -f docker/Dockerfile .

# Note: ML image is x86-only (CUDA doesn't support ARM64)
```

## Using Docker Compose

### Base Image

```bash
docker-compose -f docker/docker-compose.base.yml up
docker-compose -f docker/docker-compose.base.yml run dimtensor-base python script.py
```

### ML Image with GPU

```bash
# Prerequisites: NVIDIA Docker runtime installed
docker-compose -f docker/docker-compose.ml.yml up
docker-compose -f docker/docker-compose.ml.yml run dimtensor-ml python train.py
```

### Jupyter Image

```bash
docker-compose -f docker/docker-compose.jupyter.yml up
# Open browser to: http://localhost:8888
```

## Image Details

### Base Image

**Includes:**
- Python 3.11
- NumPy 1.x (pinned to <2.0 for compatibility)
- dimtensor core

**Size:** ~300 MB

**Best for:**
- Production deployments
- Microservices
- CI/CD pipelines
- Minimal footprint

**Example:**

```python
from dimtensor import DimArray, units

distance = DimArray([100, 200, 300], units.m)
time = DimArray([10, 20, 30], units.s)
velocity = distance / time

print(velocity)  # [10. 10. 10.] m/s
```

### ML Image

**Includes:**
- CUDA 12.1 + cuDNN 8
- PyTorch with CUDA support
- JAX with CUDA support
- dimtensor with torch/jax integrations

**Size:** ~8 GB

**Best for:**
- GPU-accelerated training
- Deep learning workflows
- Physics-informed neural networks
- Scientific ML

**Requirements:**
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime installed

**Setup NVIDIA Docker:**

```bash
# Install NVIDIA Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Verify GPU Access:**

```bash
docker run --rm --gpus all ghcr.io/marcoloco23/dimtensor:ml \
  nvidia-smi
```

**Example:**

```python
from dimtensor.torch import DimTensor
import torch

# Create tensor on GPU
x = DimTensor([1.0, 2.0, 3.0], units.m, device='cuda', requires_grad=True)
y = x ** 2
loss = y.sum()
loss.backward()

print(x.grad)  # Gradients with dimensional tracking
```

### Full Image

**Includes:**
- All base dependencies
- PyTorch (CPU-only)
- JAX (CPU-only)
- All optional integrations: pandas, xarray, h5py, scipy, scikit-learn, polars, matplotlib, plotly, mlflow, wandb, streamlit, netCDF4, pyarrow

**Size:** ~2.5 GB

**Best for:**
- Development environments
- Data analysis workflows
- Multi-framework projects
- Research

**Example:**

```python
from dimtensor import DimArray, units
import pandas as pd
import xarray as xr

# Create DimArray
data = DimArray([1, 2, 3], units.m)

# Export to pandas
df = data.to_pandas()

# Export to xarray
ds = data.to_xarray()

# Visualize
import matplotlib.pyplot as plt
plt.plot(data.data)
plt.ylabel(str(data.unit))
```

### Jupyter Image

**Includes:**
- All full image dependencies
- JupyterLab 4.x
- ipywidgets
- matplotlib, plotly, seaborn
- Example notebooks

**Size:** ~3.5 GB

**Best for:**
- Interactive exploration
- Prototyping
- Teaching
- Demos

**Ports:** 8888 (JupyterLab)

**Example:**

```bash
# Start Jupyter
docker run -p 8888:8888 -v $(pwd)/notebooks:/home/dimtensor/notebooks \
  ghcr.io/marcoloco23/dimtensor:jupyter

# Open browser to http://localhost:8888
# Default notebook: quickstart.ipynb
```

## Volume Mounts

### Common Patterns

```bash
# Mount data directory
-v $(pwd)/data:/home/dimtensor/data

# Mount notebooks
-v $(pwd)/notebooks:/home/dimtensor/notebooks

# Mount models
-v $(pwd)/models:/home/dimtensor/models

# Mount code for development
-v $(pwd)/src:/home/dimtensor/src
```

### Permissions

Images run as non-root user `dimtensor` (UID 1000) for security. Ensure mounted volumes have appropriate permissions:

```bash
# Make directory accessible
chmod 755 ./data
chown -R 1000:1000 ./data
```

## Environment Variables

### Common Variables

```bash
# Python optimization
-e PYTHONUNBUFFERED=1
-e PYTHONDONTWRITEBYTECODE=1

# NumPy settings
-e OMP_NUM_THREADS=4
-e OPENBLAS_NUM_THREADS=4

# PyTorch settings
-e TORCH_HOME=/home/dimtensor/.cache/torch

# JAX settings
-e XLA_PYTHON_CLIENT_PREALLOCATE=false
-e JAX_PLATFORM_NAME=cpu  # or 'gpu'

# CUDA settings (for ML image)
-e CUDA_VISIBLE_DEVICES=0,1
-e NVIDIA_VISIBLE_DEVICES=all
```

## Advanced Usage

### Multi-container Setup

Use Docker Compose for complex setups with multiple services:

```yaml
version: '3.8'

services:
  training:
    image: ghcr.io/marcoloco23/dimtensor:ml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./models:/home/dimtensor/models
      - ./data:/home/dimtensor/data
    command: python train.py

  jupyter:
    image: ghcr.io/marcoloco23/dimtensor:jupyter
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/dimtensor/notebooks
      - ./models:/home/dimtensor/models:ro  # Read-only access
    depends_on:
      - training
```

### Custom Dockerfile

Extend base images for custom requirements:

```dockerfile
FROM ghcr.io/marcoloco23/dimtensor:base

# Add custom dependencies
RUN pip install --no-cache-dir custom-package

# Copy custom code
COPY my_module /home/dimtensor/my_module

# Set custom entrypoint
ENTRYPOINT ["python", "-m", "my_module"]
```

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Run tests in Docker
  run: |
    docker run --rm \
      -v ${{ github.workspace }}:/workspace \
      ghcr.io/marcoloco23/dimtensor:full \
      pytest /workspace/tests
```

## Image Tags

### Versioning

```bash
# Latest stable release
ghcr.io/marcoloco23/dimtensor:latest         # Points to full:latest
ghcr.io/marcoloco23/dimtensor:base-latest
ghcr.io/marcoloco23/dimtensor:ml-latest
ghcr.io/marcoloco23/dimtensor:full-latest
ghcr.io/marcoloco23/dimtensor:jupyter-latest

# Specific version
ghcr.io/marcoloco23/dimtensor:base-4.5.0
ghcr.io/marcoloco23/dimtensor:ml-4.5.0
ghcr.io/marcoloco23/dimtensor:full-4.5.0
ghcr.io/marcoloco23/dimtensor:jupyter-4.5.0

# Development builds (from main branch)
ghcr.io/marcoloco23/dimtensor:base-dev
ghcr.io/marcoloco23/dimtensor:ml-dev
```

## Troubleshooting

### NumPy Version Issues

All images pin NumPy to <2.0 for compatibility. Verify:

```bash
docker run --rm ghcr.io/marcoloco23/dimtensor:base \
  python -c "import numpy; print(numpy.__version__)"
```

### GPU Not Detected

Check NVIDIA Docker runtime:

```bash
# Verify nvidia-docker2 installed
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check Docker daemon config
cat /etc/docker/daemon.json
# Should contain: "runtimes": { "nvidia": { ... } }
```

### Permission Denied

Images run as UID 1000. Match host permissions:

```bash
# Option 1: Change ownership
sudo chown -R 1000:1000 ./data

# Option 2: Run as root (not recommended)
docker run --user root ...
```

### Out of Memory (OOM)

Increase shared memory for PyTorch:

```bash
docker run --shm-size=8g ghcr.io/marcoloco23/dimtensor:ml python train.py
```

### Port Already in Use

Jupyter default port 8888 may conflict:

```bash
# Use different host port
docker run -p 8889:8888 ghcr.io/marcoloco23/dimtensor:jupyter
# Access at: http://localhost:8889
```

## Security

### Scanning

Images are scanned with Trivy for vulnerabilities:

```bash
# Scan an image
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image ghcr.io/marcoloco23/dimtensor:base
```

### Best Practices

- Images run as non-root user (UID 1000)
- Minimal base images (slim, runtime-only)
- Regular security updates
- No hardcoded secrets
- Health checks enabled

## Support

- GitHub Issues: https://github.com/marcoloco23/dimtensor/issues
- Documentation: https://marcoloco23.github.io/dimtensor
- Repository: https://github.com/marcoloco23/dimtensor

## License

MIT License - see LICENSE file in repository

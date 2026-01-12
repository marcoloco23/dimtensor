# Multi-stage Dockerfile for dimtensor
# Supports targets: base, ml, full, jupyter

# =============================================================================
# Base Stage: Minimal dimtensor with NumPy
# =============================================================================
FROM python:3.11-slim-bookworm AS base

LABEL maintainer="Marc Sperzel"
LABEL description="Unit-aware tensors for physics and scientific machine learning"
LABEL org.opencontainers.image.source="https://github.com/marcoloco23/dimtensor"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash dimtensor

# Set working directory
WORKDIR /home/dimtensor

# Copy project files
COPY --chown=dimtensor:dimtensor pyproject.toml README.md LICENSE ./
COPY --chown=dimtensor:dimtensor src/ ./src/

# Install dimtensor with pinned NumPy
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy>=1.24.0,<2.0.0" && \
    pip install --no-cache-dir -e . && \
    # Verify NumPy version
    python -c "import numpy; assert int(numpy.__version__.split('.')[0]) < 2, 'NumPy 2.x detected!'" && \
    # Clean up build dependencies
    apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER dimtensor

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "from dimtensor import DimArray, units; print(DimArray([1], units.m))" || exit 1

# Default command
CMD ["python"]

# =============================================================================
# ML Stage: PyTorch + JAX with CUDA support
# =============================================================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS ml

LABEL maintainer="Marc Sperzel"
LABEL description="dimtensor with PyTorch and JAX GPU support"
LABEL org.opencontainers.image.source="https://github.com/marcoloco23/dimtensor"

# Install Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash dimtensor

# Set working directory
WORKDIR /home/dimtensor

# Copy project files
COPY --chown=dimtensor:dimtensor pyproject.toml README.md LICENSE ./
COPY --chown=dimtensor:dimtensor src/ ./src/

# Install dimtensor with ML dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    # Install NumPy first (pinned to <2.0)
    pip install --no-cache-dir "numpy>=1.24.0,<2.0.0" && \
    # Install PyTorch with CUDA support
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    # Install JAX with CUDA support
    pip install --no-cache-dir "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    # Install dimtensor with torch and jax extras
    pip install --no-cache-dir -e ".[torch,jax]" && \
    # Verify installations
    python -c "import numpy; assert int(numpy.__version__.split('.')[0]) < 2, 'NumPy 2.x detected!'" && \
    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" && \
    python -c "import jax; print(f'JAX {jax.__version__}, devices: {jax.devices()}')" && \
    # Clean up
    apt-get purge -y build-essential python3.11-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    pip cache purge

# Switch to non-root user
USER dimtensor

# Health check with GPU verification
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "from dimtensor.torch import DimTensor; import torch; assert torch.cuda.is_available()" || exit 1

# Default command
CMD ["python"]

# =============================================================================
# Full Stage: All optional dependencies (no CUDA)
# =============================================================================
FROM base AS full

LABEL description="dimtensor with all optional dependencies"

# Switch back to root for installations
USER root

# Install system dependencies for various packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install all optional dependencies
RUN pip install --no-cache-dir -e ".[all]" && \
    # Verify key packages
    python -c "import numpy; assert int(numpy.__version__.split('.')[0]) < 2" && \
    python -c "import torch, jax, pandas, xarray, h5py, scipy, sklearn" && \
    python -c "from dimtensor import DimArray; from dimtensor.torch import DimTensor; from dimtensor.jax import DimArray as JAXDimArray" && \
    pip cache purge

# Switch back to non-root user
USER dimtensor

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "from dimtensor import DimArray, units; print(DimArray([1], units.m))" || exit 1

CMD ["python"]

# =============================================================================
# Jupyter Stage: Full dependencies + JupyterLab
# =============================================================================
FROM full AS jupyter

LABEL description="dimtensor with JupyterLab for interactive exploration"

# Switch back to root for installations
USER root

# Install JupyterLab and extensions
RUN pip install --no-cache-dir \
    jupyterlab>=4.0.0 \
    ipywidgets>=8.0.0 \
    jupyterlab-widgets>=3.0.0 \
    matplotlib>=3.7.0 \
    plotly>=5.15.0 \
    seaborn>=0.12.0 \
    && pip cache purge

# Create notebooks directory
RUN mkdir -p /home/dimtensor/notebooks && \
    chown -R dimtensor:dimtensor /home/dimtensor/notebooks

# Create example notebook
COPY --chown=dimtensor:dimtensor <<'EOF' /home/dimtensor/notebooks/quickstart.ipynb
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# dimtensor Quick Start\n",
    "\n",
    "Welcome to dimtensor! This notebook demonstrates basic usage."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from dimtensor import DimArray, units\n",
    "import numpy as np\n",
    "\n",
    "# Create unit-aware arrays\n",
    "distance = DimArray([1, 2, 3], units.m)\n",
    "time = DimArray([1, 2, 3], units.s)\n",
    "\n",
    "# Dimensional analysis\n",
    "velocity = distance / time\n",
    "print(f\"Velocity: {velocity}\")\n",
    "print(f\"Unit: {velocity.unit}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Physical constants\n",
    "from dimtensor.constants import c, h, k_B\n",
    "\n",
    "print(f\"Speed of light: {c}\")\n",
    "print(f\"Planck constant: {h}\")\n",
    "print(f\"Boltzmann constant: {k_B}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# PyTorch integration\n",
    "try:\n",
    "    from dimtensor.torch import DimTensor\n",
    "    import torch\n",
    "    \n",
    "    t = DimTensor([1.0, 2.0, 3.0], units.m, requires_grad=True)\n",
    "    print(f\"PyTorch DimTensor: {t}\")\n",
    "    print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
    "except ImportError:\n",
    "    print(\"PyTorch not available in this image\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Configure Jupyter
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /home/dimtensor/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /home/dimtensor/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> /home/dimtensor/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /home/dimtensor/.jupyter/jupyter_lab_config.py && \
    chown -R dimtensor:dimtensor /home/dimtensor/.jupyter

# Expose Jupyter port
EXPOSE 8888

# Switch back to non-root user
USER dimtensor

# Set working directory to notebooks
WORKDIR /home/dimtensor/notebooks

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8888/lab || exit 1

# Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]

# Plan: Docker Images for dimtensor

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner (Claude)

---

## Goal

Provide pre-built Docker images for running dimtensor in production environments with various dependency configurations (base, ML, full, Jupyter), optimized for minimal image sizes through multi-stage builds.

---

## Background

dimtensor has a complex optional dependency structure with PyTorch, JAX, CUDA, and 20+ integration packages. Users deploying dimtensor in containers need:
- Base images for lightweight production deployments
- ML images with GPU support for PyTorch/JAX workflows
- Full images with all integrations for research/development
- Jupyter images for interactive exploration
- Consistent environments across development and production

Currently, users must manually create Dockerfiles and manage dependency compatibility (especially NumPy <2.0 requirement). Pre-built images would:
- Reduce deployment friction
- Ensure tested dependency combinations
- Provide GPU-accelerated options
- Enable rapid prototyping with Jupyter

---

## Approach

### Option A: Single Dockerfile with Build Args
- One Dockerfile, use ARG to control which dependencies to install
- Pros:
  - Single file to maintain
  - Easy to see all variations
- Cons:
  - Becomes complex with many conditionals
  - Harder to optimize each variant independently
  - Build caching less effective

### Option B: Separate Dockerfiles per Variant
- Individual Dockerfiles: Dockerfile.base, Dockerfile.ml, Dockerfile.full, Dockerfile.jupyter
- Pros:
  - Clear separation of concerns
  - Easy to optimize each independently
  - Better build caching
- Cons:
  - Code duplication
  - More files to maintain

### Option C: Multi-stage Build with Shared Base
- One Dockerfile with multiple targets, shared base layer
- Different targets: base, ml, full, jupyter
- Pros:
  - Minimal duplication (shared base)
  - Good build caching
  - Single file but organized
  - Industry standard pattern
- Cons:
  - Slightly more complex than Option B
  - Requires understanding of multi-stage builds

### Decision: Option C (Multi-stage build with shared base)

Multi-stage builds are the Docker best practice for this use case. We'll create:
1. A `base` stage with Python + NumPy <2.0 + dimtensor core
2. An `ml` stage inheriting from base, adding PyTorch + JAX + CUDA
3. A `full` stage with all optional dependencies
4. A `jupyter` stage with JupyterLab + common data science tools

This approach:
- Shares base layers across all variants (efficient storage/transfer)
- Allows independent optimization of each target
- Keeps everything in one maintainable file
- Provides clear build targets: `--target=base`, `--target=ml`, etc.

---

## Implementation Steps

1. [ ] Create `docker/` directory structure
   - `docker/Dockerfile` - Main multi-stage build
   - `docker/.dockerignore` - Exclude unnecessary files
   - `docker/README.md` - Usage documentation

2. [ ] Create base stage in Dockerfile
   - Start from `python:3.11-slim-bookworm` (Debian-based, minimal)
   - Install system dependencies (build-essential for some packages)
   - Create non-root user `dimtensor` for security
   - Install NumPy 1.x (version-pinned)
   - Install dimtensor core from source or PyPI
   - Add health check
   - Set working directory and user

3. [ ] Create ml stage (GPU-enabled)
   - Start from NVIDIA CUDA base: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
   - Install Python 3.11
   - Copy dimtensor from base or reinstall
   - Install PyTorch with CUDA support
   - Install JAX with CUDA support
   - Add CUDA health check (nvidia-smi)
   - Keep image size minimal (remove build dependencies)

4. [ ] Create full stage
   - Inherit from base stage
   - Install all optional dependencies from pyproject.toml[all]
   - Clean up apt cache and pip cache
   - Document installed packages

5. [ ] Create jupyter stage
   - Inherit from full stage
   - Install JupyterLab
   - Install common extensions (ipywidgets, matplotlib, plotly)
   - Configure Jupyter (allow root, set token, expose port 8888)
   - Add example notebooks to /home/dimtensor/notebooks
   - Set CMD to start Jupyter

6. [ ] Create .dockerignore
   - Exclude: .git, .github, __pycache__, *.pyc, .pytest_cache
   - Exclude: .mypy_cache, .ruff_cache, dist, build, *.egg-info
   - Exclude: .plans, docs, examples (include separately if needed)
   - Include: src, pyproject.toml, README.md, LICENSE

7. [ ] Create docker/README.md with usage instructions
   - How to pull images from registry
   - How to build locally
   - How to run each variant
   - Volume mounting for data
   - GPU support setup
   - Example docker-compose.yml

8. [ ] Create GitHub Actions workflow: `.github/workflows/docker-build-push.yml`
   - Trigger on: push to main, version tags (v*), manual dispatch
   - Matrix build for all targets: [base, ml, full, jupyter]
   - Use docker/build-push-action@v5
   - Multi-platform builds: linux/amd64, linux/arm64
   - Tag strategy:
     - `base`: dimtensor:base, dimtensor:base-{version}, dimtensor:base-latest
     - `ml`: dimtensor:ml, dimtensor:ml-{version}, dimtensor:ml-latest
     - `full`: dimtensor:full, dimtensor:full-{version}, dimtensor:full-latest, dimtensor:latest
     - `jupyter`: dimtensor:jupyter, dimtensor:jupyter-{version}, dimtensor:jupyter-latest
   - Push to GitHub Container Registry (ghcr.io)
   - Add Docker Hub as optional secondary registry

9. [ ] Add image security scanning
   - Integrate Trivy scanner in GitHub Actions
   - Scan for vulnerabilities in dependencies
   - Fail on HIGH/CRITICAL vulnerabilities
   - Generate SBOM (Software Bill of Materials)

10. [ ] Add image size optimization
    - Use `--squash` flag to reduce layers
    - Remove unnecessary files in same RUN command
    - Use pip --no-cache-dir
    - Consider using distroless for ml/full variants
    - Document final image sizes

11. [ ] Create docker-compose examples
    - `docker-compose.base.yml` - Basic dimtensor service
    - `docker-compose.ml.yml` - ML training with GPU
    - `docker-compose.jupyter.yml` - JupyterLab with volume mounts
    - Include environment variable examples

12. [ ] Update main README.md
    - Add "Docker" section with quick start
    - Link to docker/README.md for details
    - Show example: `docker run -it ghcr.io/marcoloco23/dimtensor:jupyter`

13. [ ] Add Docker tests in CI
    - Build all image variants
    - Run smoke tests in containers:
      - `docker run dimtensor:base python -c "from dimtensor import DimArray, units; print(DimArray([1], units.m))"`
      - `docker run dimtensor:ml python -c "from dimtensor.torch import DimTensor; import torch"`
      - `docker run dimtensor:jupyter jupyter --version`
    - Verify NumPy version <2.0 in all images

---

## Files to Modify

| File | Change |
|------|--------|
| docker/Dockerfile | CREATE - Multi-stage Dockerfile with base/ml/full/jupyter targets |
| docker/.dockerignore | CREATE - Exclude unnecessary files from build context |
| docker/README.md | CREATE - Docker usage documentation |
| docker/docker-compose.base.yml | CREATE - Example compose for base image |
| docker/docker-compose.ml.yml | CREATE - Example compose for ML with GPU |
| docker/docker-compose.jupyter.yml | CREATE - Example compose for Jupyter |
| .github/workflows/docker-build-push.yml | CREATE - CI/CD for building and pushing images |
| README.md | UPDATE - Add Docker installation section |
| .gitignore | UPDATE - Add docker-related ignore patterns if needed |

---

## Testing Strategy

How will we verify this works?

- [ ] Local build tests
  - Build each target: `docker build --target=base -t dimtensor:base .`
  - Verify image sizes are reasonable (<500MB base, <5GB ml, <3GB full, <4GB jupyter)
  - Check that images run without errors

- [ ] Functional tests in CI
  - Import dimtensor in each image
  - Run basic operations (DimArray creation, arithmetic)
  - Verify PyTorch/JAX work in ml image
  - Verify Jupyter starts in jupyter image
  - Check NumPy version constraint

- [ ] Integration tests
  - Test volume mounting for data persistence
  - Test GPU access in ml image (if CUDA available)
  - Test port mapping for Jupyter
  - Verify environment variables work

- [ ] Security tests
  - Trivy scan for vulnerabilities
  - Verify running as non-root user (except Jupyter if needed)
  - Check no secrets in image layers

- [ ] Cross-platform tests
  - Build and test on linux/amd64
  - Build and test on linux/arm64 (M1/M2 Macs)

- [ ] Manual verification
  - Pull from registry: `docker pull ghcr.io/marcoloco23/dimtensor:latest`
  - Run Jupyter and open notebook
  - Run ML training example with GPU

---

## Risks / Edge Cases

- **Risk 1**: CUDA image is very large (>10GB)
  - Mitigation: Use CUDA runtime (not devel) image, clean up in same layer, consider multi-stage

- **Risk 2**: NumPy 2.0 incompatibility could break in future
  - Mitigation: Pin numpy<2.0 in Dockerfile explicitly, add CI test to verify

- **Risk 3**: PyTorch/JAX versions may conflict
  - Mitigation: Test ml image thoroughly, document known-working versions, consider separate torch/jax images if needed

- **Risk 4**: GitHub Container Registry rate limits
  - Mitigation: Cache layers aggressively, use Docker Hub as fallback, document pull limits

- **Risk 5**: Multi-platform builds are slow in CI
  - Mitigation: Use GitHub's buildx with caching, only build multi-platform on releases, use separate jobs

- **Risk 6**: Image size bloat from all dependencies in full variant
  - Mitigation: Remove build dependencies after install, use .dockerignore, document size expectations

- **Edge case**: Users want specific Python version
  - Handling: Document how to build custom images, provide Dockerfile.template in future

- **Edge case**: ARM64/M1 support for CUDA images
  - Handling: CUDA doesn't support ARM64, document that ml image is x86-only, provide CPU-only variant

- **Edge case**: Users need older dimtensor versions
  - Handling: GitHub Actions builds on version tags, images tagged with version number

- **Edge case**: Private dependencies or custom builds
  - Handling: Document how to extend base images, provide examples in docker/README.md

---

## Definition of Done

- [x] Plan reviewed and approved
- [ ] All Dockerfiles created and tested locally
- [ ] GitHub Actions workflow configured and tested
- [ ] Images successfully pushed to ghcr.io
- [ ] Docker documentation complete
- [ ] README.md updated with Docker section
- [ ] All CI tests pass (build, functional, security)
- [ ] Images verified on both amd64 and arm64
- [ ] CONTINUITY.md updated with task completion

---

## Notes / Log

**2026-01-12T15:15** - Initial plan created. Key decisions:
- Multi-stage build pattern for efficiency
- Four targets: base, ml, full, jupyter
- CUDA 12.1 for ML image
- GitHub Container Registry as primary, Docker Hub optional
- Trivy for security scanning
- Multi-platform support (amd64 + arm64) except ml

**Implementation priority**: base -> jupyter -> full -> ml (GPU last due to complexity)

**Estimated image sizes**:
- base: ~200-300 MB
- ml: ~8-10 GB (CUDA runtime + PyTorch + JAX)
- full: ~2-3 GB (all Python packages, no CUDA)
- jupyter: ~3-4 GB (full + JupyterLab)

**Alternative registries considered**: Docker Hub (rate limits), AWS ECR (requires AWS), GitLab (less common)

---

# dimtensor: Long-Term Roadmap

## Vision Statement

**dimtensor** will become the standard library for dimensionally-aware computation in scientific machine learning - the "missing link" between raw tensor operations and physically meaningful calculations.

**Ultimate Goal**: Any physicist, chemist, or engineer using Python for ML should reach for dimtensor the same way they reach for numpy.

---

## Current Status (v3.3.0) - 2026-01-09

### Completed Features

| Version | Feature | Status | Notes |
|---------|---------|--------|-------|
| v0.1-0.9 | Foundation | DONE | DimArray, units, PyTorch, JAX, serialization |
| v1.0.x | Production Release | DONE | Code review, test coverage, documentation |
| v1.1.0 | Serialization | DONE | NetCDF, Parquet, xarray |
| v1.2.0 | Domain Extensions | DONE | Astronomy, chemistry, engineering units |
| v1.3.0 | Visualization | DONE | Matplotlib, Plotly integration |
| v1.4.0 | Validation | DONE | Constraints, conservation tracking |
| v2.0.0 | Rust Backend | DONE | PyO3 bindings, ~58% overhead |
| v2.1.0 | Dimensional Inference | DONE | Variable heuristics, equation patterns, linting |
| v2.2.0 | Physics-Aware ML | DONE | DimLayers, losses, normalization, scalers |
| v3.0.0 | Physics ML Platform | DONE | Model hub, equation database, dataset registry |
| v3.1.0 | Ecosystem Integration | DONE | SciPy, scikit-learn, Polars |
| v3.2.0 | SymPy Integration | DONE | Symbolic math bridge |
| v3.3.0 | Advanced Features | DONE | Dataset loaders, unit inference, 67 equations |

### Current Metrics
- **Source Files**: 73 Python files, ~18k lines
- **Tests**: 795+ tests (with all optional deps)
- **Coverage**: ~85%
- **PyPI**: v3.3.0 deployed
- **Modules**: core, torch, jax, io, scipy, sklearn, sympy, datasets, equations, inference, visualization, validation, hub

---

## v3.4.0 - Documentation & Polish

**Theme**: Make dimtensor accessible to everyone

- [ ] Full MkDocs documentation site
- [ ] Tutorial notebooks (Jupyter)
- [ ] Migration guide from pint/astropy.units
- [ ] API reference auto-generation
- [ ] Example gallery (physics simulations)
- [ ] Video tutorials

---

## v3.5.0 - Enhanced ML Architectures

**Theme**: State-of-the-art physics ML

- [ ] Graph neural networks for physics (GNNs)
- [ ] Transformer architectures for time series
- [ ] Physics-informed attention mechanisms
- [ ] Distributed training support
- [ ] Advanced physics priors/regularizers
- [ ] Model checkpointing with units

---

## v3.6.0 - Performance & GPU

**Theme**: Production-ready speed

- [ ] CUDA kernel optimization for DimTensor
- [ ] Memory-efficient operations
- [ ] Lazy evaluation enhancements
- [ ] Profiling tools for unit overhead
- [ ] Benchmark suite against competitors

---

## v4.0.0 - Platform Maturity

**Theme**: Ecosystem and community

- [ ] VS Code extension (dimensional linting)
- [ ] Plugin system for custom units
- [ ] Community unit registry
- [ ] Web dashboard for model hub
- [ ] CI/CD templates for unit-aware projects
- [ ] Integration with MLOps tools (MLflow, W&B)

---

## v4.1.0 - More Domain Units

**Theme**: Comprehensive unit coverage

- [ ] Nuclear physics (MeV, barn, becquerel)
- [ ] Geophysics (gal, darcy)
- [ ] Biophysics (dalton, katal)
- [ ] Materials science, acoustics, photometry
- [ ] CGS, Imperial, Natural, Planck unit systems

---

## v4.2.0 - More Framework Integrations

**Theme**: Work everywhere

- [ ] TensorFlow integration
- [ ] CuPy (GPU arrays)
- [ ] Dask (distributed)
- [ ] Ray (distributed ML)
- [ ] Numba (JIT)
- [ ] Apache Arrow

---

## v4.3.0 - More Data Sources

**Theme**: Real-world physics data

- [ ] CERN Open Data, LIGO, SDSS
- [ ] Materials Project, PubChem
- [ ] NOAA, World Bank climate
- [ ] OpenFOAM, COMSOL loaders

---

## v4.4.0 - More Equations

**Theme**: Every physics equation

- [ ] QFT, General Relativity
- [ ] Statistical mechanics, Plasma physics
- [ ] Solid state, Nuclear physics
- [ ] Biophysics, Chemical kinetics
- [ ] Equation derivation trees

---

## v4.5.0 - Advanced Analysis Tools

**Theme**: Professional physics analysis

- [ ] Buckingham Pi dimensional analysis
- [ ] Scaling law finder
- [ ] Error budget calculator
- [ ] Monte Carlo uncertainty
- [ ] Interactive Jupyter widgets

---

## v5.0.0 - Research Platform

**Theme**: End-to-end physics research

- [ ] Experiment tracking with units
- [ ] Paper reproduction framework
- [ ] Model/dataset sharing protocols
- [ ] Docker, K8s, serverless deployment

---

## v5.1.0 - Education & Accessibility

**Theme**: Physics for everyone

- [ ] Interactive textbook
- [ ] Problem sets with auto-grading
- [ ] Internationalization (i18n)
- [ ] Accessibility audit
- [ ] Video course

---

## v5.2.0 - Testing & Quality

**Theme**: Production-grade reliability

- [ ] Property-based testing (Hypothesis)
- [ ] Fuzzing, mutation testing
- [ ] Chaos and load testing
- [ ] Security audit

---

## v6.0.0 - Symbolic Intelligence

**Theme**: AI-powered physics

- [ ] LLM equation generation
- [ ] Automatic unit annotation from context
- [ ] Physics copilot
- [ ] Symbolic regression with units
- [ ] Unit-aware neural ODEs
- [ ] Physics-informed transformers

---

## Success Metrics

| Milestone | Metric | Target | Current |
|-----------|--------|--------|---------|
| v3.3.0 | Test coverage | 85%+ | ~85% |
| v3.3.0 | mypy errors | 0 | 0 |
| v4.0.0 | GitHub stars | 500 | - |
| v4.0.0 | PyPI downloads/month | 50,000 | - |
| v4.0.0 | Performance overhead | <20% | ~58% |
| v4.0.0 | Contributors | 20+ | 1 |

---

## Guiding Principles

1. **Correctness over performance**: Never silently produce wrong units
2. **Zero overhead for correct code**: Optimizable when units are consistent
3. **Progressive complexity**: Simple things simple, complex things possible
4. **NumPy/PyTorch idioms**: Feel familiar to existing users
5. **Explicit over implicit**: No magical unit inference without user opt-in
6. **Interoperability**: Play nice with the ecosystem

---

## Architecture

```
src/dimtensor/
├── core/              # DimArray, units, dimensions
├── constants/         # Physical constants (CODATA 2022)
├── torch/             # PyTorch integration (DimTensor, layers, losses)
├── jax/               # JAX integration
├── io/                # Serialization (JSON, HDF5, NetCDF, Parquet, Polars)
├── scipy/             # SciPy wrappers
├── sklearn/           # Scikit-learn transformers
├── sympy/             # SymPy integration
├── datasets/          # Dataset registry and loaders
├── equations/         # Physics equation database
├── inference/         # Unit inference and heuristics
├── hub/               # Model registry
├── visualization/     # Matplotlib, Plotly
├── validation/        # Constraints, conservation
├── domains/           # Astronomy, chemistry, engineering units
├── cli/               # Linting CLI
└── benchmarks.py      # Performance measurement
```

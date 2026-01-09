# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.1.0] - 2026-01-09

### Added
- **Comprehensive Unit Coverage** - 11 new domain unit modules with 200+ units

  - **Nuclear Physics** (`from dimtensor.domains.nuclear import ...`):
    - Energy: eV, keV, MeV, GeV (electron volt family)
    - Cross-section: barn, millibarn, microbarn
    - Radioactivity: becquerel, curie
    - Absorbed dose: gray, rad
    - Dose equivalent: sievert, rem

  - **Geophysics** (`from dimtensor.domains.geophysics import ...`):
    - Acceleration: gal, milligal
    - Gravity gradient: eotvos
    - Permeability: darcy, millidarcy
    - Magnetic field: gamma, oersted

  - **Biophysics** (`from dimtensor.domains.biophysics import ...`):
    - Enzyme activity: katal, enzyme_unit
    - Cell concentrations: cells_per_mL, cells_per_uL
    - Electrophysiology: millivolt

  - **Materials Science** (`from dimtensor.domains.materials import ...`):
    - Strain: strain, microstrain, percent_strain
    - Hardness: vickers, brinell, rockwell_C, rockwell_B
    - Fracture toughness: MPa_sqrt_m, ksi_sqrt_in
    - Thermal conductivity: W_per_m_K, W_per_cm_K
    - Electrical: S_per_m, ohm_m, microohm_cm

  - **Acoustics** (`from dimtensor.domains.acoustics import ...`):
    - Sound level: decibel (dimensionless)
    - Loudness: phon, sone
    - Acoustic impedance: rayl
    - Reference pressure: micropascal

  - **Photometry** (`from dimtensor.domains.photometry import ...`):
    - Luminous flux: lumen
    - Illuminance: lux, foot_candle, phot
    - Luminance: nit, stilb, lambert
    - Luminous efficacy: lm_per_W

  - **Information Theory** (`from dimtensor.domains.information import ...`):
    - Information: bit, byte, nat, shannon
    - Binary prefixes: kilobyte, megabyte, gigabyte, terabyte (IEC 1024-based)
    - Data rates: bit_per_second, byte_per_second, baud
    - Storage density: bit_per_square_meter, gigabit_per_square_inch

  - **CGS Units** (`from dimtensor.domains.cgs import ...`):
    - Mechanical: dyne, erg, barye, poise, stokes, gal
    - Electromagnetic: gauss, maxwell, oersted, statcoulomb, statampere, statvolt
    - Variants: centipoise, centistokes, kilogauss, milligal

  - **Imperial/US Units** (`from dimtensor.domains.imperial import ...`):
    - Length: inch, foot, yard, mile, nautical_mile, furlong
    - Mass: ounce, pound, stone, ton, grain
    - Volume: fluid_ounce, cup, pint, quart, gallon, barrel
    - Temperature: rankine, fahrenheit (interval)
    - Force: pound_force, poundal
    - Energy: BTU, therm, foot_pound
    - Pressure: psi, inches_of_mercury, inches_of_water
    - Speed: mph, knot

  - **Natural Units** (`from dimtensor.domains.natural import ...`):
    - Particle physics units where c = ℏ = 1
    - Energy: GeV, MeV, eV, TeV
    - Mass: GeV_mass, MeV_mass (same as energy)
    - Momentum: GeV_momentum, MeV_momentum
    - Length/Time: GeV_inv_length, GeV_inv_time
    - Conversion functions: to_natural(), from_natural()

  - **Planck Units** (`from dimtensor.domains.planck import ...`):
    - Base: planck_length, planck_mass, planck_time, planck_charge, planck_temperature
    - Derived: planck_energy, planck_momentum, planck_force, planck_power
    - Geometric: planck_area, planck_volume
    - Dynamic: planck_density, planck_acceleration

### Changed
- Total test count: 939 → 966 passed (716 non-torch tests + 248 domain tests)
- 150 new tests for domain units
- domains/__init__.py now exports all 14 domain modules

---

## [4.0.0] - 2026-01-09

### Added
- **Platform Maturity** - Ecosystem and community features

  - **Plugin System** (`from dimtensor import plugins`):
    - `discover_plugins()` - Find all dimtensor-units-* packages via entry points
    - `load_plugin(name)` - Load and register plugin units
    - `list_plugins()` - List available plugins
    - `get_unit(plugin_name, unit_name)` - Get unit from plugin
    - `PluginRegistry` class with lazy loading and caching
    - Plugin validation and conflict resolution
    - CLI: `python -m dimtensor plugins list/info`
    - Entry point group: `dimtensor.plugins`

  - **Web Dashboard** (`pip install dimtensor[web]`):
    - Streamlit-based multi-page dashboard
    - Models browser with domain/tag filtering and search
    - Datasets browser with schema and unit display
    - Equations browser with LaTeX rendering (67+ equations)
    - Code generation snippets for each resource
    - Root entry: `streamlit run streamlit_app.py`
    - Deployment-ready for Streamlit Cloud

  - **MLflow Integration** (`pip install dimtensor[mlflow]`):
    - `log_dimarray(name, value)` - Log DimArray with unit metadata
    - `log_unit_param(name, value)` - Log parameters with units
    - `DimMLflowCallback` - Auto-log units during training
    - `compare_metrics_with_units()` - Unit-aware metric comparison
    - Units stored as MLflow tags

  - **Weights & Biases Integration** (`pip install dimtensor[wandb]`):
    - `log_dimarray(name, value)` - Log DimArray to W&B
    - `DimWandbCallback` - Training loop integration
    - `log_config_with_units(config)` - Config with unit metadata
    - `create_dimarray_table()` - W&B Tables with unit columns

  - **CI/CD Templates** (`.github/workflows/`):
    - `dimtensor-lint.yml` - Dimensional linting workflow
    - `dimtensor-test.yml` - Multi-Python, multi-OS testing
    - `dimtensor-benchmark.yml` - Performance regression detection
    - `setup-dimtensor` composite action for caching
    - Complete example in `examples/github-workflow-example.yml`
    - Documentation in `docs/guide/ci-cd.md`

### Changed
- Total test count: 903 → 939 passed (89 skipped)
- 36 new tests for plugin system
- New optional dependencies: mlflow, wandb, web (streamlit)

### Deferred
- VS Code extension (task #176-177) - Separate repository, plan exists
- CUDA kernels (task #171) - v3.7.0
- Rust optimization (task #172) - v3.7.0

---

## [3.6.0] - 2026-01-09

### Added
- **Performance Profiling Tools** - Comprehensive performance analysis for dimtensor

  - **CUDA Profiling** (`from dimtensor.torch.benchmarks import ...`):
    - `cuda_timer()` context manager for precise GPU timing with CUDA Events
    - `cuda_time_operation()` for micro-benchmarking with warmup
    - `CudaBenchmarkResult` dataclass for structured results
    - 11 benchmark functions: tensor creation, transfers, element-wise ops, matmul, autograd
    - `torch.profiler` integration via `profile_dimtensor()`
    - CPU fallback when CUDA unavailable
    - `run_all_benchmarks()` for comprehensive performance analysis

  - **Benchmark Suite** (`benchmarks/` directory):
    - ASV (Airspeed Velocity) integration for historical performance tracking
    - 97 benchmarks across NumPy, PyTorch, and competitor libraries
    - `benchmarks/suites/numpy_ops.py` - 46 NumPy DimArray benchmarks
    - `benchmarks/suites/torch_ops.py` - 22 PyTorch DimTensor benchmarks
    - `benchmarks/suites/competitors.py` - Compare vs pint, astropy.units, unyt
    - Tests array sizes: 1, 100, 10K, 1M elements
    - Statistical analysis with warmup rounds and confidence intervals
    - HTML reports via `asv publish && asv preview`

  - **Memory Profiling** (`from dimtensor import profiling`):
    - `memory_stats(obj)` - Detailed memory breakdown for DimArray/DimTensor
    - `memory_report(obj)` - Human-readable formatted report
    - `metadata_overhead(obj)` - Calculate metadata overhead in bytes
    - `get_overhead_ratio(obj)` - Get overhead as percentage
    - `compare_to_baseline(obj)` - Compare vs raw numpy/torch arrays
    - `analyze_shared_metadata(arrays)` - Analyze unit/dimension sharing
    - `MemoryProfiler` context manager for session-level profiling
    - `gpu_memory_stats(device)` - Track GPU memory for DimTensor
    - Actionable recommendations for optimization

### Changed
- Total test count: 845 → 903 passed (87 skipped)
- 58 new tests for performance profiling tools

### Deferred to v3.7.0
- CUDA kernels implementation (task #171) - Requires profiling analysis
- Rust backend optimization (task #172) - Depends on benchmark results

---

## [3.5.0] - 2026-01-09

### Added
- **Enhanced ML Architectures** (`from dimtensor.torch import ...`)
  - **Graph Neural Networks**:
    - `DimGraphConv` - Graph convolution layer with unit tracking
    - Supports message passing with dimensional validation
    - 40 new tests for GNN functionality

  - **Transformer Architectures**:
    - `DimMultiheadAttention` - Multi-head attention with unit-aware queries/keys/values
    - `DimTransformerEncoderLayer` - Transformer encoder with dimensional validation
    - `DimTransformerEncoder` - Full transformer encoder stack
    - 42 new tests for transformer functionality

  - **Physics Priors** (`from dimtensor.torch.priors import ...`):
    - `ConservationPrior` - Base class for conservation law enforcement
    - `EnergyConservationPrior` - Enforce energy conservation in ML models
    - `MomentumConservationPrior` - Enforce momentum conservation
    - `SymmetryPrior` - Enforce physical symmetries (translational, rotational)
    - `DimensionalConsistencyPrior` - Penalize dimensional inconsistencies
    - `PhysicalBoundsPrior` - Enforce physical bounds (positivity, bounded values)
    - 31 new tests for physics priors

### Changed
- Total test count: 650 → 845 passed (78 skipped)
- 93 new tests for enhanced ML architectures

---

## [3.4.0] - 2026-01-09

### Added
- **Comprehensive Documentation Overhaul**
  - **New Guide Documents** (8 new guides):
    - `docs/guide/pytorch.md` - Complete PyTorch integration guide (912 lines)
    - `docs/guide/jax.md` - JAX integration with JIT, vmap, grad (714 lines)
    - `docs/guide/physics-ml.md` - Physics-informed ML tutorial (850 lines)
    - `docs/guide/visualization.md` - Matplotlib and Plotly integration
    - `docs/guide/validation.md` - Constraints and conservation tracking (732 lines)
    - `docs/guide/inference.md` - Automatic unit inference and linting (1,121 lines)
    - `docs/guide/datasets.md` - Dataset registry and loaders (1,000 lines)
    - `docs/guide/equations.md` - Equation database usage (1,043 lines)

  - **New API Reference Documents** (6 new docs):
    - `docs/api/torch.md` - PyTorch API (DimTensor, layers, losses)
    - `docs/api/validation.md` - Validation API (constraints, conservation)
    - `docs/api/inference.md` - Inference API (heuristics, solver)
    - `docs/api/ecosystem.md` - Hub, datasets, equations API
    - `docs/api/integrations.md` - SciPy, sklearn, SymPy API
    - `docs/api/visualization.md` - Matplotlib and Plotly API

  - **Tutorial Notebooks** (5 interactive notebooks):
    - `examples/01_basics.ipynb` - DimArray fundamentals (43 cells)
    - `examples/02_physics_simulation.ipynb` - Physics simulations (36 cells)
    - `examples/03_pytorch_training.ipynb` - Train a PINN (65 cells)
    - `examples/04_data_analysis.ipynb` - Analyze exoplanet data (37 cells)
    - `examples/05_unit_inference.ipynb` - Unit inference features (45 cells)

  - **Updated Existing Documentation**:
    - `docs/index.md` - Expanded with v2.0-v3.3.0 features
    - `docs/getting-started.md` - Framework-specific installation
    - `docs/guide/examples.md` - 659 new lines of examples (204 → 863 lines)
    - `docs/troubleshooting/migration.md` - Comprehensive migration guide (360 → 1,825 lines)

  - `mkdocs.yml` - Updated navigation with all new guides and API docs
  - `examples/README.md` - Index of tutorial notebooks

### Changed
- Total documentation: ~15,000 new lines of docs and examples
- Test count: 650 passed, 80 skipped

## [3.3.0] - 2026-01-09

### Added
- **Advanced Dataset Loaders** (`from dimtensor.loaders import ...`)
  - **BaseLoader** - Abstract base class for all data loaders
  - **CSVLoader** - Generic CSV file loader with unit parsing
  - **NIST CODATA 2022 Loader** - Fundamental physical constants with uncertainties
  - **NASA Exoplanet Archive Loader** - Confirmed exoplanets with orbital parameters
  - **PRISM Climate Loader** - Historical temperature and precipitation data
  - Caching system at `~/.dimtensor/cache/` for downloaded datasets
  - 28 new tests for loader functionality

- **Expanded Equation Database** (`from dimtensor.equations import ...`)
  - 37 new equations added (67 total equations now)
  - New domains: Quantum Mechanics (Schrödinger, Heisenberg, Compton)
  - New domains: Relativity (time dilation, length contraction, Doppler)
  - New domains: Optics (lens equation, diffraction, Rayleigh criterion)
  - New domains: Acoustics (sound level, Doppler effect, resonance)
  - Expanded Fluid Dynamics with additional equations
  - Enhanced search and filtering capabilities

- **Automatic Unit Inference** (`from dimtensor.inference import infer_units`)
  - Constraint-based solver for inferring unknown units in systems
  - Expression parser for equations like "F = m * a"
  - Dimensional analysis to detect unit inconsistencies
  - `infer_units(equations, knowns, unknowns)` - solve for unknown units
  - Works with equation database for common physics relationships
  - 29 new tests for unit inference

### Fixed
- **NumPy 2.x Compatibility** - Fixed `copy=False` behavior in DimArray `__init__`

### Changed
- Total test count: 738 → 795 passed (63 skipped)
- Source files: 65 → 69 (4 new modules: loaders/)

## [3.2.0] - 2026-01-09

### Added
- **SymPy Integration** (`from dimtensor.sympy import ...`)
  - **Conversion functions**:
    - `to_sympy(arr)` - Convert DimArray to SymPy expression with units
    - `from_sympy(expr)` - Convert SymPy expression to DimArray
    - `sympy_unit_for(unit)` - Get SymPy unit for a dimtensor Unit
  - **Symbolic calculus with units**:
    - `symbolic_diff(expr, var)` - Differentiate with dimensional tracking (position -> velocity)
    - `symbolic_integrate(expr, var)` - Integrate with dimensional tracking (acceleration -> velocity)
    - `simplify_units(expr)` - Simplify unit expressions
    - `substitute(expr, values)` - Substitute values into symbolic expression

- 17 new tests for SymPy integration
- New optional dependency: `pip install dimtensor[sympy]`

### Changed
- Total test count: 721 -> 738 passed (63 skipped)
- Source files: 62 -> 65 (3 new modules)

## [3.1.0] - 2026-01-09

### Added
- **SciPy Integration** (`from dimtensor.scipy import ...`)
  - **Optimization wrappers** with unit preservation:
    - `minimize(fun, x0, ...)` - dimension-aware optimization
    - `curve_fit(f, xdata, ydata, p0)` - curve fitting with units
    - `least_squares(fun, x0, bounds)` - least squares with dimensional bounds
  - **Integration wrappers**:
    - `solve_ivp(fun, t_span, y0)` - solve ODEs preserving units
    - `quad(fun, a, b)` - numerical integration with unit-aware results
  - **Interpolation wrappers**:
    - `interp1d(x, y)` - 1D interpolation preserving units
    - `DimUnivariateSpline(x, y)` - spline with unit tracking

- **Scikit-learn Integration** (`from dimtensor.sklearn import ...`)
  - `DimStandardScaler` - StandardScaler preserving units through inverse_transform
  - `DimMinMaxScaler` - MinMaxScaler with unit-aware scaling
  - `DimTransformerMixin` - Base mixin for dimension-aware transformers

- **Polars Integration** (`from dimtensor.io.polars import ...`)
  - `to_polars(arrays)` - Convert DimArrays to Polars DataFrame
  - `from_polars(df, units_map)` - Convert Polars DataFrame to DimArrays
  - `save_polars(arrays, path)` - Save to Parquet/CSV/JSON via Polars
  - `load_polars(path, units_map)` - Load from file via Polars

- 26 new tests (17 scipy, 9 sklearn + polars tests)

### Changed
- Total test count: 695 → 721 passed (63 skipped)
- Source files: 55 → 62 (7 new modules)

## [3.0.0] - 2026-01-09

### Added
- **Model Hub** (`from dimtensor.hub import ...`)
  - `ModelInfo` - Metadata for physics models with dimensional info
  - `ModelCard` - Full model documentation with training info
  - `register_model` - Decorator/function to register models
  - `load_model`, `list_models`, `get_model_info` - Model discovery
  - `save_model_card`, `load_model_card` - Card serialization

- **Physics Equation Database** (`from dimtensor.equations import ...`)
  - `Equation` - Equation dataclass with dimensional metadata
  - 30+ built-in equations across 6 domains:
    - Mechanics: Newton's laws, kinetic/potential energy, momentum
    - Thermodynamics: Ideal gas law, heat capacity, Stefan-Boltzmann
    - Electromagnetism: Ohm's law, Coulomb's law, Lorentz force
    - Fluid dynamics: Bernoulli, Navier-Stokes, Reynolds number
    - Relativity: Mass-energy equivalence, Lorentz factor
    - Quantum: Planck-Einstein, de Broglie, Heisenberg uncertainty
  - `get_equations`, `search_equations`, `register_equation` - Database API

- **Dataset Registry** (`from dimtensor.datasets import ...`)
  - `DatasetInfo` - Metadata with feature/target dimensions
  - 10 built-in physics datasets:
    - pendulum, projectile, spring_mass (mechanics)
    - heat_diffusion, ideal_gas (thermodynamics)
    - burgers, navier_stokes_2d (fluid dynamics)
    - lorenz, three_body, wave_1d
  - `list_datasets`, `load_dataset`, `register_dataset` - Dataset API

- **Enhanced CLI** (`python -m dimtensor <command>`)
  - `convert` - Unit conversion (e.g., `dimtensor convert 1 km m`)
  - `equations` - Browse equation database with filtering
  - `datasets` - List available physics datasets
  - `constants` - Display physical constants
  - `info` - Enhanced module information

- 69 new tests (25 equations, 22 hub, 22 datasets)

### Changed
- Total test count: 626 → 695 passed (62 skipped)
- Source files: 48 → 55 (7 new modules)

## [2.2.0] - 2026-01-09

### Added
- **Physics-Aware ML Layers** (`from dimtensor.torch import DimLinear, DimConv2d, ...`)
  - `DimLayer` - Base class for dimension-aware neural network layers
  - `DimLinear` - Linear layer with dimension tracking
  - `DimConv1d`, `DimConv2d` - Convolution layers with dimension tracking
  - `DimSequential` - Container with dimension chain validation

- **Dimensional Loss Functions**
  - `DimMSELoss` - MSE loss with dimensional checking (output has dim^2)
  - `DimL1Loss` - L1 loss preserving input dimension
  - `DimHuberLoss` - Huber loss with dimensional checking
  - `PhysicsLoss` - Conservation law enforcement loss
  - `CompositeLoss` - Combine data and physics loss terms

- **Dimension-Aware Normalization**
  - `DimBatchNorm1d`, `DimBatchNorm2d` - Batch normalization preserving dimensions
  - `DimLayerNorm` - Layer normalization preserving dimensions
  - `DimInstanceNorm1d`, `DimInstanceNorm2d` - Instance normalization

- **Non-Dimensionalization Scalers**
  - `DimScaler` - Scale physical quantities to dimensionless for training
  - `MultiScaler` - Manage scaling for multiple physical quantities
  - Three scaling methods: 'characteristic', 'standard', 'minmax'
  - Automatic inverse transform back to physical units

- 52 new tests for physics ML features

### Changed
- Total test count: 574 → 626 passed (62 skipped)
- Source files: 48 (4 new: layers.py, losses.py, normalization.py, scaler.py)

## [2.1.0] - 2026-01-09

### Added
- **Dimensional linting CLI** (`python -m dimtensor lint`)
  - Static analysis of Python files for dimensional issues
  - Detects addition/subtraction of incompatible dimensions
  - Reports variable dimension inferences
  - JSON output format for IDE integration
  - Strict mode for comprehensive reporting

- **Inference configuration** (`from dimtensor import config`)
  - `config.inference.min_confidence` - minimum confidence threshold
  - `config.inference.strict_mode` - enable/disable strict checking
  - `config.inference.enabled_domains` - filter by physics domains
  - `config.inference_options()` - context manager for temporary settings
  - `config.set_inference()` - permanent configuration

- CLI entry point: `dimtensor` command with `lint` and `info` subcommands
- 26 new tests (21 lint, 5 config)

### Changed
- Total test count: 548 → 574 passed (62 skipped)
- Source files: 44 (3 new: cli/__init__.py, cli/lint.py, __main__.py)

## [2.0.0] - 2026-01-09

### Added
- **Optional Rust backend** for accelerated array operations
  - Dimension-checked array operations (add, sub, mul, div) in Rust
  - PyO3 bindings with rust-numpy for zero-copy array access
  - Automatic fallback to pure Python when Rust not available
  - Check availability: `from dimtensor._rust import HAS_RUST_BACKEND`
  - Build from source: `cd rust && maturin build --release`

- **Dimensional inference module** (`from dimtensor.inference import ...`)
  - Variable name heuristics: `infer_dimension("velocity")` returns L/T
  - 50+ physics variable patterns (velocity, force, energy, etc.)
  - Prefix handling: initial_, final_, max_, min_, delta_, etc.
  - Suffix handling: _m, _kg, _m_per_s, etc.
  - Unit suffix patterns: _meter, _newton, _joule, etc.

- **Equation pattern database** for dimensional analysis
  - 30+ physics equations (F=ma, E=mc², PV=nRT, Ohm's law, etc.)
  - 8 physics domains: mechanics, electromagnetics, thermodynamics, etc.
  - Query functions: `get_equations_by_domain()`, `find_equations_with_variable()`
  - Dimension suggestions: `suggest_dimension_from_equations("F")`

- 67 new tests (19 Rust backend, 48 inference)

### Changed
- **Major version bump** for Rust backend architecture change
- Total test count: 481 → 548 passed (62 skipped)
- Module structure: Added `_rust.py`, `inference/` package

## [1.4.0] - 2026-01-09

### Added
- **Validation & Constraints module** (`from dimtensor.validation import ...`)
  - **Value constraints** for enforcing physical validity:
    - `Positive` - values must be > 0 (mass, temperature in K, etc.)
    - `NonNegative` - values must be >= 0 (counts, magnitudes)
    - `NonZero` - values must be != 0 (divisors)
    - `Bounded(min, max)` - values must be in [min, max] (probability, efficiency)
    - `Finite` - no inf or NaN allowed
    - `NotNaN` - no NaN allowed (inf is ok)
  - `validate_all(data, constraints)` - check data against multiple constraints
  - `is_all_satisfied(data, constraints)` - boolean check
  - `DimArray.validate(constraints)` - method to validate array values

- **Conservation law tracking** for physics simulations:
  - `ConservationTracker` - track conserved quantities across computations
  - `tracker.record(value)` - record checkpoint values
  - `tracker.is_conserved(rtol, atol)` - check if quantity is conserved
  - `tracker.drift()` - calculate relative drift from initial value
  - `tracker.max_drift()` - maximum drift across all checkpoints
  - Unit consistency checking between recorded values

- `ConstraintError` exception for constraint violations
- 66 new tests for validation module

### Changed
- Total test count: 415 → 481 passed (62 skipped)
- mypy source files: 34 → 37

## [1.3.0] - 2026-01-09

### Added
- **Visualization support** via new `visualization` module
  - **Matplotlib integration** (`from dimtensor.visualization import ...`)
    - `setup_matplotlib()` - Enable automatic unit labels for DimArray plots
    - `plot()`, `scatter()`, `bar()`, `hist()` - Wrapper functions with auto-labels
    - `errorbar()` - Auto-extracts uncertainty from DimArray
    - Unit conversion via `x_unit`/`y_unit` parameters
    - Implements `matplotlib.units.ConversionInterface` for seamless integration
  - **Plotly integration** (`from dimtensor.visualization import plotly`)
    - `plotly.line()`, `plotly.scatter()`, `plotly.bar()`, `plotly.histogram()`
    - `plotly.scatter_with_errors()` - Auto-extracts uncertainty
    - Custom axis titles with automatic unit labels
    - Unit conversion support

- 21 new tests for visualization module (14 matplotlib, 7 plotly)
- Visualization examples in README

### Changed
- Total test count: 456 → 415 passed + 62 skipped
- mypy source files: 31 → 34

## [1.2.0] - 2026-01-08

### Added
- **Domain-specific units** via new `domains` module
  - **Astronomy** (`from dimtensor.domains.astronomy import ...`)
    - Distance: `parsec`, `AU`, `light_year`, `kiloparsec`, `megaparsec`
    - Mass: `solar_mass`, `earth_mass`, `jupiter_mass`
    - Radius: `solar_radius`, `earth_radius`, `jupiter_radius`
    - Luminosity: `solar_luminosity`
    - Angular: `arcsecond`, `milliarcsecond`, `microarcsecond`
    - Time: `julian_year`
  - **Chemistry** (`from dimtensor.domains.chemistry import ...`)
    - Atomic mass: `dalton`, `atomic_mass_unit`
    - Concentration: `molar`, `millimolar`, `micromolar`, `nanomolar`, `picomolar`
    - Molality: `molal`
    - Ratios: `ppm`, `ppb`, `ppt`, `percent`
    - Length: `angstrom`, `bohr_radius`
    - Energy: `hartree`, `kcal_per_mol`, `kJ_per_mol`
    - Dipole: `debye`
  - **Engineering** (`from dimtensor.domains.engineering import ...`)
    - Pressure: `MPa`, `kPa`, `GPa`, `ksi`, `millibar`
    - Energy: `BTU`, `therm`, `kWh`, `MWh`
    - Power: `horsepower`, `metric_horsepower`, `ton_refrigeration`
    - Flow: `gpm`, `cfm`, `lpm`
    - Torque: `ft_lb`, `in_lb`, `Nm`
    - Angular velocity: `rpm`, `rps`
    - Length: `mil`, `micrometer`

- 53 new tests for domain-specific units
- Domain imports available from main package: `from dimtensor import domains`

### Changed
- Total test count: 394 → 456 (excluding skipped)
- mypy source files: 27 → 31

## [1.1.0] - 2026-01-08

### Added
- **NetCDF support** (`pip install dimtensor[netcdf]`)
  - `save_netcdf()`, `load_netcdf()` - single array
  - `save_multiple_netcdf()`, `load_multiple_netcdf()` - multiple arrays
  - Compression support via zlib
  - Requires netCDF4 library

- **Parquet support** (`pip install dimtensor[parquet]`)
  - `save_parquet()`, `load_parquet()` - single array
  - `save_multiple_parquet()`, `load_multiple_parquet()` - multiple arrays
  - Compression options: snappy, gzip, lz4, zstd
  - Requires pyarrow library

- **xarray integration** (`pip install dimtensor[xarray]`)
  - `to_xarray()`, `from_xarray()` - DataArray conversion
  - `to_dataset()`, `from_dataset()` - Dataset conversion
  - Unit metadata preserved in attrs
  - Requires xarray library

- 20 new tests for serialization formats
- Optional dependency groups in pyproject.toml

### Changed
- Updated README with new I/O format examples
- `pip install dimtensor[all]` now includes all optional dependencies

## [1.0.1] - 2026-01-08

### Fixed
- **Critical**: JAX pytree registration now called automatically on import
  - Previously `register_pytree()` was defined but never executed at module level
  - JAX DimArray now works correctly with `jax.jit`, `jax.vmap`, and `jax.grad`

### Added
- 19 new tests for improved code coverage
  - Scalar arithmetic with dimensionless DimArrays
  - Right-subtract operations
  - NumPy ufuncs: `np.divide`, `np.power` edge cases
  - DimArray construction from other DimArrays with uncertainty

### Changed
- Code coverage improved: 72% → 74% overall, 85% excluding JAX module
- Core `dimarray.py` coverage: 74% → 82%
- README updated with comprehensive documentation of all features

## [1.0.0] - 2026-01-08

### Added
- **Production release**: API is now stable
- **Full type safety**: 100% mypy compliance with strict mode
- 316 tests with 72% code coverage

### Changed
- Development status upgraded to "Production/Stable"
- All public APIs are now frozen for backward compatibility

### Features Summary
- **Core**: DimArray with unit-aware arithmetic, comparison, indexing, reductions
- **Units**: Full SI system + derived units with automatic simplification
- **Constants**: CODATA 2022 physical constants with uncertainties
- **Uncertainty**: Error propagation through all operations
- **PyTorch**: DimTensor with autograd support
- **JAX**: DimArray with pytree registration for JIT/vmap/grad
- **Benchmarks**: Performance measurement utilities
- **Serialization**: JSON, Pandas, HDF5 support

## [0.9.0] - 2026-01-08

### Added
- **Serialization module**: `dimtensor.io` for saving/loading with units
  - **JSON serialization**: `to_json()`, `from_json()`, `save_json()`, `load_json()`
  - **Pandas integration**: `to_dataframe()`, `from_dataframe()`, `to_series()`, `from_series()`
  - **HDF5 support**: `save_hdf5()`, `load_hdf5()` (requires h5py)
- Unit metadata preserved through all serialization formats
- Uncertainty data preserved through serialization
- 19 new tests for serialization

### Example
```python
from dimtensor import DimArray, units
from dimtensor.io import to_json, from_json, to_dataframe, save_hdf5

# JSON serialization
arr = DimArray([1.0, 2.0, 3.0], units.m)
json_str = to_json(arr)
restored = from_json(json_str)

# Pandas DataFrame with units
distance = DimArray([1.0, 2.0], units.m)
time = DimArray([0.1, 0.2], units.s)
df = to_dataframe({"distance": distance, "time": time})

# HDF5 for large arrays
save_hdf5(arr, "data.h5", compression="gzip")
```

## [0.8.0] - 2026-01-08

### Added
- **Benchmarks module**: `dimtensor.benchmarks` for performance measurement
  - `benchmark_suite()` - run full benchmark suite
  - `quick_benchmark()` - quick overhead check
  - `print_results()` - formatted output
  - Individual benchmarks for creation, arithmetic, reductions, indexing
- **Performance metrics**:
  - Measure overhead vs raw numpy operations
  - Track performance across different array sizes
  - Document performance characteristics

### Example
```python
from dimtensor.benchmarks import quick_benchmark, benchmark_suite, print_results

# Quick overhead check
overheads = quick_benchmark()
print(f"Addition overhead: {overheads['addition']:.1f}x")

# Full benchmark suite
results = benchmark_suite(sizes=[1000, 10000])
print_results(results)
```

### Performance Notes
- Typical overhead is 2-5x for arithmetic operations
- Overhead is dominated by unit checking and object creation
- For computation-heavy workloads, overhead becomes negligible

## [0.7.0] - 2026-01-08

### Added
- **JAX integration**: `dimtensor.jax.DimArray` class
  - Wraps JAX arrays with physical unit tracking
  - Registered as JAX pytree node for JIT compatibility
  - Works with `jax.jit` - units preserved through compilation
  - Works with `jax.vmap` - vectorization with units
  - Works with `jax.grad` - differentiation with dimensional checking
  - Full arithmetic operations with dimensional checking
  - Reduction operations: sum, mean, std, var, min, max
  - Reshaping: reshape, transpose, flatten, squeeze, expand_dims
  - Linear algebra: dot, matmul, norm with dimension multiplication
- **Optional dependencies**: Install with `pip install dimtensor[jax]`
- 48 new tests for JAX integration

### Example
```python
import jax
import jax.numpy as jnp
from dimtensor.jax import DimArray
from dimtensor import units

@jax.jit
def kinetic_energy(mass, velocity):
    return 0.5 * mass * velocity**2

m = DimArray(jnp.array([1.0]), units.kg)
v = DimArray(jnp.array([10.0]), units.m / units.s)
E = kinetic_energy(m, v)  # JIT preserves units: 50.0 J
```

## [0.6.0] - 2026-01-08

### Added
- **PyTorch integration**: `dimtensor.torch.DimTensor` class
  - Wraps `torch.Tensor` with physical unit tracking
  - Full arithmetic operations with dimensional checking
  - Autograd support - gradients flow through unit-aware operations
  - Device support: CPU, CUDA, MPS via `.to()`, `.cuda()`, `.cpu()`
  - dtype support: float32, float64, float16, bfloat16
  - Reduction operations: sum, mean, std, var, min, max, norm
  - Reshaping: reshape, view, transpose, permute, flatten, squeeze, unsqueeze
  - Linear algebra: matmul, dot with dimension multiplication
  - Indexing and slicing with unit preservation
- **Optional dependencies**: Install with `pip install dimtensor[torch]`
- 60 new tests for PyTorch integration

### Example
```python
import torch
from dimtensor.torch import DimTensor
from dimtensor import units

velocity = DimTensor(torch.randn(32, 3), units.m / units.s)
velocity.requires_grad_(True)
energy = 0.5 * mass * velocity**2
energy.sum().backward()  # Gradients flow correctly
```

## [0.5.0] - 2026-01-08

### Added
- **Uncertainty propagation**: Track measurement errors through calculations
  - `DimArray([10.0], units.m, uncertainty=[0.1])` - specify uncertainty
  - `.uncertainty` property - access absolute uncertainty
  - `.relative_uncertainty` property - relative uncertainty (σ/|value|)
  - `.has_uncertainty` property - check if uncertainty is tracked
- **Propagation through operations**:
  - Addition/subtraction: σ_z = √(σ_x² + σ_y²)
  - Multiplication/division: σ_z/|z| = √((σ_x/x)² + (σ_y/y)²)
  - Power: σ_z/|z| = |n| × σ_x/|x|
  - Scalar operations scale uncertainty appropriately
- **Reduction operations** preserve uncertainty:
  - `sum()`: σ = √(Σσᵢ²)
  - `mean()`: σ = √(Σσᵢ²) / N
  - `min()`/`max()`: uncertainty of selected element
- **Unit conversion** scales uncertainty by conversion factor
- **String display**: Shows "value ± uncertainty unit" format
- **Format strings**: `f"{x:.2f}"` includes uncertainty

### Changed
- Physical constants now transfer uncertainty via `to_dimarray()`

## [0.4.0] - 2026-01-08

### Added
- **Physical constants module**: CODATA 2022 constants with proper units and uncertainties
  - `from dimtensor.constants import c, G, h, k_B, N_A, e, m_e`
  - Universal constants: `c` (speed of light), `G` (gravitational), `h`/`hbar` (Planck)
  - Electromagnetic: `e` (elementary charge), `mu_0`, `epsilon_0`, `alpha` (fine-structure)
  - Atomic: `m_e`, `m_p`, `m_n` (particle masses), `a_0` (Bohr radius), `R_inf` (Rydberg)
  - Physico-chemical: `N_A` (Avogadro), `k_B` (Boltzmann), `R` (gas), `F` (Faraday), `sigma` (Stefan-Boltzmann)
  - Derived Planck units: `l_P`, `m_P`, `t_P`, `T_P`, `E_h` (Hartree)
- **Constant class**: New `Constant` type with value, unit, and uncertainty
  - `c.uncertainty` - absolute standard uncertainty
  - `c.relative_uncertainty` - relative uncertainty
  - `c.is_exact` - True for defining constants (c, h, e, k_B, N_A)
- **Constants arithmetic**: Full interoperability with DimArray
  - `m * c**2` works seamlessly for E=mc² calculations
  - `G * m1 * m2 / r**2` for gravitational force
- **Domain submodules** for organized access:
  - `dimtensor.constants.universal`
  - `dimtensor.constants.electromagnetic`
  - `dimtensor.constants.atomic`
  - `dimtensor.constants.physico_chemical`
  - `dimtensor.constants.derived`

## [0.3.1] - 2026-01-08

### Added
- **Documentation site**: Full mkdocs documentation with Material theme
  - Getting started guide
  - API reference with mkdocstrings
  - Examples for physics calculations
  - Unit reference tables
- **Display configuration**: New `config` module for controlling output
  - `config.set_display(precision=N)` - set display precision
  - `config.precision(N)` - context manager for temporary precision
  - `config.options(...)` - context manager for multiple options
  - `config.reset_display()` - reset to defaults

## [0.3.0] - 2026-01-08

### Added
- **Module-level array functions** (NumPy-style API):
  - `concatenate(arrays, axis)` - join arrays along existing axis
  - `stack(arrays, axis)` - join arrays along new axis
  - `split(array, indices_or_sections, axis)` - split array into sub-arrays
- **Linear algebra functions**:
  - `dot(a, b)` - dot product with dimension multiplication
  - `matmul(a, b)` - matrix multiplication with dimension multiplication
  - `norm(array, ord, axis, keepdims)` - vector/matrix norm, preserves units
- **Reshaping methods** on DimArray:
  - `reshape(shape)` - reshape preserving units
  - `transpose(axes)` - permute dimensions preserving units
  - `flatten()` - flatten to 1D preserving units
- **Statistics method**:
  - `var(axis, keepdims)` - variance with squared units (m -> m^2)
- **Searching methods**:
  - `argmin(axis)` - return indices of minimum values
  - `argmax(axis)` - return indices of maximum values

### Changed
- Array functions enforce same dimension for all input arrays
- Linear algebra functions properly multiply dimensions (L * L = L^2)

## [0.2.0] - 2025-01-08

### Added
- **Unit simplification**: Compound units now display as their SI derived equivalents
  - `kg·m/s²` → `N` (newton)
  - `kg·m²/s²` → `J` (joule)
  - `kg·m²/s³` → `W` (watt)
  - `m/s·s` → `m` (cancellation)
- **Format string support**: Use f-strings with DimArray
  - `f"{distance:.2f}"` → `"1234.57 m"`
  - `f"{energy:.2e}"` → `"1.23e+03 J"`
- **NumPy ufunc integration**: Use numpy functions directly
  - `np.sin(angle)`, `np.cos(angle)` - require dimensionless input
  - `np.exp(x)`, `np.log(x)` - require dimensionless input
  - `np.sqrt(area)` - halves dimension exponents
  - `np.abs(velocity)` - preserves units
  - `np.add(a, b)`, `np.multiply(a, b)` - dimension-aware arithmetic

### Changed
- Unit display now uses simplified symbols by default

## [0.1.2] - 2025-01-08

### Fixed
- Corrected GitHub repository URLs in PyPI metadata

## [0.1.1] - 2025-01-08

### Added
- `sqrt()` method on DimArray for square root with proper dimension handling

## [0.1.0] - 2025-01-08

### Added
- Initial release
- `DimArray` class wrapping numpy arrays with unit metadata
- `Dimension` class with full algebra (multiply, divide, power)
- SI base units: meter, kilogram, second, ampere, kelvin, mole, candela
- SI derived units: newton, joule, watt, pascal, volt, hertz, etc.
- Common non-SI units: km, mile, hour, eV, atm, etc.
- Unit conversion with `.to()` method
- Dimensional error catching at operation time
- Arithmetic operations with automatic dimension checking
- Comparison operations between compatible dimensions
- Array indexing, slicing, and iteration
- Reduction operations (sum, mean, std, min, max)

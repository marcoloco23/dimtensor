# dimtensor

**Unit-aware tensors for physics and scientific machine learning.**

dimtensor wraps NumPy arrays with physical unit tracking, catching dimensional errors at operation time rather than after hours of computation.

## Why dimtensor?

- **Catch errors early**: Don't waste compute on dimensionally invalid calculations
- **Self-documenting code**: Units make physics code clearer
- **NumPy compatible**: Works seamlessly with existing NumPy code
- **Lightweight**: Just metadata tracking, minimal overhead

## Quick Example

```python
from dimtensor import DimArray, units

# Create dimension-aware arrays
velocity = DimArray([10, 20, 30], units.m / units.s)
time = DimArray([1, 2, 3], units.s)

# Operations preserve/check dimensions
distance = velocity * time
print(distance)  # [10 40 90] m

# Catch errors early
acceleration = DimArray([9.8], units.m / units.s**2)
velocity + acceleration  # DimensionError: cannot add m/s to m/s^2
```

## Installation

```bash
pip install dimtensor
```

## Core Features

| Feature | Description |
|---------|-------------|
| **Dimensional Safety** | Operations between incompatible dimensions raise `DimensionError` |
| **Unit Conversion** | Convert between compatible units with `.to()` |
| **SI Units** | Full support for SI base and derived units |
| **Unit Simplification** | `kg*m/s^2` automatically displays as `N` |
| **NumPy Integration** | Works with ufuncs like `np.sin`, `np.sqrt` |
| **Array Functions** | `concatenate`, `stack`, `split`, `dot`, `matmul`, `norm` |

## Framework Integration

dimtensor integrates seamlessly with popular scientific computing frameworks:

- **PyTorch** (`DimTensor`): Unit-aware tensors with autograd support for physics-informed neural networks
- **JAX** (`DimArray`): JAX integration with JIT, vmap, and grad compatibility
- **SciPy**: Wrappers for optimization, integration, and interpolation
- **Scikit-learn**: Unit-aware transformers (StandardScaler, MinMaxScaler)
- **SymPy**: Bridge between numerical and symbolic computation

## Serialization & I/O

Save and load your unit-aware arrays in multiple formats:

- JSON, HDF5, NetCDF, Parquet
- Pandas DataFrame integration
- Polars DataFrame support
- xarray DataArray conversion

## Advanced Features

### Physics-Aware Machine Learning (v2.2.0+)

Build neural networks that respect physical units:

- **DimLayer**: Unit-aware neural network layers (Linear, Conv1d, Conv2d)
- **Physics Losses**: Conservation-law enforcing loss functions
- **DimBatchNorm**: Normalization layers that preserve units
- **DimScaler**: Automatic non-dimensionalization for training

### Dimensional Inference & Linting (v2.1.0+)

Catch dimensional errors before runtime:

- **Automatic Unit Inference**: Infer units from variable names and equations
- **Linting CLI**: `dimtensor lint file.py` for static analysis
- **Equation Pattern Database**: 67+ physics equations across 10+ domains

### Physics ML Platform (v3.0.0+)

Complete ecosystem for physics-informed machine learning:

- **Model Hub**: Registry system for physics ML models
- **Equation Database**: Browse and search physics equations
- **Dataset Registry**: Built-in physics datasets (pendulum, Burgers, Lorenz)
- **CLI Tools**: `dimtensor equations`, `dimtensor convert`, `dimtensor datasets`

### Dataset Loaders (v3.3.0+)

Download real physics data with automatic unit tracking:

```python
from dimtensor.loaders import NISTFundamentalConstants, NASAExoplanets

# Load NIST fundamental constants
constants = NISTFundamentalConstants().load()

# Download NASA exoplanet data
exoplanets = NASAExoplanets().load()
```

Supported data sources:
- NIST CODATA fundamental constants
- NASA Exoplanet Archive
- PRISM Climate Dataset

### Validation & Constraints (v1.4.0+)

Enforce physical constraints on your arrays:

```python
from dimtensor.validation import Positive, Bounded, ConservationTracker

temperature = DimArray([300, 310, 295], units.K, constraints=[Positive()])
temperature.validate()  # Ensures all values are positive

# Track conservation laws
tracker = ConservationTracker()
tracker.record("energy", total_energy)
assert tracker.is_conserved("energy")  # Check energy conservation
```

### Visualization (v1.3.0+)

Create publication-ready plots with automatic unit labels:

```python
from dimtensor.visualization import plot, scatter

# Matplotlib integration
plot(time, position, ylabel="Position")  # Auto-labels with units

# Plotly support
from dimtensor.visualization import plotly
plotly.line(time, velocity, title="Velocity vs Time")
```

### Domain-Specific Units (v1.2.0+)

Specialized units for astronomy, chemistry, and engineering:

```python
from dimtensor.domains import astronomy, chemistry, engineering

distance = DimArray([1.0], astronomy.parsec)
concentration = DimArray([0.5], chemistry.molar)
pressure = DimArray([100], engineering.psi)
```

### Rust Backend (v2.0.0+)

Optional high-performance backend with minimal overhead:

```bash
pip install dimtensor-core  # Optional Rust extension
```

Performance optimizations for large-scale computations.

## Next Steps

- [Getting Started](getting-started.md) - Installation and first steps
- [Working with Units](guide/units.md) - Learn about the unit system
- [Working with Datasets](guide/datasets.md) - Physics datasets and data loaders
- [Examples](guide/examples.md) - Real-world physics calculations
- [API Reference](api/dimarray.md) - Complete API documentation

# dimtensor: Long-Term Roadmap

## Vision Statement

**dimtensor** will become the standard library for dimensionally-aware computation in scientific machine learning - the "missing link" between raw tensor operations and physically meaningful calculations.

**Ultimate Goal**: Any physicist, chemist, or engineer using Python for ML should reach for dimtensor the same way they reach for numpy.

---

## Current Status (v0.5.x) ✅

What we have shipped:
- `DimArray` wrapping numpy arrays with unit metadata
- `Dimension` class with full algebra (multiply, divide, power)
- SI base units (m, kg, s, A, K, mol, cd)
- Derived units (N, J, W, Pa, V, Hz, etc.)
- Common non-SI units (km, mile, hour, eV, atm, etc.)
- Unit conversion between compatible units
- Dimensional error catching at operation time
- `sqrt()` method
- **Unit simplification** (`kg·m/s²` → `N`, `m/s·s` → `m`)
- **Format string support** (`f"{distance:.2f}"`)
- **NumPy ufunc integration** (`np.sin`, `np.sqrt`, `np.exp`, etc.)
- **Array functions**: `concatenate`, `stack`, `split`
- **Linear algebra**: `dot`, `matmul`, `norm`
- **Reshaping**: `reshape`, `transpose`, `flatten`
- **Statistics**: `var` (with squared units)
- **Searching**: `argmin`, `argmax`
- **Physical constants**: CODATA 2022 values with uncertainties
- **Uncertainty propagation**: Track errors through calculations
- Published on PyPI, GitHub repo live
- 225 tests, 84% coverage

---

## Roadmap Overview

```
v0.1.x  ✅ Foundation      - Basic DimArray, units, PyPI
v0.2.x  ✅ Usability       - Unit simplification, format strings, numpy ufuncs
v0.3.x  ✅ NumPy Parity    - Array functions, linear algebra, reshaping
v0.4.x  ✅ Constants       - Physical constants library
v0.5.x  ✅ Uncertainty     - Error propagation
v0.6.x     PyTorch         - torch.Tensor integration
v0.7.x     JAX             - JAX array integration
v0.8.x     Performance     - Rust backend, lazy evaluation
v0.9.x     Serialization   - Save/load with units preserved
v1.0.0     Production      - Stable API, full docs, battle-tested

v1.x       Ecosystem       - Integrations, domain extensions
v2.x       Intelligence    - Dimensional inference, suggestions
v3.x       Platform        - Full physics ML toolkit
```

---

## Phase 1: Foundation & Usability (v0.2 - v0.5)

### v0.2.0 - Better Usability ✅
**Theme**: Make it pleasant to use daily

- [x] **Unit simplification**: `m/s·s` → `m`, `kg·m/s²` → `N`
- [x] **NumPy ufunc integration**: `np.sin(angle)`, `np.exp(dimensionless)`
- [x] **Format strings**: `f"{distance:.2f}"` works naturally
- [x] **Changelog**: Maintain CHANGELOG.md
- [x] **Repr improvements**: Configurable display precision via `config` module
- [x] **Documentation site**: mkdocs with examples and API reference

### v0.3.0 - NumPy Function Parity ✅
**Theme**: Drop-in replacement for unit-aware calculations

- [x] **Math functions**: `sin`, `cos`, `tan`, `exp`, `log` (via ufuncs in v0.2)
- [x] **Array functions**: `concatenate`, `stack`, `split` (enforce same units)
- [x] **Linear algebra**: `dot`, `matmul`, `norm` with correct unit handling
- [x] **Statistics**: `var` (squared units), `std` (preserve units)
- [x] **Searching**: `argmin`, `argmax` (return indices, not DimArrays)
- [x] **Reshaping**: `reshape`, `transpose`, `flatten` (preserve units)

### v0.4.0 - Physical Constants ✅
**Theme**: CODATA constants with proper units

```python
from dimtensor.constants import c, G, h, k_B, N_A, e, m_e

# Speed of light
print(c)  # 299792458 m/s

# Gravitational force
F = G * m1 * m2 / r**2  # Automatically in Newtons
```

- [x] **CODATA 2022 constants**: All fundamental constants
- [x] **Derived constants**: Planck length, Bohr radius, etc.
- [x] **Domain packs**: `constants.electromagnetic`, `constants.atomic`, etc.
- [x] **Uncertainty values**: `c.uncertainty`, `G.uncertainty`

### v0.5.0 - Uncertainty Propagation ✅
**Theme**: Track measurement uncertainty through calculations

```python
from dimtensor import DimArray, units

# Value ± uncertainty
length = DimArray([10.0], units.m, uncertainty=[0.1])
time = DimArray([2.0], units.s, uncertainty=[0.05])

velocity = length / time
print(velocity)  # 5.0 ± 0.13 m/s (propagated)
```

- [x] **Uncertainty storage**: Optional uncertainty array in DimArray
- [x] **Propagation rules**: Standard error propagation formulas
- [x] **Reporting**: `.uncertainty`, `.has_uncertainty`, `.relative_uncertainty`
- [x] **Constant integration**: Constants transfer uncertainty to DimArray
- [x] **String display**: Shows "value ± uncertainty unit"

---

## Phase 2: ML Framework Integration (v0.6 - v0.9)

### v0.6.0 - PyTorch Integration
**Theme**: Unit-aware deep learning

```python
import torch
from dimtensor.torch import DimTensor

# Create unit-aware tensor
velocity = DimTensor(torch.randn(32, 3), units.m/units.s)

# Works with autograd
velocity.requires_grad_(True)
energy = 0.5 * mass * velocity**2
energy.sum().backward()  # Gradients preserve dimensional correctness
```

- [ ] **DimTensor class**: Wraps `torch.Tensor` with units
- [ ] **Autograd support**: Gradients flow through unit-aware operations
- [ ] **Device support**: CPU, CUDA, MPS
- [ ] **dtype support**: float32, float64, bfloat16
- [ ] **nn.Module compatibility**: Use in neural networks

### v0.7.0 - JAX Integration
**Theme**: Unit-aware differentiable programming

```python
import jax.numpy as jnp
from dimtensor.jax import DimArray

@jax.jit
def compute_energy(mass, velocity):
    return 0.5 * mass * velocity**2

# JIT compilation preserves units
energy = compute_energy(mass, velocity)
```

- [ ] **JAX pytree registration**: DimArray as valid pytree node
- [ ] **JIT compatibility**: Units preserved through compilation
- [ ] **vmap support**: Vectorization with units
- [ ] **grad support**: Differentiation with dimensional checking

### v0.8.0 - Performance
**Theme**: Production-ready speed

- [ ] **Rust backend** (via PyO3): Core operations in Rust
- [ ] **Lazy evaluation**: Defer computation until needed
- [ ] **Operator fusion**: Combine multiple operations
- [ ] **Memory optimization**: Minimize allocations
- [ ] **Benchmarks**: Comprehensive performance test suite
- [ ] **Profiling tools**: Identify bottlenecks in user code

**Target**: <10% overhead vs raw numpy for typical operations

### v0.9.0 - Serialization & Interop
**Theme**: Save, load, and exchange data with units

```python
# Save with units preserved
data.save("simulation_results.h5")  # HDF5 with unit metadata

# Load back
loaded = DimArray.load("simulation_results.h5")
assert loaded.unit == units.m/units.s

# Pandas integration
df = data.to_dataframe()  # Column units in metadata
```

- [ ] **HDF5 support**: Save/load with unit metadata
- [ ] **NetCDF support**: Climate/geo science standard
- [ ] **Parquet support**: For large datasets
- [ ] **Pandas integration**: DataFrame with unit-aware columns
- [ ] **xarray integration**: Labeled arrays with units
- [ ] **JSON serialization**: For APIs and configs

---

## Phase 3: Production Release (v1.0)

### v1.0.0 - Stable Release
**Theme**: Ready for production use

- [ ] **API freeze**: Stable public API with deprecation policy
- [ ] **Comprehensive docs**: Tutorials, API reference, cookbook
- [ ] **100% test coverage**: All public APIs tested
- [ ] **Type stubs**: Full mypy/pyright support
- [ ] **Performance guarantees**: Documented overhead bounds
- [ ] **Migration guide**: From pint, astropy.units, unyt
- [ ] **Security audit**: No vulnerabilities
- [ ] **LTS commitment**: 2-year support for v1.x

---

## Phase 4: Ecosystem & Extensions (v1.x)

### v1.1 - Domain Extensions
**Theme**: Specialized unit systems

```python
from dimtensor.domains import astronomy, chemistry, engineering

# Astronomy
distance = DimArray([4.2], astronomy.light_year)
magnitude = DimArray([-26.74], astronomy.apparent_magnitude)

# Chemistry
concentration = DimArray([0.1], chemistry.molar)
pH = DimArray([7.0], chemistry.pH)

# Engineering
stress = DimArray([250], engineering.MPa)
```

- [ ] **Astronomy**: parsec, AU, solar mass, magnitude systems
- [ ] **Chemistry**: molar, molal, ppm, pH
- [ ] **Engineering**: MPa, ksi, BTU, horsepower
- [ ] **Particle physics**: natural units (c=ℏ=1)
- [ ] **CGS system**: Gaussian units for E&M

### v1.2 - Equation Systems
**Theme**: Solve equations with units

```python
from dimtensor.equations import solve

# Define equation: F = m * a
# Given F and m, solve for a
result = solve(
    equation="F = m * a",
    known={"F": force, "m": mass},
    solve_for="a"
)
print(result)  # Acceleration with correct units
```

- [ ] **Symbolic equation parsing**
- [ ] **Dimensional consistency checking**
- [ ] **Unit inference for unknowns**
- [ ] **System of equations support**

### v1.3 - Visualization
**Theme**: Plots that understand units

```python
import matplotlib.pyplot as plt
from dimtensor.plotting import unit_aware

with unit_aware():
    plt.plot(time, position)
    # Axes automatically labeled: "time (s)", "position (m)"
    plt.show()
```

- [ ] **Matplotlib integration**: Auto-labeled axes
- [ ] **Plotly integration**: Interactive plots with units
- [ ] **Unit conversion in plots**: Display in preferred units
- [ ] **Automatic scaling**: km instead of 1000 m

### v1.4 - Validation & Constraints
**Theme**: Catch physics errors beyond dimension mismatches

```python
from dimtensor.constraints import positive, bounded, conserved

# Ensure mass is always positive
mass = DimArray([1.0], units.kg, constraints=[positive()])

# Ensure temperature is physical
temp = DimArray([300], units.K, constraints=[bounded(min=0)])

# Track energy conservation
@conserved(energy)
def simulate(state):
    ...
```

- [ ] **Value constraints**: positive, bounded, integer
- [ ] **Conservation laws**: Energy, momentum, charge
- [ ] **Custom constraints**: User-defined validation
- [ ] **Runtime checking**: Optional strict mode

---

## Phase 5: Intelligence (v2.x)

### v2.0 - Dimensional Inference
**Theme**: AI-assisted dimensional analysis

```python
from dimtensor.inference import infer_units

# Infer units from variable names and operations
result = infer_units("""
    force = mass * acceleration
    energy = force * distance
    power = energy / time
""")
# Suggests: force→N, energy→J, power→W
```

- [ ] **Variable name heuristics**: `velocity` → m/s
- [ ] **Equation pattern matching**: Recognize physics formulas
- [ ] **Suggestions API**: IDE integration for unit hints
- [ ] **Linting**: Warn about likely dimensional errors

### v2.1 - Unit-Aware Machine Learning
**Theme**: Physics-informed ML primitives

```python
from dimtensor.ml import PhysicsLinear, DimensionalLoss

# Layer that respects unit transformations
layer = PhysicsLinear(
    in_features=3,
    out_features=1,
    in_units=units.m/units.s,
    out_units=units.J
)

# Loss function with dimensional weighting
loss = DimensionalLoss(
    position_weight=1.0,  # Automatically scaled by unit
    velocity_weight=1.0,
    energy_weight=1.0
)
```

- [ ] **Physics-aware layers**: Linear, Conv with unit semantics
- [ ] **Dimensional loss functions**: Properly weighted multi-objective
- [ ] **Unit-aware normalization**: BatchNorm that respects dimensions
- [ ] **Automatic non-dimensionalization**: Scale inputs appropriately

### v2.2 - Symbolic Integration
**Theme**: Bridge numeric and symbolic computing

```python
from dimtensor.symbolic import symbolic

@symbolic
def kinetic_energy(m, v):
    return 0.5 * m * v**2

# Get symbolic expression
expr = kinetic_energy.expression
print(expr)  # 0.5 * m * v²

# Derive new quantities
momentum = kinetic_energy.differentiate(v) * 2 / v
print(momentum.units)  # kg·m/s
```

- [ ] **SymPy integration**: Symbolic manipulation with units
- [ ] **Automatic differentiation**: Symbolic derivatives
- [ ] **Simplification**: Algebraic simplification preserving units
- [ ] **Code generation**: Generate optimized numerical code

---

## Phase 6: Platform (v3.x)

### v3.0 - Physics ML Platform
**Theme**: Complete toolkit for AI-driven science

```
dimtensor/
├── core/           # DimArray, units, dimensions (current)
├── constants/      # Physical constants
├── uncertainty/    # Error propagation
├── torch/          # PyTorch integration
├── jax/            # JAX integration
├── constraints/    # Conservation laws, validation
├── inference/      # Dimensional inference
├── ml/             # Physics-aware ML primitives
├── symbolic/       # Symbolic computing bridge
├── io/             # Serialization formats
├── viz/            # Visualization
├── domains/        # Domain-specific extensions
└── hub/            # Pre-built physics models
```

- [ ] **Model hub**: Pre-trained physics-informed models
- [ ] **Equation database**: Common physics equations
- [ ] **Dataset registry**: Curated datasets with units
- [ ] **Benchmarks**: Standard physics ML benchmarks
- [ ] **CLI tools**: `dimtensor check`, `dimtensor convert`

---

## Success Metrics

| Milestone | Metric | Target |
|-----------|--------|--------|
| v0.5 | PyPI downloads/month | 1,000 |
| v1.0 | GitHub stars | 500 |
| v1.0 | Test coverage | 95%+ |
| v1.0 | Documentation pages | 50+ |
| v1.5 | PyPI downloads/month | 10,000 |
| v2.0 | Academic citations | 10+ |
| v2.0 | Contributors | 20+ |
| v3.0 | PyPI downloads/month | 100,000 |
| v3.0 | Major framework adoption | 1+ (PyTorch/JAX) |

---

## Guiding Principles

1. **Correctness over performance**: Never silently produce wrong units
2. **Zero overhead for correct code**: Optimizable when units are consistent
3. **Progressive complexity**: Simple things simple, complex things possible
4. **NumPy/PyTorch idioms**: Feel familiar to existing users
5. **Explicit over implicit**: No magical unit inference without user opt-in
6. **Interoperability**: Play nice with the ecosystem

---

## Next Immediate Steps

**For v0.6.0 (PyTorch Integration):**
1. DimTensor class wrapping `torch.Tensor` with units
2. Autograd support for gradient flow through unit-aware operations
3. Device support (CPU, CUDA, MPS)
4. dtype support (float32, float64, bfloat16)
5. nn.Module compatibility

**Recently completed (v0.5.0):**
- [x] Uncertainty storage: Optional uncertainty array in DimArray
- [x] Standard error propagation through +, -, *, /, **
- [x] Unit conversion scales uncertainty appropriately
- [x] Reduction operations propagate uncertainty (sum, mean, min, max)
- [x] Constants transfer uncertainty via `to_dimarray()`
- [x] Properties: `.uncertainty`, `.has_uncertainty`, `.relative_uncertainty`
- [x] String representations show "value ± uncertainty unit"
- [x] 39 new tests (225 total, 84% coverage)

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**dimtensor** is a Python library providing unit-aware tensors for physics and scientific machine learning. It wraps NumPy arrays with physical unit tracking, catching dimensional errors at operation time.

## Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all tests with coverage
pytest

# Run a single test file
pytest tests/test_dimarray.py

# Run a specific test
pytest tests/test_dimarray.py::test_add_same_dimension -v

# Type checking
mypy src/dimtensor

# Linting
ruff check src/dimtensor

# Build documentation (requires docs extras)
pip install -e ".[docs]"
mkdocs serve
```

## Architecture

```
src/dimtensor/
├── core/
│   ├── dimensions.py   # Dimension class: 7-tuple of SI base dimension exponents (L,M,T,I,Θ,N,J)
│   ├── units.py        # Unit class: dimension + scale factor; all SI and common units defined here
│   └── dimarray.py     # DimArray: numpy wrapper that enforces dimensional correctness
├── functions.py        # Module-level array functions (concatenate, stack, dot, matmul, norm)
├── errors.py           # DimensionError, UnitConversionError
└── config.py           # Display options (precision, threshold)
```

**Core Design:**
- `Dimension` is a frozen dataclass storing `Fraction` exponents for 7 SI base dimensions. Algebra: multiply adds exponents, divide subtracts, power multiplies.
- `Unit` combines a `Dimension` with a scale factor (relative to SI base units). Example: kilometer has dimension L and scale 1000.0.
- `DimArray` wraps `numpy.ndarray` + `Unit`. Operations check dimensional compatibility and propagate units correctly:
  - Addition/subtraction require same dimension (auto-converts compatible units)
  - Multiplication/division multiply dimensions
  - Power operations require dimensionless exponents
  - NumPy ufuncs like `np.sin`, `np.exp` require dimensionless input; `np.sqrt` halves dimension exponents

**Unit Simplification:** The `_DIMENSION_TO_SYMBOL` dict in `units.py` maps dimensions to canonical symbols (e.g., `kg·m/s²` → `N`).

## Key Patterns

- Internal constructor `DimArray._from_data_and_unit(data, unit)` avoids copying for performance
- All operations return new DimArray instances (immutable-style)
- `__array_ufunc__` handles NumPy function calls on DimArrays
- Scalars are always wrapped in 1D arrays for API consistency

## Testing

Tests are in `tests/` using pytest. Each core module has a corresponding test file. Run `pytest -v --cov=dimtensor` for coverage report.

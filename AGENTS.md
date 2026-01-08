# AGENTS.md

Instructions for AI coding agents working on this repository.

## Build & Test

```bash
# Setup
pip install -e ".[dev]"

# Test (always run after changes)
pytest

# Single test
pytest tests/test_dimarray.py::test_name -v

# Type check
mypy src/dimtensor

# Lint
ruff check src/dimtensor
```

## Project Structure

```
src/dimtensor/
├── core/
│   ├── dimensions.py   # Dimension: 7-tuple of SI exponents (L,M,T,I,Θ,N,J)
│   ├── units.py        # Unit: dimension + scale; all unit definitions
│   └── dimarray.py     # DimArray: numpy wrapper with unit tracking
├── functions.py        # Array functions: concatenate, stack, dot, matmul, norm
├── errors.py           # DimensionError, UnitConversionError
└── config.py           # Display options
tests/
├── test_dimensions.py
├── test_dimarray.py
├── test_functions.py
└── test_config.py
```

## How It Works

1. **Dimension** - Immutable dataclass with 7 `Fraction` exponents for SI base dimensions. Multiply adds exponents, divide subtracts, power multiplies.

2. **Unit** - Combines `Dimension` with a `scale` factor (float relative to SI base). Example: `km` has dimension L, scale 1000.

3. **DimArray** - Wraps `numpy.ndarray` with a `Unit`. All arithmetic checks/propagates dimensions:
   - `+`/`-`: require same dimension (auto-converts units)
   - `*`/`/`: multiply/divide dimensions
   - `**`: requires dimensionless exponent
   - NumPy ufuncs (`np.sin`, `np.exp`): require dimensionless; `np.sqrt` halves exponents

## Code Conventions

- Use `DimArray._from_data_and_unit(data, unit)` internally (no copy)
- Operations return new DimArray instances (immutable style)
- Scalars are wrapped in 1D arrays for consistency
- Add new units to `_DIMENSION_TO_SYMBOL` dict in `units.py` for simplification

## Adding Features

**New unit:**
```python
# In src/dimtensor/core/units.py
new_unit = Unit("symbol", Dimension(...), scale)
```

**New array function:**
```python
# In src/dimtensor/functions.py
def new_func(array: DimArray) -> DimArray:
    result = np.some_func(array._data)
    return DimArray._from_data_and_unit(result, array._unit)
```

**New ufunc support:**
Add to the appropriate set in `DimArray.__array_ufunc__` in `dimarray.py`.

## Testing Requirements

- All new functionality needs tests
- Test dimensional error cases (operations on incompatible dimensions should raise `DimensionError`)
- Test unit conversion cases
- Run full test suite before committing: `pytest`

## Current Status

Version 0.3.x - NumPy parity phase complete. Next: physical constants (v0.4). See ROADMAP.md for full plan.

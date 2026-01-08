# CLAUDE.md

Guidance for Claude Code when working with this repository.

---

## Quick Start

1. **Read CONTINUITY.md first** - Contains current task queue and workflow
2. **Read AGENTS.md** - Contains detailed instructions for agents
3. **Work through the TASK QUEUE** - Don't stop to ask for approval

---

## Project Overview

**dimtensor** is a Python library providing unit-aware tensors for physics and scientific machine learning.

Features:
- NumPy DimArray with unit tracking
- PyTorch DimTensor with autograd
- JAX DimArray with JIT/vmap/grad
- Physical constants (CODATA 2022)
- Uncertainty propagation
- Serialization (JSON, Pandas, HDF5)

---

## Commands

```bash
# Install
pip install -e ".[dev]"

# Test (REQUIRED before commits)
pytest

# Type check
mypy src/dimtensor --ignore-missing-imports

# Coverage
pytest --cov=dimtensor --cov-report=term-missing

# Deploy
python -m build && twine upload dist/*
```

---

## Architecture

```
src/dimtensor/
├── core/
│   ├── dimensions.py   # Dimension: 7-tuple SI exponents (L,M,T,I,Θ,N,J)
│   ├── units.py        # Unit: dimension + scale factor
│   └── dimarray.py     # DimArray: numpy wrapper with units
├── torch/
│   └── dimtensor.py    # DimTensor: PyTorch integration
├── jax/
│   └── dimarray.py     # JAX DimArray with pytree registration
├── io/
│   ├── json.py         # JSON serialization
│   ├── pandas.py       # Pandas DataFrame integration
│   └── hdf5.py         # HDF5 support
├── constants/          # Physical constants (CODATA 2022)
│   ├── universal.py    # c, G, h, hbar
│   ├── electromagnetic.py
│   ├── atomic.py
│   └── physico_chemical.py
├── benchmarks.py       # Performance measurement
├── functions.py        # Array functions (concatenate, dot, matmul)
├── errors.py           # DimensionError, UnitConversionError
└── config.py           # Display options
```

---

## Key Files

| File | Purpose |
|------|---------|
| CONTINUITY.md | **Task queue and session log** - Read first |
| AGENTS.md | Detailed agent instructions |
| ROADMAP.md | Long-term vision |
| .plans/ | Planning documents for complex tasks |

---

## Core Design

- `Dimension`: Frozen dataclass with 7 `Fraction` exponents
- `Unit`: Dimension + scale factor (e.g., km = L with scale 1000)
- `DimArray`: numpy.ndarray + Unit, enforces dimensional correctness

**Operations:**
- `+`/`-`: Require same dimension
- `*`/`/`: Multiply dimensions
- `**`: Requires dimensionless exponent
- `np.sin`, `np.exp`: Require dimensionless input
- `np.sqrt`: Halves dimension exponents

---

## Key Patterns

```python
# Internal constructor (no copy)
DimArray._from_data_and_unit(data, unit)

# All operations return new instances (immutable)
result = a + b  # new DimArray

# Unit simplification
# kg·m/s² automatically displays as N
```

---

## Workflow

See CONTINUITY.md for the current task queue.

```
1. Read CONTINUITY.md
2. Update AGENT CHECKIN
3. Work through TASK QUEUE
4. Update CONTINUITY.md after each task
5. KEEP GOING until queue empty or blocked
```

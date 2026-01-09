# CLAUDE.md

Guidance for Claude Code when working with this repository.

---

## FIRST: Setup Verification

**Before ANY work, run this setup:**

```bash
cd "/Users/marcsperzel/Local Documents/Projects/Packages/dimtensor"

# Install with all dependencies
pip install -e ".[dev,all]"

# Verify NumPy version (must be <2.0)
python -c "import numpy; print(f'NumPy: {numpy.__version__}'); assert int(numpy.__version__.split('.')[0]) < 2"

# Quick test
pytest -x -q --tb=short 2>&1 | tail -5
```

**If NumPy 2.x**: `pip install "numpy>=1.24.0,<2.0.0" --force-reinstall`

---

## "Start Orchestrator" Command

**If the user says "start orchestrator", "orchestrate", or "run orchestrator":**

You become the ORCHESTRATOR. Do this IMMEDIATELY:

1. Read `CONTINUITY.md` to find pending tasks
2. Read `.claude/agents/README.md` for sub-agent instructions
3. Spawn sub-agents in PARALLEL using Task tool:
   - `planner` for 🗺️ tasks
   - `implementer` for feature work
   - `test-writer` for tests
   - `deployer` for PyPI releases
4. Update CONTINUITY.md with results
5. KEEP GOING until all tasks are DONE

**Spawn multiple sub-agents in ONE message for parallelism!**

Example spawn:
```
Task tool call 1: "You are a PLANNER. Read .claude/agents/planner.md. Create plan for task #X"
Task tool call 2: "You are an IMPLEMENTER. Read .claude/agents/implementer.md. Implement task #Y"
```

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

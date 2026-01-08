# Continuity Log

This file tracks the state of autonomous work sessions. Updated continuously to survive context collapses.

**READ THIS FILE FIRST ON CONTEXT RESTORE**

---

## CURRENT SESSION

**Date**: 2026-01-08
**Started**: 22:50 CET
**Current Time**: 22:55 CET
**Hours Worked**: 0.1

### Active Task
Setup complete - ready for worker agent to begin

### Task Queue (Priority Order)
1. [DONE] Set up autonomous work system
2. [PENDING] Complete v0.5.0: Correlation tracking → deploy to PyPI
3. [PENDING] Implement v0.6.0: PyTorch DimTensor → deploy to PyPI
4. [PENDING] Implement v0.7.0: JAX integration → deploy to PyPI
5. [PENDING] Implement v0.8.0: Performance optimization → deploy to PyPI
6. [PENDING] Implement v0.9.0: Serialization → deploy to PyPI
7. [PENDING] Implement v1.0.0: Production readiness → deploy to PyPI

### Target
By morning: PyPI should show v1.0.0

---

## WORKFLOW

### Before Each Task
1. Read CONTINUITY.md to get current state
2. Update this file: Set task as "IN PROGRESS", note start time
3. Update TodoWrite tool: Mark task in_progress
4. Read relevant source files

### During Task
1. Write code
2. Run tests: `pytest`
3. Fix any failures
4. Update tests if needed

### After Each Task
1. Run full test suite: `pytest` - MUST PASS
2. Update CONTINUITY.md: Mark complete, note time
3. Update TodoWrite: Mark completed
4. Update CHANGELOG.md with changes
5. Update ROADMAP.md checkboxes if milestone reached

### After Each Version Milestone (v0.5, v0.6, etc.)
1. Update version in `pyproject.toml` AND `src/dimtensor/__init__.py`
2. Git commit and push
3. Build and deploy to PyPI
4. Log deployment in session log

---

## DEPLOYMENT COMMANDS

```bash
# Working directory
cd "/Users/marcsperzel/Local Documents/Projects/Packages/dimtensor"

# Run tests (MUST PASS before any commit)
pytest

# Git commit and push
git add -A
git commit -m "Release vX.Y.0: Brief description"
git push origin main

# Build package
python -m build

# Deploy to PyPI
twine upload dist/*

# Clean dist after deploy
rm -rf dist/ build/ *.egg-info
```

### Version Update Locations
1. `pyproject.toml` line 7: `version = "X.Y.0"`
2. `src/dimtensor/__init__.py` line 35: `__version__ = "X.Y.0"`

---

## PROJECT STATE

### Current Version: 0.4.0 (on PyPI)
### Next Version: 0.5.0 (in development)

### Git Remote
- origin: https://github.com/marcoloco23/dimtensor.git

### Test Status
- 225 tests passing
- 84% coverage
- Last run: 22:45 CET 2026-01-08

---

## VERSION PLAN

| Version | Focus | Key Features | Status |
|---------|-------|--------------|--------|
| 0.5.0 | Uncertainty | Correlation tracking | IN PROGRESS |
| 0.6.0 | PyTorch | DimTensor, autograd, GPU | PENDING |
| 0.7.0 | JAX | pytree, JIT, vmap, grad | PENDING |
| 0.8.0 | Performance | Rust backend, lazy eval | PENDING |
| 0.9.0 | Serialization | HDF5, NetCDF, Parquet, Pandas | PENDING |
| 1.0.0 | Production | API freeze, 100% coverage, docs | PENDING |

---

## v0.5.0 DETAILS - Correlation Tracking

### What's Done
- Uncertainty storage in DimArray constructor
- Propagation through +, -, *, /, ** operations
- Properties: .uncertainty, .relative_uncertainty, .has_uncertainty
- String formatting with ± symbol
- Unit conversion scales uncertainty
- Reduction operations (sum, mean, min, max)

### What's Needed
Correlation tracking for correlated measurements:
- When x and y are correlated with coefficient ρ:
  - Addition: σ_z² = σ_x² + σ_y² + 2ρσ_xσ_y
  - Multiplication: (σ_z/z)² = (σ_x/x)² + (σ_y/y)² + 2ρ(σ_x/x)(σ_y/y)

### Design Decision Needed
Choose approach:
1. **Explicit correlation parameter**: `a.add(b, correlation=0.5)`
2. **Correlated group**: `with correlated(a, b, rho=0.5): result = a + b`
3. **Correlation registry**: Track correlations globally

Recommend: Option 1 (simplest, explicit, no hidden state)

---

## SESSION LOG

### 22:50 - Session Start
- Analyzed codebase state
- Found 225 tests passing, 84% coverage
- Identified v0.5.0 needs correlation tracking

### 22:55 - Setup Complete
- Created CONTINUITY.md tracking system
- Documented workflow and deployment commands
- Ready for worker agent

---

## KEY DECISIONS

(Worker agent: Record design decisions here)

---

## BLOCKERS

(Worker agent: Record blockers here, user will check)

---

## FILES MODIFIED THIS SESSION

(Track all files changed)

---

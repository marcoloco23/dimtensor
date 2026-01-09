# Plan: Comprehensive Benchmark Suite

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create a production-ready benchmark suite that tracks dimtensor performance across all backends (NumPy, PyTorch, JAX), compares against competitor libraries (pint, astropy.units, unyt), provides statistical rigor, and integrates with CI for regression detection.

---

## Background

**Current state**:
- `src/dimtensor/benchmarks.py` has basic timing utilities for NumPy DimArray
- 9 benchmark functions: creation, addition, multiplication, division, power, sum, matmul, indexing, chained_ops
- Simple `time_operation()` with warmup, returns single timing value
- Text-only output via `print_results()`
- No statistical analysis, no comparison with other libraries, no CI integration

**Why needed**:
- v3.6.0 theme is "Production-ready speed"
- Need to quantify overhead vs competitors (docs claim 2-5x but no data)
- Need regression detection to prevent performance degradation
- Need comprehensive coverage of all backends (PyTorch, JAX not benchmarked)
- Need ML-specific operations (conv2d, attention, batching) for production ML use

**Expected outcomes**:
- Overhead target: <5x for NumPy, <3x for PyTorch/JAX, <2x for GPU ops
- Statistical confidence: 95% confidence intervals, outlier detection
- CI integration: Fail build if regression >20%
- Comprehensive reports: JSON for machines, HTML for humans

---

## Approach

### Option A: asv (airspeed velocity)

**Description**: Industry-standard benchmarking tool used by NumPy, pandas, scikit-learn, etc.

**Pros**:
- Mature, battle-tested tool
- Built-in HTML report generation with graphs over time
- Tracks performance across git history
- Automatic outlier detection and statistical analysis
- Integrates with CI via `asv continuous` and `asv compare`
- Standard format (benchmarks/benchmarks.py with bench_* functions)

**Cons**:
- Separate infrastructure from pytest
- Requires dedicated environment management
- Learning curve for configuration
- Slower (builds each commit in isolated env)
- Requires benchmark machine with consistent hardware

### Option B: pytest-benchmark

**Description**: pytest plugin for benchmarking, integrates with existing test suite.

**Pros**:
- Seamless integration with existing pytest setup
- Uses pytest fixtures and marks
- JSON output for CI integration
- Statistical analysis built-in (mean, stddev, median, IQR)
- Can compare against baselines
- Faster iteration (no isolated envs)

**Cons**:
- Less mature than asv
- HTML reports require separate tool
- No automatic historical tracking
- Less feature-rich statistical analysis
- Not industry standard for performance tracking

### Option C: Custom solution (extend existing benchmarks.py)

**Description**: Build on existing benchmarks.py infrastructure.

**Pros**:
- Complete control over implementation
- Can integrate exactly as needed
- No additional dependencies
- Tailored to dimtensor's specific needs

**Cons**:
- Reinventing the wheel
- Time-consuming to implement statistical rigor
- Won't be as feature-rich as mature tools
- No community/ecosystem support
- Maintenance burden

### Decision: **Hybrid approach (asv + enhanced benchmarks.py)**

**Rationale**:
1. **Keep existing benchmarks.py** for quick developer iteration and pytest integration
2. **Add asv for production benchmarking** with historical tracking, HTML reports, CI integration
3. **Share benchmark logic** - write benchmarks once, run in both frameworks
4. **Best of both worlds**: Fast iteration + production-grade tracking

**Architecture**:
```
benchmarks/
├── benchmarks.py           # asv entry point
├── suites/
│   ├── numpy_ops.py        # NumPy DimArray benchmarks
│   ├── torch_ops.py        # PyTorch DimTensor benchmarks
│   ├── jax_ops.py          # JAX DimArray benchmarks
│   ├── ml_ops.py           # ML-specific (conv, attention, batch norm)
│   ├── competitors.py      # vs pint, astropy, unyt
│   └── conversions.py      # Unit conversions, scaling
└── asv.conf.json           # asv configuration

src/dimtensor/benchmarks.py  # Keep for quick dev use, pytest integration
```

---

## Implementation Steps

### Phase 1: Setup Infrastructure
1. [ ] Install asv and pytest-benchmark as dev dependencies
2. [ ] Create benchmarks/ directory structure
3. [ ] Create asv.conf.json configuration
4. [ ] Create benchmarks/benchmarks.py entry point that imports from suites/
5. [ ] Add .asv/ to .gitignore (benchmark results)
6. [ ] Create benchmarks/README.md with usage instructions

### Phase 2: Core Benchmarks (NumPy)
7. [ ] Extract benchmark logic from src/dimtensor/benchmarks.py to benchmarks/suites/numpy_ops.py
8. [ ] Add parametrization for array sizes: [1, 100, 10_000, 1_000_000, 10_000_000]
9. [ ] Add benchmarks for:
   - Array creation (from list, from numpy, copy)
   - Arithmetic ops (+, -, *, /, //, %, **)
   - Comparison ops (==, !=, <, >, <=, >=)
   - Reduction ops (sum, mean, std, min, max, prod)
   - Shape ops (reshape, transpose, squeeze, expand_dims)
   - Indexing (scalar, slice, fancy, boolean)
   - Broadcasting operations
   - Unit conversions (compatible units)
10. [ ] Ensure each benchmark returns timing in seconds

### Phase 3: PyTorch Benchmarks
11. [ ] Create benchmarks/suites/torch_ops.py
12. [ ] Benchmark CPU operations (all ops from Phase 2)
13. [ ] Benchmark GPU operations (if available):
    - Creation on GPU
    - Arithmetic ops on GPU
    - Reductions on GPU
    - Device transfers (CPU<->GPU with units)
14. [ ] Benchmark autograd operations:
    - Forward pass with units
    - Backward pass with gradient preservation
    - Mixed forward/backward with dimensional constraints

### Phase 4: JAX Benchmarks
15. [ ] Create benchmarks/suites/jax_ops.py
16. [ ] Benchmark eager mode operations
17. [ ] Benchmark JIT compiled operations:
    - First compilation (tracing overhead)
    - Subsequent calls (cached)
18. [ ] Benchmark vmap operations (batching)
19. [ ] Benchmark grad operations (autodiff)
20. [ ] Benchmark pytree registration overhead

### Phase 5: ML-Specific Operations
21. [ ] Create benchmarks/suites/ml_ops.py
22. [ ] Benchmark neural network operations:
    - Linear layers (matmul + bias)
    - Conv1d, Conv2d (with unit tracking)
    - Batch normalization
    - Attention mechanism (scaled dot-product)
    - Pooling (max, average)
23. [ ] Benchmark batch operations:
    - Concatenation along batch dim
    - Stacking/unstacking
    - Batch indexing
24. [ ] Benchmark loss functions with units
25. [ ] Benchmark optimizer steps with dimensional gradients

### Phase 6: Competitor Comparisons
26. [ ] Create benchmarks/suites/competitors.py
27. [ ] Implement benchmarks for pint:
    - Quantity creation
    - Arithmetic operations
    - Unit conversions
    - Comparison with DimArray
28. [ ] Implement benchmarks for astropy.units:
    - Quantity creation
    - Arithmetic operations
    - Unit conversions
    - Comparison with DimArray
29. [ ] Implement benchmarks for unyt:
    - Array creation
    - Operations
    - Conversions
    - Comparison with DimArray
30. [ ] Create comparison matrix: overhead ratios for each library

### Phase 7: Statistical Rigor
31. [ ] Configure asv for multiple iterations (default: 10-100 depending on operation)
32. [ ] Set warmup rounds (default: 2-5)
33. [ ] Configure statistical analysis:
    - Report mean, median, std dev, min, max
    - Calculate 95% confidence intervals
    - Detect outliers (>3 std devs)
34. [ ] Add memory profiling (track_memory=True in asv)
35. [ ] Add benchmark metadata (units, dtypes, devices)

### Phase 8: Reporting
36. [ ] Configure asv HTML output (asv publish)
37. [ ] Create custom comparison reports:
    - Overhead vs NumPy/PyTorch/JAX raw
    - Overhead vs competitors (pint, astropy, unyt)
    - Performance by array size (scaling analysis)
    - Performance by operation type
38. [ ] Generate JSON results for CI parsing
39. [ ] Create summary script: benchmarks/summarize.py
    - Reads JSON results
    - Prints key metrics (P50, P95, max overhead)
    - Flags regressions

### Phase 9: CI Integration
40. [ ] Create .github/workflows/benchmarks.yml
41. [ ] Run benchmarks on PRs (quick mode: smaller arrays, fewer iterations)
42. [ ] Run benchmarks on main branch (full mode: all sizes, full iterations)
43. [ ] Compare PR vs main using `asv continuous`
44. [ ] Fail build if regression >20% on key operations:
    - NumPy addition/multiplication
    - PyTorch forward/backward
    - JAX JIT operations
45. [ ] Post benchmark summary as PR comment
46. [ ] Upload benchmark results as artifacts

### Phase 10: Documentation
47. [ ] Update docs/performance/benchmarks.md with:
    - How to run benchmarks locally
    - How to interpret results
    - CI integration details
    - Performance targets
48. [ ] Create benchmarks/README.md with:
    - Setup instructions
    - Running individual benchmarks
    - Adding new benchmarks
    - Troubleshooting
49. [ ] Add performance targets to CONTRIBUTING.md
50. [ ] Update CHANGELOG.md with benchmark suite addition

### Phase 11: Enhanced Developer Experience
51. [ ] Update src/dimtensor/benchmarks.py to use shared logic from benchmarks/suites/
52. [ ] Add `python -m dimtensor benchmark` CLI command:
    - Quick benchmark (default)
    - Full benchmark (--full)
    - Compare against baseline (--compare)
    - Output formats: text (default), json, csv
53. [ ] Add pytest-benchmark integration for quick tests:
    - `pytest --benchmark-only`
    - Saves results to .benchmarks/
54. [ ] Create pre-commit hook (optional): Warn if changes affect benchmark functions

---

## Files to Modify

| File | Change |
|------|--------|
| **New: benchmarks/asv.conf.json** | asv configuration (repo, env, matrix) |
| **New: benchmarks/benchmarks.py** | asv entry point, imports from suites/ |
| **New: benchmarks/suites/numpy_ops.py** | NumPy DimArray benchmarks (extract from existing) |
| **New: benchmarks/suites/torch_ops.py** | PyTorch DimTensor benchmarks (CPU, GPU, autograd) |
| **New: benchmarks/suites/jax_ops.py** | JAX DimArray benchmarks (eager, JIT, vmap, grad) |
| **New: benchmarks/suites/ml_ops.py** | ML operations (layers, attention, batch ops) |
| **New: benchmarks/suites/competitors.py** | Benchmarks vs pint, astropy, unyt |
| **New: benchmarks/suites/conversions.py** | Unit conversion benchmarks |
| **New: benchmarks/README.md** | Benchmark usage documentation |
| **New: benchmarks/summarize.py** | Parse JSON results, generate summaries |
| **New: .github/workflows/benchmarks.yml** | CI workflow for benchmarks |
| **Modify: src/dimtensor/benchmarks.py** | Refactor to use shared logic from benchmarks/suites/ |
| **Modify: src/dimtensor/__main__.py** | Add `benchmark` subcommand |
| **Modify: pyproject.toml** | Add asv and pytest-benchmark to dev dependencies |
| **Modify: .gitignore** | Add .asv/, .benchmarks/ |
| **Modify: docs/performance/benchmarks.md** | Update with new benchmark suite info |
| **Modify: tests/test_benchmarks.py** | Update tests for refactored benchmarks.py |

---

## Testing Strategy

### Unit Tests
- [ ] Test benchmark functions return valid results
- [ ] Test parametrization works (all sizes run)
- [ ] Test competitor libraries are optional (skip if not installed)
- [ ] Test JSON output is valid
- [ ] Test summary script parses results correctly

### Integration Tests
- [ ] Run asv benchmark suite locally (smoke test)
- [ ] Verify HTML reports generate correctly
- [ ] Test asv continuous (compare two commits)
- [ ] Test CI workflow in feature branch
- [ ] Verify regression detection works (introduce intentional slowdown)

### Manual Verification
- [ ] Run full benchmark suite on clean main branch (establish baseline)
- [ ] Verify overhead ratios match documented expectations:
  - NumPy: 2-5x
  - PyTorch: 2-4x
  - JAX: 2-3x
  - GPU: <2x
- [ ] Verify competitor comparisons show dimtensor is competitive
- [ ] Check HTML reports render correctly in browser
- [ ] Verify CI passes on clean PR

---

## Risks / Edge Cases

### Risk 1: Inconsistent benchmark machine
**Problem**: Benchmarks are sensitive to hardware, CPU throttling, background processes.
**Mitigation**:
- Use dedicated CI runner (GitHub Actions) for consistent results
- Run multiple iterations, report median (robust to outliers)
- Warm up before timing
- Document expected variance (±10% is normal)

### Risk 2: Optional dependencies (torch, jax, pint, astropy, unyt)
**Problem**: Not all dependencies installed in all environments.
**Mitigation**:
- Mark benchmarks as skipped if library not available (asv supports this)
- Use try/except imports
- Document which benchmarks require which dependencies
- CI runs full suite with all dependencies

### Risk 3: GPU availability
**Problem**: GPU benchmarks require CUDA, may not be available in CI.
**Mitigation**:
- Mark GPU benchmarks as skipped if CUDA not available
- Run GPU benchmarks manually on dedicated hardware
- Document GPU benchmark results separately
- Focus CI on CPU benchmarks

### Risk 4: Benchmark code drift
**Problem**: benchmarks.py and asv benchmarks may diverge.
**Mitigation**:
- Share core logic in benchmarks/suites/
- Import from suites/ in both src/dimtensor/benchmarks.py and benchmarks/benchmarks.py
- Single source of truth for benchmark functions
- Tests verify both interfaces work

### Risk 5: Long benchmark runtime
**Problem**: Full benchmark suite may take 30+ minutes.
**Mitigation**:
- Quick mode for PRs (subset of sizes, fewer iterations)
- Full mode for main branch (nightly or on-demand)
- Cache results, only re-run changed benchmarks
- Parallelize independent benchmarks

### Risk 6: Regression false positives
**Problem**: Noise in benchmarks may trigger false positives.
**Mitigation**:
- Set threshold to 20% (avoids noise, catches real regressions)
- Run multiple iterations, report median
- Allow CI to be re-run if suspect false positive
- Document known flaky benchmarks

### Edge Case 1: Array size scaling
**Handling**: Test sizes from 1 to 10M elements, verify overhead decreases for large arrays (overhead is fixed cost, amortizes over data).

### Edge Case 2: Different dtypes
**Handling**: Benchmark float32, float64 separately (GPU prefers float32, scientific computing uses float64).

### Edge Case 3: Complex units (products, powers)
**Handling**: Benchmark m*s, m/s², kg*m/s² to test unit algebra overhead.

### Edge Case 4: Unit conversions across scales
**Handling**: Benchmark mm->m (1000x), km->m (1/1000), to test conversion logic.

---

## Definition of Done

- [ ] All implementation steps complete (1-54)
- [ ] asv benchmark suite runs successfully
- [ ] HTML reports generated and viewable
- [ ] CI workflow runs on PRs and main
- [ ] Regression detection working (tested with intentional slowdown)
- [ ] All tests pass (pytest tests/test_benchmarks.py)
- [ ] Documentation updated (docs/performance/benchmarks.md, benchmarks/README.md)
- [ ] Performance targets met:
  - NumPy overhead <5x
  - PyTorch overhead <4x
  - JAX overhead <3x
  - GPU overhead <2x
- [ ] Competitor comparisons show dimtensor is competitive (within 20% of pint/astropy/unyt)
- [ ] CONTINUITY.md updated with task completion

---

## Notes / Log

**[2026-01-09 Research Phase]** - Analyzed existing benchmarks.py:
- 9 NumPy benchmarks exist
- Simple time_operation() with warmup
- No statistical analysis
- No PyTorch/JAX coverage
- No competitor comparisons

**[2026-01-09 Decision]** - Chose hybrid approach (asv + benchmarks.py):
- asv for production/historical tracking
- Keep benchmarks.py for quick dev iteration
- Share logic via benchmarks/suites/ modules

**[2026-01-09 Scope]** - Key metrics to track:
- Time per operation (mean, median, P95)
- Memory usage (peak, average)
- Overhead ratio (dimtensor/raw)
- Scaling behavior (overhead vs array size)
- Competitor comparison (dimtensor vs pint/astropy/unyt)

---

# Plan: Profile CUDA Overhead for DimTensor

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner (agent)

---

## Goal

Profile DimTensor's CUDA performance to identify overhead sources when compared to raw PyTorch tensors, enabling targeted optimization for v3.6.0 "Production-ready speed" release.

---

## Background

**Why is this needed?**
- v3.6.0 targets production-ready performance
- DimTensor wraps PyTorch tensors with unit tracking - overhead unknown for GPU operations
- Need baseline measurements to guide optimization efforts
- CUDA operations behave differently than CPU (kernel launches, memory transfers, async execution)

**Current state:**
- `benchmarks.py` exists for NumPy DimArray CPU overhead
- No GPU-specific profiling infrastructure
- DimTensor has wrapper overhead from:
  - Unit dimension checking (`__add__`, `__sub__`)
  - Unit conversion (`to_unit()`)
  - Object creation (`_from_tensor_and_unit()`)
  - Property access (`dimension`, `unit`)

**Key questions to answer:**
1. What is the overhead for common operations (add, mul, matmul)?
2. Does overhead scale with tensor size?
3. Is overhead constant (Python wrapper) or per-element (CUDA kernel)?
4. Where does overhead come from (CPU-side checks vs GPU operations)?
5. Do autograd operations add extra overhead?

---

## Approach

### Option A: torch.profiler (PyTorch's Built-in Profiler)
- **Description**: Use `torch.profiler.profile()` with CUDA tracking
- **Pros**:
  - Built into PyTorch (no extra dependencies)
  - Tracks CPU time, CUDA time, and memory
  - Integrates with TensorBoard
  - Records kernel launches and synchronization
  - Call stack awareness
- **Cons**:
  - Some overhead from profiler itself
  - Requires interpretation of traces

### Option B: CUDA Events (Manual Timing)
- **Description**: Use `torch.cuda.Event()` for precise GPU timing
- **Pros**:
  - Minimal overhead
  - Precise GPU time measurement
  - Good for micro-benchmarks
  - Easy to understand results
- **Cons**:
  - Doesn't show CPU overhead
  - Manual synchronization needed
  - Misses kernel launch overhead

### Option C: Combined Approach (Recommended)
- **Description**: Use CUDA Events for micro-benchmarks + torch.profiler for detailed analysis
- **Pros**:
  - CUDA Events: Fast, accurate GPU time
  - torch.profiler: Understand where CPU time goes
  - Complementary insights
- **Cons**:
  - More code to write
  - Two tools to learn

### Decision: Option C - Combined Approach

**Rationale:**
- CUDA Events provide clean overhead measurements (DimTensor GPU time / PyTorch GPU time)
- torch.profiler reveals CPU-side bottlenecks (dimension checking, unit conversion)
- Together they show both Python overhead and GPU overhead
- Follows benchmarks.py pattern (simple timing) but extends it for GPU

---

## Implementation Steps

### Phase 1: Infrastructure (Benchmark Module)
1. [ ] Create `src/dimtensor/torch/benchmarks.py` (analogous to `src/dimtensor/benchmarks.py`)
2. [ ] Implement CUDA event timer utility
   - `cuda_time_operation(func, iterations, warmup)` - returns GPU time in seconds
   - Handles CUDA synchronization correctly
   - Averages over iterations
3. [ ] Implement BenchmarkResult dataclass for PyTorch
   - `name`, `pytorch_time`, `dimtensor_time`, `tensor_size`, `device`, `dtype`
   - `overhead` property (dimtensor_time / pytorch_time)
4. [ ] Add device detection utility (skip if CUDA unavailable)

### Phase 2: Micro-benchmarks (CUDA Events)
Implement benchmark functions for common operations:

5. [ ] `benchmark_creation_cuda()` - Tensor creation on GPU
6. [ ] `benchmark_device_transfer()` - CPU→GPU, GPU→CPU transfer
7. [ ] `benchmark_addition_cuda()` - Element-wise addition
8. [ ] `benchmark_multiplication_cuda()` - Element-wise multiplication
9. [ ] `benchmark_division_cuda()` - Element-wise division
10. [ ] `benchmark_matmul_cuda()` - Matrix multiplication (GEMM)
11. [ ] `benchmark_reduction_cuda()` - Sum, mean operations
12. [ ] `benchmark_autograd_forward()` - Forward pass with requires_grad=True
13. [ ] `benchmark_autograd_backward()` - Backward pass timing
14. [ ] `benchmark_mixed_ops_cuda()` - Chained operations (realistic workload)

### Phase 3: Profiler Integration (Detailed Analysis)
15. [ ] Create `profile_dimtensor()` function using torch.profiler
   - Profile a representative workload (e.g., mini neural network forward pass)
   - Compare DimTensor vs raw PyTorch
   - Output profiler results to console and optionally TensorBoard
16. [ ] Identify top CPU bottlenecks from profiler traces

### Phase 4: Variability Testing
17. [ ] Test across tensor sizes: `[1000, 10_000, 100_000, 1_000_000]`
18. [ ] Test across batch/matrix sizes: `[(32, 32), (128, 128), (1024, 1024)]`
19. [ ] Test different dtypes: `float32`, `float16`, `bfloat16`
20. [ ] Test different devices: `cuda:0`, `cpu` (for comparison)

### Phase 5: Reporting
21. [ ] Create `print_cuda_results()` function (formatted table)
22. [ ] Generate summary report with:
    - Per-operation overhead percentages
    - Total overhead vs tensor size (plot or table)
    - Recommendations (e.g., "multiplication overhead is negligible, addition has 20% overhead")
23. [ ] Document findings in `.plans/2026-01-09_cuda-profiling.md` (Notes section)

### Phase 6: Testing
24. [ ] Create `tests/test_torch_benchmarks.py` (basic smoke test)
25. [ ] Ensure benchmarks run without errors (even if CUDA unavailable)
26. [ ] Add pytest skip decorators for non-CUDA environments

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/torch/benchmarks.py` | **CREATE** - CUDA profiling utilities and benchmark suite |
| `tests/test_torch_benchmarks.py` | **CREATE** - Tests for benchmarking module |
| `src/dimtensor/torch/__init__.py` | Add exports for benchmarks (optional, for convenience) |
| `docs/performance/cuda_profiling.md` | **CREATE** - Document CUDA profiling results and methodology |

---

## Testing Strategy

How will we verify this works?

- [ ] Unit tests: Verify benchmark functions return valid BenchmarkResult objects
- [ ] Integration test: Run full benchmark suite on CUDA (if available)
- [ ] Manual verification:
  - [ ] Run benchmarks on a machine with CUDA GPU
  - [ ] Verify overhead percentages make sense (expect 5-50% overhead depending on operation)
  - [ ] Compare results to CPU benchmarks in `benchmarks.py` for sanity
- [ ] CI/CD: Tests should skip gracefully if CUDA not available (use `@pytest.mark.skipif(not torch.cuda.is_available())`)

**Success criteria:**
- Benchmarks run without errors
- Overhead measurements are reproducible (within 10% variance)
- Results clearly identify where overhead comes from

---

## Risks / Edge Cases

**Risk 1: CUDA Timing Complexity**
- CUDA operations are asynchronous - need explicit synchronization
- **Mitigation:** Use `torch.cuda.synchronize()` before starting/stopping timers

**Risk 2: Warmup Effects**
- First CUDA operation triggers context initialization (100ms+ overhead)
- **Mitigation:** Run warmup iterations (discard first 5-10 runs)

**Risk 3: Kernel Fusion and Optimization**
- PyTorch may fuse operations; DimTensor wrapper may prevent fusion
- **Mitigation:** Profile realistic workloads (chained ops) in addition to individual operations

**Risk 4: Memory Transfer Overhead**
- Small tensors may be dominated by kernel launch overhead, not compute
- **Mitigation:** Test multiple tensor sizes to identify size-dependent overhead

**Risk 5: No CUDA Available**
- CI/CD and some developers may not have GPUs
- **Mitigation:** Use `@pytest.mark.skipif`, provide CPU fallback timing (with warnings)

**Edge Case 1: Mixed CPU/GPU operations**
- DimTensor may force synchronization when mixing devices
- **Handling:** Document this behavior, benchmark device transfer explicitly

**Edge Case 2: Autograd Overhead**
- Gradient tracking may interact with unit tracking
- **Handling:** Separate benchmarks for `requires_grad=True` vs `False`

**Edge Case 3: Different CUDA Versions**
- Performance may vary by CUDA toolkit version, GPU architecture
- **Handling:** Document environment in benchmark output (GPU name, CUDA version, PyTorch version)

---

## Definition of Done

- [ ] `src/dimtensor/torch/benchmarks.py` created with all benchmark functions
- [ ] Tests pass (including skip logic for non-CUDA environments)
- [ ] Documentation shows example output and interpretation
- [ ] Benchmark suite runs successfully on at least one CUDA GPU
- [ ] Results documented in Notes section below with:
  - Overhead percentages for key operations
  - Bottleneck identification (CPU vs GPU)
  - Recommendations for optimization targets
- [ ] CONTINUITY.md updated with task completion

---

## Notes / Log

### Implementation Notes

**Expected Overhead Sources:**
1. **Python wrapper overhead (CPU):**
   - Dimension checking in `__add__`, `__sub__` (CPU-bound)
   - Unit conversion factor calculation (CPU-bound)
   - Object creation `_from_tensor_and_unit()` (CPU-bound)

2. **GPU overhead:**
   - Minimal - most operations just wrap `torch.Tensor._data`
   - Possible synchronization points

**Baseline Expectations:**
- CPU-heavy operations (creation, small tensors): 10-100% overhead
- GPU-heavy operations (large matmul): 1-10% overhead
- Autograd: Similar overhead to base operations

**Tool Choices:**
- CUDA Events: Primary measurement tool (minimal overhead)
- torch.profiler: Secondary analysis tool (understand CPU bottlenecks)
- No nvprof/nsight needed for this phase (those are for kernel-level optimization)

**Testing Matrix:**
```
Operations: [creation, add, mul, div, matmul, sum, autograd]
Sizes: [1K, 10K, 100K, 1M elements]
Dtypes: [float32, float16, bfloat16]
Devices: [cuda, cpu (baseline)]
```

---

## Next Steps After Plan Approval

1. Implementer agent: Create `src/dimtensor/torch/benchmarks.py`
2. Test-writer agent: Create `tests/test_torch_benchmarks.py`
3. Run benchmarks on GPU hardware
4. Analyze results and update this plan with findings
5. Use findings to guide v3.6.0 optimization work

---

# Plan: CUDA Kernels for DimTensor Operations

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Implement fused CUDA kernels for common DimTensor operations that combine unit checking with computation to achieve <10% GPU overhead compared to raw PyTorch tensors.

---

## Background

Currently, DimTensor operations have two separate steps:
1. Unit dimension checking (Python/Rust)
2. PyTorch tensor computation (GPU if available)

This sequential approach causes overhead from:
- Python-CUDA synchronization points
- Multiple kernel launches for unit check + computation
- Data transfers between CPU (unit metadata) and GPU (tensor data)

For v3.6.0 "Production-ready speed", we need to minimize this overhead to <10% for GPU operations.

**Current state**:
- DimTensor wraps torch.Tensor with Unit metadata
- All operations check dimensions in Python, then call PyTorch ops
- Rust backend exists for CPU, but no GPU acceleration
- Benchmarks show ~58% overhead for CPU operations (1M elements)

**Related files**:
- `src/dimtensor/torch/dimtensor.py` - Current implementation
- `src/dimtensor/benchmarks.py` - Performance measurement
- `rust/` - Existing Rust backend (CPU only)

---

## Approach

### Option A: CUDA C++ via torch.utils.cpp_extension

**Description**: Write custom CUDA kernels that encode dimension checks directly into GPU code.

**Pros**:
- Maximum control over kernel optimization
- Can fuse unit checking with computation in single kernel launch
- torch.utils.cpp_extension provides seamless PyTorch integration
- JIT compilation or ahead-of-time build
- Supports both forward and backward passes for autograd

**Cons**:
- Requires CUDA toolkit at runtime (fallback needed)
- More complex to maintain
- Need to handle different data types (float32, float64, etc.)
- Must implement custom autograd functions

### Option B: Triton Kernels

**Description**: Use OpenAI Triton to write GPU kernels in Python-like syntax.

**Pros**:
- Easier to write and maintain (Python-like)
- Automatic optimization for different GPUs
- Good integration with PyTorch
- Supports autograd

**Cons**:
- Additional dependency (Triton)
- Less mature than CUDA C++
- May have more overhead than hand-optimized CUDA
- Limited to operations Triton supports well

### Option C: Hybrid Approach

**Description**: Start with CUDA C++ for critical operations (add, mul, matmul), use Triton for experimental/less critical ops.

**Pros**:
- Best performance for critical path
- Easier iteration for new features
- Can benchmark both approaches

**Cons**:
- Two codebases to maintain
- More complex build system

### Decision: Option C - Hybrid Approach

**Rationale**:
1. Use CUDA C++ for the most common operations (add, sub, mul, div, matmul) where <10% overhead is critical
2. Use Triton for experimental features and less performance-critical operations
3. This allows us to hit the <10% target while maintaining flexibility
4. Both can coexist with proper fallback logic

**Key design principles**:
- Encode dimension as 7 int8_t values (compact, cacheable)
- Pack dimension with tensor metadata in single kernel launch
- Fuse dimension checking with computation (single kernel)
- Use template metaprogramming for different dimension operations
- Fall back to Python implementation when CUDA unavailable

---

## Implementation Steps

### Phase 1: Infrastructure Setup
1. [ ] Create `src/dimtensor/torch/cuda/` directory structure
2. [ ] Set up CMakeLists.txt or setup.py for CUDA compilation
3. [ ] Add CUDA availability detection in `__init__.py`
4. [ ] Create fallback mechanism (use Python impl if no CUDA)
5. [ ] Add GPU benchmarking utilities to `benchmarks.py`

### Phase 2: Core CUDA Kernels (C++)
6. [ ] Design dimension representation for GPU (7 int8_t packed struct)
7. [ ] Implement `cuda_kernels.cu` with fused operations:
   - `dim_add_kernel` - addition with dimension check
   - `dim_sub_kernel` - subtraction with dimension check
   - `dim_mul_kernel` - multiplication with dimension combination
   - `dim_div_kernel` - division with dimension combination
8. [ ] Implement `cuda_bindings.cpp` - C++ to Python bindings
9. [ ] Add autograd support via `torch.autograd.Function`
10. [ ] Handle different dtypes (float32, float64, float16)
11. [ ] Add proper error handling (dimension mismatch on GPU)

### Phase 3: Linear Algebra Kernels
12. [ ] Implement `dim_matmul_kernel` - fused dimension check + matmul
13. [ ] Implement `dim_dot_kernel` - fused dimension check + dot product
14. [ ] Optimize for different matrix sizes (small, medium, large)
15. [ ] Add cuBLAS integration for large matrices

### Phase 4: Build System Integration
16. [ ] Update `pyproject.toml` with CUDA build requirements
17. [ ] Create setup.py with `torch.utils.cpp_extension.CUDAExtension`
18. [ ] Add conditional compilation based on CUDA availability
19. [ ] Test JIT compilation vs ahead-of-time build
20. [ ] Create wheel build scripts for different CUDA versions

### Phase 5: Triton Kernels (Optional/Experimental)
21. [ ] Create `src/dimtensor/torch/triton/` directory
22. [ ] Implement `dim_kernels.py` with Triton kernels
23. [ ] Add Triton fallback chain: CUDA C++ → Triton → Python

### Phase 6: Integration with DimTensor
24. [ ] Modify `dimtensor.py` to detect and use CUDA kernels
25. [ ] Add `use_cuda_kernels` configuration option
26. [ ] Ensure autograd compatibility (backward passes work)
27. [ ] Add device-specific optimizations (A100, V100, etc.)

### Phase 7: Testing & Benchmarking
28. [ ] Create `tests/test_cuda_kernels.py`
29. [ ] Test correctness against Python implementation
30. [ ] Test dimension error detection on GPU
31. [ ] Test autograd (gradients flow correctly)
32. [ ] Benchmark overhead vs raw PyTorch (target: <10%)
33. [ ] Benchmark different tensor sizes (1K, 10K, 100K, 1M, 10M)
34. [ ] Profile kernel launch overhead

### Phase 8: Documentation & Polish
35. [ ] Add CUDA kernel documentation
36. [ ] Update README with GPU acceleration section
37. [ ] Create installation guide for CUDA support
38. [ ] Add performance comparison charts

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/torch/cuda/__init__.py` | Create: CUDA kernel module initialization |
| `src/dimtensor/torch/cuda/kernels.cu` | Create: CUDA C++ kernels (add, sub, mul, div, matmul) |
| `src/dimtensor/torch/cuda/bindings.cpp` | Create: PyTorch C++ bindings |
| `src/dimtensor/torch/cuda/autograd.py` | Create: Custom autograd functions |
| `src/dimtensor/torch/triton/kernels.py` | Create: Triton kernel implementations |
| `src/dimtensor/torch/dimtensor.py` | Modify: Detect and use CUDA kernels when available |
| `src/dimtensor/benchmarks.py` | Modify: Add GPU benchmarking functions |
| `src/dimtensor/config.py` | Modify: Add `use_cuda_kernels` option |
| `pyproject.toml` | Modify: Add CUDA build dependencies |
| `setup.py` | Create: CUDA extension build configuration |
| `tests/test_cuda_kernels.py` | Create: CUDA kernel tests |
| `tests/test_torch_gpu.py` | Create: GPU-specific integration tests |

---

## Testing Strategy

### Unit Tests
- [ ] Test each CUDA kernel independently with known inputs/outputs
- [ ] Test dimension checking correctness (should error on mismatches)
- [ ] Test different tensor shapes (1D, 2D, 3D, etc.)
- [ ] Test different dtypes (float32, float64, float16, bfloat16)
- [ ] Test edge cases (empty tensors, scalar tensors, large tensors)

### Integration Tests
- [ ] Test DimTensor operations use CUDA kernels when available
- [ ] Test fallback to Python implementation when CUDA unavailable
- [ ] Test autograd backward passes
- [ ] Test mixed CPU/GPU operations

### Performance Tests
- [ ] Benchmark overhead vs raw PyTorch for each operation
- [ ] Verify <10% overhead target is met
- [ ] Test scaling across tensor sizes (1K to 10M elements)
- [ ] Profile kernel launch overhead
- [ ] Compare CUDA C++ vs Triton performance

### Compatibility Tests
- [ ] Test on different CUDA versions (11.x, 12.x)
- [ ] Test on different GPUs (T4, V100, A100, etc.)
- [ ] Test with different PyTorch versions (2.0+)

---

## Risks / Edge Cases

### Risk 1: CUDA Toolkit Availability
**Problem**: Users may not have CUDA toolkit installed.
**Mitigation**:
- Provide pre-built wheels for common CUDA versions
- Implement graceful fallback to Python implementation
- Clear documentation on CUDA requirements

### Risk 2: Dimension Error Handling on GPU
**Problem**: GPU error handling is harder than CPU.
**Mitigation**:
- Use atomic operations to set error flags
- Check error flags after kernel execution
- Provide clear error messages with dimension mismatch details

### Risk 3: Autograd Compatibility
**Problem**: Custom CUDA kernels may break PyTorch autograd.
**Mitigation**:
- Use `torch.autograd.Function` for all custom operations
- Implement backward passes for all operations
- Extensive testing with `torch.autograd.gradcheck`

### Risk 4: Memory Overhead
**Problem**: Dimension metadata may increase memory usage.
**Mitigation**:
- Pack dimension as 7 int8_t (7 bytes) instead of Python objects
- Share dimension metadata across operations
- Profile memory usage in benchmarks

### Risk 5: Build Complexity
**Problem**: CUDA builds are complex and environment-specific.
**Mitigation**:
- Use torch.utils.cpp_extension for automatic configuration
- Support both JIT and AOT compilation
- Provide Docker images with pre-configured environments
- CI/CD testing on different platforms

### Edge Case 1: Mixed Device Operations
**Scenario**: DimTensor on CPU, another on GPU.
**Handling**: Detect device mismatch, automatically transfer to same device before operation.

### Edge Case 2: Non-contiguous Tensors
**Scenario**: Tensor memory is not contiguous.
**Handling**: Call `.contiguous()` before kernel launch, or implement strided access.

### Edge Case 3: Very Small Tensors
**Scenario**: Tensor size < 1000 elements, kernel launch overhead dominates.
**Handling**: Use Python implementation for small tensors, CUDA only for large.

### Edge Case 4: Dimension Mismatch During Training
**Scenario**: Dimension error occurs in middle of training loop.
**Handling**: Raise clear exception with operation context, prevent training state corruption.

---

## Definition of Done

- [ ] All implementation steps complete (1-38)
- [ ] CUDA C++ kernels for add, sub, mul, div, matmul implemented
- [ ] Autograd backward passes work correctly
- [ ] All tests pass (unit, integration, performance)
- [ ] GPU overhead <10% for tensors >10K elements
- [ ] Fallback to Python implementation works
- [ ] Documentation updated (README, API docs, installation guide)
- [ ] Benchmarks show performance improvements
- [ ] CI/CD includes GPU testing
- [ ] CONTINUITY.md updated with results

---

## Notes / Log

**Design Decision: Dimension Representation**
GPU kernels will use a compact representation:
```cpp
struct Dimension {
    int8_t length;      // L
    int8_t mass;        // M
    int8_t time;        // T
    int8_t current;     // I
    int8_t temperature; // Θ
    int8_t amount;      // N
    int8_t luminosity;  // J
};
```
Total: 7 bytes (fits in cache line, minimal overhead)

**Design Decision: Kernel Fusion**
Instead of:
1. Launch kernel to check dimensions
2. Launch PyTorch operation

We do:
1. Launch single fused kernel that checks dimensions AND computes result

This eliminates one kernel launch and one synchronization point.

**Design Decision: Error Handling**
Use device-side error flag:
```cpp
__global__ void dim_add_kernel(float* result, float* a, float* b,
                               Dimension dim_a, Dimension dim_b,
                               int* error_flag) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (!dimensions_compatible(dim_a, dim_b)) {
            *error_flag = 1;
            return;
        }
    }
    __syncthreads();

    if (*error_flag) return;

    // Actual computation...
}
```

**Performance Target Breakdown**:
- Dimension checking: <1% overhead
- Kernel launch fusion: 5-7% savings
- Memory access optimization: 2-3% savings
- Total target: <10% overhead

---

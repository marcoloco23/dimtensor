# dimtensor Benchmarks

This directory contains the comprehensive benchmark suite for dimtensor, measuring performance across all backends and comparing against competitor libraries.

## Quick Start

### Prerequisites

```bash
# Install dimtensor with benchmark dependencies
pip install -e ".[dev,benchmark]"
```

### Running Benchmarks

#### ASV (Airspeed Velocity) - Production Benchmarking

```bash
# Run all benchmarks on current commit
asv run

# Run specific benchmark
asv run --bench ArrayCreation

# Run benchmarks matching pattern
asv run --bench numpy_ops

# Compare two commits
asv continuous main HEAD

# Generate HTML reports
asv publish
asv preview
```

#### Quick Benchmarks (pytest-benchmark)

```bash
# Run quick benchmarks using pytest
pytest benchmarks/ --benchmark-only

# Save results
pytest benchmarks/ --benchmark-only --benchmark-json=output.json

# Compare against saved baseline
pytest benchmarks/ --benchmark-only --benchmark-compare=baseline.json
```

#### Using existing benchmarks.py

```bash
# Quick benchmark from Python
python -c "from dimtensor.benchmarks import benchmark_suite, print_results; print_results(benchmark_suite())"
```

## Benchmark Suites

### NumPy Operations (`suites/numpy_ops.py`)

Benchmarks core DimArray operations against raw NumPy:

- **ArrayCreation**: Array creation from lists, numpy arrays, copying
- **ArithmeticOps**: Addition, subtraction, multiplication, division, power
- **ReductionOps**: Sum, mean, std, min, max
- **MatrixOps**: Matrix multiplication, transpose
- **IndexingOps**: Scalar, slice, fancy, boolean indexing
- **ChainedOps**: Multiple operations in sequence
- **ShapeOps**: Reshape, flatten, squeeze
- **BroadcastingOps**: Broadcasting operations

**Array sizes tested**: 1, 100, 10,000, 1,000,000 elements

### PyTorch Operations (`suites/torch_ops.py`)

Benchmarks DimTensor (PyTorch backend):

- **TorchBasicOps**: Basic arithmetic and reductions (CPU)
- **TorchMatrixOps**: Matrix operations (CPU)
- **TorchAutogradOps**: Forward and backward passes with gradients
- **TorchGPUOps**: GPU operations (if CUDA available)
- **TorchDeviceTransfers**: CPU ↔ GPU transfers

**Requires**: `torch>=2.0.0`

### Competitor Comparisons (`suites/competitors.py`)

Compares dimtensor against:

- **pint**: Popular Python units library
- **astropy.units**: Astronomy units library
- **unyt**: yt-project units library

Operations compared:
- Array creation
- Addition (same units)
- Multiplication (different units)
- Reduction operations
- Unit conversions
- Chained operations

**Requires**: One or more of `pint`, `astropy`, `unyt`

## Understanding Results

### ASV Output

ASV reports timing in seconds and generates:
- Mean time per operation
- Standard deviation
- Comparison against previous commits
- HTML reports with graphs

Example output:
```
ArrayCreation.time_from_list_dimarray       10.5μs ±0.5μs
ArrayCreation.time_from_list_numpy           2.1μs ±0.1μs
Overhead: 5.0x
```

### Performance Targets

Expected overhead for dimtensor vs raw backends:

| Backend | Operation | Target Overhead |
|---------|-----------|----------------|
| NumPy   | Arithmetic | <5x |
| NumPy   | Large arrays (>1M) | <3x |
| PyTorch | CPU operations | <4x |
| PyTorch | GPU operations | <2x |
| JAX     | JIT compiled | <3x |

### Interpreting Overhead

- **1-2x overhead**: Excellent - minimal cost for unit safety
- **2-5x overhead**: Good - acceptable for most use cases
- **5-10x overhead**: Fair - may need optimization for hot loops
- **>10x overhead**: Poor - indicates performance issue

Overhead typically decreases with array size due to fixed costs amortizing over more data.

## Configuration

### ASV Configuration (`asv.conf.json`)

Key settings:
- **pythons**: Python versions to test (3.10, 3.11, 3.12)
- **matrix**: Dependency versions to test (numpy, torch, jax)
- **benchmark_dir**: Location of benchmark files
- **results_dir**: Where results are stored (`.asv/results/`)
- **html_dir**: Where HTML reports are generated (`.asv/html/`)

### Customizing Benchmarks

To add a new benchmark:

1. Add a class to appropriate suite file (e.g., `suites/numpy_ops.py`)
2. Implement `setup()` method to prepare test data
3. Implement `time_*()` methods for each operation to benchmark
4. Use `param_names` and `params` for parametrization

Example:

```python
class MyBenchmark:
    """Benchmark description."""

    param_names = ['size']
    params = [[100, 10_000, 1_000_000]]

    def setup(self, size):
        """Set up test data."""
        self.data = np.random.randn(size)
        self.da = DimArray(self.data, m)

    def time_my_operation(self, size):
        """Time my operation."""
        self.da.my_operation()
```

## CI Integration

Benchmarks run in CI:
- **On PRs**: Quick mode (smaller arrays, fewer iterations)
- **On main**: Full mode (all sizes, full statistical analysis)
- **Regression detection**: Fails if >20% slowdown on key operations

See `.github/workflows/benchmarks.yml` for CI configuration.

## Troubleshooting

### "No benchmarks found"

Make sure:
- You're in the `benchmarks/` directory
- Benchmark classes have `time_*` methods
- Imports are not failing (check `import dimtensor`)

### "ImportError: No module named 'dimtensor'"

Install dimtensor in development mode:
```bash
pip install -e ..
```

### "PyTorch/JAX benchmarks skipped"

Optional dependencies not installed. Install with:
```bash
pip install -e "..[torch,jax]"
```

### "CUDA benchmarks skipped"

GPU not available or CUDA not installed. GPU benchmarks are optional.

### Slow benchmark runs

ASV runs many iterations for statistical significance. For quick tests:
```bash
# Run fewer iterations
asv run --quick

# Run specific benchmark
asv run --bench ArrayCreation.time_add
```

## Performance Optimization Tips

If benchmarks show high overhead:

1. **Profile hot paths**: Use `cProfile` or `line_profiler`
2. **Check unit operations**: Unit algebra should be O(1)
3. **Minimize copies**: Use `_from_data_and_unit()` internally
4. **Cache unit conversions**: Don't recompute scale factors
5. **Vectorize operations**: Let NumPy/PyTorch do the heavy lifting

## Further Reading

- [ASV Documentation](https://asv.readthedocs.io/)
- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/)
- [NumPy Performance Tips](https://numpy.org/doc/stable/user/performance.html)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

## Contributing

When adding features to dimtensor:

1. Add corresponding benchmarks to appropriate suite
2. Run benchmarks locally before submitting PR
3. Ensure no regressions (>20% slowdown)
4. Update this README if adding new benchmark suites

---

For questions or issues, please open a GitHub issue: https://github.com/marcoloco23/dimtensor/issues

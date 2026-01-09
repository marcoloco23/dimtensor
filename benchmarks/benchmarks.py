"""ASV benchmark entry point for dimtensor.

This file imports all benchmark suites for airspeed velocity (asv).
Each suite is in a separate module under benchmarks/suites/.

To run benchmarks:
    asv run
    asv publish
    asv preview

See README.md for more details.
"""

# Import all benchmark suites
# ASV will discover all classes with time_* methods

# NumPy DimArray benchmarks
from suites.numpy_ops import (
    ArrayCreation,
    ArithmeticOps,
    ReductionOps,
    MatrixOps,
    IndexingOps,
    ChainedOps,
    ShapeOps,
    BroadcastingOps,
)

# PyTorch DimTensor benchmarks (optional - skipped if torch not available)
try:
    from suites.torch_ops import (
        TorchBasicOps,
        TorchMatrixOps,
        TorchAutogradOps,
        TorchGPUOps,
        TorchDeviceTransfers,
    )
except ImportError:
    # PyTorch not available, skip these benchmarks
    pass

# Competitor library benchmarks (optional - skipped if not available)
try:
    from suites.competitors import (
        CreationComparison,
        AdditionComparison,
        MultiplicationComparison,
        ReductionComparison,
        UnitConversionComparison,
        ChainedOpsComparison,
    )
except ImportError:
    # Competitor libs not available, skip these benchmarks
    pass

# Export all imported benchmark classes
# (ASV will auto-discover them, but explicit is better)
__all__ = [
    # NumPy benchmarks
    'ArrayCreation',
    'ArithmeticOps',
    'ReductionOps',
    'MatrixOps',
    'IndexingOps',
    'ChainedOps',
    'ShapeOps',
    'BroadcastingOps',
    # PyTorch benchmarks (if available)
    'TorchBasicOps',
    'TorchMatrixOps',
    'TorchAutogradOps',
    'TorchGPUOps',
    'TorchDeviceTransfers',
    # Competitor benchmarks (if available)
    'CreationComparison',
    'AdditionComparison',
    'MultiplicationComparison',
    'ReductionComparison',
    'UnitConversionComparison',
    'ChainedOpsComparison',
]

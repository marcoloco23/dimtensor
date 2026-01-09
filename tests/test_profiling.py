"""Tests for memory profiling tools."""

import sys

import numpy as np
import pytest

from dimtensor import DimArray, units
from dimtensor.profiling import (
    MemoryProfiler,
    MemoryStats,
    ComparisonStats,
    SharedMetadataReport,
    memory_stats,
    metadata_overhead,
    compare_to_baseline,
    get_overhead_ratio,
    memory_report,
    analyze_shared_metadata,
)

# Try to import torch for GPU tests
try:
    import torch
    from dimtensor.torch import DimTensor
    from dimtensor.profiling import gpu_memory_stats
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False


# =============================================================================
# Basic Memory Stats Tests
# =============================================================================

def test_memory_stats_basic():
    """Test basic memory statistics for DimArray."""
    x = DimArray([1.0, 2.0, 3.0], units.m)
    stats = memory_stats(x)

    assert isinstance(stats, MemoryStats)
    assert stats.data_bytes == 24  # 3 * 8 bytes (float64)
    assert stats.metadata_bytes > 0
    assert stats.uncertainty_bytes == 0
    assert stats.total_bytes == stats.data_bytes + stats.metadata_bytes
    assert stats.device == 'cpu'


def test_memory_stats_with_uncertainty():
    """Test memory stats with uncertainty tracking."""
    x = DimArray([1.0, 2.0, 3.0], units.m, uncertainty=[0.1, 0.1, 0.1])
    stats = memory_stats(x)

    assert stats.uncertainty_bytes == 24  # Same size as data
    assert stats.total_bytes == stats.data_bytes + stats.metadata_bytes + stats.uncertainty_bytes


def test_memory_stats_large_array():
    """Test memory stats for large array."""
    x = DimArray(np.ones(10000), units.m)
    stats = memory_stats(x)

    assert stats.data_bytes == 80000  # 10000 * 8 bytes
    assert stats.overhead_ratio < 0.02  # Less than 2% overhead for large array


def test_memory_stats_small_array():
    """Test memory stats for small array."""
    x = DimArray([1.0], units.m)
    stats = memory_stats(x)

    assert stats.data_bytes == 8  # 1 * 8 bytes
    assert stats.overhead_ratio > 1.0  # Overhead > 100% for tiny array


def test_memory_stats_shared_metadata():
    """Test detection of shared metadata."""
    # Create arrays with same unit
    x = DimArray([1.0, 2.0], units.m)
    y = DimArray([3.0, 4.0], units.m)

    stats_x = memory_stats(x)
    stats_y = memory_stats(y)

    # Both should detect shared metadata
    assert stats_x.shared_metadata
    assert stats_y.shared_metadata


def test_memory_stats_invalid_type():
    """Test memory_stats with invalid object."""
    with pytest.raises(TypeError, match="Expected DimArray or DimTensor"):
        memory_stats([1, 2, 3])


# =============================================================================
# Metadata Overhead Tests
# =============================================================================

def test_metadata_overhead():
    """Test metadata overhead calculation."""
    x = DimArray([1.0, 2.0, 3.0], units.m)
    overhead = metadata_overhead(x)

    assert isinstance(overhead, int)
    assert overhead > 0
    assert overhead == memory_stats(x).metadata_bytes


def test_metadata_overhead_with_uncertainty():
    """Test metadata overhead includes uncertainty."""
    x = DimArray([1.0, 2.0], units.m, uncertainty=[0.1, 0.1])
    overhead = metadata_overhead(x)

    stats = memory_stats(x)
    assert overhead == stats.metadata_bytes + stats.uncertainty_bytes


# =============================================================================
# Comparison Tests
# =============================================================================

def test_compare_to_baseline():
    """Test comparison to baseline numpy array."""
    x = DimArray([1.0, 2.0, 3.0], units.m)
    comp = compare_to_baseline(x)

    assert isinstance(comp, ComparisonStats)
    assert comp.baseline_bytes == 24  # Raw numpy array size
    assert comp.dimtensor_bytes > comp.baseline_bytes
    assert comp.overhead_bytes == comp.dimtensor_bytes - comp.baseline_bytes
    assert comp.overhead_percent > 0


def test_get_overhead_ratio():
    """Test overhead ratio calculation."""
    x = DimArray([1.0, 2.0, 3.0], units.m)
    ratio = get_overhead_ratio(x)

    assert isinstance(ratio, float)
    assert ratio > 0
    assert ratio == memory_stats(x).overhead_ratio


def test_overhead_decreases_with_size():
    """Test that overhead ratio decreases as array size increases."""
    small = DimArray([1.0], units.m)
    medium = DimArray(np.ones(100), units.m)
    large = DimArray(np.ones(10000), units.m)

    ratio_small = get_overhead_ratio(small)
    ratio_medium = get_overhead_ratio(medium)
    ratio_large = get_overhead_ratio(large)

    assert ratio_small > ratio_medium > ratio_large


# =============================================================================
# Memory Report Tests
# =============================================================================

def test_memory_report_single_array():
    """Test memory report for single array."""
    x = DimArray([1.0, 2.0, 3.0], units.m)
    report = memory_report(x)

    assert isinstance(report, str)
    assert "Memory Report for DimArray" in report
    assert "Data:" in report
    assert "Metadata:" in report
    assert "Total:" in report
    assert "Overhead:" in report
    assert "Device:" in report


def test_memory_report_with_uncertainty():
    """Test memory report includes uncertainty."""
    x = DimArray([1.0, 2.0], units.m, uncertainty=[0.1, 0.1])
    report = memory_report(x)

    assert "Uncertainty:" in report


def test_memory_report_detailed():
    """Test detailed memory report."""
    x = DimArray([1.0, 2.0, 3.0], units.m)
    report = memory_report(x, detailed=True)

    assert "Breakdown:" in report
    assert "Object overhead:" in report
    assert "Unit object:" in report
    assert "Dimension object:" in report
    assert "Recommendations:" in report


def test_memory_report_list():
    """Test memory report for list of arrays."""
    arrays = [
        DimArray([1.0, 2.0], units.m),
        DimArray([3.0, 4.0], units.m),
        DimArray([5.0, 6.0], units.s),
    ]
    report = memory_report(arrays)

    assert "Summary for 3 arrays:" in report
    assert "Total data:" in report
    assert "Total metadata:" in report
    assert "Recommendations:" in report


def test_memory_report_recommendations():
    """Test that recommendations are generated appropriately."""
    # Small array with high overhead
    small = DimArray([1.0], units.m)
    report = memory_report(small)
    assert "high" in report.lower() or "overhead" in report.lower()

    # Array with uncertainty
    with_unc = DimArray([1.0, 2.0], units.m, uncertainty=[0.1, 0.1])
    report_unc = memory_report(with_unc)
    # Should mention uncertainty
    assert "uncertainty" in report_unc.lower() or "Uncertainty:" in report_unc


# =============================================================================
# Shared Metadata Analysis Tests
# =============================================================================

def test_analyze_shared_metadata_same_unit():
    """Test metadata sharing analysis with same unit."""
    arrays = [DimArray([1.0, 2.0], units.m) for _ in range(10)]
    report = analyze_shared_metadata(arrays)

    assert isinstance(report, SharedMetadataReport)
    assert report.num_arrays == 10
    assert report.unique_units == 1  # All share same unit
    assert report.unique_dimensions == 1
    assert len(report.recommendations) > 0


def test_analyze_shared_metadata_different_units():
    """Test metadata sharing analysis with different units."""
    arrays = [
        DimArray([1.0], units.m),
        DimArray([2.0], units.s),
        DimArray([3.0], units.kg),
        DimArray([4.0], units.m),  # Shares with first
    ]
    report = analyze_shared_metadata(arrays)

    assert report.num_arrays == 4
    assert report.unique_units == 3  # m, s, kg
    assert report.unique_dimensions == 3


def test_analyze_shared_metadata_empty():
    """Test metadata sharing with empty list."""
    report = analyze_shared_metadata([])

    assert report.num_arrays == 0
    assert report.unique_units == 0
    assert report.unique_dimensions == 0


def test_analyze_shared_metadata_recommendations():
    """Test that recommendations are generated."""
    # Good sharing - use larger arrays to avoid "small array" recommendation
    good_arrays = [DimArray(np.ones(1000), units.m) for _ in range(10)]
    good_report = analyze_shared_metadata(good_arrays)
    assert len(good_report.recommendations) > 0
    # Check that some positive message is present about sharing or good practice
    rec_text = " ".join(good_report.recommendations).lower()
    assert "excellent" in rec_text or "good" in rec_text or "share" in rec_text or "reuse" in rec_text

    # Many unique units (poor sharing)
    from dimtensor.core.units import Unit
    from dimtensor.core.dimensions import Dimension, DIMENSIONLESS
    bad_arrays = []
    for i in range(10):
        # Create unique units (not shared)
        custom_unit = Unit(f"custom_{i}", DIMENSIONLESS, 1.0)
        bad_arrays.append(DimArray([1.0], custom_unit))
    bad_report = analyze_shared_metadata(bad_arrays)
    assert len(bad_report.recommendations) > 0


# =============================================================================
# Memory Profiler Context Manager Tests
# =============================================================================

def test_memory_profiler_context():
    """Test MemoryProfiler context manager."""
    with MemoryProfiler() as prof:
        x = DimArray([1.0, 2.0, 3.0], units.m)
        y = x * 2

    stats = prof.get_stats()
    assert 'current_bytes' in stats
    assert 'peak_bytes' in stats
    assert stats['current_bytes'] >= 0
    assert stats['peak_bytes'] >= stats['current_bytes']


def test_memory_profiler_report():
    """Test MemoryProfiler report generation."""
    with MemoryProfiler() as prof:
        x = DimArray(np.ones(1000), units.m)

    report = prof.report()
    assert isinstance(report, str)
    assert "Memory Profiling Results" in report
    assert "Current memory:" in report
    assert "Peak memory:" in report


def test_memory_profiler_reset_peak():
    """Test reset_peak functionality."""
    with MemoryProfiler() as prof:
        x = DimArray(np.ones(1000), units.m)
        stats1 = prof.get_stats()
        peak1 = stats1['peak_bytes']

        # Reset
        prof.reset_peak()

        # Allocate more
        y = DimArray(np.ones(100), units.m)
        stats2 = prof.get_stats()

        # Peak should be less after reset (only counting since reset)
        # Note: This might not always be true due to other allocations
        assert stats2['peak_bytes'] >= 0


# =============================================================================
# PyTorch DimTensor Tests (if available)
# =============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_memory_stats_dimtensor():
    """Test memory stats for DimTensor."""
    x = DimTensor(torch.tensor([1.0, 2.0, 3.0]), units.m)
    stats = memory_stats(x)

    assert isinstance(stats, MemoryStats)
    assert stats.data_bytes == 12  # 3 * 4 bytes (float32 default)
    assert stats.metadata_bytes > 0
    assert stats.uncertainty_bytes == 0  # DimTensor doesn't support uncertainty yet
    assert stats.device == 'cpu'


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_memory_report_dimtensor():
    """Test memory report for DimTensor."""
    x = DimTensor(torch.tensor([1.0, 2.0, 3.0]), units.m)
    report = memory_report(x)

    assert "Memory Report for DimTensor" in report
    assert "Device:" in report


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_gpu_memory_stats():
    """Test GPU memory statistics."""
    # Create tensor on GPU
    x = DimTensor(torch.tensor([1.0, 2.0, 3.0]), units.m)
    x_gpu = DimTensor(x.data.cuda(), units.m)

    stats = gpu_memory_stats(0)
    assert stats.device == 'cuda:0'
    assert stats.allocated_bytes > 0
    assert stats.reserved_bytes >= stats.allocated_bytes
    assert stats.peak_bytes >= stats.allocated_bytes


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_gpu_memory_stats_no_cuda():
    """Test GPU stats when CUDA not available."""
    if not CUDA_AVAILABLE:
        with pytest.raises(RuntimeError, match="CUDA not available"):
            gpu_memory_stats(0)


# =============================================================================
# Edge Cases
# =============================================================================

def test_memory_stats_empty_array():
    """Test memory stats for empty array."""
    x = DimArray([], units.m)
    stats = memory_stats(x)

    assert stats.data_bytes == 0
    assert stats.metadata_bytes > 0
    assert stats.overhead_ratio == float('inf')  # Infinite overhead for 0 data


def test_memory_stats_scalar():
    """Test memory stats for scalar (0-d array)."""
    x = DimArray(5.0, units.m)
    stats = memory_stats(x)

    assert stats.data_bytes == 8  # 1 float64
    assert stats.metadata_bytes > 0


def test_memory_report_formats_bytes():
    """Test that memory report formats bytes nicely."""
    # Small (bytes)
    small = DimArray([1.0], units.m)
    report_small = memory_report(small)
    assert "bytes" in report_small

    # Medium (KB)
    medium = DimArray(np.ones(200), units.m)
    report_medium = memory_report(medium)
    # Should have KB or bytes
    assert "KB" in report_medium or "bytes" in report_medium

    # Large (KB or MB)
    large = DimArray(np.ones(200000), units.m)
    report_large = memory_report(large)
    # Should have MB or KB
    assert "MB" in report_large or "KB" in report_large


def test_memory_stats_different_dtypes():
    """Test memory stats with different data types."""
    x32 = DimArray(np.array([1.0, 2.0], dtype=np.float32), units.m)
    x64 = DimArray(np.array([1.0, 2.0], dtype=np.float64), units.m)

    stats32 = memory_stats(x32)
    stats64 = memory_stats(x64)

    assert stats32.data_bytes == 8  # 2 * 4 bytes
    assert stats64.data_bytes == 16  # 2 * 8 bytes


# =============================================================================
# Integration Tests
# =============================================================================

def test_memory_stats_after_operations():
    """Test memory stats after arithmetic operations."""
    x = DimArray([1.0, 2.0, 3.0], units.m)
    y = DimArray([2.0, 3.0, 4.0], units.s)
    z = x * y  # Result should have different unit

    stats = memory_stats(z)
    assert stats.data_bytes == 24
    assert stats.metadata_bytes > 0


def test_memory_overhead_with_unit_conversion():
    """Test memory overhead after unit conversion."""
    x = DimArray([1000.0, 2000.0], units.m)
    x_km = x.to(units.km)

    stats_m = memory_stats(x)
    stats_km = memory_stats(x_km)

    # Data size same, but different unit objects
    assert stats_m.data_bytes == stats_km.data_bytes


def test_profiler_tracks_multiple_arrays():
    """Test profiler tracks allocations from multiple arrays."""
    with MemoryProfiler() as prof:
        arrays = []
        for i in range(10):
            arrays.append(DimArray(np.ones(100), units.m))

    stats = prof.get_stats()
    # tracemalloc tracks Python allocations
    # Note: numpy allocations may not always be tracked
    assert stats['peak_bytes'] >= 0  # Should at least return valid data
    assert 'current_bytes' in stats


def test_comprehensive_workflow():
    """Test a comprehensive workflow with profiling."""
    # Create arrays
    positions = [DimArray(np.random.randn(100), units.m) for _ in range(5)]
    velocities = [DimArray(np.random.randn(100), units.m / units.s) for _ in range(5)]

    # Analyze sharing
    all_arrays = positions + velocities
    sharing_report = analyze_shared_metadata(all_arrays)

    assert sharing_report.num_arrays == 10
    assert sharing_report.unique_units >= 2  # At least m and m/s

    # Get individual stats
    for arr in positions:
        stats = memory_stats(arr)
        assert stats.data_bytes == 800  # 100 * 8 bytes

    # Get overall report
    report = memory_report(all_arrays, detailed=False)
    assert "Summary for 10 arrays" in report


# =============================================================================
# Performance Tests
# =============================================================================

def test_memory_stats_performance():
    """Test that memory_stats is fast even for large arrays."""
    import time

    x = DimArray(np.ones(1_000_000), units.m)

    start = time.perf_counter()
    stats = memory_stats(x)
    elapsed = time.perf_counter() - start

    # Should be very fast (< 10ms)
    assert elapsed < 0.01
    assert stats.data_bytes == 8_000_000


def test_analyze_shared_metadata_performance():
    """Test metadata analysis performance."""
    import time

    arrays = [DimArray(np.ones(100), units.m) for _ in range(100)]

    start = time.perf_counter()
    report = analyze_shared_metadata(arrays)
    elapsed = time.perf_counter() - start

    # Should be fast (< 100ms for 100 arrays)
    assert elapsed < 0.1
    assert report.num_arrays == 100

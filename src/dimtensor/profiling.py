"""Memory profiling tools for dimtensor.

This module provides utilities to measure and analyze memory usage of
DimArray and DimTensor objects, helping identify optimization opportunities
and understand the overhead of dimensional safety.

Example:
    >>> from dimtensor import DimArray, units
    >>> from dimtensor.profiling import memory_stats, memory_report
    >>> x = DimArray([1.0, 2.0, 3.0], units.m)
    >>> stats = memory_stats(x)
    >>> print(f"Data: {stats.data_bytes} bytes")
    >>> print(f"Overhead: {stats.overhead_ratio:.1%}")
    >>> print(memory_report(x))
"""

from __future__ import annotations

import sys
import tracemalloc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .core.dimarray import DimArray

# Try to import torch for GPU support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MemoryStats:
    """Memory statistics for a dimtensor object.

    Attributes:
        data_bytes: Size of numerical data array in bytes.
        metadata_bytes: Size of unit/dimension metadata in bytes.
        uncertainty_bytes: Size of uncertainty array in bytes (0 if none).
        total_bytes: Total memory footprint in bytes.
        overhead_ratio: Ratio of metadata to data (metadata / data).
        shared_metadata: Whether unit/dimension are likely shared with other objects.
        device: Device location ('cpu', 'cuda:0', etc.).
    """
    data_bytes: int
    metadata_bytes: int
    uncertainty_bytes: int
    total_bytes: int
    overhead_ratio: float
    shared_metadata: bool
    device: str


@dataclass
class ComparisonStats:
    """Comparison to baseline numpy/torch array.

    Attributes:
        baseline_bytes: Size of raw array without metadata.
        dimtensor_bytes: Size with dimtensor metadata.
        overhead_bytes: Difference (dimtensor - baseline).
        overhead_percent: Percentage overhead ((overhead / baseline) * 100).
    """
    baseline_bytes: int
    dimtensor_bytes: int
    overhead_bytes: int
    overhead_percent: float


@dataclass
class SharedMetadataReport:
    """Analysis of metadata sharing across arrays.

    Attributes:
        num_arrays: Number of arrays analyzed.
        unique_units: Number of unique Unit objects.
        unique_dimensions: Number of unique Dimension objects.
        total_metadata_bytes: Total metadata size if all were separate.
        shared_savings_bytes: Bytes saved through object sharing.
        recommendations: List of optimization suggestions.
    """
    num_arrays: int
    unique_units: int
    unique_dimensions: int
    total_metadata_bytes: int
    shared_savings_bytes: int
    recommendations: list[str]


@dataclass
class GPUMemoryStats:
    """GPU memory statistics (PyTorch only).

    Attributes:
        device: GPU device name (e.g., 'cuda:0').
        allocated_bytes: Currently allocated GPU memory.
        reserved_bytes: Memory reserved by PyTorch.
        peak_bytes: Peak memory usage since last reset.
        dimtensor_estimate: Estimated memory used by DimTensor (approximation).
    """
    device: str
    allocated_bytes: int
    reserved_bytes: int
    peak_bytes: int
    dimtensor_estimate: int


# =============================================================================
# Internal Helper Functions
# =============================================================================

def _get_object_size(obj: Any) -> int:
    """Get the size of an object, handling special cases.

    Args:
        obj: Object to measure.

    Returns:
        Size in bytes.
    """
    return sys.getsizeof(obj)


def _calculate_unit_size(unit: Any) -> int:
    """Calculate the size of a Unit object.

    Args:
        unit: Unit object.

    Returns:
        Size in bytes.
    """
    # Unit has: symbol (str), dimension (ref), scale (float)
    size = _get_object_size(unit)
    size += _get_object_size(unit.symbol)  # string storage
    # Note: dimension is a reference (8 bytes on 64-bit), already counted in object
    return size


def _calculate_dimension_size(dimension: Any) -> int:
    """Calculate the size of a Dimension object.

    Args:
        dimension: Dimension object.

    Returns:
        Size in bytes.
    """
    # Dimension has: 7 Fraction exponents in a tuple
    size = _get_object_size(dimension)

    # Each exponent is a Fraction with numerator and denominator
    if hasattr(dimension, '_exponents'):
        exponents = dimension._exponents
        size += _get_object_size(exponents)  # tuple overhead
        for frac in exponents:
            size += _get_object_size(frac)  # Fraction object
            # numerator and denominator are Python ints (variable size)
            if hasattr(frac, 'numerator'):
                size += _get_object_size(frac.numerator)
            if hasattr(frac, 'denominator'):
                size += _get_object_size(frac.denominator)

    return size


def _is_likely_shared(obj: Any) -> bool:
    """Check if an object is likely shared based on refcount.

    Objects with refcount > 3 are likely shared:
    - 1 for the getrefcount call itself
    - 1 for the owning DimArray
    - 1+ for other references

    Args:
        obj: Object to check.

    Returns:
        True if likely shared with other objects.
    """
    return sys.getrefcount(obj) > 3


def _format_bytes(num_bytes: int) -> str:
    """Format bytes in human-readable form.

    Args:
        num_bytes: Number of bytes.

    Returns:
        Formatted string (e.g., "1.5 MB", "234 KB", "5 bytes").
    """
    if num_bytes < 1024:
        return f"{num_bytes} bytes"
    elif num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.1f} KB"
    elif num_bytes < 1024 ** 3:
        return f"{num_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{num_bytes / (1024 ** 3):.2f} GB"


# =============================================================================
# Core Profiling Functions
# =============================================================================

def memory_stats(obj: Any) -> MemoryStats:
    """Get detailed memory statistics for a dimtensor object.

    Analyzes the memory footprint of a DimArray or DimTensor, breaking down
    data, metadata, and uncertainty components.

    Args:
        obj: DimArray or DimTensor object to analyze.

    Returns:
        MemoryStats with detailed breakdown.

    Example:
        >>> x = DimArray([1.0, 2.0, 3.0], units.m, uncertainty=[0.1, 0.1, 0.1])
        >>> stats = memory_stats(x)
        >>> print(f"Data: {stats.data_bytes} bytes")
        >>> print(f"Overhead: {stats.overhead_ratio:.1%}")
    """
    # Detect type
    is_dimarray = hasattr(obj, '_data') and hasattr(obj._data, 'nbytes')
    is_dimtensor = TORCH_AVAILABLE and hasattr(obj, '_data') and isinstance(obj._data, torch.Tensor)

    if not (is_dimarray or is_dimtensor):
        raise TypeError(f"Expected DimArray or DimTensor, got {type(obj)}")

    # Calculate data size
    if is_dimarray:
        data_bytes = obj._data.nbytes
        device = 'cpu'
    else:  # DimTensor
        data_bytes = obj._data.element_size() * obj._data.nelement()
        device = str(obj._data.device)

    # Calculate metadata size
    # DimArray/DimTensor object overhead
    obj_overhead = _get_object_size(obj)

    # Unit size
    unit_size = _calculate_unit_size(obj._unit)

    # Dimension size
    dim_size = _calculate_dimension_size(obj._unit.dimension)

    metadata_bytes = obj_overhead + unit_size + dim_size

    # Calculate uncertainty size (DimArray only - DimTensor doesn't support uncertainty)
    uncertainty_bytes = 0
    if is_dimarray and hasattr(obj, '_uncertainty') and obj._uncertainty is not None:
        uncertainty_bytes = obj._uncertainty.nbytes

    # Total
    total_bytes = data_bytes + metadata_bytes + uncertainty_bytes

    # Overhead ratio (metadata / data)
    overhead_ratio = metadata_bytes / data_bytes if data_bytes > 0 else float('inf')

    # Check if metadata is shared
    shared_metadata = _is_likely_shared(obj._unit) or _is_likely_shared(obj._unit.dimension)

    return MemoryStats(
        data_bytes=data_bytes,
        metadata_bytes=metadata_bytes,
        uncertainty_bytes=uncertainty_bytes,
        total_bytes=total_bytes,
        overhead_ratio=overhead_ratio,
        shared_metadata=shared_metadata,
        device=device,
    )


def metadata_overhead(obj: Any) -> int:
    """Return bytes of metadata overhead.

    Args:
        obj: DimArray or DimTensor object.

    Returns:
        Total metadata overhead in bytes (unit + dimension + uncertainty).

    Example:
        >>> x = DimArray([1.0, 2.0], units.m)
        >>> overhead = metadata_overhead(x)
        >>> print(f"Overhead: {overhead} bytes")
    """
    stats = memory_stats(obj)
    return stats.metadata_bytes + stats.uncertainty_bytes


def compare_to_baseline(obj: Any) -> ComparisonStats:
    """Compare memory usage to raw numpy/torch equivalent.

    Args:
        obj: DimArray or DimTensor object.

    Returns:
        ComparisonStats showing overhead vs baseline.

    Example:
        >>> x = DimArray([1.0] * 1000, units.m)
        >>> comp = compare_to_baseline(x)
        >>> print(f"Overhead: {comp.overhead_percent:.1f}%")
    """
    stats = memory_stats(obj)
    baseline_bytes = stats.data_bytes  # Raw array only
    dimtensor_bytes = stats.total_bytes
    overhead_bytes = stats.metadata_bytes + stats.uncertainty_bytes
    overhead_percent = (overhead_bytes / baseline_bytes * 100) if baseline_bytes > 0 else 0.0

    return ComparisonStats(
        baseline_bytes=baseline_bytes,
        dimtensor_bytes=dimtensor_bytes,
        overhead_bytes=overhead_bytes,
        overhead_percent=overhead_percent,
    )


def get_overhead_ratio(obj: Any) -> float:
    """Calculate metadata overhead as a percentage.

    Args:
        obj: DimArray or DimTensor object.

    Returns:
        Overhead ratio (metadata / data).

    Example:
        >>> x = DimArray([1.0, 2.0], units.m)
        >>> ratio = get_overhead_ratio(x)
        >>> print(f"Overhead: {ratio:.1%}")
    """
    stats = memory_stats(obj)
    return stats.overhead_ratio


def memory_report(
    obj: Any | list[Any],
    detailed: bool = True
) -> str:
    """Generate formatted memory report.

    Args:
        obj: Single DimArray/DimTensor or list of arrays.
        detailed: Include per-component breakdown.

    Returns:
        Formatted string with memory usage and recommendations.

    Example:
        >>> x = DimArray([1.0, 2.0, 3.0], units.m)
        >>> print(memory_report(x))
        Memory Report for DimArray
        ===========================
        Data:        24 bytes
        Metadata:    456 bytes
        Total:       480 bytes
        Overhead:    1900.0%
        Device:      cpu
        ...
    """
    # Handle list of objects
    if isinstance(obj, list):
        reports = []
        total_data = 0
        total_metadata = 0
        total_uncertainty = 0

        for i, item in enumerate(obj):
            stats = memory_stats(item)
            total_data += stats.data_bytes
            total_metadata += stats.metadata_bytes
            total_uncertainty += stats.uncertainty_bytes

            if detailed:
                reports.append(f"Array {i+1}:")
                reports.append(f"  Data: {_format_bytes(stats.data_bytes)}")
                reports.append(f"  Metadata: {_format_bytes(stats.metadata_bytes)}")
                if stats.uncertainty_bytes > 0:
                    reports.append(f"  Uncertainty: {_format_bytes(stats.uncertainty_bytes)}")
                reports.append("")

        # Summary
        total = total_data + total_metadata + total_uncertainty
        overhead = total_metadata + total_uncertainty
        overhead_pct = (overhead / total_data * 100) if total_data > 0 else 0.0

        reports.append(f"Summary for {len(obj)} arrays:")
        reports.append("=" * 40)
        reports.append(f"Total data:        {_format_bytes(total_data)}")
        reports.append(f"Total metadata:    {_format_bytes(total_metadata)}")
        if total_uncertainty > 0:
            reports.append(f"Total uncertainty: {_format_bytes(total_uncertainty)}")
        reports.append(f"Total memory:      {_format_bytes(total)}")
        reports.append(f"Overhead:          {overhead_pct:.1f}%")

        # Add recommendations
        reports.append("")
        reports.append("Recommendations:")
        if len(obj) > 0:
            shared_report = analyze_shared_metadata(obj)
            for rec in shared_report.recommendations:
                reports.append(f"  - {rec}")

        return "\n".join(reports)

    # Single object
    stats = memory_stats(obj)
    comp = compare_to_baseline(obj)

    obj_type = type(obj).__name__

    lines = [
        f"Memory Report for {obj_type}",
        "=" * 40,
        f"Data:        {_format_bytes(stats.data_bytes)}",
        f"Metadata:    {_format_bytes(stats.metadata_bytes)}",
    ]

    if stats.uncertainty_bytes > 0:
        lines.append(f"Uncertainty: {_format_bytes(stats.uncertainty_bytes)}")

    lines.extend([
        f"Total:       {_format_bytes(stats.total_bytes)}",
        f"Overhead:    {comp.overhead_percent:.1f}%",
        f"Device:      {stats.device}",
        f"Shared:      {'Yes' if stats.shared_metadata else 'No'}",
    ])

    if detailed:
        lines.extend([
            "",
            "Breakdown:",
            f"  Object overhead:  {_format_bytes(_get_object_size(obj))}",
            f"  Unit object:      {_format_bytes(_calculate_unit_size(obj._unit))}",
            f"  Dimension object: {_format_bytes(_calculate_dimension_size(obj._unit.dimension))}",
        ])

    # Add recommendations
    lines.append("")
    lines.append("Recommendations:")

    # Small array overhead
    if stats.data_bytes < 1000 and comp.overhead_percent > 50:
        lines.append(f"  - Metadata overhead is high ({comp.overhead_percent:.0f}%) for small arrays")
        lines.append("    Consider batching operations or using larger arrays")

    # Uncertainty overhead
    if stats.uncertainty_bytes > stats.data_bytes:
        lines.append(f"  - Uncertainty arrays add {_format_bytes(stats.uncertainty_bytes)}")
        lines.append("    Consider uncertainty=None if not needed")

    # Good metadata sharing
    if stats.shared_metadata:
        lines.append("  - Good: Unit/Dimension objects are shared (low memory overhead)")

    # Large array with low overhead
    if stats.data_bytes > 1_000_000 and comp.overhead_percent < 1:
        lines.append(f"  - Excellent: Metadata overhead is only {comp.overhead_percent:.2f}% for large array")

    return "\n".join(lines)


def analyze_shared_metadata(arrays: list[Any]) -> SharedMetadataReport:
    """Analyze unit/dimension sharing across arrays.

    Identifies opportunities for memory savings through object reuse.

    Args:
        arrays: List of DimArray or DimTensor objects.

    Returns:
        Report with sharing statistics and recommendations.

    Example:
        >>> arrays = [DimArray([1, 2], units.m) for _ in range(10)]
        >>> report = analyze_shared_metadata(arrays)
        >>> print(f"Unique units: {report.unique_units}")
        >>> for rec in report.recommendations:
        ...     print(rec)
    """
    if not arrays:
        return SharedMetadataReport(
            num_arrays=0,
            unique_units=0,
            unique_dimensions=0,
            total_metadata_bytes=0,
            shared_savings_bytes=0,
            recommendations=[]
        )

    # Track unique units and dimensions by id
    unit_ids = set()
    dimension_ids = set()
    total_metadata = 0

    for arr in arrays:
        unit_ids.add(id(arr._unit))
        dimension_ids.add(id(arr._unit.dimension))
        stats = memory_stats(arr)
        total_metadata += stats.metadata_bytes

    unique_units = len(unit_ids)
    unique_dimensions = len(dimension_ids)

    # Calculate potential savings
    # If all arrays shared same unit/dimension, we'd only pay once
    if arrays:
        avg_metadata = total_metadata / len(arrays)
        ideal_metadata = avg_metadata  # Just one set of metadata
        shared_savings = total_metadata - ideal_metadata
    else:
        shared_savings = 0

    # Generate recommendations
    recommendations = []

    sharing_ratio = unique_units / len(arrays) if arrays else 0
    if sharing_ratio <= 0.1:
        recommendations.append(
            f"Excellent: {len(arrays)} arrays share {unique_units} unique units (good reuse)"
        )
    elif sharing_ratio > 0.5:
        recommendations.append(
            f"Many unique units ({unique_units} for {len(arrays)} arrays)"
        )
        recommendations.append(
            "Consider using predefined units from dimtensor.units module"
        )

    if unique_dimensions <= unique_units:
        recommendations.append(
            f"Good: Dimensions are well-shared ({unique_dimensions} unique)"
        )

    # Small arrays with high overhead
    small_arrays = sum(1 for arr in arrays if memory_stats(arr).data_bytes < 1000)
    if small_arrays > len(arrays) * 0.5:
        recommendations.append(
            f"{small_arrays}/{len(arrays)} arrays are small (<1KB data)"
        )
        recommendations.append(
            "Consider consolidating small arrays to reduce metadata overhead"
        )

    return SharedMetadataReport(
        num_arrays=len(arrays),
        unique_units=unique_units,
        unique_dimensions=unique_dimensions,
        total_metadata_bytes=total_metadata,
        shared_savings_bytes=int(shared_savings),
        recommendations=recommendations,
    )


def gpu_memory_stats(device: str | int = 0) -> GPUMemoryStats:
    """Get GPU memory statistics (PyTorch only).

    Requires PyTorch with CUDA support.

    Args:
        device: GPU device (0, 1, or 'cuda:0', etc.).

    Returns:
        GPUMemoryStats with current GPU memory usage.

    Raises:
        RuntimeError: If PyTorch or CUDA is not available.

    Example:
        >>> stats = gpu_memory_stats(0)
        >>> print(f"GPU memory: {stats.allocated_bytes / 1e9:.2f} GB")
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Install with: pip install torch")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    # Normalize device string
    if isinstance(device, int):
        device_str = f"cuda:{device}"
    else:
        device_str = device

    device_obj = torch.device(device_str)

    # Get memory stats
    allocated = torch.cuda.memory_allocated(device_obj)
    reserved = torch.cuda.memory_reserved(device_obj)

    # Peak memory (max allocated since last reset)
    peak = torch.cuda.max_memory_allocated(device_obj)

    # Estimate DimTensor memory (this is approximate)
    # We can't distinguish DimTensor from regular Tensor, so we report total
    dimtensor_estimate = allocated  # Conservative estimate

    return GPUMemoryStats(
        device=device_str,
        allocated_bytes=allocated,
        reserved_bytes=reserved,
        peak_bytes=peak,
        dimtensor_estimate=dimtensor_estimate,
    )


# =============================================================================
# Context Manager for Session Profiling
# =============================================================================

class MemoryProfiler:
    """Context manager for tracking memory allocations.

    Uses tracemalloc to track all allocations during a code block.

    Example:
        >>> with MemoryProfiler() as prof:
        ...     x = DimArray([1, 2, 3], units.m)
        ...     y = x * 2
        >>> print(prof.report())
        Memory Profiling Results
        ========================
        Peak memory: 1.5 KB
        Current memory: 1.2 KB
        ...
    """

    def __init__(self) -> None:
        """Initialize profiler."""
        self._started = False
        self._snapshot_start: Any = None
        self._snapshot_end: Any = None

    def __enter__(self) -> MemoryProfiler:
        """Start memory profiling."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._started = True

        # Clear any previous stats
        tracemalloc.clear_traces()

        # Take initial snapshot
        self._snapshot_start = tracemalloc.take_snapshot()

        return self

    def __exit__(self, *args: Any) -> None:
        """Stop memory profiling."""
        # Take final snapshot
        self._snapshot_end = tracemalloc.take_snapshot()

        # Stop tracing if we started it
        if self._started:
            tracemalloc.stop()
            self._started = False

    def get_stats(self) -> dict[str, Any]:
        """Get profiling statistics.

        Returns:
            Dictionary with memory statistics.
        """
        if self._snapshot_end is None:
            # Still inside context, take current snapshot
            current_snapshot = tracemalloc.take_snapshot()
        else:
            current_snapshot = self._snapshot_end

        # Calculate differences
        top_stats = current_snapshot.compare_to(self._snapshot_start, 'lineno')

        # Get current and peak memory
        current, peak = tracemalloc.get_traced_memory()

        return {
            'current_bytes': current,
            'peak_bytes': peak,
            'top_allocations': top_stats[:10],  # Top 10 allocations
        }

    def report(self) -> str:
        """Generate formatted profiling report.

        Returns:
            Human-readable report string.
        """
        stats = self.get_stats()

        lines = [
            "Memory Profiling Results",
            "=" * 40,
            f"Current memory: {_format_bytes(stats['current_bytes'])}",
            f"Peak memory:    {_format_bytes(stats['peak_bytes'])}",
            "",
            "Top allocations:",
        ]

        for i, stat in enumerate(stats['top_allocations'][:5], 1):
            lines.append(f"  {i}. {stat}")

        return "\n".join(lines)

    def reset_peak(self) -> None:
        """Reset peak memory statistics."""
        if tracemalloc.is_tracing():
            # Clear and take new baseline
            tracemalloc.clear_traces()
            self._snapshot_start = tracemalloc.take_snapshot()


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Data classes
    'MemoryStats',
    'ComparisonStats',
    'SharedMetadataReport',
    'GPUMemoryStats',
    # Core functions
    'memory_stats',
    'metadata_overhead',
    'compare_to_baseline',
    'get_overhead_ratio',
    'memory_report',
    'analyze_shared_metadata',
    'gpu_memory_stats',
    # Context manager
    'MemoryProfiler',
]

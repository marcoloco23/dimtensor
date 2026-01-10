"""CuPy integration for dimtensor.

Provides CuPy-compatible DimArray with GPU acceleration, uncertainty propagation,
and full compatibility with NumPy DimArray API.

Example:
    >>> import cupy as cp
    >>> from dimtensor.cupy import DimArray
    >>> from dimtensor import units
    >>>
    >>> # Create GPU arrays with units
    >>> v = DimArray(cp.array([1.0, 2.0, 3.0]), units.m / units.s)
    >>> t = DimArray(cp.array([0.5, 1.0, 1.5]), units.s)
    >>> d = v * t  # distance in meters, computed on GPU
    >>> print(d)
    [0.5 2.0 4.5] m
    >>>
    >>> # Transfer to CPU
    >>> d_numpy = d.numpy()

Requirements:
    CuPy >= 12.0.0 with CUDA support.
    Install with: pip install cupy-cuda12x
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dimarray import DimArray


def is_available() -> bool:
    """Check if CuPy and CUDA are available.

    Returns:
        True if CuPy is installed and CUDA devices are available.
    """
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def __getattr__(name: str) -> type:
    """Lazy import to avoid import errors when CuPy is not installed."""
    if name == "DimArray":
        if not is_available():
            raise ImportError(
                "CuPy is not available. Install with: pip install cupy-cuda12x\n"
                "Ensure you have a compatible CUDA installation."
            )
        from .dimarray import DimArray
        return DimArray
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DimArray", "is_available"]

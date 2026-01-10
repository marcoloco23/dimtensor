"""Dask integration for dimtensor.

Provides DaskDimArray, a distributed/chunked array with unit tracking
that supports lazy evaluation and parallel computation.

Example:
    >>> import dask.array as da
    >>> from dimtensor.dask import DaskDimArray
    >>> from dimtensor import units
    >>>
    >>> # Create from numpy array with chunks
    >>> data = np.random.randn(10000, 10000)
    >>> velocity = DaskDimArray(data, unit=units.m/units.s, chunks=(1000, 1000))
    >>>
    >>> # Lazy operations - no computation yet
    >>> speed = (velocity**2).sum(axis=1).sqrt()
    >>>
    >>> # Trigger computation
    >>> result = speed.compute()  # Returns numpy DimArray
"""

from .dimarray import DaskDimArray

__all__ = ["DaskDimArray"]

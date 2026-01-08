"""dimtensor: Unit-aware tensors for physics and scientific machine learning.

dimtensor provides DimArray, a numpy array wrapper that tracks physical
dimensions through all operations, catching dimensional errors early.

Basic usage:
    >>> from dimtensor import DimArray, units
    >>> velocity = DimArray([10, 20, 30], units.m / units.s)
    >>> time = DimArray([1, 2, 3], units.s)
    >>> distance = velocity * time
    >>> print(distance)
    [10 40 90] m

    >>> acceleration = DimArray([9.8], units.m / units.s**2)
    >>> velocity + acceleration  # DimensionError: cannot add m/s to m/s^2
"""

from .core.dimensions import Dimension, DIMENSIONLESS
from .core.dimarray import DimArray
from .core.units import Unit
from .errors import DimensionError, UnitConversionError

# Import all units into a 'units' namespace
from .core import units

__version__ = "0.1.1"

__all__ = [
    # Core classes
    "DimArray",
    "Dimension",
    "Unit",
    # Constants
    "DIMENSIONLESS",
    # Exceptions
    "DimensionError",
    "UnitConversionError",
    # Modules
    "units",
    # Version
    "__version__",
]

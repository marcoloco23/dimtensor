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
from .errors import DimensionError, UnitConversionError, ConstraintError

# Import all units into a 'units' namespace
from .core import units

# Physical constants
from . import constants

# Domain-specific units
from . import domains

# Visualization
from . import visualization

# Validation
from . import validation

# Configuration
from . import config

# Module-level array functions
from .functions import concatenate, stack, split, dot, matmul, norm

# Profiling tools
from . import profiling

__version__ = "3.6.0"

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
    "ConstraintError",
    # Module-level functions
    "concatenate",
    "stack",
    "split",
    "dot",
    "matmul",
    "norm",
    # Modules
    "units",
    "constants",
    "domains",
    "visualization",
    "validation",
    "config",
    "profiling",
    # Version
    "__version__",
]

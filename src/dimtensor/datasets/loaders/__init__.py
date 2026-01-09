"""Dataset loaders for physics data sources.

Provides loaders for downloading and caching real physics datasets
from NIST, NASA, NOAA, and other trusted sources.

Example:
    >>> from dimtensor.datasets.loaders import NISTCODATALoader
    >>> loader = NISTCODATALoader()
    >>> constants = loader.load()
    >>> c = constants['speed of light in vacuum']
"""

from .astronomy import NASAExoplanetLoader
from .base import BaseLoader, CSVLoader, ensure_cache_dir, get_cache_dir
from .climate import NOAAClimateLoader, PRISMClimateLoader
from .nist import NISTCODATALoader

__all__ = [
    "BaseLoader",
    "CSVLoader",
    "NISTCODATALoader",
    "NASAExoplanetLoader",
    "PRISMClimateLoader",
    "NOAAClimateLoader",
    "get_cache_dir",
    "ensure_cache_dir",
]

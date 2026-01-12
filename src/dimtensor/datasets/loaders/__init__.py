"""Dataset loaders for physics data sources.

Provides loaders for downloading and caching real physics datasets
from NIST, NASA, NOAA, and other trusted sources.

Example:
    >>> from dimtensor.datasets.loaders import NISTCODATALoader
    >>> loader = NISTCODATALoader()
    >>> constants = loader.load()
    >>> c = constants['speed of light in vacuum']
"""

from ..cache import CacheManager, get_cache_manager, set_cache_manager
from .astronomy import NASAExoplanetLoader
from .base import BaseLoader, CSVLoader, ensure_cache_dir, get_cache_dir
from .cern import CERNOpenDataLoader
from .climate import NOAAClimateLoader, PRISMClimateLoader
from .comsol import COMSOLLoader, PhysicsModule, load_comsol_csv, load_comsol_txt
from .gravitational_wave import GWOSCEventLoader, GWOSCStrainLoader
from .nist import NISTCODATALoader
from .noaa import NOAAWeatherLoader
from .openfoam import OpenFOAMLoader
from .pubchem import PubChemLoader
from .sdss import SDSSLoader
from .worldbank import WorldBankClimateLoader

# Optional import - mp_api has dependency issues
try:
    from .materials_project import MaterialsProjectLoader
except (ImportError, TypeError):
    MaterialsProjectLoader = None  # type: ignore

__all__ = [
    "BaseLoader",
    "CSVLoader",
    "NISTCODATALoader",
    "NASAExoplanetLoader",
    "CERNOpenDataLoader",
    "PRISMClimateLoader",
    "NOAAClimateLoader",
    "NOAAWeatherLoader",
    "MaterialsProjectLoader",
    "OpenFOAMLoader",
    "PubChemLoader",
    "SDSSLoader",
    "WorldBankClimateLoader",
    "COMSOLLoader",
    "PhysicsModule",
    "load_comsol_csv",
    "load_comsol_txt",
    "GWOSCEventLoader",
    "GWOSCStrainLoader",
    "get_cache_dir",
    "ensure_cache_dir",
    "CacheManager",
    "get_cache_manager",
    "set_cache_manager",
]

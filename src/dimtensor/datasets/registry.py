"""Dataset registry for physics-aware machine learning.

Provides a registry for discovering and loading datasets with
dimensional metadata for physics-informed neural networks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.dimensions import Dimension


# Global registry
_REGISTRY: dict[str, "DatasetInfo"] = {}
_LOADERS: dict[str, Callable[..., Any]] = {}


@dataclass
class DatasetInfo:
    """Metadata about a physics dataset.

    Attributes:
        name: Unique dataset identifier.
        description: Human-readable description.
        domain: Physics domain (e.g., "mechanics", "fluid_dynamics").
        features: Dict mapping feature names to their dimensions.
        targets: Dict mapping target names to their dimensions.
        size: Number of samples (or None if variable).
        source: URL or path to download dataset.
        license: Dataset license.
        citation: Citation for the dataset.
        tags: List of tags for filtering.
    """

    name: str
    description: str = ""
    domain: str = "general"
    features: dict[str, Dimension] = field(default_factory=dict)
    targets: dict[str, Dimension] = field(default_factory=dict)
    size: int | None = None
    source: str = ""
    license: str = ""
    citation: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "features": {k: str(v) for k, v in self.features.items()},
            "targets": {k: str(v) for k, v in self.targets.items()},
            "size": self.size,
            "source": self.source,
            "license": self.license,
            "citation": self.citation,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetInfo":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            domain=data.get("domain", "general"),
            features={
                k: _dict_to_dim(v) for k, v in data.get("features", {}).items()
            },
            targets={
                k: _dict_to_dim(v) for k, v in data.get("targets", {}).items()
            },
            size=data.get("size"),
            source=data.get("source", ""),
            license=data.get("license", ""),
            citation=data.get("citation", ""),
            tags=data.get("tags", []),
        )


def _dict_to_dim(data: dict[str, float] | str) -> Dimension:
    """Convert dict or string to Dimension."""
    if isinstance(data, str):
        # Parse string like "L¹T⁻²"
        return Dimension()  # Fallback for string format
    return Dimension(
        length=data.get("L", 0),
        mass=data.get("M", 0),
        time=data.get("T", 0),
        current=data.get("I", 0),
        temperature=data.get("Theta", 0),
        amount=data.get("N", 0),
        luminosity=data.get("J", 0),
    )


def register_dataset(
    name: str,
    info: DatasetInfo | None = None,
    loader: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]] | None:
    """Register a dataset in the registry.

    Can be used as a decorator or called directly.

    Args:
        name: Unique dataset identifier.
        info: DatasetInfo with metadata. If None, created from kwargs.
        loader: Loader function to load the dataset.
        **kwargs: Additional arguments passed to DatasetInfo.

    Returns:
        Decorator function if used as decorator, None otherwise.

    Examples:
        # As decorator
        @register_dataset("pendulum", domain="mechanics")
        def load_pendulum():
            return load_data()

        # Direct call
        register_dataset("my-data", info=info, loader=load_fn)
    """
    if info is None:
        info = DatasetInfo(name=name, **kwargs)

    _REGISTRY[name] = info

    if loader is not None:
        _LOADERS[name] = loader
        return None

    # Return decorator
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _LOADERS[name] = fn
        return fn

    return decorator


def get_dataset_info(name: str) -> DatasetInfo:
    """Get metadata for a registered dataset.

    Args:
        name: Dataset identifier.

    Returns:
        DatasetInfo with dataset metadata.

    Raises:
        KeyError: If dataset not found.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Dataset '{name}' not found in registry")
    return _REGISTRY[name]


def list_datasets(
    domain: str | None = None,
    tags: list[str] | None = None,
) -> list[DatasetInfo]:
    """List available datasets in the registry.

    Args:
        domain: Filter by physics domain.
        tags: Filter by tags (must have ALL specified tags).

    Returns:
        List of DatasetInfo matching the filters.
    """
    results = list(_REGISTRY.values())

    if domain is not None:
        results = [d for d in results if d.domain == domain]

    if tags is not None:
        results = [d for d in results if all(t in d.tags for t in tags)]

    return results


def load_dataset(name: str, **kwargs: Any) -> Any:
    """Load a dataset from the registry.

    Args:
        name: Dataset identifier.
        **kwargs: Additional arguments passed to the loader.

    Returns:
        The loaded dataset.

    Raises:
        KeyError: If dataset not found.
        RuntimeError: If no loader registered.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Dataset '{name}' not found in registry")

    if name not in _LOADERS:
        raise RuntimeError(
            f"Dataset '{name}' has no loader function. "
            "Register a loader or provide a source URL."
        )

    return _LOADERS[name](**kwargs)


def clear_datasets() -> None:
    """Clear all registered datasets (for testing)."""
    _REGISTRY.clear()
    _LOADERS.clear()


# ============================================================================
# BUILT-IN PHYSICS DATASETS
# ============================================================================

# Dimensions
_L = Dimension(length=1)
_M = Dimension(mass=1)
_T = Dimension(time=1)
_V = Dimension(length=1, time=-1)
_A = Dimension(length=1, time=-2)
_F = Dimension(mass=1, length=1, time=-2)
_E = Dimension(mass=1, length=2, time=-2)
_TEMP = Dimension(temperature=1)
_PRESSURE = Dimension(mass=1, length=-1, time=-2)
_DIMLESS = Dimension()

# Simple Pendulum Dataset
register_dataset(
    "pendulum",
    info=DatasetInfo(
        name="pendulum",
        description="Simple pendulum motion: angle and angular velocity over time",
        domain="mechanics",
        features={
            "time": _T,
            "length": _L,
            "initial_angle": _DIMLESS,  # radians
        },
        targets={
            "angle": _DIMLESS,  # radians
            "angular_velocity": Dimension(time=-1),
        },
        tags=["oscillation", "classical", "ode"],
    ),
)

# Projectile Motion Dataset
register_dataset(
    "projectile",
    info=DatasetInfo(
        name="projectile",
        description="2D projectile motion under gravity",
        domain="mechanics",
        features={
            "time": _T,
            "initial_velocity": _V,
            "launch_angle": _DIMLESS,
        },
        targets={
            "x_position": _L,
            "y_position": _L,
            "x_velocity": _V,
            "y_velocity": _V,
        },
        tags=["kinematics", "classical", "2d"],
    ),
)

# Spring-Mass System
register_dataset(
    "spring_mass",
    info=DatasetInfo(
        name="spring_mass",
        description="Damped harmonic oscillator",
        domain="mechanics",
        features={
            "time": _T,
            "mass": _M,
            "spring_constant": Dimension(mass=1, time=-2),
            "damping": Dimension(mass=1, time=-1),
        },
        targets={
            "position": _L,
            "velocity": _V,
        },
        tags=["oscillation", "ode", "damped"],
    ),
)

# Heat Diffusion Dataset
register_dataset(
    "heat_diffusion",
    info=DatasetInfo(
        name="heat_diffusion",
        description="1D heat diffusion on a rod",
        domain="thermodynamics",
        features={
            "position": _L,
            "time": _T,
            "thermal_diffusivity": Dimension(length=2, time=-1),
        },
        targets={
            "temperature": _TEMP,
        },
        tags=["pde", "diffusion", "heat"],
    ),
)

# Burgers Equation Dataset
register_dataset(
    "burgers",
    info=DatasetInfo(
        name="burgers",
        description="1D viscous Burgers equation",
        domain="fluid_dynamics",
        features={
            "position": _L,
            "time": _T,
            "viscosity": Dimension(length=2, time=-1),
        },
        targets={
            "velocity": _V,
        },
        tags=["pde", "nonlinear", "cfd"],
    ),
)

# Navier-Stokes 2D Dataset
register_dataset(
    "navier_stokes_2d",
    info=DatasetInfo(
        name="navier_stokes_2d",
        description="2D incompressible Navier-Stokes flow",
        domain="fluid_dynamics",
        features={
            "x": _L,
            "y": _L,
            "time": _T,
            "reynolds_number": _DIMLESS,
        },
        targets={
            "u_velocity": _V,
            "v_velocity": _V,
            "pressure": _PRESSURE,
        },
        tags=["pde", "cfd", "turbulence"],
    ),
)

# Three-Body Problem
register_dataset(
    "three_body",
    info=DatasetInfo(
        name="three_body",
        description="Gravitational three-body problem trajectories",
        domain="mechanics",
        features={
            "time": _T,
            "masses": _M,
            "initial_positions": _L,
            "initial_velocities": _V,
        },
        targets={
            "positions": _L,
            "velocities": _V,
        },
        tags=["gravity", "chaotic", "n-body"],
    ),
)

# Lorenz System Dataset
register_dataset(
    "lorenz",
    info=DatasetInfo(
        name="lorenz",
        description="Lorenz chaotic attractor",
        domain="fluid_dynamics",
        features={
            "time": _T,
            "sigma": _DIMLESS,
            "rho": _DIMLESS,
            "beta": _DIMLESS,
        },
        targets={
            "x": _DIMLESS,
            "y": _DIMLESS,
            "z": _DIMLESS,
        },
        tags=["chaotic", "ode", "attractor"],
    ),
)

# Wave Equation Dataset
register_dataset(
    "wave_1d",
    info=DatasetInfo(
        name="wave_1d",
        description="1D wave equation on a string",
        domain="mechanics",
        features={
            "position": _L,
            "time": _T,
            "wave_speed": _V,
        },
        targets={
            "displacement": _L,
        },
        tags=["pde", "wave", "hyperbolic"],
    ),
)

# Ideal Gas Dataset
register_dataset(
    "ideal_gas",
    info=DatasetInfo(
        name="ideal_gas",
        description="Ideal gas state variables",
        domain="thermodynamics",
        features={
            "pressure": _PRESSURE,
            "volume": Dimension(length=3),
            "moles": Dimension(amount=1),
        },
        targets={
            "temperature": _TEMP,
        },
        tags=["thermodynamics", "state"],
    ),
)


# =============================================================================
# REAL PHYSICS DATASETS WITH LOADERS
# =============================================================================


# NIST CODATA 2022 Fundamental Constants
def _load_nist_codata(**kwargs: Any) -> Any:
    """Load NIST CODATA fundamental constants."""
    from .loaders.nist import NISTCODATALoader
    loader = NISTCODATALoader()
    return loader.load(**kwargs)


register_dataset(
    "nist_codata_2022",
    info=DatasetInfo(
        name="nist_codata_2022",
        description="NIST CODATA 2022 fundamental physical constants",
        domain="constants",
        features={},
        targets={},
        source="https://physics.nist.gov/cuu/Constants/",
        license="Public Domain",
        citation="CODATA 2022",
        tags=["constants", "reference", "nist"],
    ),
    loader=_load_nist_codata,
)


# NASA Exoplanet Archive
def _load_nasa_exoplanets(**kwargs: Any) -> Any:
    """Load NASA Exoplanet Archive confirmed planets."""
    from .loaders.astronomy import NASAExoplanetLoader
    loader = NASAExoplanetLoader()
    return loader.load(**kwargs)


register_dataset(
    "nasa_exoplanets",
    info=DatasetInfo(
        name="nasa_exoplanets",
        description="NASA Exoplanet Archive: confirmed exoplanets with mass, radius, period",
        domain="astronomy",
        features={
            "pl_masse": _M,  # Planet mass (Earth masses)
            "pl_rade": _L,  # Planet radius (Earth radii)
            "pl_orbper": _T,  # Orbital period (days)
        },
        targets={},
        source="https://exoplanetarchive.ipac.caltech.edu/",
        license="Public Domain",
        citation="NASA Exoplanet Archive",
        tags=["astronomy", "exoplanets", "nasa", "real-data"],
    ),
    loader=_load_nasa_exoplanets,
)


# PRISM Climate Data
def _load_prism_climate(**kwargs: Any) -> Any:
    """Load PRISM climate data."""
    from .loaders.climate import PRISMClimateLoader
    loader = PRISMClimateLoader()
    return loader.load(**kwargs)


register_dataset(
    "prism_climate",
    info=DatasetInfo(
        name="prism_climate",
        description="PRISM climate data: temperature and precipitation time series",
        domain="climate",
        features={
            "time": _T,
        },
        targets={
            "temperature": _TEMP,
            "precipitation": _L,  # mm = length
        },
        source="https://prism.oregonstate.edu/",
        license="PRISM Climate Group",
        citation="PRISM Climate Group, Oregon State University",
        tags=["climate", "temperature", "precipitation", "real-data"],
    ),
    loader=_load_prism_climate,
)

"""Equation database for physics and engineering.

Provides a registry of physics equations with dimensional metadata
for use in dimensional analysis, validation, and physics-informed ML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.dimensions import Dimension


@dataclass
class Equation:
    """A physics equation with dimensional metadata.

    Attributes:
        name: Human-readable name (e.g., "Newton's Second Law").
        formula: Symbolic formula (e.g., "F = ma").
        variables: Dict mapping variable names to their dimensions.
        domain: Physics domain (e.g., "mechanics").
        tags: List of tags for categorization.
        description: Longer description of the equation.
        assumptions: List of assumptions/conditions.
        latex: LaTeX representation.
        related: List of related equation names.
    """

    name: str
    formula: str
    variables: dict[str, Dimension]
    domain: str = "general"
    tags: list[str] = field(default_factory=list)
    description: str = ""
    assumptions: list[str] = field(default_factory=list)
    latex: str = ""
    related: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "formula": self.formula,
            "variables": {k: str(v) for k, v in self.variables.items()},
            "domain": self.domain,
            "tags": self.tags,
            "description": self.description,
            "assumptions": self.assumptions,
            "latex": self.latex,
            "related": self.related,
        }


# Global equation registry
_EQUATIONS: dict[str, Equation] = {}


def register_equation(equation: Equation) -> None:
    """Register an equation in the database."""
    _EQUATIONS[equation.name] = equation


def get_equation(name: str) -> Equation:
    """Get an equation by name.

    Args:
        name: Equation name.

    Returns:
        Equation object.

    Raises:
        KeyError: If equation not found.
    """
    if name not in _EQUATIONS:
        raise KeyError(f"Equation '{name}' not found")
    return _EQUATIONS[name]


def get_equations(
    domain: str | None = None,
    tags: list[str] | None = None,
) -> list[Equation]:
    """Get equations, optionally filtered.

    Args:
        domain: Filter by physics domain.
        tags: Filter by tags (must have ALL tags).

    Returns:
        List of matching equations.
    """
    results = list(_EQUATIONS.values())

    if domain is not None:
        results = [eq for eq in results if eq.domain == domain]

    if tags is not None:
        results = [eq for eq in results if all(t in eq.tags for t in tags)]

    return results


def search_equations(query: str) -> list[Equation]:
    """Search equations by name, description, or variable names.

    Args:
        query: Search query (case-insensitive).

    Returns:
        List of matching equations.
    """
    query_lower = query.lower()
    results = []

    for eq in _EQUATIONS.values():
        if (
            query_lower in eq.name.lower()
            or query_lower in eq.description.lower()
            or query_lower in eq.formula.lower()
            or any(query_lower in var.lower() for var in eq.variables.keys())
            or any(query_lower in tag.lower() for tag in eq.tags)
        ):
            results.append(eq)

    return results


def list_domains() -> list[str]:
    """List all registered physics domains.

    Returns:
        Sorted list of unique domain names.
    """
    domains = set(eq.domain for eq in _EQUATIONS.values())
    return sorted(domains)


def clear_equations() -> None:
    """Clear all registered equations (for testing)."""
    _EQUATIONS.clear()


# ============================================================================
# MECHANICS EQUATIONS
# ============================================================================

# Dimensions
_L = Dimension(length=1)
_M = Dimension(mass=1)
_T = Dimension(time=1)
_V = Dimension(length=1, time=-1)
_A = Dimension(length=1, time=-2)
_F = Dimension(mass=1, length=1, time=-2)
_E = Dimension(mass=1, length=2, time=-2)
_P = Dimension(mass=1, length=1, time=-1)  # momentum
_W = Dimension(mass=1, length=2, time=-3)  # power
_DIMLESS = Dimension()

register_equation(Equation(
    name="Newton's Second Law",
    formula="F = ma",
    variables={"F": _F, "m": _M, "a": _A},
    domain="mechanics",
    tags=["newton", "force", "acceleration", "fundamental"],
    description="Force equals mass times acceleration",
    latex=r"F = ma",
    related=["Newton's First Law", "Newton's Third Law"],
))

register_equation(Equation(
    name="Kinetic Energy",
    formula="KE = (1/2)mv^2",
    variables={"KE": _E, "m": _M, "v": _V},
    domain="mechanics",
    tags=["energy", "kinetic", "motion"],
    description="Energy of motion",
    latex=r"KE = \frac{1}{2}mv^2",
    related=["Potential Energy", "Work-Energy Theorem"],
))

register_equation(Equation(
    name="Potential Energy (Gravitational)",
    formula="PE = mgh",
    variables={"PE": _E, "m": _M, "g": _A, "h": _L},
    domain="mechanics",
    tags=["energy", "potential", "gravity"],
    description="Gravitational potential energy near Earth's surface",
    latex=r"PE = mgh",
    assumptions=["Uniform gravitational field", "Near Earth's surface"],
    related=["Kinetic Energy", "Work-Energy Theorem"],
))

register_equation(Equation(
    name="Work",
    formula="W = F * d",
    variables={"W": _E, "F": _F, "d": _L},
    domain="mechanics",
    tags=["work", "energy", "force"],
    description="Work done by a constant force over a distance",
    latex=r"W = \vec{F} \cdot \vec{d}",
    assumptions=["Constant force", "Parallel to displacement"],
    related=["Power", "Kinetic Energy"],
))

register_equation(Equation(
    name="Power",
    formula="P = W/t",
    variables={"P": _W, "W": _E, "t": _T},
    domain="mechanics",
    tags=["power", "energy", "time"],
    description="Rate of doing work",
    latex=r"P = \frac{W}{t}",
    related=["Work", "Energy"],
))

register_equation(Equation(
    name="Momentum",
    formula="p = mv",
    variables={"p": _P, "m": _M, "v": _V},
    domain="mechanics",
    tags=["momentum", "motion", "mass"],
    description="Linear momentum",
    latex=r"\vec{p} = m\vec{v}",
    related=["Impulse", "Conservation of Momentum"],
))

register_equation(Equation(
    name="Impulse",
    formula="J = F * t",
    variables={"J": _P, "F": _F, "t": _T},
    domain="mechanics",
    tags=["impulse", "momentum", "force"],
    description="Change in momentum due to force over time",
    latex=r"\vec{J} = \vec{F} \Delta t",
    related=["Momentum", "Newton's Second Law"],
))

register_equation(Equation(
    name="Hooke's Law",
    formula="F = -kx",
    variables={"F": _F, "k": Dimension(mass=1, time=-2), "x": _L},
    domain="mechanics",
    tags=["spring", "elastic", "oscillation"],
    description="Force exerted by a spring",
    latex=r"F = -kx",
    assumptions=["Linear elastic region"],
    related=["Simple Harmonic Motion"],
))

register_equation(Equation(
    name="Gravitational Force",
    formula="F = G*m1*m2/r^2",
    variables={
        "F": _F,
        "G": Dimension(length=3, mass=-1, time=-2),
        "m1": _M, "m2": _M, "r": _L
    },
    domain="mechanics",
    tags=["gravity", "newton", "universal"],
    description="Newton's law of universal gravitation",
    latex=r"F = G\frac{m_1 m_2}{r^2}",
    related=["Gravitational Potential Energy"],
))

register_equation(Equation(
    name="Centripetal Acceleration",
    formula="a_c = v^2/r",
    variables={"a_c": _A, "v": _V, "r": _L},
    domain="mechanics",
    tags=["circular", "motion", "acceleration"],
    description="Acceleration toward center in circular motion",
    latex=r"a_c = \frac{v^2}{r}",
    related=["Centripetal Force"],
))

# ============================================================================
# THERMODYNAMICS EQUATIONS
# ============================================================================

_TEMP = Dimension(temperature=1)
_PRESSURE = Dimension(mass=1, length=-1, time=-2)
_VOLUME = Dimension(length=3)
_N_MOL = Dimension(amount=1)
_ENTROPY = Dimension(mass=1, length=2, time=-2, temperature=-1)
_HEAT_CAPACITY = Dimension(mass=1, length=2, time=-2, temperature=-1)

register_equation(Equation(
    name="Ideal Gas Law",
    formula="PV = nRT",
    variables={
        "P": _PRESSURE,
        "V": _VOLUME,
        "n": _N_MOL,
        "R": Dimension(mass=1, length=2, time=-2, temperature=-1, amount=-1),
        "T": _TEMP
    },
    domain="thermodynamics",
    tags=["gas", "ideal", "fundamental", "state"],
    description="Relates pressure, volume, and temperature of ideal gas",
    latex=r"PV = nRT",
    assumptions=["Ideal gas behavior", "No intermolecular forces"],
))

register_equation(Equation(
    name="First Law of Thermodynamics",
    formula="dU = Q - W",
    variables={"dU": _E, "Q": _E, "W": _E},
    domain="thermodynamics",
    tags=["energy", "conservation", "heat", "work", "fundamental"],
    description="Conservation of energy for thermodynamic systems",
    latex=r"\Delta U = Q - W",
))

register_equation(Equation(
    name="Heat Capacity",
    formula="Q = mc*dT",
    variables={"Q": _E, "m": _M, "c": _HEAT_CAPACITY / _M, "dT": _TEMP},
    domain="thermodynamics",
    tags=["heat", "temperature", "capacity"],
    description="Heat required to change temperature",
    latex=r"Q = mc\Delta T",
))

register_equation(Equation(
    name="Stefan-Boltzmann Law",
    formula="P = sigma*A*T^4",
    variables={
        "P": _W,
        "sigma": Dimension(mass=1, time=-3, temperature=-4),
        "A": Dimension(length=2),
        "T": _TEMP
    },
    domain="thermodynamics",
    tags=["radiation", "blackbody", "heat"],
    description="Power radiated by a blackbody",
    latex=r"P = \sigma A T^4",
))

register_equation(Equation(
    name="Entropy Change",
    formula="dS = dQ/T",
    variables={"dS": _ENTROPY, "dQ": _E, "T": _TEMP},
    domain="thermodynamics",
    tags=["entropy", "heat", "reversible"],
    description="Entropy change for reversible process",
    latex=r"dS = \frac{dQ}{T}",
    assumptions=["Reversible process"],
))

# ============================================================================
# ELECTROMAGNETISM EQUATIONS
# ============================================================================

_CHARGE = Dimension(current=1, time=1)
_VOLTAGE = Dimension(mass=1, length=2, time=-3, current=-1)
_CURRENT = Dimension(current=1)
_RESISTANCE = Dimension(mass=1, length=2, time=-3, current=-2)
_CAPACITANCE = Dimension(mass=-1, length=-2, time=4, current=2)
_INDUCTANCE = Dimension(mass=1, length=2, time=-2, current=-2)
_E_FIELD = Dimension(mass=1, length=1, time=-3, current=-1)
_B_FIELD = Dimension(mass=1, time=-2, current=-1)

register_equation(Equation(
    name="Ohm's Law",
    formula="V = IR",
    variables={"V": _VOLTAGE, "I": _CURRENT, "R": _RESISTANCE},
    domain="electromagnetism",
    tags=["circuits", "resistance", "current", "fundamental"],
    description="Voltage equals current times resistance",
    latex=r"V = IR",
    related=["Electric Power", "Kirchhoff's Laws"],
))

register_equation(Equation(
    name="Electric Power",
    formula="P = IV",
    variables={"P": _W, "I": _CURRENT, "V": _VOLTAGE},
    domain="electromagnetism",
    tags=["power", "circuits", "current"],
    description="Electrical power dissipation",
    latex=r"P = IV",
    related=["Ohm's Law"],
))

register_equation(Equation(
    name="Coulomb's Law",
    formula="F = k*q1*q2/r^2",
    variables={
        "F": _F,
        "k": Dimension(mass=1, length=3, time=-4, current=-2),
        "q1": _CHARGE, "q2": _CHARGE, "r": _L
    },
    domain="electromagnetism",
    tags=["electrostatic", "force", "charge"],
    description="Force between two point charges",
    latex=r"F = k\frac{q_1 q_2}{r^2}",
))

register_equation(Equation(
    name="Capacitor Energy",
    formula="E = (1/2)CV^2",
    variables={"E": _E, "C": _CAPACITANCE, "V": _VOLTAGE},
    domain="electromagnetism",
    tags=["capacitor", "energy", "circuits"],
    description="Energy stored in a capacitor",
    latex=r"E = \frac{1}{2}CV^2",
))

register_equation(Equation(
    name="Inductor Energy",
    formula="E = (1/2)LI^2",
    variables={"E": _E, "L": _INDUCTANCE, "I": _CURRENT},
    domain="electromagnetism",
    tags=["inductor", "energy", "circuits"],
    description="Energy stored in an inductor",
    latex=r"E = \frac{1}{2}LI^2",
))

register_equation(Equation(
    name="Lorentz Force",
    formula="F = q(E + v x B)",
    variables={"F": _F, "q": _CHARGE, "E": _E_FIELD, "v": _V, "B": _B_FIELD},
    domain="electromagnetism",
    tags=["force", "magnetic", "electric", "charged particle"],
    description="Force on a charged particle in electromagnetic field",
    latex=r"\vec{F} = q(\vec{E} + \vec{v} \times \vec{B})",
))

# ============================================================================
# FLUID DYNAMICS EQUATIONS
# ============================================================================

_DENSITY = Dimension(mass=1, length=-3)
_VISCOSITY = Dimension(mass=1, length=-1, time=-1)

register_equation(Equation(
    name="Bernoulli's Equation",
    formula="P + (1/2)rho*v^2 + rho*g*h = const",
    variables={
        "P": _PRESSURE,
        "rho": _DENSITY,
        "v": _V,
        "g": _A,
        "h": _L
    },
    domain="fluid_dynamics",
    tags=["fluid", "pressure", "energy", "flow"],
    description="Conservation of energy in fluid flow",
    latex=r"P + \frac{1}{2}\rho v^2 + \rho gh = \text{const}",
    assumptions=["Incompressible", "Inviscid", "Steady flow", "Along streamline"],
))

register_equation(Equation(
    name="Continuity Equation",
    formula="A1*v1 = A2*v2",
    variables={
        "A1": Dimension(length=2), "v1": _V,
        "A2": Dimension(length=2), "v2": _V
    },
    domain="fluid_dynamics",
    tags=["fluid", "mass", "conservation", "flow"],
    description="Mass conservation in fluid flow",
    latex=r"A_1 v_1 = A_2 v_2",
    assumptions=["Incompressible", "Steady flow"],
))

register_equation(Equation(
    name="Reynolds Number",
    formula="Re = rho*v*L/mu",
    variables={
        "Re": _DIMLESS,
        "rho": _DENSITY,
        "v": _V,
        "L": _L,
        "mu": _VISCOSITY
    },
    domain="fluid_dynamics",
    tags=["dimensionless", "turbulence", "laminar", "flow"],
    description="Ratio of inertial to viscous forces",
    latex=r"Re = \frac{\rho v L}{\mu}",
))

register_equation(Equation(
    name="Navier-Stokes (incompressible)",
    formula="rho(dv/dt + v.grad(v)) = -grad(P) + mu*laplacian(v) + f",
    variables={
        "rho": _DENSITY,
        "v": _V,
        "P": _PRESSURE,
        "mu": _VISCOSITY,
        "f": Dimension(mass=1, length=-2, time=-2)  # force per volume
    },
    domain="fluid_dynamics",
    tags=["pde", "momentum", "viscous", "fundamental"],
    description="Momentum equation for incompressible viscous flow",
    latex=r"\rho\left(\frac{\partial \vec{v}}{\partial t} + \vec{v} \cdot \nabla\vec{v}\right) = -\nabla P + \mu \nabla^2 \vec{v} + \vec{f}",
    assumptions=["Incompressible", "Newtonian fluid"],
))

# ============================================================================
# SPECIAL RELATIVITY
# ============================================================================

register_equation(Equation(
    name="Mass-Energy Equivalence",
    formula="E = mc^2",
    variables={
        "E": _E,
        "m": _M,
        "c": _V  # speed of light
    },
    domain="relativity",
    tags=["einstein", "energy", "mass", "fundamental"],
    description="Rest energy of a massive object",
    latex=r"E = mc^2",
))

register_equation(Equation(
    name="Lorentz Factor",
    formula="gamma = 1/sqrt(1 - v^2/c^2)",
    variables={"gamma": _DIMLESS, "v": _V, "c": _V},
    domain="relativity",
    tags=["lorentz", "time dilation", "length contraction"],
    description="Relativistic factor for time dilation and length contraction",
    latex=r"\gamma = \frac{1}{\sqrt{1 - v^2/c^2}}",
))

# ============================================================================
# QUANTUM MECHANICS
# ============================================================================

_ANGULAR_FREQ = Dimension(time=-1)
_WAVE_NUMBER = Dimension(length=-1)

register_equation(Equation(
    name="Planck-Einstein Relation",
    formula="E = hbar * omega",
    variables={
        "E": _E,
        "hbar": Dimension(mass=1, length=2, time=-1),
        "omega": _ANGULAR_FREQ
    },
    domain="quantum",
    tags=["photon", "energy", "frequency", "fundamental"],
    description="Energy of a photon",
    latex=r"E = \hbar\omega",
))

register_equation(Equation(
    name="de Broglie Wavelength",
    formula="lambda = h/p",
    variables={
        "lambda": _L,
        "h": Dimension(mass=1, length=2, time=-1),
        "p": _P
    },
    domain="quantum",
    tags=["wave-particle", "wavelength", "momentum"],
    description="Wavelength associated with a particle",
    latex=r"\lambda = \frac{h}{p}",
))

register_equation(Equation(
    name="Heisenberg Uncertainty (position-momentum)",
    formula="delta_x * delta_p >= hbar/2",
    variables={
        "delta_x": _L,
        "delta_p": _P,
        "hbar": Dimension(mass=1, length=2, time=-1)
    },
    domain="quantum",
    tags=["uncertainty", "fundamental", "measurement"],
    description="Fundamental limit on simultaneous measurement precision",
    latex=r"\Delta x \Delta p \geq \frac{\hbar}{2}",
))

register_equation(Equation(
    name="Heisenberg Uncertainty (energy-time)",
    formula="delta_E * delta_t >= hbar/2",
    variables={
        "delta_E": _E,
        "delta_t": _T,
        "hbar": Dimension(mass=1, length=2, time=-1)
    },
    domain="quantum",
    tags=["uncertainty", "fundamental", "measurement", "energy"],
    description="Uncertainty relation between energy and time",
    latex=r"\Delta E \Delta t \geq \frac{\hbar}{2}",
))

register_equation(Equation(
    name="Schrödinger Equation (time-independent)",
    formula="H*psi = E*psi",
    variables={
        "H": Dimension(mass=1, length=2, time=-2),  # Hamiltonian (energy)
        "psi": _DIMLESS,  # wavefunction (normalized)
        "E": _E
    },
    domain="quantum",
    tags=["schrodinger", "eigenvalue", "fundamental"],
    description="Time-independent Schrödinger equation for stationary states",
    latex=r"\hat{H}\psi = E\psi",
))

register_equation(Equation(
    name="Compton Scattering",
    formula="delta_lambda = (h/(m_e*c))*(1 - cos(theta))",
    variables={
        "delta_lambda": _L,
        "h": Dimension(mass=1, length=2, time=-1),
        "m_e": _M,
        "c": _V,
        "theta": _DIMLESS
    },
    domain="quantum",
    tags=["photon", "scattering", "wavelength"],
    description="Wavelength shift in photon-electron scattering",
    latex=r"\Delta\lambda = \frac{h}{m_e c}(1 - \cos\theta)",
))

register_equation(Equation(
    name="Bohr Radius",
    formula="a_0 = hbar^2/(m_e*e^2*k_e)",
    variables={
        "a_0": _L,
        "hbar": Dimension(mass=1, length=2, time=-1),
        "m_e": _M,
        "e": _CHARGE,
        "k_e": Dimension(mass=1, length=3, time=-4, current=-2)
    },
    domain="quantum",
    tags=["atomic", "hydrogen", "fundamental"],
    description="Radius of lowest-energy electron orbit in hydrogen",
    latex=r"a_0 = \frac{\hbar^2}{m_e e^2 k_e}",
))

register_equation(Equation(
    name="Rydberg Formula",
    formula="1/lambda = R*(1/n1^2 - 1/n2^2)",
    variables={
        "lambda": _L,
        "R": _WAVE_NUMBER,
        "n1": _DIMLESS,
        "n2": _DIMLESS
    },
    domain="quantum",
    tags=["atomic", "spectroscopy", "hydrogen"],
    description="Wavelengths of spectral lines in hydrogen",
    latex=r"\frac{1}{\lambda} = R\left(\frac{1}{n_1^2} - \frac{1}{n_2^2}\right)",
))

register_equation(Equation(
    name="Quantum Harmonic Oscillator Energy",
    formula="E_n = hbar*omega*(n + 1/2)",
    variables={
        "E_n": _E,
        "hbar": Dimension(mass=1, length=2, time=-1),
        "omega": _ANGULAR_FREQ,
        "n": _DIMLESS
    },
    domain="quantum",
    tags=["oscillator", "energy", "quantization"],
    description="Energy levels of quantum harmonic oscillator",
    latex=r"E_n = \hbar\omega\left(n + \frac{1}{2}\right)",
))

# ============================================================================
# RELATIVITY (EXPANDED)
# ============================================================================

register_equation(Equation(
    name="Time Dilation",
    formula="delta_t = gamma * delta_t0",
    variables={
        "delta_t": _T,
        "gamma": _DIMLESS,
        "delta_t0": _T
    },
    domain="relativity",
    tags=["time", "lorentz", "special relativity"],
    description="Time interval in moving frame appears longer",
    latex=r"\Delta t = \gamma \Delta t_0",
    related=["Lorentz Factor"],
))

register_equation(Equation(
    name="Length Contraction",
    formula="L = L0/gamma",
    variables={
        "L": _L,
        "L0": _L,
        "gamma": _DIMLESS
    },
    domain="relativity",
    tags=["length", "lorentz", "special relativity"],
    description="Length in direction of motion appears contracted",
    latex=r"L = \frac{L_0}{\gamma}",
    related=["Lorentz Factor"],
))

register_equation(Equation(
    name="Relativistic Momentum",
    formula="p = gamma*m*v",
    variables={
        "p": _P,
        "gamma": _DIMLESS,
        "m": _M,
        "v": _V
    },
    domain="relativity",
    tags=["momentum", "special relativity"],
    description="Momentum of a relativistic particle",
    latex=r"\vec{p} = \gamma m\vec{v}",
    related=["Lorentz Factor"],
))

register_equation(Equation(
    name="Relativistic Energy",
    formula="E = gamma*m*c^2",
    variables={
        "E": _E,
        "gamma": _DIMLESS,
        "m": _M,
        "c": _V
    },
    domain="relativity",
    tags=["energy", "special relativity"],
    description="Total energy of a relativistic particle",
    latex=r"E = \gamma mc^2",
    related=["Mass-Energy Equivalence", "Lorentz Factor"],
))

register_equation(Equation(
    name="Energy-Momentum Relation",
    formula="E^2 = (pc)^2 + (mc^2)^2",
    variables={
        "E": _E,
        "p": _P,
        "m": _M,
        "c": _V
    },
    domain="relativity",
    tags=["energy", "momentum", "special relativity"],
    description="Relation between energy and momentum in special relativity",
    latex=r"E^2 = (pc)^2 + (mc^2)^2",
    related=["Mass-Energy Equivalence"],
))

register_equation(Equation(
    name="Doppler Effect (relativistic)",
    formula="f = f0*sqrt((1-beta)/(1+beta))",
    variables={
        "f": Dimension(time=-1),
        "f0": Dimension(time=-1),
        "beta": _DIMLESS  # v/c
    },
    domain="relativity",
    tags=["doppler", "frequency", "special relativity"],
    description="Frequency shift for source moving radially",
    latex=r"f = f_0\sqrt{\frac{1-\beta}{1+\beta}}",
))

register_equation(Equation(
    name="Schwarzschild Radius",
    formula="r_s = 2*G*M/c^2",
    variables={
        "r_s": _L,
        "G": Dimension(length=3, mass=-1, time=-2),
        "M": _M,
        "c": _V
    },
    domain="relativity",
    tags=["black hole", "gravity", "general relativity"],
    description="Radius of the event horizon of a non-rotating black hole",
    latex=r"r_s = \frac{2GM}{c^2}",
))

# ============================================================================
# OPTICS
# ============================================================================

_FREQUENCY = Dimension(time=-1)
_REFRACTIVE_INDEX = _DIMLESS
_FOCAL_LENGTH = _L
_ANGLE = _DIMLESS

register_equation(Equation(
    name="Snell's Law",
    formula="n1*sin(theta1) = n2*sin(theta2)",
    variables={
        "n1": _REFRACTIVE_INDEX,
        "theta1": _ANGLE,
        "n2": _REFRACTIVE_INDEX,
        "theta2": _ANGLE
    },
    domain="optics",
    tags=["refraction", "fundamental", "wave"],
    description="Law of refraction at interface between media",
    latex=r"n_1\sin\theta_1 = n_2\sin\theta_2",
))

register_equation(Equation(
    name="Thin Lens Equation",
    formula="1/f = 1/d_o + 1/d_i",
    variables={
        "f": _FOCAL_LENGTH,
        "d_o": _L,  # object distance
        "d_i": _L   # image distance
    },
    domain="optics",
    tags=["lens", "imaging", "geometric optics"],
    description="Relates object and image distances to focal length",
    latex=r"\frac{1}{f} = \frac{1}{d_o} + \frac{1}{d_i}",
))

register_equation(Equation(
    name="Lens Maker's Equation",
    formula="1/f = (n-1)*(1/R1 - 1/R2)",
    variables={
        "f": _FOCAL_LENGTH,
        "n": _REFRACTIVE_INDEX,
        "R1": _L,  # radius of curvature 1
        "R2": _L   # radius of curvature 2
    },
    domain="optics",
    tags=["lens", "fabrication", "geometric optics"],
    description="Focal length from lens geometry and material",
    latex=r"\frac{1}{f} = (n-1)\left(\frac{1}{R_1} - \frac{1}{R_2}\right)",
))

register_equation(Equation(
    name="Magnification",
    formula="M = -d_i/d_o",
    variables={
        "M": _DIMLESS,
        "d_i": _L,
        "d_o": _L
    },
    domain="optics",
    tags=["lens", "imaging", "magnification"],
    description="Linear magnification of optical system",
    latex=r"M = -\frac{d_i}{d_o}",
))

register_equation(Equation(
    name="Diffraction Grating",
    formula="d*sin(theta) = m*lambda",
    variables={
        "d": _L,  # grating spacing
        "theta": _ANGLE,
        "m": _DIMLESS,  # order
        "lambda": _L
    },
    domain="optics",
    tags=["diffraction", "interference", "wave"],
    description="Condition for constructive interference in diffraction grating",
    latex=r"d\sin\theta = m\lambda",
))

register_equation(Equation(
    name="Rayleigh Criterion",
    formula="theta_min = 1.22*lambda/D",
    variables={
        "theta_min": _ANGLE,
        "lambda": _L,
        "D": _L  # aperture diameter
    },
    domain="optics",
    tags=["resolution", "diffraction", "imaging"],
    description="Minimum resolvable angle for circular aperture",
    latex=r"\theta_{\text{min}} = 1.22\frac{\lambda}{D}",
))

register_equation(Equation(
    name="Brewster's Angle",
    formula="tan(theta_B) = n2/n1",
    variables={
        "theta_B": _ANGLE,
        "n1": _REFRACTIVE_INDEX,
        "n2": _REFRACTIVE_INDEX
    },
    domain="optics",
    tags=["polarization", "reflection", "refraction"],
    description="Angle of incidence for complete p-polarization",
    latex=r"\tan\theta_B = \frac{n_2}{n_1}",
))

register_equation(Equation(
    name="Critical Angle",
    formula="sin(theta_c) = n2/n1",
    variables={
        "theta_c": _ANGLE,
        "n1": _REFRACTIVE_INDEX,
        "n2": _REFRACTIVE_INDEX
    },
    domain="optics",
    tags=["total internal reflection", "refraction"],
    description="Angle for total internal reflection (n1 > n2)",
    latex=r"\sin\theta_c = \frac{n_2}{n_1}",
    assumptions=["n1 > n2"],
))

register_equation(Equation(
    name="Malus's Law",
    formula="I = I0*cos^2(theta)",
    variables={
        "I": Dimension(mass=1, time=-3),  # intensity
        "I0": Dimension(mass=1, time=-3),
        "theta": _ANGLE
    },
    domain="optics",
    tags=["polarization", "intensity"],
    description="Intensity of light through polarizer",
    latex=r"I = I_0\cos^2\theta",
))

register_equation(Equation(
    name="Wave Equation (electromagnetic)",
    formula="c = lambda*f",
    variables={
        "c": _V,
        "lambda": _L,
        "f": _FREQUENCY
    },
    domain="optics",
    tags=["wave", "fundamental", "electromagnetic"],
    description="Relates wavelength, frequency, and speed of light",
    latex=r"c = \lambda f",
))

# ============================================================================
# ACOUSTICS
# ============================================================================

_SOUND_INTENSITY = Dimension(mass=1, time=-3)  # W/m²
_SOUND_PRESSURE = Dimension(mass=1, length=-1, time=-2)  # Pa
_BULK_MODULUS = Dimension(mass=1, length=-1, time=-2)

register_equation(Equation(
    name="Speed of Sound in Fluid",
    formula="v = sqrt(B/rho)",
    variables={
        "v": _V,
        "B": _BULK_MODULUS,
        "rho": _DENSITY
    },
    domain="acoustics",
    tags=["wave", "speed", "fluid"],
    description="Speed of sound in terms of bulk modulus and density",
    latex=r"v = \sqrt{\frac{B}{\rho}}",
))

register_equation(Equation(
    name="Speed of Sound in Ideal Gas",
    formula="v = sqrt(gamma*R*T/M)",
    variables={
        "v": _V,
        "gamma": _DIMLESS,  # heat capacity ratio
        "R": Dimension(mass=1, length=2, time=-2, temperature=-1, amount=-1),
        "T": _TEMP,
        "M": Dimension(mass=1, amount=-1)  # molar mass
    },
    domain="acoustics",
    tags=["wave", "speed", "gas"],
    description="Speed of sound in ideal gas",
    latex=r"v = \sqrt{\frac{\gamma RT}{M}}",
))

register_equation(Equation(
    name="Sound Intensity",
    formula="I = P^2/(2*rho*v)",
    variables={
        "I": _SOUND_INTENSITY,
        "P": _SOUND_PRESSURE,
        "rho": _DENSITY,
        "v": _V
    },
    domain="acoustics",
    tags=["intensity", "pressure", "wave"],
    description="Sound intensity from pressure amplitude",
    latex=r"I = \frac{P^2}{2\rho v}",
))

register_equation(Equation(
    name="Sound Intensity Level",
    formula="L = 10*log10(I/I0)",
    variables={
        "L": _DIMLESS,  # decibels
        "I": _SOUND_INTENSITY,
        "I0": _SOUND_INTENSITY  # reference intensity
    },
    domain="acoustics",
    tags=["intensity", "decibel", "logarithmic"],
    description="Sound level in decibels",
    latex=r"L = 10\log_{10}\left(\frac{I}{I_0}\right)",
))

register_equation(Equation(
    name="Doppler Effect (sound)",
    formula="f = f0*(v + v_r)/(v - v_s)",
    variables={
        "f": _FREQUENCY,
        "f0": _FREQUENCY,
        "v": _V,  # speed of sound
        "v_r": _V,  # receiver velocity
        "v_s": _V   # source velocity
    },
    domain="acoustics",
    tags=["doppler", "frequency", "wave"],
    description="Frequency shift due to relative motion in sound",
    latex=r"f = f_0\frac{v + v_r}{v - v_s}",
))

register_equation(Equation(
    name="Acoustic Impedance",
    formula="Z = rho*v",
    variables={
        "Z": Dimension(mass=1, length=-2, time=-1),
        "rho": _DENSITY,
        "v": _V
    },
    domain="acoustics",
    tags=["impedance", "wave", "material"],
    description="Characteristic acoustic impedance of medium",
    latex=r"Z = \rho v",
))

register_equation(Equation(
    name="Wave Equation (1D)",
    formula="d^2y/dt^2 = v^2*d^2y/dx^2",
    variables={
        "y": _L,  # displacement
        "t": _T,
        "v": _V,
        "x": _L
    },
    domain="acoustics",
    tags=["wave", "pde", "fundamental"],
    description="One-dimensional wave equation",
    latex=r"\frac{\partial^2 y}{\partial t^2} = v^2\frac{\partial^2 y}{\partial x^2}",
))

register_equation(Equation(
    name="Standing Wave Frequency (string)",
    formula="f_n = n*v/(2*L)",
    variables={
        "f_n": _FREQUENCY,
        "n": _DIMLESS,  # harmonic number
        "v": _V,
        "L": _L  # string length
    },
    domain="acoustics",
    tags=["standing wave", "resonance", "string"],
    description="Natural frequencies of vibrating string",
    latex=r"f_n = \frac{nv}{2L}",
    assumptions=["Fixed ends"],
))

register_equation(Equation(
    name="Acoustic Power",
    formula="P = I*A",
    variables={
        "P": _W,
        "I": _SOUND_INTENSITY,
        "A": Dimension(length=2)
    },
    domain="acoustics",
    tags=["power", "intensity"],
    description="Total acoustic power through area",
    latex=r"P = IA",
))

# ============================================================================
# FLUID DYNAMICS (EXPANDED)
# ============================================================================

register_equation(Equation(
    name="Stokes' Law",
    formula="F_d = 6*pi*mu*r*v",
    variables={
        "F_d": _F,  # drag force
        "mu": _VISCOSITY,
        "r": _L,  # sphere radius
        "v": _V
    },
    domain="fluid_dynamics",
    tags=["drag", "viscous", "sphere"],
    description="Drag force on a sphere in viscous fluid (low Re)",
    latex=r"F_d = 6\pi\mu rv",
    assumptions=["Low Reynolds number", "Spherical object"],
))

register_equation(Equation(
    name="Poiseuille's Law",
    formula="Q = pi*r^4*delta_P/(8*mu*L)",
    variables={
        "Q": Dimension(length=3, time=-1),  # volume flow rate
        "r": _L,  # pipe radius
        "delta_P": _PRESSURE,
        "mu": _VISCOSITY,
        "L": _L  # pipe length
    },
    domain="fluid_dynamics",
    tags=["flow", "viscous", "pipe"],
    description="Volume flow rate in cylindrical pipe",
    latex=r"Q = \frac{\pi r^4 \Delta P}{8\mu L}",
    assumptions=["Laminar flow", "Newtonian fluid"],
))

register_equation(Equation(
    name="Drag Equation",
    formula="F_d = (1/2)*rho*v^2*C_d*A",
    variables={
        "F_d": _F,
        "rho": _DENSITY,
        "v": _V,
        "C_d": _DIMLESS,  # drag coefficient
        "A": Dimension(length=2)  # reference area
    },
    domain="fluid_dynamics",
    tags=["drag", "turbulent", "aerodynamics"],
    description="Drag force on object in fluid flow",
    latex=r"F_d = \frac{1}{2}\rho v^2 C_d A",
))

register_equation(Equation(
    name="Froude Number",
    formula="Fr = v/sqrt(g*L)",
    variables={
        "Fr": _DIMLESS,
        "v": _V,
        "g": _A,
        "L": _L  # characteristic length
    },
    domain="fluid_dynamics",
    tags=["dimensionless", "wave", "gravity"],
    description="Ratio of inertial to gravitational forces",
    latex=r"Fr = \frac{v}{\sqrt{gL}}",
))

register_equation(Equation(
    name="Mach Number",
    formula="M = v/c",
    variables={
        "M": _DIMLESS,
        "v": _V,  # flow velocity
        "c": _V   # speed of sound
    },
    domain="fluid_dynamics",
    tags=["dimensionless", "compressible", "supersonic"],
    description="Ratio of flow velocity to speed of sound",
    latex=r"M = \frac{v}{c}",
))

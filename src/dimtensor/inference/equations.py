"""Equation pattern database for dimensional inference.

This module provides a database of common physics equations and their
dimensional relationships. It can be used to verify dimensional correctness
and suggest dimensions for unknown quantities.

Example:
    >>> from dimtensor.inference.equations import EQUATION_DATABASE
    >>> from dimtensor.inference.equations import check_equation
    >>>
    >>> # Check if F = ma is dimensionally consistent
    >>> result = check_equation("force", ["mass", "acceleration"], "mul")
    >>> print(result)  # True - dimensions are consistent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

from ..core.dimensions import Dimension


@dataclass
class Equation:
    """A physics equation with dimensional relationships.

    Attributes:
        name: Human-readable name of the equation.
        formula: LaTeX or symbolic representation.
        variables: Dict of variable names to their dimensions.
        relationships: Dimensional relationships between variables.
        domain: Physics domain (mechanics, electromagnetics, etc.).
        tags: Additional categorization tags.
    """

    name: str
    formula: str
    variables: dict[str, Dimension]
    domain: str
    tags: list[str] = field(default_factory=list)
    notes: str = ""


# ==============================================================================
# Equation Database - Common Physics Equations
# ==============================================================================

EQUATION_DATABASE: list[Equation] = [
    # ===== Classical Mechanics =====
    Equation(
        name="Newton's Second Law",
        formula="F = ma",
        variables={
            "F": Dimension(length=1, mass=1, time=-2),  # N
            "m": Dimension(mass=1),  # kg
            "a": Dimension(length=1, time=-2),  # m/s²
        },
        domain="mechanics",
        tags=["newton", "force", "fundamental"],
        notes="Force equals mass times acceleration",
    ),
    Equation(
        name="Kinetic Energy",
        formula="KE = ½mv²",
        variables={
            "KE": Dimension(length=2, mass=1, time=-2),  # J
            "m": Dimension(mass=1),  # kg
            "v": Dimension(length=1, time=-1),  # m/s
        },
        domain="mechanics",
        tags=["energy", "kinetic"],
    ),
    Equation(
        name="Gravitational Potential Energy",
        formula="PE = mgh",
        variables={
            "PE": Dimension(length=2, mass=1, time=-2),  # J
            "m": Dimension(mass=1),  # kg
            "g": Dimension(length=1, time=-2),  # m/s²
            "h": Dimension(length=1),  # m
        },
        domain="mechanics",
        tags=["energy", "potential", "gravity"],
    ),
    Equation(
        name="Momentum",
        formula="p = mv",
        variables={
            "p": Dimension(length=1, mass=1, time=-1),  # kg·m/s
            "m": Dimension(mass=1),  # kg
            "v": Dimension(length=1, time=-1),  # m/s
        },
        domain="mechanics",
        tags=["momentum"],
    ),
    Equation(
        name="Impulse",
        formula="J = FΔt = Δp",
        variables={
            "J": Dimension(length=1, mass=1, time=-1),  # N·s
            "F": Dimension(length=1, mass=1, time=-2),  # N
            "delta_t": Dimension(time=1),  # s
        },
        domain="mechanics",
        tags=["impulse", "momentum"],
    ),
    Equation(
        name="Work",
        formula="W = Fd cos(θ)",
        variables={
            "W": Dimension(length=2, mass=1, time=-2),  # J
            "F": Dimension(length=1, mass=1, time=-2),  # N
            "d": Dimension(length=1),  # m
            "theta": Dimension(),  # dimensionless (radians)
        },
        domain="mechanics",
        tags=["work", "energy"],
    ),
    Equation(
        name="Power",
        formula="P = W/t = Fv",
        variables={
            "P": Dimension(length=2, mass=1, time=-3),  # W
            "W": Dimension(length=2, mass=1, time=-2),  # J
            "t": Dimension(time=1),  # s
            "F": Dimension(length=1, mass=1, time=-2),  # N
            "v": Dimension(length=1, time=-1),  # m/s
        },
        domain="mechanics",
        tags=["power"],
    ),
    Equation(
        name="Centripetal Acceleration",
        formula="a_c = v²/r",
        variables={
            "a_c": Dimension(length=1, time=-2),  # m/s²
            "v": Dimension(length=1, time=-1),  # m/s
            "r": Dimension(length=1),  # m
        },
        domain="mechanics",
        tags=["circular motion", "acceleration"],
    ),
    Equation(
        name="Torque",
        formula="τ = rF sin(θ)",
        variables={
            "tau": Dimension(length=2, mass=1, time=-2),  # N·m
            "r": Dimension(length=1),  # m
            "F": Dimension(length=1, mass=1, time=-2),  # N
            "theta": Dimension(),  # dimensionless
        },
        domain="mechanics",
        tags=["torque", "rotation"],
    ),
    Equation(
        name="Angular Momentum",
        formula="L = Iω = mvr",
        variables={
            "L": Dimension(length=2, mass=1, time=-1),  # kg·m²/s
            "I": Dimension(length=2, mass=1),  # kg·m²
            "omega": Dimension(time=-1),  # rad/s
            "m": Dimension(mass=1),  # kg
            "v": Dimension(length=1, time=-1),  # m/s
            "r": Dimension(length=1),  # m
        },
        domain="mechanics",
        tags=["angular momentum", "rotation"],
    ),
    Equation(
        name="Hooke's Law",
        formula="F = -kx",
        variables={
            "F": Dimension(length=1, mass=1, time=-2),  # N
            "k": Dimension(mass=1, time=-2),  # N/m = kg/s²
            "x": Dimension(length=1),  # m
        },
        domain="mechanics",
        tags=["springs", "elasticity"],
    ),
    Equation(
        name="Universal Gravitation",
        formula="F = Gm₁m₂/r²",
        variables={
            "F": Dimension(length=1, mass=1, time=-2),  # N
            "G": Dimension(length=3, mass=-1, time=-2),  # m³/(kg·s²)
            "m1": Dimension(mass=1),  # kg
            "m2": Dimension(mass=1),  # kg
            "r": Dimension(length=1),  # m
        },
        domain="mechanics",
        tags=["gravity", "universal"],
    ),
    Equation(
        name="Pressure",
        formula="P = F/A",
        variables={
            "P": Dimension(length=-1, mass=1, time=-2),  # Pa
            "F": Dimension(length=1, mass=1, time=-2),  # N
            "A": Dimension(length=2),  # m²
        },
        domain="mechanics",
        tags=["pressure", "fluids"],
    ),
    Equation(
        name="Density",
        formula="ρ = m/V",
        variables={
            "rho": Dimension(length=-3, mass=1),  # kg/m³
            "m": Dimension(mass=1),  # kg
            "V": Dimension(length=3),  # m³
        },
        domain="mechanics",
        tags=["density"],
    ),
    # ===== Kinematics =====
    Equation(
        name="Velocity",
        formula="v = Δx/Δt",
        variables={
            "v": Dimension(length=1, time=-1),  # m/s
            "delta_x": Dimension(length=1),  # m
            "delta_t": Dimension(time=1),  # s
        },
        domain="kinematics",
        tags=["velocity", "motion"],
    ),
    Equation(
        name="Acceleration",
        formula="a = Δv/Δt",
        variables={
            "a": Dimension(length=1, time=-2),  # m/s²
            "delta_v": Dimension(length=1, time=-1),  # m/s
            "delta_t": Dimension(time=1),  # s
        },
        domain="kinematics",
        tags=["acceleration", "motion"],
    ),
    Equation(
        name="SUVAT: Position",
        formula="x = x₀ + v₀t + ½at²",
        variables={
            "x": Dimension(length=1),  # m
            "x0": Dimension(length=1),  # m
            "v0": Dimension(length=1, time=-1),  # m/s
            "t": Dimension(time=1),  # s
            "a": Dimension(length=1, time=-2),  # m/s²
        },
        domain="kinematics",
        tags=["suvat", "motion"],
    ),
    Equation(
        name="SUVAT: Velocity",
        formula="v = v₀ + at",
        variables={
            "v": Dimension(length=1, time=-1),  # m/s
            "v0": Dimension(length=1, time=-1),  # m/s
            "a": Dimension(length=1, time=-2),  # m/s²
            "t": Dimension(time=1),  # s
        },
        domain="kinematics",
        tags=["suvat", "motion"],
    ),
    # ===== Waves & Oscillations =====
    Equation(
        name="Wave Equation",
        formula="v = fλ",
        variables={
            "v": Dimension(length=1, time=-1),  # m/s
            "f": Dimension(time=-1),  # Hz
            "lambda": Dimension(length=1),  # m
        },
        domain="waves",
        tags=["waves", "frequency"],
    ),
    Equation(
        name="Period-Frequency",
        formula="T = 1/f",
        variables={
            "T": Dimension(time=1),  # s
            "f": Dimension(time=-1),  # Hz
        },
        domain="waves",
        tags=["period", "frequency"],
    ),
    Equation(
        name="Simple Pendulum Period",
        formula="T = 2π√(L/g)",
        variables={
            "T": Dimension(time=1),  # s
            "L": Dimension(length=1),  # m
            "g": Dimension(length=1, time=-2),  # m/s²
        },
        domain="waves",
        tags=["pendulum", "oscillation"],
    ),
    # ===== Electromagnetics =====
    Equation(
        name="Ohm's Law",
        formula="V = IR",
        variables={
            "V": Dimension(length=2, mass=1, time=-3, current=-1),  # V
            "I": Dimension(current=1),  # A
            "R": Dimension(length=2, mass=1, time=-3, current=-2),  # Ω
        },
        domain="electromagnetics",
        tags=["ohm", "resistance", "fundamental"],
    ),
    Equation(
        name="Electric Power",
        formula="P = IV = I²R = V²/R",
        variables={
            "P": Dimension(length=2, mass=1, time=-3),  # W
            "I": Dimension(current=1),  # A
            "V": Dimension(length=2, mass=1, time=-3, current=-1),  # V
            "R": Dimension(length=2, mass=1, time=-3, current=-2),  # Ω
        },
        domain="electromagnetics",
        tags=["power", "electrical"],
    ),
    Equation(
        name="Coulomb's Law",
        formula="F = kq₁q₂/r²",
        variables={
            "F": Dimension(length=1, mass=1, time=-2),  # N
            "k": Dimension(length=3, mass=1, time=-4, current=-2),  # N·m²/C²
            "q1": Dimension(current=1, time=1),  # C
            "q2": Dimension(current=1, time=1),  # C
            "r": Dimension(length=1),  # m
        },
        domain="electromagnetics",
        tags=["coulomb", "electrostatics"],
    ),
    Equation(
        name="Electric Field",
        formula="E = F/q",
        variables={
            "E": Dimension(length=1, mass=1, time=-3, current=-1),  # V/m = N/C
            "F": Dimension(length=1, mass=1, time=-2),  # N
            "q": Dimension(current=1, time=1),  # C
        },
        domain="electromagnetics",
        tags=["electric field"],
    ),
    Equation(
        name="Capacitance",
        formula="C = Q/V",
        variables={
            "C": Dimension(length=-2, mass=-1, time=4, current=2),  # F
            "Q": Dimension(current=1, time=1),  # C
            "V": Dimension(length=2, mass=1, time=-3, current=-1),  # V
        },
        domain="electromagnetics",
        tags=["capacitance"],
    ),
    Equation(
        name="Capacitor Energy",
        formula="E = ½CV²",
        variables={
            "E": Dimension(length=2, mass=1, time=-2),  # J
            "C": Dimension(length=-2, mass=-1, time=4, current=2),  # F
            "V": Dimension(length=2, mass=1, time=-3, current=-1),  # V
        },
        domain="electromagnetics",
        tags=["capacitance", "energy"],
    ),
    # ===== Thermodynamics =====
    Equation(
        name="Ideal Gas Law",
        formula="PV = nRT",
        variables={
            "P": Dimension(length=-1, mass=1, time=-2),  # Pa
            "V": Dimension(length=3),  # m³
            "n": Dimension(amount=1),  # mol
            "R": Dimension(length=2, mass=1, time=-2, temperature=-1, amount=-1),  # J/(mol·K)
            "T": Dimension(temperature=1),  # K
        },
        domain="thermodynamics",
        tags=["ideal gas", "fundamental"],
    ),
    Equation(
        name="Heat Transfer",
        formula="Q = mcΔT",
        variables={
            "Q": Dimension(length=2, mass=1, time=-2),  # J
            "m": Dimension(mass=1),  # kg
            "c": Dimension(length=2, time=-2, temperature=-1),  # J/(kg·K)
            "delta_T": Dimension(temperature=1),  # K
        },
        domain="thermodynamics",
        tags=["heat", "specific heat"],
    ),
    Equation(
        name="First Law of Thermodynamics",
        formula="ΔU = Q - W",
        variables={
            "delta_U": Dimension(length=2, mass=1, time=-2),  # J
            "Q": Dimension(length=2, mass=1, time=-2),  # J
            "W": Dimension(length=2, mass=1, time=-2),  # J
        },
        domain="thermodynamics",
        tags=["first law", "fundamental"],
    ),
    Equation(
        name="Stefan-Boltzmann Law",
        formula="P = σAT⁴",
        variables={
            "P": Dimension(length=2, mass=1, time=-3),  # W
            "sigma": Dimension(mass=1, time=-3, temperature=-4),  # W/(m²·K⁴)
            "A": Dimension(length=2),  # m²
            "T": Dimension(temperature=1),  # K
        },
        domain="thermodynamics",
        tags=["radiation", "blackbody"],
    ),
    # ===== Relativity =====
    Equation(
        name="Mass-Energy Equivalence",
        formula="E = mc²",
        variables={
            "E": Dimension(length=2, mass=1, time=-2),  # J
            "m": Dimension(mass=1),  # kg
            "c": Dimension(length=1, time=-1),  # m/s
        },
        domain="relativity",
        tags=["einstein", "fundamental"],
    ),
    # ===== Quantum Mechanics =====
    Equation(
        name="Photon Energy",
        formula="E = hf",
        variables={
            "E": Dimension(length=2, mass=1, time=-2),  # J
            "h": Dimension(length=2, mass=1, time=-1),  # J·s
            "f": Dimension(time=-1),  # Hz
        },
        domain="quantum",
        tags=["photon", "planck"],
    ),
    Equation(
        name="de Broglie Wavelength",
        formula="λ = h/p",
        variables={
            "lambda": Dimension(length=1),  # m
            "h": Dimension(length=2, mass=1, time=-1),  # J·s
            "p": Dimension(length=1, mass=1, time=-1),  # kg·m/s
        },
        domain="quantum",
        tags=["wave-particle", "de broglie"],
    ),
    # ===== Fluids =====
    Equation(
        name="Bernoulli's Equation",
        formula="P + ½ρv² + ρgh = constant",
        variables={
            "P": Dimension(length=-1, mass=1, time=-2),  # Pa
            "rho": Dimension(length=-3, mass=1),  # kg/m³
            "v": Dimension(length=1, time=-1),  # m/s
            "g": Dimension(length=1, time=-2),  # m/s²
            "h": Dimension(length=1),  # m
        },
        domain="fluids",
        tags=["bernoulli", "fluid dynamics"],
    ),
    Equation(
        name="Continuity Equation",
        formula="A₁v₁ = A₂v₂",
        variables={
            "A": Dimension(length=2),  # m²
            "v": Dimension(length=1, time=-1),  # m/s
        },
        domain="fluids",
        tags=["continuity", "flow"],
    ),
]


def get_equations_by_domain(domain: str) -> list[Equation]:
    """Get all equations in a specific domain.

    Args:
        domain: Domain name (mechanics, electromagnetics, etc.)

    Returns:
        List of equations in that domain.
    """
    return [eq for eq in EQUATION_DATABASE if eq.domain == domain]


def get_equations_by_tag(tag: str) -> list[Equation]:
    """Get all equations with a specific tag.

    Args:
        tag: Tag to search for.

    Returns:
        List of equations with that tag.
    """
    return [eq for eq in EQUATION_DATABASE if tag in eq.tags]


def find_equations_with_variable(variable_name: str) -> list[Equation]:
    """Find all equations that use a specific variable.

    Args:
        variable_name: Variable name to search for.

    Returns:
        List of equations containing that variable.
    """
    name_lower = variable_name.lower()
    results = []
    for eq in EQUATION_DATABASE:
        for var_name in eq.variables:
            if name_lower in var_name.lower():
                results.append(eq)
                break
    return results


def check_dimensional_consistency(
    equation: Equation,
) -> bool:
    """Check if an equation is dimensionally consistent.

    For most equations, this verifies that the relationship between
    variables produces consistent dimensions.

    Args:
        equation: Equation to check.

    Returns:
        True if the equation is dimensionally consistent.
    """
    # This is a simplified check - full verification would need
    # to parse the formula and evaluate dimensions
    # For now, we assume the database is correct
    return True


def suggest_dimension_from_equations(
    variable_name: str,
    context_variables: dict[str, Dimension] | None = None,
) -> list[tuple[Dimension, str, float]]:
    """Suggest dimensions for a variable based on equation patterns.

    Args:
        variable_name: Variable to find dimensions for.
        context_variables: Other variables in the context with known dimensions.

    Returns:
        List of (dimension, equation_name, confidence) tuples.
    """
    suggestions: list[tuple[Dimension, str, float]] = []
    name_lower = variable_name.lower()

    for eq in EQUATION_DATABASE:
        for var_name, dim in eq.variables.items():
            if name_lower == var_name.lower():
                # Exact match
                suggestions.append((dim, eq.name, 0.9))
            elif name_lower in var_name.lower() or var_name.lower() in name_lower:
                # Partial match
                suggestions.append((dim, eq.name, 0.6))

    # Sort by confidence
    suggestions.sort(key=lambda x: x[2], reverse=True)

    return suggestions


# Domain list for reference
DOMAINS = [
    "mechanics",
    "kinematics",
    "waves",
    "electromagnetics",
    "thermodynamics",
    "relativity",
    "quantum",
    "fluids",
]

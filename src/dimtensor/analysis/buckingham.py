"""Buckingham Pi theorem solver for dimensional analysis.

This module implements the Buckingham Pi theorem, which states that a physical
problem with n variables and k independent base dimensions can be reduced to
n-k dimensionless Pi groups.

The solver uses singular value decomposition (SVD) on the dimensional matrix
to find the nullspace, which corresponds to dimensionless combinations of variables.

Example:
    >>> from dimtensor.analysis import buckingham_pi
    >>> from dimtensor.core.units import m, s, kg
    >>>
    >>> # Fluid mechanics: drag force problem
    >>> variables = {
    ...     'F': kg * m / s**2,      # Force
    ...     'v': m / s,               # Velocity
    ...     'L': m,                   # Length
    ...     'rho': kg / m**3,         # Density
    ...     'mu': kg / (m * s),       # Dynamic viscosity
    ... }
    >>> result = buckingham_pi(variables)
    >>> for pi in result['pi_groups']:
    ...     print(f"{pi.name} = {pi.expression}")
    # Π₁ = ρvL/μ (Reynolds number)
    # Π₂ = F/(ρv²L²) (Drag coefficient)
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.dimensions import (
    Dimension,
    DIMENSIONLESS,
    _DIMENSION_SYMBOLS,
    LENGTH,
    MASS,
    TIME,
    CURRENT,
    TEMPERATURE,
    AMOUNT,
    LUMINOSITY,
)
from ..core.units import Unit
from ..core.dimarray import DimArray


@dataclass(frozen=True)
class PiGroup:
    """Represents a dimensionless Pi group.

    A Pi group is a dimensionless combination of physical variables,
    expressed as a product of powers: Π = v₁^e₁ · v₂^e₂ · ... · vₙ^eₙ

    Attributes:
        name: Name of the Pi group (e.g., 'Π₁', 'Re' for Reynolds number)
        exponents: Dictionary mapping variable names to their exponents (as Fractions)
        expression: Human-readable string like 'ρvL/μ'
        latex: LaTeX representation for pretty printing
        interpretation: Optional physical interpretation (e.g., 'inertial/viscous forces')
    """

    name: str
    exponents: dict[str, Fraction]
    expression: str
    latex: str
    interpretation: str | None = None

    def __str__(self) -> str:
        """Return the expression."""
        return self.expression

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"PiGroup({self.name}, {self.expression})"


def _build_dimensional_matrix(
    variables: dict[str, Unit | Dimension | DimArray]
) -> tuple[NDArray[np.float64], list[str], list[int]]:
    """Build the dimensional matrix from variables.

    Args:
        variables: Dictionary of {variable_name: Unit/Dimension/DimArray}

    Returns:
        Tuple of (matrix, variable_names, active_dimensions):
        - matrix: k x n array where k = number of active dimensions, n = number of variables
        - variable_names: List of variable names in column order
        - active_dimensions: List of dimension indices that are non-zero across all variables
    """
    # Extract dimensions
    dimensions_dict: dict[str, Dimension] = {}
    for name, var in variables.items():
        if isinstance(var, DimArray):
            dimensions_dict[name] = var.unit.dimension
        elif isinstance(var, Unit):
            dimensions_dict[name] = var.dimension
        elif isinstance(var, Dimension):
            dimensions_dict[name] = var
        else:
            raise TypeError(
                f"Variable {name} must be Unit, Dimension, or DimArray, "
                f"got {type(var).__name__}"
            )

    # Determine active dimensions (non-zero across all variables)
    all_indices = [LENGTH, MASS, TIME, CURRENT, TEMPERATURE, AMOUNT, LUMINOSITY]
    active_dimensions = []
    for idx in all_indices:
        if any(dim._exponents[idx] != 0 for dim in dimensions_dict.values()):
            active_dimensions.append(idx)

    if not active_dimensions:
        # All variables are dimensionless - degenerate case
        variable_names = list(dimensions_dict.keys())
        return np.zeros((0, len(variable_names))), variable_names, []

    # Build matrix: rows = active dimensions, columns = variables
    variable_names = list(dimensions_dict.keys())
    n_dims = len(active_dimensions)
    n_vars = len(variable_names)

    matrix = np.zeros((n_dims, n_vars), dtype=np.float64)
    for j, var_name in enumerate(variable_names):
        dim = dimensions_dict[var_name]
        for i, dim_idx in enumerate(active_dimensions):
            # Convert Fraction to float for SVD
            matrix[i, j] = float(dim._exponents[dim_idx])

    return matrix, variable_names, active_dimensions


def _nullspace_svd(
    matrix: NDArray[np.float64], tolerance: float = 1e-10
) -> NDArray[np.float64]:
    """Compute nullspace of matrix using SVD.

    Args:
        matrix: k x n dimensional matrix
        tolerance: Singular values below this are considered zero

    Returns:
        n x (n-r) array where r is the rank, columns are nullspace basis vectors
    """
    if matrix.size == 0:
        # Empty matrix (all dimensionless variables)
        n_vars = matrix.shape[1] if matrix.ndim > 1 else 0
        if n_vars == 0:
            return np.zeros((0, 0))
        # All variables are dimensionless, identity matrix is the nullspace
        return np.eye(n_vars)

    # Try using SymPy for exact computation if available
    try:
        import sympy as sp
        from fractions import Fraction

        # Convert to SymPy rational matrix
        rows, cols = matrix.shape
        sym_matrix = sp.Matrix([[Fraction(matrix[i, j]).limit_denominator(1000)
                                for j in range(cols)]
                               for i in range(rows)])

        # Compute exact nullspace
        nullspace_sym = sym_matrix.nullspace()

        if not nullspace_sym:
            # No nullspace
            return np.zeros((cols, 0))

        # Convert back to numpy
        nullspace = np.zeros((cols, len(nullspace_sym)))
        for j, vec in enumerate(nullspace_sym):
            for i in range(cols):
                nullspace[i, j] = float(vec[i])

        return nullspace

    except ImportError:
        # Fall back to SVD if SymPy not available
        pass

    # Compute SVD: A = U Σ V^T
    U, S, Vt = np.linalg.svd(matrix, full_matrices=True)

    # Determine rank
    rank = np.sum(S > tolerance)

    # Nullspace = last (n - rank) rows of V^T = last (n - rank) columns of V
    V = Vt.T
    nullspace = V[:, rank:]

    return nullspace


def _clean_vector(
    vector: NDArray[np.float64], tolerance: float = 1e-10
) -> list[Fraction]:
    """Clean up numerical artifacts and convert to exact Fractions.

    Args:
        vector: Nullspace vector with numerical noise
        tolerance: Values with absolute value below this are set to zero

    Returns:
        List of Fractions representing exact rational exponents
    """
    # Round near-zero values
    cleaned = np.where(np.abs(vector) < tolerance, 0.0, vector)

    # Find a common scaling factor to convert to integers
    # We want to find the smallest integer multiplier that makes all values close to integers
    max_denom = 1000  # Maximum denominator to try

    # Try to find the best common denominator
    fractions = []
    for val in cleaned:
        if abs(val) < tolerance:
            fractions.append(Fraction(0))
        else:
            # Use limit_denominator to get close rational approximation
            frac = Fraction(val).limit_denominator(max_denom)
            fractions.append(frac)

    # Find LCM of all denominators to scale to integers
    from math import gcd
    def lcm(a, b):
        return abs(a * b) // gcd(a, b) if a and b else 0

    denominators = [f.denominator for f in fractions if f != 0]
    if denominators:
        common_denom = denominators[0]
        for d in denominators[1:]:
            common_denom = lcm(common_denom, d)

        # Scale all fractions
        scaled = [f * common_denom for f in fractions]

        # Find GCD of all numerators to reduce
        numerators = [int(s) for s in scaled if s != 0]
        if numerators:
            common_gcd = numerators[0]
            for n in numerators[1:]:
                common_gcd = gcd(common_gcd, n)

            # Reduce by GCD
            fractions = [Fraction(int(f * common_denom) // common_gcd, common_denom // common_gcd) if f != 0 else Fraction(0) for f in fractions]

    return fractions


def _normalize_exponents(exponents: list[Fraction]) -> list[Fraction]:
    """Normalize exponents so first non-zero value is positive.

    Args:
        exponents: List of exponents

    Returns:
        Normalized list of exponents
    """
    # Find first non-zero exponent
    for exp in exponents:
        if exp != 0:
            if exp < 0:
                # Negate all exponents
                return [-e for e in exponents]
            break

    return exponents


def _validate_dimensionless(
    exponents: dict[str, Fraction],
    variables: dict[str, Unit | Dimension | DimArray]
) -> bool:
    """Validate that the Pi group is truly dimensionless.

    Args:
        exponents: Dictionary of variable name -> exponent
        variables: Dictionary of variable name -> Unit/Dimension/DimArray

    Returns:
        True if the combination is dimensionless
    """
    # Extract dimensions
    dimensions_dict: dict[str, Dimension] = {}
    for name, var in variables.items():
        if isinstance(var, DimArray):
            dimensions_dict[name] = var.unit.dimension
        elif isinstance(var, Unit):
            dimensions_dict[name] = var.dimension
        elif isinstance(var, Dimension):
            dimensions_dict[name] = var

    # Compute product of dimensions raised to exponents
    result_dim = DIMENSIONLESS
    for var_name, exp in exponents.items():
        if exp != 0:
            dim = dimensions_dict[var_name]
            result_dim = result_dim * (dim ** exp)

    return result_dim.is_dimensionless


def _build_expression(exponents: dict[str, Fraction]) -> str:
    """Build human-readable expression from exponents.

    Args:
        exponents: Dictionary of variable name -> exponent

    Returns:
        String like 'ρvL/μ' or 'F/(ρv²L²)'
    """
    numerator_parts = []
    denominator_parts = []

    for var_name, exp in exponents.items():
        if exp == 0:
            continue

        # Format exponent
        if exp == 1:
            term = var_name
        elif exp == -1:
            term = var_name
        else:
            # Use superscript for integer exponents
            if exp.denominator == 1:
                exp_str = _format_superscript(int(abs(exp)))
            else:
                # For fractional exponents, use ^notation
                exp_str = f"^{abs(exp)}"
            term = f"{var_name}{exp_str}"

        if exp > 0:
            numerator_parts.append(term)
        else:
            denominator_parts.append(term)

    # Build expression
    if not numerator_parts and not denominator_parts:
        return "1"
    elif not denominator_parts:
        return "".join(numerator_parts)
    elif not numerator_parts:
        if len(denominator_parts) == 1:
            return f"1/{denominator_parts[0]}"
        else:
            return f"1/({''.join(denominator_parts)})"
    else:
        num = "".join(numerator_parts)
        if len(denominator_parts) == 1:
            return f"{num}/{denominator_parts[0]}"
        else:
            return f"{num}/({''.join(denominator_parts)})"


def _build_latex(exponents: dict[str, Fraction]) -> str:
    r"""Build LaTeX expression from exponents.

    Args:
        exponents: Dictionary of variable name -> exponent

    Returns:
        LaTeX string like r'\frac{\rho v L}{\mu}'
    """
    numerator_parts = []
    denominator_parts = []

    for var_name, exp in exponents.items():
        if exp == 0:
            continue

        # Use Greek letters if appropriate
        var_latex = _to_latex_symbol(var_name)

        # Format exponent
        if exp == 1:
            term = var_latex
        elif exp == -1:
            term = var_latex
        else:
            abs_exp = abs(exp)
            if abs_exp.denominator == 1:
                term = f"{var_latex}^{{{int(abs_exp)}}}"
            else:
                term = f"{var_latex}^{{{abs_exp}}}"

        if exp > 0:
            numerator_parts.append(term)
        else:
            denominator_parts.append(term)

    # Build LaTeX
    if not numerator_parts and not denominator_parts:
        return "1"
    elif not denominator_parts:
        return " ".join(numerator_parts)
    elif not numerator_parts:
        return r"\frac{1}{" + " ".join(denominator_parts) + "}"
    else:
        return r"\frac{" + " ".join(numerator_parts) + "}{" + " ".join(denominator_parts) + "}"


def _format_superscript(n: int) -> str:
    """Convert integer to Unicode superscript."""
    superscripts = {
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    }
    return "".join(superscripts.get(c, c) for c in str(n))


def _to_latex_symbol(var_name: str) -> str:
    """Convert variable name to LaTeX symbol."""
    # Common Greek letters
    greek_map = {
        "alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma",
        "delta": r"\delta", "epsilon": r"\epsilon", "zeta": r"\zeta",
        "eta": r"\eta", "theta": r"\theta", "iota": r"\iota",
        "kappa": r"\kappa", "lambda": r"\lambda", "mu": r"\mu",
        "nu": r"\nu", "xi": r"\xi", "pi": r"\pi",
        "rho": r"\rho", "sigma": r"\sigma", "tau": r"\tau",
        "phi": r"\phi", "chi": r"\chi", "psi": r"\psi", "omega": r"\omega",
    }

    lower_name = var_name.lower()
    if lower_name in greek_map:
        return greek_map[lower_name]

    return var_name


def _recognize_pi_group(
    exponents: dict[str, Fraction],
    variables: dict[str, Unit | Dimension | DimArray]
) -> tuple[str | None, str | None]:
    """Attempt to recognize known dimensionless numbers.

    Args:
        exponents: Dictionary of variable name -> exponent
        variables: Original variables dict

    Returns:
        Tuple of (name, interpretation) if recognized, else (None, None)
    """
    # This is a simplified pattern matcher for common dimensionless numbers
    # In a full implementation, this would check against a comprehensive database

    # Extract dimension signature
    dim_signature: dict[str, Dimension] = {}
    for name, var in variables.items():
        if isinstance(var, DimArray):
            dim_signature[name] = var.unit.dimension
        elif isinstance(var, Unit):
            dim_signature[name] = var.dimension
        elif isinstance(var, Dimension):
            dim_signature[name] = var

    # Check for Reynolds number pattern: ρvL/μ
    # Structure: density * velocity * length / viscosity
    # (M/L³) * (L/T) * L / (M/(L·T)) = dimensionless

    # Check for Froude number pattern: v/√(gL)
    # Structure: velocity / sqrt(acceleration * length)
    # (L/T) / sqrt((L/T²) * L) = dimensionless

    # For now, return None (pattern matching is complex)
    # Could be extended with a proper database and matching algorithm
    return None, None


def buckingham_pi(
    variables: dict[str, Unit | Dimension | DimArray],
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Apply Buckingham Pi theorem to find dimensionless groups.

    The Buckingham Pi theorem states that for n variables with k independent
    base dimensions, there exist n-k dimensionless Pi groups that completely
    describe the relationships between variables.

    Args:
        variables: Dictionary mapping variable names to Units, Dimensions, or DimArrays
        tolerance: Numerical tolerance for SVD and zero detection

    Returns:
        Dictionary containing:
        - 'pi_groups': List of PiGroup objects
        - 'rank': Number of independent dimensions (k)
        - 'n_variables': Number of variables (n)
        - 'n_groups': Number of Pi groups (n - k)
        - 'base_dimensions': List of active dimension symbols

    Raises:
        ValueError: If no variables are provided or if dimensions are invalid

    Example:
        >>> from dimtensor.analysis import buckingham_pi
        >>> from dimtensor.core.units import m, s, kg
        >>>
        >>> variables = {
        ...     'F': kg * m / s**2,
        ...     'v': m / s,
        ...     'L': m,
        ...     'rho': kg / m**3,
        ...     'mu': kg / (m * s),
        ... }
        >>> result = buckingham_pi(variables)
        >>> print(f"Found {result['n_groups']} Pi groups")
        >>> for pi in result['pi_groups']:
        ...     print(f"{pi.name} = {pi.expression}")
    """
    if not variables:
        raise ValueError("Must provide at least one variable")

    # Build dimensional matrix
    matrix, variable_names, active_dimensions = _build_dimensional_matrix(variables)

    # Get rank
    if matrix.size == 0:
        rank = 0
    else:
        S = np.linalg.svd(matrix, compute_uv=False)
        rank = int(np.sum(S > tolerance))

    # Compute nullspace
    nullspace = _nullspace_svd(matrix, tolerance)

    n_variables = len(variable_names)
    n_groups = nullspace.shape[1] if nullspace.size > 0 else 0

    # Build Pi groups
    pi_groups = []
    for i in range(n_groups):
        # Extract and clean nullspace vector
        vector = nullspace[:, i]
        exponents_list = _clean_vector(vector, tolerance)
        exponents_list = _normalize_exponents(exponents_list)

        # Build exponents dictionary
        exponents_dict = {
            var_name: exp
            for var_name, exp in zip(variable_names, exponents_list)
        }

        # Validate dimensionlessness
        if not _validate_dimensionless(exponents_dict, variables):
            # This shouldn't happen if the algorithm is correct
            # but we check for safety
            continue

        # Recognize known Pi group
        recognized_name, interpretation = _recognize_pi_group(exponents_dict, variables)

        # Build name
        if recognized_name:
            name = recognized_name
        else:
            # Use Greek Pi with subscript
            subscripts = {
                "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
                "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
            }
            subscript = "".join(subscripts.get(c, c) for c in str(i + 1))
            name = f"Π{subscript}"

        # Build expressions
        expression = _build_expression(exponents_dict)
        latex = _build_latex(exponents_dict)

        # Create Pi group
        pi_group = PiGroup(
            name=name,
            exponents=exponents_dict,
            expression=expression,
            latex=latex,
            interpretation=interpretation,
        )
        pi_groups.append(pi_group)

    # Get active dimension symbols
    base_dimensions = [_DIMENSION_SYMBOLS[idx] for idx in active_dimensions]

    return {
        'pi_groups': pi_groups,
        'rank': rank,
        'n_variables': n_variables,
        'n_groups': n_groups,
        'base_dimensions': base_dimensions,
    }


# Known dimensionless numbers database (for future extension)
KNOWN_PI_GROUPS = {
    'reynolds': {
        'name': 'Re',
        'interpretation': 'Inertial forces / Viscous forces',
        'formula': 'ρvL/μ',
    },
    'froude': {
        'name': 'Fr',
        'interpretation': 'Inertial forces / Gravitational forces',
        'formula': 'v/√(gL)',
    },
    'mach': {
        'name': 'Ma',
        'interpretation': 'Velocity / Speed of sound',
        'formula': 'v/c',
    },
    'prandtl': {
        'name': 'Pr',
        'interpretation': 'Momentum diffusivity / Thermal diffusivity',
        'formula': 'cpμ/k',
    },
    'nusselt': {
        'name': 'Nu',
        'interpretation': 'Convective heat transfer / Conductive heat transfer',
        'formula': 'hL/k',
    },
    'weber': {
        'name': 'We',
        'interpretation': 'Inertial forces / Surface tension',
        'formula': 'ρv²L/σ',
    },
    'strouhal': {
        'name': 'St',
        'interpretation': 'Oscillation frequency / Flow velocity',
        'formula': 'fL/v',
    },
}

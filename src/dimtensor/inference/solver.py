"""Main constraint solver for unit inference.

Builds constraints from equation expressions and propagates dimensional
information to infer unknown units.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional

from ..core.dimensions import Dimension, DIMENSIONLESS
from ..core.units import Unit
from .parser import (
    ExprNode, Variable, Constant, BinaryOp, UnaryOp, FunctionCall,
    parse_equation, get_variables
)
from .constraints import (
    Constraint, EqualityConstraint, MultiplicationConstraint,
    DivisionConstraint, PowerConstraint, DimensionlessConstraint
)


def infer_units(
    equation: str,
    known_units: Dict[str, Unit],
    database: Optional[Any] = None
) -> Dict[str, Any]:
    """Infer units for unknown variables in an equation.

    Args:
        equation: String like "F = m * a" or "E = m * c**2"
        known_units: Dict mapping variable names to known units
        database: Optional equation database to match against (not yet implemented)

    Returns:
        Dict with:
        - inferred: Dict[str, Unit] - inferred units for unknowns
        - is_consistent: bool - whether equation is dimensionally valid
        - confidence: float - confidence score 0-1
        - errors: List[str] - any dimensional errors found

    Examples:
        >>> from dimtensor.core.units import kg, m, s
        >>> result = infer_units(
        ...     "F = m * a",
        ...     {"m": kg, "a": m / s**2}
        ... )
        >>> result['is_consistent']
        True
        >>> result['inferred']['F'].dimension
        Dimension(mass=1, length=1, time=-2)
    """
    try:
        # Parse the equation
        left_expr, right_expr = parse_equation(equation)

        # Build constraints
        constraints = []

        # Create temporary variables for intermediate results
        temp_counter = [0]

        def process_expr(expr: ExprNode, constraints: List[Constraint]) -> str:
            """Process expression and return the variable name holding result."""
            nonlocal temp_counter

            if isinstance(expr, Variable):
                return expr.name

            elif isinstance(expr, Constant):
                # Constants are dimensionless
                temp_name = f"_const_{temp_counter[0]}"
                temp_counter[0] += 1
                constraints.append(
                    DimensionlessConstraint(temp_name, "numeric constant")
                )
                return temp_name

            elif isinstance(expr, BinaryOp):
                left_var = process_expr(expr.left, constraints)
                right_var = process_expr(expr.right, constraints)

                # Create result variable
                result_var = f"_temp_{temp_counter[0]}"
                temp_counter[0] += 1

                if expr.op == '+' or expr.op == '-':
                    # Addition/subtraction: same dimensions
                    constraints.append(
                        EqualityConstraint(left_var, right_var, expr.op)
                    )
                    constraints.append(
                        EqualityConstraint(result_var, left_var, '=')
                    )

                elif expr.op == '*':
                    # Multiplication
                    constraints.append(
                        MultiplicationConstraint(result_var, left_var, right_var)
                    )

                elif expr.op == '/':
                    # Division
                    constraints.append(
                        DivisionConstraint(result_var, left_var, right_var)
                    )

                elif expr.op == '**':
                    # Power
                    # Try to extract constant exponent value
                    exponent_value = None
                    if isinstance(expr.right, Constant):
                        exponent_value = expr.right.value

                    constraints.append(
                        PowerConstraint(result_var, left_var, right_var, exponent_value)
                    )

                return result_var

            elif isinstance(expr, UnaryOp):
                operand_var = process_expr(expr.operand, constraints)

                # Unary +/- doesn't change dimensions
                result_var = f"_temp_{temp_counter[0]}"
                temp_counter[0] += 1
                constraints.append(
                    EqualityConstraint(result_var, operand_var, '=')
                )

                return result_var

            elif isinstance(expr, FunctionCall):
                arg_var = process_expr(expr.arg, constraints)

                # Transcendental functions require dimensionless input
                if expr.func in ['sin', 'cos', 'tan', 'exp', 'log', 'ln', 'sqrt']:
                    if expr.func != 'sqrt':
                        # sin, cos, exp, etc. require dimensionless
                        constraints.append(
                            DimensionlessConstraint(arg_var, f"argument to {expr.func}()")
                        )

                        result_var = f"_temp_{temp_counter[0]}"
                        temp_counter[0] += 1
                        constraints.append(
                            DimensionlessConstraint(result_var, f"result of {expr.func}()")
                        )
                    else:
                        # sqrt(x) has dimension sqrt(dim(x))
                        result_var = f"_temp_{temp_counter[0]}"
                        temp_counter[0] += 1
                        constraints.append(
                            PowerConstraint(result_var, arg_var, "_dummy_half", 0.5)
                        )

                    return result_var
                else:
                    raise ValueError(f"Unsupported function: {expr.func}")

            else:
                raise ValueError(f"Unknown expression type: {type(expr)}")

        # Process both sides
        left_var = process_expr(left_expr, constraints)
        right_var = process_expr(right_expr, constraints)

        # Add constraint that both sides must match
        constraints.append(EqualityConstraint(left_var, right_var, '='))

        # Extract known dimensions
        known_dimensions = {
            var: unit.dimension
            for var, unit in known_units.items()
        }

        # Solve constraints
        inferred_dimensions, errors = solve_constraints(constraints, known_dimensions)

        # Check consistency
        is_consistent = len(errors) == 0

        # Convert dimensions back to units (with simplified symbols)
        inferred_units = {}
        for var, dim in inferred_dimensions.items():
            # Skip temporary variables
            if var.startswith('_'):
                continue

            # Skip variables we already knew
            if var in known_units:
                continue

            # Create a unit with this dimension
            unit = Unit("", dim, 1.0).simplified()
            inferred_units[var] = unit

        # Calculate confidence
        if not is_consistent:
            confidence = 0.0
        elif len(inferred_units) == 0:
            confidence = 1.0  # All variables were known
        else:
            # High confidence if we fully determined all unknowns
            all_vars = get_variables(left_expr) | get_variables(right_expr)
            unknown_vars = all_vars - set(known_units.keys())
            inferred_vars = set(inferred_units.keys())

            if unknown_vars == inferred_vars:
                confidence = 1.0
            else:
                # Partial inference
                confidence = 0.7

        return {
            'inferred': inferred_units,
            'is_consistent': is_consistent,
            'confidence': confidence,
            'errors': errors,
        }

    except Exception as e:
        # Return error result
        return {
            'inferred': {},
            'is_consistent': False,
            'confidence': 0.0,
            'errors': [str(e)],
        }


def solve_constraints(
    constraints: List[Constraint],
    known: Dict[str, Dimension]
) -> tuple[Dict[str, Dimension], List[str]]:
    """Solve constraint system to infer unknown dimensions.

    Args:
        constraints: List of constraints.
        known: Dict of known variable dimensions.

    Returns:
        Tuple of (all_dimensions, errors).
        all_dimensions includes both known and inferred.
        errors is a list of error messages.
    """
    # Start with known dimensions
    dimensions = dict(known)

    # Propagate constraints iteratively until convergence
    max_iterations = 100
    errors = []

    for iteration in range(max_iterations):
        changed = False

        # Try to propagate each constraint
        for constraint in constraints:
            try:
                inferred = constraint.propagate(dimensions)

                # Add newly inferred dimensions
                for var, dim in inferred.items():
                    if var not in dimensions:
                        dimensions[var] = dim
                        changed = True
                    elif dimensions[var] != dim:
                        # Conflict!
                        errors.append(
                            f"Conflicting dimensions for {var}: "
                            f"{dimensions[var]} vs {dim}"
                        )

            except Exception as e:
                errors.append(f"Error propagating constraint {constraint}: {e}")

        if not changed:
            break

    # Check all constraints
    for constraint in constraints:
        try:
            is_satisfied, error = constraint.check(dimensions)
            if not is_satisfied and error:
                errors.append(error)
        except Exception as e:
            errors.append(f"Error checking constraint {constraint}: {e}")

    return dimensions, errors

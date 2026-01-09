"""Expression parser for equation inference.

Parses arithmetic equations into expression trees for dimensional analysis.
Supports: +, -, *, /, **, parentheses, numeric constants, and variables.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


@dataclass
class ExprNode:
    """Base class for expression tree nodes."""
    pass


@dataclass
class Variable(ExprNode):
    """A variable in the expression."""
    name: str


@dataclass
class Constant(ExprNode):
    """A numeric constant (always dimensionless)."""
    value: float


@dataclass
class BinaryOp(ExprNode):
    """A binary operation: left op right."""
    op: str  # '+', '-', '*', '/', '**'
    left: ExprNode
    right: ExprNode


@dataclass
class UnaryOp(ExprNode):
    """A unary operation: op operand."""
    op: str  # '-', '+'
    operand: ExprNode


@dataclass
class FunctionCall(ExprNode):
    """A function call: func(arg)."""
    func: str  # 'sin', 'cos', 'exp', 'sqrt', etc.
    arg: ExprNode


def parse_equation(equation: str) -> tuple[ExprNode, ExprNode]:
    """Parse an equation into left and right expression trees.

    Args:
        equation: String like "F = m * a" or "E = m * c**2"

    Returns:
        Tuple of (left_expr, right_expr)

    Raises:
        ValueError: If equation is malformed or uses unsupported syntax.

    Examples:
        >>> left, right = parse_equation("F = m * a")
        >>> isinstance(left, Variable)
        True
        >>> left.name
        'F'
    """
    # Split on '=' to get left and right sides
    if '=' not in equation:
        raise ValueError(f"Equation must contain '=': {equation}")

    parts = equation.split('=', 1)
    if len(parts) != 2:
        raise ValueError(f"Equation must have exactly one '=': {equation}")

    left_str, right_str = parts
    left_str = left_str.strip()
    right_str = right_str.strip()

    if not left_str or not right_str:
        raise ValueError(f"Both sides of equation must be non-empty: {equation}")

    left_expr = parse_expression(left_str)
    right_expr = parse_expression(right_str)

    return left_expr, right_expr


def parse_expression(expr: str) -> ExprNode:
    """Parse a single expression into an expression tree.

    Args:
        expr: Expression string like "m * a" or "0.5 * m * v**2"

    Returns:
        ExprNode representing the expression tree.

    Raises:
        ValueError: If expression is malformed.

    Examples:
        >>> node = parse_expression("m * a")
        >>> isinstance(node, BinaryOp)
        True
        >>> node.op
        '*'
    """
    try:
        tree = ast.parse(expr, mode='eval')
        return _ast_to_expr(tree.body)
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {expr}") from e


def _ast_to_expr(node: ast.expr) -> ExprNode:
    """Convert Python AST node to our ExprNode."""
    if isinstance(node, ast.Name):
        # Variable
        return Variable(name=node.id)

    elif isinstance(node, ast.Constant):
        # Numeric constant
        if isinstance(node.value, (int, float)):
            return Constant(value=float(node.value))
        else:
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

    elif isinstance(node, ast.Num):  # Python 3.7 compatibility
        return Constant(value=float(node.n))

    elif isinstance(node, ast.BinOp):
        # Binary operation
        left = _ast_to_expr(node.left)
        right = _ast_to_expr(node.right)

        if isinstance(node.op, ast.Add):
            op = '+'
        elif isinstance(node.op, ast.Sub):
            op = '-'
        elif isinstance(node.op, ast.Mult):
            op = '*'
        elif isinstance(node.op, ast.Div):
            op = '/'
        elif isinstance(node.op, ast.Pow):
            op = '**'
        else:
            raise ValueError(f"Unsupported binary operator: {type(node.op)}")

        return BinaryOp(op=op, left=left, right=right)

    elif isinstance(node, ast.UnaryOp):
        # Unary operation
        operand = _ast_to_expr(node.operand)

        if isinstance(node.op, ast.USub):
            op = '-'
        elif isinstance(node.op, ast.UAdd):
            op = '+'
        else:
            raise ValueError(f"Unsupported unary operator: {type(node.op)}")

        return UnaryOp(op=op, operand=operand)

    elif isinstance(node, ast.Call):
        # Function call
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported")

        func_name = node.func.id

        if len(node.args) != 1:
            raise ValueError(f"Function {func_name} must have exactly one argument")

        arg = _ast_to_expr(node.args[0])

        return FunctionCall(func=func_name, arg=arg)

    else:
        raise ValueError(f"Unsupported expression type: {type(node)}")


def get_variables(expr: ExprNode) -> set[str]:
    """Extract all variable names from an expression tree.

    Args:
        expr: Expression tree.

    Returns:
        Set of variable names.

    Examples:
        >>> expr = parse_expression("m * a + F")
        >>> sorted(get_variables(expr))
        ['F', 'a', 'm']
    """
    variables = set()

    if isinstance(expr, Variable):
        variables.add(expr.name)
    elif isinstance(expr, Constant):
        pass  # No variables
    elif isinstance(expr, BinaryOp):
        variables.update(get_variables(expr.left))
        variables.update(get_variables(expr.right))
    elif isinstance(expr, UnaryOp):
        variables.update(get_variables(expr.operand))
    elif isinstance(expr, FunctionCall):
        variables.update(get_variables(expr.arg))

    return variables


def expr_to_string(expr: ExprNode) -> str:
    """Convert expression tree back to string.

    Args:
        expr: Expression tree.

    Returns:
        String representation.

    Examples:
        >>> expr = parse_expression("m * a")
        >>> expr_to_string(expr)
        '(m * a)'
    """
    if isinstance(expr, Variable):
        return expr.name
    elif isinstance(expr, Constant):
        # Format nicely
        if expr.value == int(expr.value):
            return str(int(expr.value))
        else:
            return str(expr.value)
    elif isinstance(expr, BinaryOp):
        left_str = expr_to_string(expr.left)
        right_str = expr_to_string(expr.right)
        return f"({left_str} {expr.op} {right_str})"
    elif isinstance(expr, UnaryOp):
        operand_str = expr_to_string(expr.operand)
        return f"({expr.op}{operand_str})"
    elif isinstance(expr, FunctionCall):
        arg_str = expr_to_string(expr.arg)
        return f"{expr.func}({arg_str})"
    else:
        return str(expr)

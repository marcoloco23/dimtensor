"""Dimensional linting for Python source files.

This module provides static analysis of Python code to detect potential
dimensional issues using variable name heuristics.

Example:
    >>> from dimtensor.cli import lint_file
    >>> results = lint_file("physics_simulation.py")
    >>> for r in results:
    ...     print(r)
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


class LintSeverity(Enum):
    """Severity levels for lint messages."""

    ERROR = "error"  # Definite dimensional error
    WARNING = "warning"  # Potential issue, high confidence
    INFO = "info"  # Suggestion or low-confidence finding


@dataclass
class LintResult:
    """Result of a lint check.

    Attributes:
        file: Path to the source file.
        line: Line number (1-indexed).
        column: Column number (0-indexed).
        severity: Severity level.
        code: Lint code (e.g., "W001").
        message: Human-readable message.
        suggestion: Optional suggestion for fixing.
        context: Optional code context.
    """

    file: str
    line: int
    column: int
    severity: LintSeverity
    code: str
    message: str
    suggestion: str | None = None
    context: str | None = None

    def __str__(self) -> str:
        """Format as a lint message."""
        severity_char = self.severity.value[0].upper()
        result = f"{self.file}:{self.line}:{self.column}: {severity_char}{self.code} {self.message}"
        if self.context:
            result += f"\n  {self.context}"
        if self.suggestion:
            result += f"\n  Suggestion: {self.suggestion}"
        return result

    def to_dict(self) -> dict[str, str | int | None]:
        """Convert to dictionary for JSON output."""
        return {
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "suggestion": self.suggestion,
            "context": self.context,
        }


@dataclass
class VariableInfo:
    """Information about a variable in the source."""

    name: str
    line: int
    column: int
    inferred_dimension: str | None = None
    confidence: float = 0.0


class DimensionalLinter(ast.NodeVisitor):
    """AST visitor that performs dimensional linting."""

    def __init__(self, source: str, filename: str, strict: bool = False):
        """Initialize the linter.

        Args:
            source: Python source code.
            filename: Name of the file being linted.
            strict: If True, report all potential issues.
        """
        self.source = source
        self.source_lines = source.splitlines()
        self.filename = filename
        self.strict = strict
        self.results: list[LintResult] = []
        self.variables: dict[str, VariableInfo] = {}
        self._reported_nodes: set[int] = set()  # Track reported node IDs
        self._import_inference()

    def _import_inference(self) -> None:
        """Import inference module (lazy to avoid circular imports)."""
        try:
            from dimtensor.inference import infer_dimension

            self._infer_dimension: Any = infer_dimension
        except ImportError:
            self._infer_dimension = None

    def _get_line_context(self, lineno: int) -> str:
        """Get the source line for context."""
        if 0 < lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return ""

    def _infer_var_dimension(self, name: str) -> tuple[str | None, float]:
        """Infer dimension for a variable name.

        Returns:
            Tuple of (dimension_string, confidence).
        """
        if self._infer_dimension is None:
            return None, 0.0

        result = self._infer_dimension(name)
        if result is None:
            return None, 0.0

        return str(result.dimension), result.confidence

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignment statements."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                dim, conf = self._infer_var_dimension(target.id)
                self.variables[target.id] = VariableInfo(
                    name=target.id,
                    line=node.lineno,
                    column=node.col_offset,
                    inferred_dimension=dim,
                    confidence=conf,
                )

                # Report dimension inference if strict mode
                if self.strict and dim and conf >= 0.7:
                    self.results.append(
                        LintResult(
                            file=self.filename,
                            line=node.lineno,
                            column=node.col_offset,
                            severity=LintSeverity.INFO,
                            code="001",
                            message=f"Variable '{target.id}' inferred as {dim}",
                            context=self._get_line_context(node.lineno),
                            suggestion=f"Consider adding explicit unit: DimArray(..., units.{self._suggest_unit(dim)})",
                        )
                    )

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit annotated assignment statements."""
        if isinstance(node.target, ast.Name):
            dim, conf = self._infer_var_dimension(node.target.id)
            self.variables[node.target.id] = VariableInfo(
                name=node.target.id,
                line=node.lineno,
                column=node.col_offset,
                inferred_dimension=dim,
                confidence=conf,
            )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Visit binary operations to check for dimension mismatches."""
        self._check_binop(node)
        self.generic_visit(node)

    def _check_expression(self, node: ast.expr, line: int, col: int) -> None:
        """Check an expression for dimensional issues."""
        if isinstance(node, ast.BinOp):
            self._check_binop(node)

    def _check_binop(self, node: ast.BinOp) -> None:
        """Check a binary operation for dimension mismatches."""
        # Skip if we've already reported this node
        node_id = id(node)
        if node_id in self._reported_nodes:
            return

        # Only check addition and subtraction
        if not isinstance(node.op, (ast.Add, ast.Sub)):
            return

        left_dim = self._get_expr_dimension(node.left)
        right_dim = self._get_expr_dimension(node.right)

        if left_dim and right_dim and left_dim != right_dim:
            op_str = "+" if isinstance(node.op, ast.Add) else "-"
            left_name = self._get_expr_name(node.left)
            right_name = self._get_expr_name(node.right)

            self._reported_nodes.add(node_id)
            self.results.append(
                LintResult(
                    file=self.filename,
                    line=node.lineno,
                    column=node.col_offset,
                    severity=LintSeverity.WARNING,
                    code="002",
                    message=f"Potential dimension mismatch: {left_dim} {op_str} {right_dim}",
                    context=f"{left_name} {op_str} {right_name}",
                    suggestion=f"Cannot add/subtract {left_dim} and {right_dim}. Check your units.",
                )
            )

    def _get_expr_dimension(self, node: ast.expr) -> str | None:
        """Get the inferred dimension of an expression."""
        if isinstance(node, ast.Name):
            if node.id in self.variables:
                return self.variables[node.id].inferred_dimension
            # Try to infer from name
            dim, conf = self._infer_var_dimension(node.id)
            if conf >= 0.5:
                return dim
        return None

    def _get_expr_name(self, node: ast.expr) -> str:
        """Get a string representation of an expression."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.BinOp):
            left = self._get_expr_name(node.left)
            right = self._get_expr_name(node.right)
            op = self._op_str(node.op)
            return f"({left} {op} {right})"
        return "..."

    def _op_str(self, op: ast.operator) -> str:
        """Get string representation of an operator."""
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Pow: "**",
        }
        return op_map.get(type(op), "?")

    def _suggest_unit(self, dim: str) -> str:
        """Suggest a unit name for a dimension string."""
        # Common dimension to unit mappings
        suggestions = {
            "L": "m",
            "L·T⁻¹": "m/s",
            "L·T⁻²": "m/s**2",
            "M": "kg",
            "T": "s",
            "M·L·T⁻²": "N",
            "M·L²·T⁻²": "J",
            "M·L²·T⁻³": "W",
        }
        return suggestions.get(dim, "...")


def lint_file(
    filepath: str | Path,
    strict: bool = False,
) -> list[LintResult]:
    """Lint a Python source file for dimensional issues.

    Args:
        filepath: Path to the Python file.
        strict: If True, report all potential issues including suggestions.

    Returns:
        List of lint results.

    Example:
        >>> results = lint_file("physics.py")
        >>> for r in results:
        ...     print(r)
    """
    path = Path(filepath)
    if not path.exists():
        return [
            LintResult(
                file=str(path),
                line=0,
                column=0,
                severity=LintSeverity.ERROR,
                code="000",
                message=f"File not found: {path}",
            )
        ]

    if not path.suffix == ".py":
        return []

    try:
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return [
            LintResult(
                file=str(path),
                line=e.lineno or 0,
                column=e.offset or 0,
                severity=LintSeverity.ERROR,
                code="000",
                message=f"Syntax error: {e.msg}",
            )
        ]

    linter = DimensionalLinter(source, str(path), strict=strict)
    linter.visit(tree)
    return linter.results


def lint_directory(
    dirpath: str | Path,
    strict: bool = False,
    recursive: bool = True,
) -> Iterator[LintResult]:
    """Lint all Python files in a directory.

    Args:
        dirpath: Path to the directory.
        strict: If True, report all potential issues.
        recursive: If True, search subdirectories.

    Yields:
        Lint results for each file.
    """
    path = Path(dirpath)
    if not path.is_dir():
        yield LintResult(
            file=str(path),
            line=0,
            column=0,
            severity=LintSeverity.ERROR,
            code="000",
            message=f"Not a directory: {path}",
        )
        return

    pattern = "**/*.py" if recursive else "*.py"
    for pyfile in path.glob(pattern):
        # Skip common non-source directories
        if any(part.startswith(".") or part == "__pycache__" for part in pyfile.parts):
            continue
        yield from lint_file(pyfile, strict=strict)


def format_results(
    results: list[LintResult],
    format: str = "text",
) -> str:
    """Format lint results for output.

    Args:
        results: List of lint results.
        format: Output format ("text" or "json").

    Returns:
        Formatted string.
    """
    if format == "json":
        return json.dumps([r.to_dict() for r in results], indent=2)

    if not results:
        return "No issues found."

    return "\n\n".join(str(r) for r in results)


def main() -> int:
    """Main entry point for the lint command.

    Returns:
        Exit code (0 for success, 1 for issues found).
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="dimtensor lint",
        description="Check Python files for dimensional issues",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to lint",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Report all potential issues including suggestions",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories",
    )

    args = parser.parse_args()
    all_results: list[LintResult] = []

    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file():
            all_results.extend(lint_file(path, strict=args.strict))
        elif path.is_dir():
            all_results.extend(
                lint_directory(path, strict=args.strict, recursive=not args.no_recursive)
            )
        else:
            all_results.append(
                LintResult(
                    file=path_str,
                    line=0,
                    column=0,
                    severity=LintSeverity.ERROR,
                    code="000",
                    message=f"Path not found: {path_str}",
                )
            )

    print(format_results(all_results, args.format))

    # Return 1 if any warnings or errors
    has_issues = any(r.severity in (LintSeverity.WARNING, LintSeverity.ERROR) for r in all_results)
    return 1 if has_issues else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

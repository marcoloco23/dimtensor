"""Command-line interface for dimtensor."""

from .lint import lint_file, lint_directory, LintResult, LintSeverity

__all__ = [
    "lint_file",
    "lint_directory",
    "LintResult",
    "LintSeverity",
]

"""Tests for the dimensional linting CLI."""

import tempfile
from pathlib import Path

import pytest

from dimtensor.cli.lint import (
    DimensionalLinter,
    LintResult,
    LintSeverity,
    format_results,
    lint_directory,
    lint_file,
)


class TestLintResult:
    """Test LintResult dataclass."""

    def test_str_format(self):
        """Test string formatting of lint result."""
        result = LintResult(
            file="test.py",
            line=10,
            column=4,
            severity=LintSeverity.WARNING,
            code="002",
            message="Test message",
        )
        s = str(result)
        assert "test.py:10:4" in s
        assert "W002" in s
        assert "Test message" in s

    def test_str_with_context(self):
        """Test string formatting with context."""
        result = LintResult(
            file="test.py",
            line=10,
            column=4,
            severity=LintSeverity.WARNING,
            code="002",
            message="Test message",
            context="velocity + acceleration",
        )
        s = str(result)
        assert "velocity + acceleration" in s

    def test_str_with_suggestion(self):
        """Test string formatting with suggestion."""
        result = LintResult(
            file="test.py",
            line=10,
            column=4,
            severity=LintSeverity.INFO,
            code="001",
            message="Test message",
            suggestion="Use explicit units",
        )
        s = str(result)
        assert "Suggestion:" in s
        assert "Use explicit units" in s

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = LintResult(
            file="test.py",
            line=10,
            column=4,
            severity=LintSeverity.WARNING,
            code="002",
            message="Test message",
        )
        d = result.to_dict()
        assert d["file"] == "test.py"
        assert d["line"] == 10
        assert d["severity"] == "warning"
        assert d["code"] == "002"


class TestDimensionalLinter:
    """Test the AST-based linter."""

    def test_infers_variable_dimensions(self):
        """Test that variable dimensions are inferred."""
        source = """
velocity = 10
acceleration = 5
"""
        linter = DimensionalLinter(source, "test.py", strict=True)
        import ast

        tree = ast.parse(source)
        linter.visit(tree)

        assert "velocity" in linter.variables
        assert linter.variables["velocity"].inferred_dimension is not None

    def test_detects_dimension_mismatch(self):
        """Test detection of dimension mismatch in addition."""
        source = """
velocity = 10
acceleration = 5
result = velocity + acceleration
"""
        linter = DimensionalLinter(source, "test.py", strict=False)
        import ast

        tree = ast.parse(source)
        linter.visit(tree)

        # Should find a warning about dimension mismatch
        warnings = [r for r in linter.results if r.severity == LintSeverity.WARNING]
        assert len(warnings) >= 1
        assert any("mismatch" in w.message.lower() for w in warnings)

    def test_allows_same_dimension_addition(self):
        """Test that same dimensions don't trigger warnings."""
        source = """
velocity1 = 10
velocity2 = 20
total_velocity = velocity1 + velocity2
"""
        linter = DimensionalLinter(source, "test.py", strict=False)
        import ast

        tree = ast.parse(source)
        linter.visit(tree)

        # Should not find warnings for same-dimension addition
        warnings = [r for r in linter.results if r.severity == LintSeverity.WARNING]
        assert len(warnings) == 0

    def test_strict_mode_reports_inferences(self):
        """Test that strict mode reports dimension inferences."""
        source = """
velocity = 10
"""
        linter = DimensionalLinter(source, "test.py", strict=True)
        import ast

        tree = ast.parse(source)
        linter.visit(tree)

        # Should find info messages about inferred dimensions
        infos = [r for r in linter.results if r.severity == LintSeverity.INFO]
        assert len(infos) >= 1


class TestLintFile:
    """Test the lint_file function."""

    def test_lint_nonexistent_file(self):
        """Test linting a file that doesn't exist."""
        results = lint_file("/nonexistent/path/file.py")
        assert len(results) == 1
        assert results[0].severity == LintSeverity.ERROR
        assert "not found" in results[0].message.lower()

    def test_lint_non_python_file(self):
        """Test that non-Python files are skipped."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not python")
            f.flush()
            results = lint_file(f.name)
            assert results == []

    def test_lint_syntax_error(self):
        """Test handling of syntax errors."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("def broken(\n")  # Syntax error
            f.flush()
            results = lint_file(f.name)
            assert len(results) == 1
            assert results[0].severity == LintSeverity.ERROR
            assert "syntax" in results[0].message.lower()

    def test_lint_valid_file(self):
        """Test linting a valid Python file."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(
                """
velocity = 10
time = 5
distance = velocity * time
"""
            )
            f.flush()
            results = lint_file(f.name)
            # Should complete without errors
            assert all(r.severity != LintSeverity.ERROR for r in results)

    def test_lint_file_with_mismatch(self):
        """Test linting a file with dimension mismatch."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(
                """
velocity = 10
acceleration = 5
bad = velocity + acceleration
"""
            )
            f.flush()
            results = lint_file(f.name, strict=False)
            warnings = [r for r in results if r.severity == LintSeverity.WARNING]
            assert len(warnings) >= 1


class TestLintDirectory:
    """Test the lint_directory function."""

    def test_lint_nonexistent_directory(self):
        """Test linting a directory that doesn't exist."""
        results = list(lint_directory("/nonexistent/path"))
        assert len(results) == 1
        assert results[0].severity == LintSeverity.ERROR

    def test_lint_empty_directory(self):
        """Test linting an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = list(lint_directory(tmpdir))
            assert results == []

    def test_lint_directory_with_files(self):
        """Test linting a directory with Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Python file
            (Path(tmpdir) / "test.py").write_text("velocity = 10\n")
            results = list(lint_directory(tmpdir, strict=True))
            # Should find at least the file
            assert any("test.py" in r.file for r in results)


class TestFormatResults:
    """Test the format_results function."""

    def test_format_text_empty(self):
        """Test text format with no results."""
        output = format_results([], format="text")
        assert "No issues found" in output

    def test_format_text_with_results(self):
        """Test text format with results."""
        results = [
            LintResult(
                file="test.py",
                line=10,
                column=4,
                severity=LintSeverity.WARNING,
                code="002",
                message="Test message",
            )
        ]
        output = format_results(results, format="text")
        assert "test.py:10:4" in output
        assert "W002" in output

    def test_format_json(self):
        """Test JSON format."""
        results = [
            LintResult(
                file="test.py",
                line=10,
                column=4,
                severity=LintSeverity.WARNING,
                code="002",
                message="Test message",
            )
        ]
        output = format_results(results, format="json")
        import json

        data = json.loads(output)
        assert len(data) == 1
        assert data[0]["file"] == "test.py"
        assert data[0]["severity"] == "warning"


class TestCLI:
    """Test the CLI entry point."""

    def test_main_help(self):
        """Test that main runs without error."""
        import sys
        from io import StringIO
        from unittest.mock import patch

        # Test with no arguments
        with patch.object(sys, "argv", ["dimtensor"]):
            from dimtensor.__main__ import main

            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                result = main()
                assert result == 0
                output = mock_stdout.getvalue()
                assert "lint" in output

    def test_info_command(self):
        """Test the info command."""
        import sys
        from io import StringIO
        from unittest.mock import patch

        with patch.object(sys, "argv", ["dimtensor", "info"]):
            from dimtensor.__main__ import main

            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                result = main()
                assert result == 0
                output = mock_stdout.getvalue()
                assert "dimtensor" in output
                assert "Rust backend" in output

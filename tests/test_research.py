"""Tests for paper reproduction framework."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np

from dimtensor import DimArray, units
from dimtensor.core.dimensions import Dimension
from dimtensor.equations import Equation
from dimtensor.research import (
    Paper,
    ReproductionResult,
    ReproductionStatus,
    ComparisonResult,
    compare_values,
    compare_all,
    generate_report,
    to_notebook,
)


class TestPaper:
    """Tests for Paper class."""

    def test_paper_creation(self):
        """Test creating a Paper object."""
        paper = Paper(
            title="Test Paper",
            authors=["Author One", "Author Two"],
            doi="10.1234/test",
            year=2020,
            journal="Test Journal",
        )

        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert paper.doi == "10.1234/test"
        assert paper.year == 2020

    def test_paper_with_published_values(self):
        """Test paper with published values."""
        paper = Paper(
            title="Measurement of c",
            authors=["Researcher"],
            published_values={
                "speed_of_light": DimArray(299792458, units.m / units.s),
            },
        )

        assert "speed_of_light" in paper.published_values
        assert paper.published_values["speed_of_light"].unit == units.m / units.s

    def test_paper_add_published_value(self):
        """Test adding published values."""
        paper = Paper(title="Test", authors=["A"])
        paper.add_published_value("mass", DimArray(1.0, units.kg), "kg")

        assert "mass" in paper.published_values
        assert "mass" in paper.units_used
        assert paper.units_used["mass"] == "kg"

    def test_paper_add_equation(self):
        """Test adding equations to paper."""
        paper = Paper(title="Test", authors=["A"])
        eq = Equation(
            name="F=ma",
            formula="F = m * a",
            variables={
                "F": Dimension(length=1, mass=1, time=-2),
                "m": Dimension(mass=1),
                "a": Dimension(length=1, time=-2),
            },
            domain="mechanics",
        )
        paper.add_equation("newtons_law", eq)

        assert "newtons_law" in paper.equations

    def test_paper_to_dict(self):
        """Test paper serialization to dict."""
        paper = Paper(
            title="Test",
            authors=["A", "B"],
            doi="10.1234/test",
            published_values={"x": DimArray(1.0, units.m)},
        )

        data = paper.to_dict()
        assert data["title"] == "Test"
        assert len(data["authors"]) == 2
        assert "x" in data["published_values"]

    def test_paper_from_dict(self):
        """Test paper deserialization from dict."""
        paper = Paper(
            title="Test",
            authors=["A"],
            published_values={"x": DimArray(1.0, units.m)},
        )

        data = paper.to_dict()
        paper2 = Paper.from_dict(data)

        assert paper2.title == paper.title
        assert paper2.authors == paper.authors
        assert "x" in paper2.published_values

    def test_paper_json_roundtrip(self):
        """Test paper JSON serialization roundtrip."""
        paper = Paper(
            title="Test Paper",
            authors=["Researcher"],
            doi="10.1234/test",
            year=2020,
            published_values={
                "length": DimArray(10.0, units.m),
                "mass": DimArray(5.0, units.kg),
            },
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            paper.to_json(temp_path)
            paper2 = Paper.from_json(temp_path)

            assert paper2.title == paper.title
            assert paper2.authors == paper.authors
            assert paper2.doi == paper.doi
            assert "length" in paper2.published_values
            assert "mass" in paper2.published_values
        finally:
            Path(temp_path).unlink()

    def test_paper_repr(self):
        """Test paper string representation."""
        paper = Paper(title="Test", authors=["A", "B", "C"], year=2020)
        repr_str = repr(paper)

        assert "Test" in repr_str
        assert "2020" in repr_str
        assert "et al." in repr_str  # More than 2 authors


class TestComparisonResult:
    """Tests for ComparisonResult class."""

    def test_comparison_result_creation(self):
        """Test creating a ComparisonResult."""
        pub = DimArray(1.0, units.m)
        comp = DimArray(1.01, units.m)
        abs_err = comp - pub

        result = ComparisonResult(
            quantity_name="length",
            published_value=pub,
            computed_value=comp,
            absolute_error=abs_err,
            relative_error=0.01,
            matches=True,
            tolerance_used=(0.05, None),
            unit_conversion_applied=False,
        )

        assert result.quantity_name == "length"
        assert result.matches is True
        assert result.relative_error == 0.01

    def test_comparison_result_to_dict(self):
        """Test comparison result serialization."""
        pub = DimArray(1.0, units.m)
        comp = DimArray(1.01, units.m)

        result = ComparisonResult(
            quantity_name="x",
            published_value=pub,
            computed_value=comp,
            absolute_error=comp - pub,
            relative_error=0.01,
            matches=True,
            tolerance_used=(0.05, None),
            unit_conversion_applied=False,
        )

        data = result.to_dict()
        assert data["quantity_name"] == "x"
        assert data["matches"] is True
        assert data["relative_error"] == 0.01

    def test_comparison_result_from_dict(self):
        """Test comparison result deserialization."""
        pub = DimArray(1.0, units.m)
        comp = DimArray(1.01, units.m)

        result = ComparisonResult(
            quantity_name="x",
            published_value=pub,
            computed_value=comp,
            absolute_error=comp - pub,
            relative_error=0.01,
            matches=True,
            tolerance_used=(0.05, None),
            unit_conversion_applied=False,
        )

        data = result.to_dict()
        result2 = ComparisonResult.from_dict(data)

        assert result2.quantity_name == result.quantity_name
        assert result2.matches == result.matches


class TestCompareValues:
    """Tests for compare_values function."""

    def test_compare_identical_values(self):
        """Test comparing identical values."""
        pub = DimArray(1.0, units.m)
        comp = DimArray(1.0, units.m)

        result = compare_values(pub, comp)

        assert result.matches is True
        assert result.relative_error == 0.0

    def test_compare_within_tolerance(self):
        """Test comparing values within tolerance."""
        pub = DimArray(100.0, units.m)
        comp = DimArray(100.05, units.m)

        result = compare_values(pub, comp, rtol=0.001)  # 0.1% tolerance

        assert result.matches is True
        assert result.relative_error < 0.001

    def test_compare_outside_tolerance(self):
        """Test comparing values outside tolerance."""
        pub = DimArray(100.0, units.m)
        comp = DimArray(102.0, units.m)

        result = compare_values(pub, comp, rtol=0.001)  # 0.1% tolerance

        assert result.matches is False
        assert result.relative_error > 0.001

    def test_compare_different_units(self):
        """Test comparing values with different units."""
        pub = DimArray(1.0, units.km)
        comp = DimArray(1001.0, units.m)

        result = compare_values(pub, comp, rtol=0.01)

        assert result.matches is True
        assert result.unit_conversion_applied is True

    def test_compare_incompatible_dimensions(self):
        """Test comparing incompatible dimensions raises error."""
        pub = DimArray(1.0, units.m)
        comp = DimArray(1.0, units.kg)

        with pytest.raises(Exception):  # DimensionError
            compare_values(pub, comp)

    def test_compare_with_absolute_tolerance(self):
        """Test comparison with absolute tolerance."""
        pub = DimArray(0.0, units.m)
        comp = DimArray(0.001, units.m)

        result = compare_values(pub, comp, rtol=None, atol=0.01)

        assert result.matches is True

    def test_compare_arrays(self):
        """Test comparing array values."""
        pub = DimArray([1.0, 2.0, 3.0], units.m)
        comp = DimArray([1.01, 2.01, 3.01], units.m)

        result = compare_values(pub, comp, rtol=0.02)

        assert result.matches is True

    def test_compare_with_uncertainties(self):
        """Test comparison with uncertainties."""
        pub = DimArray(1.0, units.m, uncertainty=np.array(0.01))
        comp = DimArray(1.005, units.m, uncertainty=np.array(0.01))

        result = compare_values(pub, comp, rtol=0.01)  # 1% tolerance

        assert result.matches is True
        assert "uncertainties" in result.notes

    def test_compare_zero_published_value(self):
        """Test comparing when published value is zero."""
        pub = DimArray(0.0, units.m)
        comp = DimArray(0.001, units.m)

        result = compare_values(pub, comp, atol=0.01)

        assert result.relative_error is None
        assert result.matches is True


class TestCompareAll:
    """Tests for compare_all function."""

    def test_compare_all_basic(self):
        """Test comparing multiple values."""
        pub = {
            "length": DimArray(1.0, units.m),
            "mass": DimArray(2.0, units.kg),
        }
        comp = {
            "length": DimArray(1.01, units.m),
            "mass": DimArray(2.02, units.kg),
        }

        results = compare_all(pub, comp, rtol=0.05)

        assert len(results) == 2
        assert "length" in results
        assert "mass" in results
        assert results["length"].matches is True

    def test_compare_all_missing_computed(self):
        """Test compare_all with missing computed values."""
        pub = {
            "length": DimArray(1.0, units.m),
            "mass": DimArray(2.0, units.kg),
        }
        comp = {
            "length": DimArray(1.01, units.m),
        }

        results = compare_all(pub, comp)

        assert len(results) == 1  # Only length compared
        assert "mass" not in results

    def test_compare_all_dimension_error(self):
        """Test compare_all handles dimension errors gracefully."""
        pub = {
            "length": DimArray(1.0, units.m),
        }
        comp = {
            "length": DimArray(1.0, units.kg),  # Wrong dimension
        }

        results = compare_all(pub, comp)

        assert len(results) == 1
        assert results["length"].matches is False
        assert "Dimension error" in results["length"].notes


class TestReproductionResult:
    """Tests for ReproductionResult class."""

    def test_reproduction_result_creation(self):
        """Test creating a ReproductionResult."""
        paper = Paper(title="Test", authors=["A"])
        result = ReproductionResult(
            paper=paper,
            status=ReproductionStatus.SUCCESS,
            reproducer="Researcher",
        )

        assert result.paper == paper
        assert result.status == ReproductionStatus.SUCCESS
        assert result.reproducer == "Researcher"

    def test_add_computed_value(self):
        """Test adding computed values."""
        paper = Paper(title="Test", authors=["A"])
        result = ReproductionResult(paper=paper)

        result.add_computed_value("x", DimArray(1.0, units.m))

        assert "x" in result.computed_values

    def test_add_comparison(self):
        """Test adding comparison results."""
        paper = Paper(title="Test", authors=["A"])
        result = ReproductionResult(paper=paper)

        pub = DimArray(1.0, units.m)
        comp = DimArray(1.001, units.m)  # 0.1% difference
        comparison = compare_values(pub, comp, rtol=0.01)  # 1% tolerance

        result.add_comparison("x", comparison)

        assert "x" in result.comparisons
        assert result.status == ReproductionStatus.SUCCESS  # Auto-updated

    def test_status_update_partial(self):
        """Test status updates to PARTIAL."""
        paper = Paper(title="Test", authors=["A"])
        result = ReproductionResult(paper=paper)

        # Add one matching and one non-matching comparison
        pub1 = DimArray(1.0, units.m)
        comp1 = DimArray(1.0, units.m)
        result.add_comparison("match", compare_values(pub1, comp1))

        pub2 = DimArray(1.0, units.m)
        comp2 = DimArray(2.0, units.m)
        result.add_comparison("mismatch", compare_values(pub2, comp2, rtol=0.01))

        assert result.status == ReproductionStatus.PARTIAL

    def test_status_update_failed(self):
        """Test status updates to FAILED."""
        paper = Paper(title="Test", authors=["A"])
        result = ReproductionResult(paper=paper)

        # Add non-matching comparison
        pub = DimArray(1.0, units.m)
        comp = DimArray(2.0, units.m)
        result.add_comparison("mismatch", compare_values(pub, comp, rtol=0.01))

        assert result.status == ReproductionStatus.FAILED

    def test_summary(self):
        """Test summary statistics."""
        paper = Paper(title="Test", authors=["A"])
        result = ReproductionResult(paper=paper)

        result.add_computed_value("x", DimArray(1.0, units.m))
        pub = DimArray(1.0, units.m)
        comp = DimArray(1.001, units.m)  # 0.1% difference
        result.add_comparison("x", compare_values(pub, comp, rtol=0.01))  # 1% tolerance

        summary = result.summary()

        assert summary["status"] == "success"
        assert summary["num_quantities"] == 1
        assert summary["num_compared"] == 1
        assert summary["num_matches"] == 1

    def test_reproduction_json_roundtrip(self):
        """Test ReproductionResult JSON serialization."""
        paper = Paper(title="Test", authors=["A"])
        result = ReproductionResult(
            paper=paper,
            computed_values={"x": DimArray(1.0, units.m)},
            status=ReproductionStatus.SUCCESS,
            reproducer="Test",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            result.to_json(temp_path)
            result2 = ReproductionResult.from_json(temp_path)

            assert result2.paper.title == result.paper.title
            assert result2.status == result.status
            assert "x" in result2.computed_values
        finally:
            Path(temp_path).unlink()


class TestReporting:
    """Tests for report generation."""

    def test_generate_markdown_report(self):
        """Test generating markdown report."""
        paper = Paper(
            title="Test Paper",
            authors=["Researcher"],
            doi="10.1234/test",
            year=2020,
        )
        result = ReproductionResult(
            paper=paper,
            status=ReproductionStatus.SUCCESS,
            reproducer="Test User",
        )

        report = generate_report(result, format="markdown")

        assert "Test Paper" in report
        assert "Researcher" in report
        assert "SUCCESS" in report

    def test_generate_html_report(self):
        """Test generating HTML report."""
        paper = Paper(title="Test", authors=["A"])
        result = ReproductionResult(paper=paper, status=ReproductionStatus.SUCCESS)

        report = generate_report(result, format="html")

        assert "<html>" in report
        assert "Test" in report

    def test_generate_latex_report(self):
        """Test generating LaTeX report."""
        paper = Paper(title="Test", authors=["A"])
        result = ReproductionResult(paper=paper, status=ReproductionStatus.SUCCESS)

        report = generate_report(result, format="latex")

        assert "\\documentclass" in report
        assert "Test" in report

    def test_generate_text_report(self):
        """Test generating text report."""
        paper = Paper(title="Test", authors=["A"])
        result = ReproductionResult(paper=paper, status=ReproductionStatus.SUCCESS)

        report = generate_report(result, format="text")

        assert "Test" in report
        assert "SUCCESS" in report

    def test_generate_report_with_comparisons(self):
        """Test report with comparison results."""
        paper = Paper(
            title="Test",
            authors=["A"],
            published_values={"x": DimArray(1.0, units.m)},
        )
        result = ReproductionResult(paper=paper)

        comp = DimArray(1.01, units.m)
        result.add_computed_value("x", comp)
        result.add_comparison("x", compare_values(paper.published_values["x"], comp))

        report = generate_report(result, format="markdown")

        assert "Comparison Results" in report
        assert "x" in report

    def test_unsupported_format(self):
        """Test unsupported format raises error."""
        paper = Paper(title="Test", authors=["A"])
        result = ReproductionResult(paper=paper)

        with pytest.raises(ValueError):
            generate_report(result, format="pdf")

    def test_to_notebook(self):
        """Test exporting to Jupyter notebook."""
        paper = Paper(title="Test Paper", authors=["Researcher"])
        result = ReproductionResult(
            paper=paper,
            status=ReproductionStatus.SUCCESS,
            reproducer="Test User",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
            temp_path = f.name

        try:
            to_notebook(result, temp_path)

            # Verify notebook was created
            assert Path(temp_path).exists()

            # Verify notebook structure
            with open(temp_path) as f:
                nb = json.load(f)

            assert "cells" in nb
            assert len(nb["cells"]) > 0
            assert nb["cells"][0]["cell_type"] == "markdown"
            assert "Test Paper" in "".join(nb["cells"][0]["source"])

            # Verify JSON was also saved
            json_path = Path(temp_path).with_suffix(".json")
            assert json_path.exists()
        finally:
            Path(temp_path).unlink(missing_ok=True)
            Path(temp_path).with_suffix(".json").unlink(missing_ok=True)


class TestIntegration:
    """Integration tests for full reproduction workflow."""

    def test_full_reproduction_workflow(self):
        """Test complete reproduction workflow."""
        # 1. Create paper
        paper = Paper(
            title="Schwarzschild Radius Calculation",
            authors=["K. Schwarzschild"],
            doi="10.1002/asna.19160200402",
            year=1916,
            journal="Astronomische Nachrichten",
            published_values={
                "solar_schwarzschild_radius": DimArray(2.95, units.km),
            },
            assumptions=["Spherical symmetry", "Static metric"],
            tags=["general relativity", "black holes"],
        )

        # 2. Compute values using dimtensor
        from dimtensor.constants import G, c

        # Solar mass
        solar_mass = DimArray(1.989e30, units.kg)
        r_s = 2 * G * solar_mass / c**2

        # 3. Create reproduction result
        result = ReproductionResult(
            paper=paper,
            reproducer="Test User",
            code_repository="https://github.com/test/repo",
        )

        result.add_computed_value("solar_schwarzschild_radius", r_s.to(units.km))

        # 4. Compare values
        comparison = compare_values(
            paper.published_values["solar_schwarzschild_radius"],
            r_s.to(units.km),
            rtol=0.01,  # 1% tolerance
        )

        result.add_comparison("solar_schwarzschild_radius", comparison)

        # 5. Generate report
        report = generate_report(result, format="markdown")

        # Verify workflow
        assert result.status in [ReproductionStatus.SUCCESS, ReproductionStatus.PARTIAL]
        assert "Schwarzschild" in report
        assert len(result.comparisons) == 1

    def test_millikan_oil_drop_reproduction(self):
        """Test reproducing Millikan oil drop experiment."""
        # Historical value from 1913
        paper = Paper(
            title="On the Elementary Electrical Charge",
            authors=["R. A. Millikan"],
            doi="10.1103/PhysRev.2.109",
            year=1913,
            journal="Physical Review",
            published_values={
                "electron_charge": DimArray(1.592e-19, units.C),
            },
            notes="Historical value, known to have systematic error",
        )

        # Modern accepted value
        e_modern = DimArray(1.602176634e-19, units.C)

        result = ReproductionResult(paper=paper, reproducer="Test")
        result.add_computed_value("electron_charge", e_modern)

        comparison = compare_values(
            paper.published_values["electron_charge"],
            e_modern,
            rtol=0.01,
        )

        result.add_comparison("electron_charge", comparison)

        # This should show a discrepancy (Millikan's value was too low)
        assert comparison.relative_error is not None
        assert comparison.relative_error > 0.005  # More than 0.5% error

"""Report generation for paper reproduction results.

This module provides functions for generating reports in various formats
(Markdown, JSON, Jupyter notebooks) from reproduction results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .reproduction import ReproductionResult, ReproductionStatus


def generate_report(result: ReproductionResult, format: str = "markdown") -> str:
    """Generate a report from reproduction results.

    Args:
        result: ReproductionResult object.
        format: Output format ("markdown", "html", "latex", or "text").

    Returns:
        Report as a string.

    Raises:
        ValueError: If format is not supported.

    Example:
        >>> from dimtensor.research import generate_report
        >>> report = generate_report(result, format="markdown")
        >>> print(report)
    """
    if format == "markdown":
        return _generate_markdown(result)
    elif format == "html":
        return _generate_html(result)
    elif format == "latex":
        return _generate_latex(result)
    elif format == "text":
        return _generate_text(result)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _generate_markdown(result: ReproductionResult) -> str:
    """Generate Markdown report."""
    lines = [
        f"# Reproduction Report: {result.paper.title}",
        "",
        "## Paper Information",
        "",
        f"**Title:** {result.paper.title}",
        f"**Authors:** {', '.join(result.paper.authors)}",
    ]

    if result.paper.doi:
        lines.append(f"**DOI:** {result.paper.doi}")
    if result.paper.year:
        lines.append(f"**Year:** {result.paper.year}")
    if result.paper.journal:
        lines.append(f"**Journal:** {result.paper.journal}")

    lines.extend(["", "---", "", "## Reproduction Information", ""])

    if result.reproducer:
        lines.append(f"**Reproduced by:** {result.reproducer}")
    lines.append(f"**Date:** {result.reproduction_date}")
    if result.code_repository:
        lines.append(f"**Code Repository:** {result.code_repository}")
    if result.computation_time > 0:
        lines.append(f"**Computation Time:** {result.computation_time:.2f} seconds")

    # Status
    status_emoji = {
        ReproductionStatus.SUCCESS: "✅",
        ReproductionStatus.PARTIAL: "⚠️",
        ReproductionStatus.FAILED: "❌",
        ReproductionStatus.PENDING: "⏳",
    }
    emoji = status_emoji.get(result.status, "")
    lines.extend(
        [
            "",
            f"**Status:** {emoji} {result.status.value.upper()}",
            "",
        ]
    )

    # Summary statistics
    summary = result.summary()
    lines.extend(
        [
            "## Summary",
            "",
            f"- **Quantities Computed:** {summary['num_compared']} / {summary['num_quantities']}",
            f"- **Matches:** {summary.get('num_matches', 0)} / {summary['num_compared']}",
        ]
    )

    if summary.get("mean_relative_error") is not None:
        lines.append(
            f"- **Mean Relative Error:** {summary['mean_relative_error']:.2e}"
        )
    if summary.get("max_relative_error") is not None:
        lines.append(f"- **Max Relative Error:** {summary['max_relative_error']:.2e}")

    lines.extend(["", "---", ""])

    # Comparison table
    if result.comparisons:
        lines.extend(
            [
                "## Comparison Results",
                "",
                "| Quantity | Published | Computed | Rel. Error | Match |",
                "|----------|-----------|----------|------------|-------|",
            ]
        )

        for name, comp in result.comparisons.items():
            pub_str = f"{comp.published_value.data} {comp.published_value.unit}"
            comp_str = f"{comp.computed_value.data} {comp.computed_value.unit}"
            rel_err_str = (
                f"{comp.relative_error:.2e}" if comp.relative_error is not None else "N/A"
            )
            match_str = "✅" if comp.matches else "❌"

            lines.append(f"| {name} | {pub_str} | {comp_str} | {rel_err_str} | {match_str} |")

        lines.append("")

    # Discrepancies
    if result.discrepancies:
        lines.extend(["## Significant Discrepancies", ""])
        for disc in result.discrepancies:
            lines.append(f"- {disc}")
        lines.append("")

    # Notes
    if result.notes:
        lines.extend(["## Notes", "", result.notes, ""])

    # Software versions
    if result.software_versions:
        lines.extend(["## Software Versions", ""])
        for name, version in result.software_versions.items():
            lines.append(f"- **{name}:** {version}")
        lines.append("")

    # Paper metadata
    if result.paper.assumptions:
        lines.extend(["## Paper Assumptions", ""])
        for assumption in result.paper.assumptions:
            lines.append(f"- {assumption}")
        lines.append("")

    if result.paper.methods:
        lines.extend(["## Methods Used", ""])
        for method in result.paper.methods:
            lines.append(f"- {method}")
        lines.append("")

    return "\n".join(lines)


def _generate_html(result: ReproductionResult) -> str:
    """Generate HTML report."""
    # Convert markdown to HTML (simple conversion)
    md = _generate_markdown(result)
    lines = []
    lines.append("<!DOCTYPE html>")
    lines.append("<html><head><meta charset='utf-8'><title>Reproduction Report</title>")
    lines.append("<style>")
    lines.append("body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }")
    lines.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
    lines.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
    lines.append("th { background-color: #f2f2f2; }")
    lines.append("h1, h2 { color: #333; }")
    lines.append("</style></head><body>")

    # Simple markdown to HTML conversion
    for line in md.split("\n"):
        if line.startswith("# "):
            lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("**") and line.endswith("**"):
            lines.append(f"<p><strong>{line[2:-2]}</strong></p>")
        elif line.startswith("- "):
            lines.append(f"<li>{line[2:]}</li>")
        elif line.startswith("|"):
            # Table row
            if "---" in line:
                continue  # Skip separator
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if line.count("|") > 2:  # Has content
                tag = "th" if "Quantity" in line else "td"
                lines.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
        elif line == "":
            lines.append("<br>")
        else:
            lines.append(f"<p>{line}</p>")

    lines.append("</body></html>")
    return "\n".join(lines)


def _generate_latex(result: ReproductionResult) -> str:
    """Generate LaTeX report."""
    lines = [
        "\\documentclass{article}",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage{booktabs}",
        "\\title{Reproduction Report: " + result.paper.title.replace("_", "\\_") + "}",
        "\\author{" + result.reproducer.replace("_", "\\_") + "}",
        "\\date{" + result.reproduction_date + "}",
        "\\begin{document}",
        "\\maketitle",
        "",
        "\\section{Paper Information}",
        "",
    ]

    lines.append("\\textbf{Authors:} " + ", ".join(result.paper.authors) + "\\\\")
    if result.paper.journal:
        lines.append("\\textbf{Journal:} " + result.paper.journal.replace("_", "\\_") + "\\\\")
    if result.paper.year:
        lines.append(f"\\textbf{{Year:}} {result.paper.year}\\\\")
    if result.paper.doi:
        lines.append("\\textbf{DOI:} " + result.paper.doi.replace("_", "\\_") + "\\\\")

    lines.extend(["", "\\section{Reproduction Status}", ""])
    lines.append(f"Status: \\textbf{{{result.status.value.upper()}}}")

    if result.comparisons:
        lines.extend(
            [
                "",
                "\\section{Comparison Results}",
                "",
                "\\begin{tabular}{lllll}",
                "\\toprule",
                "Quantity & Published & Computed & Rel. Error & Match \\\\",
                "\\midrule",
            ]
        )

        for name, comp in result.comparisons.items():
            pub_str = f"{comp.published_value.data}"
            comp_str = f"{comp.computed_value.data}"
            rel_err_str = (
                f"{comp.relative_error:.2e}" if comp.relative_error is not None else "N/A"
            )
            match_str = "Yes" if comp.matches else "No"

            lines.append(
                f"{name} & {pub_str} & {comp_str} & {rel_err_str} & {match_str} \\\\"
            )

        lines.extend(["\\bottomrule", "\\end{tabular}", ""])

    lines.extend(["", "\\end{document}"])
    return "\n".join(lines)


def _generate_text(result: ReproductionResult) -> str:
    """Generate plain text report."""
    lines = [
        "=" * 80,
        f"REPRODUCTION REPORT: {result.paper.title}",
        "=" * 80,
        "",
        "PAPER INFORMATION",
        "-" * 80,
        f"Authors: {', '.join(result.paper.authors)}",
    ]

    if result.paper.journal:
        lines.append(f"Journal: {result.paper.journal}")
    if result.paper.year:
        lines.append(f"Year: {result.paper.year}")
    if result.paper.doi:
        lines.append(f"DOI: {result.paper.doi}")

    lines.extend(
        [
            "",
            "REPRODUCTION INFORMATION",
            "-" * 80,
            f"Reproduced by: {result.reproducer}",
            f"Date: {result.reproduction_date}",
            f"Status: {result.status.value.upper()}",
            "",
        ]
    )

    if result.comparisons:
        lines.extend(["COMPARISON RESULTS", "-" * 80])
        for name, comp in result.comparisons.items():
            lines.append(f"\n{name}:")
            lines.append(f"  Published: {comp.published_value.data} {comp.published_value.unit}")
            lines.append(f"  Computed:  {comp.computed_value.data} {comp.computed_value.unit}")
            if comp.relative_error is not None:
                lines.append(f"  Rel. Error: {comp.relative_error:.2e}")
            lines.append(f"  Match: {'YES' if comp.matches else 'NO'}")

    lines.extend(["", "=" * 80])
    return "\n".join(lines)


def to_notebook(result: ReproductionResult, path: str | Path) -> None:
    """Export reproduction result to Jupyter notebook.

    Creates a Jupyter notebook with all metadata, comparisons,
    and code cells for verification.

    Args:
        result: ReproductionResult object.
        path: Path to save notebook (.ipynb file).

    Example:
        >>> from dimtensor.research import to_notebook
        >>> to_notebook(result, "reproduction_report.ipynb")
    """
    path = Path(path)

    # Create notebook structure
    cells = []

    # Title cell
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# Reproduction Report: {result.paper.title}\n",
                "\n",
                f"**Authors:** {', '.join(result.paper.authors)}\n",
                f"**Status:** {result.status.value.upper()}\n",
            ],
        }
    )

    # Paper information cell
    paper_info = ["## Paper Information\n", "\n"]
    if result.paper.doi:
        paper_info.append(f"**DOI:** {result.paper.doi}\n")
    if result.paper.year:
        paper_info.append(f"**Year:** {result.paper.year}\n")
    if result.paper.journal:
        paper_info.append(f"**Journal:** {result.paper.journal}\n")
    if result.paper.abstract:
        paper_info.extend(["\n", "**Abstract:**\n", "\n", result.paper.abstract, "\n"])

    cells.append({"cell_type": "markdown", "metadata": {}, "source": paper_info})

    # Comparison results cell
    if result.comparisons:
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Comparison Results\n"],
            }
        )

        # Code cell to display results
        code = [
            "from dimtensor.research import ReproductionResult\n",
            "\n",
            f"# Load result from JSON\n",
            f"result = ReproductionResult.from_json('{path.with_suffix('.json')}')\n",
            "\n",
            "# Display comparisons\n",
            "for name, comp in result.comparisons.items():\n",
            "    print(f'{name}:')\n",
            "    print(f'  Published: {comp.published_value}')\n",
            "    print(f'  Computed:  {comp.computed_value}')\n",
            "    if comp.relative_error:\n",
            "        print(f'  Rel. Error: {comp.relative_error:.2e}')\n",
            "    print(f'  Match: {comp.matches}')\n",
            "    print()\n",
        ]

        cells.append(
            {"cell_type": "code", "metadata": {}, "source": code, "outputs": [], "execution_count": None}
        )

    # Notes cell
    if result.notes:
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Notes\n", "\n", result.notes, "\n"],
            }
        )

    # Create notebook
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    # Save notebook
    with open(path, "w") as f:
        json.dump(notebook, f, indent=2)

    # Also save result as JSON for loading in notebook
    result.to_json(path.with_suffix(".json"))

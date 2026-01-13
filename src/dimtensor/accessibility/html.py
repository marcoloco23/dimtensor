"""HTML formatters for Jupyter notebook accessibility.

Provides semantic HTML with ARIA labels for screen reader compatibility
and high-contrast CSS themes for visual accessibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..core.dimarray import DimArray

from .formatters import expand_unit_name


# ==============================================================================
# CSS Styles
# ==============================================================================

DEFAULT_CSS = """
<style>
.dimarray-container {
    font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Menlo, 'Courier New', monospace;
    margin: 10px 0;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background-color: #f8f9fa;
}

.dimarray-value {
    font-weight: 600;
    color: #0066cc;
}

.dimarray-unit {
    font-style: italic;
    color: #666;
    margin-left: 4px;
}

.dimarray-shape {
    font-size: 0.9em;
    color: #888;
}

.dimarray-uncertainty {
    color: #cc6600;
    font-size: 0.95em;
}

.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border-width: 0;
}
</style>
"""

HIGH_CONTRAST_CSS = """
<style>
.dimarray-container {
    font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Menlo, 'Courier New', monospace;
    margin: 10px 0;
    padding: 12px;
    border: 3px solid #000;
    border-radius: 0;
    background-color: #fff;
    color: #000;
}

.dimarray-container.high-contrast {
    background-color: #000;
    color: #fff;
    border-color: #fff;
}

.dimarray-value {
    font-weight: 700;
    color: #000;
    font-size: 1.1em;
}

.dimarray-container.high-contrast .dimarray-value {
    color: #ffff00;
}

.dimarray-unit {
    font-style: normal;
    font-weight: 600;
    color: #000;
    margin-left: 6px;
}

.dimarray-container.high-contrast .dimarray-unit {
    color: #00ffff;
}

.dimarray-shape {
    font-size: 1em;
    color: #333;
    font-weight: 500;
}

.dimarray-container.high-contrast .dimarray-shape {
    color: #ccc;
}

.dimarray-uncertainty {
    color: #cc0000;
    font-size: 1em;
    font-weight: 600;
}

.dimarray-container.high-contrast .dimarray-uncertainty {
    color: #ff6666;
}

.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border-width: 0;
}
</style>
"""


# ==============================================================================
# HTML Formatter
# ==============================================================================


class HTMLFormatter:
    """Formatter for accessible HTML output in Jupyter notebooks."""

    def __init__(self, high_contrast: bool = False, include_css: bool = True):
        """Initialize the HTML formatter.

        Args:
            high_contrast: If True, use high-contrast theme.
            include_css: If True, include CSS styles in output.
        """
        self.high_contrast = high_contrast
        self.include_css = include_css

    def _get_css(self) -> str:
        """Get the appropriate CSS for the current mode."""
        if not self.include_css:
            return ""
        return HIGH_CONTRAST_CSS if self.high_contrast else DEFAULT_CSS

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    def format_dimarray(self, arr: DimArray, max_elements: int = 100) -> str:
        """Format a DimArray as accessible HTML.

        Args:
            arr: The DimArray to format.
            max_elements: Maximum number of elements to show.

        Returns:
            HTML string with ARIA labels and semantic markup.
        """
        from ..core.dimarray import DimArray

        # Build screen reader text
        unit_name = expand_unit_name(arr.unit.symbol)
        sr_text = self._build_screen_reader_text(arr, unit_name)

        # Build visual representation
        visual_html = self._build_visual_html(arr, max_elements)

        # Combine with ARIA
        container_class = "dimarray-container"
        if self.high_contrast:
            container_class += " high-contrast"

        html = f"""{self._get_css()}
<div class="{container_class}" role="img" aria-label="{sr_text}">
    <span class="sr-only">{sr_text}</span>
    {visual_html}
</div>
"""
        return html

    def _build_screen_reader_text(self, arr: DimArray, unit_name: str) -> str:
        """Build screen reader text for ARIA label.

        Args:
            arr: The DimArray.
            unit_name: Full unit name.

        Returns:
            Screen reader text.
        """
        data = arr._data.flatten()
        n_values = len(data)

        if n_values == 0:
            return f"Empty array with unit {unit_name}"
        elif n_values == 1:
            value = float(data[0])
            if arr.has_uncertainty:
                unc = float(arr.uncertainty.flatten()[0])
                return f"{value:.4g} plus or minus {unc:.4g} {unit_name}"
            else:
                return f"{value:.4g} {unit_name}"
        else:
            return f"Array of {n_values} values in {unit_name}, shape {arr.shape}"

    def _build_visual_html(self, arr: DimArray, max_elements: int) -> str:
        """Build visual HTML representation.

        Args:
            arr: The DimArray.
            max_elements: Maximum elements to display.

        Returns:
            HTML string.
        """
        # Format array data
        n_elements = arr._data.size

        if n_elements <= max_elements:
            data_str = np.array2string(
                arr._data,
                precision=4,
                separator=", ",
                suppress_small=True,
                threshold=max_elements,
            )
        else:
            # Show abbreviated version
            data_str = np.array2string(
                arr._data,
                precision=4,
                separator=", ",
                suppress_small=True,
                threshold=10,
                edgeitems=3,
            )

        # Escape HTML
        data_str = self._escape_html(data_str)

        # Build components
        shape_html = f'<span class="dimarray-shape">shape: {arr.shape}</span>'
        value_html = f'<span class="dimarray-value">{data_str}</span>'
        unit_html = f'<span class="dimarray-unit">[{self._escape_html(arr.unit.symbol)}]</span>'

        # Add uncertainty if present
        if arr.has_uncertainty:
            unc_str = np.array2string(
                arr.uncertainty,
                precision=4,
                separator=", ",
                suppress_small=True,
                threshold=max_elements if n_elements <= max_elements else 10,
            )
            unc_str = self._escape_html(unc_str)
            unc_html = f'<br><span class="dimarray-uncertainty">± {unc_str}</span>'
        else:
            unc_html = ""

        return f"{shape_html}<br>{value_html} {unit_html}{unc_html}"

    def format_table(
        self,
        arr: DimArray,
        max_rows: int = 10,
        max_cols: int = 10,
    ) -> str:
        """Format a 2D DimArray as an accessible HTML table.

        Args:
            arr: The DimArray (should be 2D).
            max_rows: Maximum rows to display.
            max_cols: Maximum columns to display.

        Returns:
            HTML table with semantic markup.
        """
        from ..core.dimarray import DimArray

        if arr.ndim != 2:
            # Fall back to regular format for non-2D arrays
            return self.format_dimarray(arr)

        rows, cols = arr.shape
        show_rows = min(rows, max_rows)
        show_cols = min(cols, max_cols)

        # Build caption
        unit_name = expand_unit_name(arr.unit.symbol)
        caption = f"Table of {rows}×{cols} values in {unit_name}"

        # Start table
        html_parts = [self._get_css()]
        html_parts.append(
            f'<table class="dimarray-table" role="table" aria-label="{caption}">'
        )
        html_parts.append(f"<caption>{caption}</caption>")

        # Table body
        html_parts.append("<tbody>")
        for i in range(show_rows):
            html_parts.append("<tr>")
            for j in range(show_cols):
                value = arr._data[i, j]
                html_parts.append(
                    f'<td role="cell"><span class="dimarray-value">{value:.4g}</span></td>'
                )
            if cols > show_cols:
                html_parts.append('<td role="cell">...</td>')
            html_parts.append("</tr>")

        if rows > show_rows:
            html_parts.append("<tr>")
            for _ in range(min(cols, show_cols + 1)):
                html_parts.append('<td role="cell">...</td>')
            html_parts.append("</tr>")

        html_parts.append("</tbody>")
        html_parts.append("</table>")

        return "\n".join(html_parts)


# ==============================================================================
# Convenience Functions
# ==============================================================================


def to_html(
    arr: Any,
    high_contrast: bool = False,
    include_css: bool = True,
    as_table: bool = False,
) -> str:
    """Convert a DimArray to accessible HTML.

    Args:
        arr: The DimArray to format.
        high_contrast: If True, use high-contrast theme.
        include_css: If True, include CSS styles.
        as_table: If True and arr is 2D, format as a table.

    Returns:
        HTML string.

    Examples:
        >>> arr = DimArray([1, 2, 3], units.m)
        >>> html = to_html(arr)
        >>> print(html)  # Shows semantic HTML with ARIA
    """
    formatter = HTMLFormatter(high_contrast=high_contrast, include_css=include_css)

    if as_table and arr.ndim == 2:
        return formatter.format_table(arr)
    else:
        return formatter.format_dimarray(arr)


def add_repr_html_to_dimarray() -> None:
    """Add _repr_html_ method to DimArray for Jupyter notebooks.

    This enables rich HTML display in Jupyter notebooks with accessibility features.

    Examples:
        >>> from dimtensor.accessibility import add_repr_html_to_dimarray
        >>> add_repr_html_to_dimarray()
        >>> # Now DimArrays will display with accessible HTML in Jupyter
    """
    from ..core.dimarray import DimArray
    from .. import config

    def _repr_html_(self: DimArray) -> str:
        """Generate HTML representation for Jupyter notebooks."""
        # Check if accessibility options are set
        high_contrast = False
        if hasattr(config, "accessibility"):
            high_contrast = config.accessibility.high_contrast

        formatter = HTMLFormatter(high_contrast=high_contrast)
        return formatter.format_dimarray(self)

    # Monkey-patch the method onto DimArray
    DimArray._repr_html_ = _repr_html_  # type: ignore


# ==============================================================================
# Plot Accessibility
# ==============================================================================


def add_plot_alt_text(fig: Any, description: str) -> None:
    """Add alt text to a matplotlib figure for accessibility.

    Args:
        fig: Matplotlib figure object.
        description: Alt text description of the plot.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> add_plot_alt_text(fig, "Plot showing quadratic growth")
    """
    # Matplotlib doesn't have built-in alt text, but we can add metadata
    if hasattr(fig, "set_label"):
        fig.set_label(description)

    # Store as metadata for potential export to HTML
    if not hasattr(fig, "_accessibility_metadata"):
        fig._accessibility_metadata = {}  # type: ignore
    fig._accessibility_metadata["alt_text"] = description  # type: ignore

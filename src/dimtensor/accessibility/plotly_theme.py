"""Plotly theme integration for accessibility.

Provides colorblind-safe color schemes and accessible templates for Plotly plots.
"""

from __future__ import annotations

from typing import Any

from .colors import get_palette

# Track if plotly is available
_HAS_PLOTLY = False
try:
    import plotly.graph_objects as go
    import plotly.io as pio

    _HAS_PLOTLY = True
except ImportError:
    pass


def _check_plotly() -> None:
    """Raise ImportError if plotly is not available."""
    if not _HAS_PLOTLY:
        raise ImportError(
            "plotly is required for this feature. " "Install it with: pip install plotly"
        )


# ==============================================================================
# Plotly Template Creation
# ==============================================================================


def create_accessible_template(
    palette: str = "colorblind_safe",
    high_contrast: bool = False,
    name: str = "accessible",
) -> Any:
    """Create an accessible Plotly template.

    Args:
        palette: Color palette name.
        high_contrast: If True, use high-contrast settings.
        name: Name for the template.

    Returns:
        Plotly template object.

    Examples:
        >>> template = create_accessible_template('colorblind_safe')
        >>> fig = go.Figure(data=[...], layout={'template': template})
    """
    _check_plotly()

    colors = get_palette(palette)

    # Create base template
    template = go.layout.Template()

    # Set colorway
    template.layout.colorway = colors

    if high_contrast:
        # High contrast settings
        template.layout.plot_bgcolor = "white"
        template.layout.paper_bgcolor = "white"

        template.layout.xaxis = {
            "showgrid": True,
            "gridcolor": "#000000",
            "gridwidth": 1,
            "linecolor": "#000000",
            "linewidth": 2,
            "showline": True,
            "ticks": "outside",
            "tickcolor": "#000000",
            "tickwidth": 2,
        }

        template.layout.yaxis = {
            "showgrid": True,
            "gridcolor": "#000000",
            "gridwidth": 1,
            "linecolor": "#000000",
            "linewidth": 2,
            "showline": True,
            "ticks": "outside",
            "tickcolor": "#000000",
            "tickwidth": 2,
        }

        template.layout.font = {"color": "#000000", "size": 14}

        # Thicker lines for traces
        template.data.scatter = [
            go.Scatter(
                line={"width": 3},
                marker={"size": 8, "line": {"width": 1, "color": "#000000"}},
            )
        ]

        template.data.bar = [go.Bar(marker={"line": {"width": 2, "color": "#000000"}})]

    else:
        # Standard accessible settings
        template.layout.plot_bgcolor = "#f8f9fa"
        template.layout.paper_bgcolor = "white"

        template.layout.xaxis = {
            "showgrid": True,
            "gridcolor": "#e0e0e0",
            "linecolor": "#333333",
            "showline": True,
        }

        template.layout.yaxis = {
            "showgrid": True,
            "gridcolor": "#e0e0e0",
            "linecolor": "#333333",
            "showline": True,
        }

        template.layout.font = {"color": "#333333", "size": 12}

    return template


def register_accessible_templates() -> None:
    """Register accessible templates with Plotly.

    After calling this, you can use templates like 'accessible',
    'accessible_high_contrast', etc.

    Examples:
        >>> register_accessible_templates()
        >>> import plotly.express as px
        >>> fig = px.line(x=[1, 2, 3], y=[1, 4, 9], template='accessible')
    """
    _check_plotly()

    # Create and register various accessible templates
    templates = {
        "accessible": create_accessible_template("colorblind_safe", False),
        "accessible_high_contrast": create_accessible_template(
            "high_contrast", True
        ),
        "accessible_grayscale": create_accessible_template("grayscale", False),
        "accessible_tol": create_accessible_template("tol_bright", False),
    }

    for name, template in templates.items():
        pio.templates[name] = template


def set_default_template(template: str = "accessible") -> None:
    """Set the default Plotly template.

    Args:
        template: Template name ('accessible', 'accessible_high_contrast', etc.).

    Examples:
        >>> set_default_template('accessible')
        >>> # All subsequent plots will use the accessible template
    """
    _check_plotly()

    # Register templates if not already done
    if "accessible" not in pio.templates:
        register_accessible_templates()

    pio.templates.default = template


# ==============================================================================
# Color Utilities
# ==============================================================================


def get_plotly_colorway(palette: str = "colorblind_safe") -> list[str]:
    """Get a color sequence for Plotly plots.

    Args:
        palette: Palette name.

    Returns:
        List of color hex codes.

    Examples:
        >>> colors = get_plotly_colorway('colorblind_safe')
        >>> fig = go.Figure(layout={'colorway': colors})
    """
    return get_palette(palette)


def get_plotly_colorscale(name: str = "colorblind_safe_sequential") -> list[list[Any]]:
    """Get a colorscale for Plotly heatmaps and continuous plots.

    Args:
        name: Colorscale name ('colorblind_safe_sequential',
              'colorblind_safe_diverging').

    Returns:
        Plotly colorscale (list of [position, color] pairs).

    Examples:
        >>> colorscale = get_plotly_colorscale('colorblind_safe_sequential')
        >>> fig = go.Figure(data=go.Heatmap(z=data, colorscale=colorscale))
    """
    from .colors import COLORBLIND_SAFE_DIVERGING, COLORBLIND_SAFE_SEQUENTIAL

    if name == "colorblind_safe_sequential":
        colors = COLORBLIND_SAFE_SEQUENTIAL
    elif name == "colorblind_safe_diverging":
        colors = COLORBLIND_SAFE_DIVERGING
    else:
        raise ValueError(
            f"Unknown colorscale '{name}'. "
            "Use 'colorblind_safe_sequential' or 'colorblind_safe_diverging'."
        )

    # Convert to Plotly colorscale format
    n = len(colors)
    colorscale = [[i / (n - 1), color] for i, color in enumerate(colors)]
    return colorscale


# ==============================================================================
# Figure Enhancement
# ==============================================================================


def make_figure_accessible(
    fig: Any,
    palette: str = "colorblind_safe",
    high_contrast: bool = False,
    add_alt_text: str | None = None,
) -> Any:
    """Enhance a Plotly figure for accessibility.

    Args:
        fig: Plotly Figure object.
        palette: Color palette to apply.
        high_contrast: If True, apply high-contrast settings.
        add_alt_text: Optional alt text description for screen readers.

    Returns:
        Modified Figure object.

    Examples:
        >>> fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 4, 9])])
        >>> fig = make_figure_accessible(fig, palette='colorblind_safe')
    """
    _check_plotly()

    # Apply color palette
    colors = get_palette(palette)
    fig.update_layout(colorway=colors)

    if high_contrast:
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font={"color": "#000000", "size": 14},
            xaxis={
                "showgrid": True,
                "gridcolor": "#000000",
                "gridwidth": 1,
                "linecolor": "#000000",
                "linewidth": 2,
                "showline": True,
            },
            yaxis={
                "showgrid": True,
                "gridcolor": "#000000",
                "gridwidth": 1,
                "linecolor": "#000000",
                "linewidth": 2,
                "showline": True,
            },
        )

        # Update trace styles
        fig.update_traces(
            line={"width": 3},
            marker={"size": 8, "line": {"width": 1, "color": "#000000"}},
        )

    # Add alt text as metadata (not natively supported by Plotly, but stored)
    if add_alt_text:
        if not hasattr(fig, "_accessibility_metadata"):
            fig._accessibility_metadata = {}  # type: ignore
        fig._accessibility_metadata["alt_text"] = add_alt_text  # type: ignore

    return fig


# ==============================================================================
# Marker and Line Style Cycles
# ==============================================================================

PLOTLY_MARKER_SYMBOLS = [
    "circle",
    "square",
    "diamond",
    "cross",
    "x",
    "triangle-up",
    "triangle-down",
    "pentagon",
    "hexagon",
    "star",
]

PLOTLY_LINE_DASH = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]


def get_marker_symbols() -> list[str]:
    """Get a list of distinguishable marker symbols for Plotly.

    Returns:
        List of Plotly marker symbol names.

    Examples:
        >>> symbols = get_marker_symbols()
        >>> for i, symbol in enumerate(symbols[:3]):
        ...     fig.add_trace(go.Scatter(x=[i], y=[i], mode='markers',
        ...                               marker={'symbol': symbol, 'size': 12}))
    """
    return PLOTLY_MARKER_SYMBOLS


def get_line_dash_styles() -> list[str]:
    """Get a list of distinguishable line dash styles for Plotly.

    Returns:
        List of Plotly line dash style names.

    Examples:
        >>> dashes = get_line_dash_styles()
        >>> for i, dash in enumerate(dashes[:3]):
        ...     fig.add_trace(go.Scatter(x=[0, 1], y=[i, i],
        ...                               line={'dash': dash}))
    """
    return PLOTLY_LINE_DASH


def create_accessible_scatter(
    x: list[Any],
    y: list[list[Any]],
    labels: list[str] | None = None,
    palette: str = "colorblind_safe",
    use_markers: bool = True,
    high_contrast: bool = False,
) -> Any:
    """Create an accessible scatter plot with multiple series.

    Uses colors, markers, and line styles for maximum distinguishability.

    Args:
        x: X data (shared by all series).
        y: List of Y data arrays for each series.
        labels: Optional labels for each series.
        palette: Color palette to use.
        use_markers: If True, add markers to lines.
        high_contrast: If True, use high-contrast settings.

    Returns:
        Plotly Figure object.

    Examples:
        >>> x = [1, 2, 3, 4, 5]
        >>> y = [[1, 4, 9, 16, 25], [1, 2, 3, 4, 5]]
        >>> fig = create_accessible_scatter(x, y, labels=['Quadratic', 'Linear'])
        >>> fig.show()
    """
    _check_plotly()

    colors = get_palette(palette)
    markers = get_marker_symbols()
    dashes = get_line_dash_styles()

    fig = go.Figure()

    for i, y_data in enumerate(y):
        color = colors[i % len(colors)]
        marker_symbol = markers[i % len(markers)]
        line_dash = dashes[i % len(dashes)]

        label = labels[i] if labels and i < len(labels) else f"Series {i+1}"

        trace_kwargs = {
            "x": x,
            "y": y_data,
            "name": label,
            "mode": "lines+markers" if use_markers else "lines",
            "line": {"color": color, "width": 3 if high_contrast else 2, "dash": line_dash},
        }

        if use_markers:
            trace_kwargs["marker"] = {"symbol": marker_symbol, "size": 8, "color": color}

        fig.add_trace(go.Scatter(**trace_kwargs))

    # Apply accessible layout
    fig = make_figure_accessible(fig, palette=palette, high_contrast=high_contrast)

    return fig

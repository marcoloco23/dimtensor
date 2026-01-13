"""Accessibility features for dimtensor.

Provides colorblind-safe palettes, screen reader support, and high-contrast
display modes to make dimtensor accessible to all users.

Modules:
    colors: Colorblind-safe color palettes and CVD simulation
    formatters: Screen reader and high-contrast text formatters
    html: Semantic HTML with ARIA labels for Jupyter notebooks
    matplotlib_theme: Matplotlib accessibility integration
    plotly_theme: Plotly accessibility integration

Examples:
    >>> from dimtensor import accessibility
    >>> # Apply colorblind-safe theme to matplotlib
    >>> accessibility.apply_accessible_theme('colorblind_safe')
    >>>
    >>> # Format array for screen readers
    >>> arr = DimArray([3.14], units.m)
    >>> print(accessibility.format_for_screen_reader(arr))
    '3.14 meters'
    >>>
    >>> # Get accessible color palette
    >>> colors = accessibility.get_palette('colorblind_safe')
"""

from __future__ import annotations

# Color palettes and utilities
from .colors import (
    COLORBLIND_SAFE_DIVERGING,
    COLORBLIND_SAFE_QUALITATIVE,
    COLORBLIND_SAFE_SEQUENTIAL,
    GRAYSCALE_PALETTE,
    HIGH_CONTRAST_PALETTE,
    TOL_BRIGHT,
    TOL_HIGH_CONTRAST,
    TOL_MUTED,
    WONG_PALETTE,
    check_distinguishability,
    check_wcag_contrast,
    color_distance,
    contrast_ratio,
    get_palette,
    hex_to_rgb,
    relative_luminance,
    rgb_to_hex,
    simulate_cvd,
    simulate_palette_cvd,
    suggest_palette,
)

# Text formatters
from .formatters import (
    HighContrastFormatter,
    ScreenReaderFormatter,
    expand_unit_name,
    format_for_screen_reader,
    format_high_contrast,
)

# HTML/Jupyter support
from .html import HTMLFormatter, add_plot_alt_text, add_repr_html_to_dimarray, to_html

# Matplotlib integration
try:
    from .matplotlib_theme import (
        ACCESSIBLE_LINE_STYLES,
        ACCESSIBLE_MARKERS,
        accessible_plot,
        apply_accessible_theme,
        get_accessible_colormap,
        get_linestyle_cycle,
        get_marker_cycle,
        plot_with_patterns,
        reset_matplotlib_theme,
    )

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

# Plotly integration
try:
    from .plotly_theme import (
        PLOTLY_LINE_DASH,
        PLOTLY_MARKER_SYMBOLS,
        create_accessible_scatter,
        create_accessible_template,
        get_line_dash_styles,
        get_marker_symbols,
        get_plotly_colorscale,
        get_plotly_colorway,
        make_figure_accessible,
        register_accessible_templates,
        set_default_template,
    )

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


__all__ = [
    # Color palettes
    "COLORBLIND_SAFE_QUALITATIVE",
    "COLORBLIND_SAFE_SEQUENTIAL",
    "COLORBLIND_SAFE_DIVERGING",
    "HIGH_CONTRAST_PALETTE",
    "GRAYSCALE_PALETTE",
    "WONG_PALETTE",
    "TOL_BRIGHT",
    "TOL_MUTED",
    "TOL_HIGH_CONTRAST",
    # Color functions
    "get_palette",
    "simulate_cvd",
    "simulate_palette_cvd",
    "check_distinguishability",
    "suggest_palette",
    "hex_to_rgb",
    "rgb_to_hex",
    "color_distance",
    "contrast_ratio",
    "relative_luminance",
    "check_wcag_contrast",
    # Formatters
    "ScreenReaderFormatter",
    "HighContrastFormatter",
    "format_for_screen_reader",
    "format_high_contrast",
    "expand_unit_name",
    # HTML/Jupyter
    "HTMLFormatter",
    "to_html",
    "add_repr_html_to_dimarray",
    "add_plot_alt_text",
]

# Add matplotlib exports if available
if _HAS_MATPLOTLIB:
    __all__.extend(
        [
            "apply_accessible_theme",
            "reset_matplotlib_theme",
            "get_accessible_colormap",
            "accessible_plot",
            "ACCESSIBLE_MARKERS",
            "ACCESSIBLE_LINE_STYLES",
            "get_marker_cycle",
            "get_linestyle_cycle",
            "plot_with_patterns",
        ]
    )

# Add plotly exports if available
if _HAS_PLOTLY:
    __all__.extend(
        [
            "create_accessible_template",
            "register_accessible_templates",
            "set_default_template",
            "get_plotly_colorway",
            "get_plotly_colorscale",
            "make_figure_accessible",
            "PLOTLY_MARKER_SYMBOLS",
            "PLOTLY_LINE_DASH",
            "get_marker_symbols",
            "get_line_dash_styles",
            "create_accessible_scatter",
        ]
    )

"""Matplotlib theme integration for accessibility.

Provides colorblind-safe themes and high-contrast options for matplotlib plots.
"""

from __future__ import annotations

from typing import Any

from .colors import get_palette

# Track if matplotlib is available
_HAS_MATPLOTLIB = False
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from cycler import cycler

    _HAS_MATPLOTLIB = True
except ImportError:
    pass


def _check_matplotlib() -> None:
    """Raise ImportError if matplotlib is not available."""
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for this feature. "
            "Install it with: pip install matplotlib"
        )


# ==============================================================================
# Theme Application
# ==============================================================================


def apply_accessible_theme(
    palette: str = "colorblind_safe",
    high_contrast: bool = False,
    larger_fonts: bool = False,
) -> None:
    """Apply an accessible theme to matplotlib.

    Args:
        palette: Palette name ('colorblind_safe', 'high_contrast', 'grayscale', etc.).
        high_contrast: If True, increase line widths and marker sizes.
        larger_fonts: If True, increase font sizes for better readability.

    Examples:
        >>> from dimtensor.accessibility import apply_accessible_theme
        >>> apply_accessible_theme('colorblind_safe')
        >>> plt.plot([1, 2, 3], [1, 4, 9])
        >>> plt.show()  # Uses colorblind-safe colors
    """
    _check_matplotlib()

    # Get color palette
    colors = get_palette(palette)

    # Set color cycle
    mpl.rcParams["axes.prop_cycle"] = cycler(color=colors)

    if high_contrast:
        # Increase line widths
        mpl.rcParams["lines.linewidth"] = 2.5
        mpl.rcParams["lines.markersize"] = 8
        mpl.rcParams["patch.linewidth"] = 2.0
        mpl.rcParams["axes.linewidth"] = 1.5
        mpl.rcParams["grid.linewidth"] = 1.2

        # Use solid lines (avoid dashed/dotted which are hard to see)
        mpl.rcParams["lines.linestyle"] = "-"

        # Increase contrast for axes
        mpl.rcParams["axes.edgecolor"] = "black"
        mpl.rcParams["axes.labelcolor"] = "black"
        mpl.rcParams["xtick.color"] = "black"
        mpl.rcParams["ytick.color"] = "black"
        mpl.rcParams["text.color"] = "black"

        # White background
        mpl.rcParams["figure.facecolor"] = "white"
        mpl.rcParams["axes.facecolor"] = "white"
        mpl.rcParams["savefig.facecolor"] = "white"

    if larger_fonts:
        # Increase font sizes
        mpl.rcParams["font.size"] = 12
        mpl.rcParams["axes.titlesize"] = 14
        mpl.rcParams["axes.labelsize"] = 12
        mpl.rcParams["xtick.labelsize"] = 11
        mpl.rcParams["ytick.labelsize"] = 11
        mpl.rcParams["legend.fontsize"] = 11
        mpl.rcParams["figure.titlesize"] = 16


def reset_matplotlib_theme() -> None:
    """Reset matplotlib to default theme.

    Examples:
        >>> reset_matplotlib_theme()
    """
    _check_matplotlib()
    mpl.rcdefaults()


def get_accessible_colormap(
    name: str = "colorblind_safe_sequential", n_colors: int = 256
) -> Any:
    """Get an accessible colormap for heatmaps and continuous data.

    Args:
        name: Colormap type ('colorblind_safe_sequential', 'colorblind_safe_diverging').
        n_colors: Number of colors in the colormap.

    Returns:
        Matplotlib colormap object.

    Examples:
        >>> cmap = get_accessible_colormap('colorblind_safe_sequential')
        >>> plt.imshow(data, cmap=cmap)
    """
    _check_matplotlib()
    from matplotlib.colors import LinearSegmentedColormap

    from .colors import COLORBLIND_SAFE_DIVERGING, COLORBLIND_SAFE_SEQUENTIAL

    if name == "colorblind_safe_sequential":
        colors = COLORBLIND_SAFE_SEQUENTIAL
    elif name == "colorblind_safe_diverging":
        colors = COLORBLIND_SAFE_DIVERGING
    else:
        raise ValueError(
            f"Unknown colormap '{name}'. "
            "Use 'colorblind_safe_sequential' or 'colorblind_safe_diverging'."
        )

    # Create colormap from color list
    cmap = LinearSegmentedColormap.from_list(name, colors, N=n_colors)
    return cmap


# ==============================================================================
# Context Manager
# ==============================================================================


class accessible_plot:
    """Context manager for temporarily applying accessible theme.

    Examples:
        >>> from dimtensor.accessibility import accessible_plot
        >>> with accessible_plot('colorblind_safe'):
        ...     plt.plot([1, 2, 3], [1, 4, 9])
        ...     plt.show()
    """

    def __init__(
        self,
        palette: str = "colorblind_safe",
        high_contrast: bool = False,
        larger_fonts: bool = False,
    ):
        """Initialize the context manager.

        Args:
            palette: Palette name.
            high_contrast: If True, use high-contrast settings.
            larger_fonts: If True, use larger fonts.
        """
        self.palette = palette
        self.high_contrast = high_contrast
        self.larger_fonts = larger_fonts
        self.old_rc = None

    def __enter__(self) -> None:
        """Enter the context."""
        _check_matplotlib()
        # Save current rcParams
        self.old_rc = mpl.rcParams.copy()
        # Apply accessible theme
        apply_accessible_theme(self.palette, self.high_contrast, self.larger_fonts)

    def __exit__(self, *args: Any) -> None:
        """Exit the context and restore original settings."""
        if self.old_rc is not None:
            mpl.rcParams.update(self.old_rc)


# ==============================================================================
# Marker Styles for Pattern Differentiation
# ==============================================================================

# Use different markers in addition to colors for maximum accessibility
ACCESSIBLE_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "p", "h"]

# Line styles for grayscale plots
ACCESSIBLE_LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]


def get_marker_cycle() -> list[str]:
    """Get a list of distinguishable markers for scatter plots.

    Returns:
        List of matplotlib marker symbols.

    Examples:
        >>> markers = get_marker_cycle()
        >>> for i, marker in enumerate(markers[:5]):
        ...     plt.scatter(i, i, marker=marker, s=100)
    """
    return ACCESSIBLE_MARKERS


def get_linestyle_cycle() -> list[str | tuple[int, tuple[int, ...]]]:
    """Get a list of distinguishable line styles.

    Returns:
        List of matplotlib line styles.

    Examples:
        >>> linestyles = get_linestyle_cycle()
        >>> for i, ls in enumerate(linestyles[:5]):
        ...     plt.plot([0, 1], [i, i], linestyle=ls)
    """
    return ACCESSIBLE_LINE_STYLES


def plot_with_patterns(
    x: list[Any],
    y: list[Any],
    labels: list[str] | None = None,
    palette: str = "colorblind_safe",
    **kwargs: Any,
) -> None:
    """Plot multiple lines with both colors and patterns for accessibility.

    Uses different colors, line styles, and markers to ensure maximum
    distinguishability, including for colorblind users and grayscale printing.

    Args:
        x: X data (single array or list of arrays).
        y: List of Y data arrays.
        labels: Optional labels for each line.
        palette: Color palette to use.
        **kwargs: Additional keyword arguments passed to plt.plot.

    Examples:
        >>> x = [1, 2, 3, 4, 5]
        >>> y1 = [1, 4, 9, 16, 25]
        >>> y2 = [1, 2, 3, 4, 5]
        >>> plot_with_patterns(x, [y1, y2], labels=['Quadratic', 'Linear'])
    """
    _check_matplotlib()

    colors = get_palette(palette)
    markers = get_marker_cycle()
    linestyles = get_linestyle_cycle()

    n_lines = len(y)

    for i in range(n_lines):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]

        label = labels[i] if labels and i < len(labels) else None

        plt.plot(
            x,
            y[i],
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=6,
            markevery=max(1, len(x) // 10),  # Don't overcrowd with markers
            label=label,
            **kwargs,
        )

    if labels:
        plt.legend()

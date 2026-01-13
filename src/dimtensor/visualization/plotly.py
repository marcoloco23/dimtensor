"""Plotly integration for DimArray.

Provides wrapper functions that handle DimArray objects and automatically
add unit labels to axes.

Usage:
    >>> from dimtensor.visualization.plotly import line, scatter
    >>> fig = line(time, distance)  # Auto-labels with units
    >>> fig.show()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..core.dimarray import DimArray
    from ..core.units import Unit

# Track if plotly is available
_HAS_PLOTLY = False
try:
    import plotly.express as px
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except ImportError:
    pass


def _check_plotly() -> None:
    """Raise ImportError if plotly is not available."""
    if not _HAS_PLOTLY:
        raise ImportError(
            "plotly is required for this visualization. "
            "Install it with: pip install plotly"
        )


def _get_colorway(palette: str | None) -> list[str] | None:
    """Get colorway for a palette, respecting config settings.

    Args:
        palette: Explicit palette name, or None to use config settings.

    Returns:
        List of color hex codes, or None if using default.
    """
    # Determine which palette to use
    palette_to_use = palette
    if palette_to_use is None:
        try:
            from .. import config

            if (
                hasattr(config, "accessibility")
                and config.accessibility.color_palette != "default"
            ):
                palette_to_use = config.accessibility.color_palette
        except ImportError:
            pass

    # Get the colorway
    if palette_to_use and palette_to_use != "default":
        try:
            from ..accessibility.plotly_theme import get_plotly_colorway

            return get_plotly_colorway(palette_to_use)
        except ImportError:
            pass  # Accessibility module not available

    return None


def _extract_data_and_label(
    arr: Any, target_unit: Unit | None = None
) -> tuple[np.ndarray[Any, Any], str]:
    """Extract numpy data and unit label from a value.

    Args:
        arr: Value to extract (DimArray or array-like).
        target_unit: Optional unit to convert to.

    Returns:
        Tuple of (numpy array, unit label string).
    """
    from ..core.dimarray import DimArray

    if isinstance(arr, DimArray):
        if target_unit is not None:
            arr = arr.to(target_unit)
        label = f"[{arr.unit.symbol}]" if not arr.is_dimensionless else ""
        return arr._data, label
    return np.asarray(arr), ""


def line(
    x: Any,
    y: Any,
    *,
    x_unit: Unit | None = None,
    y_unit: Unit | None = None,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    palette: str | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Create a line plot with automatic unit labels.

    Args:
        x: X-axis data (DimArray or array-like).
        y: Y-axis data (DimArray or array-like).
        x_unit: Optional unit to convert x data to.
        y_unit: Optional unit to convert y data to.
        title: Plot title.
        x_title: X-axis title (units appended automatically).
        y_title: Y-axis title (units appended automatically).
        palette: Color palette to use. If None, uses config.accessibility.color_palette.
        **kwargs: Additional arguments passed to px.line.

    Returns:
        Plotly Figure object.

    Example:
        >>> time = DimArray([0, 1, 2, 3], units.s)
        >>> distance = DimArray([0, 10, 40, 90], units.m)
        >>> fig = line(time, distance, title="Motion")
        >>> fig.show()
    """
    _check_plotly()

    x_data, x_label = _extract_data_and_label(x, x_unit)
    y_data, y_label = _extract_data_and_label(y, y_unit)

    fig = px.line(x=x_data, y=y_data, **kwargs)

    # Build axis titles
    x_axis_title = x_title or ""
    y_axis_title = y_title or ""

    if x_label:
        x_axis_title = f"{x_axis_title} {x_label}".strip()
    if y_label:
        y_axis_title = f"{y_axis_title} {y_label}".strip()

    # Apply palette
    colorway = _get_colorway(palette)
    layout_update = {
        "title": title,
        "xaxis_title": x_axis_title if x_axis_title else None,
        "yaxis_title": y_axis_title if y_axis_title else None,
    }
    if colorway:
        layout_update["colorway"] = colorway

    fig.update_layout(**layout_update)

    return fig


def scatter(
    x: Any,
    y: Any,
    *,
    x_unit: Unit | None = None,
    y_unit: Unit | None = None,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Create a scatter plot with automatic unit labels.

    Args:
        x: X-axis data (DimArray or array-like).
        y: Y-axis data (DimArray or array-like).
        x_unit: Optional unit to convert x data to.
        y_unit: Optional unit to convert y data to.
        title: Plot title.
        x_title: X-axis title (units appended automatically).
        y_title: Y-axis title (units appended automatically).
        **kwargs: Additional arguments passed to px.scatter.

    Returns:
        Plotly Figure object.

    Example:
        >>> mass = DimArray([1, 2, 3, 4], units.kg)
        >>> force = DimArray([9.8, 19.6, 29.4, 39.2], units.N)
        >>> fig = scatter(mass, force, title="Force vs Mass")
        >>> fig.show()
    """
    _check_plotly()

    x_data, x_label = _extract_data_and_label(x, x_unit)
    y_data, y_label = _extract_data_and_label(y, y_unit)

    fig = px.scatter(x=x_data, y=y_data, **kwargs)

    x_axis_title = x_title or ""
    y_axis_title = y_title or ""

    if x_label:
        x_axis_title = f"{x_axis_title} {x_label}".strip()
    if y_label:
        y_axis_title = f"{y_axis_title} {y_label}".strip()

    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title if x_axis_title else None,
        yaxis_title=y_axis_title if y_axis_title else None,
    )

    return fig


def bar(
    x: Any,
    y: Any,
    *,
    x_unit: Unit | None = None,
    y_unit: Unit | None = None,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Create a bar chart with automatic unit labels.

    Args:
        x: X-axis positions/categories (DimArray or array-like).
        y: Bar heights (DimArray or array-like).
        x_unit: Optional unit to convert x data to.
        y_unit: Optional unit to convert y data to.
        title: Plot title.
        x_title: X-axis title (units appended automatically).
        y_title: Y-axis title (units appended automatically).
        **kwargs: Additional arguments passed to px.bar.

    Returns:
        Plotly Figure object.

    Example:
        >>> categories = ["A", "B", "C", "D"]
        >>> values = DimArray([10, 25, 15, 30], units.m)
        >>> fig = bar(categories, values, title="Heights")
        >>> fig.show()
    """
    _check_plotly()

    x_data, x_label = _extract_data_and_label(x, x_unit)
    y_data, y_label = _extract_data_and_label(y, y_unit)

    fig = px.bar(x=x_data, y=y_data, **kwargs)

    x_axis_title = x_title or ""
    y_axis_title = y_title or ""

    if x_label:
        x_axis_title = f"{x_axis_title} {x_label}".strip()
    if y_label:
        y_axis_title = f"{y_axis_title} {y_label}".strip()

    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title if x_axis_title else None,
        yaxis_title=y_axis_title if y_axis_title else None,
    )

    return fig


def histogram(
    x: Any,
    *,
    x_unit: Unit | None = None,
    title: str | None = None,
    x_title: str | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Create a histogram with automatic unit labels.

    Args:
        x: Data to histogram (DimArray or array-like).
        x_unit: Optional unit to convert x data to.
        title: Plot title.
        x_title: X-axis title (units appended automatically).
        **kwargs: Additional arguments passed to px.histogram.

    Returns:
        Plotly Figure object.

    Example:
        >>> measurements = DimArray([1.1, 1.2, 0.9, 1.0, 1.1], units.m)
        >>> fig = histogram(measurements, title="Measurements")
        >>> fig.show()
    """
    _check_plotly()

    x_data, x_label = _extract_data_and_label(x, x_unit)

    fig = px.histogram(x=x_data, **kwargs)

    x_axis_title = x_title or ""
    if x_label:
        x_axis_title = f"{x_axis_title} {x_label}".strip()

    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title if x_axis_title else None,
    )

    return fig


def scatter_with_errors(
    x: Any,
    y: Any,
    *,
    xerr: Any | None = None,
    yerr: Any | None = None,
    x_unit: Unit | None = None,
    y_unit: Unit | None = None,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Create a scatter plot with error bars and automatic unit labels.

    If y is a DimArray with uncertainty and yerr is not provided,
    uses the uncertainty from y. Same for x and xerr.

    Args:
        x: X-axis data (DimArray or array-like).
        y: Y-axis data (DimArray or array-like).
        xerr: X error bars (optional).
        yerr: Y error bars (optional).
        x_unit: Optional unit to convert x data to.
        y_unit: Optional unit to convert y data to.
        title: Plot title.
        x_title: X-axis title (units appended automatically).
        y_title: Y-axis title (units appended automatically).
        **kwargs: Additional arguments passed to go.Scatter.

    Returns:
        Plotly Figure object.

    Example:
        >>> x = DimArray([1, 2, 3], units.s)
        >>> y = DimArray([10, 20, 30], units.m, uncertainty=[0.5, 0.5, 0.5])
        >>> fig = scatter_with_errors(x, y, title="With Errors")
        >>> fig.show()
    """
    _check_plotly()
    from ..core.dimarray import DimArray

    x_data, x_label = _extract_data_and_label(x, x_unit)
    y_data, y_label = _extract_data_and_label(y, y_unit)

    # Auto-extract uncertainty if not provided
    if yerr is None and isinstance(y, DimArray) and y.has_uncertainty:
        if y_unit is not None:
            yerr = y.to(y_unit).uncertainty
        else:
            yerr = y.uncertainty
    elif isinstance(yerr, DimArray):
        yerr, _ = _extract_data_and_label(yerr, y_unit)

    if xerr is None and isinstance(x, DimArray) and x.has_uncertainty:
        if x_unit is not None:
            xerr = x.to(x_unit).uncertainty
        else:
            xerr = x.uncertainty
    elif isinstance(xerr, DimArray):
        xerr, _ = _extract_data_and_label(xerr, x_unit)

    # Build error_x and error_y dicts
    error_x = None
    error_y = None

    if xerr is not None:
        error_x = {"type": "data", "array": xerr, "visible": True}
    if yerr is not None:
        error_y = {"type": "data", "array": yerr, "visible": True}

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode="markers",
            error_x=error_x,
            error_y=error_y,
            **kwargs,
        )
    )

    x_axis_title = x_title or ""
    y_axis_title = y_title or ""

    if x_label:
        x_axis_title = f"{x_axis_title} {x_label}".strip()
    if y_label:
        y_axis_title = f"{y_axis_title} {y_label}".strip()

    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title if x_axis_title else None,
        yaxis_title=y_axis_title if y_axis_title else None,
    )

    return fig

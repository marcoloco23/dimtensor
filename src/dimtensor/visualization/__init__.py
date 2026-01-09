"""Visualization support for dimtensor.

This module provides integration with plotting libraries for DimArray.

Matplotlib Integration:
    >>> from dimtensor.visualization import setup_matplotlib
    >>> setup_matplotlib()  # Enable automatic unit labels

    Or use wrapper functions:
    >>> from dimtensor.visualization import plot, scatter, bar, hist
    >>> plot(time, distance)  # Automatic axis labels

Plotly Integration:
    >>> from dimtensor.visualization import plotly
    >>> fig = plotly.line(time, distance)  # Automatic axis labels
    >>> fig.show()
"""

from __future__ import annotations

# Re-export matplotlib functions
from .matplotlib import (
    setup_matplotlib,
    plot,
    scatter,
    bar,
    hist,
    errorbar,
)

# Import plotly module (available as visualization.plotly)
from . import plotly

__all__ = [
    # Setup
    "setup_matplotlib",
    # Matplotlib wrapper functions
    "plot",
    "scatter",
    "bar",
    "hist",
    "errorbar",
    # Plotly module
    "plotly",
]

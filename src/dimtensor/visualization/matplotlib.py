"""Matplotlib integration for DimArray.

Provides automatic axis labeling with units when plotting DimArray objects.

Usage:
    >>> from dimtensor.visualization import setup_matplotlib
    >>> setup_matplotlib()  # Enable automatic unit labels
    >>> plt.plot(time, distance)  # Labels added automatically

Or use the wrapper functions:
    >>> from dimtensor.visualization import plot, scatter
    >>> plot(time, distance)  # Works with DimArray
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..core.dimarray import DimArray
    from ..core.units import Unit

# Track if matplotlib is available
_HAS_MATPLOTLIB = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.units as munits
    from matplotlib.axis import Axis

    _HAS_MATPLOTLIB = True
except ImportError:
    pass


def _check_matplotlib() -> None:
    """Raise ImportError if matplotlib is not available."""
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


class DimArrayConverter:
    """Matplotlib unit converter for DimArray objects.

    Implements matplotlib.units.ConversionInterface to enable automatic
    unit handling when plotting DimArray objects.
    """

    @staticmethod
    def convert(value: Any, unit: Unit | None, axis: Axis) -> Any:
        """Convert DimArray to numpy array for plotting.

        Args:
            value: Value to convert (DimArray or array-like).
            unit: Target unit for axis (from axisinfo).
            axis: The matplotlib axis.

        Returns:
            Numpy array of values, converted to target unit if specified.
        """
        from ..core.dimarray import DimArray

        if isinstance(value, DimArray):
            if unit is not None and hasattr(unit, "symbol"):
                # Convert to the axis unit
                try:
                    return value.to(unit)._data
                except Exception:
                    # If conversion fails, just return raw data
                    return value._data
            return value._data
        elif hasattr(value, "__iter__"):
            # Handle sequences of DimArrays
            result = []
            for v in value:
                if isinstance(v, DimArray):
                    if unit is not None and hasattr(unit, "symbol"):
                        try:
                            result.append(v.to(unit)._data)
                        except Exception:
                            result.append(v._data)
                    else:
                        result.append(v._data)
                else:
                    result.append(v)
            return np.array(result)
        return value

    @staticmethod
    def axisinfo(unit: Unit | None, axis: Axis) -> munits.AxisInfo | None:
        """Return axis labeling information.

        Args:
            unit: The unit for this axis.
            axis: The matplotlib axis.

        Returns:
            AxisInfo with unit label, or None if no unit.
        """
        if unit is None:
            return None
        if hasattr(unit, "symbol"):
            label = f"[{unit.symbol}]"
            return munits.AxisInfo(label=label)  # type: ignore[no-untyped-call]
        return None

    @staticmethod
    def default_units(x: Any, axis: Axis) -> Unit | None:
        """Determine the default unit for the axis.

        Args:
            x: The data being plotted.
            axis: The matplotlib axis.

        Returns:
            The Unit from the first DimArray found, or None.
        """
        from ..core.dimarray import DimArray

        if isinstance(x, DimArray):
            return x._unit
        elif hasattr(x, "__iter__"):
            for item in x:
                if isinstance(item, DimArray):
                    return item._unit
        return None


def setup_matplotlib(enable: bool = True) -> None:
    """Enable or disable matplotlib integration for DimArray.

    When enabled, DimArray objects can be passed directly to matplotlib
    plotting functions, and axes will be automatically labeled with units.

    Args:
        enable: If True (default), enable integration. If False, disable.

    Example:
        >>> setup_matplotlib()
        >>> time = DimArray([0, 1, 2, 3], units.s)
        >>> distance = DimArray([0, 10, 40, 90], units.m)
        >>> plt.plot(time, distance)  # Axes labeled with [s] and [m]
        >>> plt.show()
    """
    _check_matplotlib()

    from ..core.dimarray import DimArray

    if enable:
        munits.registry[DimArray] = DimArrayConverter()
    else:
        if DimArray in munits.registry:
            del munits.registry[DimArray]


# =============================================================================
# Wrapper functions for convenient plotting
# =============================================================================


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


def plot(
    x: Any,
    y: Any,
    *args: Any,
    x_unit: Unit | None = None,
    y_unit: Unit | None = None,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot y vs x with automatic unit labels.

    Wrapper around matplotlib.pyplot.plot that handles DimArray objects
    and automatically labels axes with units.

    Args:
        x: X-axis data (DimArray or array-like).
        y: Y-axis data (DimArray or array-like).
        *args: Additional positional arguments passed to plt.plot.
        x_unit: Optional unit to convert x data to.
        y_unit: Optional unit to convert y data to.
        ax: Matplotlib axes to plot on. If None, uses current axes.
        **kwargs: Keyword arguments passed to plt.plot.

    Returns:
        List of Line2D objects (from plt.plot).

    Example:
        >>> time = DimArray([0, 1, 2, 3], units.s)
        >>> distance = DimArray([0, 10, 40, 90], units.m)
        >>> plot(time, distance)  # Labels: x=[s], y=[m]
        >>> plot(time, distance, y_unit=units.km)  # y in kilometers
    """
    _check_matplotlib()

    if ax is None:
        ax = plt.gca()

    x_data, x_label = _extract_data_and_label(x, x_unit)
    y_data, y_label = _extract_data_and_label(y, y_unit)

    result = ax.plot(x_data, y_data, *args, **kwargs)

    # Add unit labels if not already set
    if x_label and not ax.get_xlabel():
        ax.set_xlabel(x_label)
    if y_label and not ax.get_ylabel():
        ax.set_ylabel(y_label)

    return result


def scatter(
    x: Any,
    y: Any,
    *args: Any,
    x_unit: Unit | None = None,
    y_unit: Unit | None = None,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Create a scatter plot with automatic unit labels.

    Wrapper around matplotlib.pyplot.scatter that handles DimArray objects
    and automatically labels axes with units.

    Args:
        x: X-axis data (DimArray or array-like).
        y: Y-axis data (DimArray or array-like).
        *args: Additional positional arguments passed to plt.scatter.
        x_unit: Optional unit to convert x data to.
        y_unit: Optional unit to convert y data to.
        ax: Matplotlib axes to plot on. If None, uses current axes.
        **kwargs: Keyword arguments passed to plt.scatter.

    Returns:
        PathCollection object (from plt.scatter).

    Example:
        >>> mass = DimArray([1, 2, 3, 4], units.kg)
        >>> force = DimArray([9.8, 19.6, 29.4, 39.2], units.N)
        >>> scatter(mass, force)
    """
    _check_matplotlib()

    if ax is None:
        ax = plt.gca()

    x_data, x_label = _extract_data_and_label(x, x_unit)
    y_data, y_label = _extract_data_and_label(y, y_unit)

    result = ax.scatter(x_data, y_data, *args, **kwargs)

    if x_label and not ax.get_xlabel():
        ax.set_xlabel(x_label)
    if y_label and not ax.get_ylabel():
        ax.set_ylabel(y_label)

    return result


def bar(
    x: Any,
    height: Any,
    *args: Any,
    x_unit: Unit | None = None,
    height_unit: Unit | None = None,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Create a bar chart with automatic unit labels.

    Wrapper around matplotlib.pyplot.bar that handles DimArray objects
    and automatically labels axes with units.

    Args:
        x: X-axis positions (DimArray or array-like).
        height: Bar heights (DimArray or array-like).
        *args: Additional positional arguments passed to plt.bar.
        x_unit: Optional unit to convert x data to.
        height_unit: Optional unit to convert height data to.
        ax: Matplotlib axes to plot on. If None, uses current axes.
        **kwargs: Keyword arguments passed to plt.bar.

    Returns:
        BarContainer object (from plt.bar).

    Example:
        >>> categories = [1, 2, 3, 4]
        >>> values = DimArray([10, 25, 15, 30], units.m)
        >>> bar(categories, values)
    """
    _check_matplotlib()

    if ax is None:
        ax = plt.gca()

    x_data, x_label = _extract_data_and_label(x, x_unit)
    height_data, height_label = _extract_data_and_label(height, height_unit)

    result = ax.bar(x_data, height_data, *args, **kwargs)

    if x_label and not ax.get_xlabel():
        ax.set_xlabel(x_label)
    if height_label and not ax.get_ylabel():
        ax.set_ylabel(height_label)

    return result


def hist(
    x: Any,
    *args: Any,
    x_unit: Unit | None = None,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Create a histogram with automatic unit labels.

    Wrapper around matplotlib.pyplot.hist that handles DimArray objects
    and automatically labels the x-axis with units.

    Args:
        x: Data to histogram (DimArray or array-like).
        *args: Additional positional arguments passed to plt.hist.
        x_unit: Optional unit to convert x data to.
        ax: Matplotlib axes to plot on. If None, uses current axes.
        **kwargs: Keyword arguments passed to plt.hist.

    Returns:
        Tuple of (n, bins, patches) from plt.hist.

    Example:
        >>> measurements = DimArray([1.1, 1.2, 0.9, 1.0, 1.1], units.m)
        >>> hist(measurements, bins=5)
    """
    _check_matplotlib()

    if ax is None:
        ax = plt.gca()

    x_data, x_label = _extract_data_and_label(x, x_unit)

    result = ax.hist(x_data, *args, **kwargs)

    if x_label and not ax.get_xlabel():
        ax.set_xlabel(x_label)

    return result


def errorbar(
    x: Any,
    y: Any,
    yerr: Any | None = None,
    xerr: Any | None = None,
    *args: Any,
    x_unit: Unit | None = None,
    y_unit: Unit | None = None,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Create an error bar plot with automatic unit labels.

    Wrapper around matplotlib.pyplot.errorbar that handles DimArray objects.
    If y is a DimArray with uncertainty, uses that uncertainty for yerr
    unless yerr is explicitly provided.

    Args:
        x: X-axis data (DimArray or array-like).
        y: Y-axis data (DimArray or array-like).
        yerr: Y error bars. If None and y is DimArray with uncertainty,
              uses y.uncertainty.
        xerr: X error bars. If None and x is DimArray with uncertainty,
              uses x.uncertainty.
        *args: Additional positional arguments passed to plt.errorbar.
        x_unit: Optional unit to convert x data to.
        y_unit: Optional unit to convert y data to.
        ax: Matplotlib axes to plot on. If None, uses current axes.
        **kwargs: Keyword arguments passed to plt.errorbar.

    Returns:
        ErrorbarContainer object.

    Example:
        >>> x = DimArray([1, 2, 3], units.s)
        >>> y = DimArray([10, 20, 30], units.m, uncertainty=[0.5, 0.5, 0.5])
        >>> errorbar(x, y)  # Uses uncertainty from y automatically
    """
    _check_matplotlib()
    from ..core.dimarray import DimArray

    if ax is None:
        ax = plt.gca()

    x_data, x_label = _extract_data_and_label(x, x_unit)
    y_data, y_label = _extract_data_and_label(y, y_unit)

    # Auto-extract uncertainty if not provided
    if yerr is None and isinstance(y, DimArray) and y.has_uncertainty:
        if y_unit is not None:
            yerr = y.to(y_unit).uncertainty
        else:
            yerr = y.uncertainty
    elif isinstance(yerr, DimArray):
        yerr_data, _ = _extract_data_and_label(yerr, y_unit)
        yerr = yerr_data

    if xerr is None and isinstance(x, DimArray) and x.has_uncertainty:
        if x_unit is not None:
            xerr = x.to(x_unit).uncertainty
        else:
            xerr = x.uncertainty
    elif isinstance(xerr, DimArray):
        xerr_data, _ = _extract_data_and_label(xerr, x_unit)
        xerr = xerr_data

    result = ax.errorbar(x_data, y_data, yerr=yerr, xerr=xerr, *args, **kwargs)

    if x_label and not ax.get_xlabel():
        ax.set_xlabel(x_label)
    if y_label and not ax.get_ylabel():
        ax.set_ylabel(y_label)

    return result

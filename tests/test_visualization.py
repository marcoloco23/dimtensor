"""Tests for visualization module."""

import numpy as np
import pytest

from dimtensor import DimArray, units

# Skip all tests if matplotlib is not available
pytest.importorskip("matplotlib")


class TestMatplotlibConverter:
    """Tests for the DimArrayConverter class."""

    def test_setup_matplotlib_enables_converter(self):
        """Test that setup_matplotlib registers the converter."""
        import matplotlib.units as munits

        from dimtensor.visualization import setup_matplotlib

        # Enable
        setup_matplotlib(enable=True)
        assert DimArray in munits.registry

        # Disable
        setup_matplotlib(enable=False)
        assert DimArray not in munits.registry

    def test_converter_extracts_data(self):
        """Test that converter extracts numpy data."""
        from dimtensor.visualization.matplotlib import DimArrayConverter

        arr = DimArray([1.0, 2.0, 3.0], units.m)
        result = DimArrayConverter.convert(arr, None, None)

        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_converter_handles_unit_conversion(self):
        """Test that converter can convert units."""
        from dimtensor.visualization.matplotlib import DimArrayConverter

        arr = DimArray([1000.0], units.m)
        result = DimArrayConverter.convert(arr, units.km, None)

        np.testing.assert_array_almost_equal(result, [1.0])

    def test_default_units_extracts_unit(self):
        """Test default_units returns the DimArray's unit."""
        from dimtensor.visualization.matplotlib import DimArrayConverter

        arr = DimArray([1.0, 2.0], units.m)
        unit = DimArrayConverter.default_units(arr, None)

        assert unit is arr.unit

    def test_axisinfo_creates_label(self):
        """Test axisinfo creates proper label."""
        from dimtensor.visualization.matplotlib import DimArrayConverter

        info = DimArrayConverter.axisinfo(units.m, None)

        assert info is not None
        assert info.label == "[m]"


class TestPlotWrapper:
    """Tests for the plot() wrapper function."""

    def test_plot_with_dimarrays(self):
        """Test plotting DimArrays."""
        import matplotlib.pyplot as plt

        from dimtensor.visualization import plot

        fig, ax = plt.subplots()
        x = DimArray([0, 1, 2, 3], units.s)
        y = DimArray([0, 1, 4, 9], units.m)

        lines = plot(x, y, ax=ax)

        assert len(lines) == 1
        assert ax.get_xlabel() == "[s]"
        assert ax.get_ylabel() == "[m]"
        plt.close(fig)

    def test_plot_with_unit_conversion(self):
        """Test plotting with unit conversion."""
        import matplotlib.pyplot as plt

        from dimtensor.visualization import plot

        fig, ax = plt.subplots()
        x = DimArray([0, 1000, 2000], units.m)
        y = DimArray([0, 100, 200], units.cm)

        plot(x, y, x_unit=units.km, y_unit=units.m, ax=ax)

        # Check labels show converted units
        assert ax.get_xlabel() == "[km]"
        assert ax.get_ylabel() == "[m]"
        plt.close(fig)

    def test_plot_with_dimensionless(self):
        """Test plotting dimensionless data doesn't add labels."""
        import matplotlib.pyplot as plt

        from dimtensor.visualization import plot

        fig, ax = plt.subplots()
        x = DimArray([0, 1, 2, 3], units.dimensionless)
        y = DimArray([0, 1, 4, 9], units.dimensionless)

        plot(x, y, ax=ax)

        # No labels for dimensionless
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == ""
        plt.close(fig)


class TestScatterWrapper:
    """Tests for the scatter() wrapper function."""

    def test_scatter_with_dimarrays(self):
        """Test scatter plot with DimArrays."""
        import matplotlib.pyplot as plt

        from dimtensor.visualization import scatter

        fig, ax = plt.subplots()
        mass = DimArray([1, 2, 3, 4], units.kg)
        force = DimArray([9.8, 19.6, 29.4, 39.2], units.N)

        result = scatter(mass, force, ax=ax)

        assert ax.get_xlabel() == "[kg]"
        assert ax.get_ylabel() == "[N]"
        plt.close(fig)


class TestBarWrapper:
    """Tests for the bar() wrapper function."""

    def test_bar_with_dimarray(self):
        """Test bar chart with DimArray heights."""
        import matplotlib.pyplot as plt

        from dimtensor.visualization import bar

        fig, ax = plt.subplots()
        x = [1, 2, 3, 4]
        heights = DimArray([10, 25, 15, 30], units.m)

        bar(x, heights, ax=ax)

        assert ax.get_ylabel() == "[m]"
        plt.close(fig)


class TestHistWrapper:
    """Tests for the hist() wrapper function."""

    def test_hist_with_dimarray(self):
        """Test histogram with DimArray data."""
        import matplotlib.pyplot as plt

        from dimtensor.visualization import hist

        fig, ax = plt.subplots()
        data = DimArray([1.1, 1.2, 0.9, 1.0, 1.1, 0.95, 1.05], units.m)

        result = hist(data, bins=5, ax=ax)

        assert ax.get_xlabel() == "[m]"
        plt.close(fig)


class TestErrorbarWrapper:
    """Tests for the errorbar() wrapper function."""

    def test_errorbar_with_uncertainty(self):
        """Test errorbar auto-extracts uncertainty."""
        import matplotlib.pyplot as plt

        from dimtensor.visualization import errorbar

        fig, ax = plt.subplots()
        x = DimArray([1, 2, 3], units.s)
        y = DimArray([10, 20, 30], units.m, uncertainty=[0.5, 0.5, 0.5])

        result = errorbar(x, y, ax=ax)

        assert ax.get_xlabel() == "[s]"
        assert ax.get_ylabel() == "[m]"
        plt.close(fig)

    def test_errorbar_with_explicit_errors(self):
        """Test errorbar with explicit error values."""
        import matplotlib.pyplot as plt

        from dimtensor.visualization import errorbar

        fig, ax = plt.subplots()
        x = DimArray([1, 2, 3], units.s)
        y = DimArray([10, 20, 30], units.m)
        yerr = [1, 1, 1]

        result = errorbar(x, y, yerr=yerr, ax=ax)

        assert ax.get_xlabel() == "[s]"
        assert ax.get_ylabel() == "[m]"
        plt.close(fig)


class TestDirectPlotting:
    """Test direct plotting with setup_matplotlib enabled."""

    def test_direct_plot_after_setup(self):
        """Test that plt.plot works directly after setup_matplotlib."""
        import matplotlib.pyplot as plt

        from dimtensor.visualization import setup_matplotlib

        setup_matplotlib(enable=True)

        fig, ax = plt.subplots()
        x = DimArray([0, 1, 2], units.s)
        y = DimArray([0, 10, 40], units.m)

        ax.plot(x, y)

        # Cleanup
        setup_matplotlib(enable=False)
        plt.close(fig)


# =============================================================================
# Plotly tests
# =============================================================================

# Check if plotly is available
try:
    import plotly
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
class TestPlotlyLine:
    """Tests for plotly line() function."""

    def test_line_with_dimarrays(self):
        """Test line plot with DimArrays."""
        from dimtensor.visualization.plotly import line

        x = DimArray([0, 1, 2, 3], units.s)
        y = DimArray([0, 10, 40, 90], units.m)

        fig = line(x, y)

        assert fig.layout.xaxis.title.text == "[s]"
        assert fig.layout.yaxis.title.text == "[m]"

    def test_line_with_unit_conversion(self):
        """Test line plot with unit conversion."""
        from dimtensor.visualization.plotly import line

        x = DimArray([0, 1000, 2000], units.m)
        y = DimArray([0, 100, 200], units.cm)

        fig = line(x, y, x_unit=units.km, y_unit=units.m)

        assert "[km]" in fig.layout.xaxis.title.text
        assert "[m]" in fig.layout.yaxis.title.text

    def test_line_with_titles(self):
        """Test line plot with custom titles."""
        from dimtensor.visualization.plotly import line

        x = DimArray([0, 1, 2], units.s)
        y = DimArray([0, 10, 40], units.m)

        fig = line(x, y, title="Motion", x_title="Time", y_title="Distance")

        assert fig.layout.title.text == "Motion"
        assert "Time" in fig.layout.xaxis.title.text
        assert "Distance" in fig.layout.yaxis.title.text


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
class TestPlotlyScatter:
    """Tests for plotly scatter() function."""

    def test_scatter_with_dimarrays(self):
        """Test scatter plot with DimArrays."""
        from dimtensor.visualization.plotly import scatter

        mass = DimArray([1, 2, 3, 4], units.kg)
        force = DimArray([9.8, 19.6, 29.4, 39.2], units.N)

        fig = scatter(mass, force)

        assert fig.layout.xaxis.title.text == "[kg]"
        assert fig.layout.yaxis.title.text == "[N]"


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
class TestPlotlyBar:
    """Tests for plotly bar() function."""

    def test_bar_with_dimarray(self):
        """Test bar chart with DimArray heights."""
        from dimtensor.visualization.plotly import bar

        x = ["A", "B", "C", "D"]
        heights = DimArray([10, 25, 15, 30], units.m)

        fig = bar(x, heights)

        assert fig.layout.yaxis.title.text == "[m]"


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
class TestPlotlyHistogram:
    """Tests for plotly histogram() function."""

    def test_histogram_with_dimarray(self):
        """Test histogram with DimArray data."""
        from dimtensor.visualization.plotly import histogram

        data = DimArray([1.1, 1.2, 0.9, 1.0, 1.1, 0.95, 1.05], units.m)

        fig = histogram(data)

        assert fig.layout.xaxis.title.text == "[m]"


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
class TestPlotlyErrorBars:
    """Tests for plotly scatter_with_errors() function."""

    def test_scatter_with_uncertainty(self):
        """Test scatter with auto-extracted uncertainty."""
        from dimtensor.visualization.plotly import scatter_with_errors

        x = DimArray([1, 2, 3], units.s)
        y = DimArray([10, 20, 30], units.m, uncertainty=[0.5, 0.5, 0.5])

        fig = scatter_with_errors(x, y)

        assert fig.layout.xaxis.title.text == "[s]"
        assert fig.layout.yaxis.title.text == "[m]"
        # Check that error bars are present
        assert fig.data[0].error_y is not None
        assert fig.data[0].error_y.array is not None

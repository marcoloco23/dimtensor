"""Tests for configuration module."""

import numpy as np

from dimtensor import DimArray, config, units


class TestDisplayPrecision:
    """Test display precision configuration."""

    def test_default_precision(self):
        """Default precision is 4."""
        assert config.display.precision == 4

    def test_set_precision(self):
        """Can set precision permanently."""
        config.set_display(precision=2)
        assert config.display.precision == 2
        config.reset_display()
        assert config.display.precision == 4

    def test_precision_context_manager(self):
        """Precision context manager temporarily changes precision."""
        assert config.display.precision == 4
        with config.precision(6):
            assert config.display.precision == 6
        assert config.display.precision == 4

    def test_precision_affects_str(self):
        """Precision affects string output."""
        arr = DimArray([1.23456789], units.m)

        config.set_display(precision=2)
        s2 = str(arr)
        assert "1.23" in s2

        config.set_display(precision=6)
        s6 = str(arr)
        assert "1.234568" in s6

        config.reset_display()


class TestDisplayOptions:
    """Test display options configuration."""

    def test_options_context_manager(self):
        """Options context manager temporarily changes multiple options."""
        with config.options(precision=2, threshold=5):
            assert config.display.precision == 2
            assert config.display.threshold == 5
        assert config.display.precision == 4
        assert config.display.threshold == 10

    def test_threshold_affects_display(self):
        """Threshold controls when array is summarized."""
        arr = DimArray(np.arange(15), units.m)

        # Default threshold is 10, so 15 elements should be summarized
        s = str(arr)
        assert "..." in s

        # Increase threshold to show all
        with config.options(threshold=20):
            s = str(arr)
            assert "..." not in s

    def test_reset_display(self):
        """Reset returns all options to defaults."""
        config.set_display(precision=2, threshold=5, edgeitems=1)
        config.reset_display()

        assert config.display.precision == 4
        assert config.display.threshold == 10
        assert config.display.edgeitems == 3
        assert config.display.linewidth == 75
        assert config.display.suppress_small is True


class TestScalarDisplay:
    """Test scalar value display."""

    def test_scalar_uses_precision(self):
        """Scalar display uses configured precision."""
        arr = DimArray(1.23456789, units.m)

        config.set_display(precision=2)
        s = str(arr)
        assert "1.2" in s

        config.set_display(precision=6)
        s = str(arr)
        assert "1.23457" in s

        config.reset_display()

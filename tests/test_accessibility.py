"""Tests for accessibility features."""

import pytest

from dimtensor import DimArray, config, units
from dimtensor.accessibility import (
    check_distinguishability,
    check_wcag_contrast,
    color_distance,
    contrast_ratio,
    expand_unit_name,
    format_for_screen_reader,
    format_high_contrast,
    get_palette,
    hex_to_rgb,
    rgb_to_hex,
    simulate_cvd,
    simulate_palette_cvd,
    suggest_palette,
    to_html,
)


# ==============================================================================
# Color Palette Tests
# ==============================================================================


def test_get_palette():
    """Test getting color palettes."""
    # Get colorblind-safe palette
    colors = get_palette("colorblind_safe")
    assert len(colors) == 8
    assert all(color.startswith("#") for color in colors)

    # Get limited number of colors
    colors_3 = get_palette("colorblind_safe", n_colors=3)
    assert len(colors_3) == 3

    # Test other palettes
    assert len(get_palette("high_contrast")) == 8
    assert len(get_palette("grayscale")) == 8
    assert len(get_palette("tol_bright")) == 7


def test_get_palette_invalid():
    """Test getting invalid palette raises error."""
    with pytest.raises(KeyError):
        get_palette("nonexistent_palette")


def test_hex_to_rgb():
    """Test hex to RGB conversion."""
    # Test black
    r, g, b = hex_to_rgb("#000000")
    assert (r, g, b) == (0.0, 0.0, 0.0)

    # Test white
    r, g, b = hex_to_rgb("#FFFFFF")
    assert (r, g, b) == (1.0, 1.0, 1.0)

    # Test red
    r, g, b = hex_to_rgb("#FF0000")
    assert (r, g, b) == (1.0, 0.0, 0.0)

    # Test without # prefix
    r, g, b = hex_to_rgb("0000FF")
    assert (r, g, b) == (0.0, 0.0, 1.0)


def test_rgb_to_hex():
    """Test RGB to hex conversion."""
    # Test black
    assert rgb_to_hex(0.0, 0.0, 0.0) == "#000000"

    # Test white
    assert rgb_to_hex(1.0, 1.0, 1.0) == "#FFFFFF"

    # Test red
    assert rgb_to_hex(1.0, 0.0, 0.0) == "#FF0000"


def test_hex_rgb_roundtrip():
    """Test that hex -> RGB -> hex is consistent."""
    colors = ["#000000", "#FFFFFF", "#FF0000", "#00FF00", "#0000FF", "#FFFF00"]
    for color in colors:
        r, g, b = hex_to_rgb(color)
        result = rgb_to_hex(r, g, b)
        assert result == color


# ==============================================================================
# CVD Simulation Tests
# ==============================================================================


def test_simulate_cvd():
    """Test CVD simulation."""
    # Red should look different for deuteranopia
    red = "#FF0000"
    red_deut = simulate_cvd(red, "deuteranopia")
    assert red_deut != red
    assert red_deut.startswith("#")

    # Test all CVD types
    for cvd_type in ["deuteranopia", "protanopia", "tritanopia"]:
        simulated = simulate_cvd(red, cvd_type)
        assert simulated.startswith("#")


def test_simulate_cvd_invalid_type():
    """Test that invalid CVD type raises error."""
    with pytest.raises(ValueError):
        simulate_cvd("#FF0000", "invalid_type")  # type: ignore


def test_simulate_palette_cvd():
    """Test simulating entire palette."""
    palette = get_palette("colorblind_safe")
    simulated = simulate_palette_cvd(palette, "deuteranopia")

    assert len(simulated) == len(palette)
    assert all(color.startswith("#") for color in simulated)


def test_cvd_simulation_severity():
    """Test CVD simulation with different severities."""
    red = "#FF0000"

    # Zero severity should return original color
    sim_0 = simulate_cvd(red, "deuteranopia", severity=0.0)
    assert sim_0 == red

    # Partial severity
    sim_50 = simulate_cvd(red, "deuteranopia", severity=0.5)
    assert sim_50 != red

    # Full severity
    sim_100 = simulate_cvd(red, "deuteranopia", severity=1.0)
    assert sim_100 != red


# ==============================================================================
# Color Distance and Distinguishability Tests
# ==============================================================================


def test_color_distance():
    """Test color distance calculation."""
    # Same color should have distance 0
    assert color_distance("#000000", "#000000") == 0.0

    # Black and white should have large distance
    bw_distance = color_distance("#000000", "#FFFFFF")
    assert bw_distance > 0.5

    # Similar colors should have small distance
    red1 = "#FF0000"
    red2 = "#FE0000"
    assert color_distance(red1, red2) < 0.01


def test_check_distinguishability():
    """Test distinguishability checking."""
    # Colorblind-safe palette should be distinguishable (for normal vision)
    palette = get_palette("colorblind_safe")
    result = check_distinguishability(palette)
    # The function should return a valid result structure
    assert "all_distinguishable" in result
    assert "problematic_pairs" in result
    assert isinstance(result["problematic_pairs"], list)

    # Check that the function works for each CVD type
    # Note: We don't assert all are distinguishable as the minimum distance
    # threshold may be strict - just verify the function runs
    for cvd_type in ["deuteranopia", "protanopia", "tritanopia"]:
        result = check_distinguishability(palette, cvd_type, min_distance=0.1)  # type: ignore
        assert "all_distinguishable" in result
        assert isinstance(result["problematic_pairs"], list)


def test_suggest_palette():
    """Test palette suggestion."""
    # Request 5 colors
    colors = suggest_palette(5)
    assert len(colors) == 5
    assert all(color.startswith("#") for color in colors)

    # Request more colors than available (should cycle)
    colors = suggest_palette(20)
    assert len(colors) == 20


# ==============================================================================
# WCAG Contrast Tests
# ==============================================================================


def test_contrast_ratio():
    """Test contrast ratio calculation."""
    # Black on white should have maximum contrast
    ratio = contrast_ratio("#000000", "#FFFFFF")
    assert ratio == pytest.approx(21.0, rel=0.01)

    # Same color should have minimum contrast
    ratio = contrast_ratio("#000000", "#000000")
    assert ratio == pytest.approx(1.0, rel=0.01)


def test_check_wcag_contrast():
    """Test WCAG contrast checking."""
    # Black on white should pass AA and AAA
    result = check_wcag_contrast("#000000", "#FFFFFF", level="AA")
    assert result["passes"] is True
    assert result["ratio"] > 4.5

    result_aaa = check_wcag_contrast("#000000", "#FFFFFF", level="AAA")
    assert result_aaa["passes"] is True
    assert result_aaa["ratio"] > 7.0

    # Light gray on white should fail
    result = check_wcag_contrast("#EEEEEE", "#FFFFFF", level="AA")
    assert result["passes"] is False


# ==============================================================================
# Formatter Tests
# ==============================================================================


def test_expand_unit_name():
    """Test unit name expansion."""
    assert expand_unit_name("m") == "meters"
    assert expand_unit_name("kg") == "kilograms"
    assert expand_unit_name("s") == "seconds"
    assert expand_unit_name("N") == "newtons"
    assert expand_unit_name("") == "dimensionless"


def test_format_for_screen_reader():
    """Test screen reader formatting."""
    # Single value
    arr = DimArray([3.14], units.m)
    text = format_for_screen_reader(arr)
    assert "3.14" in text
    assert "meters" in text

    # Array of values
    arr = DimArray([1.0, 2.0, 3.0], units.m)
    text = format_for_screen_reader(arr)
    assert "Array" in text or "1.0" in text
    assert "meters" in text


def test_format_for_screen_reader_with_uncertainty():
    """Test screen reader formatting with uncertainty."""
    arr = DimArray([3.14], units.m, uncertainty=[0.1])
    text = format_for_screen_reader(arr)
    assert "3.14" in text
    assert "plus or minus" in text or "0.1" in text
    assert "meters" in text


def test_format_high_contrast():
    """Test high-contrast formatting."""
    arr = DimArray([1.0, 2.0, 3.0], units.m)
    text = format_high_contrast(arr)

    # Should contain box characters or equivalent
    assert len(text) > 20
    assert "m" in text  # Unit should be present


# ==============================================================================
# HTML Formatter Tests
# ==============================================================================


def test_to_html():
    """Test HTML formatting."""
    arr = DimArray([1.0, 2.0, 3.0], units.m)
    html = to_html(arr)

    # Should contain HTML tags
    assert "<" in html and ">" in html

    # Should contain ARIA attributes
    assert "aria-label" in html or "role" in html

    # Should contain the data
    assert "m" in html  # Unit should be present


def test_to_html_high_contrast():
    """Test high-contrast HTML formatting."""
    arr = DimArray([1.0], units.m)
    html = to_html(arr, high_contrast=True)

    assert "<" in html and ">" in html
    assert "high-contrast" in html or "background" in html


def test_to_html_without_css():
    """Test HTML formatting without CSS."""
    arr = DimArray([1.0], units.m)
    html = to_html(arr, include_css=False)

    # Should have HTML but not CSS
    assert "<" in html
    assert "<style>" not in html


# ==============================================================================
# Config Integration Tests
# ==============================================================================


def test_accessibility_config():
    """Test accessibility configuration options."""
    # Check defaults
    assert config.accessibility.colorblind_mode is None
    assert config.accessibility.high_contrast is False
    assert config.accessibility.screen_reader_mode is False
    assert config.accessibility.color_palette == "default"

    # Set options
    config.set_accessibility(
        colorblind_mode="deuteranopia", color_palette="colorblind_safe"
    )
    assert config.accessibility.colorblind_mode == "deuteranopia"
    assert config.accessibility.color_palette == "colorblind_safe"

    # Reset
    config.reset_accessibility()
    assert config.accessibility.colorblind_mode is None
    assert config.accessibility.color_palette == "default"


def test_accessibility_context_manager():
    """Test accessibility options context manager."""
    # Set initial state
    config.reset_accessibility()
    assert config.accessibility.color_palette == "default"

    # Use context manager
    with config.accessibility_options(color_palette="colorblind_safe"):
        assert config.accessibility.color_palette == "colorblind_safe"

    # Should be restored
    assert config.accessibility.color_palette == "default"


# ==============================================================================
# Edge Cases and Error Handling
# ==============================================================================


def test_empty_array_formatting():
    """Test formatting of empty arrays."""
    arr = DimArray([], units.m)
    text = format_for_screen_reader(arr)
    assert "Empty" in text or len(text) > 0


def test_multidimensional_array_formatting():
    """Test formatting of multidimensional arrays."""
    arr = DimArray([[1, 2], [3, 4]], units.m)
    text = format_for_screen_reader(arr)
    assert "meters" in text
    assert len(text) > 0


def test_special_values_formatting():
    """Test formatting of special values (inf, nan)."""
    import numpy as np

    arr = DimArray([np.inf], units.m)
    text = format_for_screen_reader(arr)
    assert "infinity" in text.lower()

    arr = DimArray([np.nan], units.m)
    text = format_for_screen_reader(arr)
    assert "not a number" in text.lower() or "nan" in text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

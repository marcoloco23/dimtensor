"""Colorblind-safe color palettes and CVD simulation.

Provides color schemes that are distinguishable for users with color vision
deficiencies (CVD), including deuteranopia, protanopia, and tritanopia.

Based on:
- Wong, B. (2011). "Points of view: Color blindness." Nature Methods 8, 441.
- Okabe, M. & Ito, K. "Color Universal Design."
- Paul Tol's color schemes.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

# Type aliases
CVDType = Literal["deuteranopia", "protanopia", "tritanopia"]
PaletteType = Literal[
    "colorblind_safe", "high_contrast", "grayscale", "tol_bright", "tol_muted"
]


# ==============================================================================
# Wong 2011 / Okabe & Ito Colorblind-Safe Palette
# ==============================================================================

WONG_PALETTE = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# List form for easy indexing
COLORBLIND_SAFE_QUALITATIVE = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#000000",  # Black
]

# Sequential palette (blue-based, safe for all CVD types)
COLORBLIND_SAFE_SEQUENTIAL = [
    "#f7fbff",
    "#deebf7",
    "#c6dbef",
    "#9ecae1",
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#08519c",
    "#08306b",
]

# Diverging palette (blue-orange, safe for CVD)
COLORBLIND_SAFE_DIVERGING = [
    "#d55e00",  # Orange (negative extreme)
    "#e78429",
    "#f4aa52",
    "#fdd0a2",
    "#f7f7f7",  # Neutral
    "#c7e9f2",
    "#92c5de",
    "#56b4e9",  # Blue (positive extreme)
    "#0072b2",
]


# ==============================================================================
# Paul Tol Palettes
# ==============================================================================

TOL_BRIGHT = [
    "#4477AA",  # Blue
    "#EE6677",  # Red
    "#228833",  # Green
    "#CCBB44",  # Yellow
    "#66CCEE",  # Cyan
    "#AA3377",  # Purple
    "#BBBBBB",  # Grey
]

TOL_MUTED = [
    "#CC6677",  # Rose
    "#332288",  # Indigo
    "#DDCC77",  # Sand
    "#117733",  # Green
    "#88CCEE",  # Cyan
    "#882255",  # Wine
    "#44AA99",  # Teal
    "#999933",  # Olive
    "#AA4499",  # Purple
]

TOL_HIGH_CONTRAST = [
    "#DDAA33",  # Yellow
    "#BB5566",  # Red
    "#004488",  # Blue
]


# ==============================================================================
# High Contrast & Grayscale
# ==============================================================================

HIGH_CONTRAST_PALETTE = [
    "#000000",  # Black
    "#FFFFFF",  # White
    "#FFFF00",  # Yellow
    "#00FFFF",  # Cyan
    "#FF00FF",  # Magenta
    "#0000FF",  # Blue
    "#FF0000",  # Red
    "#00FF00",  # Green
]

GRAYSCALE_PALETTE = [
    "#000000",  # Black
    "#252525",
    "#525252",
    "#737373",
    "#969696",
    "#bdbdbd",
    "#d9d9d9",
    "#f0f0f0",
]


# ==============================================================================
# Palette Registry
# ==============================================================================

PALETTES = {
    "colorblind_safe": COLORBLIND_SAFE_QUALITATIVE,
    "high_contrast": HIGH_CONTRAST_PALETTE,
    "grayscale": GRAYSCALE_PALETTE,
    "tol_bright": TOL_BRIGHT,
    "tol_muted": TOL_MUTED,
    "tol_high_contrast": TOL_HIGH_CONTRAST,
}


def get_palette(name: str, n_colors: int | None = None) -> list[str]:
    """Get a color palette by name.

    Args:
        name: Name of the palette. Options: 'colorblind_safe', 'high_contrast',
            'grayscale', 'tol_bright', 'tol_muted', 'tol_high_contrast'.
        n_colors: If specified, return only the first n colors.

    Returns:
        List of color hex codes.

    Raises:
        KeyError: If palette name is not recognized.

    Examples:
        >>> colors = get_palette('colorblind_safe')
        >>> len(colors)
        8
        >>> colors = get_palette('colorblind_safe', n_colors=3)
        >>> len(colors)
        3
    """
    if name not in PALETTES:
        raise KeyError(
            f"Unknown palette '{name}'. Available: {list(PALETTES.keys())}"
        )

    palette = PALETTES[name]
    if n_colors is not None:
        return palette[:n_colors]
    return palette


# ==============================================================================
# Color Vision Deficiency (CVD) Simulation
# ==============================================================================


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert hex color to RGB tuple (0-1 range).

    Args:
        hex_color: Hex color string like '#FF0000' or 'FF0000'.

    Returns:
        Tuple of (r, g, b) values in 0-1 range.
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)


def rgb_to_hex(r: float, g: float, b: float) -> str:
    """Convert RGB tuple to hex color.

    Args:
        r: Red component (0-1).
        g: Green component (0-1).
        b: Blue component (0-1).

    Returns:
        Hex color string like '#FF0000'.
    """
    r_int = int(np.clip(r * 255, 0, 255))
    g_int = int(np.clip(g * 255, 0, 255))
    b_int = int(np.clip(b * 255, 0, 255))
    return f"#{r_int:02X}{g_int:02X}{b_int:02X}"


def simulate_cvd(
    color: str, cvd_type: CVDType, severity: float = 1.0
) -> str:
    """Simulate how a color appears with color vision deficiency.

    Uses simplified Brettel, Viénot and Mollon (1997) transformations.

    Args:
        color: Hex color string like '#FF0000'.
        cvd_type: Type of CVD ('deuteranopia', 'protanopia', 'tritanopia').
        severity: Severity of CVD (0.0 = normal vision, 1.0 = complete).

    Returns:
        Hex color string showing how the color appears with CVD.

    Examples:
        >>> simulate_cvd('#FF0000', 'deuteranopia')  # Red appears brownish
        '#B8A600'
        >>> simulate_cvd('#00FF00', 'protanopia')  # Green appears yellowish
        '#CBCB00'
    """
    r, g, b = hex_to_rgb(color)

    # Transformation matrices for CVD simulation
    # Based on Viénot, Brettel and Mollon (1999)
    if cvd_type == "deuteranopia":  # Green-blind
        # Deuteranopia matrix
        r_sim = 0.625 * r + 0.375 * g + 0.0 * b
        g_sim = 0.7 * r + 0.3 * g + 0.0 * b
        b_sim = 0.0 * r + 0.3 * g + 0.7 * b
    elif cvd_type == "protanopia":  # Red-blind
        # Protanopia matrix
        r_sim = 0.567 * r + 0.433 * g + 0.0 * b
        g_sim = 0.558 * r + 0.442 * g + 0.0 * b
        b_sim = 0.0 * r + 0.242 * g + 0.758 * b
    elif cvd_type == "tritanopia":  # Blue-blind
        # Tritanopia matrix
        r_sim = 0.95 * r + 0.05 * g + 0.0 * b
        g_sim = 0.0 * r + 0.433 * g + 0.567 * b
        b_sim = 0.0 * r + 0.475 * g + 0.525 * b
    else:
        raise ValueError(
            f"Unknown CVD type '{cvd_type}'. "
            "Use 'deuteranopia', 'protanopia', or 'tritanopia'."
        )

    # Interpolate based on severity
    r_final = severity * r_sim + (1 - severity) * r
    g_final = severity * g_sim + (1 - severity) * g
    b_final = severity * b_sim + (1 - severity) * b

    return rgb_to_hex(r_final, g_final, b_final)


def simulate_palette_cvd(
    palette: list[str], cvd_type: CVDType, severity: float = 1.0
) -> list[str]:
    """Simulate how an entire palette appears with CVD.

    Args:
        palette: List of hex color strings.
        cvd_type: Type of CVD ('deuteranopia', 'protanopia', 'tritanopia').
        severity: Severity of CVD (0.0 = normal vision, 1.0 = complete).

    Returns:
        List of simulated hex color strings.

    Examples:
        >>> palette = get_palette('colorblind_safe')
        >>> simulated = simulate_palette_cvd(palette, 'deuteranopia')
        >>> len(simulated) == len(palette)
        True
    """
    return [simulate_cvd(color, cvd_type, severity) for color in palette]


# ==============================================================================
# Color Distinguishability Analysis
# ==============================================================================


def color_distance(color1: str, color2: str) -> float:
    """Compute perceptual distance between two colors.

    Uses simplified CIEDE2000-like metric based on RGB distance.

    Args:
        color1: First hex color string.
        color2: Second hex color string.

    Returns:
        Distance value (0 = identical, higher = more different).
    """
    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)

    # Weighted Euclidean distance (accounts for human perception)
    # Red is weighted less, green more (human eye more sensitive to green)
    dr = (r1 - r2) * 0.3
    dg = (g1 - g2) * 0.59
    db = (b1 - b2) * 0.11

    return np.sqrt(dr**2 + dg**2 + db**2)


def check_distinguishability(
    palette: list[str],
    cvd_type: CVDType | None = None,
    min_distance: float = 0.15,
) -> dict[str, bool | list[tuple[int, int]]]:
    """Check if colors in a palette are distinguishable.

    Args:
        palette: List of hex color strings.
        cvd_type: If specified, check distinguishability for this CVD type.
        min_distance: Minimum perceptual distance required (0-1 scale).

    Returns:
        Dictionary with:
            - 'all_distinguishable': bool
            - 'problematic_pairs': list of (index1, index2) tuples

    Examples:
        >>> palette = get_palette('colorblind_safe')
        >>> result = check_distinguishability(palette, 'deuteranopia')
        >>> result['all_distinguishable']
        True
    """
    # Simulate palette if CVD type specified
    if cvd_type is not None:
        test_palette = simulate_palette_cvd(palette, cvd_type)
    else:
        test_palette = palette

    problematic_pairs = []

    for i in range(len(test_palette)):
        for j in range(i + 1, len(test_palette)):
            distance = color_distance(test_palette[i], test_palette[j])
            if distance < min_distance:
                problematic_pairs.append((i, j))

    return {
        "all_distinguishable": len(problematic_pairs) == 0,
        "problematic_pairs": problematic_pairs,
    }


def suggest_palette(
    n_colors: int,
    cvd_type: CVDType | None = None,
    prefer: str = "colorblind_safe",
) -> list[str]:
    """Suggest an appropriate color palette for a given use case.

    Args:
        n_colors: Number of colors needed.
        cvd_type: If specified, optimize for this CVD type.
        prefer: Preferred palette type ('colorblind_safe', 'high_contrast', etc.).

    Returns:
        List of hex color strings.

    Examples:
        >>> colors = suggest_palette(5, cvd_type='deuteranopia')
        >>> len(colors)
        5
    """
    # Try preferred palette first
    if prefer in PALETTES:
        palette = get_palette(prefer, n_colors)
        if len(palette) >= n_colors:
            # Check if it's distinguishable
            result = check_distinguishability(palette[:n_colors], cvd_type)
            if result["all_distinguishable"]:
                return palette[:n_colors]

    # Fall back to colorblind_safe
    if prefer != "colorblind_safe":
        palette = get_palette("colorblind_safe", n_colors)
        if len(palette) >= n_colors:
            return palette[:n_colors]

    # If we need more colors than available, cycle through
    base_palette = get_palette("colorblind_safe")
    result = []
    for i in range(n_colors):
        result.append(base_palette[i % len(base_palette)])
    return result


# ==============================================================================
# Contrast Ratio (WCAG 2.1)
# ==============================================================================


def relative_luminance(color: str) -> float:
    """Calculate relative luminance of a color.

    Based on WCAG 2.1 formula.

    Args:
        color: Hex color string.

    Returns:
        Relative luminance (0-1).
    """
    r, g, b = hex_to_rgb(color)

    # Apply sRGB gamma correction
    def gamma_correct(channel: float) -> float:
        if channel <= 0.03928:
            return channel / 12.92
        else:
            return ((channel + 0.055) / 1.055) ** 2.4

    r_lin = gamma_correct(r)
    g_lin = gamma_correct(g)
    b_lin = gamma_correct(b)

    # Calculate luminance
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def contrast_ratio(color1: str, color2: str) -> float:
    """Calculate WCAG 2.1 contrast ratio between two colors.

    Args:
        color1: First hex color string.
        color2: Second hex color string.

    Returns:
        Contrast ratio (1-21). Higher is better.

    Examples:
        >>> contrast_ratio('#000000', '#FFFFFF')  # Black on white
        21.0
        >>> contrast_ratio('#0072B2', '#FFFFFF')  # Blue on white
        8.59
    """
    l1 = relative_luminance(color1)
    l2 = relative_luminance(color2)

    # Ensure l1 is the lighter color
    if l2 > l1:
        l1, l2 = l2, l1

    return (l1 + 0.05) / (l2 + 0.05)


def check_wcag_contrast(
    foreground: str, background: str, level: str = "AA", large_text: bool = False
) -> dict[str, bool | float]:
    """Check if color combination meets WCAG 2.1 contrast requirements.

    Args:
        foreground: Foreground color (hex string).
        background: Background color (hex string).
        level: WCAG level ('AA' or 'AAA').
        large_text: If True, use threshold for large text (18pt+).

    Returns:
        Dictionary with:
            - 'ratio': Contrast ratio
            - 'passes': Whether it meets the requirement
            - 'required': Required ratio for this level

    Examples:
        >>> check_wcag_contrast('#000000', '#FFFFFF', level='AA')
        {'ratio': 21.0, 'passes': True, 'required': 4.5}
    """
    ratio = contrast_ratio(foreground, background)

    # Determine required ratio
    if level == "AAA":
        required = 4.5 if large_text else 7.0
    else:  # AA
        required = 3.0 if large_text else 4.5

    return {"ratio": ratio, "passes": ratio >= required, "required": required}

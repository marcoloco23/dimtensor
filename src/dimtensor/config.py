"""Configuration options for dimtensor display and behavior."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class DisplayOptions:
    """Options for controlling DimArray display.

    Attributes:
        precision: Number of decimal places for floating point values.
        threshold: Total number of array elements which trigger summarization
            rather than full repr.
        edgeitems: Number of array items in summary at beginning and end of
            each dimension.
        linewidth: Characters per line for the purpose of inserting line breaks.
        suppress_small: If True, very small numbers are printed as zero.

    Examples:
        >>> from dimtensor import config
        >>> config.display.precision = 2
        >>> arr = DimArray([1.23456, 2.34567], units.m)
        >>> print(arr)  # Shows 2 decimal places

        >>> with config.precision(6):
        ...     print(arr)  # Temporarily shows 6 decimal places
    """

    precision: int = 4
    threshold: int = 10
    edgeitems: int = 3
    linewidth: int = 75
    suppress_small: bool = True


# Global display options instance
display = DisplayOptions()


@contextmanager
def precision(n: int) -> Iterator[None]:
    """Temporarily set display precision.

    Args:
        n: Number of decimal places.

    Examples:
        >>> with config.precision(2):
        ...     print(arr)  # Shows 2 decimal places
    """
    old = display.precision
    display.precision = n
    try:
        yield
    finally:
        display.precision = old


@contextmanager
def options(
    precision: int | None = None,
    threshold: int | None = None,
    edgeitems: int | None = None,
    linewidth: int | None = None,
    suppress_small: bool | None = None,
) -> Iterator[None]:
    """Temporarily set multiple display options.

    Args:
        precision: Number of decimal places.
        threshold: Elements that trigger summarization.
        edgeitems: Items shown at array edges in summary.
        linewidth: Characters per line.
        suppress_small: Print tiny numbers as zero.

    Examples:
        >>> with config.options(precision=2, threshold=5):
        ...     print(arr)
    """
    old_precision = display.precision
    old_threshold = display.threshold
    old_edgeitems = display.edgeitems
    old_linewidth = display.linewidth
    old_suppress_small = display.suppress_small

    if precision is not None:
        display.precision = precision
    if threshold is not None:
        display.threshold = threshold
    if edgeitems is not None:
        display.edgeitems = edgeitems
    if linewidth is not None:
        display.linewidth = linewidth
    if suppress_small is not None:
        display.suppress_small = suppress_small

    try:
        yield
    finally:
        display.precision = old_precision
        display.threshold = old_threshold
        display.edgeitems = old_edgeitems
        display.linewidth = old_linewidth
        display.suppress_small = old_suppress_small


def set_display(
    precision: int | None = None,
    threshold: int | None = None,
    edgeitems: int | None = None,
    linewidth: int | None = None,
    suppress_small: bool | None = None,
) -> None:
    """Permanently set display options.

    Args:
        precision: Number of decimal places.
        threshold: Elements that trigger summarization.
        edgeitems: Items shown at array edges in summary.
        linewidth: Characters per line.
        suppress_small: Print tiny numbers as zero.

    Examples:
        >>> config.set_display(precision=2)
    """
    if precision is not None:
        display.precision = precision
    if threshold is not None:
        display.threshold = threshold
    if edgeitems is not None:
        display.edgeitems = edgeitems
    if linewidth is not None:
        display.linewidth = linewidth
    if suppress_small is not None:
        display.suppress_small = suppress_small


def reset_display() -> None:
    """Reset display options to defaults."""
    display.precision = 4
    display.threshold = 10
    display.edgeitems = 3
    display.linewidth = 75
    display.suppress_small = True


# ==============================================================================
# Inference Configuration
# ==============================================================================


@dataclass
class InferenceOptions:
    """Options for controlling dimensional inference.

    Attributes:
        min_confidence: Minimum confidence level (0.0-1.0) for inference results.
            Results below this threshold are filtered out.
        strict_mode: If True, report all potential issues including low-confidence
            suggestions. If False, only report high-confidence warnings.
        enabled_domains: Set of physics domains to use for inference. If None,
            all domains are enabled. Valid domains: 'mechanics', 'electromagnetics',
            'thermodynamics', 'fluid_dynamics', 'optics', 'waves', 'relativity',
            'quantum'.
        infer_from_context: If True, use surrounding variable names to improve
            inference accuracy.
        warn_on_mismatch: If True, warn when adding/subtracting quantities with
            different inferred dimensions.

    Examples:
        >>> from dimtensor import config
        >>> config.inference.min_confidence = 0.8  # Only high-confidence results
        >>> config.inference.strict_mode = True     # Report everything
    """

    min_confidence: float = 0.5
    strict_mode: bool = False
    enabled_domains: set[str] | None = None
    infer_from_context: bool = True
    warn_on_mismatch: bool = True


# Global inference options instance
inference = InferenceOptions()


@contextmanager
def inference_options(
    min_confidence: float | None = None,
    strict_mode: bool | None = None,
    enabled_domains: set[str] | None = None,
    infer_from_context: bool | None = None,
    warn_on_mismatch: bool | None = None,
) -> Iterator[None]:
    """Temporarily set inference options.

    Args:
        min_confidence: Minimum confidence threshold.
        strict_mode: Report all potential issues.
        enabled_domains: Physics domains to use.
        infer_from_context: Use context for better inference.
        warn_on_mismatch: Warn on dimension mismatches.

    Examples:
        >>> with config.inference_options(min_confidence=0.9):
        ...     # Only very high-confidence inferences here
        ...     result = infer_dimension("velocity")
    """
    old_min_confidence = inference.min_confidence
    old_strict_mode = inference.strict_mode
    old_enabled_domains = inference.enabled_domains
    old_infer_from_context = inference.infer_from_context
    old_warn_on_mismatch = inference.warn_on_mismatch

    if min_confidence is not None:
        inference.min_confidence = min_confidence
    if strict_mode is not None:
        inference.strict_mode = strict_mode
    if enabled_domains is not None:
        inference.enabled_domains = enabled_domains
    if infer_from_context is not None:
        inference.infer_from_context = infer_from_context
    if warn_on_mismatch is not None:
        inference.warn_on_mismatch = warn_on_mismatch

    try:
        yield
    finally:
        inference.min_confidence = old_min_confidence
        inference.strict_mode = old_strict_mode
        inference.enabled_domains = old_enabled_domains
        inference.infer_from_context = old_infer_from_context
        inference.warn_on_mismatch = old_warn_on_mismatch


def set_inference(
    min_confidence: float | None = None,
    strict_mode: bool | None = None,
    enabled_domains: set[str] | None = None,
    infer_from_context: bool | None = None,
    warn_on_mismatch: bool | None = None,
) -> None:
    """Permanently set inference options.

    Args:
        min_confidence: Minimum confidence threshold.
        strict_mode: Report all potential issues.
        enabled_domains: Physics domains to use.
        infer_from_context: Use context for better inference.
        warn_on_mismatch: Warn on dimension mismatches.

    Examples:
        >>> config.set_inference(min_confidence=0.8, strict_mode=True)
    """
    if min_confidence is not None:
        inference.min_confidence = min_confidence
    if strict_mode is not None:
        inference.strict_mode = strict_mode
    if enabled_domains is not None:
        inference.enabled_domains = enabled_domains
    if infer_from_context is not None:
        inference.infer_from_context = infer_from_context
    if warn_on_mismatch is not None:
        inference.warn_on_mismatch = warn_on_mismatch


def reset_inference() -> None:
    """Reset inference options to defaults."""
    inference.min_confidence = 0.5
    inference.strict_mode = False
    inference.enabled_domains = None
    inference.infer_from_context = True
    inference.warn_on_mismatch = True


# ==============================================================================
# Internationalization Configuration
# ==============================================================================


@dataclass
class I18nOptions:
    """Options for internationalization (i18n).

    Attributes:
        locale: Current locale code (e.g., 'en', 'es', 'fr'). If None,
            auto-detects from system settings.
        use_localized_units: If True, display unit names in localized form.
        use_localized_errors: If True, translate error messages.
        fallback_locale: Locale to use when translation is missing.

    Examples:
        >>> from dimtensor import config
        >>> config.i18n.locale = 'es'  # Use Spanish
        >>> config.i18n.use_localized_units = True
    """

    locale: str | None = None  # None = auto-detect
    use_localized_units: bool = True
    use_localized_errors: bool = True
    fallback_locale: str = "en"


# Global i18n options instance
i18n = I18nOptions()


@contextmanager
def locale(locale_code: str) -> Iterator[None]:
    """Temporarily switch to a different locale.

    This is a convenience wrapper around dimtensor.i18n.locale_context().

    Args:
        locale_code: Locale code to use (e.g., 'en', 'es', 'fr')

    Examples:
        >>> from dimtensor import config, units
        >>> with config.locale('es'):
        ...     print(units.meter.localized_name())
        metro
        >>> print(units.meter.localized_name())
        meter
    """
    try:
        from dimtensor.i18n import locale_context

        with locale_context(locale_code):
            yield
    except ImportError:
        # i18n not available, just yield without changing locale
        yield


# ==============================================================================
# Accessibility Configuration
# ==============================================================================


@dataclass
class AccessibilityOptions:
    """Options for accessibility features.

    Attributes:
        colorblind_mode: Color vision deficiency mode for simulation.
            Options: None (disabled), 'deuteranopia', 'protanopia', 'tritanopia'.
        high_contrast: If True, use high-contrast display modes for better readability.
        screen_reader_mode: If True, format output for screen readers (expanded units).
        color_palette: Default color palette for visualizations.
            Options: 'default', 'colorblind_safe', 'high_contrast', 'grayscale',
            'tol_bright', 'tol_muted'.

    Examples:
        >>> from dimtensor import config
        >>> config.accessibility.colorblind_mode = 'deuteranopia'
        >>> config.accessibility.color_palette = 'colorblind_safe'
        >>> config.accessibility.screen_reader_mode = True
    """

    colorblind_mode: str | None = None
    high_contrast: bool = False
    screen_reader_mode: bool = False
    color_palette: str = "default"


# Global accessibility options instance
accessibility = AccessibilityOptions()


@contextmanager
def accessibility_options(
    colorblind_mode: str | None = None,
    high_contrast: bool | None = None,
    screen_reader_mode: bool | None = None,
    color_palette: str | None = None,
) -> Iterator[None]:
    """Temporarily set accessibility options.

    Args:
        colorblind_mode: CVD mode ('deuteranopia', 'protanopia', 'tritanopia', None).
        high_contrast: Use high-contrast display modes.
        screen_reader_mode: Format output for screen readers.
        color_palette: Default color palette for visualizations.

    Examples:
        >>> with config.accessibility_options(colorblind_mode='deuteranopia'):
        ...     # Visualizations will use colorblind-safe palette
        ...     plt.plot(x, y)
    """
    old_colorblind_mode = accessibility.colorblind_mode
    old_high_contrast = accessibility.high_contrast
    old_screen_reader_mode = accessibility.screen_reader_mode
    old_color_palette = accessibility.color_palette

    if colorblind_mode is not None:
        accessibility.colorblind_mode = colorblind_mode
    if high_contrast is not None:
        accessibility.high_contrast = high_contrast
    if screen_reader_mode is not None:
        accessibility.screen_reader_mode = screen_reader_mode
    if color_palette is not None:
        accessibility.color_palette = color_palette

    try:
        yield
    finally:
        accessibility.colorblind_mode = old_colorblind_mode
        accessibility.high_contrast = old_high_contrast
        accessibility.screen_reader_mode = old_screen_reader_mode
        accessibility.color_palette = old_color_palette


def set_accessibility(
    colorblind_mode: str | None = None,
    high_contrast: bool | None = None,
    screen_reader_mode: bool | None = None,
    color_palette: str | None = None,
) -> None:
    """Permanently set accessibility options.

    Args:
        colorblind_mode: CVD mode ('deuteranopia', 'protanopia', 'tritanopia', None).
        high_contrast: Use high-contrast display modes.
        screen_reader_mode: Format output for screen readers.
        color_palette: Default color palette for visualizations.

    Examples:
        >>> config.set_accessibility(
        ...     colorblind_mode='deuteranopia',
        ...     color_palette='colorblind_safe'
        ... )
    """
    if colorblind_mode is not None:
        accessibility.colorblind_mode = colorblind_mode
    if high_contrast is not None:
        accessibility.high_contrast = high_contrast
    if screen_reader_mode is not None:
        accessibility.screen_reader_mode = screen_reader_mode
    if color_palette is not None:
        accessibility.color_palette = color_palette


def reset_accessibility() -> None:
    """Reset accessibility options to defaults."""
    accessibility.colorblind_mode = None
    accessibility.high_contrast = False
    accessibility.screen_reader_mode = False
    accessibility.color_palette = "default"

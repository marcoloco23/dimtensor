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

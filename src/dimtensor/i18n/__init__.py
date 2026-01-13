"""Internationalization (i18n) support for dimtensor.

Provides translation for unit names, error messages, and CLI output in multiple
languages. Supports thread-safe locale management with automatic fallback chains.

Supported locales:
- en (English)
- es (Spanish)
- fr (French)
- de (German)
- zh_CN (Simplified Chinese)
- ja (Japanese)

Examples:
    >>> from dimtensor import i18n
    >>> i18n.set_locale('es')
    >>> i18n.translate_unit_name('m')
    'metro'

    >>> with i18n.locale_context('fr'):
    ...     print(i18n.translate_unit_name('kg'))
    kilogramme
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

# Import public API
from .locale import (
    available_locales,
    detect_locale,
    get_fallback_chain,
    get_locale,
    reset_locale,
    set_default_locale,
    set_locale,
)
from .translations import (
    clear_cache,
    translate,
    translate_cli,
    translate_dimension_name,
    translate_error,
    translate_unit_name,
    translate_unit_symbol,
)

__all__ = [
    # Locale management
    "get_locale",
    "set_locale",
    "set_default_locale",
    "reset_locale",
    "detect_locale",
    "available_locales",
    "get_fallback_chain",
    # Translation functions
    "translate",
    "_",  # gettext-compatible alias
    "translate_unit_name",
    "translate_unit_symbol",
    "translate_dimension_name",
    "translate_error",
    "translate_cli",
    # Utilities
    "clear_cache",
    "locale_context",
]


def _(message: str, **kwargs: Any) -> str:
    """Translate a message (gettext-compatible naming).

    This is a shorthand for translate() that uses the 'messages' category
    by default. The name '_' is compatible with gettext conventions.

    Args:
        message: Message key to translate
        **kwargs: Format arguments

    Returns:
        Translated and formatted message

    Examples:
        >>> from dimtensor.i18n import _
        >>> _('conversion_warning', from_unit='m', to_unit='km')
        'Warning: converting from m to km'
    """
    return translate(message, category="messages", **kwargs)


@contextmanager
def locale_context(locale_code: str) -> Iterator[None]:
    """Context manager for temporarily switching locale.

    Args:
        locale_code: Locale code to use within context

    Yields:
        None

    Examples:
        >>> with locale_context('es'):
        ...     print(translate_unit_name('m'))
        metro
        >>> print(translate_unit_name('m'))
        meter
    """
    old_locale = get_locale()
    try:
        set_locale(locale_code)
        yield
    finally:
        if old_locale != get_locale():
            try:
                set_locale(old_locale)
            except ValueError:
                # Old locale no longer available, reset to default
                reset_locale()

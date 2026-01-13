"""Translation loading and lookup for dimtensor i18n.

Provides lazy loading and caching of translation files with fallback support.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .locale import get_locale, get_fallback_chain


# Cache for loaded translations
_translation_cache: dict[str, dict[str, Any]] = {}

# Path to locales directory
_LOCALES_DIR = Path(__file__).parent / "locales"


def _load_locale_file(locale_code: str) -> dict[str, Any]:
    """Load a locale JSON file.

    Args:
        locale_code: Locale code (e.g., 'en', 'es')

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If locale file doesn't exist
    """
    if locale_code in _translation_cache:
        return _translation_cache[locale_code]

    locale_file = _LOCALES_DIR / f"{locale_code}.json"

    if not locale_file.exists():
        raise FileNotFoundError(f"Locale file not found: {locale_file}")

    with open(locale_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    _translation_cache[locale_code] = data
    return data


def translate(
    key: str,
    locale: Optional[str] = None,
    category: str = "messages",
    **kwargs: Any,
) -> str:
    """Translate a message key.

    Args:
        key: Translation key
        locale: Locale code (None = use current)
        category: Category in JSON (e.g., 'errors', 'messages', 'cli')
        **kwargs: Format arguments for the translation string

    Returns:
        Translated and formatted string

    Examples:
        >>> translate('conversion_warning', from_unit='m', to_unit='km')
        'Warning: converting from m to km'
        >>> translate('conversion_warning', locale='es', from_unit='m', to_unit='km')
        'Advertencia: convirtiendo de m a km'
    """
    if locale is None:
        locale = get_locale()

    # Try fallback chain
    for fallback_locale in get_fallback_chain(locale):
        try:
            data = _load_locale_file(fallback_locale)

            # Navigate to category
            if category in data and key in data[category]:
                template = data[category][key]
                return template.format(**kwargs)

        except (FileNotFoundError, KeyError):
            continue

    # No translation found, return key with args
    if kwargs:
        return f"{key}({', '.join(f'{k}={v}' for k, v in kwargs.items())})"
    return key


def translate_unit_name(
    symbol: str,
    locale: Optional[str] = None,
    plural: bool = False,
) -> str:
    """Translate a unit name.

    Args:
        symbol: Unit symbol (e.g., 'm', 'kg')
        locale: Locale code (None = use current)
        plural: Return plural form

    Returns:
        Translated unit name

    Examples:
        >>> translate_unit_name('m')
        'meter'
        >>> translate_unit_name('m', plural=True)
        'meters'
        >>> translate_unit_name('m', locale='es')
        'metro'
        >>> translate_unit_name('m', locale='es', plural=True)
        'metros'
    """
    if locale is None:
        locale = get_locale()

    # Try fallback chain
    for fallback_locale in get_fallback_chain(locale):
        try:
            data = _load_locale_file(fallback_locale)

            if "units" in data and symbol in data["units"]:
                unit_data = data["units"][symbol]
                if plural:
                    return unit_data.get("name_plural", unit_data.get("name", symbol))
                return unit_data.get("name", symbol)

        except (FileNotFoundError, KeyError):
            continue

    # No translation found, return symbol
    return symbol


def translate_unit_symbol(
    symbol: str,
    locale: Optional[str] = None,
) -> str:
    """Translate a unit symbol (usually returns same symbol).

    Most units keep the same symbol across locales, but this provides
    a hook for exceptions.

    Args:
        symbol: Unit symbol (e.g., 'm', 'kg')
        locale: Locale code (None = use current)

    Returns:
        Translated symbol (usually same as input)

    Examples:
        >>> translate_unit_symbol('m')
        'm'
        >>> translate_unit_symbol('m', locale='es')
        'm'
    """
    if locale is None:
        locale = get_locale()

    # Try fallback chain
    for fallback_locale in get_fallback_chain(locale):
        try:
            data = _load_locale_file(fallback_locale)

            if "units" in data and symbol in data["units"]:
                return data["units"][symbol].get("symbol", symbol)

        except (FileNotFoundError, KeyError):
            continue

    # No translation found, return original symbol
    return symbol


def translate_dimension_name(
    dimension_name: str,
    locale: Optional[str] = None,
) -> str:
    """Translate a dimension name.

    Args:
        dimension_name: Dimension name (e.g., 'length', 'mass')
        locale: Locale code (None = use current)

    Returns:
        Translated dimension name

    Examples:
        >>> translate_dimension_name('length')
        'length'
        >>> translate_dimension_name('length', locale='es')
        'longitud'
        >>> translate_dimension_name('mass', locale='fr')
        'masse'
    """
    if locale is None:
        locale = get_locale()

    # Try fallback chain
    for fallback_locale in get_fallback_chain(locale):
        try:
            data = _load_locale_file(fallback_locale)

            if "dimensions" in data and dimension_name in data["dimensions"]:
                return data["dimensions"][dimension_name]

        except (FileNotFoundError, KeyError):
            continue

    # No translation found, return original name
    return dimension_name


def translate_error(error_key: str, locale: Optional[str] = None, **kwargs: Any) -> str:
    """Translate an error message.

    Args:
        error_key: Error message key
        locale: Locale code (None = use current)
        **kwargs: Format arguments

    Returns:
        Translated error message

    Examples:
        >>> translate_error('unit_not_found', symbol='xyz')
        "Unit 'xyz' not found"
        >>> translate_error('unit_not_found', locale='es', symbol='xyz')
        "Unidad 'xyz' no encontrada"
    """
    return translate(error_key, locale=locale, category="errors", **kwargs)


def translate_cli(cli_key: str, locale: Optional[str] = None, **kwargs: Any) -> str:
    """Translate a CLI message.

    Args:
        cli_key: CLI message key
        locale: Locale code (None = use current)
        **kwargs: Format arguments

    Returns:
        Translated CLI message

    Examples:
        >>> translate_cli('checking_file', filename='test.py')
        'Checking file: test.py'
        >>> translate_cli('issues_found', locale='fr', count=5)
        '5 problèmes trouvés'
    """
    return translate(cli_key, locale=locale, category="cli", **kwargs)


def clear_cache() -> None:
    """Clear the translation cache.

    Useful for testing or if locale files are updated at runtime.

    Examples:
        >>> clear_cache()
    """
    _translation_cache.clear()

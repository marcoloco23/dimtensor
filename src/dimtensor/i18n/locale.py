"""Locale detection and management for dimtensor i18n.

Provides thread-safe locale management with fallback chains for regional variants.
"""

from __future__ import annotations

import locale as system_locale
import os
import threading
from typing import Optional


# Thread-local storage for per-thread locale
_thread_local = threading.local()

# Global default locale
_default_locale: str = "en"

# Available locales (loaded from locales directory)
_AVAILABLE_LOCALES = frozenset(["en", "es", "fr", "de", "zh_CN", "ja"])


def detect_locale() -> str:
    """Detect system locale.

    Checks in order:
    1. DIMTENSOR_LOCALE environment variable
    2. LANG environment variable
    3. System locale
    4. Falls back to 'en'

    Returns:
        Locale code (e.g., 'en', 'es', 'zh_CN')

    Examples:
        >>> locale = detect_locale()
        >>> print(locale)
        en
    """
    # Check environment variables
    env_locale = os.environ.get("DIMTENSOR_LOCALE")
    if env_locale:
        return _normalize_locale(env_locale)

    lang = os.environ.get("LANG")
    if lang:
        return _normalize_locale(lang)

    # Try system locale
    try:
        sys_locale, _ = system_locale.getlocale()
        if sys_locale:
            return _normalize_locale(sys_locale)
    except (ValueError, TypeError):
        pass

    # Fallback to English
    return "en"


def _normalize_locale(locale_str: str) -> str:
    """Normalize locale string to our format.

    Converts:
    - en_US.UTF-8 -> en
    - zh_CN.UTF-8 -> zh_CN
    - es_MX -> es (if es_MX not available)

    Args:
        locale_str: Locale string from system or environment

    Returns:
        Normalized locale code
    """
    # Remove encoding
    if "." in locale_str:
        locale_str = locale_str.split(".")[0]

    # Remove @modifiers
    if "@" in locale_str:
        locale_str = locale_str.split("@")[0]

    # Check if exact match exists
    if locale_str in _AVAILABLE_LOCALES:
        return locale_str

    # Try base language (e.g., es_MX -> es)
    if "_" in locale_str:
        base_lang = locale_str.split("_")[0]
        if base_lang in _AVAILABLE_LOCALES:
            return base_lang

    # Fallback to English
    return "en"


def get_locale() -> str:
    """Get current locale for this thread.

    Returns:
        Current locale code

    Examples:
        >>> get_locale()
        'en'
        >>> set_locale('es')
        >>> get_locale()
        'es'
    """
    # Check thread-local first
    if hasattr(_thread_local, "locale"):
        return _thread_local.locale

    # Use default
    return _default_locale


def set_locale(locale_code: str) -> None:
    """Set locale for this thread.

    Args:
        locale_code: Locale code (e.g., 'en', 'es', 'zh_CN')

    Raises:
        ValueError: If locale is not available

    Examples:
        >>> set_locale('es')
        >>> get_locale()
        'es'
    """
    # Check if exact match or base language exists before normalizing
    if locale_code not in _AVAILABLE_LOCALES:
        # Check if base language exists for regional variants
        if "_" in locale_code:
            base_lang = locale_code.split("_")[0]
            if base_lang not in _AVAILABLE_LOCALES:
                raise ValueError(
                    f"Locale '{locale_code}' not available. "
                    f"Available: {sorted(_AVAILABLE_LOCALES)}"
                )
        else:
            # Not a regional variant and not in available locales
            raise ValueError(
                f"Locale '{locale_code}' not available. "
                f"Available: {sorted(_AVAILABLE_LOCALES)}"
            )

    normalized = _normalize_locale(locale_code)
    _thread_local.locale = normalized


def set_default_locale(locale_code: str) -> None:
    """Set the default locale for all threads.

    This affects new threads and threads that haven't set their own locale.

    Args:
        locale_code: Locale code (e.g., 'en', 'es')

    Raises:
        ValueError: If locale is not available

    Examples:
        >>> set_default_locale('es')
    """
    global _default_locale

    normalized = _normalize_locale(locale_code)

    if normalized not in _AVAILABLE_LOCALES:
        raise ValueError(
            f"Locale '{locale_code}' not available. "
            f"Available: {sorted(_AVAILABLE_LOCALES)}"
        )

    _default_locale = normalized


def reset_locale() -> None:
    """Reset this thread's locale to the default.

    Examples:
        >>> set_locale('es')
        >>> get_locale()
        'es'
        >>> reset_locale()
        >>> get_locale()
        'en'
    """
    if hasattr(_thread_local, "locale"):
        delattr(_thread_local, "locale")


def available_locales() -> list[str]:
    """Get list of available locales.

    Returns:
        Sorted list of locale codes

    Examples:
        >>> locales = available_locales()
        >>> 'en' in locales
        True
        >>> 'es' in locales
        True
    """
    return sorted(_AVAILABLE_LOCALES)


def get_fallback_chain(locale_code: str) -> list[str]:
    """Get fallback chain for a locale.

    For example, 'es_MX' falls back to 'es', then 'en'.

    Args:
        locale_code: Locale code

    Returns:
        List of locale codes in fallback order

    Examples:
        >>> get_fallback_chain('es_MX')
        ['es_MX', 'es', 'en']
        >>> get_fallback_chain('en')
        ['en']
    """
    chain = [locale_code]

    # Add base language if this is a regional variant
    if "_" in locale_code:
        base_lang = locale_code.split("_")[0]
        if base_lang not in chain:
            chain.append(base_lang)

    # Always fallback to English
    if "en" not in chain:
        chain.append("en")

    return chain

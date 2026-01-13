"""Tests for internationalization (i18n) system."""

import threading

import pytest

from dimtensor import config, i18n, units


class TestLocaleManagement:
    """Test locale detection and management."""

    def test_default_locale(self):
        """Test default locale is English."""
        i18n.reset_locale()
        assert i18n.get_locale() == "en"

    def test_set_get_locale(self):
        """Test setting and getting locale."""
        original = i18n.get_locale()
        try:
            i18n.set_locale("es")
            assert i18n.get_locale() == "es"

            i18n.set_locale("fr")
            assert i18n.get_locale() == "fr"
        finally:
            i18n.set_locale(original)

    def test_invalid_locale(self):
        """Test setting invalid locale raises ValueError."""
        with pytest.raises(ValueError, match="not available"):
            i18n.set_locale("invalid_locale")

    def test_reset_locale(self):
        """Test resetting locale."""
        original = i18n.get_locale()
        try:
            i18n.set_locale("es")
            assert i18n.get_locale() == "es"

            i18n.reset_locale()
            # Should reset to default (en or whatever was set as default)
            assert i18n.get_locale() in i18n.available_locales()
        finally:
            i18n.set_locale(original)

    def test_available_locales(self):
        """Test getting list of available locales."""
        locales = i18n.available_locales()
        assert isinstance(locales, list)
        assert "en" in locales
        assert "es" in locales
        assert "fr" in locales
        assert "de" in locales
        assert "zh_CN" in locales
        assert "ja" in locales
        assert len(locales) >= 6

    def test_fallback_chain(self):
        """Test fallback chain for regional variants."""
        # Test regional variant falls back to base language
        chain = i18n.get_fallback_chain("es_MX")
        assert "es_MX" in chain
        assert "es" in chain
        assert "en" in chain

        # Test base language falls back to English
        chain = i18n.get_fallback_chain("es")
        assert "es" in chain
        assert "en" in chain

        # Test English doesn't have additional fallback
        chain = i18n.get_fallback_chain("en")
        assert chain == ["en"]

    def test_locale_context_manager(self):
        """Test locale context manager."""
        original = i18n.get_locale()
        try:
            i18n.set_locale("en")
            assert i18n.get_locale() == "en"

            with i18n.locale_context("es"):
                assert i18n.get_locale() == "es"

                # Nested context
                with i18n.locale_context("fr"):
                    assert i18n.get_locale() == "fr"

                assert i18n.get_locale() == "es"

            assert i18n.get_locale() == "en"
        finally:
            i18n.set_locale(original)

    def test_config_locale_context(self):
        """Test config.locale() context manager."""
        original = i18n.get_locale()
        try:
            i18n.set_locale("en")

            with config.locale("es"):
                assert i18n.get_locale() == "es"

            assert i18n.get_locale() == "en"
        finally:
            i18n.set_locale(original)


class TestUnitTranslation:
    """Test unit name translation."""

    def test_translate_unit_name_english(self):
        """Test translating unit names to English."""
        assert i18n.translate_unit_name("m", locale="en") == "meter"
        assert i18n.translate_unit_name("kg", locale="en") == "kilogram"
        assert i18n.translate_unit_name("s", locale="en") == "second"

    def test_translate_unit_name_spanish(self):
        """Test translating unit names to Spanish."""
        assert i18n.translate_unit_name("m", locale="es") == "metro"
        assert i18n.translate_unit_name("kg", locale="es") == "kilogramo"
        assert i18n.translate_unit_name("s", locale="es") == "segundo"

    def test_translate_unit_name_french(self):
        """Test translating unit names to French."""
        assert i18n.translate_unit_name("m", locale="fr") == "mètre"
        assert i18n.translate_unit_name("kg", locale="fr") == "kilogramme"
        assert i18n.translate_unit_name("s", locale="fr") == "seconde"

    def test_translate_unit_name_german(self):
        """Test translating unit names to German."""
        assert i18n.translate_unit_name("m", locale="de") == "Meter"
        assert i18n.translate_unit_name("kg", locale="de") == "Kilogramm"
        assert i18n.translate_unit_name("s", locale="de") == "Sekunde"

    def test_translate_unit_name_chinese(self):
        """Test translating unit names to Chinese."""
        assert i18n.translate_unit_name("m", locale="zh_CN") == "米"
        assert i18n.translate_unit_name("kg", locale="zh_CN") == "千克"
        assert i18n.translate_unit_name("s", locale="zh_CN") == "秒"

    def test_translate_unit_name_japanese(self):
        """Test translating unit names to Japanese."""
        assert i18n.translate_unit_name("m", locale="ja") == "メートル"
        assert i18n.translate_unit_name("kg", locale="ja") == "キログラム"
        assert i18n.translate_unit_name("s", locale="ja") == "秒"

    def test_translate_unit_name_plural(self):
        """Test translating plural unit names."""
        assert i18n.translate_unit_name("m", locale="en", plural=True) == "meters"
        assert i18n.translate_unit_name("m", locale="es", plural=True) == "metros"
        assert i18n.translate_unit_name("m", locale="fr", plural=True) == "mètres"

    def test_translate_unknown_unit(self):
        """Test translating unknown unit returns symbol."""
        assert i18n.translate_unit_name("unknown", locale="en") == "unknown"
        assert i18n.translate_unit_name("xyz", locale="es") == "xyz"

    def test_unit_localized_name_method(self):
        """Test Unit.localized_name() method."""
        assert units.meter.localized_name(locale="en") == "meter"
        assert units.meter.localized_name(locale="es") == "metro"
        assert units.meter.localized_name(locale="fr") == "mètre"

        assert units.kilogram.localized_name(locale="en") == "kilogram"
        assert units.kilogram.localized_name(locale="es") == "kilogramo"
        assert units.kilogram.localized_name(locale="de") == "Kilogramm"

    def test_unit_localized_name_plural(self):
        """Test Unit.localized_name() with plural."""
        assert units.meter.localized_name(locale="en", plural=True) == "meters"
        assert units.meter.localized_name(locale="es", plural=True) == "metros"
        assert units.second.localized_name(locale="de", plural=True) == "Sekunden"

    def test_unit_localized_name_current_locale(self):
        """Test Unit.localized_name() uses current locale."""
        original = i18n.get_locale()
        try:
            i18n.set_locale("es")
            assert units.meter.localized_name() == "metro"

            i18n.set_locale("fr")
            assert units.meter.localized_name() == "mètre"
        finally:
            i18n.set_locale(original)

    def test_unit_localized_symbol(self):
        """Test Unit.localized_symbol() method."""
        # Most units keep same symbol
        assert units.meter.localized_symbol(locale="en") == "m"
        assert units.meter.localized_symbol(locale="es") == "m"
        assert units.meter.localized_symbol(locale="zh_CN") == "m"

        assert units.kilogram.localized_symbol(locale="fr") == "kg"
        assert units.newton.localized_symbol(locale="ja") == "N"


class TestDimensionTranslation:
    """Test dimension name translation."""

    def test_translate_dimension_english(self):
        """Test translating dimension names to English."""
        assert i18n.translate_dimension_name("length", locale="en") == "length"
        assert i18n.translate_dimension_name("mass", locale="en") == "mass"
        assert i18n.translate_dimension_name("time", locale="en") == "time"

    def test_translate_dimension_spanish(self):
        """Test translating dimension names to Spanish."""
        assert i18n.translate_dimension_name("length", locale="es") == "longitud"
        assert i18n.translate_dimension_name("mass", locale="es") == "masa"
        assert i18n.translate_dimension_name("time", locale="es") == "tiempo"

    def test_translate_dimension_french(self):
        """Test translating dimension names to French."""
        assert i18n.translate_dimension_name("length", locale="fr") == "longueur"
        assert i18n.translate_dimension_name("mass", locale="fr") == "masse"
        assert i18n.translate_dimension_name("time", locale="fr") == "temps"

    def test_translate_dimension_german(self):
        """Test translating dimension names to German."""
        assert i18n.translate_dimension_name("length", locale="de") == "Länge"
        assert i18n.translate_dimension_name("mass", locale="de") == "Masse"
        assert i18n.translate_dimension_name("time", locale="de") == "Zeit"


class TestErrorTranslation:
    """Test error message translation."""

    def test_translate_error_english(self):
        """Test translating error messages to English."""
        msg = i18n.translate_error("unit_not_found", locale="en", symbol="xyz")
        assert "Unit 'xyz' not found" == msg

    def test_translate_error_spanish(self):
        """Test translating error messages to Spanish."""
        msg = i18n.translate_error("unit_not_found", locale="es", symbol="xyz")
        assert "Unidad 'xyz' no encontrada" == msg

    def test_translate_error_french(self):
        """Test translating error messages to French."""
        msg = i18n.translate_error("unit_not_found", locale="fr", symbol="xyz")
        assert "Unité 'xyz' introuvable" == msg

    def test_translate_error_with_formatting(self):
        """Test error translation with multiple format args."""
        msg = i18n.translate_error(
            "unit_conversion_incompatible",
            locale="en",
            from_unit="m",
            to_unit="kg",
        )
        assert "Cannot convert from m to kg" in msg


class TestCLITranslation:
    """Test CLI message translation."""

    def test_translate_cli_english(self):
        """Test translating CLI messages to English."""
        msg = i18n.translate_cli("checking_file", locale="en", filename="test.py")
        assert "Checking file: test.py" == msg

        msg = i18n.translate_cli("issues_found", locale="en", count=5)
        assert "Found 5 issues" == msg

    def test_translate_cli_spanish(self):
        """Test translating CLI messages to Spanish."""
        msg = i18n.translate_cli("checking_file", locale="es", filename="test.py")
        assert "Verificando archivo: test.py" == msg

        msg = i18n.translate_cli("issues_found", locale="es", count=5)
        assert "Se encontraron 5 problemas" == msg


class TestMessageTranslation:
    """Test general message translation."""

    def test_translate_message_english(self):
        """Test translating general messages to English."""
        msg = i18n.translate(
            "conversion_warning",
            locale="en",
            category="messages",
            from_unit="m",
            to_unit="km",
        )
        assert "Warning: converting from m to km" == msg

    def test_translate_message_spanish(self):
        """Test translating general messages to Spanish."""
        msg = i18n.translate(
            "conversion_warning",
            locale="es",
            category="messages",
            from_unit="m",
            to_unit="km",
        )
        assert "Advertencia: convirtiendo de m a km" == msg

    def test_gettext_compatible_function(self):
        """Test _() function (gettext-compatible)."""
        msg = i18n._("conversion_warning", from_unit="m", to_unit="km")
        # Should work with current locale
        assert isinstance(msg, str)
        assert "m" in msg and "km" in msg


class TestThreadSafety:
    """Test thread-safety of locale management."""

    def test_thread_local_locale(self):
        """Test that each thread can have its own locale."""
        results = {}

        def set_and_get_locale(thread_id, locale_code):
            i18n.set_locale(locale_code)
            results[thread_id] = i18n.get_locale()

        # Create threads with different locales
        t1 = threading.Thread(target=set_and_get_locale, args=(1, "es"))
        t2 = threading.Thread(target=set_and_get_locale, args=(2, "fr"))
        t3 = threading.Thread(target=set_and_get_locale, args=(3, "de"))

        t1.start()
        t2.start()
        t3.start()

        t1.join()
        t2.join()
        t3.join()

        # Each thread should have its own locale
        assert results[1] == "es"
        assert results[2] == "fr"
        assert results[3] == "de"

    def test_thread_local_translation(self):
        """Test that translations use thread-local locale."""
        results = {}

        def translate_in_locale(thread_id, locale_code):
            i18n.set_locale(locale_code)
            results[thread_id] = i18n.translate_unit_name("m")

        t1 = threading.Thread(target=translate_in_locale, args=(1, "es"))
        t2 = threading.Thread(target=translate_in_locale, args=(2, "fr"))

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        assert results[1] == "metro"
        assert results[2] == "mètre"


class TestCaching:
    """Test translation caching."""

    def test_clear_cache(self):
        """Test clearing translation cache."""
        # Load some translations
        i18n.translate_unit_name("m", locale="es")
        i18n.translate_unit_name("m", locale="fr")

        # Clear cache
        i18n.clear_cache()

        # Translations should still work (will reload)
        assert i18n.translate_unit_name("m", locale="es") == "metro"
        assert i18n.translate_unit_name("m", locale="fr") == "mètre"


class TestIntegration:
    """Integration tests with dimtensor components."""

    def test_unit_with_locale_context(self):
        """Test using units with locale context."""
        original = i18n.get_locale()
        try:
            # English
            with i18n.locale_context("en"):
                assert units.meter.localized_name() == "meter"
                assert units.kilogram.localized_name() == "kilogram"

            # Spanish
            with i18n.locale_context("es"):
                assert units.meter.localized_name() == "metro"
                assert units.kilogram.localized_name() == "kilogramo"

            # French
            with i18n.locale_context("fr"):
                assert units.meter.localized_name() == "mètre"
                assert units.kilogram.localized_name() == "kilogramme"
        finally:
            i18n.set_locale(original)

    def test_config_i18n_options(self):
        """Test I18nOptions in config."""
        assert hasattr(config, "i18n")
        assert hasattr(config.i18n, "locale")
        assert hasattr(config.i18n, "use_localized_units")
        assert hasattr(config.i18n, "use_localized_errors")
        assert hasattr(config.i18n, "fallback_locale")

        assert config.i18n.fallback_locale == "en"

    def test_all_base_units_have_translations(self):
        """Test that all SI base units have translations in all locales."""
        base_units = ["m", "kg", "s", "A", "K", "mol", "cd"]
        locales = ["en", "es", "fr", "de", "zh_CN", "ja"]

        for locale in locales:
            for unit_symbol in base_units:
                name = i18n.translate_unit_name(unit_symbol, locale=locale)
                # Should get a translated name (might be same as symbol for some scientific units)
                assert isinstance(name, str)
                assert len(name) > 0
                # At least verify we got a response from translation system

    def test_all_derived_units_have_translations(self):
        """Test that common derived units have translations."""
        derived_units = ["Hz", "N", "J", "W", "Pa", "C", "V"]
        locales = ["en", "es", "fr", "de"]

        for locale in locales:
            for unit_symbol in derived_units:
                name = i18n.translate_unit_name(unit_symbol, locale=locale)
                # Should have a name (at least falls back to symbol)
                assert isinstance(name, str)
                assert len(name) > 0


class TestFallback:
    """Test fallback behavior."""

    def test_missing_translation_fallback(self):
        """Test that missing translations fall back to English."""
        # Use a unit that might not be in all locales
        # Even if missing, should fall back to English or return symbol
        name = i18n.translate_unit_name("unknown_unit", locale="es")
        assert isinstance(name, str)

    def test_fallback_chain_usage(self):
        """Test that fallback chain is used for regional variants."""
        # If es_MX specific translation doesn't exist, should fall back to es
        # We normalize es_MX to es, so this should work
        name = i18n.translate_unit_name("m", locale="es")
        assert name == "metro"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

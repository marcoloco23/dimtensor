"""Fuzz tests for dimtensor (v5.2.0 task #270).

These tests randomly generate inputs (well-formed or adversarial) and
verify that:
1. The library never panics with an unhelpful exception (only
   well-typed errors like ``DimensionError``, ``TypeError``, ``ValueError``).
2. Round-trips through serialization formats are lossless.
3. Stateful sequences of operations preserve unit semantics.

Fuzzing complements property-based tests by stressing the *inputs*: we
build malformed JSON, weird strings, NaN-laden arrays, and watch for
unexpected crashes.
"""

from __future__ import annotations

import io
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

from dimtensor import DimArray, Dimension, Unit, units
from dimtensor.errors import DimensionError, UnitConversionError
from dimtensor.io.json import from_dict, from_json, to_dict, to_json


# Allowed exceptions: all errors thrown by user-facing APIs MUST be one of
# these. If a fuzz test triggers anything else, that's a bug.
ALLOWED_EXCEPTIONS = (
    DimensionError,
    UnitConversionError,
    TypeError,
    ValueError,
    KeyError,
    OverflowError,
    ZeroDivisionError,
)


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------


def safe_floats() -> st.SearchStrategy[float]:
    """Floats unlikely to overflow when squared."""

    return st.floats(
        min_value=-1e3,
        max_value=1e3,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )


def positive_safe_floats() -> st.SearchStrategy[float]:
    """Strictly positive safe floats."""

    return st.floats(
        min_value=1e-3,
        max_value=1e3,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )


def small_int_exponent() -> st.SearchStrategy[int]:
    return st.integers(min_value=-3, max_value=3)


def random_dimension() -> st.SearchStrategy[Dimension]:
    return st.builds(
        Dimension,
        length=small_int_exponent(),
        mass=small_int_exponent(),
        time=small_int_exponent(),
        current=small_int_exponent(),
        temperature=small_int_exponent(),
        amount=small_int_exponent(),
        luminosity=small_int_exponent(),
    )


def random_unit() -> st.SearchStrategy[Unit]:
    return st.builds(
        lambda d, s: Unit("u", d, s), random_dimension(), positive_safe_floats()
    )


def random_array_data() -> st.SearchStrategy[list[float]]:
    return st.lists(safe_floats(), min_size=1, max_size=8)


def random_dim_array() -> st.SearchStrategy[DimArray]:
    return st.builds(
        lambda data, unit: DimArray(np.array(data, dtype=float), unit),
        random_array_data(),
        random_unit(),
    )


# ---------------------------------------------------------------------------
# JSON round-trip fuzzing
# ---------------------------------------------------------------------------


class TestJSONRoundTrip:
    """Round-tripping a DimArray through JSON should preserve all info."""

    @given(random_dim_array())
    def test_to_dict_from_dict_preserves(self, arr: DimArray) -> None:
        """from_dict(to_dict(arr)) recovers the same data and unit."""
        recovered = from_dict(to_dict(arr))
        np.testing.assert_allclose(
            recovered._data, arr._data, equal_nan=True
        )
        assert recovered.unit.dimension == arr.unit.dimension
        assert recovered.unit.scale == pytest.approx(arr.unit.scale)

    @given(random_dim_array())
    def test_to_json_from_json_preserves(self, arr: DimArray) -> None:
        """JSON string round-trip preserves data and unit."""
        recovered = from_json(to_json(arr))
        np.testing.assert_allclose(
            recovered._data, arr._data, equal_nan=True
        )
        assert recovered.unit.dimension == arr.unit.dimension

    @given(random_dim_array())
    def test_json_is_valid_utf8(self, arr: DimArray) -> None:
        """Generated JSON must parse back as a dict."""
        text = to_json(arr)
        decoded = json.loads(text)
        assert isinstance(decoded, dict)
        assert "data" in decoded
        assert "unit" in decoded


# ---------------------------------------------------------------------------
# Adversarial JSON input fuzzing
# ---------------------------------------------------------------------------


class TestMalformedJSONInput:
    """from_dict should fail gracefully on malformed input (no segfault)."""

    @given(st.text(min_size=0, max_size=200))
    def test_random_text_does_not_crash(self, text: str) -> None:
        """Garbage JSON strings must raise an allowed exception type."""
        try:
            from_json(text)
        except ALLOWED_EXCEPTIONS + (json.JSONDecodeError,):
            pass

    @given(
        st.dictionaries(
            keys=st.text(min_size=0, max_size=10),
            values=st.one_of(
                st.none(),
                st.integers(),
                st.text(),
                st.lists(st.integers(), max_size=4),
            ),
            max_size=8,
        )
    )
    def test_random_dict_does_not_crash(self, payload: dict) -> None:
        """Random dicts: from_dict must raise an allowed exception."""
        try:
            from_dict(payload)
        except ALLOWED_EXCEPTIONS:
            pass

    @given(random_dim_array(), st.text(min_size=0, max_size=20))
    def test_corrupted_unit_symbol_does_not_crash(
        self, arr: DimArray, garbage_symbol: str
    ) -> None:
        """Corrupting the unit symbol shouldn't break round-trip."""
        d = to_dict(arr)
        d["unit"]["symbol"] = garbage_symbol
        try:
            recovered = from_dict(d)
            assert recovered.unit.symbol == garbage_symbol
        except ALLOWED_EXCEPTIONS:
            pass


# ---------------------------------------------------------------------------
# Operation sequence fuzzing (stateful-style)
# ---------------------------------------------------------------------------


@st.composite
def operation_sequence(
    draw: st.DrawFn,
) -> tuple[DimArray, list[tuple[str, float]]]:
    """Generate a starting DimArray and a sequence of safe operations."""

    initial = draw(random_dim_array())
    n_ops = draw(st.integers(min_value=0, max_value=8))
    ops: list[tuple[str, float]] = []
    for _ in range(n_ops):
        op = draw(st.sampled_from(["mul", "div", "add_self", "neg", "pow_int"]))
        if op in ("mul", "div"):
            ops.append((op, draw(positive_safe_floats())))
        elif op == "pow_int":
            ops.append((op, draw(st.integers(min_value=1, max_value=3))))
        else:
            ops.append((op, 0.0))
    return initial, ops


class TestOperationSequence:
    """Apply random sequences of operations and check invariants."""

    @given(operation_sequence())
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_dimension_is_predictable(
        self, sequence: tuple[DimArray, list[tuple[str, float]]]
    ) -> None:
        """After any sequence, the dimension matches what we expect analytically."""
        arr, ops = sequence
        expected_dim = arr.unit.dimension
        cur = arr
        for op, val in ops:
            if op == "mul":
                cur = cur * val
            elif op == "div":
                # Avoid division by zero
                if val == 0:
                    continue
                cur = cur / val
            elif op == "add_self":
                cur = cur + cur
            elif op == "neg":
                cur = -cur
            elif op == "pow_int":
                expected_dim = expected_dim ** int(val)
                cur = cur ** int(val)
        assert cur.unit.dimension == expected_dim

    @given(random_dim_array(), random_dim_array())
    def test_addition_either_succeeds_or_raises(
        self, a: DimArray, b: DimArray
    ) -> None:
        """a + b must either succeed (matching dims) or raise DimensionError."""
        try:
            result = a + b
            assert result.unit.dimension == a.unit.dimension
            assert result.unit.dimension == b.unit.dimension
        except DimensionError:
            assert a.unit.dimension != b.unit.dimension
        except ValueError:
            # Shape mismatch - acceptable
            pass


# ---------------------------------------------------------------------------
# Dimension construction fuzzing (extreme exponents)
# ---------------------------------------------------------------------------


class TestDimensionFuzz:
    """Dimension should accept any rational exponent without crashing."""

    @given(
        st.integers(min_value=-1000, max_value=1000),
        st.integers(min_value=1, max_value=1000),
    )
    def test_fraction_construction(self, num: int, den: int) -> None:
        """Constructing with a Fraction shouldn't crash."""
        f = Fraction(num, den).limit_denominator(1000)
        dim = Dimension(length=f)
        assert dim.length == f

    @given(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    def test_float_construction(self, val: float) -> None:
        """Constructing with floats shouldn't crash and rounds to small denom."""
        dim = Dimension(mass=val)
        # The resulting fraction should have denom <= 1000
        assert dim.mass.denominator <= 1000

    @given(small_int_exponent(), small_int_exponent())
    def test_string_representation_is_str(self, a: int, b: int) -> None:
        """str() and repr() should always return strings."""
        dim = Dimension(length=a, mass=b)
        assert isinstance(str(dim), str)
        assert isinstance(repr(dim), str)


# ---------------------------------------------------------------------------
# Unit fuzzing
# ---------------------------------------------------------------------------


class TestUnitFuzz:
    """Unit operations should never panic with non-typed exceptions."""

    @given(random_unit(), st.floats(allow_nan=False, allow_infinity=False))
    def test_unit_scalar_multiplication_doesnt_crash(
        self, unit: Unit, scalar: float
    ) -> None:
        """unit * scalar should always work or raise an allowed exception."""
        try:
            _ = unit * scalar
        except ALLOWED_EXCEPTIONS:
            pass

    @given(random_unit(), small_int_exponent())
    def test_unit_power_doesnt_crash(self, unit: Unit, p: int) -> None:
        """unit ** p should always succeed for small integer exponents."""
        result = unit ** p
        assert result.dimension == unit.dimension ** p

    @given(random_unit(), random_unit())
    def test_conversion_factor_either_works_or_raises(
        self, a: Unit, b: Unit
    ) -> None:
        """conversion_factor either returns finite or raises ValueError."""
        try:
            f = a.conversion_factor(b)
            assert np.isfinite(f)
        except ValueError:
            assert not a.is_compatible(b)


# ---------------------------------------------------------------------------
# DimArray with NaN/inf data fuzzing
# ---------------------------------------------------------------------------


class TestDimArrayWithSpecialFloats:
    """DimArray should handle NaN and inf without crashing."""

    @given(
        st.lists(
            st.one_of(
                st.just(np.nan),
                st.just(np.inf),
                st.just(-np.inf),
                safe_floats(),
            ),
            min_size=1,
            max_size=6,
        ),
        random_unit(),
    )
    def test_special_floats_are_accepted(
        self, data: list[float], unit: Unit
    ) -> None:
        """DimArray should accept NaN/inf data without crashing."""
        arr = DimArray(np.array(data, dtype=float), unit)
        assert arr._data.shape == (len(data),)

    @given(
        st.lists(safe_floats(), min_size=1, max_size=4),
        random_unit(),
    )
    def test_negation_of_safe_floats_is_safe(
        self, data: list[float], unit: Unit
    ) -> None:
        """Negation should preserve shape and unit."""
        arr = DimArray(np.array(data, dtype=float), unit)
        neg = -arr
        np.testing.assert_array_equal(neg._data, -np.array(data, dtype=float))
        assert neg.unit == arr.unit


# Use a smaller default profile - fuzz tests can be slow.
settings.register_profile(
    "fuzz",
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.load_profile("fuzz")

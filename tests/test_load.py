"""Load tests: verify dimtensor stays performant under high load (v5.2.0 task #273).

These tests confirm that key operations still complete in reasonable wall-clock
time when given large arrays or repeated workloads. They are NOT intended to be
microbenchmarks (use ``benchmarks/`` for that). They flag *catastrophic*
performance regressions like accidental O(n²) loops.

Run with: ``pytest tests/test_load.py -v -m "not slow"``
or include the slow ones: ``pytest tests/test_load.py -v``
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from dimtensor import DimArray, Dimension, Unit, units


# Performance ceilings - chosen to catch O(n²) regressions, not to enforce
# specific micro-benchmarks. Generous so they pass on slow CI runners.
LARGE_ARRAY_SIZE = 1_000_000
HUGE_ARRAY_SIZE = 10_000_000
TIME_BUDGET_SECONDS_LARGE = 5.0
TIME_BUDGET_SECONDS_HUGE = 30.0


def _timed(func) -> tuple[float, object]:
    """Run a callable and return (elapsed_seconds, result)."""
    start = time.perf_counter()
    result = func()
    return time.perf_counter() - start, result


# ---------------------------------------------------------------------------
# Large array operations
# ---------------------------------------------------------------------------


class TestLargeArrayOperations:
    """Operations on large DimArrays should still finish quickly."""

    def test_addition_on_large_array(self) -> None:
        """Adding 1M-element arrays must finish under the time budget."""
        a = DimArray(np.arange(LARGE_ARRAY_SIZE, dtype=float), units.m)
        b = DimArray(np.arange(LARGE_ARRAY_SIZE, dtype=float), units.m)
        elapsed, result = _timed(lambda: a + b)
        assert result._data.shape == (LARGE_ARRAY_SIZE,)
        assert elapsed < TIME_BUDGET_SECONDS_LARGE, (
            f"addition took {elapsed:.2f}s (budget {TIME_BUDGET_SECONDS_LARGE}s)"
        )

    def test_multiplication_on_large_array(self) -> None:
        """Element-wise multiplication on 1M elements must be fast."""
        a = DimArray(np.arange(LARGE_ARRAY_SIZE, dtype=float), units.m)
        b = DimArray(np.arange(LARGE_ARRAY_SIZE, dtype=float), units.s)
        elapsed, result = _timed(lambda: a * b)
        assert result.unit.dimension == Dimension(length=1, time=1)
        assert elapsed < TIME_BUDGET_SECONDS_LARGE

    def test_scalar_multiplication_on_large_array(self) -> None:
        a = DimArray(np.arange(LARGE_ARRAY_SIZE, dtype=float), units.m)
        elapsed, result = _timed(lambda: a * 3.14)
        assert elapsed < TIME_BUDGET_SECONDS_LARGE

    def test_unit_conversion_on_large_array(self) -> None:
        a = DimArray(np.arange(LARGE_ARRAY_SIZE, dtype=float), units.m)
        elapsed, result = _timed(lambda: a.to(units.km))
        assert elapsed < TIME_BUDGET_SECONDS_LARGE


# ---------------------------------------------------------------------------
# Repeated operations stress
# ---------------------------------------------------------------------------


class TestRepeatedOperations:
    """Repeated small operations should not allocate unbounded memory."""

    def test_thousand_additions(self) -> None:
        """1000 additions on a moderate-size array must finish quickly."""
        a = DimArray(np.arange(1000, dtype=float), units.kg)
        start = time.perf_counter()
        result = a
        for _ in range(1000):
            result = result + a
        elapsed = time.perf_counter() - start
        # Generous budget: 1000 additions on 1000-element arrays.
        assert elapsed < 2.0, f"1000 additions took {elapsed:.2f}s"

    def test_thousand_dimension_constructions(self) -> None:
        """Dimension construction must be cheap (no surprising overhead)."""
        start = time.perf_counter()
        for i in range(1000):
            Dimension(length=i % 5, mass=i % 3, time=-i % 2)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"1000 Dimensions took {elapsed:.2f}s"

    def test_thousand_unit_conversions(self) -> None:
        """1000 conversions of a small array should be fast."""
        a = DimArray(np.arange(100, dtype=float), units.m)
        start = time.perf_counter()
        for _ in range(1000):
            _ = a.to(units.km)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"1000 conversions took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Memory pressure (huge arrays) - marked slow so excluded by default
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestHugeArrayOperations:
    """Stress tests with very large arrays - skipped by default."""

    def test_huge_addition(self) -> None:
        """10M element addition (~80 MB) should still complete."""
        a = DimArray(np.zeros(HUGE_ARRAY_SIZE, dtype=float), units.m)
        b = DimArray(np.ones(HUGE_ARRAY_SIZE, dtype=float), units.m)
        elapsed, result = _timed(lambda: a + b)
        assert result._data.shape == (HUGE_ARRAY_SIZE,)
        assert elapsed < TIME_BUDGET_SECONDS_HUGE


# ---------------------------------------------------------------------------
# Hash/equality stress (ensure Dimension can be a dict key at scale)
# ---------------------------------------------------------------------------


class TestDimensionHashStress:
    """Dimension hashes should be well-distributed enough for dict use."""

    def test_many_unique_dimensions_in_dict(self) -> None:
        """Putting 1000 unique dimensions in a dict should not collide badly."""
        d: dict[Dimension, int] = {}
        for i in range(-50, 50):
            for j in range(-10, 10):
                dim = Dimension(length=i, mass=j)
                d[dim] = i * 100 + j
        # We added 100 * 20 = 2000 entries; all should be retrievable.
        assert len(d) == 2000

        # Spot check
        sample = Dimension(length=3, mass=4)
        assert d[sample] == 3 * 100 + 4

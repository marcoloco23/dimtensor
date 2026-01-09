"""Competitor library benchmarks.

Compares dimtensor performance against other unit-aware libraries:
- pint: Popular Python units library
- astropy.units: Astronomy units library
- unyt: yt-project units library
"""

from __future__ import annotations

import numpy as np

# Import dimtensor
try:
    from dimtensor.core.dimarray import DimArray
    from dimtensor.core.units import m, s, kg
    DIMTENSOR_AVAILABLE = True
except ImportError:
    DIMTENSOR_AVAILABLE = False
    DimArray = None
    m = s = kg = None

# Try importing competitors
try:
    import pint
    PINT_AVAILABLE = True
    ureg = pint.UnitRegistry()
except ImportError:
    PINT_AVAILABLE = False
    pint = None
    ureg = None

try:
    import astropy.units as u
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    u = None

try:
    import unyt
    UNYT_AVAILABLE = True
except ImportError:
    UNYT_AVAILABLE = False
    unyt = None


class CreationComparison:
    """Compare array creation across libraries."""

    param_names = ['size']
    params = [[100, 10_000, 1_000_000]]

    def setup(self, size):
        """Set up test data."""
        self.data = np.random.randn(size)

    def time_creation_numpy_baseline(self, size):
        """Baseline: numpy array creation."""
        np.array(self.data)

    def time_creation_dimtensor(self, size):
        """DimArray creation."""
        if not DIMTENSOR_AVAILABLE:
            raise NotImplementedError("dimtensor not available")
        DimArray(self.data, m)

    def time_creation_pint(self, size):
        """Pint Quantity creation."""
        if not PINT_AVAILABLE:
            raise NotImplementedError("pint not available")
        self.data * ureg.meter

    def time_creation_astropy(self, size):
        """Astropy Quantity creation."""
        if not ASTROPY_AVAILABLE:
            raise NotImplementedError("astropy not available")
        self.data * u.m

    def time_creation_unyt(self, size):
        """Unyt array creation."""
        if not UNYT_AVAILABLE:
            raise NotImplementedError("unyt not available")
        unyt.unyt_array(self.data, 'm')


class AdditionComparison:
    """Compare addition operations across libraries."""

    param_names = ['size']
    params = [[100, 10_000, 1_000_000]]

    def setup(self, size):
        """Set up test arrays."""
        self.np_a = np.random.randn(size)
        self.np_b = np.random.randn(size)

        if DIMTENSOR_AVAILABLE:
            self.da_a = DimArray(self.np_a, m)
            self.da_b = DimArray(self.np_b, m)

        if PINT_AVAILABLE:
            self.pint_a = self.np_a * ureg.meter
            self.pint_b = self.np_b * ureg.meter

        if ASTROPY_AVAILABLE:
            self.astropy_a = self.np_a * u.m
            self.astropy_b = self.np_b * u.m

        if UNYT_AVAILABLE:
            self.unyt_a = unyt.unyt_array(self.np_a, 'm')
            self.unyt_b = unyt.unyt_array(self.np_b, 'm')

    def time_addition_numpy_baseline(self, size):
        """Baseline: numpy addition."""
        self.np_a + self.np_b

    def time_addition_dimtensor(self, size):
        """DimArray addition."""
        if not DIMTENSOR_AVAILABLE:
            raise NotImplementedError("dimtensor not available")
        self.da_a + self.da_b

    def time_addition_pint(self, size):
        """Pint addition."""
        if not PINT_AVAILABLE:
            raise NotImplementedError("pint not available")
        self.pint_a + self.pint_b

    def time_addition_astropy(self, size):
        """Astropy addition."""
        if not ASTROPY_AVAILABLE:
            raise NotImplementedError("astropy not available")
        self.astropy_a + self.astropy_b

    def time_addition_unyt(self, size):
        """Unyt addition."""
        if not UNYT_AVAILABLE:
            raise NotImplementedError("unyt not available")
        self.unyt_a + self.unyt_b


class MultiplicationComparison:
    """Compare multiplication operations across libraries."""

    param_names = ['size']
    params = [[100, 10_000, 1_000_000]]

    def setup(self, size):
        """Set up test arrays."""
        self.np_a = np.random.randn(size)
        self.np_b = np.random.randn(size)

        if DIMTENSOR_AVAILABLE:
            self.da_a = DimArray(self.np_a, m)
            self.da_b = DimArray(self.np_b, s)

        if PINT_AVAILABLE:
            self.pint_a = self.np_a * ureg.meter
            self.pint_b = self.np_b * ureg.second

        if ASTROPY_AVAILABLE:
            self.astropy_a = self.np_a * u.m
            self.astropy_b = self.np_b * u.s

        if UNYT_AVAILABLE:
            self.unyt_a = unyt.unyt_array(self.np_a, 'm')
            self.unyt_b = unyt.unyt_array(self.np_b, 's')

    def time_multiplication_numpy_baseline(self, size):
        """Baseline: numpy multiplication."""
        self.np_a * self.np_b

    def time_multiplication_dimtensor(self, size):
        """DimArray multiplication."""
        if not DIMTENSOR_AVAILABLE:
            raise NotImplementedError("dimtensor not available")
        self.da_a * self.da_b

    def time_multiplication_pint(self, size):
        """Pint multiplication."""
        if not PINT_AVAILABLE:
            raise NotImplementedError("pint not available")
        self.pint_a * self.pint_b

    def time_multiplication_astropy(self, size):
        """Astropy multiplication."""
        if not ASTROPY_AVAILABLE:
            raise NotImplementedError("astropy not available")
        self.astropy_a * self.astropy_b

    def time_multiplication_unyt(self, size):
        """Unyt multiplication."""
        if not UNYT_AVAILABLE:
            raise NotImplementedError("unyt not available")
        self.unyt_a * self.unyt_b


class ReductionComparison:
    """Compare reduction operations across libraries."""

    param_names = ['size']
    params = [[100, 10_000, 1_000_000]]

    def setup(self, size):
        """Set up test arrays."""
        self.np_a = np.random.randn(size)

        if DIMTENSOR_AVAILABLE:
            self.da_a = DimArray(self.np_a, m)

        if PINT_AVAILABLE:
            self.pint_a = self.np_a * ureg.meter

        if ASTROPY_AVAILABLE:
            self.astropy_a = self.np_a * u.m

        if UNYT_AVAILABLE:
            self.unyt_a = unyt.unyt_array(self.np_a, 'm')

    def time_sum_numpy_baseline(self, size):
        """Baseline: numpy sum."""
        self.np_a.sum()

    def time_sum_dimtensor(self, size):
        """DimArray sum."""
        if not DIMTENSOR_AVAILABLE:
            raise NotImplementedError("dimtensor not available")
        self.da_a.sum()

    def time_sum_pint(self, size):
        """Pint sum."""
        if not PINT_AVAILABLE:
            raise NotImplementedError("pint not available")
        self.pint_a.sum()

    def time_sum_astropy(self, size):
        """Astropy sum."""
        if not ASTROPY_AVAILABLE:
            raise NotImplementedError("astropy not available")
        self.astropy_a.sum()

    def time_sum_unyt(self, size):
        """Unyt sum."""
        if not UNYT_AVAILABLE:
            raise NotImplementedError("unyt not available")
        self.unyt_a.sum()


class UnitConversionComparison:
    """Compare unit conversion operations across libraries."""

    param_names = ['size']
    params = [[100, 10_000, 1_000_000]]

    def setup(self, size):
        """Set up test arrays."""
        self.np_a = np.random.randn(size)

        if DIMTENSOR_AVAILABLE:
            from dimtensor.core.units import km
            self.da_a = DimArray(self.np_a, m)
            self.km = km

        if PINT_AVAILABLE:
            self.pint_a = self.np_a * ureg.meter

        if ASTROPY_AVAILABLE:
            self.astropy_a = self.np_a * u.m

        if UNYT_AVAILABLE:
            self.unyt_a = unyt.unyt_array(self.np_a, 'm')

    def time_conversion_dimtensor(self, size):
        """DimArray unit conversion (m to km)."""
        if not DIMTENSOR_AVAILABLE:
            raise NotImplementedError("dimtensor not available")
        self.da_a.to(self.km)

    def time_conversion_pint(self, size):
        """Pint unit conversion (m to km)."""
        if not PINT_AVAILABLE:
            raise NotImplementedError("pint not available")
        self.pint_a.to(ureg.kilometer)

    def time_conversion_astropy(self, size):
        """Astropy unit conversion (m to km)."""
        if not ASTROPY_AVAILABLE:
            raise NotImplementedError("astropy not available")
        self.astropy_a.to(u.km)

    def time_conversion_unyt(self, size):
        """Unyt unit conversion (m to km)."""
        if not UNYT_AVAILABLE:
            raise NotImplementedError("unyt not available")
        self.unyt_a.to('km')


class ChainedOpsComparison:
    """Compare chained operations across libraries."""

    param_names = ['size']
    params = [[100, 10_000, 1_000_000]]

    def setup(self, size):
        """Set up test arrays."""
        self.np_a = np.random.randn(size)
        self.np_b = np.random.randn(size)
        self.np_c = np.random.randn(size) + 1

        if DIMTENSOR_AVAILABLE:
            self.da_a = DimArray(self.np_a, m)
            self.da_b = DimArray(self.np_b, m)
            self.da_c = DimArray(self.np_c, s)

        if PINT_AVAILABLE:
            self.pint_a = self.np_a * ureg.meter
            self.pint_b = self.np_b * ureg.meter
            self.pint_c = self.np_c * ureg.second

        if ASTROPY_AVAILABLE:
            self.astropy_a = self.np_a * u.m
            self.astropy_b = self.np_b * u.m
            self.astropy_c = self.np_c * u.s

        if UNYT_AVAILABLE:
            self.unyt_a = unyt.unyt_array(self.np_a, 'm')
            self.unyt_b = unyt.unyt_array(self.np_b, 'm')
            self.unyt_c = unyt.unyt_array(self.np_c, 's')

    def time_chained_numpy_baseline(self, size):
        """Baseline: numpy chained ops."""
        ((self.np_a + self.np_b) * 2.0) / self.np_c

    def time_chained_dimtensor(self, size):
        """DimArray chained ops."""
        if not DIMTENSOR_AVAILABLE:
            raise NotImplementedError("dimtensor not available")
        ((self.da_a + self.da_b) * 2.0) / self.da_c

    def time_chained_pint(self, size):
        """Pint chained ops."""
        if not PINT_AVAILABLE:
            raise NotImplementedError("pint not available")
        ((self.pint_a + self.pint_b) * 2.0) / self.pint_c

    def time_chained_astropy(self, size):
        """Astropy chained ops."""
        if not ASTROPY_AVAILABLE:
            raise NotImplementedError("astropy not available")
        ((self.astropy_a + self.astropy_b) * 2.0) / self.astropy_c

    def time_chained_unyt(self, size):
        """Unyt chained ops."""
        if not UNYT_AVAILABLE:
            raise NotImplementedError("unyt not available")
        ((self.unyt_a + self.unyt_b) * 2.0) / self.unyt_c

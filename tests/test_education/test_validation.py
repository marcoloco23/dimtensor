"""Tests for validation module."""

import numpy as np
import pytest

from dimtensor import DimArray, units
from dimtensor.core.dimensions import Dimension
from dimtensor.education.validation import AnswerValidator


class TestAnswerValidator:
    """Tests for AnswerValidator."""

    def test_validate_numeric_correct(self):
        """Test validating correct numeric answer."""
        validator = AnswerValidator()
        result = validator.validate_numeric(10.0, 10.0, tolerance=0.01)

        assert result.valid is True
        assert "correct" in result.message.lower()

    def test_validate_numeric_within_tolerance(self):
        """Test answer within tolerance."""
        validator = AnswerValidator()
        result = validator.validate_numeric(10.05, 10.0, tolerance=0.01)

        assert result.valid is True

    def test_validate_numeric_outside_tolerance(self):
        """Test answer outside tolerance."""
        validator = AnswerValidator()
        result = validator.validate_numeric(11.0, 10.0, tolerance=0.01)

        assert result.valid is False
        assert "incorrect" in result.message.lower()

    def test_validate_numeric_zero(self):
        """Test validation with zero."""
        validator = AnswerValidator()

        # Correct zero
        result = validator.validate_numeric(0.0, 0.0)
        assert result.valid is True

        # Near zero
        result = validator.validate_numeric(1e-15, 0.0)
        assert result.valid is True

        # Not zero
        result = validator.validate_numeric(0.1, 0.0)
        assert result.valid is False

    def test_validate_numeric_with_units_correct(self):
        """Test validation with correct units."""
        validator = AnswerValidator()

        answer = DimArray([10.0], units.m)
        correct = DimArray([10.0], units.m)

        result = validator.validate_numeric_with_units(answer, correct, tolerance=0.01)
        assert result.valid is True

    def test_validate_numeric_with_units_conversion(self):
        """Test validation with unit conversion."""
        validator = AnswerValidator()

        answer = DimArray([1.0], units.km)
        correct = DimArray([1000.0], units.m)

        result = validator.validate_numeric_with_units(answer, correct, tolerance=0.01)
        assert result.valid is True

    def test_validate_numeric_with_wrong_dimension(self):
        """Test validation with wrong dimension."""
        validator = AnswerValidator()

        answer = DimArray([10.0], units.m)
        correct = DimArray([10.0], units.s)

        result = validator.validate_numeric_with_units(answer, correct, tolerance=0.01)
        assert result.valid is False
        assert "dimension" in result.message.lower()

    def test_validate_numeric_dimensionless_when_should_have_units(self):
        """Test error when answer lacks units."""
        validator = AnswerValidator()

        answer = 10.0  # No units
        correct = DimArray([10.0], units.m)

        result = validator.validate_numeric_with_units(answer, correct, tolerance=0.01)
        assert result.valid is False
        assert "must have units" in result.message.lower()

    def test_validate_dimension_correct(self):
        """Test dimension validation."""
        validator = AnswerValidator()

        dim = Dimension(length=1, mass=0, time=-1)  # velocity dimension
        result = validator.validate_dimension(dim, dim)

        assert result.valid is True

    def test_validate_dimension_from_dimarray(self):
        """Test dimension validation from DimArray."""
        validator = AnswerValidator()

        velocity = DimArray([10], units.m / units.s)
        correct_dim = Dimension(length=1, mass=0, time=-1)

        result = validator.validate_dimension(velocity, correct_dim)
        assert result.valid is True

    def test_validate_dimension_incorrect(self):
        """Test incorrect dimension."""
        validator = AnswerValidator()

        dim1 = Dimension(length=1, mass=0, time=-1)  # velocity
        dim2 = Dimension(length=1, mass=0, time=-2)  # acceleration

        result = validator.validate_dimension(dim1, dim2)
        assert result.valid is False

    def test_validate_array_correct(self):
        """Test array validation."""
        validator = AnswerValidator()

        answer = DimArray([1, 2, 3], units.m)
        correct = DimArray([1, 2, 3], units.m)

        result = validator.validate_array(answer, correct, tolerance=0.01)
        assert result.valid is True

    def test_validate_array_with_tolerance(self):
        """Test array validation within tolerance."""
        validator = AnswerValidator()

        answer = DimArray([1.01, 2.01, 3.01], units.m)
        correct = DimArray([1, 2, 3], units.m)

        result = validator.validate_array(answer, correct, tolerance=0.02)
        assert result.valid is True

    def test_validate_array_wrong_shape(self):
        """Test array with wrong shape."""
        validator = AnswerValidator()

        answer = DimArray([1, 2], units.m)
        correct = DimArray([1, 2, 3], units.m)

        result = validator.validate_array(answer, correct, tolerance=0.01)
        assert result.valid is False
        assert "shape" in result.message.lower()

    def test_validate_array_wrong_values(self):
        """Test array with wrong values."""
        validator = AnswerValidator()

        answer = DimArray([1, 2, 10], units.m)
        correct = DimArray([1, 2, 3], units.m)

        result = validator.validate_array(answer, correct, tolerance=0.01)
        assert result.valid is False
        assert result.details["num_incorrect"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for exercise types."""

import pytest

from dimtensor import DimArray, units
from dimtensor.core.dimensions import Dimension
from dimtensor.education import (
    CodeExercise,
    DimensionalAnalysisExercise,
    MultipleChoiceExercise,
    NumericAnswerExercise,
    UnitConversionExercise,
    WordProblem,
)


class TestMultipleChoiceExercise:
    """Tests for MultipleChoiceExercise."""

    def test_correct_answer(self):
        """Test correct answer returns success."""
        ex = MultipleChoiceExercise(
            id="test1",
            question="What is 2 + 2?",
            choices=["3", "4", "5", "6"],
            correct_index=1,
        )

        result = ex.check(1)
        assert result.correct is True
        assert result.score == 1.0

    def test_incorrect_answer(self):
        """Test incorrect answer returns failure."""
        ex = MultipleChoiceExercise(
            id="test2",
            question="What is 2 + 2?",
            choices=["3", "4", "5", "6"],
            correct_index=1,
        )

        result = ex.check(2)
        assert result.correct is False
        assert result.score == 0.0

    def test_invalid_index(self):
        """Test invalid index returns error."""
        ex = MultipleChoiceExercise(
            id="test3",
            question="What is 2 + 2?",
            choices=["3", "4", "5", "6"],
            correct_index=1,
        )

        result = ex.check(10)
        assert result.correct is False
        assert "must be between" in result.message

    def test_hints(self):
        """Test hint system."""
        ex = MultipleChoiceExercise(
            id="test4",
            question="What is 2 + 2?",
            choices=["3", "4", "5", "6"],
            correct_index=1,
            hints=["Think about basic arithmetic", "The answer is 4"],
        )

        hint1 = ex.get_hint()
        assert hint1 == "Think about basic arithmetic"

        hint2 = ex.get_hint()
        assert hint2 == "The answer is 4"

        hint3 = ex.get_hint()
        assert hint3 is None


class TestNumericAnswerExercise:
    """Tests for NumericAnswerExercise."""

    def test_correct_dimensionless_answer(self):
        """Test correct dimensionless answer."""
        ex = NumericAnswerExercise(
            id="test5",
            question="What is 5 * 3?",
            correct_answer=15.0,
            tolerance=0.01,
        )

        result = ex.check(15.0)
        assert result.correct is True
        assert result.score == 1.0

    def test_answer_within_tolerance(self):
        """Test answer within tolerance is accepted."""
        ex = NumericAnswerExercise(
            id="test6",
            question="What is pi?",
            correct_answer=3.14159,
            tolerance=0.01,
        )

        result = ex.check(3.14)
        assert result.correct is True

    def test_answer_outside_tolerance(self):
        """Test answer outside tolerance is rejected."""
        ex = NumericAnswerExercise(
            id="test7",
            question="What is pi?",
            correct_answer=3.14159,
            tolerance=0.001,
        )

        result = ex.check(3.0)
        assert result.correct is False

    def test_correct_answer_with_units(self):
        """Test correct answer with units."""
        ex = NumericAnswerExercise(
            id="test8",
            question="Calculate velocity",
            correct_answer=DimArray([10.0], units.m / units.s),
            tolerance=0.01,
        )

        answer = DimArray([10.0], units.m / units.s)
        result = ex.check(answer)
        assert result.correct is True

    def test_wrong_units(self):
        """Test answer with wrong units is rejected."""
        ex = NumericAnswerExercise(
            id="test9",
            question="Calculate velocity",
            correct_answer=DimArray([10.0], units.m / units.s),
            tolerance=0.01,
        )

        answer = DimArray([10.0], units.m)
        result = ex.check(answer)
        assert result.correct is False
        assert "dimension" in result.message.lower()

    def test_unit_conversion(self):
        """Test that equivalent units are accepted."""
        ex = NumericAnswerExercise(
            id="test10",
            question="Distance",
            correct_answer=DimArray([1000.0], units.m),
            tolerance=0.01,
        )

        # Answer in kilometers
        answer = DimArray([1.0], units.km)
        result = ex.check(answer)
        assert result.correct is True


class TestDimensionalAnalysisExercise:
    """Tests for DimensionalAnalysisExercise."""

    def test_correct_dimension(self):
        """Test correct dimension is accepted."""
        # Force has dimension M L T^-2
        force_dim = Dimension(length=1, mass=1, time=-2)

        ex = DimensionalAnalysisExercise(
            id="test11", question="What is the dimension of force?", correct_dimension=force_dim
        )

        result = ex.check(force_dim)
        assert result.correct is True

    def test_dimension_from_dimarray(self):
        """Test extracting dimension from DimArray."""
        force_dim = Dimension(length=1, mass=1, time=-2)

        ex = DimensionalAnalysisExercise(
            id="test12", question="What is the dimension of force?", correct_dimension=force_dim
        )

        force = DimArray([10], units.N)
        result = ex.check(force)
        assert result.correct is True

    def test_incorrect_dimension(self):
        """Test incorrect dimension is rejected."""
        force_dim = Dimension(length=1, mass=1, time=-2)
        energy_dim = Dimension(length=2, mass=1, time=-2)

        ex = DimensionalAnalysisExercise(
            id="test13", question="What is the dimension of force?", correct_dimension=force_dim
        )

        result = ex.check(energy_dim)
        assert result.correct is False


class TestCodeExercise:
    """Tests for CodeExercise."""

    def test_all_tests_pass(self):
        """Test code that passes all tests."""

        def test1(namespace):
            return "add" in namespace and namespace["add"](2, 3) == 5

        def test2(namespace):
            return namespace["add"](0, 0) == 0

        ex = CodeExercise(
            id="test14",
            question="Write a function that adds two numbers",
            test_cases=[test1, test2],
        )

        code = """
def add(a, b):
    return a + b
"""

        result = ex.check(code)
        assert result.correct is True
        assert result.score == 1.0

    def test_some_tests_fail(self):
        """Test code that fails some tests."""

        def test1(namespace):
            return "add" in namespace and namespace["add"](2, 3) == 5

        def test2(namespace):
            return namespace["add"](0, 0) == 0

        ex = CodeExercise(
            id="test15",
            question="Write a function that adds two numbers",
            test_cases=[test1, test2],
        )

        code = """
def add(a, b):
    return a + b + 1  # Wrong!
"""

        result = ex.check(code)
        assert result.correct is False
        assert result.score == 0.0  # Both tests fail

    def test_forbidden_functions(self):
        """Test that forbidden functions are caught."""
        ex = CodeExercise(
            id="test16",
            question="Implement your own max function",
            test_cases=[lambda ns: True],
            forbidden_functions=["max"],
        )

        code = "result = max([1, 2, 3])"

        result = ex.check(code)
        assert result.correct is False
        assert "forbidden" in result.message


class TestUnitConversionExercise:
    """Tests for UnitConversionExercise."""

    def test_correct_conversion(self):
        """Test correct unit conversion."""
        ex = UnitConversionExercise(
            id="test17",
            question="Convert 5 km to meters",
            value=5.0,
            from_unit=units.km,
            to_unit=units.m,
            correct_answer=5000.0,
            tolerance=0.01,
        )

        result = ex.check(5000.0)
        assert result.correct is True

    def test_conversion_with_dimarray(self):
        """Test conversion with DimArray answer."""
        ex = UnitConversionExercise(
            id="test18",
            question="Convert 100 cm to meters",
            value=100.0,
            from_unit=units.cm,
            to_unit=units.m,
            correct_answer=1.0,
            tolerance=0.01,
        )

        answer = DimArray([1.0], units.m)
        result = ex.check(answer)
        assert result.correct is True

    def test_wrong_units_in_answer(self):
        """Test that wrong units are caught."""
        ex = UnitConversionExercise(
            id="test19",
            question="Convert 5 km to meters",
            value=5.0,
            from_unit=units.km,
            to_unit=units.m,
            correct_answer=5000.0,
            tolerance=0.01,
        )

        # Answer in km instead of m
        answer = DimArray([5.0], units.km)
        result = ex.check(answer)
        assert result.correct is False


class TestWordProblem:
    """Tests for WordProblem."""

    def test_correct_final_answer(self):
        """Test correct final answer."""
        ex = WordProblem(
            id="test20",
            question="A car travels 100 m in 10 s. What is its velocity?",
            steps=["Identify distance and time", "Use v = d/t", "Calculate result"],
            correct_answer=DimArray([10.0], units.m / units.s),
            tolerance=0.01,
        )

        answer = DimArray([10.0], units.m / units.s)
        result = ex.check(answer)
        assert result.correct is True

    def test_incorrect_final_answer(self):
        """Test incorrect final answer."""
        ex = WordProblem(
            id="test21",
            question="A car travels 100 m in 10 s. What is its velocity?",
            steps=["Identify distance and time", "Use v = d/t", "Calculate result"],
            correct_answer=DimArray([10.0], units.m / units.s),
            tolerance=0.01,
        )

        answer = DimArray([20.0], units.m / units.s)
        result = ex.check(answer)
        assert result.correct is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

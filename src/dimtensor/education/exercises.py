"""Exercise types for the interactive textbook.

This module defines various types of exercises including multiple choice,
numeric answers, code exercises, and dimensional analysis problems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.dimarray import DimArray
from ..core.dimensions import Dimension
from ..core.units import Unit
from .validation import AnswerValidator, ValidationResult


@dataclass
class ExerciseResult:
    """Result of checking an exercise answer.

    Attributes:
        correct: Whether the answer was correct.
        message: Feedback message for the student.
        score: Numeric score (0.0 to 1.0).
        details: Additional details about the result.
    """

    correct: bool
    message: str
    score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


class Exercise(ABC):
    """Base class for all exercise types.

    Attributes:
        id: Unique identifier for the exercise.
        question: The question text.
        difficulty: Difficulty level (1-5).
        topics: List of topic tags.
        hints: List of hints (progressively more detailed).
        solution_text: Full solution explanation.
    """

    def __init__(
        self,
        id: str,
        question: str,
        difficulty: int = 1,
        topics: list[str] | None = None,
        hints: list[str] | None = None,
        solution_text: str = "",
    ):
        """Initialize an exercise.

        Args:
            id: Unique identifier for the exercise.
            question: The question text.
            difficulty: Difficulty level (1-5).
            topics: List of topic tags.
            hints: List of hints (progressively more detailed).
            solution_text: Full solution explanation.
        """
        self.id = id
        self.question = question
        self.difficulty = difficulty
        self.topics = topics or []
        self.hints = hints or []
        self.solution_text = solution_text
        self._hints_revealed = 0

    @abstractmethod
    def check(self, answer: Any) -> ExerciseResult:
        """Check if the student's answer is correct.

        Args:
            answer: The student's answer.

        Returns:
            ExerciseResult with feedback.
        """
        pass

    def get_hint(self) -> str | None:
        """Get the next hint for this exercise.

        Returns:
            The next hint, or None if no more hints available.
        """
        if self._hints_revealed >= len(self.hints):
            return None
        hint = self.hints[self._hints_revealed]
        self._hints_revealed += 1
        return hint

    def reset_hints(self) -> None:
        """Reset the hint counter."""
        self._hints_revealed = 0

    def reveal_solution(self) -> str:
        """Reveal the full solution."""
        return self.solution_text


class MultipleChoiceExercise(Exercise):
    """Multiple choice exercise.

    Attributes:
        choices: List of possible answers.
        correct_index: Index of the correct answer.
        explanations: Optional explanations for each choice.
    """

    def __init__(
        self,
        id: str,
        question: str,
        choices: list[str],
        correct_index: int,
        explanations: list[str] | None = None,
        **kwargs,
    ):
        """Initialize a multiple choice exercise.

        Args:
            id: Unique identifier.
            question: The question text.
            choices: List of possible answers.
            correct_index: Index of the correct answer (0-based).
            explanations: Optional explanations for each choice.
            **kwargs: Additional arguments passed to Exercise.__init__.
        """
        super().__init__(id, question, **kwargs)
        self.choices = choices
        self.correct_index = correct_index
        self.explanations = explanations or []

    def check(self, answer: int) -> ExerciseResult:
        """Check if the selected answer is correct.

        Args:
            answer: The index of the selected answer.

        Returns:
            ExerciseResult with feedback.
        """
        if not isinstance(answer, int):
            return ExerciseResult(
                correct=False, message="Answer must be an integer index.", score=0.0
            )

        if answer < 0 or answer >= len(self.choices):
            return ExerciseResult(
                correct=False,
                message=f"Answer must be between 0 and {len(self.choices) - 1}.",
                score=0.0,
            )

        correct = answer == self.correct_index
        score = 1.0 if correct else 0.0

        # Build message
        if correct:
            message = f"Correct! The answer is: {self.choices[answer]}"
        else:
            message = (
                f"Incorrect. You selected: {self.choices[answer]}. "
                f"The correct answer is: {self.choices[self.correct_index]}"
            )

        # Add explanation if available
        if self.explanations and answer < len(self.explanations):
            message += f"\n\nExplanation: {self.explanations[answer]}"

        return ExerciseResult(correct=correct, message=message, score=score)


class NumericAnswerExercise(Exercise):
    """Numeric answer exercise with optional units.

    Attributes:
        correct_answer: The correct answer (number or DimArray).
        tolerance: Acceptable relative tolerance (default 0.01 = 1%).
        unit: Expected unit if answer should have units.
    """

    def __init__(
        self,
        id: str,
        question: str,
        correct_answer: float | DimArray,
        tolerance: float = 0.01,
        unit: Unit | None = None,
        **kwargs,
    ):
        """Initialize a numeric answer exercise.

        Args:
            id: Unique identifier.
            question: The question text.
            correct_answer: The correct answer (number or DimArray).
            tolerance: Acceptable relative tolerance (default 0.01 = 1%).
            unit: Expected unit if answer should have units.
            **kwargs: Additional arguments passed to Exercise.__init__.
        """
        super().__init__(id, question, **kwargs)
        self.correct_answer = correct_answer
        self.tolerance = tolerance
        self.unit = unit

    def check(self, answer: float | DimArray) -> ExerciseResult:
        """Check if the numeric answer is correct within tolerance.

        Args:
            answer: The student's answer (number or DimArray).

        Returns:
            ExerciseResult with feedback.
        """
        validator = AnswerValidator()

        # If correct answer is a DimArray, validate with units
        if isinstance(self.correct_answer, DimArray):
            validation = validator.validate_numeric_with_units(
                answer, self.correct_answer, self.tolerance
            )
        else:
            # Dimensionless numeric answer
            validation = validator.validate_numeric(
                answer, self.correct_answer, self.tolerance
            )

        return ExerciseResult(
            correct=validation.valid,
            message=validation.message,
            score=1.0 if validation.valid else 0.0,
            details=validation.details,
        )


class CodeExercise(Exercise):
    """Code exercise where students write Python code.

    Attributes:
        test_cases: List of test functions that validate the code.
        template: Optional code template to start with.
        forbidden_functions: List of forbidden function names.
    """

    def __init__(
        self,
        id: str,
        question: str,
        test_cases: list[Callable],
        template: str = "",
        forbidden_functions: list[str] | None = None,
        **kwargs,
    ):
        """Initialize a code exercise.

        Args:
            id: Unique identifier.
            question: The question text.
            test_cases: List of test functions that return True if passed.
            template: Optional code template to start with.
            forbidden_functions: List of forbidden function names.
            **kwargs: Additional arguments passed to Exercise.__init__.
        """
        super().__init__(id, question, **kwargs)
        self.test_cases = test_cases
        self.template = template
        self.forbidden_functions = forbidden_functions or []

    def check(self, answer: str | Callable) -> ExerciseResult:
        """Check if the code passes all test cases.

        Args:
            answer: Either code string to execute or a callable function.

        Returns:
            ExerciseResult with feedback.
        """
        # Check for forbidden functions
        if isinstance(answer, str):
            for forbidden in self.forbidden_functions:
                if forbidden in answer:
                    return ExerciseResult(
                        correct=False,
                        message=f"Your code uses the forbidden function: {forbidden}",
                        score=0.0,
                    )

        # Run test cases
        passed_tests = 0
        failed_test_messages = []

        for i, test in enumerate(self.test_cases):
            try:
                if isinstance(answer, str):
                    # Execute code string
                    local_namespace = {}
                    exec(answer, local_namespace)
                    # Assume the function is named as expected by test
                    result = test(local_namespace)
                else:
                    # Direct function call
                    result = test(answer)

                if result:
                    passed_tests += 1
                else:
                    failed_test_messages.append(f"Test {i + 1} failed")
            except Exception as e:
                failed_test_messages.append(f"Test {i + 1} raised error: {str(e)}")

        total_tests = len(self.test_cases)
        score = passed_tests / total_tests if total_tests > 0 else 0.0
        correct = score == 1.0

        if correct:
            message = f"Excellent! All {total_tests} tests passed."
        else:
            message = (
                f"Passed {passed_tests}/{total_tests} tests.\n"
                + "\n".join(failed_test_messages)
            )

        return ExerciseResult(
            correct=correct,
            message=message,
            score=score,
            details={"passed": passed_tests, "total": total_tests},
        )


class DimensionalAnalysisExercise(Exercise):
    """Exercise for dimensional analysis problems.

    Attributes:
        correct_dimension: The correct dimension.
    """

    def __init__(
        self,
        id: str,
        question: str,
        correct_dimension: Dimension,
        **kwargs,
    ):
        """Initialize a dimensional analysis exercise.

        Args:
            id: Unique identifier.
            question: The question text.
            correct_dimension: The correct dimension.
            **kwargs: Additional arguments passed to Exercise.__init__.
        """
        super().__init__(id, question, **kwargs)
        self.correct_dimension = correct_dimension

    def check(self, answer: Dimension | DimArray) -> ExerciseResult:
        """Check if the derived dimension is correct.

        Args:
            answer: The student's dimension (Dimension or DimArray).

        Returns:
            ExerciseResult with feedback.
        """
        if isinstance(answer, DimArray):
            answer_dim = answer.unit.dimension
        elif isinstance(answer, Dimension):
            answer_dim = answer
        else:
            return ExerciseResult(
                correct=False,
                message="Answer must be a Dimension or DimArray.",
                score=0.0,
            )

        correct = answer_dim == self.correct_dimension

        if correct:
            message = f"Correct! The dimension is {self.correct_dimension}."
        else:
            message = (
                f"Incorrect. You got {answer_dim}, "
                f"but the correct dimension is {self.correct_dimension}."
            )

        return ExerciseResult(correct=correct, message=message, score=1.0 if correct else 0.0)


class UnitConversionExercise(Exercise):
    """Exercise for unit conversion problems.

    Attributes:
        value: The value to convert.
        from_unit: The starting unit.
        to_unit: The target unit.
        correct_answer: The correct answer after conversion.
        tolerance: Acceptable relative tolerance.
    """

    def __init__(
        self,
        id: str,
        question: str,
        value: float,
        from_unit: Unit,
        to_unit: Unit,
        correct_answer: float,
        tolerance: float = 0.01,
        **kwargs,
    ):
        """Initialize a unit conversion exercise.

        Args:
            id: Unique identifier.
            question: The question text.
            value: The value to convert.
            from_unit: The starting unit.
            to_unit: The target unit.
            correct_answer: The correct answer after conversion.
            tolerance: Acceptable relative tolerance.
            **kwargs: Additional arguments passed to Exercise.__init__.
        """
        super().__init__(id, question, **kwargs)
        self.value = value
        self.from_unit = from_unit
        self.to_unit = to_unit
        self.correct_answer = correct_answer
        self.tolerance = tolerance

    def check(self, answer: float | DimArray) -> ExerciseResult:
        """Check if the conversion is correct.

        Args:
            answer: The student's answer (number or DimArray).

        Returns:
            ExerciseResult with feedback.
        """
        validator = AnswerValidator()

        # Extract numeric value if DimArray
        if isinstance(answer, DimArray):
            # Check if units are correct - must be in target unit
            if answer.unit != self.to_unit:
                # Check if it's at least the right dimension
                if answer.unit.dimension != self.to_unit.dimension:
                    return ExerciseResult(
                        correct=False,
                        message=f"Wrong dimension. Expected {self.to_unit}, got {answer.unit}.",
                        score=0.0,
                    )
                else:
                    # Right dimension, wrong specific unit
                    return ExerciseResult(
                        correct=False,
                        message=f"Answer should be in {self.to_unit}, not {answer.unit}. Did you complete the conversion?",
                        score=0.0,
                    )
            # Convert to target unit for comparison
            converted = answer.to(self.to_unit).data
            answer_value = float(converted.flat[0]) if converted.size == 1 else float(converted)
        else:
            answer_value = float(answer)

        validation = validator.validate_numeric(
            answer_value, self.correct_answer, self.tolerance
        )

        return ExerciseResult(
            correct=validation.valid,
            message=validation.message,
            score=1.0 if validation.valid else 0.0,
            details=validation.details,
        )


class WordProblem(Exercise):
    """Multi-step word problem.

    Attributes:
        steps: List of step descriptions.
        correct_answer: The final correct answer.
        tolerance: Acceptable relative tolerance.
        partial_credit: Whether to award partial credit for intermediate steps.
    """

    def __init__(
        self,
        id: str,
        question: str,
        steps: list[str],
        correct_answer: float | DimArray,
        tolerance: float = 0.01,
        partial_credit: bool = True,
        **kwargs,
    ):
        """Initialize a word problem.

        Args:
            id: Unique identifier.
            question: The question text.
            steps: List of step descriptions.
            correct_answer: The final correct answer.
            tolerance: Acceptable relative tolerance.
            partial_credit: Whether to award partial credit.
            **kwargs: Additional arguments passed to Exercise.__init__.
        """
        super().__init__(id, question, **kwargs)
        self.steps = steps
        self.correct_answer = correct_answer
        self.tolerance = tolerance
        self.partial_credit = partial_credit

    def check(self, answer: float | DimArray) -> ExerciseResult:
        """Check if the final answer is correct.

        Args:
            answer: The student's final answer.

        Returns:
            ExerciseResult with feedback.
        """
        validator = AnswerValidator()

        # Check answer similar to NumericAnswerExercise
        if isinstance(self.correct_answer, DimArray):
            validation = validator.validate_numeric_with_units(
                answer, self.correct_answer, self.tolerance
            )
        else:
            validation = validator.validate_numeric(
                answer, self.correct_answer, self.tolerance
            )

        message = validation.message
        if validation.valid:
            message += f"\n\nThis problem required {len(self.steps)} steps to solve."

        return ExerciseResult(
            correct=validation.valid,
            message=message,
            score=1.0 if validation.valid else 0.0,
            details=validation.details,
        )

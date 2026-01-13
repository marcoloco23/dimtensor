"""Answer validation with dimensional checking.

This module provides validation logic for student answers, with special
support for dimensional correctness checking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.dimarray import DimArray
from ..core.dimensions import Dimension
from ..errors import DimensionError


@dataclass
class ValidationResult:
    """Result of validating a student answer.

    Attributes:
        valid: Whether the answer is valid.
        message: Feedback message.
        details: Additional details about the validation.
    """

    valid: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class AnswerValidator:
    """Validator for student answers with dimensional checking."""

    def validate_numeric(
        self, answer: float, correct: float, tolerance: float = 0.01
    ) -> ValidationResult:
        """Validate a dimensionless numeric answer.

        Args:
            answer: The student's answer.
            correct: The correct answer.
            tolerance: Relative tolerance for comparison.

        Returns:
            ValidationResult indicating if answer is within tolerance.
        """
        try:
            answer_val = float(answer)
            correct_val = float(correct)

            # Handle zero case
            if correct_val == 0:
                if abs(answer_val) < 1e-10:
                    return ValidationResult(
                        valid=True,
                        message="Correct!",
                        details={"answer": answer_val, "correct": correct_val},
                    )
                else:
                    return ValidationResult(
                        valid=False,
                        message=f"Incorrect. Expected 0, got {answer_val}.",
                        details={"answer": answer_val, "correct": correct_val},
                    )

            # Calculate relative error
            rel_error = abs((answer_val - correct_val) / correct_val)

            if rel_error <= tolerance:
                return ValidationResult(
                    valid=True,
                    message=f"Correct! (Your answer: {answer_val}, within {tolerance * 100}% tolerance)",
                    details={
                        "answer": answer_val,
                        "correct": correct_val,
                        "rel_error": rel_error,
                    },
                )
            else:
                return ValidationResult(
                    valid=False,
                    message=f"Incorrect. Your answer: {answer_val}, Correct answer: {correct_val} (relative error: {rel_error * 100:.2f}%)",
                    details={
                        "answer": answer_val,
                        "correct": correct_val,
                        "rel_error": rel_error,
                    },
                )

        except (TypeError, ValueError) as e:
            return ValidationResult(
                valid=False, message=f"Invalid numeric answer: {str(e)}", details={}
            )

    def validate_numeric_with_units(
        self, answer: DimArray | float, correct: DimArray, tolerance: float = 0.01
    ) -> ValidationResult:
        """Validate a numeric answer with units.

        Args:
            answer: The student's answer (DimArray or number).
            correct: The correct answer (DimArray).
            tolerance: Relative tolerance for comparison.

        Returns:
            ValidationResult indicating if answer matches correct value and units.
        """
        # Check if answer is dimensionless when it shouldn't be
        if not isinstance(answer, DimArray):
            return ValidationResult(
                valid=False,
                message=f"Answer must have units. Expected unit: {correct.unit}",
                details={"expected_unit": str(correct.unit)},
            )

        # Check dimensional compatibility
        if answer.unit.dimension != correct.unit.dimension:
            return ValidationResult(
                valid=False,
                message=f"Incorrect dimensions. Your answer has dimension {answer.unit.dimension}, but correct answer has dimension {correct.unit.dimension}.",
                details={
                    "answer_dimension": str(answer.unit.dimension),
                    "correct_dimension": str(correct.unit.dimension),
                },
            )

        # Convert answer to same unit as correct answer for comparison
        try:
            answer_converted = answer.to(correct.unit)
        except Exception as e:
            return ValidationResult(
                valid=False,
                message=f"Error converting units: {str(e)}",
                details={},
            )

        # Compare numeric values
        answer_val = float(answer_converted.data.flat[0]) if answer_converted.data.size == 1 else float(answer_converted.data)
        correct_val = float(correct.data.flat[0]) if correct.data.size == 1 else float(correct.data)

        # Handle zero case
        if correct_val == 0:
            if abs(answer_val) < 1e-10:
                return ValidationResult(
                    valid=True,
                    message=f"Correct! (0 {correct.unit})",
                    details={"answer": answer_val, "correct": correct_val},
                )
            else:
                return ValidationResult(
                    valid=False,
                    message=f"Incorrect. Expected 0 {correct.unit}, got {answer_val} {correct.unit}.",
                    details={"answer": answer_val, "correct": correct_val},
                )

        # Calculate relative error
        rel_error = abs((answer_val - correct_val) / correct_val)

        if rel_error <= tolerance:
            return ValidationResult(
                valid=True,
                message=f"Correct! (Your answer: {answer}, within {tolerance * 100}% tolerance)",
                details={
                    "answer": answer_val,
                    "correct": correct_val,
                    "rel_error": rel_error,
                    "unit": str(correct.unit),
                },
            )
        else:
            return ValidationResult(
                valid=False,
                message=f"Incorrect. Your answer: {answer}, Correct answer: {correct} (relative error: {rel_error * 100:.2f}%)",
                details={
                    "answer": answer_val,
                    "correct": correct_val,
                    "rel_error": rel_error,
                    "unit": str(correct.unit),
                },
            )

    def validate_dimension(
        self, answer: Dimension | DimArray, correct: Dimension
    ) -> ValidationResult:
        """Validate that a dimension is correct.

        Args:
            answer: The student's dimension (Dimension or DimArray).
            correct: The correct dimension.

        Returns:
            ValidationResult indicating if dimensions match.
        """
        if isinstance(answer, DimArray):
            answer_dim = answer.unit.dimension
        elif isinstance(answer, Dimension):
            answer_dim = answer
        else:
            return ValidationResult(
                valid=False,
                message="Answer must be a Dimension or DimArray.",
                details={},
            )

        if answer_dim == correct:
            return ValidationResult(
                valid=True,
                message=f"Correct! The dimension is {correct}.",
                details={"dimension": str(correct)},
            )
        else:
            return ValidationResult(
                valid=False,
                message=f"Incorrect dimension. Your answer: {answer_dim}, Correct: {correct}",
                details={
                    "answer_dimension": str(answer_dim),
                    "correct_dimension": str(correct),
                },
            )

    def validate_array(
        self,
        answer: DimArray | np.ndarray,
        correct: DimArray | np.ndarray,
        tolerance: float = 0.01,
    ) -> ValidationResult:
        """Validate an array answer (element-wise comparison).

        Args:
            answer: The student's answer array.
            correct: The correct answer array.
            tolerance: Relative tolerance for comparison.

        Returns:
            ValidationResult indicating if arrays match element-wise.
        """
        # Handle DimArray
        if isinstance(answer, DimArray) and isinstance(correct, DimArray):
            # Check dimensions
            if answer.unit.dimension != correct.unit.dimension:
                return ValidationResult(
                    valid=False,
                    message=f"Incorrect dimensions. Your answer: {answer.unit.dimension}, Correct: {correct.unit.dimension}",
                    details={},
                )

            # Convert to same units
            try:
                answer_converted = answer.to(correct.unit)
                answer_data = answer_converted.data
                correct_data = correct.data
            except Exception as e:
                return ValidationResult(
                    valid=False, message=f"Error converting units: {str(e)}", details={}
                )

        elif isinstance(answer, np.ndarray) and isinstance(correct, np.ndarray):
            answer_data = answer
            correct_data = correct
        else:
            return ValidationResult(
                valid=False,
                message="Answer and correct must both be arrays of the same type.",
                details={},
            )

        # Check shapes
        if answer_data.shape != correct_data.shape:
            return ValidationResult(
                valid=False,
                message=f"Incorrect shape. Your answer: {answer_data.shape}, Correct: {correct_data.shape}",
                details={
                    "answer_shape": answer_data.shape,
                    "correct_shape": correct_data.shape,
                },
            )

        # Element-wise comparison with tolerance
        rel_errors = np.abs((answer_data - correct_data) / (correct_data + 1e-10))
        max_error = np.max(rel_errors)

        if max_error <= tolerance:
            return ValidationResult(
                valid=True,
                message=f"Correct! All elements within {tolerance * 100}% tolerance.",
                details={"max_rel_error": float(max_error)},
            )
        else:
            # Find indices of incorrect elements
            incorrect_indices = np.where(rel_errors > tolerance)
            num_incorrect = len(incorrect_indices[0])

            return ValidationResult(
                valid=False,
                message=f"Incorrect. {num_incorrect} element(s) outside tolerance. Max relative error: {max_error * 100:.2f}%",
                details={
                    "max_rel_error": float(max_error),
                    "num_incorrect": num_incorrect,
                    "incorrect_indices": [
                        tuple(idx) for idx in zip(*incorrect_indices)
                    ],
                },
            )

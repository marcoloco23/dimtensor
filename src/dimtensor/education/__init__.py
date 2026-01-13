"""Interactive textbook system for teaching physics with dimtensor.

This module provides an interactive educational framework for learning physics
through unit-aware computation. It includes:

- Structured content organization (chapters, sections, lessons)
- Multiple exercise types with auto-grading
- Progress tracking and learning analytics
- Dimensional validation for answers

Example:
    >>> from dimtensor.education import MultipleChoiceExercise
    >>> ex = MultipleChoiceExercise(
    ...     question="What is the SI unit of force?",
    ...     choices=["Joule", "Newton", "Watt", "Pascal"],
    ...     correct_index=1
    ... )
    >>> result = ex.check(1)
    >>> print(result.correct)  # True
"""

from .exercises import (
    Exercise,
    MultipleChoiceExercise,
    NumericAnswerExercise,
    CodeExercise,
    DimensionalAnalysisExercise,
    UnitConversionExercise,
    WordProblem,
)
from .grading import AutoGrader, GradeResult
from .progress import ProgressTracker, ProgressStorage
from .textbook import Chapter, Section, Lesson, Example
from .validation import AnswerValidator, ValidationResult

__all__ = [
    # Content structure
    "Chapter",
    "Section",
    "Lesson",
    "Example",
    # Exercise types
    "Exercise",
    "MultipleChoiceExercise",
    "NumericAnswerExercise",
    "CodeExercise",
    "DimensionalAnalysisExercise",
    "UnitConversionExercise",
    "WordProblem",
    # Validation
    "AnswerValidator",
    "ValidationResult",
    # Grading
    "AutoGrader",
    "GradeResult",
    # Progress tracking
    "ProgressTracker",
    "ProgressStorage",
]

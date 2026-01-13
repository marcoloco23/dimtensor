"""Auto-grading logic for exercises and quizzes.

This module provides automatic grading functionality for exercises,
including partial credit and rubric-based grading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .exercises import Exercise, ExerciseResult


@dataclass
class GradeResult:
    """Result of grading an exercise or quiz.

    Attributes:
        total_score: Total score achieved (0.0 to max_score).
        max_score: Maximum possible score.
        percentage: Percentage score (0.0 to 100.0).
        feedback: Overall feedback message.
        exercise_results: Individual exercise results.
        passed: Whether the student passed (percentage >= passing_threshold).
    """

    total_score: float
    max_score: float
    percentage: float
    feedback: str
    exercise_results: list[ExerciseResult] = field(default_factory=list)
    passed: bool = False

    def __str__(self) -> str:
        """Return formatted string representation."""
        status = "PASSED" if self.passed else "NOT PASSED"
        return (
            f"Grade: {self.total_score:.1f}/{self.max_score:.1f} "
            f"({self.percentage:.1f}%) - {status}"
        )


class AutoGrader:
    """Automatic grading system for exercises and quizzes.

    This class handles grading of individual exercises and collections
    of exercises (quizzes, assignments), with support for partial credit.
    """

    def __init__(self, passing_threshold: float = 70.0):
        """Initialize the auto-grader.

        Args:
            passing_threshold: Minimum percentage to pass (default 70%).
        """
        self.passing_threshold = passing_threshold

    def grade_exercise(
        self, exercise: Exercise, answer: Any, weight: float = 1.0
    ) -> GradeResult:
        """Grade a single exercise.

        Args:
            exercise: The exercise to grade.
            answer: The student's answer.
            weight: Weight of this exercise in overall grade.

        Returns:
            GradeResult for the exercise.
        """
        result = exercise.check(answer)

        total_score = result.score * weight
        max_score = weight
        percentage = (total_score / max_score * 100) if max_score > 0 else 0.0

        return GradeResult(
            total_score=total_score,
            max_score=max_score,
            percentage=percentage,
            feedback=result.message,
            exercise_results=[result],
            passed=percentage >= self.passing_threshold,
        )

    def grade_exercises(
        self,
        exercises: list[Exercise],
        answers: list[Any],
        weights: list[float] | None = None,
    ) -> GradeResult:
        """Grade multiple exercises (e.g., a quiz).

        Args:
            exercises: List of exercises to grade.
            answers: List of student answers (must match length of exercises).
            weights: Optional weights for each exercise (default: equal weights).

        Returns:
            GradeResult for all exercises combined.
        """
        if len(exercises) != len(answers):
            raise ValueError(
                f"Number of exercises ({len(exercises)}) must match "
                f"number of answers ({len(answers)})"
            )

        # Default to equal weights
        if weights is None:
            weights = [1.0] * len(exercises)
        elif len(weights) != len(exercises):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of exercises ({len(exercises)})"
            )

        # Grade each exercise
        results = []
        total_score = 0.0
        max_score = 0.0

        for exercise, answer, weight in zip(exercises, answers, weights):
            result = exercise.check(answer)
            results.append(result)
            total_score += result.score * weight
            max_score += weight

        # Calculate overall percentage
        percentage = (total_score / max_score * 100) if max_score > 0 else 0.0
        passed = percentage >= self.passing_threshold

        # Generate feedback
        num_correct = sum(1 for r in results if r.correct)
        feedback = (
            f"You answered {num_correct}/{len(exercises)} questions correctly. "
            f"Your score: {percentage:.1f}%."
        )

        if passed:
            feedback += " Great job!"
        else:
            feedback += f" You need {self.passing_threshold}% to pass. Keep practicing!"

        return GradeResult(
            total_score=total_score,
            max_score=max_score,
            percentage=percentage,
            feedback=feedback,
            exercise_results=results,
            passed=passed,
        )

    def grade_with_rubric(
        self,
        exercise: Exercise,
        answer: Any,
        rubric: dict[str, float],
        rubric_scores: dict[str, float],
    ) -> GradeResult:
        """Grade an exercise using a rubric.

        Args:
            exercise: The exercise to grade.
            answer: The student's answer.
            rubric: Rubric criteria with max points per criterion.
            rubric_scores: Actual scores achieved per criterion.

        Returns:
            GradeResult with rubric-based scoring.
        """
        # First check if the answer is correct
        result = exercise.check(answer)

        # Calculate rubric scores
        total_score = sum(rubric_scores.values())
        max_score = sum(rubric.values())
        percentage = (total_score / max_score * 100) if max_score > 0 else 0.0

        # Generate detailed feedback
        feedback_lines = [result.message, "", "Rubric Breakdown:"]
        for criterion, max_points in rubric.items():
            earned = rubric_scores.get(criterion, 0.0)
            feedback_lines.append(f"  {criterion}: {earned:.1f}/{max_points:.1f}")

        feedback = "\n".join(feedback_lines)

        return GradeResult(
            total_score=total_score,
            max_score=max_score,
            percentage=percentage,
            feedback=feedback,
            exercise_results=[result],
            passed=percentage >= self.passing_threshold,
        )

    def calculate_partial_credit(
        self, steps_completed: int, total_steps: int, full_credit: float = 1.0
    ) -> float:
        """Calculate partial credit based on completed steps.

        Args:
            steps_completed: Number of steps completed correctly.
            total_steps: Total number of steps.
            full_credit: Maximum score for full completion.

        Returns:
            Partial credit score.
        """
        if total_steps == 0:
            return 0.0
        return (steps_completed / total_steps) * full_credit


@dataclass
class Quiz:
    """A quiz containing multiple exercises.

    Attributes:
        title: Quiz title.
        exercises: List of exercises in the quiz.
        time_limit: Optional time limit in minutes.
        passing_threshold: Minimum percentage to pass.
        weights: Optional weights for each exercise.
    """

    title: str
    exercises: list[Exercise]
    time_limit: int | None = None
    passing_threshold: float = 70.0
    weights: list[float] | None = None

    def grade(self, answers: list[Any]) -> GradeResult:
        """Grade the quiz with the provided answers.

        Args:
            answers: List of answers (must match number of exercises).

        Returns:
            GradeResult for the entire quiz.
        """
        grader = AutoGrader(passing_threshold=self.passing_threshold)
        return grader.grade_exercises(self.exercises, answers, self.weights)

    def get_exercise_count(self) -> int:
        """Get the number of exercises in this quiz."""
        return len(self.exercises)

    def get_max_score(self) -> float:
        """Get the maximum possible score."""
        if self.weights:
            return sum(self.weights)
        return float(len(self.exercises))

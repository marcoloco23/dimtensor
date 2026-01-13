"""Tests for auto-grading module."""

import pytest

from dimtensor import DimArray, units
from dimtensor.education import MultipleChoiceExercise, NumericAnswerExercise
from dimtensor.education.grading import AutoGrader, Quiz


class TestAutoGrader:
    """Tests for AutoGrader."""

    def test_grade_single_exercise_correct(self):
        """Test grading a single correct exercise."""
        grader = AutoGrader(passing_threshold=70.0)

        ex = MultipleChoiceExercise(
            id="test1",
            question="What is 2+2?",
            choices=["3", "4", "5"],
            correct_index=1,
        )

        result = grader.grade_exercise(ex, answer=1, weight=1.0)

        assert result.total_score == 1.0
        assert result.max_score == 1.0
        assert result.percentage == 100.0
        assert result.passed is True

    def test_grade_single_exercise_incorrect(self):
        """Test grading a single incorrect exercise."""
        grader = AutoGrader(passing_threshold=70.0)

        ex = MultipleChoiceExercise(
            id="test2",
            question="What is 2+2?",
            choices=["3", "4", "5"],
            correct_index=1,
        )

        result = grader.grade_exercise(ex, answer=0, weight=1.0)

        assert result.total_score == 0.0
        assert result.max_score == 1.0
        assert result.percentage == 0.0
        assert result.passed is False

    def test_grade_multiple_exercises(self):
        """Test grading multiple exercises."""
        grader = AutoGrader(passing_threshold=70.0)

        ex1 = MultipleChoiceExercise(
            id="test3", question="Q1", choices=["A", "B"], correct_index=0
        )

        ex2 = MultipleChoiceExercise(
            id="test4", question="Q2", choices=["A", "B"], correct_index=1
        )

        ex3 = MultipleChoiceExercise(
            id="test5", question="Q3", choices=["A", "B"], correct_index=0
        )

        exercises = [ex1, ex2, ex3]
        answers = [0, 1, 1]  # First two correct, last one wrong

        result = grader.grade_exercises(exercises, answers)

        assert result.total_score == 2.0
        assert result.max_score == 3.0
        assert abs(result.percentage - 66.67) < 0.1
        assert result.passed is False  # Below 70%

    def test_grade_with_weights(self):
        """Test grading with weighted exercises."""
        grader = AutoGrader(passing_threshold=70.0)

        ex1 = MultipleChoiceExercise(
            id="test6", question="Easy", choices=["A", "B"], correct_index=0
        )

        ex2 = MultipleChoiceExercise(
            id="test7", question="Hard", choices=["A", "B"], correct_index=1
        )

        exercises = [ex1, ex2]
        answers = [0, 0]  # First correct, second wrong
        weights = [1.0, 2.0]  # Second question worth double

        result = grader.grade_exercises(exercises, answers, weights)

        assert result.total_score == 1.0
        assert result.max_score == 3.0
        assert abs(result.percentage - 33.33) < 0.1

    def test_grade_with_rubric(self):
        """Test rubric-based grading."""
        grader = AutoGrader(passing_threshold=70.0)

        ex = NumericAnswerExercise(
            id="test8", question="Calculate", correct_answer=10.0, tolerance=0.01
        )

        rubric = {"Correct Answer": 5.0, "Show Work": 3.0, "Explanation": 2.0}

        rubric_scores = {"Correct Answer": 5.0, "Show Work": 2.0, "Explanation": 1.0}

        result = grader.grade_with_rubric(ex, 10.0, rubric, rubric_scores)

        assert result.total_score == 8.0
        assert result.max_score == 10.0
        assert result.percentage == 80.0
        assert "Rubric Breakdown" in result.feedback

    def test_partial_credit_calculation(self):
        """Test partial credit calculation."""
        grader = AutoGrader()

        # Completed 3 out of 5 steps
        credit = grader.calculate_partial_credit(
            steps_completed=3, total_steps=5, full_credit=1.0
        )

        assert credit == 0.6

    def test_passing_threshold(self):
        """Test custom passing threshold."""
        grader = AutoGrader(passing_threshold=80.0)

        ex = MultipleChoiceExercise(
            id="test9", question="Q", choices=["A", "B"], correct_index=0
        )

        # 75% score
        result = grader.grade_exercises([ex, ex, ex, ex], [0, 0, 0, 1])

        assert result.percentage == 75.0
        assert result.passed is False  # Threshold is 80%

        # But with 70% threshold
        grader2 = AutoGrader(passing_threshold=70.0)
        result2 = grader2.grade_exercises([ex, ex, ex, ex], [0, 0, 0, 1])
        assert result2.passed is True


class TestQuiz:
    """Tests for Quiz."""

    def test_quiz_creation(self):
        """Test creating a quiz."""
        ex1 = MultipleChoiceExercise(
            id="q1", question="Q1", choices=["A", "B"], correct_index=0
        )

        ex2 = MultipleChoiceExercise(
            id="q2", question="Q2", choices=["A", "B"], correct_index=1
        )

        quiz = Quiz(
            title="Test Quiz", exercises=[ex1, ex2], time_limit=30, passing_threshold=75.0
        )

        assert quiz.title == "Test Quiz"
        assert quiz.get_exercise_count() == 2
        assert quiz.get_max_score() == 2.0

    def test_quiz_grading(self):
        """Test quiz grading."""
        ex1 = MultipleChoiceExercise(
            id="q1", question="Q1", choices=["A", "B"], correct_index=0
        )

        ex2 = MultipleChoiceExercise(
            id="q2", question="Q2", choices=["A", "B"], correct_index=1
        )

        quiz = Quiz(title="Test Quiz", exercises=[ex1, ex2], passing_threshold=50.0)

        result = quiz.grade([0, 1])  # Both correct
        assert result.percentage == 100.0
        assert result.passed is True

        result = quiz.grade([0, 0])  # One correct
        assert result.percentage == 50.0
        assert result.passed is True

        result = quiz.grade([1, 0])  # Neither correct
        assert result.percentage == 0.0
        assert result.passed is False

    def test_quiz_with_weights(self):
        """Test quiz with weighted questions."""
        ex1 = MultipleChoiceExercise(
            id="q1", question="Easy", choices=["A", "B"], correct_index=0
        )

        ex2 = MultipleChoiceExercise(
            id="q2", question="Hard", choices=["A", "B"], correct_index=1
        )

        quiz = Quiz(
            title="Weighted Quiz",
            exercises=[ex1, ex2],
            weights=[1.0, 3.0],  # Second worth 3x
            passing_threshold=50.0,
        )

        assert quiz.get_max_score() == 4.0

        result = quiz.grade([0, 0])  # First correct, second wrong
        assert result.total_score == 1.0
        assert result.max_score == 4.0
        assert result.percentage == 25.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

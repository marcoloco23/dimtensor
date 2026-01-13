"""Progress tracking for student learning.

This module provides progress tracking functionality, storing completion
status, scores, and learning analytics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ExerciseAttempt:
    """Record of a student's attempt at an exercise.

    Attributes:
        exercise_id: Unique identifier for the exercise.
        timestamp: When the attempt was made.
        correct: Whether the answer was correct.
        score: Numeric score (0.0 to 1.0).
        hints_used: Number of hints used.
    """

    exercise_id: str
    timestamp: str
    correct: bool
    score: float
    hints_used: int = 0


@dataclass
class SectionProgress:
    """Progress for a section.

    Attributes:
        section_id: Section identifier (e.g., "1.2").
        completed: Whether the section is marked as complete.
        exercises_attempted: List of exercise IDs attempted.
        exercises_completed: List of exercise IDs completed correctly.
        time_spent: Time spent in minutes.
    """

    section_id: str
    completed: bool = False
    exercises_attempted: list[str] = field(default_factory=list)
    exercises_completed: list[str] = field(default_factory=list)
    time_spent: float = 0.0


@dataclass
class ChapterProgress:
    """Progress for a chapter.

    Attributes:
        chapter_number: Chapter number.
        started: Whether the chapter has been started.
        completed: Whether the chapter is marked as complete.
        sections: Progress for each section.
    """

    chapter_number: int
    started: bool = False
    completed: bool = False
    sections: dict[str, SectionProgress] = field(default_factory=dict)


class ProgressStorage:
    """Storage backend for progress data.

    Stores progress data to a JSON file in ~/.dimtensor/progress/
    """

    def __init__(self, student_id: str = "default"):
        """Initialize storage for a student.

        Args:
            student_id: Unique identifier for the student.
        """
        self.student_id = student_id
        self.storage_dir = Path.home() / ".dimtensor" / "progress"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.storage_dir / f"{student_id}.json"

    def save(self, data: dict[str, Any]) -> None:
        """Save progress data to file.

        Args:
            data: Progress data dictionary.
        """
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> dict[str, Any]:
        """Load progress data from file.

        Returns:
            Progress data dictionary, or empty dict if file doesn't exist.
        """
        if not self.file_path.exists():
            return {}

        with open(self.file_path, "r") as f:
            return json.load(f)

    def clear(self) -> None:
        """Clear all progress data."""
        if self.file_path.exists():
            self.file_path.unlink()


class ProgressTracker:
    """Track student progress through the textbook.

    This class maintains a record of completed chapters, sections, and
    exercises, along with scores and timestamps.
    """

    def __init__(self, student_id: str = "default"):
        """Initialize a progress tracker.

        Args:
            student_id: Unique identifier for the student.
        """
        self.student_id = student_id
        self.storage = ProgressStorage(student_id)
        self._data = self.storage.load()

        # Initialize data structure if empty
        if not self._data:
            self._data = {
                "student_id": student_id,
                "started_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
                "chapters": {},
                "exercise_attempts": [],
            }
            self._save()

    def _save(self) -> None:
        """Save current progress to storage."""
        self._data["last_active"] = datetime.now().isoformat()
        self.storage.save(self._data)

    def start_chapter(self, chapter_number: int) -> None:
        """Mark a chapter as started.

        Args:
            chapter_number: The chapter number.
        """
        chapter_key = str(chapter_number)
        if chapter_key not in self._data["chapters"]:
            self._data["chapters"][chapter_key] = {
                "chapter_number": chapter_number,
                "started": True,
                "completed": False,
                "sections": {},
            }
        else:
            self._data["chapters"][chapter_key]["started"] = True

        self._save()

    def complete_chapter(self, chapter_number: int) -> None:
        """Mark a chapter as completed.

        Args:
            chapter_number: The chapter number.
        """
        chapter_key = str(chapter_number)
        if chapter_key not in self._data["chapters"]:
            self.start_chapter(chapter_number)

        self._data["chapters"][chapter_key]["completed"] = True
        self._save()

    def complete_section(self, chapter_number: int, section_number: int) -> None:
        """Mark a section as completed.

        Args:
            chapter_number: The chapter number.
            section_number: The section number.
        """
        chapter_key = str(chapter_number)
        section_key = f"{chapter_number}.{section_number}"

        if chapter_key not in self._data["chapters"]:
            self.start_chapter(chapter_number)

        if section_key not in self._data["chapters"][chapter_key]["sections"]:
            self._data["chapters"][chapter_key]["sections"][section_key] = {
                "section_id": section_key,
                "completed": True,
                "exercises_attempted": [],
                "exercises_completed": [],
                "time_spent": 0.0,
            }
        else:
            self._data["chapters"][chapter_key]["sections"][section_key][
                "completed"
            ] = True

        self._save()

    def record_exercise_attempt(
        self,
        exercise_id: str,
        correct: bool,
        score: float,
        hints_used: int = 0,
        chapter_number: int | None = None,
        section_number: int | None = None,
    ) -> None:
        """Record an exercise attempt.

        Args:
            exercise_id: Unique identifier for the exercise.
            correct: Whether the answer was correct.
            score: Numeric score (0.0 to 1.0).
            hints_used: Number of hints used.
            chapter_number: Optional chapter number for tracking.
            section_number: Optional section number for tracking.
        """
        attempt = {
            "exercise_id": exercise_id,
            "timestamp": datetime.now().isoformat(),
            "correct": correct,
            "score": score,
            "hints_used": hints_used,
        }
        self._data["exercise_attempts"].append(attempt)

        # Update chapter/section progress if provided
        if chapter_number is not None and section_number is not None:
            chapter_key = str(chapter_number)
            section_key = f"{chapter_number}.{section_number}"

            if chapter_key not in self._data["chapters"]:
                self.start_chapter(chapter_number)

            if section_key not in self._data["chapters"][chapter_key]["sections"]:
                self._data["chapters"][chapter_key]["sections"][section_key] = {
                    "section_id": section_key,
                    "completed": False,
                    "exercises_attempted": [],
                    "exercises_completed": [],
                    "time_spent": 0.0,
                }

            section = self._data["chapters"][chapter_key]["sections"][section_key]

            if exercise_id not in section["exercises_attempted"]:
                section["exercises_attempted"].append(exercise_id)

            if correct and exercise_id not in section["exercises_completed"]:
                section["exercises_completed"].append(exercise_id)

        self._save()

    def get_chapter_progress(self, chapter_number: int) -> dict[str, Any]:
        """Get progress for a specific chapter.

        Args:
            chapter_number: The chapter number.

        Returns:
            Dictionary with chapter progress data.
        """
        chapter_key = str(chapter_number)
        return self._data["chapters"].get(
            chapter_key,
            {
                "chapter_number": chapter_number,
                "started": False,
                "completed": False,
                "sections": {},
            },
        )

    def get_overall_progress(self) -> dict[str, Any]:
        """Get overall progress summary.

        Returns:
            Dictionary with overall progress statistics.
        """
        total_chapters = len(self._data["chapters"])
        completed_chapters = sum(
            1 for ch in self._data["chapters"].values() if ch["completed"]
        )

        total_exercises = len(self._data["exercise_attempts"])
        correct_exercises = sum(
            1 for att in self._data["exercise_attempts"] if att["correct"]
        )

        return {
            "student_id": self.student_id,
            "started_at": self._data["started_at"],
            "last_active": self._data["last_active"],
            "chapters_started": total_chapters,
            "chapters_completed": completed_chapters,
            "total_exercises_attempted": total_exercises,
            "exercises_correct": correct_exercises,
            "accuracy": correct_exercises / total_exercises if total_exercises > 0 else 0.0,
        }

    def get_exercise_history(self, exercise_id: str) -> list[dict[str, Any]]:
        """Get all attempts for a specific exercise.

        Args:
            exercise_id: The exercise identifier.

        Returns:
            List of attempt records for the exercise.
        """
        return [
            att
            for att in self._data["exercise_attempts"]
            if att["exercise_id"] == exercise_id
        ]

    def reset_progress(self) -> None:
        """Reset all progress data."""
        self.storage.clear()
        self._data = {
            "student_id": self.student_id,
            "started_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "chapters": {},
            "exercise_attempts": [],
        }
        self._save()

    def export_report(self) -> str:
        """Export a text report of progress.

        Returns:
            Formatted text report of student progress.
        """
        overall = self.get_overall_progress()

        report = []
        report.append("=" * 60)
        report.append("PROGRESS REPORT")
        report.append("=" * 60)
        report.append(f"Student ID: {overall['student_id']}")
        report.append(f"Started: {overall['started_at']}")
        report.append(f"Last Active: {overall['last_active']}")
        report.append("")
        report.append(
            f"Chapters: {overall['chapters_completed']}/{overall['chapters_started']} completed"
        )
        report.append(
            f"Exercises: {overall['exercises_correct']}/{overall['total_exercises_attempted']} correct"
        )
        report.append(f"Accuracy: {overall['accuracy'] * 100:.1f}%")
        report.append("")

        # Chapter details
        report.append("CHAPTER DETAILS")
        report.append("-" * 60)
        for chapter_key in sorted(self._data["chapters"].keys(), key=int):
            chapter = self._data["chapters"][chapter_key]
            status = "✓ Completed" if chapter["completed"] else "○ In Progress"
            report.append(f"Chapter {chapter_key}: {status}")

            # Section details
            for section_key in sorted(chapter["sections"].keys()):
                section = chapter["sections"][section_key]
                exercises = section["exercises_completed"]
                attempted = section["exercises_attempted"]
                report.append(
                    f"  Section {section_key}: {len(exercises)}/{len(attempted)} exercises"
                )

        report.append("=" * 60)

        return "\n".join(report)

"""Tests for progress tracking."""

import json
from pathlib import Path

import pytest

from dimtensor.education.progress import ProgressStorage, ProgressTracker


class TestProgressStorage:
    """Tests for ProgressStorage."""

    def test_save_and_load(self, tmp_path):
        """Test saving and loading progress."""
        storage = ProgressStorage("test_student")
        storage.storage_dir = tmp_path

        data = {"test": "data", "score": 100}
        storage.save(data)

        loaded = storage.load()
        assert loaded == data

    def test_load_nonexistent(self, tmp_path):
        """Test loading non-existent file returns empty dict."""
        storage = ProgressStorage("nonexistent")
        storage.storage_dir = tmp_path

        loaded = storage.load()
        assert loaded == {}

    def test_clear(self, tmp_path):
        """Test clearing progress."""
        storage = ProgressStorage("test_student")
        storage.storage_dir = tmp_path

        data = {"test": "data"}
        storage.save(data)

        storage.clear()
        loaded = storage.load()
        assert loaded == {}


class TestProgressTracker:
    """Tests for ProgressTracker."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a tracker with temporary storage."""
        tracker = ProgressTracker("test_student")
        tracker.storage.storage_dir = tmp_path
        tracker.storage.file_path = tmp_path / "test_student.json"
        return tracker

    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.student_id == "test_student"
        assert "started_at" in tracker._data
        assert "chapters" in tracker._data
        assert "exercise_attempts" in tracker._data

    def test_start_chapter(self, tracker):
        """Test starting a chapter."""
        tracker.start_chapter(1)

        chapter = tracker.get_chapter_progress(1)
        assert chapter["started"] is True
        assert chapter["completed"] is False

    def test_complete_chapter(self, tracker):
        """Test completing a chapter."""
        tracker.start_chapter(1)
        tracker.complete_chapter(1)

        chapter = tracker.get_chapter_progress(1)
        assert chapter["completed"] is True

    def test_complete_section(self, tracker):
        """Test completing a section."""
        tracker.complete_section(1, 1)

        chapter = tracker.get_chapter_progress(1)
        assert "1.1" in chapter["sections"]
        assert chapter["sections"]["1.1"]["completed"] is True

    def test_record_exercise_attempt(self, tracker):
        """Test recording exercise attempt."""
        tracker.record_exercise_attempt(
            exercise_id="ex1",
            correct=True,
            score=1.0,
            hints_used=1,
            chapter_number=1,
            section_number=1,
        )

        # Check exercise was recorded
        history = tracker.get_exercise_history("ex1")
        assert len(history) == 1
        assert history[0]["correct"] is True
        assert history[0]["score"] == 1.0
        assert history[0]["hints_used"] == 1

        # Check section was updated
        chapter = tracker.get_chapter_progress(1)
        section = chapter["sections"]["1.1"]
        assert "ex1" in section["exercises_attempted"]
        assert "ex1" in section["exercises_completed"]

    def test_record_incorrect_exercise(self, tracker):
        """Test recording incorrect exercise."""
        tracker.record_exercise_attempt(
            exercise_id="ex2",
            correct=False,
            score=0.0,
            chapter_number=1,
            section_number=1,
        )

        chapter = tracker.get_chapter_progress(1)
        section = chapter["sections"]["1.1"]
        assert "ex2" in section["exercises_attempted"]
        assert "ex2" not in section["exercises_completed"]

    def test_get_overall_progress(self, tracker):
        """Test getting overall progress."""
        tracker.start_chapter(1)
        tracker.complete_chapter(1)
        tracker.record_exercise_attempt("ex1", correct=True, score=1.0)
        tracker.record_exercise_attempt("ex2", correct=False, score=0.0)

        progress = tracker.get_overall_progress()

        assert progress["chapters_started"] == 1
        assert progress["chapters_completed"] == 1
        assert progress["total_exercises_attempted"] == 2
        assert progress["exercises_correct"] == 1
        assert progress["accuracy"] == 0.5

    def test_reset_progress(self, tracker):
        """Test resetting progress."""
        tracker.start_chapter(1)
        tracker.record_exercise_attempt("ex1", correct=True, score=1.0)

        tracker.reset_progress()

        progress = tracker.get_overall_progress()
        assert progress["chapters_started"] == 0
        assert progress["total_exercises_attempted"] == 0

    def test_export_report(self, tracker):
        """Test exporting report."""
        tracker.start_chapter(1)
        tracker.complete_chapter(1)

        report = tracker.export_report()

        assert "PROGRESS REPORT" in report
        assert "Student ID: test_student" in report
        assert "Chapter 1" in report

    def test_persistence_across_sessions(self, tracker):
        """Test that progress persists across sessions."""
        # First session
        tracker.start_chapter(1)
        tracker.record_exercise_attempt("ex1", correct=True, score=1.0)
        tracker._save()

        # Create new tracker (simulating new session)
        new_tracker = ProgressTracker("test_student")
        new_tracker.storage.storage_dir = tracker.storage.storage_dir
        new_tracker.storage.file_path = tracker.storage.file_path
        new_tracker._data = new_tracker.storage.load()

        # Check data persisted
        chapter = new_tracker.get_chapter_progress(1)
        assert chapter["started"] is True

        history = new_tracker.get_exercise_history("ex1")
        assert len(history) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

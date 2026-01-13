"""Tests for textbook structure."""

import pytest

from dimtensor.education.textbook import Chapter, Example, Lesson, Section, Textbook


class TestExample:
    """Tests for Example."""

    def test_example_creation(self):
        """Test creating an example."""
        ex = Example(
            title="Example 1.1",
            problem="Calculate 2 + 2",
            solution="2 + 2 = 4",
            code="result = 2 + 2",
            explanation="Addition is commutative",
        )

        assert ex.title == "Example 1.1"
        assert ex.problem == "Calculate 2 + 2"
        assert ex.solution == "2 + 2 = 4"

    def test_example_display(self):
        """Test example display formatting."""
        ex = Example(
            title="Test Example", problem="Problem text", solution="Solution text"
        )

        display = ex.display()

        assert "Test Example" in display
        assert "Problem:" in display
        assert "Solution:" in display


class TestLesson:
    """Tests for Lesson."""

    def test_lesson_creation(self):
        """Test creating a lesson."""
        lesson = Lesson(
            title="Introduction to Units",
            content="Units are important...",
            learning_objectives=["Understand units", "Apply units"],
            key_concepts=["SI units", "Dimensions"],
        )

        assert lesson.title == "Introduction to Units"
        assert len(lesson.learning_objectives) == 2
        assert len(lesson.key_concepts) == 2

    def test_add_example(self):
        """Test adding examples to lesson."""
        lesson = Lesson(title="Test Lesson", content="Content")

        ex1 = Example(title="Ex 1", problem="P1", solution="S1")
        ex2 = Example(title="Ex 2", problem="P2", solution="S2")

        lesson.add_example(ex1)
        lesson.add_example(ex2)

        assert len(lesson.examples) == 2
        assert lesson.examples[0].title == "Ex 1"


class TestSection:
    """Tests for Section."""

    def test_section_creation(self):
        """Test creating a section."""
        section = Section(
            number=1, title="Units and Measurements", description="Learn about units"
        )

        assert section.number == 1
        assert section.title == "Units and Measurements"
        assert section.full_title == "Section 1: Units and Measurements"

    def test_add_lesson(self):
        """Test adding lessons to section."""
        section = Section(number=1, title="Test Section")

        lesson1 = Lesson(title="Lesson 1", content="Content 1")
        lesson2 = Lesson(title="Lesson 2", content="Content 2")

        section.add_lesson(lesson1)
        section.add_lesson(lesson2)

        assert len(section.lessons) == 2

    def test_add_exercise(self):
        """Test adding exercise IDs."""
        section = Section(number=1, title="Test Section")

        section.add_exercise("ex1.1")
        section.add_exercise("ex1.2")

        assert len(section.exercises) == 2
        assert "ex1.1" in section.exercises


class TestChapter:
    """Tests for Chapter."""

    def test_chapter_creation(self):
        """Test creating a chapter."""
        chapter = Chapter(
            number=1,
            title="Introduction",
            description="First chapter",
            prerequisites=[],
            difficulty=1,
        )

        assert chapter.number == 1
        assert chapter.title == "Introduction"
        assert chapter.full_title == "Chapter 1: Introduction"
        assert chapter.difficulty == 1

    def test_add_section(self):
        """Test adding sections to chapter."""
        chapter = Chapter(number=1, title="Test Chapter")

        section1 = Section(number=1, title="Section 1")
        section2 = Section(number=2, title="Section 2")

        chapter.add_section(section1)
        chapter.add_section(section2)

        assert len(chapter.sections) == 2

    def test_get_all_exercises(self):
        """Test getting all exercises from chapter."""
        chapter = Chapter(number=1, title="Test Chapter")

        section1 = Section(number=1, title="Section 1")
        section1.add_exercise("ex1.1")
        section1.add_exercise("ex1.2")

        section2 = Section(number=2, title="Section 2")
        section2.add_exercise("ex1.3")

        chapter.add_section(section1)
        chapter.add_section(section2)

        exercises = chapter.get_all_exercises()
        assert len(exercises) == 3
        assert "ex1.1" in exercises
        assert "ex1.3" in exercises

    def test_prerequisites(self):
        """Test chapter prerequisites."""
        chapter = Chapter(
            number=5, title="Advanced Topics", prerequisites=[1, 2, 3, 4], difficulty=4
        )

        assert len(chapter.prerequisites) == 4
        assert 1 in chapter.prerequisites


class TestTextbook:
    """Tests for Textbook."""

    def test_textbook_creation(self):
        """Test creating a textbook."""
        textbook = Textbook(
            title="Physics with dimtensor",
            authors=["Author 1", "Author 2"],
            version="1.0.0",
        )

        assert textbook.title == "Physics with dimtensor"
        assert len(textbook.authors) == 2
        assert textbook.version == "1.0.0"

    def test_add_chapter(self):
        """Test adding chapters to textbook."""
        textbook = Textbook()

        chapter1 = Chapter(number=1, title="Chapter 1")
        chapter2 = Chapter(number=2, title="Chapter 2")

        textbook.add_chapter(chapter1)
        textbook.add_chapter(chapter2)

        assert len(textbook.chapters) == 2

    def test_get_chapter(self):
        """Test getting chapter by number."""
        textbook = Textbook()

        chapter1 = Chapter(number=1, title="Chapter 1")
        chapter2 = Chapter(number=2, title="Chapter 2")

        textbook.add_chapter(chapter1)
        textbook.add_chapter(chapter2)

        retrieved = textbook.get_chapter(2)
        assert retrieved is not None
        assert retrieved.number == 2
        assert retrieved.title == "Chapter 2"

        # Non-existent chapter
        missing = textbook.get_chapter(99)
        assert missing is None

    def test_get_chapter_count(self):
        """Test getting chapter count."""
        textbook = Textbook()

        assert textbook.get_chapter_count() == 0

        textbook.add_chapter(Chapter(number=1, title="Chapter 1"))
        textbook.add_chapter(Chapter(number=2, title="Chapter 2"))

        assert textbook.get_chapter_count() == 2

    def test_get_exercise_count(self):
        """Test getting total exercise count."""
        textbook = Textbook()

        chapter1 = Chapter(number=1, title="Chapter 1")
        section1 = Section(number=1, title="Section 1.1")
        section1.add_exercise("ex1.1")
        section1.add_exercise("ex1.2")
        chapter1.add_section(section1)

        chapter2 = Chapter(number=2, title="Chapter 2")
        section2 = Section(number=1, title="Section 2.1")
        section2.add_exercise("ex2.1")
        chapter2.add_section(section2)

        textbook.add_chapter(chapter1)
        textbook.add_chapter(chapter2)

        assert textbook.get_exercise_count() == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

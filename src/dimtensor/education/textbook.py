"""Content structure for the interactive textbook.

This module defines the organizational structure of the textbook: chapters,
sections, lessons, and examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Example:
    """A worked example showing how to solve a problem.

    Attributes:
        title: Brief title of the example.
        problem: Description of the problem.
        solution: Step-by-step solution.
        code: Python code demonstrating the solution (optional).
        explanation: Additional explanation or insights (optional).
    """

    title: str
    problem: str
    solution: str
    code: str | None = None
    explanation: str | None = None

    def display(self) -> str:
        """Return formatted string representation of the example."""
        output = [f"## {self.title}", "", "**Problem:**", self.problem, ""]

        if self.code:
            output.extend(["**Solution (Code):**", "```python", self.code, "```", ""])

        output.extend(["**Solution:**", self.solution, ""])

        if self.explanation:
            output.extend(["**Explanation:**", self.explanation, ""])

        return "\n".join(output)


@dataclass
class Lesson:
    """A single lesson within a section.

    Attributes:
        title: Lesson title.
        content: Main content of the lesson (markdown).
        learning_objectives: List of learning objectives.
        examples: Worked examples.
        key_concepts: Key concepts covered in this lesson.
    """

    title: str
    content: str
    learning_objectives: list[str] = field(default_factory=list)
    examples: list[Example] = field(default_factory=list)
    key_concepts: list[str] = field(default_factory=list)

    def add_example(self, example: Example) -> None:
        """Add an example to this lesson."""
        self.examples.append(example)


@dataclass
class Section:
    """A section within a chapter, containing multiple lessons.

    Attributes:
        number: Section number within the chapter (e.g., 1, 2, 3).
        title: Section title.
        description: Brief description of what this section covers.
        lessons: Lessons in this section.
        exercises: Exercise IDs associated with this section.
    """

    number: int
    title: str
    description: str = ""
    lessons: list[Lesson] = field(default_factory=list)
    exercises: list[str] = field(default_factory=list)

    def add_lesson(self, lesson: Lesson) -> None:
        """Add a lesson to this section."""
        self.lessons.append(lesson)

    def add_exercise(self, exercise_id: str) -> None:
        """Add an exercise ID to this section."""
        self.exercises.append(exercise_id)

    @property
    def full_title(self) -> str:
        """Return the full section title with number."""
        return f"Section {self.number}: {self.title}"


@dataclass
class Chapter:
    """A chapter in the textbook, containing multiple sections.

    Attributes:
        number: Chapter number.
        title: Chapter title.
        description: Brief description of the chapter.
        sections: Sections in this chapter.
        prerequisites: List of prerequisite chapter numbers.
        difficulty: Difficulty level (1-5).
    """

    number: int
    title: str
    description: str = ""
    sections: list[Section] = field(default_factory=list)
    prerequisites: list[int] = field(default_factory=list)
    difficulty: int = 1

    def add_section(self, section: Section) -> None:
        """Add a section to this chapter."""
        self.sections.append(section)

    @property
    def full_title(self) -> str:
        """Return the full chapter title with number."""
        return f"Chapter {self.number}: {self.title}"

    def get_all_exercises(self) -> list[str]:
        """Get all exercise IDs from all sections in this chapter."""
        exercises = []
        for section in self.sections:
            exercises.extend(section.exercises)
        return exercises


@dataclass
class Textbook:
    """The complete textbook containing all chapters.

    Attributes:
        title: Textbook title.
        authors: List of authors.
        version: Textbook version.
        chapters: All chapters in the textbook.
    """

    title: str = "Physics with dimtensor"
    authors: list[str] = field(default_factory=lambda: ["dimtensor team"])
    version: str = "1.0.0"
    chapters: list[Chapter] = field(default_factory=list)

    def add_chapter(self, chapter: Chapter) -> None:
        """Add a chapter to the textbook."""
        self.chapters.append(chapter)

    def get_chapter(self, number: int) -> Chapter | None:
        """Get a chapter by number."""
        for chapter in self.chapters:
            if chapter.number == number:
                return chapter
        return None

    def get_chapter_count(self) -> int:
        """Get the total number of chapters."""
        return len(self.chapters)

    def get_exercise_count(self) -> int:
        """Get the total number of exercises across all chapters."""
        count = 0
        for chapter in self.chapters:
            count += len(chapter.get_all_exercises())
        return count

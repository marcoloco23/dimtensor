# Plan: Interactive Textbook System

**Date**: 2026-01-13
**Status**: PLANNING
**Author**: planner-agent
**Task ID**: #260 (v5.1.0 - Education & Accessibility)

---

## Goal

Design and implement an interactive textbook system that teaches physics concepts using dimtensor. The system should provide structured learning paths from introductory mechanics to advanced topics (E&M, thermodynamics, quantum mechanics), with interactive code exercises, progress tracking, and automatic grading.

---

## Background

**Why is this needed?**

dimtensor v5.0.0 provides powerful physics computation capabilities, but there's no structured learning path for:
- Students learning physics for the first time
- Researchers transitioning from traditional units to dimensional programming
- Educators wanting interactive physics teaching materials

**Current state:**
- 5 tutorial notebooks exist (examples/01-05), but they're disconnected
- Comprehensive API documentation exists (docs/guide/)
- 246 equations across 10+ domains in equation database
- No structured curriculum or learning progression
- No progress tracking or assessment tools
- No interactive exercises with feedback

**Target audience:**
- Undergraduate/graduate physics students
- Self-learners interested in computational physics
- Educators teaching physics with Python
- Researchers learning dimtensor for their work

---

## Approach

### Architecture Overview

The interactive textbook will be a **Jupyter-based system** with three layers:

1. **Content Layer**: Markdown + code cells organized into chapters/sections
2. **Exercise Layer**: Interactive problems with validation and hints
3. **Progress Layer**: Track completion, scores, and learning path

### Option A: Pure Jupyter Notebooks
- Description: Enhanced .ipynb files with custom metadata
- Pros: Familiar interface, no new tools, easy to deploy
- Cons: Limited interactivity, no progress persistence across sessions

### Option B: JupyterBook with Extensions
- Description: Use JupyterBook for structured content + custom exercise plugin
- Pros: Beautiful rendering, navigation, existing ecosystem
- Cons: More complex setup, requires build step, harder to customize

### Option C: Custom Jupyter Extension + Content Management
- Description: Jupyter extension for exercises + Python module for content/progress
- Pros: Full control, rich interactivity, can save progress to database
- Cons: More development work, requires installation of extension

### Option D: Hybrid (JupyterBook + Python Module) **RECOMMENDED**
- Description: JupyterBook for static content rendering + dimtensor.education module for interactive components
- Pros:
  - Beautiful static docs (can host on web)
  - Interactive exercises work in any Jupyter environment
  - Progress tracking via Python API
  - Can generate both static HTML and interactive notebooks
  - Minimal dependencies (no custom Jupyter extension needed)
- Cons:
  - Slight complexity managing both systems
  - Need to design API carefully for both static and interactive use

### Decision: Option D (Hybrid Approach)

**Rationale:**
- JupyterBook provides excellent content navigation and static rendering
- Python module keeps interactivity simple (no browser extensions)
- Students can use either static website OR download notebooks
- Educators can easily customize content
- Progress tracking works across sessions

---

## Implementation Steps

### Phase 1: Content Structure & Module Skeleton

1. [ ] Create `src/dimtensor/education/` module structure:
   - `__init__.py` - Public API exports
   - `textbook.py` - Chapter/Section/Lesson classes
   - `exercises.py` - Exercise types (MultipleChoice, CodeExercise, NumericAnswer)
   - `validation.py` - Answer checking with dimensional validation
   - `progress.py` - ProgressTracker for student progress
   - `hints.py` - Hint system for exercises
   - `grading.py` - Auto-grading logic

2. [ ] Design content organization structure:
   ```
   textbook/
   ├── intro/
   │   ├── chapter_01_units.ipynb
   │   ├── chapter_02_dimensions.ipynb
   │   └── chapter_03_basic_operations.ipynb
   ├── mechanics/
   │   ├── chapter_04_kinematics.ipynb
   │   ├── chapter_05_forces.ipynb
   │   ├── chapter_06_energy.ipynb
   │   └── chapter_07_momentum.ipynb
   ├── waves/
   │   ├── chapter_08_oscillations.ipynb
   │   └── chapter_09_wave_motion.ipynb
   ├── thermodynamics/
   │   ├── chapter_10_temperature.ipynb
   │   ├── chapter_11_heat_transfer.ipynb
   │   └── chapter_12_entropy.ipynb
   ├── electromagnetism/
   │   ├── chapter_13_electrostatics.ipynb
   │   ├── chapter_14_circuits.ipynb
   │   └── chapter_15_magnetism.ipynb
   ├── quantum/
   │   ├── chapter_16_photoelectric.ipynb
   │   ├── chapter_17_wave_functions.ipynb
   │   └── chapter_18_operators.ipynb
   └── advanced/
       ├── chapter_19_relativity.ipynb
       └── chapter_20_statistical_mechanics.ipynb
   ```

3. [ ] Create JupyterBook configuration:
   - `textbook/_config.yml` - JupyterBook settings
   - `textbook/_toc.yml` - Table of contents
   - `textbook/index.md` - Textbook landing page

### Phase 2: Exercise System

4. [ ] Implement base Exercise class:
   - `Exercise.check(answer)` - Validate student answer
   - `Exercise.hint(level)` - Get progressively detailed hints
   - `Exercise.solution()` - Reveal full solution
   - Metadata: difficulty, topic, learning_objectives

5. [ ] Implement exercise types:
   - `MultipleChoiceExercise` - Select from options
   - `NumericAnswerExercise` - Enter number with units
   - `CodeExercise` - Write code that passes tests
   - `DimensionalAnalysisExercise` - Derive dimensions
   - `UnitConversionExercise` - Convert between unit systems
   - `PhysicsWordProblem` - Multi-step calculation

6. [ ] Implement validation logic:
   - Numeric comparison with tolerance
   - Dimensional validation (reject wrong dimensions)
   - Unit-aware comparison (accept equivalent units)
   - Code execution in sandbox
   - Test case validation for CodeExercise

7. [ ] Implement hint system:
   - 3-tier hints: gentle nudge → approach → partial solution
   - Track hint usage in progress
   - Option to disable hints for assessment mode

### Phase 3: Progress Tracking

8. [ ] Implement ProgressTracker:
   - `track_completion(chapter, section)` - Mark as complete
   - `track_exercise(exercise_id, correct, attempts)` - Record exercise attempts
   - `get_progress()` - Return completion percentage
   - `get_weak_topics()` - Identify areas needing review
   - `export_report()` - Generate progress report

9. [ ] Implement storage backends:
   - `LocalFileStorage` - Save to JSON in ~/.dimtensor/progress/
   - `SQLiteStorage` - Save to SQLite database (optional)
   - `LMSStorage` - Integration with Canvas/Moodle (future)

10. [ ] Add progress visualization:
    - `plot_progress()` - Bar chart of chapter completion
    - `plot_mastery()` - Heatmap of topic mastery
    - `plot_timeline()` - Learning activity over time

### Phase 4: Content Creation (Intro → Mechanics)

11. [ ] Create Chapter 1: Units and Dimensions
    - Lesson 1.1: What are physical units?
    - Lesson 1.2: SI base units
    - Lesson 1.3: Creating DimArray with units
    - 5 exercises: unit identification, creating arrays
    - Quiz: 10 questions on unit fundamentals

12. [ ] Create Chapter 2: Dimensional Analysis
    - Lesson 2.1: The seven fundamental dimensions
    - Lesson 2.2: Dimension objects in dimtensor
    - Lesson 2.3: Buckingham Pi theorem
    - 5 exercises: dimensional analysis problems
    - Project: Derive drag equation using dimensions

13. [ ] Create Chapter 3: Basic Operations
    - Lesson 3.1: Addition and subtraction (same dimension)
    - Lesson 3.2: Multiplication and division
    - Lesson 3.3: Powers and roots
    - Lesson 3.4: Transcendental functions (dimensionless only)
    - 8 exercises: array operations
    - Quiz: 12 questions on operations

14. [ ] Create Chapter 4: Kinematics
    - Lesson 4.1: Position, velocity, acceleration
    - Lesson 4.2: Equations of motion
    - Lesson 4.3: 2D and 3D kinematics
    - 10 exercises: motion problems
    - Project: Projectile motion simulator

15. [ ] Create Chapter 5: Forces and Newton's Laws
    - Lesson 5.1: Force as a vector quantity
    - Lesson 5.2: Newton's second law (F=ma)
    - Lesson 5.3: Free body diagrams
    - Lesson 5.4: Friction and drag
    - 12 exercises: force calculations
    - Project: Multiple-object system simulation

16. [ ] Create Chapter 6: Energy and Work
    - Lesson 6.1: Work and kinetic energy
    - Lesson 6.2: Potential energy
    - Lesson 6.3: Conservation of energy
    - Lesson 6.4: Power
    - 10 exercises: energy problems
    - Project: Pendulum energy analysis

17. [ ] Create Chapter 7: Momentum and Collisions
    - Lesson 7.1: Linear momentum
    - Lesson 7.2: Impulse
    - Lesson 7.3: Conservation of momentum
    - Lesson 7.4: Elastic and inelastic collisions
    - 10 exercises: collision problems
    - Project: Two-body collision simulator

### Phase 5: Content Creation (Waves → Quantum)

18. [ ] Create Chapters 8-9: Waves and Oscillations
    - Simple harmonic motion
    - Wave equation
    - Interference and resonance
    - 15 exercises total

19. [ ] Create Chapters 10-12: Thermodynamics
    - Temperature and heat
    - Laws of thermodynamics
    - Entropy and free energy
    - 20 exercises total

20. [ ] Create Chapters 13-15: Electromagnetism
    - Electrostatics (Coulomb's law)
    - Electric circuits (Ohm's law)
    - Magnetism and induction
    - 25 exercises total

21. [ ] Create Chapters 16-18: Quantum Mechanics
    - Photoelectric effect
    - Wave-particle duality
    - Schrödinger equation basics
    - 15 exercises total

22. [ ] Create Chapters 19-20: Advanced Topics
    - Special relativity
    - Statistical mechanics
    - 10 exercises total

### Phase 6: Auto-Grading and Assessment

23. [ ] Implement Quiz system:
    - `Quiz` class with multiple exercises
    - Time limits (optional)
    - Randomized question order
    - Immediate vs. delayed feedback modes

24. [ ] Implement auto-grading:
    - Test suite execution for code exercises
    - Dimensional correctness checking
    - Partial credit for numeric answers within tolerance
    - Rubric-based grading for multi-part problems

25. [ ] Implement learning analytics:
    - Common mistakes identification
    - Time-per-exercise tracking
    - Retry patterns analysis
    - Suggestion engine for review topics

### Phase 7: Jupyter Integration

26. [ ] Create Jupyter cell magic:
    - `%%exercise` - Mark cell as exercise
    - `%%solution` - Mark cell as hidden solution
    - `%%check` - Auto-run validation after cell execution

27. [ ] Create interactive widgets:
    - Exercise submission widget (button + feedback area)
    - Hint request widget (progressive reveal)
    - Progress dashboard widget (show completion)

28. [ ] Create notebook utilities:
    - `load_chapter(n)` - Load chapter n with progress tracking
    - `submit_exercise(id, answer)` - Submit and validate
    - `get_hints(exercise_id)` - Request hints
    - `show_solution(exercise_id)` - Reveal solution

### Phase 8: Documentation and Examples

29. [ ] Create usage guide:
    - `docs/education/getting-started.md` - Setup instructions
    - `docs/education/for-students.md` - Student guide
    - `docs/education/for-educators.md` - Educator guide
    - `docs/education/creating-content.md` - Content creation guide

30. [ ] Create example exercises:
    - `examples/textbook/sample_exercises.ipynb` - All exercise types
    - `examples/textbook/custom_exercise.ipynb` - Creating new exercises
    - `examples/textbook/progress_tracking.ipynb` - Using ProgressTracker

### Phase 9: Testing and Polish

31. [ ] Write comprehensive tests:
    - Test all exercise types
    - Test validation logic (dimensional, numeric, code)
    - Test progress tracking (save/load)
    - Test hint system
    - Test grading logic

32. [ ] Test content notebooks:
    - Verify all code examples run
    - Check all exercises have solutions
    - Validate all equations render correctly
    - Test in fresh Jupyter environment

33. [ ] User testing:
    - Pilot with 5-10 students
    - Gather feedback on difficulty progression
    - Identify confusing sections
    - Iterate based on feedback

### Phase 10: Deployment

34. [ ] Build JupyterBook static site:
    - `jupyter-book build textbook/`
    - Deploy to GitHub Pages or ReadTheDocs
    - Add search functionality

35. [ ] Create distribution package:
    - Ensure textbook content is included in package
    - Update pyproject.toml with education extras
    - Test pip install with textbook

36. [ ] Create educator resources:
    - Answer keys (separate repository or encrypted)
    - Slide decks for each chapter
    - Assessment bank (additional exercises)
    - Customization guide

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/education/__init__.py` | **CREATE** - Module initialization, public API |
| `src/dimtensor/education/textbook.py` | **CREATE** - Chapter/Section/Lesson classes (300 lines) |
| `src/dimtensor/education/exercises.py` | **CREATE** - Exercise type implementations (600 lines) |
| `src/dimtensor/education/validation.py` | **CREATE** - Answer validation logic (400 lines) |
| `src/dimtensor/education/progress.py` | **CREATE** - ProgressTracker and storage (500 lines) |
| `src/dimtensor/education/hints.py` | **CREATE** - Hint system (200 lines) |
| `src/dimtensor/education/grading.py` | **CREATE** - Auto-grading logic (300 lines) |
| `src/dimtensor/education/widgets.py` | **CREATE** - Jupyter widgets (400 lines) |
| `textbook/_config.yml` | **CREATE** - JupyterBook configuration |
| `textbook/_toc.yml` | **CREATE** - Table of contents |
| `textbook/index.md` | **CREATE** - Textbook homepage |
| `textbook/intro/chapter_01_units.ipynb` | **CREATE** - Chapter 1 notebook (~200 cells) |
| `textbook/intro/chapter_02_dimensions.ipynb` | **CREATE** - Chapter 2 notebook (~180 cells) |
| `textbook/intro/chapter_03_operations.ipynb` | **CREATE** - Chapter 3 notebook (~200 cells) |
| `textbook/mechanics/chapter_04_kinematics.ipynb` | **CREATE** - Chapter 4 notebook (~250 cells) |
| `textbook/mechanics/chapter_05_forces.ipynb` | **CREATE** - Chapter 5 notebook (~280 cells) |
| `textbook/mechanics/chapter_06_energy.ipynb` | **CREATE** - Chapter 6 notebook (~250 cells) |
| `textbook/mechanics/chapter_07_momentum.ipynb` | **CREATE** - Chapter 7 notebook (~250 cells) |
| `textbook/waves/chapter_08_oscillations.ipynb` | **CREATE** - Chapter 8 notebook (~200 cells) |
| `textbook/waves/chapter_09_wave_motion.ipynb` | **CREATE** - Chapter 9 notebook (~200 cells) |
| `textbook/thermodynamics/chapter_10_*.ipynb` | **CREATE** - Chapters 10-12 (3 files, ~600 cells) |
| `textbook/electromagnetism/chapter_13_*.ipynb` | **CREATE** - Chapters 13-15 (3 files, ~750 cells) |
| `textbook/quantum/chapter_16_*.ipynb` | **CREATE** - Chapters 16-18 (3 files, ~600 cells) |
| `textbook/advanced/chapter_19_*.ipynb` | **CREATE** - Chapters 19-20 (2 files, ~400 cells) |
| `tests/test_education/` | **CREATE** - Test suite for education module |
| `docs/education/getting-started.md` | **CREATE** - Setup guide |
| `docs/education/for-students.md` | **CREATE** - Student guide |
| `docs/education/for-educators.md` | **CREATE** - Educator guide |
| `pyproject.toml` | **UPDATE** - Add jupyter-book to education extras |
| `src/dimtensor/__init__.py` | **UPDATE** - Expose education module |

**Total estimate:**
- ~2,500 lines of Python code (education module)
- ~20 notebook files (~4,000 notebook cells total)
- ~150+ exercises across all chapters
- ~50+ interactive examples
- ~15 markdown documentation pages

---

## Testing Strategy

### Unit Tests
- [ ] Test each exercise type independently
- [ ] Test validation with correct/incorrect answers
- [ ] Test dimensional validation edge cases
- [ ] Test progress tracker save/load
- [ ] Test hint revelation system
- [ ] Test grading logic with partial credit

### Integration Tests
- [ ] Test full exercise submission workflow
- [ ] Test progress tracking across multiple chapters
- [ ] Test quiz mode with time limits
- [ ] Test export/import of progress

### Content Tests
- [ ] Run all notebook cells in sequence
- [ ] Verify all exercises have solutions
- [ ] Check all dimensional annotations are correct
- [ ] Validate all equation references exist in database
- [ ] Test exercises with student-like incorrect answers

### User Acceptance Tests
- [ ] Pilot with physics students (n=5-10)
- [ ] Pilot with educators (n=2-3)
- [ ] Measure completion rates
- [ ] Gather qualitative feedback
- [ ] Test on Windows, Mac, Linux

### Performance Tests
- [ ] Load time for notebooks with many exercises
- [ ] Progress save/load time
- [ ] JupyterBook build time
- [ ] Widget responsiveness

---

## Risks / Edge Cases

### Risk 1: Content Creation is Massive
- **Impact**: 20 chapters × ~200 cells = 4,000 cells to create
- **Mitigation**:
  - Phase the rollout (intro + mechanics first, then expand)
  - Reuse existing examples/ notebooks where possible
  - Create template notebooks to speed up authoring
  - Generate some exercises programmatically (e.g., unit conversions)
- **Fallback**: Launch with 8-10 chapters, expand in v5.2.0

### Risk 2: Exercise Validation is Complex
- **Impact**: Need to handle many answer formats (numeric, code, symbolic)
- **Mitigation**:
  - Start with simple exercise types (multiple choice, numeric)
  - Use existing dimtensor validation (dimensional checking)
  - Sandbox code execution for security
  - Extensive testing with edge cases
- **Fallback**: Limit to multiple choice + numeric for v5.1.0

### Risk 3: Progress Persistence Across Sessions
- **Impact**: Students expect their progress to be saved
- **Mitigation**:
  - Use local file storage (~/.dimtensor/progress/)
  - JSON format for portability
  - Auto-save after each exercise
  - Provide export/import for backup
- **Fallback**: Session-only progress, manual save

### Risk 4: Jupyter Widget Compatibility
- **Impact**: Widgets may not work in all Jupyter environments (JupyterLab, Colab, etc.)
- **Mitigation**:
  - Use ipywidgets (widely supported)
  - Provide fallback to function calls
  - Test in JupyterLab, Jupyter Notebook, Google Colab
  - Document compatibility matrix
- **Fallback**: Pure Python API without widgets

### Risk 5: Content May Become Outdated
- **Impact**: API changes break notebook examples
- **Mitigation**:
  - Pin dimtensor version in textbook requirements
  - Run CI tests on all notebooks
  - Version textbook alongside library (textbook v5.1 uses dimtensor v5.1)
  - Keep content focused on stable core API

### Edge Case: Students Without Jupyter
- **Handling**: Provide Binder links for cloud execution
- **Handling**: Static HTML version via JupyterBook (read-only)

### Edge Case: Educators Want Custom Content
- **Handling**: Document how to create custom exercises
- **Handling**: Provide templates for new chapters
- **Handling**: Make all exercise classes easily extendable

### Edge Case: Accessibility (Screen Readers)
- **Handling**: Ensure all equations have alt text
- **Handling**: Test with screen reader software
- **Handling**: Provide text descriptions of visualizations

### Edge Case: International Students
- **Handling**: Use clear, simple English
- **Handling**: Avoid cultural references
- **Handling**: Plan for i18n in v5.2.0 (task #263)

---

## Definition of Done

### Core Infrastructure
- [ ] education module created with all classes
- [ ] All exercise types implemented and tested
- [ ] Progress tracking saves/loads correctly
- [ ] Auto-grading works for all exercise types
- [ ] Jupyter widgets functional in JupyterLab

### Content (Minimum Viable Textbook)
- [ ] Chapters 1-7 complete (intro + mechanics)
- [ ] 50+ exercises with solutions
- [ ] 20+ interactive examples
- [ ] All code cells tested and working

### Documentation
- [ ] Student guide complete
- [ ] Educator guide complete
- [ ] Content creation guide complete
- [ ] API documentation for education module

### Quality
- [ ] 90%+ test coverage for education module
- [ ] User testing with 5+ students
- [ ] Educator review (2+ educators)
- [ ] All notebooks pass CI tests

### Deployment
- [ ] JupyterBook builds without errors
- [ ] Static site deployed (GitHub Pages)
- [ ] pip install dimtensor[education] works
- [ ] Binder links functional

---

## Notes / Log

**[Initial Research - 2026-01-13]**

Analyzed existing educational infrastructure:
- 5 tutorial notebooks exist but are disconnected
- Comprehensive equation database (246 equations)
- Dataset registry with real physics data
- No structured curriculum or progress tracking
- No interactive exercise system

**[Design Decisions]**

1. **Why Hybrid (JupyterBook + Python module)?**
   - Static site for browsing (no Jupyter needed)
   - Interactive notebooks for hands-on learning
   - Progress tracking via Python API
   - Best of both worlds

2. **Why 20 Chapters?**
   - Covers undergraduate physics curriculum
   - Intro (3) + Mechanics (4) + Waves (2) + Thermo (3) + EM (3) + Quantum (3) + Advanced (2)
   - Can be taught as 1-2 semester course
   - Modular: students can skip to relevant chapters

3. **Why Multiple Exercise Types?**
   - Multiple choice: Quick concept checks
   - Numeric: Traditional physics problems
   - Code: Develop programming skills
   - Dimensional analysis: Core dimtensor skill
   - Word problems: Real-world application

4. **Why Auto-Grading?**
   - Immediate feedback improves learning
   - Reduces instructor workload
   - Enables self-paced learning
   - Provides analytics for educators

**[Content Reuse Opportunities]**

Existing notebooks can be integrated:
- `examples/01_basics.ipynb` → Chapter 1-2 material
- `examples/02_physics_simulation.ipynb` → Chapter 4-7 projects
- `examples/03_pytorch_training.ipynb` → Advanced chapter
- Equations database → Reference in relevant chapters
- Datasets → Use in exercises and projects

**[Phased Rollout Plan]**

v5.1.0 (current):
- Core education module
- Chapters 1-7 (intro + mechanics)
- Basic exercise types
- Local progress tracking

v5.2.0 (future):
- Chapters 8-15 (waves + thermo + EM)
- Advanced exercise types
- LMS integration
- i18n support

v5.3.0 (future):
- Chapters 16-20 (quantum + advanced)
- Video content integration
- Gamification (badges, leaderboards)
- Social features (discussion forums)

**[Estimated Complexity]**

**HIGH** - This is a substantial feature:
- ~2,500 lines of new Python code
- ~4,000 notebook cells to create
- ~150 exercises with validation logic
- Comprehensive testing required
- User testing and iteration

Estimated time: 40-60 hours for MVP (Chapters 1-7)
Full 20-chapter system: 100-120 hours

**[Success Metrics]**

How will we measure success?
1. Adoption: 100+ students complete Chapter 1 in first 3 months
2. Engagement: 70%+ completion rate for started chapters
3. Quality: 4.0+ star rating from student feedback
4. Learning: Students correctly solve 80%+ of exercises
5. Retention: 50%+ of students complete 3+ chapters

---

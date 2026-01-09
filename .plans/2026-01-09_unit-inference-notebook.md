# Plan: Unit Inference Notebook Example

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create an interactive Jupyter notebook (`examples/05_unit_inference.ipynb`) that demonstrates dimtensor's automatic unit inference system, including variable name heuristics, equation pattern matching, the constraint solver, and the CLI linting tool. The notebook should serve as both a tutorial and a showcase of dimtensor's ability to catch dimensional bugs before runtime.

---

## Background

Dimtensor v3.3.0 includes a comprehensive unit inference system with:
- Variable name pattern matching (150+ patterns across mechanics, EM, thermo, etc.)
- Physics equation database (67+ equations)
- Constraint solver for inferring unknown units from equations
- CLI linting tool for static code analysis

Currently there are no example notebooks in the repository. This will be the first comprehensive tutorial demonstrating these features in an interactive format.

---

## Approach

### Option A: Reference-Style Documentation
- Exhaustive coverage of all features
- Organized by feature category
- Pros: Complete reference, good for documentation
- Cons: May be dry, less engaging for learning

### Option B: Tutorial-Style with Real Physics Problems
- Start with motivation (catching bugs)
- Build up from simple to complex examples
- Include real-world physics simulations with bugs
- Demonstrate how inference catches errors
- Pros: Engaging, educational, shows practical value
- Cons: May not cover every feature exhaustively

### Option C: Mixed Approach
- Start with tutorial-style motivation and bug examples
- Progress through features systematically
- Include both simple demonstrations and realistic applications
- Pros: Best of both worlds - engaging and comprehensive
- Cons: May be longer (but 20-25 cells should accommodate this)

### Decision: Option C - Mixed Approach

The notebook will:
1. Start with a compelling bug example that inference catches
2. Systematically cover each inference feature
3. Build up to realistic physics examples
4. Demonstrate the CLI linting tool
5. End with best practices and limitations

This approach is engaging, educational, and comprehensive.

---

## Implementation Steps

### Notebook Structure (20-25 cells)

1. [ ] **Title & Overview Cell** (markdown)
   - Introduction to unit inference
   - Why it matters for scientific computing
   - What features are covered

2. [ ] **Motivation: The Bug** (markdown + code)
   - Show a buggy physics calculation (velocity + time)
   - Run it without dimtensor - no error!
   - Explain the problem

3. [ ] **Setup Cell** (code)
   - Import statements
   - Basic dimtensor setup

4. [ ] **Variable Name Heuristics - Basics** (markdown + code)
   - Demonstrate `infer_dimension()` with simple names
   - Show confidence scores
   - Examples: velocity, force, energy

5. [ ] **Exact Pattern Matches** (code)
   - Show high-confidence exact matches
   - Multiple physics domains (mechanics, EM, thermo)
   - Display a table of common patterns

6. [ ] **Prefix and Suffix Patterns** (code)
   - Demonstrate prefix stripping (initial_velocity, max_force)
   - Unit suffixes (distance_m, energy_j)
   - Component notation (velocity_x, force_y)

7. [ ] **Ambiguous Names** (code)
   - Show `get_matching_patterns()` for ambiguous names
   - Example: "v" could be velocity or voltage
   - Display all possible interpretations sorted by confidence

8. [ ] **Equation Database Overview** (markdown + code)
   - Introduce the equation database
   - Show total number of equations and domains
   - List available domains

9. [ ] **Searching Equations by Domain** (code)
   - `get_equations_by_domain()` examples
   - Display mechanics equations
   - Show equation details (variables, formula, tags)

10. [ ] **Searching Equations by Tag** (code)
    - `get_equations_by_tag()` examples
    - Find all energy-related equations
    - Find all fundamental equations

11. [ ] **Finding Equations with Variables** (code)
    - `find_equations_with_variable()` examples
    - Search for equations containing "F", "v", etc.
    - Show how to get dimension suggestions from equations

12. [ ] **Constraint Solver - Basic** (markdown + code)
    - Introduce `infer_units()` function
    - Simple example: F = m * a
    - Show inferred units and confidence

13. [ ] **Constraint Solver - Complex Equations** (code)
    - Kinetic energy: KE = 0.5 * m * v^2
    - Power equations with division
    - Multiple variable inference

14. [ ] **Detecting Inconsistencies** (code)
    - Demonstrate dimensionally inconsistent equation
    - Show error detection: F = m + a (invalid!)
    - Display error messages

15. [ ] **Real Physics Example: Projectile Motion** (markdown + code)
    - Build a projectile motion calculator
    - Intentionally introduce a dimensional bug
    - Show how inference catches it

16. [ ] **Bug Example #1: Unit Confusion** (code)
    - Common bug: mixing units (km + m without conversion)
    - Show inference detection
    - Correct the code

17. [ ] **Bug Example #2: Dimensional Mismatch** (code)
    - Adding incompatible quantities (velocity + acceleration)
    - Demonstrate error catching
    - Show the fix

18. [ ] **CLI Linting Tool - Introduction** (markdown + code)
    - Write buggy code to a temporary file
    - Run `dimtensor lint` programmatically
    - Display lint output

19. [ ] **CLI Linting - Strict Mode** (code)
    - Run linting with --strict flag
    - Show inference suggestions
    - Demonstrate JSON output format

20. [ ] **CLI Linting - Integration** (markdown)
    - Explain CI/CD integration
    - Pre-commit hooks
    - IDE integration
    - Exit codes for automation

21. [ ] **Confidence Levels & Accuracy** (code)
    - Explain confidence scores (0.9, 0.7, 0.5)
    - Show when inference is reliable
    - When to use explicit units instead

22. [ ] **Best Practices** (markdown)
    - When to rely on inference vs explicit units
    - Naming conventions for maximum accuracy
    - Production code guidelines
    - Testing strategies

23. [ ] **Limitations & Caveats** (markdown + code)
    - Ambiguous variable names
    - Non-standard naming conventions
    - Complex coupled equations
    - When solver may fail

24. [ ] **Summary & Next Steps** (markdown)
    - Recap of inference capabilities
    - Links to documentation
    - Encourage experimentation

25. [ ] **Resources** (markdown)
    - Links to API docs
    - Equation database reference
    - CLI tool documentation
    - Other example notebooks (future)

---

## Files to Modify

| File | Change |
|------|--------|
| examples/05_unit_inference.ipynb | Create new Jupyter notebook with 20-25 cells |

---

## Testing Strategy

How will we verify this works?

- [ ] Execute notebook cell-by-cell to ensure no errors
- [ ] Verify all code examples run successfully
- [ ] Check that imports work (dimtensor, inference module)
- [ ] Confirm CLI linting demonstrations work
- [ ] Test on fresh environment with dimtensor installed
- [ ] Verify output displays are clear and informative
- [ ] Check that markdown formatting renders correctly
- [ ] Ensure file I/O for linting examples works properly

### Manual Testing Checklist
- [ ] Run notebook in Jupyter Lab
- [ ] Run notebook in Jupyter Notebook
- [ ] Run notebook in VS Code
- [ ] Execute with `jupyter nbconvert --execute`
- [ ] Verify no deprecation warnings
- [ ] Check that temporary files are cleaned up

---

## Risks / Edge Cases

### Risk 1: CLI Tool Execution in Notebook
**Risk**: Running subprocess commands for `dimtensor lint` may behave differently in notebook environment
**Mitigation**: Use Python API (`lint_file()`) directly when possible, only demonstrate CLI for illustration

### Risk 2: Temporary File Management
**Risk**: Creating temporary Python files for linting demos could leave files behind or have permission issues
**Mitigation**: Use `tempfile` module with proper cleanup, or write to examples/ directory with clear naming

### Risk 3: Import Dependencies
**Risk**: Inference module is optional and may not be installed
**Mitigation**: Include clear setup instructions, check imports with try/except, provide helpful error messages

### Risk 4: Notebook Length
**Risk**: 20-25 cells might be too long or too short
**Mitigation**: Aim for 22-23 cells as baseline, can adjust based on content density

### Risk 5: Output Verbosity
**Risk**: Some examples (like equation database listing) may produce very long output
**Mitigation**: Use `head()` style truncation, show representative samples, provide links to full docs

### Edge Case 1: Empty Examples Directory
**Edge case**: This is the first notebook, so no other examples to reference
**Handling**: Structure as standalone, mention "more examples coming soon"

### Edge Case 2: Equation Database Evolution
**Edge case**: Number of equations may change in future versions
**Handling**: Use dynamic queries (`len(EQUATION_DATABASE)`) rather than hardcoded counts

### Edge Case 3: NumPy Version Compatibility
**Edge case**: Notebook might run on NumPy 1.x or 2.x
**Handling**: Test with project's pinned NumPy version (<2.0), add version check cell if needed

---

## Definition of Done

- [ ] Notebook created with 20-25 cells covering all required features
- [ ] All code cells execute without errors
- [ ] Markdown cells are well-formatted and informative
- [ ] Examples demonstrate:
  - [ ] Variable name heuristics (exact, prefix, suffix, component)
  - [ ] Equation pattern database (search by domain, tag, variable)
  - [ ] Constraint solver (basic, complex, error detection)
  - [ ] CLI linting tool (basic, strict mode, integration)
  - [ ] Real bug catching examples
- [ ] Output is clear and educational
- [ ] Notebook can be executed from top to bottom without errors
- [ ] Temporary files are properly managed
- [ ] Best practices and limitations are documented
- [ ] CONTINUITY.md updated with task completion

---

## Notes / Log

**[PLANNING]** - Initial plan created based on:
- inference module API (heuristics.py, equations.py, solver.py)
- CLI linting tool (cli/lint.py)
- Documentation (docs/guide/inference.md)
- Project structure (no existing examples, first notebook)

**Key Decisions**:
1. Mixed tutorial/reference approach for engagement and completeness
2. Target 22-23 cells as baseline (within 20-25 range)
3. Use Python API for linting when possible, subprocess for demonstration
4. Focus on practical bug-catching examples
5. Include real physics simulations
6. Clear best practices and limitations section

**Structure Philosophy**:
- Start with "why" (motivation, bugs)
- Build up systematically through features
- Mix simple demos with realistic examples
- End with practical guidance (best practices, limitations)

**Content Balance**:
- 30% variable name heuristics
- 25% equation database and solver
- 25% CLI linting and bug examples
- 20% best practices, limitations, resources

---

# AGENTS.md

Instructions for AI coding agents working on this repository.

---

## FIRST: Read CONTINUITY.md

**Before doing anything else**, read `CONTINUITY.md` in the project root.

It contains:
- Current task queue (what to work on)
- Session log (what's been done)
- Workflow rules (how to work)

**Your job**: Work through the TASK QUEUE in CONTINUITY.md until it's empty.

---

## Workflow

```
1. Read CONTINUITY.md
2. Update AGENT CHECKIN section
3. Check for 🗺️ PLAN REQUIRED markers in task queue
4. CREATE PLAN BEFORE CODING (if required)
5. Work through TASK QUEUE
6. Update CONTINUITY.md after each task
7. KEEP GOING - don't stop to ask for approval
8. Only stop if: tests fail, blocked, or queue empty
```

---

## Planning (MANDATORY FOR NEW FILES)

⚠️ **BEFORE creating any new .py file, you MUST create a plan first.**

```bash
# Step 1: Copy template
cp .plans/_TEMPLATE.md .plans/$(date +%Y-%m-%d)_feature-name.md

# Step 2: Fill out the plan
# - Goal: What are we building?
# - Approach: How will we build it?
# - Implementation Steps: Ordered list of steps
# - Files to Modify: What files will be created/changed?

# Step 3: THEN start coding
```

**PLAN REQUIRED when**:
- Creating a new file (any .py file)
- Task has 🗺️ marker in CONTINUITY.md
- Adding a new module or feature

**Plan NOT required when**:
- Editing existing files only
- Running tests/deploys
- Updating docs

**Why?** Your context WILL be compacted. Plans survive. Your working memory doesn't.

---

## Skills (Auto-Loaded Guidance)

Skills provide specialized knowledge that's auto-loaded based on your task. Located in `.claude/skills/`.

| Skill | When It Activates |
|-------|-------------------|
| `units-design` | Designing new unit modules (astronomy, chemistry, etc.) |
| `deploy` | Deploying to PyPI |
| `code-review` | Reviewing code for correctness |

**How skills work**: Skills are automatically detected based on your task description. When you're working on something that matches a skill's description, Claude will load the skill's instructions automatically.

**To explicitly use a skill**: Just mention what you're doing. For example:
- "I'm designing astronomy units" → loads `units-design` skill
- "Deploy to PyPI" → loads `deploy` skill
- "Review the new module" → loads `code-review` skill

---

## Spawning Specialized Agents

For complex sub-tasks, you can spawn specialized agents using the Task tool:

```
Use the Task tool with subagent_type="general-purpose" and a prompt like:
"Use the code-reviewer agent to review src/dimtensor/domains/astronomy.py.
Read .claude/agents/code-reviewer.md for instructions."
```

Available agents in `.claude/agents/`:
- `code-reviewer` - Detailed code review with dimtensor patterns
- `test-writer` - Write tests following dimtensor conventions

Agents run in separate contexts and return results when done.

---

## Build & Test

```bash
# Setup
pip install -e ".[dev]"

# Test (REQUIRED before any commit)
pytest

# Type check
mypy src/dimtensor --ignore-missing-imports

# Coverage
pytest --cov=dimtensor --cov-report=term-missing

# Lint
ruff check src/dimtensor
```

---

## Deploy

```bash
# Update version in BOTH:
# - pyproject.toml (line ~7)
# - src/dimtensor/__init__.py (line ~35)

# Build and deploy
rm -rf dist/ build/
python -m build
twine upload dist/*

# Commit
git add -A
git commit -m "Release vX.Y.Z: Description"
git push origin main
```

---

## Project Structure

```
src/dimtensor/
├── core/
│   ├── dimensions.py   # Dimension: 7-tuple SI exponents
│   ├── units.py        # Unit: dimension + scale
│   └── dimarray.py     # DimArray: numpy wrapper
├── torch/
│   └── dimtensor.py    # DimTensor: PyTorch wrapper
├── jax/
│   └── dimarray.py     # JAX DimArray with pytree
├── io/
│   ├── json.py         # JSON serialization
│   ├── pandas.py       # Pandas integration
│   └── hdf5.py         # HDF5 support
├── constants/          # Physical constants (CODATA 2022)
├── benchmarks.py       # Performance measurement
├── functions.py        # Array functions
├── errors.py           # Custom exceptions
└── config.py           # Display options

tests/                  # pytest tests for each module
.plans/                 # Planning documents
CONTINUITY.md           # Task queue and session log
ROADMAP.md              # Long-term vision
```

---

## Code Patterns

- Use `DimArray._from_data_and_unit(data, unit)` internally (no copy)
- Operations return new instances (immutable style)
- Follow existing patterns in core/dimarray.py
- All new functionality needs tests
- Run `pytest` before any commit

---

## Adding Features

**New serialization format** (io/):
```python
# Follow pattern from io/hdf5.py
def save_format(arr: DimArray, path: str) -> None:
    # Store data + unit metadata

def load_format(path: str) -> DimArray:
    # Reconstruct with unit
```

**New unit domain** (e.g., astronomy):
```python
# Create new file in domains/ or add to units.py
parsec = Unit("pc", Dimension(length=1), 3.0857e16)
```

---

## Current Status

See CONTINUITY.md for current version and task queue.
See ROADMAP.md for long-term plans.

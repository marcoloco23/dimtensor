# Dimensional Linting CLI

## Goal
Implement a `dimtensor lint` command that analyzes Python files for potential dimensional issues using variable name heuristics.

## Approach
Create a CLI tool that:
1. Parses Python source files using AST
2. Extracts variable names and assignments
3. Uses inference module to suggest dimensions
4. Detects potential mismatches (e.g., adding velocity + acceleration)
5. Reports findings with confidence levels

## Implementation

### File Structure
```
src/dimtensor/
├── cli/
│   ├── __init__.py
│   └── lint.py      # Main linting logic
└── __main__.py      # Entry point for python -m dimtensor
```

### CLI Interface
```bash
# Lint a single file
dimtensor lint script.py

# Lint a directory
dimtensor lint src/

# Output formats
dimtensor lint --format=text script.py   # Default
dimtensor lint --format=json script.py   # JSON for IDE integration

# Strictness levels
dimtensor lint --strict script.py        # Report all potential issues
dimtensor lint --relaxed script.py       # Only high-confidence issues
```

### Linting Rules
1. **Addition/Subtraction mismatch**: `velocity + acceleration` (L/T + L/T²)
2. **Suspicious assignments**: `result = mass * time` (suggests mass*time = M·T)
3. **Missing dimension hints**: Variables that could benefit from explicit units
4. **Known physics violations**: E = mc (missing c²)

### Output Format
```
script.py:15:4: W001 Potential dimension mismatch
  velocity + acceleration
  ~~~~~~~~   ~~~~~~~~~~~~
  L·T⁻¹      L·T⁻²
  Suggestion: These have different dimensions and cannot be added

script.py:23:4: I001 Dimension inference
  result = mass * velocity
  Inferred dimension: M·L·T⁻¹ (momentum)
```

## Files to Modify/Create
1. CREATE: src/dimtensor/cli/__init__.py
2. CREATE: src/dimtensor/cli/lint.py
3. CREATE: src/dimtensor/__main__.py
4. MODIFY: pyproject.toml (add CLI entry point)
5. CREATE: tests/test_lint.py

## Dependencies
- Standard library only (ast, argparse, pathlib)
- Uses existing inference module

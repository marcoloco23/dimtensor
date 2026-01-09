# Contributing to dimtensor

Thank you for your interest in contributing to dimtensor! This document provides guidelines and instructions for contributing.

## Ways to Contribute

### Report Bugs

Found a bug? [Open an issue](https://github.com/marcoloco23/dimtensor/issues/new) with:

- Your dimtensor version (`pip show dimtensor`)
- Python version (`python --version`)
- Operating system
- Minimal code to reproduce the issue
- Full error traceback

### Suggest Features

Have an idea? [Start a discussion](https://github.com/marcoloco23/dimtensor/discussions) or open an issue describing:

- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

### Improve Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples
- Improve API documentation
- Write tutorials

### Contribute Code

Ready to contribute code? Follow the development setup below.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git

### Setup

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/dimtensor.git
   cd dimtensor
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**

   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify installation**

   ```bash
   pytest
   ```

## Development Workflow

### 1. Create a branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make your changes

Write your code following our style guidelines (below).

### 3. Run tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dimtensor --cov-report=term-missing

# Run specific tests
pytest tests/test_dimarray.py -v
```

### 4. Check types

```bash
mypy src/dimtensor --ignore-missing-imports
```

### 5. Format and lint

```bash
ruff check src/dimtensor
ruff format src/dimtensor
```

### 6. Commit your changes

```bash
git add .
git commit -m "Add feature: description of your change"
```

### 7. Push and create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub.

## Code Style

### Python Style

We follow PEP 8 with these specifics:

- **Line length**: 88 characters (Black default)
- **Imports**: Sorted with isort (ruff handles this)
- **Quotes**: Double quotes for strings
- **Type hints**: Required for public APIs

### Example

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dimtensor.core.dimensions import Dimension
from dimtensor.errors import DimensionError

if TYPE_CHECKING:
    from dimtensor.core.units import Unit


def example_function(
    data: np.ndarray,
    unit: Unit,
    *,
    validate: bool = True,
) -> DimArray:
    """Create a DimArray from data and unit.

    Args:
        data: The numerical data.
        unit: The physical unit.
        validate: Whether to validate inputs.

    Returns:
        A new DimArray instance.

    Raises:
        DimensionError: If dimensions are incompatible.
    """
    if validate:
        _validate_data(data)
    return DimArray._from_data_and_unit(data, unit)
```

### Docstrings

Use Google-style docstrings:

```python
def function(arg1: int, arg2: str) -> bool:
    """Short description.

    Longer description if needed.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When something is wrong.

    Examples:
        >>> function(1, "test")
        True
    """
```

## Testing Guidelines

### Test Structure

- Tests go in `tests/` directory
- Mirror the source structure (e.g., `tests/test_dimarray.py`)
- Use pytest fixtures for common setup

### Writing Tests

```python
import pytest
import numpy as np
from dimtensor import DimArray, units


class TestDimArrayCreation:
    """Tests for DimArray creation."""

    def test_create_with_unit(self):
        """DimArray can be created with a unit."""
        arr = DimArray([1, 2, 3], units.m)
        assert arr.unit == units.m
        np.testing.assert_array_equal(arr.data, [1, 2, 3])

    def test_dimension_error_on_invalid_add(self):
        """Adding incompatible dimensions raises DimensionError."""
        a = DimArray([1], units.m)
        b = DimArray([1], units.s)
        with pytest.raises(DimensionError):
            a + b
```

### Test Coverage

We aim for high test coverage. Run coverage to check:

```bash
pytest --cov=dimtensor --cov-report=html
open htmlcov/index.html  # View report
```

## Pull Request Guidelines

### Before Submitting

- [ ] Tests pass (`pytest`)
- [ ] Type checking passes (`mypy src/dimtensor --ignore-missing-imports`)
- [ ] Code is formatted (`ruff format src/dimtensor`)
- [ ] Linting passes (`ruff check src/dimtensor`)
- [ ] Documentation is updated if needed
- [ ] CHANGELOG.md is updated for user-facing changes

### PR Description

Include:

- What the PR does
- Why it's needed
- How to test it
- Any breaking changes

### Review Process

1. A maintainer will review your PR
2. Address any feedback
3. Once approved, your PR will be merged

## Release Process

Releases are handled by maintainers:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a GitHub release
4. CI publishes to PyPI

## Getting Help

- [GitHub Discussions](https://github.com/marcoloco23/dimtensor/discussions) - Questions and ideas
- [GitHub Issues](https://github.com/marcoloco23/dimtensor/issues) - Bug reports and feature requests

## Code of Conduct

Be respectful and inclusive. We're all here to build great software together.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

# Unit Inference and Linting

> Automatic dimension inference and static analysis for dimensional correctness

Dimtensor provides sophisticated tools to automatically infer physical dimensions from variable names, match equation patterns, and detect dimensional bugs in your code before runtime.

## Table of Contents

- [Overview](#overview)
- [Variable Name Patterns](#variable-name-patterns)
- [Equation Pattern Database](#equation-pattern-database)
- [Unit Inference from Equations](#unit-inference-from-equations)
- [CLI Linting Tool](#cli-linting-tool)
- [IDE Integration](#ide-integration)
- [Best Practices](#best-practices)

---

## Overview

### What is Inference?

Dimtensor's inference system uses three complementary techniques to help catch dimensional bugs:

1. **Variable name heuristics**: Infers dimensions from names like `velocity`, `force`, or `energy_j`
2. **Equation pattern matching**: Recognizes common physics equations like F = ma or E = mc²
3. **Constraint solving**: Propagates dimensional information through equations to infer unknown units
4. **Static code analysis**: Scans Python files to detect potential dimensional mismatches

### Why Use Inference?

Inference helps you:

- **Catch bugs early**: Detect dimensional errors before running your code
- **Write cleaner code**: Reduce boilerplate when prototyping
- **Learn physics**: Get suggestions for correct dimensions based on variable names
- **Enforce correctness**: Add dimensional checks to your CI/CD pipeline

### Quick Example: Catching a Bug

Without inference, this dimensional error goes undetected until runtime:

```python
# Buggy physics simulation
velocity = 10  # m/s (forgot units!)
time = 5       # s (forgot units!)
distance = velocity + time  # BUG: adding velocity + time!
```

With dimtensor linting:

```bash
$ dimtensor lint buggy.py
buggy.py:3:12: W002 Potential dimension mismatch: L·T⁻¹ + T
  distance = velocity + time
  Suggestion: Cannot add/subtract L·T⁻¹ and T. Check your units.
```

---

## Variable Name Patterns

### How It Works

Dimtensor analyzes variable names using pattern matching to suggest physical dimensions. It recognizes:

- Exact matches: `velocity`, `force`, `energy`
- Prefixed names: `initial_velocity`, `max_force`
- Suffixed units: `distance_m`, `time_s`
- Component notation: `velocity_x`, `force_y`

### Confidence Levels

Each inference has a confidence score:

| Confidence | Source | Example |
|------------|--------|---------|
| **0.9** | Exact match | `velocity` → L·T⁻¹ |
| **0.7** | Prefix/suffix stripped | `initial_velocity_x` → L·T⁻¹ |
| **0.7** | Unit suffix | `distance_m` → L |
| **0.5** | Partial match | `particle_velocity` → L·T⁻¹ |

### Basic Usage

```python
from dimtensor.inference import infer_dimension

# Exact match - high confidence
result = infer_dimension("velocity")
print(result.dimension)   # L·T⁻¹
print(result.confidence)  # 0.9
print(result.source)      # "exact"

# With prefix - medium confidence
result = infer_dimension("initial_velocity")
print(result.dimension)   # L·T⁻¹
print(result.confidence)  # 0.7
print(result.source)      # "prefix_stripped"

# With component suffix
result = infer_dimension("velocity_x")
print(result.dimension)   # L·T⁻¹
print(result.confidence)  # 0.7

# No match
result = infer_dimension("foo")
print(result)  # None
```

### Multiple Matches

Some names have multiple possible interpretations:

```python
from dimtensor.inference import get_matching_patterns

# Get all possible interpretations
results = get_matching_patterns("v")
for r in results:
    print(f"{r.pattern}: {r.dimension} (confidence={r.confidence:.1f}, source={r.source})")

# Output:
# velocity: L·T⁻¹ (confidence=0.9, source=exact)
# voltage: L²·M·T⁻³·I⁻¹ (confidence=0.9, source=exact)
```

The function returns matches sorted by confidence, so the first result is the most likely interpretation.

### Common Patterns

#### Mechanics

| Pattern | Dimension | Description |
|---------|-----------|-------------|
| `velocity`, `v`, `speed` | L·T⁻¹ | Velocity (m/s) |
| `acceleration`, `accel`, `a` | L·T⁻² | Acceleration (m/s²) |
| `force`, `f` | M·L·T⁻² | Force (N) |
| `momentum`, `p` | M·L·T⁻¹ | Momentum (kg·m/s) |
| `energy`, `e` | M·L²·T⁻² | Energy (J) |
| `power` | M·L²·T⁻³ | Power (W) |
| `pressure` | M·L⁻¹·T⁻² | Pressure (Pa) |
| `mass`, `m` | M | Mass (kg) |
| `density`, `rho` | M·L⁻³ | Density (kg/m³) |

#### Geometry

| Pattern | Dimension | Description |
|---------|-----------|-------------|
| `distance`, `length`, `x`, `y`, `z` | L | Length (m) |
| `radius`, `r` | L | Radius (m) |
| `area` | L² | Area (m²) |
| `volume` | L³ | Volume (m³) |
| `angle`, `theta`, `phi` | 1 | Angle (rad, dimensionless) |

#### Time

| Pattern | Dimension | Description |
|---------|-----------|-------------|
| `time`, `t`, `duration` | T | Time (s) |
| `period` | T | Period (s) |
| `frequency`, `freq`, `f` | T⁻¹ | Frequency (Hz) |

#### Electromagnetics

| Pattern | Dimension | Description |
|---------|-----------|-------------|
| `current`, `i` | I | Current (A) |
| `voltage`, `potential`, `v` | M·L²·T⁻³·I⁻¹ | Voltage (V) |
| `resistance`, `r` | M·L²·T⁻³·I⁻² | Resistance (Ω) |
| `charge`, `q` | I·T | Charge (C) |
| `capacitance` | M⁻¹·L⁻²·T⁴·I² | Capacitance (F) |
| `inductance` | M·L²·T⁻²·I⁻² | Inductance (H) |

#### Thermodynamics

| Pattern | Dimension | Description |
|---------|-----------|-------------|
| `temperature`, `temp`, `t` | Θ | Temperature (K) |
| `heat`, `q` | M·L²·T⁻² | Heat energy (J) |
| `entropy` | M·L²·T⁻²·Θ⁻¹ | Entropy (J/K) |
| `specific_heat` | L²·T⁻²·Θ⁻¹ | Specific heat (J/(kg·K)) |

### Unit Suffixes

Variables with unit suffixes are recognized with 0.7 confidence:

```python
# SI base units
distance_m      # L (meters)
mass_kg         # M (kilograms)
time_s          # T (seconds)
current_a       # I (amperes)
temp_k          # Θ (kelvin)

# Derived units
force_n         # M·L·T⁻² (newtons)
energy_j        # M·L²·T⁻² (joules)
power_w         # M·L²·T⁻³ (watts)
pressure_pa     # M·L⁻¹·T⁻² (pascals)

# Compound units
velocity_m_per_s   # L·T⁻¹
velocity_mps       # L·T⁻¹
density_kg_per_m3  # M·L⁻³
```

### Prefixes and Modifiers

These prefixes are recognized and stripped (0.7 confidence):

```python
initial_velocity    # L·T⁻¹ (prefix: initial_)
final_position      # L (prefix: final_)
max_force           # M·L·T⁻² (prefix: max_)
min_energy          # M·L²·T⁻² (prefix: min_)
avg_power           # M·L²·T⁻³ (prefix: avg_)
total_momentum      # M·L·T⁻¹ (prefix: total_)
delta_velocity      # L·T⁻¹ (prefix: delta_)
```

---

## Equation Pattern Database

### Overview

Dimtensor includes a database of 50+ common physics equations spanning multiple domains. Each equation stores:

- Variable names and their dimensions
- Formula representation
- Physics domain (mechanics, electromagnetics, etc.)
- Tags for categorization

### Available Domains

The database covers these physics domains:

```python
from dimtensor.inference import DOMAINS

print(DOMAINS)
# ['mechanics', 'kinematics', 'waves', 'electromagnetics',
#  'thermodynamics', 'relativity', 'quantum', 'fluids']
```

### Searching by Domain

```python
from dimtensor.inference import get_equations_by_domain

# Get all mechanics equations
mechanics = get_equations_by_domain("mechanics")
for eq in mechanics[:5]:
    print(f"{eq.name}: {eq.formula}")

# Output:
# Newton's Second Law: F = ma
# Kinetic Energy: KE = ½mv²
# Gravitational Potential Energy: PE = mgh
# Momentum: p = mv
# Impulse: J = FΔt = Δp
```

### Searching by Tag

```python
from dimtensor.inference import get_equations_by_tag

# Get all equations related to energy
energy_equations = get_equations_by_tag("energy")
for eq in energy_equations:
    print(f"{eq.name}: {eq.formula}")

# Output:
# Kinetic Energy: KE = ½mv²
# Gravitational Potential Energy: PE = mgh
# Work: W = Fd cos(θ)
# Capacitor Energy: E = ½CV²
```

### Finding Equations with Specific Variables

```python
from dimtensor.inference import find_equations_with_variable

# Find equations using 'v' (velocity)
velocity_equations = find_equations_with_variable("v")
for eq in velocity_equations[:3]:
    print(f"{eq.name}: {eq.formula}")

# Output:
# Kinetic Energy: KE = ½mv²
# Velocity: v = Δx/Δt
# SUVAT: Velocity: v = v₀ + at
```

### Exploring the Database

```python
from dimtensor.inference import EQUATION_DATABASE

# See all available equations
print(f"Total equations: {len(EQUATION_DATABASE)}")

# Examine a specific equation
eq = EQUATION_DATABASE[0]  # Newton's Second Law
print(f"Name: {eq.name}")
print(f"Formula: {eq.formula}")
print(f"Domain: {eq.domain}")
print(f"Tags: {eq.tags}")
print(f"Variables:")
for var, dim in eq.variables.items():
    print(f"  {var}: {dim}")

# Output:
# Name: Newton's Second Law
# Formula: F = ma
# Domain: mechanics
# Tags: ['newton', 'force', 'fundamental']
# Variables:
#   F: M·L·T⁻²
#   m: M
#   a: L·T⁻²
```

### Using Equations for Suggestions

```python
from dimtensor.inference import suggest_dimension_from_equations

# Suggest dimensions for variable 'F' based on equation database
suggestions = suggest_dimension_from_equations("F")
for dim, eq_name, confidence in suggestions[:3]:
    print(f"{eq_name}: {dim} (confidence={confidence:.1f})")

# Output:
# Newton's Second Law: M·L·T⁻² (confidence=0.9)
# Impulse: M·L·T⁻¹ (confidence=0.9)
# Work: M·L²·T⁻² (confidence=0.9)
```

### Example Equations by Domain

#### Mechanics

- Newton's Second Law: **F = ma**
- Kinetic Energy: **KE = ½mv²**
- Momentum: **p = mv**
- Gravitational Force: **F = Gm₁m₂/r²**
- Pressure: **P = F/A**

#### Kinematics

- Velocity: **v = Δx/Δt**
- Acceleration: **a = Δv/Δt**
- SUVAT Position: **x = x₀ + v₀t + ½at²**
- SUVAT Velocity: **v = v₀ + at**

#### Electromagnetics

- Ohm's Law: **V = IR**
- Electric Power: **P = IV = I²R = V²/R**
- Coulomb's Law: **F = kq₁q₂/r²**
- Capacitance: **C = Q/V**

#### Thermodynamics

- Ideal Gas Law: **PV = nRT**
- Heat Transfer: **Q = mcΔT**
- First Law: **ΔU = Q - W**
- Stefan-Boltzmann: **P = σAT⁴**

#### Waves

- Wave Equation: **v = fλ**
- Period-Frequency: **T = 1/f**
- Pendulum Period: **T = 2π√(L/g)**

#### Relativity

- Mass-Energy: **E = mc²**

#### Quantum

- Photon Energy: **E = hf**
- de Broglie Wavelength: **λ = h/p**

---

## Unit Inference from Equations

### Overview

The constraint solver can infer unknown units from equation structures when some variables have known units.

### Basic API

```python
from dimtensor.inference import infer_units
from dimtensor.core.units import kg, m, s

# Infer force from F = m * a
result = infer_units(
    "F = m * a",
    known_units={"m": kg, "a": m / s**2}
)

print(result['inferred']['F'])      # N (kg·m/s²)
print(result['is_consistent'])      # True
print(result['confidence'])         # 1.0
print(result['errors'])             # []
```

### How It Works

The solver:

1. Parses the equation into an expression tree
2. Builds dimensional constraints for each operation
3. Propagates known dimensions through the constraint system
4. Detects inconsistencies and conflicts
5. Returns inferred units with confidence scores

### Example: Inferring Energy

```python
from dimtensor.inference import infer_units
from dimtensor.core.units import kg, m, s

# E = 0.5 * m * v^2
result = infer_units(
    "E = 0.5 * m * v**2",
    known_units={
        "m": kg,
        "v": m / s
    }
)

print(result['inferred']['E'])      # J (kg·m²/s²)
print(result['is_consistent'])      # True
```

### Example: Detecting Inconsistencies

```python
from dimtensor.inference import infer_units
from dimtensor.core.units import kg, m, s

# Dimensionally inconsistent equation
result = infer_units(
    "F = m + a",  # Can't add mass + acceleration!
    known_units={"m": kg, "a": m / s**2}
)

print(result['is_consistent'])  # False
print(result['errors'])
# ['Conflicting dimensions for _temp_0: M vs M·L·T⁻²']
```

### Example: Inferring Multiple Variables

```python
from dimtensor.inference import infer_units
from dimtensor.core.units import m, s

# Given distance and time, infer velocity
result = infer_units(
    "d = v * t",
    known_units={"d": m, "t": s}
)

print(result['inferred']['v'])  # m/s
```

### Example: Complex Equations

```python
from dimtensor.inference import infer_units
from dimtensor.core.units import kg, m, s

# Kinematic equation with power
result = infer_units(
    "KE = 0.5 * m * v**2",
    known_units={"m": kg, "v": m / s}
)

print(result['inferred']['KE'])     # J
print(result['confidence'])         # 1.0

# Division
result = infer_units(
    "rho = m / V",
    known_units={"m": kg, "V": m**3}
)

print(result['inferred']['rho'])    # kg/m³
```

### Confidence Scores

| Confidence | Meaning |
|------------|---------|
| **1.0** | All unknowns fully determined, dimensionally consistent |
| **0.7** | Partial inference, some variables remain unknown |
| **0.0** | Dimensional inconsistency detected or parse error |

### Supported Operations

The solver handles:

- **Arithmetic**: `+`, `-`, `*`, `/`, `**`
- **Functions**: `sqrt(x)`
- **Constants**: Numeric values (treated as dimensionless)
- **Parentheses**: Full expression grouping

### Limitations

The solver has some limitations:

1. **Nonlinear equations**: May not fully solve systems with coupled unknowns
2. **Transcendental functions**: `sin()`, `cos()`, `exp()` require dimensionless arguments
3. **Multiple solutions**: Returns the first consistent solution found
4. **Variable exponents**: Best with numeric constant exponents

Example of limitation:

```python
# This works (constant exponent)
result = infer_units("E = m * c**2", {"m": kg, "c": m/s})
# E = J ✓

# This is harder (variable exponent)
result = infer_units("y = a * x**n", {"x": m})
# May not fully determine dimensions of y
```

---

## CLI Linting Tool

### Installation

The `dimtensor lint` command is included with the dimtensor package:

```bash
pip install dimtensor
```

### Basic Usage

Lint a single Python file:

```bash
dimtensor lint myfile.py
```

Lint all files in a directory:

```bash
dimtensor lint src/
```

Lint multiple paths:

```bash
dimtensor lint file1.py file2.py src/ tests/
```

### Severity Levels

Linting produces three severity levels:

| Severity | Code | Description |
|----------|------|-------------|
| **ERROR** | E000 | File not found, syntax errors |
| **WARNING** | W002 | Potential dimension mismatch |
| **INFO** | I001 | Suggestions in strict mode |

### Example: Catching Dimension Mismatches

Create a buggy file:

```python
# physics_sim.py
velocity = 10.0  # m/s
time = 5.0       # s
acceleration = 2.0  # m/s²

# BUG: Can't add velocity and time!
result = velocity + time

# BUG: Can't add velocity and acceleration!
total = velocity + acceleration
```

Run the linter:

```bash
$ dimtensor lint physics_sim.py

physics_sim.py:6:10: W002 Potential dimension mismatch: L·T⁻¹ + T
  result = velocity + time
  Suggestion: Cannot add/subtract L·T⁻¹ and T. Check your units.

physics_sim.py:9:9: W002 Potential dimension mismatch: L·T⁻¹ + L·T⁻²
  total = velocity + acceleration
  Suggestion: Cannot add/subtract L·T⁻¹ and L·T⁻². Check your units.
```

### Strict Mode

Use `--strict` to get suggestions for all inferred variables:

```bash
$ dimtensor lint physics_sim.py --strict

physics_sim.py:2:0: I001 Variable 'velocity' inferred as L·T⁻¹
  velocity = 10.0
  Suggestion: Consider adding explicit unit: DimArray(..., units.m/s)

physics_sim.py:3:0: I001 Variable 'time' inferred as T
  time = 5.0
  Suggestion: Consider adding explicit unit: DimArray(..., units.s)

physics_sim.py:4:0: I001 Variable 'acceleration' inferred as L·T⁻²
  acceleration = 2.0
  Suggestion: Consider adding explicit unit: DimArray(..., units.m/s**2)

physics_sim.py:7:10: W002 Potential dimension mismatch: L·T⁻¹ + T
  ...
```

### Output Formats

#### Text Format (Default)

```bash
$ dimtensor lint myfile.py
myfile.py:10:5: W002 Potential dimension mismatch: L·T⁻¹ + T
  result = velocity + time
  Suggestion: Cannot add/subtract L·T⁻¹ and T. Check your units.
```

#### JSON Format

```bash
$ dimtensor lint myfile.py --format json
[
  {
    "file": "myfile.py",
    "line": 10,
    "column": 5,
    "severity": "warning",
    "code": "002",
    "message": "Potential dimension mismatch: L·T⁻¹ + T",
    "context": "result = velocity + time",
    "suggestion": "Cannot add/subtract L·T⁻¹ and T. Check your units."
  }
]
```

### Command-Line Options

```bash
dimtensor lint --help
```

| Option | Description |
|--------|-------------|
| `paths` | Files or directories to lint (required) |
| `--format text\|json` | Output format (default: text) |
| `--strict` | Report all potential issues including suggestions |
| `--no-recursive` | Don't search subdirectories |

### Examples

```bash
# Lint current directory recursively
dimtensor lint .

# Lint with JSON output for CI integration
dimtensor lint src/ --format json > lint-results.json

# Strict mode for code review
dimtensor lint PR_changes.py --strict

# Lint specific directory without recursion
dimtensor lint src/ --no-recursive
```

### Exit Codes

The linter returns:

- **0**: No warnings or errors found
- **1**: Warnings or errors detected

This makes it suitable for CI/CD pipelines:

```bash
# Fail CI build if dimensional issues found
dimtensor lint src/ || exit 1
```

---

## IDE Integration

### Pre-commit Hooks

Add dimensional linting to your pre-commit hooks:

#### Using pre-commit framework

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: dimtensor-lint
        name: Dimensional Linting
        entry: python -m dimtensor.cli.lint
        language: system
        types: [python]
        pass_filenames: true
```

Install and run:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

#### Manual Git hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Run dimensional linting before commit

echo "Running dimensional linting..."
python -m dimtensor.cli.lint $(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')

if [ $? -ne 0 ]; then
    echo "Dimensional linting failed. Fix issues or use 'git commit --no-verify' to skip."
    exit 1
fi
```

Make it executable:

```bash
chmod +x .git/hooks/pre-commit
```

### CI/CD Integration

#### GitHub Actions

Add to `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install dimtensor
      - name: Dimensional linting
        run: |
          dimtensor lint src/ --format json > lint-results.json
          cat lint-results.json
          dimtensor lint src/  # Fail on warnings
      - name: Upload lint results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: lint-results
          path: lint-results.json
```

#### GitLab CI

Add to `.gitlab-ci.yml`:

```yaml
lint:
  stage: test
  image: python:3.10
  script:
    - pip install dimtensor
    - dimtensor lint src/ --format json | tee lint-results.json
    - dimtensor lint src/
  artifacts:
    reports:
      codequality: lint-results.json
    when: always
```

#### Jenkins

Add to `Jenkinsfile`:

```groovy
stage('Dimensional Linting') {
    steps {
        sh 'pip install dimtensor'
        sh 'dimtensor lint src/ --format json > lint-results.json'
        sh 'dimtensor lint src/'
    }
    post {
        always {
            archiveArtifacts artifacts: 'lint-results.json', allowEmptyArchive: true
        }
    }
}
```

### VS Code Integration

While there's no official extension yet, you can integrate linting into VS Code:

#### Task Configuration

Create `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Dimensional Lint",
      "type": "shell",
      "command": "dimtensor lint ${file}",
      "group": "test",
      "presentation": {
        "reveal": "always",
        "panel": "new"
      }
    }
  ]
}
```

Run with: **Terminal > Run Task > Dimensional Lint**

#### Keyboard Shortcut

Add to `.vscode/keybindings.json`:

```json
[
  {
    "key": "ctrl+shift+l",
    "command": "workbench.action.tasks.runTask",
    "args": "Dimensional Lint"
  }
]
```

### PyCharm Integration

#### External Tool

1. Go to **Settings > Tools > External Tools**
2. Click **+** to add a new tool
3. Configure:
   - Name: `Dimensional Lint`
   - Program: `python`
   - Arguments: `-m dimtensor.cli.lint $FilePath$`
   - Working directory: `$ProjectFileDir$`
4. Click **OK**

Run with: **Tools > External Tools > Dimensional Lint**

---

## Best Practices

### When to Use Inference

| Scenario | Recommendation |
|----------|----------------|
| **Prototyping** | ✅ Use inference to explore and experiment |
| **Learning** | ✅ Use inference to understand dimensional relationships |
| **Code review** | ✅ Use linting to catch errors before merging |
| **Production code** | ⚠️ Prefer explicit units for clarity |
| **Public APIs** | ❌ Always use explicit units, never rely on inference |

### Naming Conventions for Maximum Accuracy

Follow these conventions to improve inference accuracy:

#### DO: Use descriptive physics names

```python
# Good - clear intention
velocity_initial = ...
acceleration_gravity = ...
force_applied = ...
```

#### DON'T: Use ambiguous abbreviations

```python
# Bad - unclear
v = ...  # velocity? voltage? volume?
f = ...  # force? frequency?
t = ...  # time? temperature?
```

#### DO: Include units in variable names when needed

```python
# Good - explicit about units
distance_km = ...
energy_ev = ...
time_ns = ...
```

#### DO: Use standard prefixes

```python
# Good - recognized modifiers
initial_velocity = ...
max_pressure = ...
delta_temperature = ...
```

#### DON'T: Mix conventions

```python
# Bad - inconsistent
vel = ...           # abbreviation
acceleration = ...  # full name
f_net = ...         # hybrid
```

### Combining Inference with Explicit Units

Use inference during development, explicit units in production:

```python
# Development: Quick prototyping
def simulate_motion(velocity, time, acceleration):
    """Quick physics simulation (units inferred from names)."""
    return velocity * time + 0.5 * acceleration * time**2

# Production: Explicit and documented
def simulate_motion(
    velocity: DimArray,  # m/s
    time: DimArray,      # s
    acceleration: DimArray  # m/s²
) -> DimArray:  # m
    """Simulate motion with explicit unit checking.

    Args:
        velocity: Initial velocity in m/s
        time: Time duration in seconds
        acceleration: Acceleration in m/s²

    Returns:
        Distance traveled in meters
    """
    assert velocity.unit.dimension == Dimension(length=1, time=-1)
    assert time.unit.dimension == Dimension(time=1)
    assert acceleration.unit.dimension == Dimension(length=1, time=-2)

    return velocity * time + 0.5 * acceleration * time**2
```

### Using Linting in Your Workflow

#### During Development

```bash
# Check your changes frequently
dimtensor lint myfile.py

# Get suggestions while coding
dimtensor lint myfile.py --strict
```

#### Before Committing

```bash
# Add to your pre-commit hook
dimtensor lint $(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')
```

#### In Code Review

```bash
# Check PR changes
git diff main...feature-branch --name-only | grep '\.py$' | xargs dimtensor lint
```

#### In CI/CD

```bash
# Fail build on dimensional issues
dimtensor lint src/ || exit 1
```

### Combining with Unit Tests

Use inference to catch bugs early, but still write comprehensive unit tests:

```python
import pytest
from dimtensor import DimArray, units
from dimtensor.errors import DimensionError

def test_force_calculation():
    """Test Newton's second law with explicit units."""
    mass = DimArray([2.0], units.kg)
    accel = DimArray([3.0], units.m / units.s**2)
    force = mass * accel

    assert force.to(units.N).data[0] == pytest.approx(6.0)

def test_dimension_mismatch_caught():
    """Test that dimensional errors are caught at runtime."""
    velocity = DimArray([10.0], units.m / units.s)
    time = DimArray([5.0], units.s)

    with pytest.raises(DimensionError):
        result = velocity + time  # Can't add velocity + time!
```

### Performance Considerations

Inference has minimal runtime overhead:

- **Variable name inference**: Fast lookups in dictionaries
- **Equation matching**: Only runs when explicitly called
- **Static linting**: No runtime cost (runs on source code)

For performance-critical code:

```python
# Inference has no runtime cost here
def physics_loop(n_iterations):
    velocity = DimArray([10.0], units.m / units.s)
    time = DimArray([0.0], units.s)
    dt = DimArray([0.01], units.s)

    for i in range(n_iterations):
        time += dt
        # No inference overhead in this loop
        distance = velocity * time
```

### Documentation

Document when you rely on inference:

```python
def analyze_trajectory(
    initial_velocity,  # Inferred as L·T⁻¹
    acceleration,      # Inferred as L·T⁻²
    duration           # Inferred as T
):
    """Analyze projectile trajectory.

    Args:
        initial_velocity: Initial velocity (m/s, inferred from name)
        acceleration: Acceleration (m/s², inferred from name)
        duration: Time duration (s, inferred from name)

    Note:
        This function relies on variable name inference.
        Run 'dimtensor lint' to verify dimensional correctness.

    Returns:
        Final position in meters.
    """
    return initial_velocity * duration + 0.5 * acceleration * duration**2
```

---

## Summary

Dimtensor's inference system provides powerful tools for dimensional correctness:

- **Variable name patterns** suggest dimensions from common physics names
- **Equation database** recognizes 50+ physics equations across multiple domains
- **Constraint solver** infers unknown units from equation structure
- **CLI linting** detects dimensional bugs before runtime
- **CI/CD integration** enforces dimensional correctness in your pipeline

Use inference to:
- Catch bugs early during development
- Learn physics through automated suggestions
- Enforce correctness in code reviews
- Reduce boilerplate when prototyping

For production code, combine inference (as a safety check) with explicit units (for clarity and documentation).

---

## See Also

- [Getting Started](getting-started.md) - Basic dimtensor usage
- [Examples](examples.md) - Real-world physics calculations
- [Operations](operations.md) - Mathematical operations with units
- [API Reference](../api/inference.md) - Detailed function signatures

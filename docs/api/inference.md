# Dimensional Inference

Automatic inference of physical dimensions from equations, variable names, and context.

## Overview

The `dimtensor.inference` module provides tools to automatically infer physical dimensions:

- **Equation Solver**: Parse physics equations and infer unknown units
- **Heuristics**: Infer dimensions from variable names (e.g., "velocity" → L/T)
- **Constraint Solver**: Propagate dimensional constraints through expressions
- **Parser**: Parse arithmetic equations into expression trees

```python
from dimtensor.inference import infer_units, infer_dimension
from dimtensor import units

# Infer from equation
result = infer_units(
    "F = m * a",
    known_units={"m": units.kg, "a": units.m / units.s**2}
)
print(result['inferred']['F'])  # N (newton)

# Infer from variable name
dim, confidence = infer_dimension("velocity")
print(dim)  # Dimension(length=1, time=-1)
print(confidence)  # 0.9
```

## Equation-Based Inference

### infer_units

Infer units for unknown variables in an equation using dimensional analysis.

```python
from dimtensor.inference import infer_units
from dimtensor import units

# Newton's second law: F = m * a
result = infer_units(
    equation="F = m * a",
    known_units={
        "m": units.kg,
        "a": units.m / units.s**2
    }
)

print(result['is_consistent'])  # True
print(result['inferred']['F'])  # N (newton)
print(result['confidence'])     # 1.0 (high confidence)

# Kinetic energy: E = 0.5 * m * v**2
result = infer_units(
    equation="E = 0.5 * m * v**2",
    known_units={
        "m": units.kg,
        "v": units.m / units.s
    }
)
print(result['inferred']['E'])  # J (joule)
```

::: dimtensor.inference.infer_units
    options:
      members:
        - equation
        - known_units
        - database

## Name-Based Inference

### infer_dimension

Infer physical dimension from a variable name using heuristics.

```python
from dimtensor.inference import infer_dimension

# Common physics variables
dim, conf = infer_dimension("velocity")
# Dimension(length=1, time=-1), confidence=0.9

dim, conf = infer_dimension("acceleration")
# Dimension(length=1, time=-2), confidence=0.9

dim, conf = infer_dimension("force")
# Dimension(mass=1, length=1, time=-2), confidence=0.9

dim, conf = infer_dimension("energy")
# Dimension(mass=1, length=2, time=-2), confidence=0.9

# Abbreviated names
dim, conf = infer_dimension("v")  # velocity
dim, conf = infer_dimension("a")  # acceleration
dim, conf = infer_dimension("F")  # force

# Compound names
dim, conf = infer_dimension("angular_velocity")
# Dimension(time=-1), confidence=0.9

dim, conf = infer_dimension("kinetic_energy")
# Dimension(mass=1, length=2, time=-2), confidence=0.9
```

::: dimtensor.inference.infer_dimension

### InferenceResult

Result of dimensional inference with confidence scoring.

::: dimtensor.inference.InferenceResult
    options:
      members:
        - dimension
        - confidence
        - pattern
        - source

## Expression Parsing

### parse_equation

Parse an equation string into left and right expression trees.

```python
from dimtensor.inference import parse_equation

left, right = parse_equation("F = m * a")
# left: Variable('F')
# right: BinaryOp('*', Variable('m'), Variable('a'))

left, right = parse_equation("E = 0.5 * m * v**2")
# right: BinaryOp('*', BinaryOp('*', Constant(0.5), Variable('m')),
#                     BinaryOp('**', Variable('v'), Constant(2)))
```

::: dimtensor.inference.parse_equation

### parse_expression

Parse a single expression into an expression tree.

```python
from dimtensor.inference import parse_expression

expr = parse_expression("m * a")
# BinaryOp('*', Variable('m'), Variable('a'))

expr = parse_expression("0.5 * m * v**2")
# Nested BinaryOp nodes
```

::: dimtensor.inference.parse_expression

### Expression Node Types

Base classes for expression tree nodes:

::: dimtensor.inference.ExprNode

::: dimtensor.inference.Variable

::: dimtensor.inference.Constant

::: dimtensor.inference.BinaryOp

::: dimtensor.inference.UnaryOp

::: dimtensor.inference.FunctionCall

### get_variables

Extract all variable names from an expression.

```python
from dimtensor.inference import get_variables, parse_expression

expr = parse_expression("F = m * a + b")
variables = get_variables(expr)
print(variables)  # ['F', 'm', 'a', 'b']
```

::: dimtensor.inference.get_variables

## Constraint Solving

### Constraint Types

Dimensional constraints for propagation through expressions:

::: dimtensor.inference.Constraint

::: dimtensor.inference.EqualityConstraint

::: dimtensor.inference.MultiplicationConstraint

::: dimtensor.inference.DivisionConstraint

::: dimtensor.inference.PowerConstraint

::: dimtensor.inference.DimensionlessConstraint

## Built-in Patterns

The inference system recognizes common physics variable names:

### Mechanics

| Pattern | Dimension | Examples |
|---------|-----------|----------|
| velocity, speed | L/T | v, vel, speed |
| acceleration | L/T^2 | a, accel |
| force | MLT^-2 | F, force |
| momentum | MLT^-1 | p, momentum |
| energy | ML^2T^-2 | E, energy, KE, PE |
| power | ML^2T^-3 | P, power |
| pressure | ML^-1T^-2 | p, pressure |

### Geometry

| Pattern | Dimension | Examples |
|---------|-----------|----------|
| position, distance | L | x, y, z, r, position |
| area | L^2 | A, area |
| volume | L^3 | V, volume |
| angle | 1 (dimensionless) | theta, phi, angle |
| angular_velocity | T^-1 | omega, angular_vel |

### Thermodynamics

| Pattern | Dimension | Examples |
|---------|-----------|----------|
| temperature | Θ | T, temp, temperature |
| heat | ML^2T^-2 | Q, heat |
| entropy | ML^2T^-2Θ^-1 | S, entropy |

### Electromagnetism

| Pattern | Dimension | Examples |
|---------|-----------|----------|
| charge | IT | q, Q, charge |
| current | I | I, current |
| voltage | ML^2T^-3I^-1 | V, voltage |
| resistance | ML^2T^-3I^-2 | R, resistance |
| capacitance | M^-1L^-2T^4I^2 | C, capacitance |
| magnetic_field | MT^-2I^-1 | B, field |

## Examples

### Inferring Units from Equations

```python
from dimtensor.inference import infer_units
from dimtensor import units

# Example 1: Newton's law of gravitation
result = infer_units(
    equation="F = G * m1 * m2 / r**2",
    known_units={
        "F": units.N,
        "m1": units.kg,
        "m2": units.kg,
        "r": units.m
    }
)
print(result['inferred']['G'])  # Gravitational constant
# Unit: m^3/(kg*s^2)

# Example 2: Ideal gas law: PV = nRT
result = infer_units(
    equation="P * V = n * R * T",
    known_units={
        "P": units.Pa,
        "V": units.m**3,
        "n": units.mol,
        "T": units.K
    }
)
print(result['inferred']['R'])  # Gas constant
# Unit: J/(mol*K)

# Example 3: Check dimensional consistency
result = infer_units(
    equation="E = m * c**2",
    known_units={
        "E": units.J,
        "m": units.kg,
        "c": units.m / units.s
    }
)
print(result['is_consistent'])  # True
print(result['confidence'])     # 1.0
```

### Variable Name Inference

```python
from dimtensor.inference import infer_dimension
from dimtensor import DimArray, Dimension

def create_array_from_name(name: str, data):
    """Create DimArray inferring units from variable name."""
    dim, confidence = infer_dimension(name)

    if confidence < 0.5:
        print(f"Warning: Low confidence ({confidence}) for '{name}'")

    # Map dimension to a default unit
    unit = dimension_to_default_unit(dim)
    return DimArray(data, unit)

# Usage
velocity = create_array_from_name("velocity", [10, 20, 30])
# Infers dimension L/T, uses m/s

acceleration = create_array_from_name("acceleration", [1, 2, 3])
# Infers dimension L/T^2, uses m/s^2
```

### Smart DataFrame Loading

```python
from dimtensor.inference import infer_dimension
from dimtensor import DimArray, units
import pandas as pd

def load_physics_dataframe(df: pd.DataFrame) -> dict:
    """Load DataFrame columns as DimArrays with inferred units."""
    result = {}

    for column in df.columns:
        data = df[column].values

        # Infer dimension from column name
        dim, confidence = infer_dimension(column)

        if confidence > 0.7:
            # High confidence: use inferred dimension
            unit = dimension_to_default_unit(dim)
            result[column] = DimArray(data, unit)
        else:
            # Low confidence: ask user or use dimensionless
            print(f"Cannot infer units for '{column}' (confidence={confidence})")
            result[column] = DimArray(data, units.dimensionless)

    return result

# Usage
df = pd.DataFrame({
    'time': [0, 1, 2, 3],
    'velocity': [0, 10, 20, 30],
    'acceleration': [10, 10, 10, 10]
})

arrays = load_physics_dataframe(df)
# time: DimArray with units.s
# velocity: DimArray with units.m / units.s
# acceleration: DimArray with units.m / units.s**2
```

### Interactive Equation Solver

```python
from dimtensor.inference import infer_units
from dimtensor import units

def solve_for_unknown(equation: str, **known_vars):
    """Interactive equation solver with unit inference.

    Args:
        equation: Physics equation string
        **known_vars: Known variables as keyword arguments

    Example:
        >>> solve_for_unknown("F = m * a", m=units.kg, a=units.m/units.s**2)
    """
    result = infer_units(equation, known_units=known_vars)

    print(f"Equation: {equation}")
    print(f"Dimensionally consistent: {result['is_consistent']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print()

    if result['inferred']:
        print("Inferred units:")
        for var, unit in result['inferred'].items():
            print(f"  {var}: {unit}")
    else:
        print("No unknowns to infer")

    if result['errors']:
        print("\nErrors:")
        for error in result['errors']:
            print(f"  - {error}")

# Usage
solve_for_unknown("E = m * c**2", m=units.kg, c=units.m/units.s)
# Infers: E has units J (joule)

solve_for_unknown("F = m * a", m=units.kg, a=units.m/units.s**2)
# Infers: F has units N (newton)
```

### Custom Pattern Extension

```python
from dimtensor.inference import VARIABLE_PATTERNS, infer_dimension
from dimtensor import Dimension

# Add custom patterns for your domain
VARIABLE_PATTERNS.update({
    "reynolds_number": Dimension(),  # Dimensionless
    "mach_number": Dimension(),      # Dimensionless
    "viscosity": Dimension(mass=1, length=-1, time=-1),
    "diffusivity": Dimension(length=2, time=-1),
})

# Now they'll be recognized
dim, conf = infer_dimension("reynolds_number")
# Dimension(), confidence=0.9

dim, conf = infer_dimension("viscosity")
# Dimension(mass=1, length=-1, time=-1), confidence=0.9
```

## Advanced Usage

### Equation Database Integration

```python
from dimtensor.inference import infer_units
from dimtensor.equations import EquationDatabase

# Create equation database
db = EquationDatabase()
db.add("newton_second_law", "F = m * a",
       {"F": "force", "m": "mass", "a": "acceleration"})
db.add("kinetic_energy", "E = 0.5 * m * v**2",
       {"E": "energy", "m": "mass", "v": "velocity"})

# Use database for inference
result = infer_units(
    "F = m * a",
    known_units={"m": units.kg},
    database=db
)
# Matches known equation and provides suggestions
```

### Multi-Equation Systems

```python
from dimtensor.inference import infer_units

# System of equations
equations = [
    "F = m * a",
    "a = v / t",
    "p = m * v"
]

known = {
    "m": units.kg,
    "t": units.s,
    "F": units.N
}

# Solve system iteratively
inferred = {}
for eq in equations:
    result = infer_units(eq, {**known, **inferred})
    inferred.update(result['inferred'])

print(inferred)
# All variables inferred consistently
```

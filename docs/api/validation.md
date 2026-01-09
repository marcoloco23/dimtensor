# Validation

Value constraints and conservation law checking for dimensional arrays.

## Overview

The `dimtensor.validation` module provides tools to enforce physical constraints and conservation laws on dimensional arrays:

- **Constraints**: Enforce value requirements (positive, bounded, finite, etc.)
- **Conservation**: Check and enforce conservation of energy, momentum, mass, etc.

```python
from dimtensor import DimArray, units
from dimtensor.validation import Positive, Bounded, check_conservation

# Create with constraints
mass = DimArray([1.0, 2.0, 3.0], units.kg, constraints=[Positive()])
probability = DimArray([0.3, 0.5, 0.2], units.dimensionless,
                       constraints=[Bounded(0, 1)])

# Validate
mass.validate()  # OK

# Check conservation laws
E_initial = DimArray([100.0], units.J)
E_final = DimArray([99.5], units.J)
is_conserved = check_conservation(E_initial, E_final, rtol=1e-2)  # True
```

## Constraints

### Constraint Base Class

::: dimtensor.validation.Constraint
    options:
      members:
        - name
        - description
        - check
        - is_satisfied
        - validate

### Positive

Requires all values > 0. Use for quantities that must be strictly positive like mass, temperature (Kelvin), distances.

```python
from dimtensor.validation import Positive

mass = DimArray([1.0, 2.0], units.kg, constraints=[Positive()])
```

::: dimtensor.validation.Positive

### NonNegative

Requires all values >= 0. Use for counts, magnitudes, or quantities that can be zero.

```python
from dimtensor.validation import NonNegative

count = DimArray([0, 5, 10], units.dimensionless, constraints=[NonNegative()])
```

::: dimtensor.validation.NonNegative

### NonZero

Requires all values != 0. Use for divisors or quantities that cannot be zero.

::: dimtensor.validation.NonZero

### Bounded

Requires all values in [min, max]. Use for probabilities, efficiencies, normalized quantities.

```python
from dimtensor.validation import Bounded

# Probability must be in [0, 1]
prob = DimArray([0.3, 0.5, 0.2], units.dimensionless,
                constraints=[Bounded(0, 1)])

# Angle in degrees [0, 360]
angle = DimArray([45, 90, 180], units.deg,
                 constraints=[Bounded(0, 360)])
```

::: dimtensor.validation.Bounded
    options:
      members:
        - __init__
        - check
        - min_val
        - max_val

### Finite

Requires all values to be finite (no inf or NaN). Use for physical measurements and most numerical computations.

```python
from dimtensor.validation import Finite

data = DimArray([1.0, 2.0, 3.0], units.m, constraints=[Finite()])
```

::: dimtensor.validation.Finite

### NotNaN

Requires no NaN values. Allows infinity but rejects NaN.

::: dimtensor.validation.NotNaN

## Utility Functions

### validate_all

Validate data against multiple constraints.

```python
from dimtensor.validation import validate_all, Positive, Finite

constraints = [Positive(), Finite()]
validate_all(data, constraints)  # Raises ConstraintError if any fail
```

::: dimtensor.validation.validate_all

### is_all_satisfied

Check if data satisfies all constraints without raising errors.

```python
from dimtensor.validation import is_all_satisfied

if is_all_satisfied(data, constraints):
    print("All constraints satisfied")
```

::: dimtensor.validation.is_all_satisfied

## Conservation Laws

### check_conservation

Check if a quantity is conserved within tolerance.

```python
from dimtensor.validation import check_conservation

E_initial = DimArray([100.0], units.J)
E_final = DimArray([99.9], units.J)

is_conserved = check_conservation(E_initial, E_final, rtol=1e-2)
# True: 0.1% change is within 1% tolerance
```

::: dimtensor.validation.check_conservation

### ConservationChecker

Class-based interface for checking multiple conservation laws.

```python
from dimtensor.validation import ConservationChecker

checker = ConservationChecker(rtol=1e-6, atol=1e-10)
checker.add('energy', E_initial, E_final)
checker.add('momentum', p_initial, p_final)

violations = checker.check_all()
if violations:
    print(f"Conservation violated: {violations}")
```

::: dimtensor.validation.ConservationChecker
    options:
      members:
        - __init__
        - add
        - check
        - check_all
        - report

## Examples

### Using Constraints in Physics

```python
from dimtensor import DimArray, units
from dimtensor.validation import Positive, Bounded, Finite

# Mass must be positive
mass = DimArray([1.0, 2.0, 3.0], units.kg, constraints=[Positive()])

# Velocity can be any finite value
velocity = DimArray([10.0, -5.0, 0.0], units.m / units.s,
                    constraints=[Finite()])

# Efficiency between 0 and 1
efficiency = DimArray([0.85, 0.90, 0.78], units.dimensionless,
                      constraints=[Bounded(0, 1), Positive()])

# Validate all
mass.validate()
velocity.validate()
efficiency.validate()

# This would raise ConstraintError:
# bad_mass = DimArray([-1.0], units.kg, constraints=[Positive()])
```

### Checking Energy Conservation

```python
from dimtensor import DimArray, units
from dimtensor.validation import check_conservation, ConservationChecker

# Simulate a physics system
def simulate_collision(m1, v1, m2, v2):
    """Elastic collision."""
    # ... physics computation ...
    return v1_final, v2_final

# Initial conditions
m1 = DimArray([1.0], units.kg)
v1 = DimArray([10.0], units.m / units.s)
m2 = DimArray([2.0], units.kg)
v2 = DimArray([0.0], units.m / units.s)

# Compute initial energy and momentum
E_initial = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
p_initial = m1 * v1 + m2 * v2

# Run simulation
v1_f, v2_f = simulate_collision(m1, v1, m2, v2)

# Compute final quantities
E_final = 0.5 * m1 * v1_f**2 + 0.5 * m2 * v2_f**2
p_final = m1 * v1_f + m2 * v2_f

# Check conservation
checker = ConservationChecker(rtol=1e-6)
checker.add('energy', E_initial, E_final)
checker.add('momentum', p_initial, p_final)

violations = checker.check_all()
if not violations:
    print("Conservation laws satisfied!")
else:
    print(f"Violations: {violations}")
```

### Custom Constraints

```python
from dimtensor.validation import Constraint
import numpy as np

class EvenOnly(Constraint):
    """Constraint requiring all values to be even integers."""

    @property
    def name(self) -> str:
        return "EvenOnly"

    @property
    def description(self) -> str:
        return "values must be even integers"

    def check(self, data):
        return (data % 2 == 0) & (data == data.astype(int))

# Use custom constraint
counts = DimArray([2, 4, 6, 8], units.dimensionless,
                  constraints=[EvenOnly()])
counts.validate()  # OK

# This would fail:
# bad_counts = DimArray([1, 3, 5], units.dimensionless,
#                       constraints=[EvenOnly()])
```

### Constraints in Machine Learning

```python
from dimtensor import DimArray, units
from dimtensor.validation import Positive, Bounded

# Training data validation
temperature = DimArray(temp_data, units.K, constraints=[Positive()])
pressure = DimArray(pressure_data, units.Pa, constraints=[Positive()])

# Ensure data quality before training
temperature.validate()
pressure.validate()

# Neural network output constraints
predicted_prob = DimArray(nn_output, units.dimensionless,
                          constraints=[Bounded(0, 1)])

# Validate predictions
predicted_prob.validate()
```

## Error Handling

When constraints are violated, a `ConstraintError` is raised with details:

```python
from dimtensor import DimArray, units
from dimtensor.validation import Positive
from dimtensor.errors import ConstraintError

try:
    mass = DimArray([-1.0, 2.0], units.kg, constraints=[Positive()])
    mass.validate()
except ConstraintError as e:
    print(f"Constraint: {e.constraint}")
    print(f"Indices: {e.indices}")
    print(f"Message: {e}")
    # Constraint: Positive
    # Indices: [0]
    # Message: Constraint 'Positive' violated: values must be > 0.
    #          Invalid values at indices [0]: [-1.0]
```

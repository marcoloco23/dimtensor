# Validation and Constraints

## Introduction

Physical simulations require values that obey physical laws. Negative masses, probabilities outside [0, 1], and violations of conservation laws indicate bugs or numerical errors.

dimtensor provides **validation constraints** to catch these errors early:

- **Value constraints** enforce bounds (e.g., mass > 0, probability in [0,1])
- **Conservation tracking** monitors quantities that should remain constant (energy, momentum, mass)
- **Custom constraints** for domain-specific validation

Validation complements dimension checking: dimensions ensure *units* are correct, while constraints ensure *values* are physically valid.

## Built-in Constraints

dimtensor provides six built-in constraint types:

| Constraint | Condition | Use Cases |
|------------|-----------|-----------|
| `Positive` | value > 0 | Mass, temperature (K), time intervals |
| `NonNegative` | value >= 0 | Counts, magnitudes, absolute values |
| `NonZero` | value != 0 | Divisors, non-trivial quantities |
| `Bounded(min, max)` | min <= value <= max | Probability, efficiency, angles |
| `Finite` | No inf or NaN | Physical measurements, numerical results |
| `NotNaN` | No NaN | Computed results where NaN indicates error |

### Positive

Requires all values to be strictly positive (> 0). Use for quantities that physically cannot be zero or negative:

```python
from dimtensor import DimArray, units
from dimtensor.validation import Positive

# Mass must be positive
mass = DimArray([1.5, 2.3, 0.8], units.kg)
mass.validate([Positive()])  # OK

# This will raise ConstraintError
try:
    invalid_mass = DimArray([-1.0, 2.0], units.kg)
    invalid_mass.validate([Positive()])
except Exception as e:
    print(f"Error: {e}")
    # Error: Constraint 'Positive' violated: values must be > 0
```

**Common use cases:**
- Mass: `m > 0`
- Temperature in Kelvin: `T > 0`
- Time intervals: `dt > 0`
- Distances and lengths: `r > 0`

### NonNegative

Requires all values to be non-negative (>= 0). Use when zero is physically valid but negative values are not:

```python
from dimtensor.validation import NonNegative

# Particle count can be zero
count = DimArray([0, 5, 10], units.dimensionless)
count.validate([NonNegative()])  # OK - zero is allowed

# Magnitude of a vector
import numpy as np
velocity = DimArray([3.0, 4.0], units.m / units.s)
speed = DimArray([np.linalg.norm(velocity.data)], units.m / units.s)
speed.validate([NonNegative()])
```

**Common use cases:**
- Counts and populations
- Magnitudes and absolute values
- Probabilities (combined with upper bound)

### NonZero

Requires all values to be non-zero (!= 0). Use when zero would cause mathematical or physical problems:

```python
from dimtensor.validation import NonZero

# Denominator must be non-zero
time = DimArray([1.0, 2.0, 3.0], units.s)
time.validate([NonZero()])  # Check before using as denominator
distance = DimArray([100.0, 200.0, 300.0], units.m)
velocity = distance / time  # Safe - time validated to be non-zero

# This will raise ConstraintError
try:
    bad_time = DimArray([1.0, 0.0, 3.0], units.s)
    bad_time.validate([NonZero()])
except Exception as e:
    print("Cannot use zero in denominator")
```

### Bounded

Requires all values to be within [min, max] (inclusive bounds). Use for quantities with natural or physical limits:

```python
from dimtensor.validation import Bounded

# Probability must be in [0, 1]
probability = DimArray([0.25, 0.5, 0.75], units.dimensionless)
probability.validate([Bounded(0, 1)])  # OK

# Efficiency must be in [0, 1]
efficiency = DimArray([0.85], units.dimensionless)
efficiency.validate([Bounded(0, 1)])

# Angle in degrees [0, 360)
angle = DimArray([45, 90, 180], units.deg)
angle.validate([Bounded(0, 360)])

# Temperature range for liquid water at 1 atm [273.15, 373.15] K
water_temp = DimArray([300, 350], units.K)
water_temp.validate([Bounded(273.15, 373.15)])
```

**Common use cases:**
- Probability: `Bounded(0, 1)`
- Efficiency: `Bounded(0, 1)`
- Angles: `Bounded(0, 2*pi)` or `Bounded(0, 360)`
- Material phase ranges (e.g., liquid water temperature)

### Finite

Requires all values to be finite (no inf or NaN). Use when infinite values indicate numerical errors:

```python
from dimtensor.validation import Finite

# Physical measurements must be finite
position = DimArray([1.0, 2.0, 3.0], units.m)
position.validate([Finite()])  # OK

# Catch numerical overflow
try:
    huge_value = DimArray([1e308, 1e309], units.J)
    huge_value.validate([Finite()])
except Exception as e:
    print("Numerical overflow detected")
```

### NotNaN

Requires no NaN values. Use when NaN indicates a computation error but infinity is acceptable:

```python
from dimtensor.validation import NotNaN

# Computed results should not be NaN
result = DimArray([1.0, 2.0, 3.0], units.m)
result.validate([NotNaN()])  # OK

# NaN indicates an error
import numpy as np
try:
    bad_result = DimArray([1.0, np.nan, 3.0], units.m)
    bad_result.validate([NotNaN()])
except Exception as e:
    print("NaN detected in computation")
```

## Using Constraints with DimArray

### Immediate Validation

Validate immediately after creating a DimArray to enforce constraints:

```python
from dimtensor import DimArray, units
from dimtensor.validation import Positive, Bounded

# Create and validate immediately
mass = DimArray([1.0, 2.0], units.kg)
mass.validate([Positive()])  # Raises if constraint violated

# Raises ConstraintError
try:
    bad_mass = DimArray([-1.0], units.kg)
    bad_mass.validate([Positive()])
except Exception as e:
    print(f"Validation failed: {e}")
```

### Manual Validation

Use `.validate()` to check constraints after operations:

```python
# Create array
mass = DimArray([1.0, 2.0], units.kg)

# Perform operations
mass = mass - DimArray([0.5, 0.5], units.kg)

# Validate after computation
mass.validate([Positive()])  # Check if still positive
```

### Multiple Constraints

Combine multiple constraints for complex validation:

```python
from dimtensor.validation import NonNegative, Bounded, Finite

# Probability: must be in [0,1] and finite
probability = DimArray([0.3, 0.7], units.dimensionless)
probability.validate([Bounded(0, 1), Finite()])  # All constraints checked

# Temperature: must be positive and finite
temperature = DimArray([300], units.K)
temperature.validate([Positive(), Finite()])
```

### Error Handling

Constraint violations raise `ConstraintError` with detailed information:

```python
from dimtensor.errors import ConstraintError

try:
    data = DimArray([-1.0, 2.0, -3.0], units.kg)
    data.validate([Positive()])
except ConstraintError as e:
    print(f"Constraint: {e.constraint}")  # 'Positive'
    print(f"Indices: {e.indices}")        # [0, 2] - where violations occurred
    print(f"Message: {e}")
    # Constraint 'Positive' violated: values must be > 0.
    # Invalid values at indices [0, 2]: [-1.0, -3.0]
```

## Conservation Tracking

Conservation laws (energy, momentum, mass) are fundamental in physics. The `ConservationTracker` monitors these quantities to detect numerical drift or bugs.

### Basic Usage

```python
from dimtensor import DimArray, units
from dimtensor.validation import ConservationTracker

# Track total energy in a simulation
tracker = ConservationTracker("Total Energy")

# Record initial energy
E_initial = DimArray([100.0], units.J)
tracker.record(E_initial)

# After computation
E_final = DimArray([99.99999], units.J)
tracker.record(E_final)

# Check conservation
if tracker.is_conserved(rtol=1e-6):
    print("Energy conserved within tolerance")
else:
    print(f"Energy drift: {tracker.drift():.2e}")
```

### Tracking Through Simulation

Record checkpoints throughout a simulation:

```python
import numpy as np

# Free fall simulation - mechanical energy should be conserved
tracker = ConservationTracker("Mechanical Energy")

# Initial conditions
mass = DimArray([1.0], units.kg)
height = DimArray([100.0], units.m)
velocity = DimArray([0.0], units.m / units.s)
g = DimArray([9.8], units.m / units.s**2)

# Initial energy: E = mgh + (1/2)mv^2
E_potential = mass * g * height
E_kinetic = 0.5 * mass * velocity**2
E_total = E_potential + E_kinetic
tracker.record(E_total)

# Simulate fall (simplified)
dt = DimArray([0.1], units.s)
for step in range(10):
    # Update position and velocity
    velocity = velocity + g * dt
    height = height - velocity * dt

    # Calculate energy at this step
    E_potential = mass * g * height
    E_kinetic = 0.5 * mass * velocity**2
    E_total = E_potential + E_kinetic
    tracker.record(E_total)

# Check conservation
print(f"Steps tracked: {len(tracker)}")
print(f"Energy conserved: {tracker.is_conserved(rtol=1e-6)}")
print(f"Maximum drift: {tracker.max_drift():.2e}")
```

### Conservation Tolerances

Choose appropriate tolerances based on your simulation:

```python
# Tight tolerance for analytical solutions
tracker.is_conserved(rtol=1e-12)  # 1 part in 10^12

# Moderate tolerance for numerical integration
tracker.is_conserved(rtol=1e-6)   # 1 part in 10^6

# Looser tolerance for approximate methods
tracker.is_conserved(rtol=1e-3)   # 0.1%

# Absolute tolerance for near-zero quantities
tracker.is_conserved(rtol=1e-6, atol=1e-10)
```

### Measuring Drift

Track how much a quantity has drifted from its initial value:

```python
# Drift: (current - initial) / |initial|
drift = tracker.drift()
print(f"Relative drift: {drift:.2e}")

# Maximum drift across all timesteps
max_drift = tracker.max_drift()
print(f"Maximum drift: {max_drift:.2e}")

# Reset for new simulation
tracker.reset()
```

## Custom Constraints

Create domain-specific constraints by extending the `Constraint` base class:

### Basic Custom Constraint

```python
from dimtensor.validation import Constraint
import numpy as np

class AboveAbsoluteZero(Constraint):
    """Temperature must be above absolute zero (0 K)."""

    @property
    def name(self) -> str:
        return "AboveAbsoluteZero"

    @property
    def description(self) -> str:
        return "temperature must be > 0 K"

    def check(self, data):
        return data > 0

# Use the custom constraint
temperature = DimArray([273.15, 300.0], units.K)
temperature.validate([AboveAbsoluteZero()])
```

### Parameterized Custom Constraint

```python
class MaterialTemperatureRange(Constraint):
    """Temperature must be within material's stable range."""

    def __init__(self, min_temp: float, max_temp: float, material: str):
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.material = material

    @property
    def name(self) -> str:
        return f"{self.material}TemperatureRange"

    @property
    def description(self) -> str:
        return f"temperature must be in [{self.min_temp}, {self.max_temp}] K for {self.material}"

    def check(self, data):
        return (data >= self.min_temp) & (data <= self.max_temp)

    def __repr__(self) -> str:
        return f"MaterialTemperatureRange({self.min_temp}, {self.max_temp}, '{self.material}')"

# Use for specific materials
iron_temp = DimArray([1800], units.K)

try:
    iron_temp.validate([MaterialTemperatureRange(1811, 3134, "Iron")])
except Exception as e:
    print(f"Iron melts at 1811 K: {e}")
```

### Complex Custom Constraint

```python
class ValidQuantumNumbers(Constraint):
    """Validate quantum numbers obey n >= 1, 0 <= l < n, -l <= m <= l."""

    def __init__(self, n: int, l: int):
        if n < 1:
            raise ValueError("n must be >= 1")
        if l < 0 or l >= n:
            raise ValueError(f"l must be in [0, {n-1}] for n={n}")
        self.n = n
        self.l = l

    @property
    def name(self) -> str:
        return "ValidQuantumNumbers"

    @property
    def description(self) -> str:
        return f"m must be in [-{self.l}, {self.l}] for n={self.n}, l={self.l}"

    def check(self, data):
        # Check magnetic quantum number m is in valid range
        return (data >= -self.l) & (data <= self.l) & (np.round(data) == data)

    def __repr__(self) -> str:
        return f"ValidQuantumNumbers(n={self.n}, l={self.l})"

# Magnetic quantum numbers for n=3, l=2 (d orbital)
m_values = DimArray([-2, -1, 0, 1, 2], units.dimensionless)
m_values.validate([ValidQuantumNumbers(n=3, l=2)])  # OK - all valid m values for d orbital
```

## Physics Examples

### Example 1: Temperature Validation

Temperature in Kelvin cannot be negative:

```python
from dimtensor import DimArray, units
from dimtensor.validation import Positive, Bounded

# Temperature must be above absolute zero
room_temp = DimArray([293.15], units.K)
room_temp.validate([Positive()])

# Water freezing/boiling points with phase constraints
ice = DimArray([270], units.K)
ice.validate([Bounded(0, 273.15)])

liquid_water = DimArray([300], units.K)
liquid_water.validate([Bounded(273.15, 373.15)])

steam = DimArray([400], units.K)
steam.validate([Bounded(373.15, 1000)])

# Convert to Celsius for display
print(f"Room temperature: {(room_temp.data[0] - 273.15):.1f} °C")
```

### Example 2: Mass Constraints

Mass must be positive with realistic bounds:

```python
from dimtensor.validation import Positive, Bounded

# Particle mass - must be positive
electron_mass = DimArray([9.109e-31], units.kg)
electron_mass.validate([Positive()])

# Astronomical mass with realistic bounds
star_mass_min = 0.08 * 1.989e30  # Minimum for hydrogen fusion (solar masses)
star_mass_max = 150 * 1.989e30   # Theoretical maximum before instability
star_mass = DimArray([2.0 * 1.989e30], units.kg)
star_mass.validate([Bounded(star_mass_min, star_mass_max)])  # OK - 2 solar masses is realistic
```

### Example 3: Probability Validation

Probabilities must be in [0, 1]:

```python
# Quantum state probabilities
probabilities = DimArray([0.25, 0.25, 0.25, 0.25], units.dimensionless)
probabilities.validate([Bounded(0, 1)])

# Ensure normalization
total_prob = probabilities.data.sum()
assert abs(total_prob - 1.0) < 1e-10, "Probabilities must sum to 1"

# Measurement outcome probability
measurement_prob = DimArray([0.64], units.dimensionless)
measurement_prob.validate([Bounded(0, 1), Finite()])
```

### Example 4: Energy Conservation in Orbital Mechanics

Track total energy during orbit simulation:

```python
import numpy as np
from dimtensor.validation import ConservationTracker

# Orbital mechanics - track mechanical energy
tracker = ConservationTracker("Orbital Energy")

# Constants
G = DimArray([6.674e-11], units.m**3 / (units.kg * units.s**2))  # Gravitational constant
M = DimArray([5.972e24], units.kg)  # Earth mass
m = DimArray([1000], units.kg)      # Satellite mass

# Initial orbital state (circular orbit)
r0 = DimArray([7000e3], units.m)    # 7000 km from Earth's center
v0 = DimArray([7.5e3], units.m / units.s)

# Initial energy: E = (1/2)mv^2 - GMm/r
E_kinetic = 0.5 * m * v0**2
E_potential = -G * M * m / r0
E_total = E_kinetic + E_potential
tracker.record(E_total)

# Simulate orbit (simplified)
r = r0
v = v0
dt = DimArray([10], units.s)

for step in range(100):
    # Gravitational acceleration
    a = -G * M / r**2

    # Update velocity and position (Euler integration)
    v = v + a * dt
    r = r + v * dt

    # Calculate energy
    E_kinetic = 0.5 * m * v**2
    E_potential = -G * M * m / r
    E_total = E_kinetic + E_potential
    tracker.record(E_total)

# Check energy conservation
print(f"Energy conserved: {tracker.is_conserved(rtol=1e-3)}")
print(f"Energy drift: {tracker.drift():.2e}")
print(f"Max drift: {tracker.max_drift():.2e}")

# Note: Euler integration has poor energy conservation
# Use better integrators (Verlet, RK4) for real simulations
```

### Example 5: Momentum Conservation in Collisions

Track momentum before and after collision:

```python
from dimtensor.validation import ConservationTracker

# Two-body collision - track total momentum
tracker = ConservationTracker("Total Momentum")

# Initial state
m1 = DimArray([2.0], units.kg)
m2 = DimArray([3.0], units.kg)
v1_initial = DimArray([5.0], units.m / units.s)
v2_initial = DimArray([-2.0], units.m / units.s)

# Initial momentum
p_initial = m1 * v1_initial + m2 * v2_initial
tracker.record(p_initial)

# Elastic collision formulas
v1_final = ((m1 - m2) * v1_initial + 2 * m2 * v2_initial) / (m1 + m2)
v2_final = ((m2 - m1) * v2_initial + 2 * m1 * v1_initial) / (m1 + m2)

# Final momentum
p_final = m1 * v1_final + m2 * v2_final
tracker.record(p_final)

# Verify conservation
print(f"Momentum conserved: {tracker.is_conserved(rtol=1e-10)}")
print(f"Initial: {p_initial}")
print(f"Final: {p_final}")
print(f"Drift: {tracker.drift():.2e}")
```

## Best Practices

### When to Validate

**Immediately after creation** for invariants that must always hold:

```python
# Mass is always positive
mass = DimArray([1.0], units.kg)
mass.validate([Positive()])
```

**After operations** for computed values:

```python
# Position may become invalid after computation
position = initial_position + velocity * time
position.validate([Finite()])
```

**Before critical operations** to prevent errors:

```python
# Ensure denominator is non-zero before division
time.validate([NonZero()])
velocity = distance / time
```

### Performance Considerations

Validation adds overhead. Use strategically:

```python
# DON'T: Validate inside tight loops
for i in range(1000000):
    result = compute(data)
    result.validate([Finite()])  # Expensive!

# DO: Validate once after loop
for i in range(1000000):
    result = compute(data)
result.validate([Finite()])  # Once at end

# DO: Sample validation in long simulations
for step in range(1000):
    result = simulate_step(result)
    if step % 100 == 0:  # Every 100 steps
        result.validate([Finite()])
```

### Choosing Tolerances

For conservation checking, choose tolerances based on:

**Analytical solutions:** Very tight tolerances (rtol=1e-12)

```python
tracker.is_conserved(rtol=1e-12)  # Exact analytical result
```

**Numerical integration:** Moderate tolerances based on method

```python
# Euler: rtol=1e-3
# RK4: rtol=1e-6
# Verlet: rtol=1e-8
tracker.is_conserved(rtol=1e-6)
```

**Long simulations:** Account for accumulated drift

```python
# 1000 steps, expect ~sqrt(1000) accumulation
tracker.is_conserved(rtol=1e-6 * np.sqrt(1000))
```

### Combining with Dimension Checking

Dimensions and validation work together:

```python
from dimtensor import DimArray, units
from dimtensor.validation import Positive, Bounded

# Dimensions ensure units are correct
mass = DimArray([1.0], units.kg)  # Must be mass dimension
velocity = DimArray([10.0], units.m / units.s)  # Must be velocity

# Operations check dimensions automatically
momentum = mass * velocity  # Dimensions checked: kg * m/s = kg⋅m/s

# Validation ensures values are physically reasonable
mass.validate([Positive()])
velocity.validate([Finite()])

# Both together provide complete safety
temperature = DimArray([300], units.K)
temperature.validate([Positive()])
# ✓ Correct dimension (temperature)
# ✓ Valid value (above absolute zero)
```

### Error Recovery

Handle validation errors gracefully:

```python
from dimtensor.errors import ConstraintError

def safe_compute(data):
    """Compute with validation and fallback."""
    try:
        result = expensive_computation(data)
        result.validate([Finite()])
        return result
    except ConstraintError as e:
        print(f"Validation failed: {e}")
        # Return safe default or raise
        return None

# Check before proceeding
result = safe_compute(input_data)
if result is not None:
    continue_simulation(result)
else:
    print("Computation produced invalid results")
```

## Summary

dimtensor's validation system provides:

- **Built-in constraints** for common physics requirements (positive, bounded, finite)
- **Conservation tracking** to detect numerical drift in conserved quantities
- **Custom constraints** for domain-specific validation rules
- **Dimension checking + validation** = complete safety for physical computations

Use validation to catch physics errors early, verify conservation laws, and ensure numerical stability in your simulations.

For more examples, see the [Examples Guide](examples.md). For automatic dimension checking, see [Operations](operations.md).

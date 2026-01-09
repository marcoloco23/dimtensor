# Working with Equations

> Discover, browse, and validate against a rich database of physics equations

Dimtensor includes a comprehensive database of 50+ physics equations spanning mechanics, thermodynamics, electromagnetism, fluid dynamics, relativity, quantum mechanics, optics, and acoustics. Each equation contains dimensional metadata, LaTeX representations, assumptions, and relationships to other equations.

## Table of Contents

- [Overview](#overview)
- [Browsing Equations](#browsing-equations)
- [Searching Equations](#searching-equations)
- [Equation Metadata](#equation-metadata)
- [Validating Calculations](#validating-calculations)
- [Custom Equations](#custom-equations)
- [Reference Tables](#reference-tables)
- [Advanced Examples](#advanced-examples)

---

## Overview

### What is the Equation Database?

The equation database provides structured metadata about common physics equations:

- **Variables**: Names mapped to their physical dimensions
- **Formulas**: Symbolic and LaTeX representations
- **Domains**: Categorization by physics field (mechanics, optics, etc.)
- **Tags**: Additional categorization (fundamental, energy, wave, etc.)
- **Assumptions**: Conditions under which the equation applies
- **Related equations**: Links to similar or derived equations

### Why Use It?

The equation database helps you:

- **Discover equations**: Browse by domain or search by topic
- **Validate calculations**: Check that your results have correct dimensions
- **Learn physics**: Explore equation metadata and relationships
- **Extend the system**: Register custom equations for your domain
- **Guide inference**: Use equations to infer units for unknown variables

### What Domains Are Covered?

The database spans 8 physics domains:

```python
from dimtensor.equations.database import list_domains

domains = list_domains()
print(domains)
# ['acoustics', 'electromagnetism', 'fluid_dynamics', 'mechanics',
#  'optics', 'quantum', 'relativity', 'thermodynamics']
```

---

## Browsing Equations

### Listing All Domains

Get a list of available physics domains:

```python
from dimtensor.equations.database import list_domains

domains = list_domains()
for domain in domains:
    print(domain)

# Output:
# acoustics
# electromagnetism
# fluid_dynamics
# mechanics
# optics
# quantum
# relativity
# thermodynamics
```

### Getting Equations by Domain

Retrieve all equations in a specific domain:

```python
from dimtensor.equations.database import get_equations

# Get all mechanics equations
mechanics_eqs = get_equations(domain="mechanics")
print(f"Found {len(mechanics_eqs)} mechanics equations")

# Display the first few
for eq in mechanics_eqs[:5]:
    print(f"{eq.name}: {eq.formula}")

# Output:
# Newton's Second Law: F = ma
# Kinetic Energy: KE = (1/2)mv^2
# Potential Energy (Gravitational): PE = mgh
# Work: W = F * d
# Power: P = W/t
```

### Filtering by Tags

Get equations with specific tags:

```python
from dimtensor.equations.database import get_equations

# Find all fundamental equations
fundamental = get_equations(tags=["fundamental"])
for eq in fundamental:
    print(f"{eq.name} ({eq.domain})")

# Output:
# Newton's Second Law (mechanics)
# Ideal Gas Law (thermodynamics)
# Ohm's Law (electromagnetism)
# Mass-Energy Equivalence (relativity)
```

Multiple tags can be specified (equations must have ALL tags):

```python
# Find equations tagged as both "energy" and "kinetic"
kinetic_energy_eqs = get_equations(tags=["energy", "kinetic"])
for eq in kinetic_energy_eqs:
    print(eq.name)

# Output:
# Kinetic Energy
```

### Getting All Equations

Retrieve the complete database:

```python
from dimtensor.equations.database import get_equations

# Get all equations (no filters)
all_eqs = get_equations()
print(f"Total equations in database: {len(all_eqs)}")

# Count equations per domain
from collections import Counter
domain_counts = Counter(eq.domain for eq in all_eqs)
for domain, count in sorted(domain_counts.items()):
    print(f"  {domain}: {count}")
```

---

## Searching Equations

### Searching by Name or Description

Search across equation names, descriptions, formulas, variable names, and tags:

```python
from dimtensor.equations.database import search_equations

# Search for equations related to energy
energy_eqs = search_equations("energy")
for eq in energy_eqs[:5]:
    print(f"{eq.name}: {eq.formula}")

# Output:
# Kinetic Energy: KE = (1/2)mv^2
# Potential Energy (Gravitational): PE = mgh
# Work: W = F * d
# Inductor Energy: E = (1/2)LI^2
# Capacitor Energy: E = (1/2)CV^2
```

### Searching by Variable

Find equations that use specific variables:

```python
# Find equations involving velocity
velocity_eqs = search_equations("velocity")
for eq in velocity_eqs[:3]:
    print(f"{eq.name}: {eq.formula}")

# Output:
# Kinetic Energy: KE = (1/2)mv^2
# Momentum: p = mv
# Centripetal Acceleration: a_c = v^2/r
```

### Searching by Physics Topic

Search for equations in a specific area:

```python
# Search for wave-related equations
wave_eqs = search_equations("wave")
for eq in wave_eqs:
    print(f"{eq.name} ({eq.domain}): {eq.formula}")

# Output:
# Acoustic Impedance (acoustics): Z = rho*v
# Wave Equation (1D) (acoustics): d^2y/dt^2 = v^2*d^2y/dx^2
# Wave Equation (electromagnetic) (optics): c = lambda*f
```

### Case-Insensitive Search

All searches are case-insensitive:

```python
# These all return the same results
search_equations("Newton")
search_equations("newton")
search_equations("NEWTON")
```

---

## Equation Metadata

### Getting a Specific Equation

Retrieve an equation by its exact name:

```python
from dimtensor.equations.database import get_equation

# Get Newton's Second Law
eq = get_equation("Newton's Second Law")
print(f"Name: {eq.name}")
print(f"Formula: {eq.formula}")
print(f"Domain: {eq.domain}")
print(f"Tags: {eq.tags}")

# Output:
# Name: Newton's Second Law
# Formula: F = ma
# Domain: mechanics
# Tags: ['newton', 'force', 'acceleration', 'fundamental']
```

!!! note "Exact Name Required"
    `get_equation()` requires the exact equation name. Use `search_equations()` for fuzzy matching.

### Accessing Formula and LaTeX

Every equation has both a symbolic formula and LaTeX representation:

```python
eq = get_equation("Ideal Gas Law")
print(f"Formula: {eq.formula}")
print(f"LaTeX: {eq.latex}")

# Output:
# Formula: PV = nRT
# LaTeX: PV = nRT
```

LaTeX can be used for rendering in Jupyter notebooks or documentation:

```python
from IPython.display import display, Math

eq = get_equation("Kinetic Energy")
display(Math(eq.latex))
# Renders: KE = \frac{1}{2}mv^2
```

### Viewing Variable Dimensions

Inspect the physical dimensions of each variable:

```python
eq = get_equation("Newton's Second Law")
print("Variables:")
for var_name, dimension in eq.variables.items():
    print(f"  {var_name}: {dimension}")

# Output:
# Variables:
#   F: M·L·T⁻²
#   m: M
#   a: L·T⁻²
```

### Checking Assumptions

Many equations have conditions or assumptions:

```python
eq = get_equation("Ideal Gas Law")
print(f"Assumptions:")
for assumption in eq.assumptions:
    print(f"  - {assumption}")

# Output:
# Assumptions:
#   - Ideal gas behavior
#   - No intermolecular forces
```

If no assumptions are listed, the list will be empty:

```python
eq = get_equation("Newton's Second Law")
print(f"Assumptions: {eq.assumptions}")
# Output: Assumptions: []
```

### Finding Related Equations

Explore relationships between equations:

```python
eq = get_equation("Kinetic Energy")
print(f"Related equations:")
for related_name in eq.related:
    print(f"  - {related_name}")

# Output:
# Related equations:
#   - Potential Energy
#   - Work-Energy Theorem
```

You can then retrieve these related equations:

```python
for related_name in eq.related:
    try:
        related_eq = get_equation(related_name)
        print(f"{related_eq.name}: {related_eq.formula}")
    except KeyError:
        print(f"{related_name}: (not in database)")
```

### Converting to Dictionary

Export equation metadata as a dictionary:

```python
eq = get_equation("Ohm's Law")
eq_dict = eq.to_dict()

import json
print(json.dumps(eq_dict, indent=2))

# Output:
# {
#   "name": "Ohm's Law",
#   "formula": "V = IR",
#   "variables": {
#     "V": "M·L²·T⁻³·I⁻¹",
#     "I": "I",
#     "R": "M·L²·T⁻³·I⁻²"
#   },
#   "domain": "electromagnetism",
#   "tags": ["circuits", "resistance", "current", "fundamental"],
#   "description": "Voltage equals current times resistance",
#   "assumptions": [],
#   "latex": "V = IR",
#   "related": ["Electric Power", "Kirchhoff's Laws"]
# }
```

---

## Validating Calculations

### Checking Dimensional Correctness

Use equations to verify your calculations have the right dimensions:

```python
from dimtensor import DimArray, units
from dimtensor.equations.database import get_equation

# Get Newton's Second Law
eq = get_equation("Newton's Second Law")

# Perform calculation
mass = DimArray([2.0], units.kg)
accel = DimArray([10.0], units.m / units.s**2)
force = mass * accel

# Validate dimensions match F = ma
F_dim = eq.variables["F"]
assert force.dimension == F_dim, "Dimension mismatch!"
print(f"✓ Dimensional validation passed: {force}")

# Output:
# ✓ Dimensional validation passed: [20.] N
```

### Validation Workflow Example

Create a validation function for a specific equation:

```python
from dimtensor import DimArray
from dimtensor.equations.database import get_equation

def validate_kinetic_energy(KE, m, v):
    """Validate kinetic energy calculation against KE = (1/2)mv²."""
    eq = get_equation("Kinetic Energy")

    # Check input dimensions
    assert m.dimension == eq.variables["m"], f"Mass dimension mismatch"
    assert v.dimension == eq.variables["v"], f"Velocity dimension mismatch"

    # Check output dimension
    assert KE.dimension == eq.variables["KE"], f"Energy dimension mismatch"

    # Check numerical relationship
    expected_KE = 0.5 * m * v**2
    assert (KE.to(expected_KE.unit).data == expected_KE.data).all(), \
        f"Numerical mismatch: {KE} != {expected_KE}"

    print(f"✓ Validation passed: KE = {KE}")

# Use it
from dimtensor import units
m = DimArray([5.0], units.kg)
v = DimArray([10.0], units.m / units.s)
KE = 0.5 * m * v**2

validate_kinetic_energy(KE, m, v)
# Output: ✓ Validation passed: KE = [250.] J
```

### Using Equations as Templates

Extract dimension requirements from equations:

```python
from dimtensor.equations.database import get_equation

eq = get_equation("Ideal Gas Law")
print(f"To use {eq.formula}, you need:")
for var_name, dimension in eq.variables.items():
    print(f"  {var_name}: {dimension}")

# Output:
# To use PV = nRT, you need:
#   P: M·L⁻¹·T⁻²
#   V: L³
#   n: N
#   R: M·L²·T⁻²·Θ⁻¹·N⁻¹
#   T: Θ
```

### Validating Against Multiple Equations

Check if a calculation is consistent with related equations:

```python
from dimtensor import DimArray, units
from dimtensor.equations.database import get_equations

# Calculate power
voltage = DimArray([12.0], units.V)
current = DimArray([2.0], units.A)
power = voltage * current

# Find all power equations
power_eqs = get_equations(tags=["power"])
for eq in power_eqs:
    if "P" in eq.variables:
        expected_dim = eq.variables["P"]
        if power.dimension == expected_dim:
            print(f"✓ Consistent with {eq.name}")

# Output:
# ✓ Consistent with Power
# ✓ Consistent with Electric Power
```

---

## Custom Equations

### Creating Custom Equations

Define equations for your specific domain:

```python
from dimtensor.equations.database import Equation
from dimtensor.core.dimensions import Dimension

# Define Tsiolkovsky rocket equation
rocket_eq = Equation(
    name="Rocket Equation",
    formula="delta_v = v_e * ln(m0/mf)",
    variables={
        "delta_v": Dimension(length=1, time=-1),
        "v_e": Dimension(length=1, time=-1),
        "m0": Dimension(mass=1),
        "mf": Dimension(mass=1),
    },
    domain="aerospace",
    tags=["rocket", "propulsion"],
    description="Tsiolkovsky rocket equation for ideal rocket",
    latex=r"\Delta v = v_e \ln\frac{m_0}{m_f}",
    assumptions=[
        "Constant exhaust velocity",
        "No external forces",
        "All mass is propellant or payload"
    ],
    related=["Momentum Conservation"]
)

print(f"Created: {rocket_eq.name}")
print(f"Formula: {rocket_eq.formula}")
```

### Registering Custom Equations

Add your equation to the global database:

```python
from dimtensor.equations.database import register_equation, get_equation

# Register the rocket equation
register_equation(rocket_eq)

# Now it's available globally
eq = get_equation("Rocket Equation")
print(f"Retrieved: {eq.name}")
print(f"Domain: {eq.domain}")

# It appears in domain listings
from dimtensor.equations.database import get_equations
aerospace_eqs = get_equations(domain="aerospace")
print(f"Found {len(aerospace_eqs)} aerospace equations")
```

!!! warning "Name Conflicts"
    If you register an equation with the same name as an existing one, it will overwrite the original. Use unique names for custom equations.

### Organizing Domain-Specific Equation Sets

Create modules for specific domains:

```python
# equations/chemistry.py
from dimtensor.equations.database import Equation, register_equation
from dimtensor.core.dimensions import Dimension

# Define chemistry equations
CHEMISTRY_EQUATIONS = [
    Equation(
        name="Nernst Equation",
        formula="E = E0 - (RT/nF)*ln(Q)",
        variables={
            "E": Dimension(mass=1, length=2, time=-3, current=-1),
            "E0": Dimension(mass=1, length=2, time=-3, current=-1),
            "R": Dimension(mass=1, length=2, time=-2, temperature=-1, amount=-1),
            "T": Dimension(temperature=1),
            "n": Dimension(),  # dimensionless
            "F": Dimension(current=1, time=1, amount=-1),
            "Q": Dimension(),  # dimensionless
        },
        domain="chemistry",
        tags=["electrochemistry", "potential"],
        description="Electrode potential as function of concentration",
        latex=r"E = E^0 - \frac{RT}{nF}\ln Q"
    ),
    # Add more chemistry equations...
]

def register_chemistry_equations():
    """Register all chemistry equations."""
    for eq in CHEMISTRY_EQUATIONS:
        register_equation(eq)

# Use in your code
# from equations.chemistry import register_chemistry_equations
# register_chemistry_equations()
```

### Best Practices for Custom Equations

**DO: Use descriptive names**

```python
# Good
Equation(name="Bernoulli's Equation for Incompressible Flow", ...)

# Avoid
Equation(name="Eq1", ...)
```

**DO: Include comprehensive metadata**

```python
Equation(
    name="...",
    formula="...",
    variables={...},
    domain="...",
    tags=["tag1", "tag2"],  # Add multiple tags
    description="Detailed description",
    latex=r"...",  # LaTeX for rendering
    assumptions=[...],  # List all conditions
    related=[...]  # Link to related equations
)
```

**DO: Document assumptions clearly**

```python
Equation(
    name="Drag Force",
    assumptions=[
        "Turbulent flow regime (Re > 1000)",
        "Constant drag coefficient",
        "Uniform density"
    ],
    ...
)
```

**DON'T: Forget to specify all variables**

```python
# Bad - missing 'r' variable
Equation(
    formula="F = ma + r",
    variables={"F": ..., "m": ..., "a": ...},  # Where's 'r'?
)

# Good - all variables included
Equation(
    formula="F = ma + r",
    variables={"F": ..., "m": ..., "a": ..., "r": ...},
)
```

---

## Reference Tables

### Available Domains

| Domain | Description | Example Equations |
|--------|-------------|-------------------|
| **mechanics** | Classical mechanics | Newton's laws, energy, momentum |
| **thermodynamics** | Heat and temperature | Ideal gas law, entropy, heat capacity |
| **electromagnetism** | Electric and magnetic | Ohm's law, Coulomb's law, Lorentz force |
| **fluid_dynamics** | Fluid flow | Bernoulli, Navier-Stokes, Reynolds number |
| **relativity** | Special/general relativity | E=mc², time dilation, Lorentz factor |
| **quantum** | Quantum mechanics | Schrödinger, de Broglie, uncertainty |
| **optics** | Light and waves | Snell's law, thin lens, diffraction |
| **acoustics** | Sound and vibration | Wave equation, Doppler, sound intensity |

### Example Equations by Domain

#### Mechanics (10 equations)

| Equation | Formula | Description |
|----------|---------|-------------|
| Newton's Second Law | F = ma | Force equals mass times acceleration |
| Kinetic Energy | KE = ½mv² | Energy of motion |
| Momentum | p = mv | Linear momentum |
| Gravitational Force | F = Gm₁m₂/r² | Universal gravitation |
| Hooke's Law | F = -kx | Spring force |

#### Thermodynamics (5 equations)

| Equation | Formula | Description |
|----------|---------|-------------|
| Ideal Gas Law | PV = nRT | Gas state equation |
| First Law | ΔU = Q - W | Energy conservation |
| Heat Capacity | Q = mcΔT | Heat transfer |
| Stefan-Boltzmann | P = σAT⁴ | Blackbody radiation |
| Entropy Change | dS = dQ/T | Entropy definition |

#### Electromagnetism (6 equations)

| Equation | Formula | Description |
|----------|---------|-------------|
| Ohm's Law | V = IR | Voltage-current relation |
| Coulomb's Law | F = kq₁q₂/r² | Electrostatic force |
| Electric Power | P = IV | Electrical power |
| Capacitor Energy | E = ½CV² | Energy in capacitor |
| Inductor Energy | E = ½LI² | Energy in inductor |
| Lorentz Force | F = q(E + v×B) | EM force on charge |

#### Fluid Dynamics (5 equations)

| Equation | Formula | Description |
|----------|---------|-------------|
| Bernoulli's Equation | P + ½ρv² + ρgh = const | Energy in flow |
| Continuity | A₁v₁ = A₂v₂ | Mass conservation |
| Reynolds Number | Re = ρvL/μ | Flow regime indicator |
| Stokes' Law | Fd = 6πμrv | Low Re drag |
| Drag Equation | Fd = ½ρv²CdA | High Re drag |

#### Relativity (6 equations)

| Equation | Formula | Description |
|----------|---------|-------------|
| Mass-Energy | E = mc² | Rest energy |
| Lorentz Factor | γ = 1/√(1-v²/c²) | Relativistic factor |
| Time Dilation | Δt = γΔt₀ | Moving clocks run slow |
| Length Contraction | L = L₀/γ | Moving lengths contract |
| Relativistic Momentum | p = γmv | Momentum at high speed |
| Energy-Momentum | E² = (pc)² + (mc²)² | Invariant mass relation |

#### Quantum (8 equations)

| Equation | Formula | Description |
|----------|---------|-------------|
| Planck-Einstein | E = ℏω | Photon energy |
| de Broglie | λ = h/p | Matter wavelength |
| Heisenberg (Δx·Δp) | ΔxΔp ≥ ℏ/2 | Position-momentum uncertainty |
| Heisenberg (ΔE·Δt) | ΔEΔt ≥ ℏ/2 | Energy-time uncertainty |
| Schrödinger | Ĥψ = Eψ | Time-independent |
| Bohr Radius | a₀ = ℏ²/(mₑe²kₑ) | Hydrogen ground state |
| Rydberg Formula | 1/λ = R(1/n₁² - 1/n₂²) | Hydrogen spectrum |
| Harmonic Oscillator | Eₙ = ℏω(n + ½) | QHO energy levels |

#### Optics (9 equations)

| Equation | Formula | Description |
|----------|---------|-------------|
| Snell's Law | n₁sinθ₁ = n₂sinθ₂ | Refraction |
| Thin Lens | 1/f = 1/dₒ + 1/dᵢ | Image formation |
| Magnification | M = -dᵢ/dₒ | Image size ratio |
| Diffraction Grating | d sinθ = mλ | Interference condition |
| Rayleigh Criterion | θₘᵢₙ = 1.22λ/D | Resolution limit |
| Wave Equation | c = λf | Wave speed |

#### Acoustics (8 equations)

| Equation | Formula | Description |
|----------|---------|-------------|
| Speed of Sound (fluid) | v = √(B/ρ) | Sound speed in medium |
| Speed of Sound (gas) | v = √(γRT/M) | Sound speed in ideal gas |
| Sound Intensity | I = P²/(2ρv) | Intensity from pressure |
| Intensity Level | L = 10log₁₀(I/I₀) | Decibel scale |
| Doppler Effect | f = f₀(v+vᵣ)/(v-vₛ) | Frequency shift |
| Acoustic Impedance | Z = ρv | Medium property |
| Wave Equation (1D) | ∂²y/∂t² = v²∂²y/∂x² | Wave PDE |
| Standing Wave | fₙ = nv/(2L) | String harmonics |

### Equation Metadata Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `name` | str | Human-readable name | "Newton's Second Law" |
| `formula` | str | Symbolic formula | "F = ma" |
| `variables` | dict | Variable → Dimension mapping | {"F": M·L·T⁻², "m": M, ...} |
| `domain` | str | Physics domain | "mechanics" |
| `tags` | list[str] | Categorization tags | ["newton", "force", "fundamental"] |
| `description` | str | Longer description | "Force equals mass times..." |
| `assumptions` | list[str] | Conditions/assumptions | ["Constant mass", "Inertial frame"] |
| `latex` | str | LaTeX representation | r"F = ma" |
| `related` | list[str] | Related equation names | ["Newton's First Law", ...] |

---

## Advanced Examples

### Integration with Inference System

Use equations to guide unit inference:

```python
from dimtensor.equations.database import get_equation
from dimtensor import units

# Get equation metadata
eq = get_equation("Ideal Gas Law")
print(f"Equation: {eq.formula}")
print(f"Variables needed:")
for var, dim in eq.variables.items():
    print(f"  {var}: {dim}")

# Use equation to validate inference
# If we know P, V, n, T, we can check if R has the right dimension
P = units.Pa
V = units.m**3
n = units.mol
T = units.K
R_expected = eq.variables["R"]

# Calculate what R should be dimensionally
R_calculated = (P * V) / (n * T)
print(f"\nExpected R dimension: {R_expected}")
print(f"Calculated R dimension: {R_calculated.dimension}")
print(f"Match: {R_calculated.dimension == R_expected}")

# Output:
# Equation: PV = nRT
# Variables needed:
#   P: M·L⁻¹·T⁻²
#   V: L³
#   n: N
#   R: M·L²·T⁻²·Θ⁻¹·N⁻¹
#   T: Θ
#
# Expected R dimension: M·L²·T⁻²·Θ⁻¹·N⁻¹
# Calculated R dimension: M·L²·T⁻²·Θ⁻¹·N⁻¹
# Match: True
```

### Building Equation Validators

Create a validator class for physics equations:

```python
from dimtensor import DimArray
from dimtensor.equations.database import get_equation

class EquationValidator:
    """Validate calculations against physics equations."""

    def __init__(self, equation_name):
        self.equation = get_equation(equation_name)

    def validate(self, **variables):
        """Validate that provided variables have correct dimensions.

        Args:
            **variables: Variable names and DimArray values

        Returns:
            bool: True if all dimensions match

        Raises:
            ValueError: If dimensions don't match
        """
        errors = []

        for var_name, var_value in variables.items():
            if var_name not in self.equation.variables:
                errors.append(f"Unknown variable: {var_name}")
                continue

            expected_dim = self.equation.variables[var_name]
            actual_dim = var_value.dimension

            if actual_dim != expected_dim:
                errors.append(
                    f"{var_name}: expected {expected_dim}, got {actual_dim}"
                )

        if errors:
            raise ValueError(f"Validation failed:\n" + "\n".join(errors))

        return True

    def __repr__(self):
        return f"EquationValidator('{self.equation.name}')"

# Use it
from dimtensor import units

validator = EquationValidator("Newton's Second Law")
print(validator)

# Valid inputs
mass = DimArray([2.0], units.kg)
accel = DimArray([10.0], units.m / units.s**2)
force = DimArray([20.0], units.N)

validator.validate(F=force, m=mass, a=accel)
print("✓ Validation passed")

# Invalid inputs (wrong dimension for acceleration)
try:
    bad_accel = DimArray([10.0], units.m / units.s)  # Wrong: should be m/s²
    validator.validate(F=force, m=mass, a=bad_accel)
except ValueError as e:
    print(f"✗ Validation failed: {e}")
```

### Creating Interactive Equation Browsers

Build tools to explore the equation database:

```python
from dimtensor.equations.database import get_equations, list_domains

def browse_equations_by_domain(domain=None):
    """Interactive equation browser."""
    if domain is None:
        # List all domains
        print("Available domains:")
        for i, d in enumerate(list_domains(), 1):
            eqs = get_equations(domain=d)
            print(f"  {i}. {d} ({len(eqs)} equations)")
        return

    # Show equations in domain
    equations = get_equations(domain=domain)
    print(f"\n{domain.upper()} ({len(equations)} equations)")
    print("=" * 60)

    for i, eq in enumerate(equations, 1):
        print(f"\n{i}. {eq.name}")
        print(f"   Formula: {eq.formula}")
        print(f"   Tags: {', '.join(eq.tags)}")
        if eq.description:
            print(f"   Description: {eq.description}")

# Browse all domains
browse_equations_by_domain()

# Browse specific domain
browse_equations_by_domain("quantum")
```

### Multi-Equation Validation

Validate calculations against multiple related equations:

```python
from dimtensor import DimArray, units
from dimtensor.equations.database import get_equation

def validate_against_related(equation_name, **variables):
    """Validate against an equation and all its related equations."""

    eq = get_equation(equation_name)
    results = []

    # Validate against main equation
    try:
        for var_name, var_value in variables.items():
            if var_name in eq.variables:
                expected = eq.variables[var_name]
                actual = var_value.dimension
                match = expected == actual
                results.append({
                    'equation': eq.name,
                    'variable': var_name,
                    'expected': expected,
                    'actual': actual,
                    'match': match
                })
    except Exception as e:
        results.append({
            'equation': eq.name,
            'error': str(e)
        })

    # Validate against related equations
    for related_name in eq.related:
        try:
            related_eq = get_equation(related_name)
            for var_name, var_value in variables.items():
                if var_name in related_eq.variables:
                    expected = related_eq.variables[var_name]
                    actual = var_value.dimension
                    match = expected == actual
                    results.append({
                        'equation': related_eq.name,
                        'variable': var_name,
                        'expected': expected,
                        'actual': actual,
                        'match': match
                    })
        except KeyError:
            # Related equation not in database
            pass

    return results

# Example: Validate energy calculation
m = DimArray([5.0], units.kg)
v = DimArray([10.0], units.m / units.s)
KE = 0.5 * m * v**2

results = validate_against_related("Kinetic Energy", KE=KE, m=m, v=v)

print("Validation Results:")
print("-" * 70)
for r in results:
    if 'error' in r:
        print(f"✗ {r['equation']}: ERROR - {r['error']}")
    elif r['match']:
        print(f"✓ {r['equation']}: {r['variable']} = {r['actual']}")
    else:
        print(f"✗ {r['equation']}: {r['variable']} expected {r['expected']}, got {r['actual']}")
```

### Equation-Based Unit Conversion Hints

Use equations to suggest appropriate unit conversions:

```python
from dimtensor import DimArray, units
from dimtensor.equations.database import search_equations

def suggest_conversions(value, context=""):
    """Suggest unit conversions based on physics context."""

    # Find equations matching context
    equations = search_equations(context) if context else []

    suggestions = []
    dim = value.dimension

    # Check which equation variables match this dimension
    for eq in equations[:5]:  # Top 5 matches
        for var_name, var_dim in eq.variables.items():
            if var_dim == dim:
                suggestions.append({
                    'equation': eq.name,
                    'variable': var_name,
                    'formula': eq.formula,
                    'domain': eq.domain
                })

    return suggestions

# Example
energy = DimArray([1000.0], units.J)
suggestions = suggest_conversions(energy, "energy")

print(f"Value: {energy}")
print(f"\nThis dimension appears in:")
for s in suggestions[:3]:
    print(f"  - {s['equation']} as '{s['variable']}'")
    print(f"    Formula: {s['formula']}")
    print(f"    Domain: {s['domain']}")
```

---

## See Also

- [Dimensional Inference](inference.md) - Automatic unit inference and linting
- [Working with Units](units.md) - Unit system and conversions
- [Validation](validation.md) - Dimensional validation tools
- [Examples](examples.md) - Real-world physics calculations
- [API Reference](../api/validation.md) - Detailed API documentation

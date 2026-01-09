# JAX Integration

dimtensor provides seamless integration with [JAX](https://github.com/google/jax), Google's high-performance numerical computing library. The JAX-compatible `DimArray` class works with all JAX transformations while preserving unit safety.

## What is JAX?

JAX is a numerical computing library that combines:

- **NumPy-like API**: Familiar array operations with `jax.numpy`
- **JIT compilation**: Just-in-time compilation to XLA for speed
- **Automatic differentiation**: Forward and reverse-mode autodiff
- **Vectorization**: Automatic batching with `vmap`
- **Hardware acceleration**: GPU/TPU support

dimtensor's JAX integration brings physical units and dimensional safety to all these features.

## Installation

To use JAX with dimtensor, install both libraries:

```bash
pip install dimtensor jax jaxlib
```

!!! note "Platform Support"
    JAX installation varies by platform and hardware. See [JAX installation guide](https://github.com/google/jax#installation) for details.

## Getting Started

Import the JAX-compatible `DimArray` from `dimtensor.jax`:

```python
import jax
import jax.numpy as jnp
from dimtensor.jax import DimArray
from dimtensor import units

# Create DimArray from JAX array
velocities = DimArray(jnp.array([1.0, 2.0, 3.0]), units.m / units.s)

# Create from Python list (auto-converts to JAX array)
masses = DimArray([10.0, 20.0, 30.0], units.kg)

# Arithmetic preserves units
momentum = masses * velocities  # kg*m/s
print(momentum)  # [10. 40. 90.] kg*m/s
```

All arithmetic operations preserve dimensional correctness:

```python
# Same dimension required for addition
distance1 = DimArray([100.0], units.m)
distance2 = DimArray([50.0], units.m)
total = distance1 + distance2  # [150.0] m

# Dimensions multiply/divide
time = DimArray([5.0], units.s)
velocity = total / time  # [30.0] m/s

# Power scales dimensions
area = distance1 ** 2  # [10000.0] m^2
```

## Understanding Pytrees

JAX transformations like `jit`, `vmap`, and `grad` work on **pytrees** - a tree-like structure of arrays and containers. dimtensor's `DimArray` is registered as a JAX pytree, enabling seamless integration.

### How DimArray is Registered

When you import `dimtensor.jax`, `DimArray` is automatically registered with JAX's pytree system:

```python
# Happens automatically on import
from dimtensor.jax import DimArray
```

This registration tells JAX how to:

1. **Flatten**: Extract the JAX array (traceable) and unit metadata (static)
2. **Unflatten**: Reconstruct `DimArray` from the array and metadata

### Why This Matters

Pytree registration enables:

- **JIT compilation**: Units preserved through compilation
- **Batching**: `vmap` works on `DimArray` directly
- **Differentiation**: `grad` understands `DimArray` structure
- **Nested structures**: Lists/dicts of `DimArray` work transparently

```python
# Example: pytree flatten/unflatten
da = DimArray([1.0, 2.0], units.m)

# JAX can flatten it
leaves, treedef = jax.tree_util.tree_flatten(da)
print(leaves)  # [Array([1., 2.], dtype=float32)]

# And unflatten it
reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
print(reconstructed)  # [1. 2.] m
```

## JIT Compilation

`jax.jit` compiles functions to XLA for dramatic speedups. Units are preserved through JIT compilation.

### Basic JIT Example

```python
import jax
from dimtensor.jax import DimArray
from dimtensor import units

@jax.jit
def double_velocity(v):
    return v * 2.0

velocity = DimArray([10.0, 20.0, 30.0], units.m / units.s)
result = double_velocity(velocity)
print(result)  # [20. 40. 60.] m/s
```

### Physics Function Example

JIT compilation shines with complex physics calculations:

```python
@jax.jit
def kinetic_energy(mass, velocity):
    """Calculate kinetic energy: KE = 0.5 * m * v^2"""
    return 0.5 * mass * velocity ** 2

m = DimArray([1.0, 2.0, 5.0], units.kg)
v = DimArray([10.0, 5.0, 2.0], units.m / units.s)

# First call: compiles and runs
E = kinetic_energy(m, v)
print(E)  # [50. 25. 10.] J

# Subsequent calls: reuses compiled code (fast!)
E2 = kinetic_energy(m * 2, v * 0.5)
print(E2)  # [25. 12.5 5.] J
```

### Chaining Operations

JIT works with complex expressions:

```python
@jax.jit
def position_after_time(initial_pos, velocity, time):
    """Calculate position: x = x0 + v*t"""
    return initial_pos + velocity * time

x0 = DimArray([0.0], units.m)
v = DimArray([5.0], units.m / units.s)
t = DimArray([10.0], units.s)

x = position_after_time(x0, v, t)
print(x)  # [50.0] m
```

### Performance Benefits

JIT compilation provides significant speedups for repeated calls:

```python
import time

def slow_function(x):
    """Without JIT"""
    result = x
    for _ in range(100):
        result = result * 1.01 + x * 0.001
    return result

@jax.jit
def fast_function(x):
    """With JIT"""
    result = x
    for _ in range(100):
        result = result * 1.01 + x * 0.001
    return result

x = DimArray(jnp.ones(1000), units.m)

# Warm up JIT
_ = fast_function(x)

# Time comparison (JIT is much faster)
start = time.time()
for _ in range(100):
    _ = slow_function(x)
slow_time = time.time() - start

start = time.time()
for _ in range(100):
    _ = fast_function(x)
fast_time = time.time() - start

print(f"Speedup: {slow_time / fast_time:.1f}x")
```

!!! tip "When to Use JIT"
    Use JIT for:

    - Functions called repeatedly with similar shapes
    - Numerical computations with many operations
    - Performance-critical inner loops

    Avoid JIT for:

    - Functions called once
    - Operations with frequently changing shapes
    - Code with Python control flow dependent on array values

## Vectorization with vmap

`jax.vmap` automatically vectorizes functions over a batch dimension, eliminating manual loops.

### Basic vmap Example

```python
def square(x):
    """Square a scalar-like DimArray"""
    return x ** 2

# Batch of lengths
lengths = DimArray(jnp.array([[1.0], [2.0], [3.0], [4.0]]), units.m)

# vmap applies square to each element in the batch
areas = jax.vmap(square)(lengths)
print(areas)  # [[1.0], [4.0], [9.0], [16.0]] m^2
```

### Batching Multiple Arguments

`vmap` works with functions taking multiple arguments:

```python
def multiply(a, b):
    """Multiply two quantities"""
    return a * b

# Batches of lengths and widths
lengths = DimArray(jnp.array([[2.0], [3.0], [4.0]]), units.m)
widths = DimArray(jnp.array([[5.0], [6.0], [7.0]]), units.m)

# vmap over both arguments
areas = jax.vmap(multiply)(lengths, widths)
print(areas)  # [[10.0], [18.0], [28.0]] m^2
```

### Physics Example: Force Calculations

Calculate forces on multiple masses with different accelerations:

```python
def force(mass, acceleration):
    """F = m * a"""
    return mass * acceleration

# 5 objects with different masses and accelerations
masses = DimArray(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), units.kg)
accelerations = DimArray(
    jnp.array([9.8, 5.0, 2.5, 10.0, 1.0]),
    units.m / units.s**2
)

forces = jax.vmap(force)(masses, accelerations)
print(forces)  # [9.8, 10.0, 7.5, 40.0, 5.0] N
```

### Specifying Batch Axes

Control which axes to vectorize over:

```python
# Matrix with batch in first dimension: (batch, features)
data = DimArray(jnp.ones((10, 3)), units.m)

def process_features(row):
    return row.sum()

# vmap over axis 0 (batch dimension)
results = jax.vmap(process_features)(data)
print(results.shape)  # (10,) - one result per batch element
```

!!! note "Dimension Requirements"
    `vmap` preserves dimensional correctness. All batch elements must have compatible dimensions for operations to succeed.

## Automatic Differentiation

`jax.grad` computes gradients automatically. When working with `DimArray`, remember that gradients have dimensions too.

### Basic Gradient Example

```python
def energy(velocity):
    """Kinetic energy as function of velocity (mass = 1 kg)"""
    mass = DimArray(1.0, units.kg)
    return 0.5 * mass * velocity ** 2

# Gradient of energy with respect to velocity
grad_energy = jax.grad(lambda v: energy(v).magnitude().sum())

v = DimArray(jnp.array([10.0]), units.m / units.s)
dE_dv = grad_energy(v)

# Gradient: dE/dv = m*v (with dimensions)
print(dE_dv)  # [10.0] kg*m/s (momentum!)
```

### Understanding Gradient Dimensions

Gradients have dimensions based on the derivative:

```python
# If f(x) has dimension [F] and x has dimension [X]
# Then df/dx has dimension [F]/[X]

def quadratic(x):
    """Dimensionless quadratic function"""
    return (x ** 2).sum()

x = DimArray(jnp.array([2.0, 3.0]), units.m)

# d(m^2)/d(m) = 2*m, dimension = [m^2]/[m] = [m]
grad_f = jax.grad(lambda x: quadratic(x).magnitude().sum())
df_dx = grad_f(x)
print(df_dx)  # [4.0, 6.0] m
```

### Optimization Example

Use gradients for physics-based optimization:

```python
def trajectory_height(launch_angle):
    """Maximum height of projectile (simplified)"""
    # For fixed initial velocity
    v0 = DimArray(20.0, units.m / units.s)
    g = DimArray(9.8, units.m / units.s**2)

    # h_max = (v0 * sin(angle))^2 / (2*g)
    # angle must be dimensionless (radians)
    vertical_v = v0 * jnp.sin(launch_angle)
    height = vertical_v ** 2 / (2 * g)
    return height

# Find optimal angle (should be 45 degrees = π/4)
grad_h = jax.grad(lambda angle: trajectory_height(angle).magnitude())

angle = 0.7  # Initial guess (radians, dimensionless)
for _ in range(10):
    dh = grad_h(angle)
    angle = angle + 0.01 * dh  # Gradient ascent

print(f"Optimal angle: {angle:.3f} radians = {angle * 180 / 3.14159:.1f} degrees")
# Optimal angle: 0.785 radians = 45.0 degrees
```

### value_and_grad Pattern

Compute both value and gradient efficiently:

```python
def potential_energy(height):
    """PE = m * g * h"""
    mass = DimArray(5.0, units.kg)
    g = DimArray(9.8, units.m / units.s**2)
    return mass * g * height

# Get both energy and gradient in one pass
value_and_grad_fn = jax.value_and_grad(
    lambda h: potential_energy(h).magnitude()
)

h = DimArray(10.0, units.m)
pe_value, dpe_dh = value_and_grad_fn(h)

print(f"PE = {pe_value:.1f}")  # PE = 490.0
print(f"dPE/dh = {dpe_dh:.1f}")  # dPE/dh = 49.0 (= m*g)
```

!!! warning "Gradient Limitations"
    - JAX's `grad` differentiates with respect to the first argument by default
    - When using `DimArray`, you often need to extract `.magnitude()` for scalar outputs
    - Gradients preserve dimensional relationships, but may require unit-aware interpretation

## Combining Transformations

JAX transformations compose naturally. Combine them for powerful patterns.

### JIT + vmap: Fast Batched Operations

```python
@jax.jit
def kinetic_energy(mass, velocity):
    return 0.5 * mass * velocity ** 2

# Vectorize the JIT-compiled function
kinetic_energy_batched = jax.vmap(kinetic_energy)

# Batches of masses and velocities
masses = DimArray(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), units.kg)
velocities = DimArray(jnp.array([10.0, 8.0, 6.0, 4.0, 2.0]), units.m / units.s)

# Fast batched computation
energies = kinetic_energy_batched(masses, velocities)
print(energies)  # [50.0, 64.0, 54.0, 32.0, 10.0] J
```

### JIT + grad: Fast Gradients

```python
@jax.jit
def loss_function(params):
    """Some physics-based loss"""
    return (params ** 2).sum()

# JIT-compiled gradient
grad_loss = jax.jit(jax.grad(lambda p: loss_function(p).magnitude()))

params = DimArray(jnp.array([1.0, 2.0, 3.0]), units.m)
gradient = grad_loss(params)
print(gradient)  # [2.0, 4.0, 6.0] m
```

### vmap + grad: Batched Gradients (Jacobians)

Compute gradients for multiple inputs simultaneously:

```python
def f(x):
    """Quadratic function"""
    return (x ** 2).sum()

# Gradient function
grad_f = jax.grad(lambda x: f(x).magnitude())

# Batch of inputs
inputs = DimArray(jnp.array([[1.0], [2.0], [3.0], [4.0]]), units.m)

# Compute gradient for each input in batch
gradients = jax.vmap(grad_f)(inputs)
print(gradients)  # [[2.0], [4.0], [6.0], [8.0]] m
```

### Real Example: Neural Network with Units

Combine all transformations for a physics-informed neural network:

```python
@jax.jit
def network_forward(weights, x):
    """Simple network: y = w * x^2"""
    return weights * x ** 2

# Batched and differentiated loss
@jax.jit
def train_step(weights, batch_x, batch_y):
    """One training step"""
    def loss_fn(w):
        predictions = jax.vmap(lambda x: network_forward(w, x))(batch_x)
        errors = predictions - batch_y
        # Return scalar loss (magnitude)
        return (errors ** 2).sum().magnitude()

    loss = loss_fn(weights)
    grad = jax.grad(loss_fn)(weights)

    # Update weights
    learning_rate = 0.01
    new_weights = weights - learning_rate * grad

    return new_weights, loss

# Training data
weights = DimArray(1.0, units.kg / units.m**2)
batch_x = DimArray(jnp.array([[1.0], [2.0], [3.0]]), units.m)
batch_y = DimArray(jnp.array([[2.0], [8.0], [18.0]]), units.kg)

# Training loop
for epoch in range(10):
    weights, loss = train_step(weights, batch_x, batch_y)
    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

## Performance Considerations

### When to Use JIT

**Use JIT for:**

- Functions called repeatedly
- Compute-intensive operations
- Stable input shapes

**Avoid JIT for:**

- One-off computations
- Dynamic shapes
- Heavy Python control flow

### Memory Efficiency

JAX arrays are immutable. Avoid creating unnecessary copies:

```python
# Good: Chain operations
result = ((x + y) * z).sum()

# Less efficient: Many intermediate variables
temp1 = x + y
temp2 = temp1 * z
result = temp2.sum()
```

### XLA Optimization Tips

1. **Batch operations**: Use `vmap` instead of loops
2. **Avoid Python loops**: Use JAX array operations
3. **Keep computations on device**: Minimize `.numpy()` conversions
4. **Profile code**: Use `jax.profiler` to identify bottlenecks

```python
# Bad: Python loop
result = velocities[0]
for v in velocities[1:]:
    result = result + v

# Good: JAX reduction
result = velocities.sum()
```

## Common Patterns

### Physics Simulations

Simulate particle motion with JIT and vmap:

```python
@jax.jit
def update_position(pos, vel, dt):
    """Euler integration: x(t+dt) = x(t) + v*dt"""
    return pos + vel * dt

@jax.jit
def update_velocity(vel, acc, dt):
    """v(t+dt) = v(t) + a*dt"""
    return vel + acc * dt

# Initial conditions
positions = DimArray(jnp.zeros(10), units.m)
velocities = DimArray(jnp.ones(10), units.m / units.s)
accelerations = DimArray(jnp.full(10, -9.8), units.m / units.s**2)
dt = DimArray(0.01, units.s)

# Simulation loop
for step in range(100):
    positions = update_position(positions, velocities, dt)
    velocities = update_velocity(velocities, accelerations, dt)

print(f"Final positions: {positions}")
```

### Unit Conversions in Batch

Convert units efficiently with vmap:

```python
def convert_to_mph(speed_ms):
    """Convert m/s to mph"""
    mph = units.mile / units.hour
    return speed_ms.to(mph)

# Batch of speeds in m/s
speeds_ms = DimArray(
    jnp.array([10.0, 20.0, 30.0]),
    units.m / units.s
)

# Convert all at once
speeds_mph = jax.vmap(convert_to_mph)(speeds_ms.reshape(3, 1))
print(speeds_mph)  # [~22.4, ~44.7, ~67.1] mph
```

### Uncertainty Propagation

Propagate uncertainties using gradients:

```python
def propagate_uncertainty(f, x, x_std):
    """
    Linear uncertainty propagation: σ_f ≈ |df/dx| * σ_x
    """
    grad_f = jax.grad(lambda val: f(val).magnitude())
    df_dx = grad_f(x)
    return jnp.abs(df_dx) * x_std

# Example: uncertainty in kinetic energy
def KE(v):
    m = DimArray(1.0, units.kg)
    return 0.5 * m * v ** 2

v = DimArray(10.0, units.m / units.s)
v_std = 0.5  # m/s uncertainty

E_std = propagate_uncertainty(KE, v, v_std)
print(f"KE = {KE(v)} ± {E_std}")
```

## Limitations and Troubleshooting

### What Works

- ✅ All arithmetic operations
- ✅ JIT compilation
- ✅ vmap vectorization
- ✅ grad differentiation
- ✅ Nested pytrees (lists/dicts of DimArray)
- ✅ Reductions (sum, mean, max, etc.)
- ✅ Reshaping and indexing
- ✅ Linear algebra (dot, matmul)

### Current Limitations

- ❌ `grad` with respect to `DimArray` requires extracting `.magnitude()` for scalar outputs
- ❌ Some JAX functions (e.g., `jax.lax` primitives) may not automatically handle `DimArray`
- ❌ Control flow primitives (`cond`, `while_loop`) need careful unit handling

### Common Errors

#### DimensionError in JIT Functions

```python
@jax.jit
def bad_function(x, y):
    return x + y  # Will fail if x and y have incompatible dimensions

# Fix: Ensure compatible dimensions
a = DimArray([1.0], units.m)
b = DimArray([2.0], units.s)  # Incompatible!
# bad_function(a, b)  # DimensionError
```

#### Tracing Errors with Units

If you get tracing errors, the issue is often unit-dependent control flow:

```python
# Bad: Control flow depends on traced values
@jax.jit
def bad_conditional(x):
    if x.magnitude() > 5.0:  # Can't trace this!
        return x * 2
    return x

# Fix: Use jax.lax.cond
@jax.jit
def good_conditional(x):
    return jax.lax.cond(
        x.magnitude() > 5.0,
        lambda x: x * 2,
        lambda x: x,
        x
    )
```

#### When to Strip Units

Sometimes you need raw arrays for certain JAX operations:

```python
# Get magnitude (plain JAX array) when needed
da = DimArray([1.0, 2.0, 3.0], units.m)
plain_array = da.magnitude()  # or da.data

# Use with JAX functions that don't support DimArray
result = jax.lax.scan(some_function, init, plain_array)

# Re-wrap result if appropriate
result_with_units = DimArray(result, units.m)
```

### Debugging JIT Compilation

Enable JAX debugging to inspect compilation:

```python
import os
os.environ['JAX_DISABLE_JIT'] = '1'  # Disable JIT for debugging

# Now JIT functions execute eagerly
result = my_jit_function(data)

# Re-enable when done
os.environ['JAX_DISABLE_JIT'] = '0'
```

## Further Reading

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX Pytrees Guide](https://jax.readthedocs.io/en/latest/pytrees.html)
- [dimtensor Operations Guide](operations.md)
- [dimtensor Examples](examples.md)

!!! tip "Performance Testing"
    Always profile your code. JAX's compilation overhead means JIT may not help for small arrays or simple operations. Use `%timeit` in Jupyter or Python's `timeit` module to verify speedups.

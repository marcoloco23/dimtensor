# PyTorch Integration Guide

Physics-informed neural networks with automatic dimensional checking.

## Introduction

The `dimtensor.torch` module provides PyTorch tensors with physical units, enabling:

- **Automatic dimension checking** during neural network operations
- **Gradient flow** through unit-aware computations
- **Dimension-aware layers** (DimLinear, DimConv1d/2d)
- **Physics-informed loss functions** for conservation laws
- **Automatic non-dimensionalization** for better training

This guide shows how to build physics-informed neural networks (PINNs) that catch dimensional errors at training time while maintaining full PyTorch functionality including autograd, GPU acceleration, and standard optimizers.

## Prerequisites

Install dimtensor with PyTorch support:

```bash
pip install dimtensor[torch]
```

Basic imports used throughout this guide:

```python
import torch
import torch.nn as nn
from dimtensor import DimArray, units, Dimension
from dimtensor.torch import DimTensor, DimLinear, DimSequential
```

## DimTensor Basics

### Creating DimTensors

DimTensor wraps a `torch.Tensor` and tracks its physical units:

```python
from dimtensor.torch import DimTensor
from dimtensor import units

# From Python lists
position = DimTensor([1.0, 2.0, 3.0], units.m)

# From torch tensors
velocity = DimTensor(torch.tensor([10.0, 20.0]), units.m / units.s)

# With device and dtype
temperature = DimTensor(
    [300.0, 400.0, 500.0],
    units.K,
    dtype=torch.float64,
    device='cuda'  # or 'cpu', 'mps'
)

# With gradient tracking
x = DimTensor(torch.tensor([1.0, 2.0]), units.m, requires_grad=True)
```

### Arithmetic Operations

DimTensor enforces dimensional correctness:

```python
# Compatible operations
v = DimTensor([10.0], units.m / units.s)
t = DimTensor([5.0], units.s)
d = v * t  # Result: DimTensor with unit 'm'

# Addition/subtraction require same dimension
a = DimTensor([1.0], units.m)
b = DimTensor([2.0], units.m)
c = a + b  # OK: both meters

# This raises DimensionError:
# bad = a + v  # Can't add length to velocity!

# Multiplication and division combine dimensions
force = DimTensor([10.0], units.N)
mass = DimTensor([2.0], units.kg)
acceleration = force / mass  # m/s^2

print(acceleration.unit)  # m/s^2
```

### Autograd Support

Gradients flow through DimTensor operations:

```python
# Define computation with gradient tracking
x = DimTensor([2.0], units.m, requires_grad=True)
v = DimTensor([5.0], units.m / units.s)

# Compute distance
t = x / v  # time = distance / velocity
loss = (t.data ** 2).sum()  # Use .data for raw tensor

# Backpropagation
loss.backward()
print(x.grad)  # Gradient in original units

# Detach from computation graph
x_detached = x.detach()
```

### Device Management

Move DimTensors between devices while preserving units:

```python
# Create on CPU
x = DimTensor([1.0, 2.0, 3.0], units.m)

# Move to GPU
if torch.cuda.is_available():
    x_gpu = x.cuda()
    # or
    x_gpu = x.to('cuda')

# Back to CPU
x_cpu = x_gpu.cpu()

# Change dtype while preserving device
x_double = x.double()  # float64
x_half = x.half()      # float16
```

## Dimension-Aware Layers

### DimLinear: Linear Transformation

`DimLinear` performs linear transformations while tracking physical dimensions:

```python
from dimtensor.torch import DimLinear
from dimtensor import Dimension

# Define dimensions
L = Dimension(L=1)           # Length [m]
V = Dimension(L=1, T=-1)     # Velocity [m/s]
A = Dimension(L=1, T=-2)     # Acceleration [m/s^2]

# Layer that transforms position to velocity
layer = DimLinear(
    in_features=3,
    out_features=3,
    input_dim=L,      # Expects meters
    output_dim=V      # Outputs m/s
)

# Create input
position = DimTensor(torch.randn(32, 3), units.m)

# Forward pass
velocity = layer(position)
print(velocity.unit)  # m/s

# The weights implicitly carry dimension (V/L) = [1/s]
```

### DimConv1d and DimConv2d

Convolutional layers for spatial physics problems:

```python
from dimtensor.torch import DimConv1d, DimConv2d

# 1D convolution for time series (e.g., temperature sensor data)
conv1d = DimConv1d(
    in_channels=1,
    out_channels=8,
    kernel_size=5,
    input_dim=Dimension(Theta=1),   # Temperature
    output_dim=Dimension(Theta=1)   # Temperature
)

# Input: (batch, channels, length)
temp_signal = DimTensor(torch.randn(16, 1, 100), units.K)
features = conv1d(temp_signal)  # (16, 8, 96) in K

# 2D convolution for spatial fields (e.g., heat distribution)
conv2d = DimConv2d(
    in_channels=1,
    out_channels=16,
    kernel_size=3,
    input_dim=Dimension(Theta=1),   # Temperature
    output_dim=Dimension(Theta=1)   # Temperature
)

# Input: (batch, channels, height, width)
temp_field = DimTensor(torch.randn(8, 1, 64, 64), units.K)
feature_map = conv2d(temp_field)  # (8, 16, 62, 62) in K
```

### DimSequential: Chaining Layers

Build networks by chaining dimension-aware layers:

```python
from dimtensor.torch import DimSequential

# Define a physics-informed network
# Position -> Velocity -> Acceleration
model = DimSequential(
    DimLinear(10, 64, input_dim=L, output_dim=V),
    DimLinear(64, 64, input_dim=V, output_dim=V),
    DimLinear(64, 10, input_dim=V, output_dim=A)
)

# The chain validates dimensions at construction
print(f"Input: {model.input_dim}, Output: {model.output_dim}")

# Forward pass
x = DimTensor(torch.randn(32, 10), units.m)
y = model(x)  # Output has dimension [m/s^2]
```

### Mixing with Standard PyTorch Layers

DimTensors can be used with standard PyTorch modules:

```python
import torch.nn.functional as F

class PhysicsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim_layer = DimLinear(3, 32, input_dim=L, output_dim=V)
        self.dropout = nn.Dropout(0.2)
        self.output = DimLinear(32, 3, input_dim=V, output_dim=A)

    def forward(self, x):
        # x is a DimTensor
        x = self.dim_layer(x)      # DimTensor output
        x = F.relu(x.data)          # Use .data for standard ops
        x = self.dropout(x)
        x = DimTensor._from_tensor_and_unit(x, units.m / units.s)
        x = self.output(x)         # DimTensor output
        return x

model = PhysicsNet()
```

## Loss Functions

### DimMSELoss

Mean squared error with dimensional checking:

```python
from dimtensor.torch import DimMSELoss

loss_fn = DimMSELoss()

# Predictions and targets must have same dimension
pred = DimTensor(torch.randn(100, 3), units.m / units.s)
target = DimTensor(torch.randn(100, 3), units.m / units.s)

loss = loss_fn(pred, target)
print(loss.unit)  # m^2/s^2 (squared error)

# This would raise DimensionError:
# bad_target = DimTensor(torch.randn(100, 3), units.m)
# loss_fn(pred, bad_target)  # Can't compare velocity to position!
```

### DimL1Loss and DimHuberLoss

Alternative loss functions:

```python
from dimtensor.torch import DimL1Loss, DimHuberLoss

# L1 (MAE) loss
l1_loss = DimL1Loss()
mae = l1_loss(pred, target)  # Preserves dimension

# Huber loss (smooth L1)
huber = DimHuberLoss(delta=1.0)
smooth_loss = huber(pred, target)
```

### PhysicsLoss: Conservation Law Enforcement

Penalize violations of physical conservation laws:

```python
from dimtensor.torch import PhysicsLoss

physics_loss = PhysicsLoss(rtol=1e-6)

# Check energy conservation
E_initial = DimTensor(torch.tensor([100.0, 200.0]), units.J)
E_final = DimTensor(torch.tensor([99.9, 200.1]), units.J)

loss = physics_loss(E_initial, E_final)
# Dimensionless loss penalizing relative change
```

### CompositeLoss: Combining Data and Physics

Combine multiple loss terms:

```python
from dimtensor.torch import CompositeLoss, DimMSELoss, PhysicsLoss

composite = CompositeLoss(
    data_loss=DimMSELoss(),
    physics_losses={
        'energy': (PhysicsLoss(), 0.1),      # 10% weight
        'momentum': (PhysicsLoss(), 0.05)    # 5% weight
    }
)

# In training loop
pred = model(x)
target = y

# Compute conservation quantities
E_init, E_final = compute_energy(x, pred)
p_init, p_final = compute_momentum(x, pred)

loss = composite(
    pred, target,
    physics_terms={
        'energy': (E_init, E_final),
        'momentum': (p_init, p_final)
    }
)
```

## Normalization Layers

Normalization layers preserve physical dimensions:

### DimBatchNorm1d and DimBatchNorm2d

```python
from dimtensor.torch import DimBatchNorm1d, DimBatchNorm2d

# Batch norm for 1D features
bn1d = DimBatchNorm1d(
    num_features=10,
    dimension=Dimension(L=1, T=-1)  # Velocity
)

v = DimTensor(torch.randn(32, 10), units.m / units.s)
v_normalized = bn1d(v)  # Still has dimension m/s

# Batch norm for 2D spatial data
bn2d = DimBatchNorm2d(
    num_features=16,
    dimension=Dimension(Theta=1)  # Temperature
)

T = DimTensor(torch.randn(8, 16, 32, 32), units.K)
T_normalized = bn2d(T)  # Still in Kelvin
```

### DimLayerNorm

```python
from dimtensor.torch import DimLayerNorm

# Layer normalization
ln = DimLayerNorm(
    normalized_shape=[64],
    dimension=Dimension(M=1, L=2, T=-2)  # Energy
)

energy = DimTensor(torch.randn(32, 64), units.J)
energy_normalized = ln(energy)  # Still in Joules
```

### Why Normalization Preserves Dimensions

Normalization transforms data to zero mean and unit variance, but the *physical dimension* doesn't change. A normalized temperature is still a temperature, just scaled. The learnable affine parameters (γ, β) maintain the same dimension as the input.

## Non-dimensionalization

Neural networks train best with inputs scaled to [-1, 1] or [0, 1]. `DimScaler` automates this:

### Basic Usage

```python
from dimtensor import DimArray
from dimtensor.torch import DimScaler

# Create scaler
scaler = DimScaler(method='characteristic')

# Fit on training data
velocities = DimArray([10, 100, 1000], units.m / units.s)
temperatures = DimArray([300, 400, 500], units.K)

scaler.fit(velocities, temperatures)

# Transform to dimensionless tensors
v_scaled = scaler.transform(velocities)      # torch.Tensor
T_scaled = scaler.transform(temperatures)    # torch.Tensor

print(v_scaled)  # Values in [-1, 1] range

# Train neural network with scaled data
model = nn.Sequential(
    nn.Linear(3, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
)

# Inverse transform predictions back to physical units
output_scaled = model(v_scaled)
v_pred = scaler.inverse_transform(output_scaled, units.m / units.s)
print(v_pred.unit)  # m/s
```

### Scaling Methods

Three methods are available:

```python
# 1. Characteristic scaling: divide by max absolute value
scaler_char = DimScaler(method='characteristic')
# Output range: [-1, 1]

# 2. Standard scaling: (x - mean) / std (z-score)
scaler_std = DimScaler(method='standard')
# Output: mean=0, std=1

# 3. Min-max scaling: scale to [0, 1]
scaler_minmax = DimScaler(method='minmax')
# Output range: [0, 1]
```

### MultiScaler for Complex Problems

Manage scaling for multiple physical quantities:

```python
from dimtensor.torch import MultiScaler

scaler = MultiScaler(method='characteristic')

# Add quantities with their training data
scaler.add('position', position_train)
scaler.add('velocity', velocity_train)
scaler.add('temperature', temperature_train)

# Transform
x_scaled = scaler.transform('position', x)
v_scaled = scaler.transform('velocity', v)
T_scaled = scaler.transform('temperature', T)

# Inverse transform by name
x_pred = scaler.inverse_transform('position', output, units.m)
```

## Complete PINN Example

Let's build a physics-informed neural network to solve the 1D heat equation:

∂T/∂t = α ∂²T/∂x²

where T is temperature, t is time, x is position, and α is thermal diffusivity.

### Problem Setup

```python
import torch
import torch.nn as nn
from dimtensor import DimArray, units, Dimension
from dimtensor.torch import (
    DimTensor, DimLinear, DimScaler,
    DimMSELoss, PhysicsLoss, CompositeLoss
)

# Physical parameters
L = 1.0  # Domain length [m]
T_left = 100.0  # Left boundary temperature [K]
T_right = 300.0  # Right boundary temperature [K]
alpha = 1e-4  # Thermal diffusivity [m^2/s]
t_max = 100.0  # Simulation time [s]

# Create training data
n_points = 1000
x_data = DimArray(torch.rand(n_points, 1).numpy() * L, units.m)
t_data = DimArray(torch.rand(n_points, 1).numpy() * t_max, units.s)

# Boundary condition points
n_bc = 100
x_bc_left = DimArray(torch.zeros(n_bc, 1).numpy(), units.m)
x_bc_right = DimArray(torch.ones(n_bc, 1).numpy() * L, units.m)
t_bc = DimArray(torch.rand(n_bc, 1).numpy() * t_max, units.s)
```

### Non-dimensionalize Data

```python
# Scale inputs to [-1, 1]
scaler = MultiScaler(method='characteristic')
scaler.add('position', x_data)
scaler.add('time', t_data)

# We'll predict temperature, add reference scale
T_ref = DimArray([T_left, T_right], units.K)
scaler.add('temperature', T_ref)

# Transform training data
x_scaled = scaler.transform('position', x_data)
t_scaled = scaler.transform('time', t_data)

# Boundary conditions
x_bc_left_scaled = scaler.transform('position', x_bc_left)
x_bc_right_scaled = scaler.transform('position', x_bc_right)
t_bc_scaled = scaler.transform('time', t_bc)
```

### Define Physics-Informed Network

```python
class HeatPINN(nn.Module):
    """Physics-informed neural network for heat equation."""

    def __init__(self, hidden_dim=64):
        super().__init__()

        # Network: (x, t) -> T
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, t):
        """Forward pass.

        Args:
            x, t: Scaled dimensionless tensors

        Returns:
            T: Scaled dimensionless temperature
        """
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

    def pde_residual(self, x, t, T):
        """Compute PDE residual: ∂T/∂t - α ∂²T/∂x²

        All inputs are dimensionless (scaled).
        """
        # Compute gradients (automatic differentiation)
        dT_dt = torch.autograd.grad(
            T, t,
            grad_outputs=torch.ones_like(T),
            create_graph=True
        )[0]

        dT_dx = torch.autograd.grad(
            T, x,
            grad_outputs=torch.ones_like(T),
            create_graph=True
        )[0]

        d2T_dx2 = torch.autograd.grad(
            dT_dx, x,
            grad_outputs=torch.ones_like(dT_dx),
            create_graph=True
        )[0]

        # Scale alpha to dimensionless form
        # In scaled coordinates: α_scaled = α * (t_scale / x_scale^2)
        x_scale = scaler.get_scaler('position').get_scale(Dimension(L=1))
        t_scale = scaler.get_scaler('time').get_scale(Dimension(T=1))
        alpha_scaled = alpha * (t_scale / x_scale**2)

        # PDE residual (should be zero)
        residual = dT_dt - alpha_scaled * d2T_dx2
        return residual

model = HeatPINN(hidden_dim=64)
```

### Training Loop

```python
# Loss functions
mse_loss = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Boundary condition targets (scaled)
T_left_scaled = scaler.transform('temperature',
                                  DimArray([T_left], units.K))
T_right_scaled = scaler.transform('temperature',
                                   DimArray([T_right], units.K))

# Training
n_epochs = 5000
for epoch in range(n_epochs):
    optimizer.zero_grad()

    # Enable gradient tracking for inputs (needed for PDE)
    x_train = x_scaled.clone().requires_grad_(True)
    t_train = t_scaled.clone().requires_grad_(True)

    # Forward pass
    T_pred = model(x_train, t_train)

    # PDE residual loss (should be zero everywhere)
    pde_residual = model.pde_residual(x_train, t_train, T_pred)
    pde_loss = torch.mean(pde_residual ** 2)

    # Boundary condition loss (left boundary)
    T_bc_left_pred = model(x_bc_left_scaled, t_bc_scaled)
    bc_left_loss = mse_loss(
        T_bc_left_pred,
        T_left_scaled.expand_as(T_bc_left_pred)
    )

    # Boundary condition loss (right boundary)
    T_bc_right_pred = model(x_bc_right_scaled, t_bc_scaled)
    bc_right_loss = mse_loss(
        T_bc_right_pred,
        T_right_scaled.expand_as(T_bc_right_pred)
    )

    # Total loss (weighted combination)
    loss = pde_loss + 10.0 * (bc_left_loss + bc_right_loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, "
              f"PDE = {pde_loss.item():.6f}, "
              f"BC = {(bc_left_loss + bc_right_loss).item():.6f}")
```

### Evaluate and Visualize

```python
import matplotlib.pyplot as plt

# Create evaluation grid
x_eval = torch.linspace(0, 1, 100).reshape(-1, 1)  # Scaled to [0, 1]
t_eval = torch.linspace(0, 1, 50).reshape(-1, 1)   # Scaled to [0, 1]

# Evaluate model
model.eval()
with torch.no_grad():
    T_results = []
    for t_val in t_eval:
        t_repeated = t_val.expand_as(x_eval)
        T_pred_scaled = model(x_eval, t_repeated)

        # Inverse transform to physical units
        T_pred = scaler.inverse_transform(
            'temperature',
            T_pred_scaled.squeeze(),
            units.K
        )
        T_results.append(T_pred.data.numpy())

# Convert back to physical coordinates
x_physical = scaler.inverse_transform(
    'position', x_eval, units.m
).data.numpy()
t_physical = scaler.inverse_transform(
    'time', t_eval, units.s
).data.numpy()

# Plot temperature evolution
plt.figure(figsize=(10, 6))
for i in [0, len(t_eval)//4, len(t_eval)//2, -1]:
    plt.plot(x_physical, T_results[i],
             label=f't = {t_physical[i, 0]:.1f} s')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.title('Heat Equation Solution')
plt.legend()
plt.grid(True)
plt.show()
```

### Validation

Verify physical correctness:

```python
# Check boundary conditions
T_at_left = scaler.inverse_transform(
    'temperature',
    model(torch.zeros(1, 1), torch.tensor([[0.5]])),
    units.K
)
print(f"T at x=0: {T_at_left.data.item():.1f} K (expected {T_left:.1f} K)")

T_at_right = scaler.inverse_transform(
    'temperature',
    model(torch.ones(1, 1), torch.tensor([[0.5]])),
    units.K
)
print(f"T at x=L: {T_at_right.data.item():.1f} K (expected {T_right:.1f} K)")

# Check steady-state (should be linear at large t)
t_steady = torch.ones(100, 1)
T_steady = []
for x_val in x_eval:
    x_repeated = x_val.expand(1, 1)
    T_pred_scaled = model(x_repeated, torch.ones(1, 1))
    T_pred = scaler.inverse_transform(
        'temperature', T_pred_scaled.squeeze(), units.K
    )
    T_steady.append(T_pred.data.item())

# Should be approximately linear
print(f"Steady-state temperature gradient: "
      f"{(T_steady[-1] - T_steady[0]) / L:.1f} K/m")
```

## Best Practices

### When to Use Dimension Checking

**Use DimTensor for:**
- Physics calculations with mixed units
- Model inputs/outputs that represent physical quantities
- Loss functions comparing physical predictions

**Use raw tensors for:**
- Pure numerical operations (activations, normalization parameters)
- Dimensionless intermediate representations
- Performance-critical inner loops

### Performance Considerations

```python
# DimTensor adds minimal overhead, but for best performance:

# 1. Use .data for inner computations
x = DimTensor(torch.randn(1000, 10), units.m)
for layer in layers:
    x_data = layer(x.data)  # Raw tensor through PyTorch layers
    x = DimTensor._from_tensor_and_unit(x_data, output_unit)

# 2. Batch dimension checking at model boundaries
class PhysicsModel(nn.Module):
    def forward(self, x: DimTensor) -> DimTensor:
        # Check dimension once at input
        assert x.dimension == self.input_dim

        # Internal computation with raw tensors
        h = self.layers(x.data)

        # Restore dimension at output
        return DimTensor._from_tensor_and_unit(h, self.output_unit)
```

### Common Pitfalls

```python
# 1. Forgetting to enable gradients for PINN inputs
x = torch.tensor([[1.0, 2.0]])  # BAD: no gradients
x = torch.tensor([[1.0, 2.0]], requires_grad=True)  # GOOD

# 2. Mixing scaled and unscaled data
x_scaled = scaler.transform(x_physical)
# Don't forget to scale ALL inputs consistently!

# 3. Using wrong dimension for loss
loss = mse_loss(pred, target)
# loss is a DimTensor! Use .data for optimizer:
loss.data.backward()  # Extract raw tensor first

# 4. Not detaching when needed
T_for_plotting = model(x).detach()  # Don't accumulate gradients
```

### Unit Testing Physics Models

```python
def test_energy_conservation():
    """Ensure model conserves energy."""
    model = PhysicsModel()

    x = DimTensor(torch.randn(10, 3), units.m)
    v = DimTensor(torch.randn(10, 3), units.m / units.s)

    # Compute initial energy
    E_initial = 0.5 * mass * (v ** 2).sum()

    # Evolve system
    x_new, v_new = model(x, v)

    # Compute final energy
    E_final = 0.5 * mass * (v_new ** 2).sum()

    # Check conservation
    relative_error = ((E_final - E_initial) / E_initial).abs()
    assert relative_error < 1e-4, f"Energy not conserved: {relative_error}"
```

## Tips and Tricks

### Converting Between DimTensor and Raw Tensors

```python
# DimTensor -> raw tensor (for PyTorch ops)
dim_tensor = DimTensor([1, 2, 3], units.m)
raw_tensor = dim_tensor.data  # or .magnitude()

# Raw tensor -> DimTensor (after computation)
result_tensor = torch.sin(raw_tensor)
result_dim = DimTensor._from_tensor_and_unit(
    result_tensor,
    dimensionless  # sin produces dimensionless output
)
```

### Debugging Dimension Errors

```python
# Print dimensions at each step
print(f"Input: {x.dimension}")
print(f"After layer 1: {h1.dimension}")
print(f"After layer 2: {h2.dimension}")

# Use Python debugger
import pdb; pdb.set_trace()

# Check dimension compatibility before operations
if x.dimension != expected_dim:
    print(f"Warning: Expected {expected_dim}, got {x.dimension}")
```

### Integration with Existing PyTorch Code

```python
# Wrap an existing model
class DimWrapper(nn.Module):
    def __init__(self, base_model, input_unit, output_unit):
        super().__init__()
        self.model = base_model
        self.input_unit = input_unit
        self.output_unit = output_unit

    def forward(self, x: DimTensor) -> DimTensor:
        # Validate input
        if x.unit != self.input_unit:
            x = x.to_unit(self.input_unit)

        # Run base model
        output = self.model(x.data)

        # Attach output unit
        return DimTensor._from_tensor_and_unit(output, self.output_unit)

# Use it
base_model = torch.load('pretrained_model.pt')
dim_model = DimWrapper(base_model, units.m, units.m / units.s)
```

### Working with Multiple Physics Domains

```python
# Electromagnetic + Mechanical system
E_field = DimTensor(torch.randn(10, 3), units.V / units.m)
charge = DimTensor([1e-6], units.C)
force_em = charge * E_field  # Electrostatic force [N]

velocity = DimTensor(torch.randn(10, 3), units.m / units.s)
mass = DimTensor([0.1], units.kg)
momentum = mass * velocity  # Momentum [kg⋅m/s]

# Combine in a loss function
loss_force = mse_loss(force_pred, force_em)
loss_momentum = mse_loss(momentum_pred, momentum)

# Losses have different dimensions! Normalize by characteristic scales:
loss = (loss_force.data / force_scale**2 +
        loss_momentum.data / momentum_scale**2)
```

### GPU Acceleration

```python
# Move model and data to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = PhysicsModel().to(device)

# DimTensor automatically follows device
x = DimTensor(torch.randn(1000, 10), units.m).to(device)
y = model(x)  # Computed on GPU

print(f"Result on {y.device}")  # cuda:0
```

---

This guide covers the essential PyTorch integration features of dimtensor. For more examples, see the [Examples Guide](examples.md) and the [API Reference](../api/torch.md).

For questions or issues, visit the [GitHub repository](https://github.com/yourusername/dimtensor).

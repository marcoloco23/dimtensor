# Physics-Informed Machine Learning Guide

Build neural networks that respect the laws of physics.

## Introduction

Physics-informed neural networks (PINNs) combine data-driven learning with physical laws encoded as differential equations and conservation principles. dimtensor makes building PINNs easier by:

- **Catching dimension errors** during training (can't predict velocity from temperature!)
- **Enforcing conservation laws** through specialized loss functions
- **Providing equation databases** for automatic constraint generation
- **Automating non-dimensionalization** for stable training
- **Tracking physical units** through the entire pipeline

This guide shows how to build PINNs that are both accurate and physically consistent.

## Prerequisites

Install dimtensor with PyTorch support:

```bash
pip install dimtensor[torch]
```

Basic imports:

```python
import torch
import torch.nn as nn
from dimtensor import DimArray, units, Dimension
from dimtensor.torch import (
    DimTensor, DimLinear, DimConv1d, DimSequential,
    DimMSELoss, PhysicsLoss, CompositeLoss, DimScaler
)
```

## Simple Example: Pendulum Dynamics

Let's start with a simple pendulum: predict angle θ(t) from time t.

### Problem Setup

We want a network that learns the pendulum equation: d²θ/dt² = -(g/L)sin(θ)

```python
import torch
from dimtensor import units, Dimension
from dimtensor.torch import DimTensor, DimLinear, DimMSELoss
import numpy as np

# Generate training data: simple harmonic approximation
g = 9.8  # m/s^2
L = 1.0  # m
omega = np.sqrt(g / L)  # angular frequency

t_data = np.linspace(0, 10, 100)  # 10 seconds
theta_data = 0.2 * np.cos(omega * t_data)  # 0.2 radian amplitude

# Create DimTensors
t = DimTensor(torch.tensor(t_data, dtype=torch.float32).reshape(-1, 1), units.s)
theta_target = DimTensor(
    torch.tensor(theta_data, dtype=torch.float32).reshape(-1, 1),
    units.rad
)
```

### Define Dimensional Network

The network transforms time [s] to angle [rad]:

```python
# Define dimensions
T_dim = Dimension(time=1)           # Time [s]
ANGLE_dim = Dimension()             # Angle [rad] - dimensionless

# Build network: time -> angle
model = nn.Sequential(
    DimLinear(1, 32, input_dim=T_dim, output_dim=ANGLE_dim),
    nn.Tanh(),  # Activation doesn't change dimension
    DimLinear(32, 32, input_dim=ANGLE_dim, output_dim=ANGLE_dim),
    nn.Tanh(),
    DimLinear(32, 1, input_dim=ANGLE_dim, output_dim=ANGLE_dim)
)
```

### Train with Dimensional Loss

```python
# Loss function checks dimension compatibility
loss_fn = DimMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()

    # Forward pass
    theta_pred = model(t)

    # Compute loss (dimensions checked automatically)
    loss = loss_fn(theta_pred, theta_target)

    # Backward pass (extract raw tensor for autograd)
    loss.data.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data.item():.6f}")

# Evaluate
model.eval()
with torch.no_grad():
    t_test = DimTensor(torch.linspace(0, 15, 200).reshape(-1, 1), units.s)
    theta_pred = model(t_test)
    print(f"Prediction at t=15s: {theta_pred.data[-1, 0].item():.4f} rad")
```

This basic example shows dimensional checking at work. Now let's add physics constraints.

## Dimensional Layers in Detail

### DimLinear: The Foundation

`DimLinear` is like `nn.Linear` but tracks physical dimensions:

```python
from dimtensor.torch import DimLinear
from dimtensor import Dimension

# Define physical dimensions
L = Dimension(length=1)              # Position [m]
V = Dimension(length=1, time=-1)     # Velocity [m/s]
A = Dimension(length=1, time=-2)     # Acceleration [m/s²]

# Layer that computes velocity from position
# Implicitly, weights have dimension [1/s]
layer = DimLinear(
    in_features=3,
    out_features=3,
    input_dim=L,
    output_dim=V
)

# Input: positions
x = DimTensor(torch.randn(32, 3), units.m)

# Output: velocities (dimension enforced)
v = layer(x)
print(f"Output dimension: {v.dimension}")  # L=1, T=-1 (m/s)

# This would raise DimensionError:
# bad_input = DimTensor(torch.randn(32, 3), units.K)  # temperature!
# layer(bad_input)  # ERROR: Expected dimension L=1, got Theta=1
```

### DimConv1d: Time Series Physics

Process temporal physics data while preserving dimensions:

```python
from dimtensor.torch import DimConv1d

# Temperature sensor data over time
# Input: (batch, channels, time_steps)
conv = DimConv1d(
    in_channels=1,
    out_channels=16,
    kernel_size=5,
    input_dim=Dimension(temperature=1),
    output_dim=Dimension(temperature=1)
)

# Temperature time series: 32 samples, 1 channel, 100 time steps
temp_signal = DimTensor(
    torch.randn(32, 1, 100) * 300 + 273,  # Around room temp
    units.K
)

# Extract features (still in Kelvin!)
features = conv(temp_signal)
print(f"Feature shape: {features.shape}")  # (32, 16, 96)
print(f"Feature unit: {features.unit}")     # K
```

### DimConv2d: Spatial Physics Fields

For 2D spatial problems like heat distribution:

```python
from dimtensor.torch import DimConv2d

# Process temperature field on a 2D surface
conv2d = DimConv2d(
    in_channels=1,
    out_channels=32,
    kernel_size=3,
    padding=1,
    input_dim=Dimension(temperature=1),
    output_dim=Dimension(temperature=1)
)

# 2D temperature field: (batch, channels, height, width)
T_field = DimTensor(torch.randn(16, 1, 64, 64) * 50 + 300, units.K)

# Convolve (output maintains temperature dimension)
T_features = conv2d(T_field)
print(f"Output shape: {T_features.shape}")      # (16, 32, 64, 64)
print(f"Output dimension: {T_features.dimension}")  # Theta=1
```

### DimSequential: Chaining Layers

Build physics-aware pipelines:

```python
from dimtensor.torch import DimSequential

# Multi-step transformation: Position -> Velocity -> Acceleration
physics_net = DimSequential(
    DimLinear(10, 64, input_dim=L, output_dim=V),
    DimLinear(64, 64, input_dim=V, output_dim=V),
    DimLinear(64, 10, input_dim=V, output_dim=A)
)

# Dimensions validated at construction
print(f"Network: {physics_net.input_dim} -> {physics_net.output_dim}")

# Forward pass
position = DimTensor(torch.randn(100, 10), units.m)
acceleration = physics_net(position)
print(f"Output: {acceleration.unit}")  # m/s²
```

## Loss Functions for Physics

### Data Fidelity Losses

Standard losses with dimension checking:

```python
from dimtensor.torch import DimMSELoss, DimL1Loss, DimHuberLoss

# Mean squared error
mse = DimMSELoss()
pred = DimTensor(torch.randn(100, 3), units.m / units.s)
target = DimTensor(torch.randn(100, 3), units.m / units.s)
loss_mse = mse(pred, target)
print(f"MSE loss dimension: {loss_mse.unit}")  # (m/s)² = m²/s²

# Mean absolute error
mae = DimL1Loss()
loss_mae = mae(pred, target)
print(f"MAE loss dimension: {loss_mae.unit}")  # m/s

# Huber loss (smooth L1)
huber = DimHuberLoss(delta=1.0)
loss_huber = huber(pred, target)
```

### PhysicsLoss: Conservation Law Enforcement

Penalize violations of conservation laws:

```python
from dimtensor.torch import PhysicsLoss

# Create physics loss with 1e-6 relative tolerance
physics_loss = PhysicsLoss(rtol=1e-6)

# Example: Energy conservation
# Initial kinetic energy
m = DimTensor([2.0], units.kg)
v_initial = DimTensor(torch.tensor([[10.0, 0.0, 0.0]]), units.m / units.s)
E_initial = 0.5 * m * (v_initial ** 2).sum()

# Final kinetic energy (after network prediction)
v_final = DimTensor(torch.tensor([[9.8, 1.0, 0.5]]), units.m / units.s)
E_final = 0.5 * m * (v_final ** 2).sum()

# Compute conservation loss (dimensionless)
conservation_loss = physics_loss(E_initial, E_final)
print(f"Energy conservation loss: {conservation_loss.item():.8f}")
```

### CompositeLoss: Combining Data + Physics

The key to physics-informed learning:

```python
from dimtensor.torch import CompositeLoss

# Combine MSE loss with conservation constraints
composite = CompositeLoss(
    data_loss=DimMSELoss(),
    physics_losses={
        'energy': (PhysicsLoss(rtol=1e-4), 0.1),    # 10% weight
        'momentum': (PhysicsLoss(rtol=1e-4), 0.05), # 5% weight
    }
)

# In training loop:
def compute_energies(positions, velocities, mass):
    """Helper to compute kinetic energy."""
    KE = 0.5 * mass * (velocities ** 2).sum(dim=-1, keepdim=True)
    return KE

def compute_momentum(velocities, mass):
    """Helper to compute momentum."""
    return mass * velocities

# Training step
x = DimTensor(torch.randn(32, 3, requires_grad=True), units.m)
v_target = DimTensor(torch.randn(32, 3), units.m / units.s)
mass = DimTensor([1.0], units.kg)

# Network prediction
v_pred = model(x)

# Initial state (from input)
E_initial = compute_energies(x, v_target, mass)
p_initial = compute_momentum(v_target, mass)

# Final state (from prediction)
E_final = compute_energies(x, v_pred, mass)
p_final = compute_momentum(v_pred, mass)

# Total loss combines data fidelity + physics
total_loss = composite(
    pred=v_pred,
    target=v_target,
    physics_terms={
        'energy': (E_initial, E_final),
        'momentum': (p_initial, p_final)
    }
)
```

## Non-dimensionalization with DimScaler

Neural networks train best with inputs in [-1, 1]. `DimScaler` automates this:

### Basic Scaling

```python
from dimtensor import DimArray
from dimtensor.torch import DimScaler

# Create training data with large range
velocities = DimArray([1.0, 100.0, 1000.0], units.m / units.s)
temperatures = DimArray([273.0, 373.0, 573.0], units.K)

# Create and fit scaler
scaler = DimScaler(method='characteristic')  # Divide by max value
scaler.fit(velocities, temperatures)

# Transform to dimensionless tensors
v_scaled = scaler.transform(velocities)
T_scaled = scaler.transform(temperatures)

print(f"Original velocities: {velocities.data}")
print(f"Scaled velocities: {v_scaled}")  # Values in [-1, 1]

# Train network with scaled data...
# model = train(v_scaled, T_scaled)

# Inverse transform predictions back to physical units
v_pred_scaled = torch.tensor([[0.5, -0.3, 0.8]])
v_pred = scaler.inverse_transform(v_pred_scaled, units.m / units.s)
print(f"Predicted velocity: {v_pred.data} {v_pred.unit}")
```

### Scaling Methods

Three methods available:

```python
# 1. Characteristic scaling: x' = x / max(|x|)
#    Output range: [-1, 1]
scaler_char = DimScaler(method='characteristic')

# 2. Standard scaling: x' = (x - mean) / std
#    Output: mean=0, std=1 (z-score normalization)
scaler_std = DimScaler(method='standard')

# 3. Min-max scaling: x' = (x - min) / (max - min)
#    Output range: [0, 1]
scaler_minmax = DimScaler(method='minmax')

# Example comparison
data = DimArray([10, 20, 30, 40, 50], units.m)

scaler_char.fit(data)
print(f"Characteristic: {scaler_char.transform(data)}")  # [-1, ..., 1]

scaler_std.fit(data)
print(f"Standard: {scaler_std.transform(data)}")  # ~[-1.4, ..., 1.4]

scaler_minmax.fit(data)
print(f"MinMax: {scaler_minmax.transform(data)}")  # [0, ..., 1]
```

### MultiScaler for Complex Problems

Manage many physical quantities:

```python
from dimtensor.torch import MultiScaler

# Create multi-quantity scaler
scaler = MultiScaler(method='characteristic')

# Add quantities (fit on training data)
scaler.add('position', DimArray([0, 10, 20], units.m))
scaler.add('velocity', DimArray([0, 5, 10], units.m / units.s))
scaler.add('temperature', DimArray([273, 373, 473], units.K))

# Transform by name
x = DimArray([5.0], units.m)
x_scaled = scaler.transform('position', x)

v = DimArray([2.5], units.m / units.s)
v_scaled = scaler.transform('velocity', v)

# Inverse transform by name
output_scaled = torch.tensor([[0.5]])
x_pred = scaler.inverse_transform('position', output_scaled, units.m)
print(f"Predicted position: {x_pred.data[0, 0]:.1f} m")
```

## Using the Equation Database

dimtensor includes 67 physics equations across 10 domains. Use them to validate models and generate constraints.

### Searching for Equations

```python
from dimtensor.equations import search_equations, get_equation, list_domains

# Search for equations about energy
energy_eqs = search_equations("energy")
print(f"Found {len(energy_eqs)} equations about energy:")
for eq in energy_eqs[:3]:
    print(f"  - {eq.name}: {eq.formula}")

# Search for heat-related equations
heat_eqs = search_equations("heat")
for eq in heat_eqs:
    print(f"  - {eq.name} ({eq.domain}): {eq.formula}")

# List all available domains
domains = list_domains()
print(f"\nAvailable domains: {domains}")
```

### Getting Specific Equations

```python
# Get Newton's Second Law
newton = get_equation("Newton's Second Law")
print(f"Name: {newton.name}")
print(f"Formula: {newton.formula}")
print(f"Variables: {newton.variables}")
print(f"LaTeX: {newton.latex}")

# Variables have dimension metadata
for var, dim in newton.variables.items():
    print(f"  {var}: {dim}")
```

### Using Equations for Validation

Validate that your network outputs have correct dimensions:

```python
from dimtensor.equations import get_equation
from dimtensor import DimArray, units

# Get kinetic energy equation
ke_eq = get_equation("Kinetic Energy")

# Check our calculation matches expected dimension
m = DimArray([2.0], units.kg)
v = DimArray([10.0], units.m / units.s)
KE_computed = 0.5 * m * v**2

# Validate dimension
expected_dim = ke_eq.variables['KE']
actual_dim = KE_computed.dimension

assert actual_dim == expected_dim, \
    f"Dimension mismatch! Expected {expected_dim}, got {actual_dim}"

print(f"✓ Kinetic energy calculation is dimensionally correct")
print(f"  KE = {KE_computed.to(units.J)}")
```

### Browsing by Domain

```python
from dimtensor.equations import get_equations

# Get all thermodynamics equations
thermo_eqs = get_equations(domain="thermodynamics")
print(f"Thermodynamics equations ({len(thermo_eqs)}):")
for eq in thermo_eqs:
    print(f"  - {eq.name}: {eq.formula}")

# Get equations with specific tags
fundamental_eqs = get_equations(tags=["fundamental"])
print(f"\nFundamental equations: {len(fundamental_eqs)}")
```

## Using the Dataset Registry

dimtensor provides 10+ physics datasets with dimensional metadata.

### Listing Available Datasets

```python
from dimtensor.datasets import list_datasets, get_dataset_info

# List all datasets
all_datasets = list_datasets()
print(f"Available datasets: {len(all_datasets)}")
for ds in all_datasets[:5]:
    print(f"  - {ds.name} ({ds.domain}): {ds.description}")

# Filter by domain
mechanics_datasets = list_datasets(domain="mechanics")
print(f"\nMechanics datasets: {len(mechanics_datasets)}")
for ds in mechanics_datasets:
    print(f"  - {ds.name}: {ds.description}")

# Get dataset metadata
info = get_dataset_info("pendulum")
print(f"\nDataset: {info.name}")
print(f"Description: {info.description}")
print(f"Features: {info.features}")
print(f"Targets: {info.targets}")
```

### Dataset Dimensions

Each dataset includes dimensional metadata for features and targets:

```python
# Inspect heat diffusion dataset
heat_info = get_dataset_info("heat_diffusion")

print(f"Heat Diffusion Dataset:")
print(f"  Features:")
for name, dim in heat_info.features.items():
    print(f"    {name}: {dim}")

print(f"  Targets:")
for name, dim in heat_info.targets.items():
    print(f"    {name}: {dim}")

# Use dimensions to build network architecture
# Input: position (L=1) + time (T=1) -> concatenate
# Output: temperature (Theta=1)
```

### Loading Real Datasets

Some datasets have actual data loaders:

```python
from dimtensor.datasets import load_dataset

# Load NIST physical constants
# constants = load_dataset("nist_codata_2022")
# print(constants['speed of light in vacuum'])

# Load NASA exoplanet data
# exoplanets = load_dataset("nasa_exoplanets")
# print(f"Loaded {len(exoplanets)} exoplanets")

# Note: Real datasets require internet connection and may cache data locally
```

## Complete Example: 1D Burgers Equation PINN

Let's solve the viscous Burgers equation, a nonlinear PDE important in fluid dynamics:

∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²

where u(x,t) is velocity, and ν is viscosity.

### Problem Definition

```python
import torch
import torch.nn as nn
from dimtensor import units, Dimension
from dimtensor.torch import DimTensor, DimScaler
import numpy as np

# Physical parameters
nu = 0.01  # Viscosity [m²/s]
L = 1.0    # Domain length [m]
T = 1.0    # Simulation time [s]

# Boundary conditions
u_left = 0.0   # Left boundary: u(0, t) = 0
u_right = 0.0  # Right boundary: u(L, t) = 0

# Initial condition: sine wave
def u_initial(x):
    return np.sin(np.pi * x / L)
```

### Generate Training Data

```python
# Collocation points for PDE (interior)
n_interior = 1000
x_interior = np.random.rand(n_interior, 1) * L
t_interior = np.random.rand(n_interior, 1) * T

# Boundary points
n_boundary = 100
x_left = np.zeros((n_boundary, 1))
x_right = np.ones((n_boundary, 1)) * L
t_boundary = np.random.rand(n_boundary, 1) * T

# Initial condition points
n_initial = 100
x_initial = np.random.rand(n_initial, 1) * L
t_initial = np.zeros((n_initial, 1))
u_initial_vals = u_initial(x_initial)

# Convert to tensors
x_int = torch.tensor(x_interior, dtype=torch.float32, requires_grad=True)
t_int = torch.tensor(t_interior, dtype=torch.float32, requires_grad=True)

x_bc_left = torch.tensor(x_left, dtype=torch.float32)
x_bc_right = torch.tensor(x_right, dtype=torch.float32)
t_bc = torch.tensor(t_boundary, dtype=torch.float32)

x_ic = torch.tensor(x_initial, dtype=torch.float32)
t_ic = torch.tensor(t_initial, dtype=torch.float32)
u_ic = torch.tensor(u_initial_vals, dtype=torch.float32)
```

### Define PINN Architecture

```python
class BurgersPINN(nn.Module):
    """Physics-informed neural network for Burgers equation."""

    def __init__(self, hidden_dim=64, num_layers=4):
        super().__init__()

        layers = []
        layers.append(nn.Linear(2, hidden_dim))  # Input: (x, t)
        layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1))  # Output: u

        self.net = nn.Sequential(*layers)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, t):
        """Predict velocity u(x, t).

        Args:
            x: Position tensor [m]
            t: Time tensor [s]

        Returns:
            u: Velocity tensor [m/s]
        """
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

    def pde_residual(self, x, t, nu):
        """Compute Burgers equation residual.

        Residual = ∂u/∂t + u·∂u/∂x - ν·∂²u/∂x²

        Should be zero everywhere in the domain.
        """
        # Enable gradient computation
        u = self.forward(x, t)

        # First derivatives
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        # Second derivative
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0]

        # Burgers equation residual
        residual = u_t + u * u_x - nu * u_xx

        return residual

# Create model
model = BurgersPINN(hidden_dim=64, num_layers=4)
```

### Training Loop

```python
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Loss weights
lambda_data = 1.0
lambda_pde = 1.0
lambda_bc = 10.0

# Training
n_epochs = 5000
for epoch in range(n_epochs):
    optimizer.zero_grad()

    # PDE loss (interior points)
    residual = model.pde_residual(x_int, t_int, nu)
    loss_pde = torch.mean(residual ** 2)

    # Boundary condition loss
    u_bc_left = model(x_bc_left, t_bc)
    u_bc_right = model(x_bc_right, t_bc)
    loss_bc = torch.mean(u_bc_left ** 2) + torch.mean(u_bc_right ** 2)

    # Initial condition loss
    u_pred_ic = model(x_ic, t_ic)
    loss_ic = torch.mean((u_pred_ic - u_ic) ** 2)

    # Total loss
    loss = lambda_pde * loss_pde + lambda_bc * (loss_bc + loss_ic)

    # Backward pass
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | "
              f"PDE: {loss_pde.item():.6f} | "
              f"BC: {loss_bc.item():.6f} | "
              f"IC: {loss_ic.item():.6f}")

print("\nTraining complete!")
```

### Evaluate and Visualize

```python
# Create evaluation grid
x_eval = np.linspace(0, L, 100)
t_eval = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

model.eval()
with torch.no_grad():
    for t_val in t_eval:
        x_test = torch.tensor(x_eval.reshape(-1, 1), dtype=torch.float32)
        t_test = torch.tensor(np.full_like(x_eval, t_val).reshape(-1, 1),
                              dtype=torch.float32)
        u_pred = model(x_test, t_test).numpy()

        print(f"\nSolution at t={t_val:.2f}s:")
        print(f"  u(0.5m) = {u_pred[50, 0]:.4f} m/s")
```

### Physical Validation

```python
# Check boundary conditions
with torch.no_grad():
    t_check = torch.tensor([[0.5]], dtype=torch.float32)

    # Left boundary
    x_left_check = torch.tensor([[0.0]], dtype=torch.float32)
    u_left = model(x_left_check, t_check)
    print(f"u(0, 0.5s) = {u_left.item():.6f} (should be ≈0)")

    # Right boundary
    x_right_check = torch.tensor([[L]], dtype=torch.float32)
    u_right = model(x_right_check, t_check)
    print(f"u({L}, 0.5s) = {u_right.item():.6f} (should be ≈0)")

    # Check PDE residual
    x_test = torch.tensor([[0.5]], dtype=torch.float32, requires_grad=True)
    t_test = torch.tensor([[0.5]], dtype=torch.float32, requires_grad=True)
    residual = model.pde_residual(x_test, t_test, nu)
    print(f"PDE residual at (0.5m, 0.5s) = {residual.item():.6e} (should be ≈0)")
```

## Conservation Laws in Practice

Enforce multiple conservation principles simultaneously.

### Energy Conservation Example

```python
import torch
import torch.nn as nn
from dimtensor import units, Dimension
from dimtensor.torch import DimTensor, PhysicsLoss, CompositeLoss, DimMSELoss

class ConservativeNet(nn.Module):
    """Network that predicts dynamics while conserving energy."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128),  # Input: (x, y, z, vx, vy, vz)
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 6)   # Output: (x', y', z', vx', vy', vz')
        )

    def forward(self, state):
        """Predict next state.

        Args:
            state: (position, velocity) as tensor [batch, 6]

        Returns:
            next_state: Predicted next state [batch, 6]
        """
        return self.net(state)

def compute_total_energy(state, mass):
    """Compute kinetic + potential energy.

    Args:
        state: Tensor [batch, 6] = (x, y, z, vx, vy, vz)
        mass: Mass [kg]

    Returns:
        Total energy [J]
    """
    position = state[:, :3]  # (x, y, z)
    velocity = state[:, 3:]  # (vx, vy, vz)

    # Kinetic energy
    KE = 0.5 * mass * (velocity ** 2).sum(dim=1, keepdim=True)

    # Potential energy (gravitational)
    g = 9.8  # m/s²
    PE = mass * g * position[:, 2:3]  # z-component only

    return KE + PE

# Training with energy conservation
model = ConservativeNet()
mass = 1.0  # kg

# Loss functions
data_loss = nn.MSELoss()
energy_loss = PhysicsLoss(rtol=1e-4)
composite = CompositeLoss(
    data_loss=data_loss,
    physics_losses={'energy': (energy_loss, 0.2)}  # 20% weight on conservation
)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1000):
    # Sample initial states
    state_init = torch.randn(32, 6)
    state_target = torch.randn(32, 6)  # From simulation/data

    # Predict next state
    state_pred = model(state_init)

    # Compute energies
    E_init = compute_total_energy(state_init, mass)
    E_pred = compute_total_energy(state_pred, mass)

    # Total loss
    loss = composite(
        pred=state_pred,
        target=state_target,
        physics_terms={'energy': (E_init, E_pred)}
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        with torch.no_grad():
            avg_violation = torch.abs((E_pred - E_init) / E_init).mean()
            print(f"Epoch {epoch}: Energy violation = {avg_violation.item():.6f}")
```

### Momentum Conservation

```python
def compute_momentum(state, mass):
    """Compute total momentum.

    Args:
        state: Tensor [batch, 6] = (x, y, z, vx, vy, vz)
        mass: Mass [kg]

    Returns:
        Momentum vector [batch, 3] in [kg⋅m/s]
    """
    velocity = state[:, 3:]  # (vx, vy, vz)
    return mass * velocity

# Add momentum conservation to composite loss
momentum_loss = PhysicsLoss(rtol=1e-4)
composite_full = CompositeLoss(
    data_loss=data_loss,
    physics_losses={
        'energy': (energy_loss, 0.2),
        'momentum': (momentum_loss, 0.1)
    }
)

# Training with both constraints
for epoch in range(1000):
    state_init = torch.randn(32, 6)
    state_target = torch.randn(32, 6)
    state_pred = model(state_init)

    # Energies
    E_init = compute_total_energy(state_init, mass)
    E_pred = compute_total_energy(state_pred, mass)

    # Momenta
    p_init = compute_momentum(state_init, mass)
    p_pred = compute_momentum(state_pred, mass)

    # Both energies and momenta are DimTensors with correct units
    # PhysicsLoss computes relative error for each conserved quantity

    loss = composite_full(
        pred=state_pred,
        target=state_target,
        physics_terms={
            'energy': (E_init, E_pred),
            'momentum': (p_init, p_pred)
        }
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Best Practices

### When to Use Dimensional Layers vs Standard PyTorch

**Use DimLinear/DimConv when:**
- Model inputs/outputs are physical quantities
- You want automatic dimension validation
- Building physics-informed architectures
- Prototyping scientific models

**Use standard nn.Linear when:**
- Working with dimensionless embeddings
- Internal hidden representations
- Performance is critical (minimal overhead with DimTensor, but raw tensors are fastest)
- Integrating with existing PyTorch models

**Hybrid approach** (recommended for large models):

```python
class HybridPhysicsNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Dimensional input/output layers
        self.input_layer = DimLinear(3, 64,
                                     input_dim=Dimension(length=1),
                                     output_dim=Dimension())

        # Standard PyTorch for internal computation
        self.hidden = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Dimensional output layer
        self.output_layer = DimLinear(64, 3,
                                      input_dim=Dimension(),
                                      output_dim=Dimension(length=1, time=-1))

    def forward(self, x):
        # x is DimTensor with length dimension
        h = self.input_layer(x)        # DimTensor -> dimensionless
        h = self.hidden(h.data)         # Use .data for standard layers
        h = DimTensor._from_tensor_and_unit(h, dimensionless)
        return self.output_layer(h)     # dimensionless -> velocity
```

### Debugging Dimension Errors

```python
# 1. Print dimensions at each step
def debug_forward(model, x):
    print(f"Input: {x.dimension if isinstance(x, DimTensor) else 'raw tensor'}")

    for i, layer in enumerate(model.layers):
        x = layer(x)
        print(f"After layer {i}: {x.dimension}")

    return x

# 2. Use try-except for informative errors
try:
    output = model(input_tensor)
except DimensionError as e:
    print(f"Dimension mismatch: {e}")
    print(f"Input dimension: {input_tensor.dimension}")
    print(f"Expected: {model.layers[0].input_dim}")

# 3. Validate intermediate results
def validate_physics(position, velocity, acceleration):
    """Check dimensional consistency of physics quantities."""
    assert position.dimension == Dimension(length=1)
    assert velocity.dimension == Dimension(length=1, time=-1)
    assert acceleration.dimension == Dimension(length=1, time=-2)

    # Also check magnitudes are reasonable
    assert (position.data.abs() < 1e6).all(), "Position too large!"
    assert (velocity.data.abs() < 1e4).all(), "Velocity too large!"
```

### Training Tips

```python
# 1. Always non-dimensionalize inputs
scaler = DimScaler(method='characteristic')
scaler.fit(train_positions, train_velocities, train_temperatures)

x_train = scaler.transform(positions)  # Now in [-1, 1]
v_train = scaler.transform(velocities)

# 2. Use learning rate scheduling
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=100
)

# 3. Balance loss terms
#    Start with pure data loss, gradually increase physics weight
epoch_schedule = {
    0: {'data': 1.0, 'physics': 0.0},
    1000: {'data': 1.0, 'physics': 0.1},
    2000: {'data': 1.0, 'physics': 0.5},
    3000: {'data': 1.0, 'physics': 1.0},
}

def get_loss_weights(epoch):
    for e in sorted(epoch_schedule.keys(), reverse=True):
        if epoch >= e:
            return epoch_schedule[e]
    return epoch_schedule[0]

# 4. Monitor PDE residuals during training
with torch.no_grad():
    x_test = torch.randn(100, 1, requires_grad=True)
    t_test = torch.randn(100, 1, requires_grad=True)
    residual = model.pde_residual(x_test, t_test, alpha)
    avg_residual = residual.abs().mean().item()
    print(f"Average PDE residual: {avg_residual:.6e}")
```

### Common Pitfalls

```python
# ❌ WRONG: Forgetting to enable gradients for PDE inputs
x = torch.tensor([[1.0]])  # No requires_grad!
residual = model.pde_residual(x, t)  # ERROR in autograd

# ✅ CORRECT:
x = torch.tensor([[1.0]], requires_grad=True)

# ❌ WRONG: Mixing scaled and unscaled data
x_scaled = scaler.transform(x)
y = model(x_scaled)  # Dimensionless output
loss = mse(y, y_target)  # y_target is NOT scaled! Wrong!

# ✅ CORRECT:
y_target_scaled = scaler.transform(y_target)
loss = mse(y, y_target_scaled)

# ❌ WRONG: Using dimensioned loss directly in backward()
loss = dim_mse_loss(pred, target)  # Returns DimTensor
loss.backward()  # ERROR: DimTensor has no backward()

# ✅ CORRECT:
loss = dim_mse_loss(pred, target)
loss.data.backward()  # Extract raw tensor first

# ❌ WRONG: Not detaching when evaluating
with torch.no_grad():
    pred = model(x)
    result = some_expensive_computation(pred)  # Builds computation graph anyway!

# ✅ CORRECT:
with torch.no_grad():
    pred = model(x).detach()  # Explicitly detach
    result = some_expensive_computation(pred)
```

### Performance Optimization

```python
# 1. Batch PDE residual computations
def compute_residuals_batched(model, x_batch, t_batch):
    """Compute PDE residuals for entire batch at once."""
    residuals = model.pde_residual(x_batch, t_batch, nu)
    return residuals

# 2. Use mixed precision training for large models
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(n_epochs):
    optimizer.zero_grad()

    with autocast():  # Automatic mixed precision
        pred = model(x)
        loss = loss_fn(pred, target)

    scaler.scale(loss.data).backward()
    scaler.step(optimizer)
    scaler.update()

# 3. Checkpoint gradients for very deep networks
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(model, x):
    return checkpoint(model, x)  # Saves memory at cost of recomputation
```

## Summary

You've learned how to:

- ✅ Build **dimension-aware neural networks** with DimLinear, DimConv
- ✅ Enforce **conservation laws** with PhysicsLoss and CompositeLoss
- ✅ **Non-dimensionalize** data automatically with DimScaler
- ✅ Search and use the **equation database** for validation
- ✅ Discover datasets with the **dataset registry**
- ✅ Build complete **PINNs** for PDEs like Burgers equation
- ✅ Debug dimension errors and optimize training

## Next Steps

- **[PyTorch Integration Guide](pytorch.md)** - More PyTorch-specific features
- **[Examples Guide](examples.md)** - Additional physics calculations
- **[API Reference](../api/torch.md)** - Full API documentation
- **[JAX Guide](jax.md)** - Use dimtensor with JAX for JIT and vmap

## Further Reading

**Papers on PINNs:**
- Raissi et al. (2019): "Physics-informed neural networks"
- Karniadakis et al. (2021): "Physics-informed machine learning"

**dimtensor Resources:**
- GitHub: [github.com/user/dimtensor](https://github.com/user/dimtensor)
- Documentation: Full API reference and tutorials
- Examples: More complete PINN implementations

---

*Happy physics-informed learning!* 🔬🧠

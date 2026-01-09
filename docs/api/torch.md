# PyTorch Integration

Dimension-aware neural network layers, loss functions, and utilities for physics-informed machine learning.

## Overview

The `dimtensor.torch` module provides:

- **DimTensor**: PyTorch tensor with physical dimensions and autograd support
- **Neural Network Layers**: Dimension-tracking versions of Linear, Conv1d, Conv2d
- **Loss Functions**: MSE, L1, Huber losses with dimensional checking
- **Normalization**: BatchNorm and LayerNorm preserving dimensions
- **Scaling**: Automatic non-dimensionalization for training

```python
from dimtensor.torch import DimTensor, DimLinear, DimMSELoss
from dimtensor import units, Dimension

# Create dimension-aware layer
layer = DimLinear(
    in_features=3, out_features=3,
    input_dim=Dimension(L=1),      # position [m]
    output_dim=Dimension(L=1, T=-1) # velocity [m/s]
)

# Forward pass tracks dimensions
x = DimTensor(torch.randn(10, 3), units.m)
v = layer(x)  # Dimension: m/s

# Loss functions check dimensional compatibility
loss_fn = DimMSELoss()
loss = loss_fn(v_pred, v_target)  # Error if dimensions don't match
```

## Core Classes

### DimTensor

::: dimtensor.torch.DimTensor
    options:
      members:
        - __init__
        - data
        - unit
        - dimension
        - shape
        - dtype
        - device
        - requires_grad
        - to
        - detach
        - numpy

## Neural Network Layers

### DimLayer

::: dimtensor.torch.DimLayer
    options:
      members:
        - __init__
        - forward
        - input_dim
        - output_dim
        - validate_input

### DimLinear

Linear layer with dimension tracking. Transforms input from one physical dimension to another.

```python
from dimtensor.torch import DimLinear
from dimtensor import Dimension

# Convert position [m] to velocity [m/s]
layer = DimLinear(
    in_features=3, out_features=3,
    input_dim=Dimension(L=1),
    output_dim=Dimension(L=1, T=-1)
)
```

::: dimtensor.torch.DimLinear
    options:
      members:
        - __init__
        - forward

### DimConv1d

1D convolution with dimension tracking. Useful for time series with physical units.

```python
from dimtensor.torch import DimConv1d
from dimtensor import units

# Process a temperature time series
conv = DimConv1d(
    in_channels=1, out_channels=16, kernel_size=5,
    input_dim=Dimension(Theta=1),
    output_dim=Dimension(Theta=1)
)
```

::: dimtensor.torch.DimConv1d

### DimConv2d

2D convolution with dimension tracking. For spatial fields with physical units.

```python
# Process a pressure field
conv = DimConv2d(
    in_channels=1, out_channels=32, kernel_size=3,
    input_dim=Dimension(M=1, L=-1, T=-2),  # pressure
    output_dim=Dimension(M=1, L=-1, T=-2)
)
```

::: dimtensor.torch.DimConv2d

### DimSequential

Sequential container for chaining dimension-aware layers. Validates that dimensions match between layers.

```python
from dimtensor.torch import DimSequential, DimLinear
from dimtensor import Dimension

L = Dimension(L=1)       # length
V = Dimension(L=1, T=-1) # velocity
A = Dimension(L=1, T=-2) # acceleration

model = DimSequential(
    DimLinear(3, 16, input_dim=L, output_dim=V),
    DimLinear(16, 3, input_dim=V, output_dim=A),
)  # Validates V matches between layers
```

::: dimtensor.torch.DimSequential

## Loss Functions

### DimMSELoss

Mean squared error with dimensional checking. Output has dimension of input^2.

```python
from dimtensor.torch import DimMSELoss

loss_fn = DimMSELoss()
loss = loss_fn(pred, target)  # Checks dimensions match
# If pred has dimension [m], loss has dimension [m^2]
```

::: dimtensor.torch.DimMSELoss

### DimL1Loss

Mean absolute error (L1) with dimensional checking. Output preserves input dimension.

::: dimtensor.torch.DimL1Loss

### DimHuberLoss

Huber loss (smooth L1) with dimensional checking. Quadratic for small errors, linear for large.

::: dimtensor.torch.DimHuberLoss

### PhysicsLoss

Loss function for enforcing conservation laws. Penalizes relative changes in conserved quantities.

```python
from dimtensor.torch import PhysicsLoss

physics_loss = PhysicsLoss(rtol=1e-6)
loss = physics_loss(E_initial, E_final)  # Penalize energy change
```

::: dimtensor.torch.PhysicsLoss

### CompositeLoss

Combines data fidelity loss with physics constraints.

```python
from dimtensor.torch import CompositeLoss, DimMSELoss, PhysicsLoss

loss_fn = CompositeLoss(
    data_loss=DimMSELoss(),
    physics_losses={
        'energy': (PhysicsLoss(), 0.1),
        'momentum': (PhysicsLoss(), 0.1),
    }
)

loss = loss_fn(pred, target, physics_terms={
    'energy': (E_initial, E_final),
    'momentum': (p_initial, p_final)
})
```

::: dimtensor.torch.CompositeLoss

## Normalization Layers

### DimBatchNorm1d

Batch normalization over 2D/3D input, preserving physical dimensions.

```python
from dimtensor.torch import DimBatchNorm1d

bn = DimBatchNorm1d(num_features=10, dimension=Dimension(L=1))
x_norm = bn(x)  # Normalized, still in meters
```

::: dimtensor.torch.DimBatchNorm1d

### DimBatchNorm2d

Batch normalization over 4D input (N, C, H, W).

::: dimtensor.torch.DimBatchNorm2d

### DimLayerNorm

Layer normalization preserving physical dimensions.

```python
ln = DimLayerNorm(
    normalized_shape=10,
    dimension=Dimension(M=1, L=2, T=-2)  # energy
)
```

::: dimtensor.torch.DimLayerNorm

### DimInstanceNorm1d

Instance normalization for 3D input.

::: dimtensor.torch.DimInstanceNorm1d

### DimInstanceNorm2d

Instance normalization for 4D input.

::: dimtensor.torch.DimInstanceNorm2d

## Scaling and Preprocessing

### DimScaler

Automatic non-dimensionalization for neural network training.

Neural networks train best with inputs in [-1, 1] or [0, 1]. DimScaler transforms physical quantities to dimensionless values and inverts the transformation on outputs.

```python
from dimtensor import DimArray, units
from dimtensor.torch import DimScaler

# Create and fit scaler
scaler = DimScaler(method='characteristic')
scaler.fit(velocities, temperatures, pressures)

# Transform to dimensionless
v_scaled = scaler.transform(velocities)  # torch.Tensor

# Train model...

# Inverse transform predictions
v_pred = scaler.inverse_transform(output, units.m / units.s)
```

**Methods:**

- `characteristic`: Divide by max absolute value
- `standard`: Z-score normalization (mean=0, std=1)
- `minmax`: Scale to [0, 1] range

::: dimtensor.torch.DimScaler
    options:
      members:
        - __init__
        - fit
        - transform
        - inverse_transform
        - get_scale
        - get_offset
        - dimensions

### MultiScaler

Manages multiple DimScalers for complex physics problems.

```python
from dimtensor.torch import MultiScaler

scaler = MultiScaler()
scaler.add('position', position_data)
scaler.add('velocity', velocity_data)
scaler.add('temperature', temperature_data)

# Transform
x_scaled = scaler.transform('position', x)
v_scaled = scaler.transform('velocity', v)

# Inverse transform
x_pred = scaler.inverse_transform('position', output)
```

::: dimtensor.torch.MultiScaler
    options:
      members:
        - __init__
        - add
        - transform
        - inverse_transform
        - get_scaler
        - quantities

## Examples

### Physics-Informed Neural Network

```python
import torch
import torch.nn as nn
from dimtensor.torch import (
    DimTensor, DimLinear, DimSequential,
    DimMSELoss, PhysicsLoss, CompositeLoss,
    DimScaler
)
from dimtensor import units, Dimension

# Define dimensions
L = Dimension(L=1)
V = Dimension(L=1, T=-1)

# Build network
model = DimSequential(
    DimLinear(1, 64, input_dim=L, output_dim=V),
    DimLinear(64, 64, input_dim=V, output_dim=V),
    DimLinear(64, 1, input_dim=V, output_dim=V)
)

# Scale data
scaler = DimScaler(method='standard')
scaler.fit(positions, velocities)

x_train = scaler.transform(positions)
v_train = scaler.transform(velocities)

# Define loss with physics constraint
loss_fn = CompositeLoss(
    data_loss=DimMSELoss(),
    physics_losses={'momentum': (PhysicsLoss(), 0.1)}
)

# Training loop
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(100):
    optimizer.zero_grad()

    # Forward pass (on scaled data)
    v_pred_scaled = model(x_train)

    # Convert back to physical units for loss
    v_pred = scaler.inverse_transform(v_pred_scaled, units.m / units.s)

    # Compute loss
    loss = loss_fn(v_pred, v_target)

    loss.backward()
    optimizer.step()
```

### Convolutional Model for Spatial Fields

```python
from dimtensor.torch import DimConv2d, DimBatchNorm2d
from dimtensor import units, Dimension

# Temperature field processing
T_dim = Dimension(Theta=1)

model = nn.Sequential(
    DimConv2d(1, 32, 3, input_dim=T_dim, output_dim=T_dim),
    DimBatchNorm2d(32, dimension=T_dim),
    nn.ReLU(),
    DimConv2d(32, 64, 3, input_dim=T_dim, output_dim=T_dim),
    DimBatchNorm2d(64, dimension=T_dim),
    nn.ReLU(),
    DimConv2d(64, 1, 3, input_dim=T_dim, output_dim=T_dim)
)

# Process temperature field
T_input = DimTensor(torch.randn(16, 1, 64, 64), units.K)
T_output = model(T_input)  # Dimension preserved: K
```

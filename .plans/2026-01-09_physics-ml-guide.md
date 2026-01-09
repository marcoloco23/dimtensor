# Plan: Physics-Informed Machine Learning Guide

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent
**Task ID**: #145

---

## Goal

Create a comprehensive tutorial guide (docs/guide/physics-ml.md) that teaches ML researchers how to build physics-informed neural networks using dimtensor. The guide will demonstrate how to leverage dimensional tracking, conservation laws, physics datasets, and equation databases for scientific machine learning.

---

## Background

**Why is this needed?**

1. dimtensor v2.2.0 introduced physics-aware ML capabilities (DimLayers, losses, scalers)
2. v3.0.0 added equation database and dataset registry
3. v3.3.0 deployed with advanced dataset loaders
4. However, there's no comprehensive guide showing how to USE these features together
5. ML researchers need end-to-end examples showing real physics applications

**Current state:**
- docs/guide/examples.md has basic physics calculations (kinematics, forces)
- docs/guide/operations.md covers array operations
- NO guide on physics-informed neural networks
- NO examples of using equation database for constraints
- NO examples of using dataset registry for training

**Target audience:**
- Machine learning researchers/engineers working on physics problems
- Scientists who want to incorporate physics constraints into neural networks
- Practitioners familiar with PyTorch but new to physics-informed ML

---

## Approach

### Content Structure

The guide will follow a progressive complexity approach:

1. **Introduction** - What are PINNs and why dimensional tracking matters
2. **Basic Example** - Simple pendulum prediction with DimTensor
3. **Dimensional Layers** - Building networks that preserve units
4. **Loss Functions** - Data fidelity + physics constraints
5. **Non-dimensionalization** - Scaling for effective training
6. **Equation Database** - Using registered equations for constraints
7. **Dataset Registry** - Loading physics datasets
8. **Complete Example** - End-to-end PINN for heat diffusion PDE
9. **Advanced Topics** - Conservation laws, composite losses, multi-physics

### Key Design Decisions

**Decision 1: Code-Heavy vs. Theory-Heavy**
- **Choice**: Code-heavy with minimal theory
- **Rationale**: Matches existing guide style (see examples.md), target audience knows ML
- **Approach**: Brief 1-2 sentence explanations, then runnable code examples

**Decision 2: Example Complexity**
- **Choice**: Start simple (1D ODE), build to complex (2D PDE)
- **Rationale**: Progressive learning, each section builds on previous
- **Approach**:
  - Section 2: Simple pendulum (ODE, 1 equation)
  - Section 8: Heat diffusion (PDE, spatial + temporal)

**Decision 3: Integration with Existing Code**
- **Choice**: Use ONLY existing dimtensor features (no new code needed)
- **Rationale**: This is documentation, not feature development
- **Available features**:
  - DimTensor with autograd
  - DimLinear, DimConv1d, DimConv2d layers
  - DimMSELoss, PhysicsLoss, CompositeLoss
  - DimScaler for non-dimensionalization
  - Equation database (67 equations across 10 domains)
  - Dataset registry (10 synthetic + 3 real datasets)

---

## Implementation Steps

### Phase 1: Structure and Outline
1. [ ] Create docs/guide/physics-ml.md skeleton with all section headers
2. [ ] Write introduction (what is physics-informed ML + why dimtensor)
3. [ ] Add navigation links to index.md

### Phase 2: Basic Examples
4. [ ] Section 2: Simple pendulum forward prediction
   - Train network to predict theta(t) given initial conditions
   - Use DimLinear layers with correct dimensions
   - DimMSELoss for data fidelity
5. [ ] Section 3: Dimensional layers tutorial
   - Show DimLinear with input_dim/output_dim
   - Show DimConv1d for time series
   - Show dimension validation catching errors

### Phase 3: Physics Constraints
6. [ ] Section 4: Loss functions
   - DimMSELoss example
   - PhysicsLoss for energy conservation
   - CompositeLoss combining data + physics
7. [ ] Section 5: Non-dimensionalization
   - DimScaler.fit() on physics data
   - Transform/inverse_transform workflow
   - Show why it improves training

### Phase 4: Database Integration
8. [ ] Section 6: Equation database
   - Search equations: `search_equations("pendulum")`
   - Get equation: `get_equation("Newton's Second Law")`
   - Use equation dimensions to validate model outputs
   - Extract equation for physics loss
9. [ ] Section 7: Dataset registry
   - List datasets: `list_datasets(domain="mechanics")`
   - Load dataset: `load_dataset("pendulum")`
   - Use dataset features/targets for training

### Phase 5: Complete End-to-End Example
10. [ ] Section 8: Heat diffusion PINN
    - Problem: Solve ∂u/∂t = α∇²u with boundary conditions
    - Load equation from database
    - Build DimConv1d network
    - Physics loss = PDE residual
    - Data loss = boundary conditions
    - Train and visualize solution
11. [ ] Add code that readers can copy-paste and run

### Phase 6: Advanced Topics
12. [ ] Section 9: Conservation law enforcement
    - Energy conservation example
    - Momentum conservation example
    - Multi-quantity conservation with CompositeLoss
13. [ ] Section 10: Best practices and tips
    - When to use dimensional layers vs. raw PyTorch
    - Debugging dimension errors
    - Performance considerations
    - Links to further reading

### Phase 7: Polish
14. [ ] Add cross-references to API docs
15. [ ] Add "Next Steps" section linking to other guides
16. [ ] Proofread and test all code examples
17. [ ] Add to docs navigation

---

## Files to Modify

| File | Change |
|------|--------|
| docs/guide/physics-ml.md | CREATE - Main tutorial document (~500-800 lines) |
| docs/index.md | ADD link to physics-ml guide in table of contents |
| docs/guide/README.md | ADD physics-ml to guide index (if exists) |

---

## Code Examples to Include

### Example 1: Simple Pendulum (Section 2)
```python
import torch
from dimtensor.torch import DimTensor, DimLinear, DimMSELoss
from dimtensor import units, Dimension

# Network: time [s] → angle [rad]
model = DimLinear(
    in_features=1, out_features=1,
    input_dim=Dimension(time=1),
    output_dim=Dimension()  # dimensionless (radians)
)

# Training data
t = DimTensor(torch.linspace(0, 10, 100).unsqueeze(1), units.s)
theta_true = DimTensor(torch.sin(t.data), units.rad)

# Train with dimensional loss
loss_fn = DimMSELoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(100):
    theta_pred = model(t)
    loss = loss_fn(theta_pred, theta_true)
    optimizer.zero_grad()
    loss.data.backward()  # Extract tensor for autograd
    optimizer.step()
```

### Example 2: Physics Loss (Section 4)
```python
from dimtensor.torch import PhysicsLoss, CompositeLoss

# Energy conservation constraint
physics_loss = PhysicsLoss(rtol=1e-3)

# Combine data + physics
composite = CompositeLoss(
    data_loss=DimMSELoss(),
    physics_losses={'energy': (physics_loss, 0.1)}
)

# Training loop
loss = composite(
    pred=prediction,
    target=ground_truth,
    physics_terms={'energy': (E_initial, E_final)}
)
```

### Example 3: Equation Database (Section 6)
```python
from dimtensor.equations import search_equations, get_equation

# Find relevant equations
eqs = search_equations("heat diffusion")
for eq in eqs:
    print(f"{eq.name}: {eq.formula}")

# Get specific equation
heat_eq = get_equation("Heat Diffusion (1D)")
print(f"Variables: {heat_eq.variables}")
# Use variables to validate network output dimensions
```

### Example 4: Dataset Loading (Section 7)
```python
from dimtensor.datasets import list_datasets, load_dataset

# Discover available datasets
datasets = list_datasets(domain="mechanics")
for ds in datasets:
    print(f"{ds.name}: {ds.description}")

# Load pendulum dataset
data = load_dataset("pendulum", num_samples=1000)
# Use data.features and data.targets for training
```

### Example 5: Heat Diffusion PINN (Section 8)
```python
import torch
import torch.nn as nn
from dimtensor.torch import DimTensor, DimLinear, DimSequential
from dimtensor.torch import CompositeLoss, PhysicsLoss
from dimtensor import units, Dimension

# Problem: ∂u/∂t = α∇²u
# Build network: (x, t) → u(x, t)

class HeatPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: concatenate position [m] and time [s]
        # Output: temperature [K]
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x, t):
        # x: DimTensor [m], t: DimTensor [s]
        # Normalize inputs (dimensionless for NN)
        x_norm = x.data / 1.0  # characteristic length
        t_norm = t.data / 1.0  # characteristic time

        inputs = torch.cat([x_norm, t_norm], dim=-1)
        u_norm = self.net(inputs)

        # Convert back to temperature
        return DimTensor(u_norm * 300, units.K)  # scale * characteristic temp

    def pde_residual(self, x, t, alpha):
        """Compute PDE residual: ∂u/∂t - α∇²u"""
        u = self.forward(x, t)

        # Automatic differentiation for derivatives
        u_t = torch.autograd.grad(u.data.sum(), t.data, create_graph=True)[0]
        u_x = torch.autograd.grad(u.data.sum(), x.data, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x.data, create_graph=True)[0]

        # PDE residual (should be zero)
        residual = u_t - alpha.data * u_xx
        return residual

# Training with physics constraint
model = HeatPINN()
physics_loss = lambda x, t, alpha: model.pde_residual(x, t, alpha).pow(2).mean()
data_loss = DimMSELoss()

# Combine losses
total_loss = data_loss(u_pred, u_boundary) + 0.1 * physics_loss(x, t, alpha)
```

---

## Testing Strategy

**Manual verification:**
- [ ] Copy each code example into a Python script
- [ ] Run each example and verify it executes without errors
- [ ] Verify outputs make physical sense (units, magnitudes)
- [ ] Test on Python 3.9, 3.10, 3.11

**Documentation quality:**
- [ ] All equations render correctly in markdown
- [ ] All links work (cross-references, API docs)
- [ ] Code formatting is consistent
- [ ] Navigation from index.md works

**Completeness:**
- [ ] Each section flows logically to the next
- [ ] No forward references to undefined concepts
- [ ] All dimtensor features mentioned are actually available
- [ ] Examples are self-contained (can copy-paste and run)

---

## Risks / Edge Cases

**Risk 1: Code examples don't work**
- Mitigation: Test EVERY code snippet before finalizing
- Mitigation: Use only features that exist in current codebase

**Risk 2: Too theoretical for target audience**
- Mitigation: Lead with code, keep explanations to 1-2 sentences
- Mitigation: Follow style of existing examples.md guide

**Risk 3: Examples too simple or too complex**
- Mitigation: Progressive complexity (ODE → PDE)
- Mitigation: Start with 10-line examples, build to 50-line complete example

**Risk 4: Equation database doesn't have needed equations**
- Mitigation: Use equations that already exist (checked database.py - has 67 equations)
- Fallback: Show how to create custom equations if needed

**Edge case: User doesn't have torch installed**
- Handling: Add note at top: "Requires: pip install dimtensor[torch]"

**Edge case: Autograd with DimTensor**
- Handling: Show .data extraction for backward() pass (see existing DimTensor code)

---

## Definition of Done

- [x] Research existing codebase (equations, datasets, torch modules)
- [ ] All sections written with code examples
- [ ] All code examples tested and working
- [ ] Cross-references to API docs added
- [ ] Navigation from docs/index.md works
- [ ] Guide follows style of existing docs
- [ ] No references to non-existent features
- [ ] Proofread for clarity and correctness
- [ ] File created: docs/guide/physics-ml.md

---

## Notes / Log

**2026-01-09 - Research Phase**
- Analyzed existing physics ML infrastructure:
  - torch/dimtensor.py: DimTensor with autograd ✓
  - torch/layers.py: DimLinear, DimConv1d/2d, DimSequential ✓
  - torch/losses.py: DimMSELoss, DimL1Loss, PhysicsLoss, CompositeLoss ✓
  - torch/scaler.py: DimScaler for non-dimensionalization ✓
  - equations/database.py: 67 equations across 10 domains ✓
  - datasets/registry.py: 10 synthetic + 3 real datasets ✓

- Reviewed existing guide style:
  - docs/guide/examples.md: Code-heavy, minimal theory ✓
  - docs/getting-started.md: Installation + quick examples ✓
  - Pattern: Brief intro, code example, practical notes

- Key finding: ALL features needed for guide already exist!
  - No new code required
  - Just need to show how to combine them

- Plan structure: 10 sections
  1. Intro (2-3 paragraphs)
  2. Basic example (pendulum, ~30 lines)
  3. Dimensional layers (~50 lines)
  4. Loss functions (~60 lines)
  5. Non-dimensionalization (~40 lines)
  6. Equation database (~30 lines)
  7. Dataset registry (~30 lines)
  8. Complete PINN example (~100 lines)
  9. Conservation laws (~50 lines)
  10. Best practices (~40 lines)

Total: ~500-600 lines of markdown + code

**Estimated complexity: MEDIUM**
- No new features to implement
- Requires testing all examples
- Needs careful integration of existing pieces
- Must maintain consistency with existing docs

---

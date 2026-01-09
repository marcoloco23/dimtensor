# Plan: PyTorch Training Example Notebook (03_pytorch_training.ipynb)

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create a comprehensive Jupyter notebook demonstrating how to train a Physics-Informed Neural Network (PINN) using dimtensor's PyTorch integration. The notebook will solve the 1D heat equation as an example, showcasing DimTensor, DimLinear layers, physics-aware loss functions, and dimensional validation during training.

---

## Background

dimtensor provides dimension-aware PyTorch layers (DimLinear, DimConv1d/2d), loss functions (DimMSELoss, PhysicsLoss, CompositeLoss), and training utilities. This notebook is part of v3.4.0's documentation effort (task #153) to make dimtensor accessible through hands-on examples.

PINNs combine data-driven learning with physics constraints encoded as differential equations. Using dimtensor ensures dimensional consistency throughout the training process.

---

## Approach

### Option A: 1D Heat Equation PINN
- Description: Solve ∂T/∂t = α·∂²T/∂x² with boundary/initial conditions
- Pros:
  - Well-understood physics (thermal diffusion)
  - Simple 1D domain
  - Clear physical units (temperature K, position m, time s)
  - Easy to visualize results
  - Demonstrates conservation checking (energy)
- Cons:
  - Relatively simple problem

### Option B: Wave Equation or Burgers Equation
- Description: More complex PDEs with richer dynamics
- Pros:
  - More impressive results
  - Shows dimtensor handling complex physics
- Cons:
  - Harder for beginners to understand
  - More complex setup
  - Less intuitive physical validation

### Decision: Option A (Heat Equation)

The heat equation is pedagogically superior - readers will understand the physics, making it easier to see how dimtensor adds value. The simplicity lets us focus on the dimtensor features rather than mathematical complexity.

---

## Implementation Steps

1. [ ] Create `examples/` directory structure
2. [ ] Create notebook file `examples/03_pytorch_training.ipynb`
3. [ ] Write notebook cells in this order:
   - [ ] Title and overview cell
   - [ ] Installation instructions
   - [ ] Imports (torch, dimtensor, matplotlib)
   - [ ] Problem definition (heat equation, domain, parameters)
   - [ ] Generate training data (analytical solution or simulation)
   - [ ] Define PINN architecture using DimLinear
   - [ ] Define composite loss (data + physics terms)
   - [ ] Training loop with dimensional validation
   - [ ] Visualize predictions vs ground truth
   - [ ] Physical validation (conservation checks)
   - [ ] Extensions/exercises section
4. [ ] Test notebook execution (run all cells)
5. [ ] Add to documentation index/navigation

---

## Files to Modify

| File | Change |
|------|--------|
| examples/03_pytorch_training.ipynb | CREATE - Main training notebook (~30-40 cells) |
| examples/README.md | CREATE - Index of example notebooks |
| docs/guide/examples.md | UPDATE - Add link to new notebook |
| .gitignore | UPDATE - Add Jupyter checkpoint folders |

---

## Notebook Structure (Detailed)

### Cell Breakdown (~35 cells total)

**Part 1: Setup (5 cells)**
1. Title markdown: "Training a Physics-Informed Neural Network with DimTensor"
2. Overview markdown: Problem description, learning goals
3. Installation code: `pip install dimtensor[torch]`
4. Imports
5. Set random seeds for reproducibility

**Part 2: Problem Definition (6 cells)**
6. Markdown: Heat equation theory
7. Define physical parameters with units (α = 0.01 m²/s, domain size, time)
8. Markdown: Boundary and initial conditions
9. Define analytical solution function (if available) or reference solution
10. Markdown: Training data strategy
11. Generate collocation points (space-time grid) with units

**Part 3: Data Generation (4 cells)**
12. Markdown: Ground truth generation
13. Compute temperature field at collocation points
14. Visualize initial/boundary conditions
15. Split into train/validation sets

**Part 4: Model Architecture (5 cells)**
16. Markdown: PINN architecture design
17. Define input/output dimensions (Dimension objects)
18. Build model using DimSequential and DimLinear:
    - Input: (x, t) → [L, T]
    - Hidden layers with dimensionless intermediate representations
    - Output: T(x,t) → [Θ] (temperature)
19. Print model summary showing dimensions
20. Markdown: Explain dimension flow

**Part 5: Loss Functions (5 cells)**
21. Markdown: Physics-aware loss design
22. Define data fidelity loss (DimMSELoss)
23. Define physics loss (compute PDE residual, check dimensions)
24. Define CompositeLoss combining both terms
25. Markdown: Loss weighting strategy

**Part 6: Training (7 cells)**
26. Markdown: Training configuration
27. Define optimizer (Adam), learning rate, epochs
28. Training loop implementation:
    - Forward pass with DimTensor
    - Compute composite loss
    - Dimension validation checks
    - Backpropagation
    - Log metrics
29. Plot training curves (data loss, physics loss, total loss)
30. Markdown: Training observations
31. Checkpoint best model
32. Markdown: Convergence discussion

**Part 7: Evaluation (5 cells)**
33. Markdown: Model evaluation
34. Generate predictions on test grid
35. Visualize predictions vs ground truth (heatmap/animation)
36. Compute error metrics (MSE, MAE) with units
37. Plot residuals

**Part 8: Physical Validation (4 cells)**
38. Markdown: Conservation law checking
39. Compute total energy at different time steps
40. Use PhysicsLoss to check energy conservation
41. Visualize conservation metrics

**Part 9: Conclusion (2 cells)**
42. Markdown: Summary and key takeaways
43. Markdown: Extensions and exercises:
    - Try different α values
    - Add noise to training data
    - Use different boundary conditions
    - Solve 2D heat equation
    - Try other PDEs (wave, diffusion-reaction)

---

## Testing Strategy

How will we verify this works?

- [ ] Execute notebook end-to-end in fresh environment
- [ ] Verify all cells run without errors
- [ ] Check training converges (loss decreases)
- [ ] Verify dimensional consistency (no DimensionError raised)
- [ ] Visual inspection of predictions (should match physics)
- [ ] Test with different PyTorch versions (1.x, 2.x)
- [ ] Test on CPU and GPU (if available)
- [ ] Verify plots render correctly
- [ ] Check notebook size is reasonable (<5MB)

---

## Risks / Edge Cases

- **Risk**: Notebook takes too long to train (>5 minutes)
  - **Mitigation**: Use small network (2-3 hidden layers, 32-64 neurons), limited epochs (1000), small domain

- **Risk**: Training doesn't converge
  - **Mitigation**: Provide known-good hyperparameters (learning rate, loss weights), use simple problem with smooth solution

- **Risk**: Users don't have GPU, training too slow
  - **Mitigation**: Design for CPU execution (<2 min), add note about GPU speedup

- **Edge case**: Different PyTorch versions behave differently
  - **Mitigation**: Test with torch 1.13+ and 2.0+, document version requirements

- **Edge case**: Matplotlib backend issues in different environments
  - **Mitigation**: Use %matplotlib inline, provide clear error messages

- **Risk**: Dimensional errors confuse beginners
  - **Mitigation**: Add explanatory markdown cells before each dimension-related operation, show what units flow through model

---

## Definition of Done

- [ ] Notebook created with all 35+ cells
- [ ] All cells execute successfully
- [ ] Training converges to reasonable accuracy (<1% error)
- [ ] All plots render correctly
- [ ] Dimensional validation passes throughout
- [ ] Physical conservation check passes
- [ ] Code is well-commented and documented
- [ ] Markdown cells explain each step clearly
- [ ] Examples README created linking to notebook
- [ ] Tested in fresh conda environment
- [ ] Added to docs/guide/examples.md
- [ ] CONTINUITY.md updated with completion

---

## Notes / Log

**[2026-01-09 - Planning]**
- Researched dimtensor torch modules: DimTensor, DimLinear, DimMSELoss, PhysicsLoss, CompositeLoss
- Chose 1D heat equation as example problem (pedagogically sound, clear physics)
- Planned 35-cell notebook structure balancing education and completeness
- Key insight: Use dimensionless intermediate layers to avoid complex dimension algebra
- Key insight: Show dimension flow explicitly to teach dimensional consistency
- Expected training time: <2 minutes on CPU with suggested hyperparameters
- Target audience: ML practitioners familiar with PyTorch, learning physics-informed ML

---

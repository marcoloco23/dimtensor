# Plan: PyTorch Integration Guide

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create a comprehensive user guide for dimtensor's PyTorch integration that teaches users how to build physics-informed neural networks with automatic dimensional checking, covering DimTensor operations, dimension-aware layers, loss functions, normalization, and non-dimensionalization strategies.

---

## Background

dimtensor provides a powerful PyTorch integration that wraps torch.Tensor with physical units, enabling:
- Automatic dimension checking during neural network operations
- Gradient flow through unit-aware computations
- Dimension-aware layers (DimLinear, DimConv1d/2d)
- Physics-informed loss functions
- Automatic non-dimensionalization (DimScaler)

Currently, users need to piece together information from docstrings and tests. A comprehensive guide will:
1. Lower the barrier to entry for physics-informed ML
2. Demonstrate best practices for training with physical units
3. Show how to build complete PINN workflows
4. Provide reusable examples for common physics problems

---

## Approach

### Option A: Tutorial-style (Beginner to Advanced)
- Description: Start with basics (creating DimTensors) and progressively build to complex PINNs
- Pros:
  - Accessible to newcomers
  - Natural learning progression
  - Easy to follow step-by-step
- Cons:
  - May be too slow for experienced users
  - Advanced users might need to skip sections

### Option B: Reference-style (Feature-focused)
- Description: Organize by feature (layers, losses, etc.) with complete examples for each
- Pros:
  - Easy to find specific features
  - Good for reference lookup
  - Each section stands alone
- Cons:
  - May not show how features work together
  - Harder to see big picture

### Option C: Hybrid (Overview + Examples + Complete PINN)
- Description: Brief overview of all features, then detailed examples, culminating in a complete PINN example
- Pros:
  - Best of both worlds
  - Quick reference + deep dive
  - Shows integration of all features
  - Matches existing docs style (see docs/guide/examples.md)
- Cons:
  - Slightly longer document
  - Requires careful organization

### Decision: Option C (Hybrid)

This matches the existing documentation style in docs/guide/examples.md and provides both reference material and tutorial content. Users can read top-to-bottom or jump to specific sections.

---

## Implementation Steps

1. [ ] Create docs/guide/pytorch.md with document structure
2. [ ] Write "Introduction" section explaining when to use PyTorch integration
3. [ ] Write "DimTensor Basics" section:
   - Creating DimTensors
   - Arithmetic operations
   - Autograd support
   - Device management
4. [ ] Write "Dimension-Aware Layers" section:
   - DimLinear with dimension tracking
   - DimConv1d and DimConv2d
   - DimSequential for chaining layers
   - Example: Building a physics-informed network
5. [ ] Write "Loss Functions" section:
   - DimMSELoss, DimL1Loss, DimHuberLoss
   - PhysicsLoss for conservation laws
   - CompositeLoss for combining data + physics terms
6. [ ] Write "Normalization Layers" section:
   - DimBatchNorm1d/2d
   - DimLayerNorm
   - DimInstanceNorm1d/2d
   - Why normalization preserves dimensions
7. [ ] Write "Non-dimensionalization" section:
   - DimScaler (characteristic, standard, minmax methods)
   - MultiScaler for multiple quantities
   - Best practices for scaling physical data
8. [ ] Write "Complete PINN Example" section:
   - Problem: 1D heat equation or wave equation
   - Data preparation with DimScaler
   - Model architecture with DimLayers
   - Physics-informed loss (PDE residual + boundary conditions)
   - Training loop
   - Inverse transform predictions
   - Validation of results
9. [ ] Add "Best Practices" section:
   - When to use dimension checking vs raw tensors
   - Performance considerations
   - Common pitfalls
10. [ ] Add "Tips and Tricks" section:
    - Converting between DimTensor and raw tensors
    - Debugging dimension errors
    - Integration with existing PyTorch code
11. [ ] Review and test all code examples
12. [ ] Update docs/index.md to link to pytorch.md guide

---

## Files to Modify

| File | Change |
|------|--------|
| docs/guide/pytorch.md | Create new guide document (main deliverable) |
| docs/index.md | Add link to PyTorch guide in navigation/TOC |
| docs/getting-started.md | (Optional) Add brief mention of PyTorch support with link |

---

## Testing Strategy

How will we verify this works?

- [ ] Run all code examples in the guide to verify they execute without errors
- [ ] Test code with both NumPy 1.x and 2.x (if applicable for PyTorch tensors)
- [ ] Verify all imports work correctly
- [ ] Check that dimension errors are caught as expected in examples
- [ ] Validate PINN example produces physically reasonable results
- [ ] Run examples on both CPU and (if available) CUDA devices
- [ ] Get feedback from 1-2 test readers on clarity

---

## Risks / Edge Cases

- **Risk**: PINN example too complex for beginners
  - **Mitigation**: Start with simplest possible PDE (1D heat equation), add extensive comments

- **Risk**: Code examples become outdated as API evolves
  - **Mitigation**: Keep examples simple and focused on stable API features; add doctest-style testing later

- **Risk**: Guide too long and overwhelming
  - **Mitigation**: Use clear section headers, add table of contents, keep each section focused

- **Edge case**: Users without PyTorch installed
  - **Handling**: Add clear prerequisites section at top; show pip install command

- **Edge case**: Users on GPU vs CPU
  - **Handling**: All examples use CPU by default; add note about .cuda() or .to('cuda') for GPU users

- **Edge case**: Integration with standard PyTorch modules (dropout, activation functions)
  - **Handling**: Show examples of mixing DimTensor with standard nn.Module layers

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] All code examples tested and working
- [ ] Guide follows existing documentation style
- [ ] Navigation/links updated in docs/index.md
- [ ] PINN example produces correct physics results
- [ ] No broken internal links
- [ ] CONTINUITY.md updated with completion

---

## Notes / Log

**Key Design Decisions:**

1. **PINN Problem Selection**: Use 1D heat equation (∂u/∂t = α ∂²u/∂x²) because:
   - Simple enough to understand
   - Shows spatial + temporal derivatives
   - Demonstrates boundary conditions
   - Physically intuitive (temperature diffusion)

2. **Code Example Style**: Follow docs/guide/examples.md style:
   - Complete, runnable code blocks
   - Clear comments explaining physics
   - Print statements showing results with units
   - Progressive complexity

3. **Sections Organization**:
   - Start with fundamentals (DimTensor)
   - Build up to complex features (layers, losses)
   - Culminate in complete example (PINN)
   - End with practical advice (best practices)

4. **Target Audience**:
   - Primary: ML researchers/engineers working on physics problems
   - Secondary: Physics students/researchers learning ML
   - Assumption: Basic PyTorch knowledge, physics background helpful but not required

---

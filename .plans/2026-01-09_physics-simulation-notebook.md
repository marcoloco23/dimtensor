# Plan: Physics Simulation Notebook (02_physics_simulation.ipynb)

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent
**Task**: #152 from CONTINUITY.md

---

## Goal

Create an interactive Jupyter notebook demonstrating dimtensor's capabilities for classical physics simulations. The notebook will showcase dimensional safety, unit conversions, physical constants, uncertainty propagation, and visualization through three progressively complex examples: projectile motion with air resistance, simple pendulum, and orbital mechanics.

---

## Background

**Why is this needed?**
- Part of v3.4.0 "Documentation & Polish" milestone (ROADMAP.md)
- Demonstrates dimtensor's value proposition for physics/engineering users
- Shows real-world application of dimensional analysis in computational physics
- Provides educational examples for students and researchers
- Highlights features: DimArray, constants, visualization, validation

**Target audience**: Physics students, engineers, researchers who want to see dimtensor in action with familiar problems.

---

## Approach

### Option A: Separate notebook per simulation
- Description: Create 3 separate notebooks (02a, 02b, 02c) for each simulation
- Pros: Easier to maintain, focused content, can be used independently
- Cons: Fragmented user experience, doesn't show progression of complexity

### Option B: Single unified notebook with 3 simulations
- Description: One notebook with clear sections for each simulation, building complexity
- Pros: Shows progression, better storytelling, easier to navigate
- Cons: Longer notebook, potential for overwhelming beginners

### Decision: Option B - Single unified notebook

**Reasoning**:
1. Better educational flow (simple → complex)
2. Can reuse setup code and patterns
3. Demonstrates dimtensor's versatility in one place
4. Estimated 25-35 cells is manageable
5. Clear section headers will aid navigation

---

## Implementation Steps

### Setup & Introduction (Cells 1-4)
1. [ ] Title cell with notebook overview
2. [ ] Import cell: dimtensor, numpy, matplotlib, constants
3. [ ] Setup visualization (matplotlib integration)
4. [ ] Introduction markdown: What we'll simulate and why dimtensor helps

### Example 1: Projectile Motion with Air Resistance (Cells 5-12)
5. [ ] Theory markdown: Equations of motion with drag force
6. [ ] Define parameters: mass, drag coefficient, area, initial velocity/angle
7. [ ] Implement differential equation solver (Euler or RK4)
8. [ ] Run simulation with units tracking
9. [ ] Plot trajectory (x vs y with units)
10. [ ] Plot velocity magnitude over time
11. [ ] Demonstrate dimension error catching (try invalid operation)
12. [ ] Validation: Check energy dissipation, compare with/without drag

### Example 2: Simple Pendulum (Cells 13-20)
13. [ ] Theory markdown: Pendulum equation (nonlinear)
14. [ ] Define parameters: length, mass, g (from constants or custom)
15. [ ] Implement ODE solver for θ(t)
16. [ ] Run simulation from various initial angles
17. [ ] Plot angle vs time with multiple curves
18. [ ] Plot phase space (θ vs ω)
19. [ ] Demonstrate period analysis with units
20. [ ] Validation: Energy conservation check, small-angle approximation comparison

### Example 3: Orbital Mechanics (Cells 21-30)
21. [ ] Theory markdown: Gravitational 2-body problem, Kepler's laws
22. [ ] Import gravitational constant G from dimtensor.constants
23. [ ] Define central mass (e.g., Sun), satellite mass, initial conditions
24. [ ] Implement 2D orbital integrator (position + velocity)
25. [ ] Run simulation for elliptical orbit
26. [ ] Plot orbit (x vs y) with central body
27. [ ] Plot orbital speed vs radius
28. [ ] Demonstrate eccentricity calculation
29. [ ] Validation: Energy conservation, angular momentum conservation
30. [ ] Show unit conversion: orbital period in different time units

### Conclusion & Extensions (Cells 31-35)
31. [ ] Summary markdown: What we learned about dimtensor
32. [ ] Key benefits demonstrated: dimension safety, physical constants, viz
33. [ ] Suggested exercises: 3D motion, coupled oscillators, etc.
34. [ ] Links to other notebooks and documentation
35. [ ] Optional: Performance note on computational overhead

---

## Files to Modify

| File | Change |
|------|--------|
| examples/ (directory) | Create if doesn't exist |
| examples/02_physics_simulation.ipynb | **NEW FILE** - Full notebook implementation |

**Note**: May also need to update README.md or index to link to examples/.

---

## Testing Strategy

How will we verify this works?

### Automated Testing
- [ ] Run notebook kernel start-to-finish without errors
- [ ] Verify all imports succeed
- [ ] Check that plots are generated (output cells contain figures)
- [ ] Validate that dimensional errors are caught as expected

### Manual Verification
- [ ] Review all plots for correctness (trajectory shapes, units on axes)
- [ ] Verify physical validity (e.g., pendulum period ≈ 2π√(L/g))
- [ ] Check energy/momentum conservation within numerical tolerance
- [ ] Ensure markdown cells are clear and educational
- [ ] Test on fresh kernel (no residual state)

### Code Quality
- [ ] Follow dimtensor usage patterns from existing codebase
- [ ] Use internal constructors (_from_data_and_unit) appropriately
- [ ] Proper error handling demonstrations
- [ ] Clear variable names and comments
- [ ] Consistent formatting

### Physics Validation
- [ ] Projectile: Compare max range with analytic formula (no drag case)
- [ ] Pendulum: Verify period matches small-angle approximation for θ₀ < 10°
- [ ] Orbit: Check vis-viva equation μ(2/r - 1/a) = v² holds

---

## Risks / Edge Cases

### Risk 1: Numerical instability in integrators
- **Description**: Euler method may be unstable for stiff equations or long times
- **Mitigation**: Use RK4 (4th order Runge-Kutta) or scipy.integrate.solve_ivp
- **Handling**: Add note about integration methods, keep timesteps reasonable

### Risk 2: Unit dimension mismatches in ODE solvers
- **Description**: Integrators expect raw numpy arrays, not DimArray
- **Mitigation**: Extract ._data for solver, re-wrap results with units
- **Handling**: Document the pattern clearly, explain why it's necessary

### Risk 3: Matplotlib compatibility
- **Description**: Auto-labeling might not work perfectly with all plot types
- **Mitigation**: Use dimtensor.visualization.setup_matplotlib()
- **Handling**: Fall back to manual labels if needed

### Risk 4: Notebook execution time
- **Description**: Long simulations might take too long for tutorial
- **Mitigation**: Choose simulation parameters for ~1-5 second execution
- **Handling**: Add note that users can increase resolution/duration

### Edge Case: Zero or negative masses/distances
- **Handling**: Use realistic physical parameters, add validation cells if needed

### Edge Case: Uncertainty propagation overhead
- **Handling**: Keep uncertainty optional, mention as extension possibility

### Edge Case: Users without matplotlib/scipy
- **Handling**: Clear dependency statement at top, graceful ImportError handling

---

## Definition of Done

- [x] Plan document created and reviewed
- [ ] Notebook created at examples/02_physics_simulation.ipynb
- [ ] All 3 simulations implemented with correct physics
- [ ] All plots render correctly with unit labels
- [ ] Energy/momentum conservation validated numerically
- [ ] Dimension error demonstration included
- [ ] Notebook runs cleanly in fresh kernel (no errors)
- [ ] CONTINUITY.md updated with completion status
- [ ] Optional: README.md updated to reference examples/

---

## Notes / Log

**2026-01-09 13:00** - Plan created. Key decisions:
- Single notebook with 3 examples (progressive complexity)
- Focus on educational value and dimtensor feature showcase
- Use RK4 or scipy for numerical stability
- Clear separation between theory, implementation, validation
- Expected cell count: 30-35 (within 25-35 target)

**Physics constants available**:
- `dimtensor.constants.G` - Gravitational constant
- `dimtensor.constants.c` - Speed of light
- Can define custom `g = DimArray(9.81, units.m/units.s**2)` for Earth gravity

**Visualization features**:
- `dimtensor.visualization.setup_matplotlib()` for auto-labeling
- `dimtensor.visualization.plot()` wrapper functions
- DimArrayConverter handles unit conversions in plots

**Code patterns to follow**:
- Use `DimArray(data, unit)` for construction
- Use `._data` to extract numpy arrays for solvers
- Use `DimArray._from_data_and_unit()` for efficient re-wrapping
- Use `.to(unit)` for conversions
- Catch `DimensionError` for invalid operations

---

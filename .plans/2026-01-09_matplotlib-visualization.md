# Plan: Matplotlib Visualization Integration

**Date**: 2026-01-09
**Status**: IN PROGRESS
**Author**: agent

---

## Goal

Add matplotlib integration for dimtensor DimArray with automatic axis labeling based on units. Enable scientists to plot dimensioned data with proper unit labels automatically appearing on axes.

---

## Background

Scientists need plots with proper unit labels. Currently, users must manually extract `.data` and add labels. Libraries like pint and astropy provide matplotlib integration via the `matplotlib.units` registry system. dimtensor should follow this established pattern.

---

## Approach

### Option A: matplotlib.units.ConversionInterface
- Implement a converter class that registers with matplotlib's unit registry
- Automatic: Once enabled, all DimArray plots get unit labels
- Follows established pattern used by pint, astropy, unyt
- Pros: Standard approach, works with any matplotlib plotting function
- Cons: Global state (registry), requires explicit setup call

### Option B: Wrapper functions (plot, scatter, etc.)
- Create standalone functions that wrap matplotlib functions
- More explicit control over behavior
- Pros: No global state, clear API
- Cons: Need to wrap every matplotlib function, doesn't work with seaborn/etc.

### Option C: Hybrid (both)
- Provide both: converter for automatic integration, plus convenient wrapper functions
- Pros: Flexibility, works with external libs and has easy API
- Cons: More code to maintain

### Decision: Option C (Hybrid)

Best of both worlds - the converter handles automatic integration, wrapper functions provide convenience and explicit unit conversion options (e.g., plot in km instead of m).

---

## Implementation Steps

1. [x] Research matplotlib.units API (Task #42)
2. [ ] Create visualization/ folder with __init__.py
3. [ ] Implement DimArrayConverter class in visualization/matplotlib.py:
   - `convert(value, unit, axis)` - Extract .data, apply unit conversion
   - `axisinfo(unit, axis)` - Return AxisInfo with unit label
   - `default_units(x, axis)` - Return Unit from DimArray
4. [ ] Implement `setup_matplotlib(enable=True)` to register/unregister converter
5. [ ] Implement wrapper functions:
   - `plot(x, y, unit=None, **kwargs)` - line plot
   - `scatter(x, y, unit=None, **kwargs)` - scatter plot
   - `bar(x, height, unit=None, **kwargs)` - bar chart
   - `hist(x, unit=None, **kwargs)` - histogram
6. [ ] Add unit conversion option to wrappers (e.g., plot distance in km)
7. [ ] Update visualization/__init__.py with exports
8. [ ] Add tests
9. [ ] Update __init__.py to expose visualization module

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/visualization/__init__.py | NEW: Export setup_matplotlib, plot, scatter, bar, hist |
| src/dimtensor/visualization/matplotlib.py | NEW: DimArrayConverter, wrapper functions |
| src/dimtensor/__init__.py | Add visualization module export |
| tests/test_visualization.py | NEW: Tests for matplotlib integration |

---

## Testing Strategy

- [ ] Test converter registration/unregistration
- [ ] Test automatic axis labeling with registered converter
- [ ] Test wrapper functions (plot, scatter, bar, hist)
- [ ] Test unit conversion in wrappers
- [ ] Test with 1D and 2D data
- [ ] Test error handling (incompatible units on same axis)
- [ ] Skip tests if matplotlib not installed

---

## Risks / Edge Cases

- Risk: matplotlib not installed → Mitigation: Optional dependency, import guard
- Edge case: Mixing units on same axis → Return error with helpful message
- Edge case: Dimensionless data → No unit label needed
- Edge case: Complex unit symbols → Ensure readable labels
- Edge case: Data with uncertainty → Include error bars option

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Tests pass
- [ ] Can plot DimArray with automatic unit labels
- [ ] Can convert units during plotting
- [ ] CONTINUITY.md updated

---

## Notes / Log

**2026-01-09 morning** - Plan created based on research of pint/astropy patterns

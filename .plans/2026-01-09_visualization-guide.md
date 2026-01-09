# Plan: Visualization Guide Documentation

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create comprehensive user documentation for creating publication-quality plots with DimArray objects, teaching users how to leverage automatic unit labeling and conversion in both matplotlib and plotly.

---

## Background

The dimtensor library has visualization modules for matplotlib and plotly (src/dimtensor/visualization/) that automatically extract units from DimArray objects and label axes appropriately. However, there is no user-facing guide that teaches users how to use these features effectively. This guide will be a critical resource for users creating plots for papers, presentations, and analysis.

---

## Approach

### Option A: Single comprehensive guide
- Description: Create one guide file (docs/guide/visualization.md) covering both matplotlib and plotly
- Pros:
  - Single place to look for visualization help
  - Easy to compare matplotlib vs plotly approaches
  - Follows pattern of other guides in docs/guide/
- Cons:
  - Could be long (but manageable)

### Option B: Separate files for each library
- Description: Create docs/guide/visualization-matplotlib.md and docs/guide/visualization-plotly.md
- Pros:
  - More focused documentation
  - Easier to maintain separately
- Cons:
  - Duplication of common concepts
  - Less convenient for users to compare

### Decision: Option A

Reasoning: The existing guides (units.md, operations.md, examples.md) are comprehensive single files. The visualization functionality is cohesive enough to be covered in one guide. Using sections and subsections will keep it organized.

---

## Implementation Steps

1. [ ] Create docs/guide/visualization.md file
2. [ ] Write introduction section explaining automatic unit labeling benefit
3. [ ] Write "Quick Start" section with minimal working examples
4. [ ] Write "Matplotlib Integration" section covering:
   - [ ] setup_matplotlib() for direct plt.plot usage
   - [ ] Wrapper functions (plot, scatter, bar, hist, errorbar)
   - [ ] Unit conversion with x_unit/y_unit parameters
   - [ ] Working with dimensionless quantities
5. [ ] Write "Plotly Integration" section covering:
   - [ ] Import structure (dimtensor.visualization.plotly)
   - [ ] Wrapper functions (line, scatter, bar, histogram, scatter_with_errors)
   - [ ] Unit conversion and custom titles
6. [ ] Write "Error Bars and Uncertainty" section covering:
   - [ ] Auto-extraction from DimArray.uncertainty
   - [ ] errorbar() function (matplotlib)
   - [ ] scatter_with_errors() function (plotly)
   - [ ] Explicit error specification
7. [ ] Write "Customization" section covering:
   - [ ] Passing matplotlib kwargs (color, linestyle, etc)
   - [ ] Using ax parameter for subplots
   - [ ] Plotly styling and layout options
8. [ ] Write "Best Practices" section
9. [ ] Add admonitions (tips, warnings) where appropriate
10. [ ] Review for clarity and completeness

---

## Files to Modify

| File | Change |
|------|--------|
| docs/guide/visualization.md | Create new file with complete visualization guide |

---

## Testing Strategy

Documentation quality checks:

- [ ] All code examples are runnable and produce correct output
- [ ] All functions mentioned are correctly imported
- [ ] Examples cover common use cases from tests/test_visualization.py
- [ ] Links to API reference are correct (if added)
- [ ] Follows style/tone of existing guides (units.md, operations.md)

Manual verification:

- [ ] Run each example in a Python interpreter to ensure it works
- [ ] Check that matplotlib examples work with both setup_matplotlib() and wrapper functions
- [ ] Verify plotly examples produce interactive figures
- [ ] Test unit conversion examples show correct values

---

## Risks / Edge Cases

- Risk: Users might not have matplotlib or plotly installed
  - Mitigation: Include clear installation instructions at the top

- Risk: Examples might break with matplotlib/plotly version changes
  - Mitigation: Use stable, well-established APIs (plot, scatter, etc)

- Edge case: Dimensionless quantities don't show units
  - Handling: Explicitly document this behavior with example

- Edge case: Unit conversion might fail for incompatible dimensions
  - Handling: Show what happens and reference UnitConversionError

- Risk: Guide might be too long
  - Mitigation: Use clear headings, table of contents anchor links, and code folding if needed

---

## Definition of Done

- [ ] docs/guide/visualization.md created with all sections
- [ ] At least 10 working code examples included
- [ ] Covers both matplotlib (2 modes) and plotly
- [ ] Explains unit conversion in plots
- [ ] Explains error bar auto-extraction
- [ ] Includes best practices section
- [ ] All examples tested and runnable
- [ ] Plan file marked as COMPLETED

---

## Key Content Sections

### 1. Introduction (1-2 paragraphs)
- Why automatic unit labeling matters for scientific plots
- Overview of supported libraries

### 2. Quick Start
- Minimal matplotlib example (3-4 lines)
- Minimal plotly example (3-4 lines)

### 3. Matplotlib Integration
- **Mode 1: Direct Integration**
  - setup_matplotlib() explanation
  - Example with plt.plot, plt.scatter
  - When units get extracted and labeled

- **Mode 2: Wrapper Functions**
  - plot() example with physics data
  - scatter() example
  - bar() example
  - hist() example
  - errorbar() with uncertainty

- **Unit Conversion**
  - x_unit and y_unit parameters
  - Example converting km to m, cm to m

- **Dimensionless Quantities**
  - Example showing no labels for dimensionless

### 4. Plotly Integration
- Import structure (from dimtensor.visualization import plotly)
- line() example
- scatter() example
- bar() example
- histogram() example
- scatter_with_errors() with uncertainty
- Custom titles and layout

### 5. Error Bars and Uncertainty
- Creating DimArray with uncertainty
- Auto-extraction in errorbar() (matplotlib)
- Auto-extraction in scatter_with_errors() (plotly)
- Explicit yerr/xerr specification
- Unit conversion of error bars

### 6. Customization
- Matplotlib: color, linestyle, marker, etc
- Using ax parameter for subplots/multi-panel figures
- Plotly: styling traces, layout customization
- Combining custom labels with unit labels

### 7. Best Practices
- Always convert to convenient units for display (km not m for long distances)
- Use wrapper functions for simplicity
- Leverage uncertainty auto-extraction
- Add titles and legends for publication plots

---

## Example Code Snippets to Include

1. Basic plot with automatic labels
2. Scatter plot with unit conversion
3. Multiple subplots with consistent units
4. Error bars with auto-extracted uncertainty
5. Bar chart comparing quantities
6. Histogram of measurements
7. Plotly interactive line plot
8. Plotly scatter with custom titles and colors
9. Side-by-side matplotlib vs plotly comparison
10. Real physics example (projectile motion, energy, etc)

---

## Notes / Log

**2026-01-09** - Plan created. Research complete:
- Reviewed matplotlib.py (451 lines): setup_matplotlib(), plot, scatter, bar, hist, errorbar
- Reviewed plotly.py (374 lines): line, scatter, bar, histogram, scatter_with_errors
- Reviewed test_visualization.py (357 lines): Full test coverage for all functions
- Reviewed units.md: Similar structure and style to follow
- Ready for implementation phase

---

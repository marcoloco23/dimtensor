# Plan: Data Analysis Example Notebook (04_data_analysis.ipynb)

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Create an educational Jupyter notebook demonstrating real-world data analysis workflows using dimtensor, including loading physical constants from CODATA, analyzing NASA exoplanet data, performing unit conversions, visualization, and statistical analysis.

---

## Background

The dimtensor library provides unit-aware tensors for scientific computing. This example notebook will showcase practical data analysis workflows that scientists and engineers encounter, demonstrating how dimtensor catches unit errors, simplifies conversions, and integrates with the scientific Python ecosystem (pandas, matplotlib, numpy).

This is the fourth example notebook in the series, aimed at intermediate users who want to see dimtensor in real data analysis scenarios.

---

## Approach

### Option A: Focus on Real External Data Sources
- Download NASA exoplanet archive data
- Use CODATA constants from dimtensor
- Show integration with pandas, matplotlib
- Pros: Most realistic, shows real-world workflow
- Cons: Requires network access, external dependencies

### Option B: Synthetic/Bundled Data
- Create synthetic exoplanet-like data
- Use only dimtensor built-in features
- Pros: No external dependencies, always works
- Cons: Less authentic

### Decision: **Option A with fallback to synthetic data**

Use real NASA exoplanet data but provide synthetic fallback data in the notebook for offline use. This gives users the best of both worlds - they see real workflows but can run it anywhere.

---

## Implementation Steps

1. [ ] Create examples/ directory structure
2. [ ] Create 04_data_analysis.ipynb with 25-30 cells:
   - **Introduction (1-2 cells)**: What we'll analyze, imports
   - **Section 1: CODATA Constants (3-4 cells)**:
     - Load universal constants (c, G, h)
     - Load particle masses (m_e, m_p)
     - Show constant metadata (uncertainty, references)
     - Demonstrate dimensional correctness
   - **Section 2: Loading Data (4-5 cells)**:
     - Load NASA exoplanet data (or synthetic fallback)
     - Convert pandas DataFrame columns to DimArray
     - Show unit metadata preservation
     - Display first few rows with units
   - **Section 3: Unit Conversions (4-5 cells)**:
     - Convert planetary masses (M_jup to M_earth to kg)
     - Convert orbital distances (AU to km to m)
     - Convert periods (days to years to seconds)
     - Show automatic simplification (kg·m/s² → N)
   - **Section 4: Statistical Analysis (5-6 cells)**:
     - Calculate mean, std, median of orbital periods
     - Find min/max planetary masses
     - Compute derived quantities (surface gravity, escape velocity)
     - Filter data by physical criteria
   - **Section 5: Visualization (4-5 cells)**:
     - Mass-radius relationship scatter plot
     - Period-distance histogram
     - Use matplotlib with unit-aware labels
     - Customize plot with dimtensor display config
   - **Section 6: Data Export (3-4 cells)**:
     - Save to JSON with units
     - Save to pandas DataFrame
     - Save to HDF5 (if h5py available)
     - Round-trip test (save and reload)
3. [ ] Add markdown cells with explanations
4. [ ] Add error handling examples (dimension mismatches)
5. [ ] Test notebook end-to-end
6. [ ] Verify all outputs render correctly

---

## Files to Modify

| File | Change |
|------|--------|
| examples/04_data_analysis.ipynb | Create new Jupyter notebook (~25-30 cells) |
| examples/data/exoplanets_sample.csv | Optional: Bundled sample data for offline use |
| examples/README.md | Add entry for notebook 04 (if exists) |

---

## Testing Strategy

How will we verify this works?

- [ ] Run notebook top-to-bottom without errors
- [ ] Verify all plots render correctly
- [ ] Test both online (NASA data) and offline (synthetic) modes
- [ ] Verify unit conversions are mathematically correct
- [ ] Test save/load round-trips preserve units
- [ ] Check that error examples raise expected DimensionError
- [ ] Validate against real exoplanet database values
- [ ] Test in fresh environment with minimal dependencies

---

## Risks / Edge Cases

- **Risk**: NASA API changes or is unavailable
  - **Mitigation**: Include synthetic fallback data in notebook

- **Risk**: Missing optional dependencies (h5py, pandas, matplotlib)
  - **Mitigation**: Wrap in try/except, show graceful degradation

- **Edge case**: Exoplanet data contains NaN values
  - **Handling**: Show how to filter/clean data with dimtensor

- **Edge case**: Very large/small values (Jupiter mass vs electron mass)
  - **Handling**: Demonstrate automatic unit scaling, scientific notation

- **Risk**: Notebook too long/complex for beginners
  - **Mitigation**: Clear section headers, progressive complexity

---

## Definition of Done

- [ ] Notebook created with 25-30 cells
- [ ] All sections implemented as specified
- [ ] Notebook executes without errors
- [ ] Plots render correctly
- [ ] Unit conversions validated
- [ ] Save/load round-trips work
- [ ] Error examples demonstrate dimension checking
- [ ] Markdown documentation is clear
- [ ] CONTINUITY.md updated

---

## Notes / Log

**Design decisions**:

1. **Data source**: NASA Exoplanet Archive (https://exoplanetarchive.ipac.caltech.edu/)
   - Use TAP API or CSV download
   - Fields: pl_name, pl_bmassj (mass in Jupiter masses), pl_orbsmax (orbital distance in AU), pl_orbper (period in days)

2. **Synthetic fallback**: 20-30 synthetic exoplanets with realistic parameters
   - Use power law distributions matching real exoplanet statistics
   - Include some edge cases (hot Jupiters, super-Earths)

3. **Statistical analysis examples**:
   - Mean orbital period with uncertainty propagation
   - Mass-radius relationship (if radius data available)
   - Escape velocity calculation: v_esc = sqrt(2*G*M/R)
   - Surface gravity: g = G*M/R²

4. **Visualization approach**:
   - Use matplotlib with dimtensor-aware axis labels
   - Extract numeric values with arr.to(unit)._data
   - Show unit symbols in axis labels: "Mass (M_jup)", "Distance (AU)"

5. **IO formats to demonstrate**:
   - JSON: Universal, human-readable
   - pandas: Integration with data science ecosystem
   - HDF5: High-performance, large datasets (optional)

6. **Error examples to include**:
   - Adding mass + distance (dimension error)
   - Using dimensional value where dimensionless expected (e.g., exp(distance))
   - Show error messages are clear and helpful

---

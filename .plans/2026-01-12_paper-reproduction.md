# Plan: Paper Reproduction Framework for dimtensor

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner (agent)

---

## Goal

Create a framework that helps researchers reproduce physics papers with unit-aware computations. The framework will store paper metadata, equations, published values, and compare computed results against published values with proper unit handling and statistical analysis.

---

## Background

dimtensor has evolved into a comprehensive physics ML platform with:
- Unit-aware arrays (DimArray) with uncertainty propagation
- 246 physics equations in the database
- Analysis tools (Buckingham Pi, sensitivity analysis, error budgets)
- Integration with SciPy, SymPy, PyTorch, JAX
- Dataset registry with real physics data loaders

However, there's no structured way to:
- **Reproduce published results** from physics papers
- **Compare computed vs published values** with proper unit conversion
- **Track reproduction accuracy** and identify discrepancies
- **Document reproduction attempts** with metadata and citations
- **Share reproduction protocols** with the community

This framework addresses the "reproducibility crisis" in computational physics by making paper reproduction a first-class workflow. It's essential for:
- **Verification**: Validating published results before building on them
- **Education**: Teaching students to critically evaluate literature
- **Research integrity**: Detecting errors in published computations
- **Method benchmarking**: Comparing numerical methods against known results
- **Community building**: Creating a repository of verified reproductions

Related work:
- Jupyter notebooks are popular for reproduction but lack structure
- ReScience journal publishes reproductions but needs tooling
- dimtensor's equation database + unit system is ideal foundation

---

## Approach

### Option A: Notebook-Based Approach

- **Description**: Provide Jupyter notebook templates with standardized sections for paper metadata, equations, data, comparison
- **Pros**:
  - Familiar workflow for scientists
  - Flexible (can include plots, derivations)
  - Easy to share via nbviewer/Binder
  - No new data structures needed
- **Cons**:
  - Lacks programmatic access to results
  - Hard to aggregate across multiple reproductions
  - No standardized comparison metrics
  - Difficult to build dashboards/search functionality

### Option B: Class-Based API with Optional Notebooks

- **Description**: Create Paper and ReproductionResult classes for structured metadata, then optionally export to notebooks for documentation
- **Pros**:
  - Programmatic access enables automation
  - Can build search/comparison tools
  - Standardized comparison metrics
  - Export to multiple formats (JSON, notebook, PDF)
  - Machine-readable for meta-analysis
- **Cons**:
  - More upfront implementation effort
  - Users learn new API (but simple dataclasses)
  - Less flexible than pure notebooks

### Option C: Database-Backed Repository

- **Description**: Full repository system with SQLite/PostgreSQL backend for storing reproductions
- **Pros**:
  - Scalable to thousands of reproductions
  - Advanced querying capabilities
  - Multi-user collaboration
  - Version control for reproductions
- **Cons**:
  - Major infrastructure project
  - Overkill for v5.0.0
  - Requires server/hosting
  - Complex deployment

### Decision: Option B (Class-Based API) with export to notebooks

**Rationale**:
1. **Best of both worlds**: Structured API + notebook export for documentation
2. **Aligns with dimtensor patterns**: Similar to ModelCard, DatasetInfo, Equation classes
3. **Enables automation**: Can programmatically compare 100s of reproductions
4. **Future-proof**: Can add database backend later without API changes
5. **Immediate value**: Works standalone without infrastructure

**Implementation philosophy**:
- Simple dataclasses for metadata (like ModelCard, DatasetInfo)
- Functions for comparison logic (like error_budget, sensitivity analysis)
- Export utilities to various formats
- Option C (database) can be added in v5.1.0+ as `research.repository` module

---

## Implementation Steps

### Phase 1: Core Data Structures

1. [ ] Create `src/dimtensor/research/` subpackage
   - [ ] `__init__.py` with main exports
   - [ ] `paper.py` for Paper class
   - [ ] `reproduction.py` for ReproductionResult class
   - [ ] `comparison.py` for comparison utilities

2. [ ] Implement Paper class
   ```python
   @dataclass
   class Paper:
       """Metadata for a physics paper to be reproduced.

       Attributes:
           title: Paper title
           authors: List of author names
           doi: Digital Object Identifier
           year: Publication year
           journal: Journal name
           abstract: Brief abstract/summary
           equations: Dict mapping equation names to Equation objects
           published_values: Dict mapping quantity names to DimArrays
           units_used: Dict mapping quantity to original units in paper
           methods: List of numerical methods used
           assumptions: List of key assumptions
           tags: List of tags for filtering
       """
   ```

3. [ ] Implement ReproductionResult class
   ```python
   @dataclass
   class ReproductionResult:
       """Results from reproducing a paper's computations.

       Attributes:
           paper: Reference to Paper object
           computed_values: Dict mapping quantity names to computed DimArrays
           comparison_metrics: Dict with accuracy statistics
           reproduction_date: When reproduction was performed
           reproducer: Name/email of person reproducing
           code_repository: Link to reproduction code
           notes: Additional notes about reproduction process
           status: SUCCESS | PARTIAL | FAILED
           discrepancies: List of significant differences found
       """
   ```

4. [ ] Implement comparison functions
   - [ ] `compare_values(published, computed, rtol=1e-3, atol=None)` → ComparisonResult
   - [ ] Handle unit conversion automatically
   - [ ] Compute relative error, absolute error
   - [ ] Propagate uncertainties if available
   - [ ] Flag significant discrepancies

5. [ ] Implement ComparisonResult class
   ```python
   @dataclass
   class ComparisonResult:
       """Results from comparing published vs computed values.

       Attributes:
           quantity_name: Name of compared quantity
           published_value: Original published value (DimArray)
           computed_value: Reproduced value (DimArray)
           absolute_error: |computed - published| with units
           relative_error: |computed - published| / |published|
           matches: True if within tolerance
           tolerance_used: (rtol, atol) tuple
           unit_conversion_applied: True if units were different
       """
   ```

### Phase 2: Analysis & Reporting

6. [ ] Implement statistical analysis functions
   - [ ] `analyze_reproduction(result: ReproductionResult)` → dict
   - [ ] Compute summary statistics (mean error, max error, RMS)
   - [ ] Identify systematic vs random discrepancies
   - [ ] Check if errors are within uncertainty bounds
   - [ ] Test for unit conversion errors specifically

7. [ ] Implement report generation
   - [ ] `generate_report(result, format='markdown')` → str
   - [ ] Formats: markdown, HTML, LaTeX, PDF
   - [ ] Include table of all compared values
   - [ ] Highlight discrepancies
   - [ ] Add citations and metadata
   - [ ] Include equations used

8. [ ] Implement visualization functions
   - [ ] `plot_comparison(result, quantity=None)` → matplotlib figure
   - [ ] Scatter plot: computed vs published
   - [ ] Residual plot: error vs magnitude
   - [ ] Error distribution histogram
   - [ ] Use dimtensor.visualization for consistent styling

### Phase 3: Export & Templates

9. [ ] Implement export to Jupyter notebook
   - [ ] `result.to_notebook(path)` creates IPython notebook
   - [ ] Includes all metadata, equations, comparisons
   - [ ] Executable cells for verification
   - [ ] Formatted markdown sections

10. [ ] Create reproduction templates
    - [ ] `create_reproduction_template(paper: Paper)` → notebook
    - [ ] Pre-filled metadata sections
    - [ ] Equation definitions from paper.equations
    - [ ] Placeholder code cells for computation
    - [ ] Automatic comparison cells

11. [ ] Implement serialization
    - [ ] `paper.to_json(path)` and `from_json(path)`
    - [ ] `result.to_json(path)` and `from_json(path)`
    - [ ] Handle DimArray serialization via existing io.json
    - [ ] Include schema version for compatibility

### Phase 4: Examples & Documentation

12. [ ] Create example reproductions
    - [ ] Classic mechanics: Millikan oil drop experiment
    - [ ] Thermodynamics: Clausius-Clapeyron equation verification
    - [ ] Quantum: Hydrogen energy levels
    - [ ] Relativity: Schwarzschild radius calculation
    - [ ] Cosmology: Hubble constant from distance ladder

13. [ ] Add CLI commands
    - [ ] `dimtensor reproduce --paper paper.json --notebook output.ipynb`
    - [ ] `dimtensor compare --published pub.json --computed comp.json`
    - [ ] `dimtensor papers list` (show registered papers)

14. [ ] Create comprehensive documentation
    - [ ] User guide: "Reproducing Physics Papers with dimtensor"
    - [ ] API reference for research module
    - [ ] Tutorial notebook: Step-by-step reproduction
    - [ ] Best practices guide

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/research/__init__.py` | Create subpackage, export Paper, ReproductionResult, etc. |
| `src/dimtensor/research/paper.py` | Implement Paper dataclass |
| `src/dimtensor/research/reproduction.py` | Implement ReproductionResult dataclass |
| `src/dimtensor/research/comparison.py` | Implement ComparisonResult and comparison functions |
| `src/dimtensor/research/analysis.py` | Implement statistical analysis functions |
| `src/dimtensor/research/reporting.py` | Implement report generation |
| `src/dimtensor/research/export.py` | Implement notebook export |
| `src/dimtensor/research/templates.py` | Reproduction templates |
| `src/dimtensor/__init__.py` | Add research module to imports |
| `src/dimtensor/__main__.py` | Add reproduce, compare CLI commands |
| `tests/test_paper.py` | Test Paper class and serialization |
| `tests/test_reproduction.py` | Test ReproductionResult and comparison |
| `tests/test_research_analysis.py` | Test statistical analysis |
| `tests/test_research_export.py` | Test notebook export |
| `examples/paper_reproduction/` | Directory with example reproductions |
| `examples/paper_reproduction/millikan_oil_drop.ipynb` | Classic experiment |
| `examples/paper_reproduction/hydrogen_spectrum.ipynb` | Quantum example |
| `docs/guide/paper-reproduction.md` | User guide |

---

## Testing Strategy

### Unit Tests

**test_paper.py**:
- [ ] Paper class initialization with all fields
- [ ] Validation of DOI format
- [ ] Serialization to/from JSON
- [ ] Equation attachment and retrieval
- [ ] Published values with different units

**test_reproduction.py**:
- [ ] ReproductionResult class initialization
- [ ] Status enum values (SUCCESS, PARTIAL, FAILED)
- [ ] Discrepancy tracking
- [ ] Serialization to/from JSON

**test_comparison.py**:
- [ ] compare_values() with identical values → matches=True
- [ ] compare_values() with different units → automatic conversion
- [ ] compare_values() with discrepancy → matches=False
- [ ] Relative error calculation
- [ ] Absolute error with proper units
- [ ] Tolerance parameters (rtol, atol)
- [ ] Array-valued comparisons (element-wise)
- [ ] Uncertainty propagation in comparison

**test_research_analysis.py**:
- [ ] analyze_reproduction() summary statistics
- [ ] Systematic error detection
- [ ] Random error estimation
- [ ] Unit conversion error identification
- [ ] Uncertainty bounds checking

**test_research_export.py**:
- [ ] Notebook export creates valid .ipynb file
- [ ] Markdown report generation
- [ ] HTML export with plots
- [ ] Template creation with pre-filled metadata

### Integration Tests

**test_full_reproduction.py**:
- [ ] End-to-end reproduction workflow
  1. Create Paper object with equations and published values
  2. Compute results using dimtensor
  3. Create ReproductionResult
  4. Compare values
  5. Generate report
  6. Export to notebook
- [ ] Verify exported notebook is executable
- [ ] Verify report contains all expected sections

### Example Reproductions (Manual Verification)

**Millikan Oil Drop**:
- [ ] Published: e = 1.592×10⁻¹⁹ C (1913 value)
- [ ] Modern: e = 1.602176634×10⁻¹⁹ C (exact, SI 2019)
- [ ] Reproduction should flag systematic error in original

**Hydrogen Spectrum**:
- [ ] Rydberg formula: 1/λ = R_∞ (1/n₁² - 1/n₂²)
- [ ] Compare computed wavelengths vs published spectral lines
- [ ] Should match within measurement uncertainty

**Schwarzschild Radius**:
- [ ] r_s = 2GM/c²
- [ ] Solar mass: r_s ≈ 2.95 km
- [ ] Earth mass: r_s ≈ 8.87 mm
- [ ] Verify unit handling (m, km, cm)

### Edge Cases

- [ ] Paper with no published values (template only)
- [ ] Missing uncertainties in published values
- [ ] Incompatible units (e.g., eV vs Joules - should convert)
- [ ] Array vs scalar mismatches
- [ ] Empty reproduction (no computed values yet)
- [ ] Multiple reproductions of same paper (tracking)

---

## Risks / Edge Cases

### Risk 1: Unit ambiguity in published papers
**Problem**: Papers often don't specify units explicitly or use non-standard conventions
**Mitigation**:
- Require explicit units in Paper.published_values
- Document unit assumption in Paper.units_used field
- Flag unit conversions clearly in reports
- Allow manual override if automatic detection fails

### Risk 2: Numerical method differences
**Problem**: Different numerical methods (finite diff vs spectral) give different results
**Mitigation**:
- Document method in ReproductionResult.notes
- Provide tolerances appropriate for numerical methods
- Don't expect bit-level exactness
- Focus on physical agreement, not numerical identity

### Risk 3: Missing information in papers
**Problem**: Papers may not report enough detail to reproduce
**Mitigation**:
- Paper.assumptions field documents what we had to assume
- ReproductionResult.notes explains gaps
- Status=PARTIAL for incomplete reproductions
- Consider this "documentation of non-reproducibility"

### Risk 4: Comparison with uncertain published values
**Problem**: Published values often have uncertainties, need error budget comparison
**Mitigation**:
- Support uncertainty in both published and computed values
- Use dimtensor's existing uncertainty propagation
- Check if discrepancy is within combined uncertainty
- Report: "3σ discrepancy" vs "within error bars"

### Risk 5: Large arrays/datasets
**Problem**: Some papers involve large simulation data
**Mitigation**:
- Don't store full arrays in Paper/Result objects, just summaries
- Use external files for large data, reference by path
- Compare statistical summaries rather than point-by-point
- Provide downsampled data for quick checks

### Edge Case: Time-dependent results
**Problem**: Simulations produce time series, not single values
**Mitigation**:
- Store time series as DimArrays with time dimension
- Compare at specific time points or via integrated metrics (RMS error over time)

### Edge Case: Figures without data tables
**Problem**: Papers show plots but don't tabulate exact values
**Mitigation**:
- Allow approximate extraction from figures (with large tolerance)
- Document this uncertainty in notes
- Focus on qualitative agreement for plots

### Edge Case: Different conventions
**Problem**: Sign conventions (charge, phase), coordinate systems differ
**Mitigation**:
- Document conventions in Paper.assumptions
- Provide conversion functions for common convention changes
- Consider both matches in comparison

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] Core functionality works:
  - [ ] Paper objects can be created and serialized
  - [ ] ReproductionResult tracks computed values and comparisons
  - [ ] compare_values() correctly handles unit conversion and tolerances
  - [ ] Reports generate with all metadata and comparisons
  - [ ] Notebook export creates executable notebooks
- [ ] Tests pass:
  - [ ] Unit tests achieve >90% coverage of new code
  - [ ] Integration tests verify end-to-end workflow
  - [ ] Example reproductions match published values (within tolerance)
  - [ ] Edge cases handled gracefully
- [ ] Documentation complete:
  - [ ] API docstrings follow dimtensor conventions
  - [ ] User guide explains reproduction workflow
  - [ ] Tutorial notebook demonstrates full reproduction
  - [ ] Example reproductions for 3+ classic papers
- [ ] CLI commands work:
  - [ ] `dimtensor reproduce` creates templates
  - [ ] `dimtensor compare` runs comparisons
  - [ ] `dimtensor papers list` shows registered papers
- [ ] Code review:
  - [ ] Follows dimtensor patterns (dataclasses, unit safety)
  - [ ] Type hints complete
  - [ ] Proper integration with existing modules (equations, io, visualization)
- [ ] CONTINUITY.md updated with task completion

---

## Notes / Log

### Research Findings

**Existing dimtensor infrastructure to leverage**:
1. **Equations database**: 246 equations across domains - can attach to Paper
2. **Unit conversion**: Automatic via DimArray.to() method
3. **Uncertainty propagation**: Built into DimArray operations
4. **Error budgets**: `uncertainty.error_budget` module for contribution analysis
5. **Serialization**: `io.json` handles DimArray → JSON → DimArray
6. **Visualization**: matplotlib/plotly integration for plots
7. **Dataclass patterns**: ModelCard, DatasetInfo, Equation are good templates

**Similar patterns in codebase**:
- **DatasetInfo**: Metadata + registry + loader pattern
- **ModelCard**: Structured metadata with to_dict/from_dict
- **ErrorBudget**: Container class with analysis + visualization methods
- **Equation**: Database of physics knowledge with search/filter

**Key dimtensor patterns to follow**:
1. Dataclasses for structured data
2. to_dict()/from_dict() for serialization
3. Type hints throughout
4. Optional dependencies (pandas, matplotlib) with HAS_* guards
5. Export to multiple formats
6. Integration with existing modules

**Reproducibility research**:
- **ReScience journal**: Publishes computational reproductions
- **Nature reproducibility crisis**: ~70% of researchers failed to reproduce others' results
- **ACM badges**: Artifacts Available, Artifacts Evaluated, Results Reproduced
- **Jupyter notebooks**: Popular but lack structure for systematic comparison
- **FAIR principles**: Findable, Accessible, Interoperable, Reusable

### Design Decisions Made

1. **Class-based API** over pure notebooks
   - Enables programmatic access and automation
   - Can export to notebooks for documentation
   - Aligns with dimtensor's structured approach

2. **Separate Paper and ReproductionResult** classes
   - Paper = static metadata (title, DOI, published values)
   - ReproductionResult = dynamic computation results
   - One paper can have multiple reproduction attempts
   - Clear separation of concerns

3. **Automatic unit conversion** in comparisons
   - dimtensor's strength is unit handling
   - Published values might use different units than computed
   - Comparison should "just work" across unit systems
   - Flag when conversion happens (for transparency)

4. **Uncertainty-aware comparisons**
   - Published values often have error bars
   - Use dimtensor's uncertainty propagation
   - Check if discrepancy is significant (beyond combined uncertainty)
   - Report confidence level of agreement/disagreement

5. **Multiple export formats**
   - JSON for machine-readable storage
   - Notebooks for interactive documentation
   - Markdown/HTML for reports
   - Flexibility for different use cases

6. **Status enum** for reproduction attempts
   - SUCCESS: All values match within tolerance
   - PARTIAL: Some values match, some don't
   - FAILED: Significant discrepancies or couldn't compute
   - Honest reporting of reproducibility state

7. **Extensible for future database backend**
   - Current design uses in-memory objects + file serialization
   - API won't change if we add database later
   - Just add research.repository module with DB operations
   - Gradual migration path

---

## Future Enhancements (Post-v5.0.0)

### v5.1.0: Repository & Collaboration
- [ ] Database backend for storing reproductions
- [ ] Web interface for browsing reproductions
- [ ] Community contributions via GitHub PRs
- [ ] Voting/verification system for reproductions
- [ ] DOI minting for reproduction artifacts

### v5.2.0: Advanced Analysis
- [ ] Automatic equation extraction from paper PDFs (ML-based)
- [ ] Parameter fitting to match published results
- [ ] Sensitivity analysis: which parameters most affect match
- [ ] Monte Carlo sampling for robustness testing
- [ ] Outlier detection in reproduction attempts

### v5.3.0: Integration with External Services
- [ ] Crossref API integration for DOI lookup
- [ ] arXiv API for paper metadata
- [ ] Zenodo upload for reproduction archives
- [ ] ORCID integration for researcher attribution
- [ ] ReScience journal submission workflow

---

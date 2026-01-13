# Plan: Accessibility Audit

**Date**: 2026-01-13
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Design and implement comprehensive accessibility improvements for dimtensor, ensuring the library is usable by physicists and researchers with visual impairments, color vision deficiencies, and those using assistive technologies.

---

## Background

### Why is this needed?

Currently, dimtensor provides no specific accessibility features. As an educational and research tool (v5.1.0 theme: "Physics for everyone"), the library should be accessible to all scientists regardless of visual abilities or assistive technology needs.

### Context from existing codebase

1. **Current output system**:
   - `DimArray.__str__()` and `__repr__()` format output using `DisplayOptions` (config.py)
   - No screen reader-specific hints or semantic information
   - No high-contrast or alternative display modes

2. **Current visualization**:
   - matplotlib.py and plotly.py integration exists
   - Uses default color schemes (not colorblind-safe)
   - No configuration for accessible color palettes

3. **Configuration system**:
   - config.py has `DisplayOptions` and `InferenceOptions` dataclasses
   - Context managers for temporary settings already exist
   - Pattern established for global options with set/reset functions

### Accessibility requirements

**Color vision deficiencies (CVD)**:
- Deuteranopia (~5% of males): red-green confusion (green-blind)
- Protanopia (~2% of males): red-green confusion (red-blind)
- Tritanopia (~0.01%): blue-yellow confusion (blue-blind)

**Screen reader compatibility**:
- Semantic markup for structured output
- Descriptive labels for data
- Avoid ASCII art that doesn't read well

**High-contrast needs**:
- Sufficient contrast ratios (WCAG 2.1: 4.5:1 for text, 3:1 for graphics)
- Configurable themes for different lighting conditions

---

## Approach

### Option A: Minimal - Color Palettes Only
- **Description**: Add colorblind-safe palettes to visualization modules
- **Pros**:
  - Quick to implement
  - Addresses most common accessibility issue
  - Low risk
- **Cons**:
  - Doesn't help screen reader users
  - Doesn't address text output readability
  - Incomplete solution

### Option B: Comprehensive - Full Accessibility System
- **Description**: Create dedicated accessibility module with:
  - Accessible output formatters (text/HTML/Jupyter)
  - Colorblind-safe color palettes
  - High-contrast display modes
  - Screen reader hints and semantic markup
  - Configuration system for user preferences
- **Pros**:
  - Complete accessibility solution
  - Future-proof architecture
  - Aligns with v5.1.0 "Physics for everyone" theme
  - Can be extended for i18n later
- **Cons**:
  - More implementation work
  - Requires careful testing with real assistive technology

### Option C: Hybrid - Pragmatic Approach
- **Description**: Implement Option B but phase it:
  - Phase 1: Color palettes + basic screen reader support
  - Phase 2: High-contrast modes + HTML output
- **Pros**:
  - Delivers value quickly
  - Allows for user feedback
  - Spreads work across releases
- **Cons**:
  - Architecture must be correct from start
  - Partial solution in v5.1.0

### Decision: Option B - Comprehensive Solution

**Rationale**:
1. v5.1.0 theme is "Education & Accessibility" - this is the right time for full solution
2. Config system pattern already established, can reuse architecture
3. Better to design holistically than patch incrementally
4. Educational use case (task #260-262) will benefit from complete solution
5. Relatively small codebase addition (~1,500 lines total)

---

## Implementation Steps

### Phase 1: Architecture & Configuration
1. [ ] Create `src/dimtensor/accessibility/` module folder
2. [ ] Create `__init__.py` with public API exports
3. [ ] Create `config.py` with `AccessibilityOptions` dataclass
4. [ ] Add accessibility options to main `config.py`:
   - `colorblind_mode: str | None` (deuteranopia, protanopia, tritanopia, None)
   - `high_contrast: bool` (default False)
   - `screen_reader_mode: bool` (default False)
   - `color_palette: str` (default, colorblind_safe, high_contrast, grayscale)

### Phase 2: Colorblind-Safe Palettes
5. [ ] Create `accessibility/colors.py` module
6. [ ] Implement color palette definitions:
   - `COLORBLIND_SAFE_QUALITATIVE`: 8 colors (Wong 2011, Okabe & Ito)
   - `COLORBLIND_SAFE_SEQUENTIAL`: Blue-based sequential
   - `COLORBLIND_SAFE_DIVERGING`: Blue-Orange diverging
   - `HIGH_CONTRAST`: Black/white with yellow/cyan accents
   - `GRAYSCALE`: Pure grayscale with pattern differentiation
7. [ ] Implement CVD simulation functions:
   - `simulate_deuteranopia(color)`: Simulate green-blindness
   - `simulate_protanopia(color)`: Simulate red-blindness
   - `simulate_tritanopia(color)`: Simulate blue-blindness
8. [ ] Implement palette validation:
   - `check_distinguishability(palette, cvd_type)`: Verify colors remain distinct
   - `suggest_palette(n_colors, cvd_type)`: Recommend palette for use case

### Phase 3: Visualization Integration
9. [ ] Create `accessibility/matplotlib_theme.py`
10. [ ] Implement `apply_colorblind_safe_theme(palette_name)` for matplotlib
11. [ ] Update `visualization/matplotlib.py`:
    - Respect `config.accessibility.color_palette` setting
    - Add `palette` parameter to plot functions
    - Default to colorblind-safe if `colorblind_mode` is set
12. [ ] Create `accessibility/plotly_theme.py`
13. [ ] Implement `get_plotly_colorway(palette_name)` for plotly
14. [ ] Update `visualization/plotly.py`:
    - Respect `config.accessibility.color_palette` setting
    - Add `palette` parameter to plot functions

### Phase 4: Text Output Accessibility
15. [ ] Create `accessibility/formatters.py` module
16. [ ] Implement `ScreenReaderFormatter` class:
    - `format_dimarray(arr)`: "3.14 meters" instead of "3.14 m"
    - `format_with_uncertainty(arr)`: "3.14 plus or minus 0.1 meters"
    - `format_equation(eq)`: Spell out symbols
17. [ ] Implement `HighContrastFormatter` class:
    - Use ASCII box-drawing characters for structure
    - Bold numerical values
    - Extra spacing for readability
18. [ ] Update `DimArray.__str__()` to respect `screen_reader_mode`
19. [ ] Update `DimArray._repr_html_()` for Jupyter notebooks:
    - Add ARIA labels
    - Semantic HTML (not just `<div>`)
    - Respect high-contrast mode with CSS

### Phase 5: HTML/Jupyter Support
20. [ ] Create `accessibility/html.py` module
21. [ ] Implement `HTMLFormatter` class with:
    - `format_dimarray_html(arr, options)`: Semantic HTML output
    - ARIA labels: `aria-label="distance: 3.14 meters"`
    - `role="math"` for scientific notation
    - CSS classes for user styling
22. [ ] Implement `_repr_html_()` method for DimArray:
    - Use semantic `<table>` for multi-dimensional arrays
    - Add `<caption>` with unit information
    - Include screen reader text: `<span class="sr-only">`
23. [ ] Create high-contrast CSS for Jupyter:
    - `accessibility/jupyter.css` with themes
    - Inject CSS based on `high_contrast` setting

### Phase 6: Documentation & Examples
24. [ ] Create `docs/guide/accessibility.md`:
    - Color vision deficiency support
    - Screen reader configuration
    - High-contrast mode usage
    - Best practices for inclusive physics visualization
25. [ ] Create `examples/accessibility_demo.ipynb`:
    - Demonstrate all palettes
    - Show before/after for CVD simulation
    - Screen reader output examples
26. [ ] Update main docs with accessibility features
27. [ ] Add accessibility section to README.md

### Phase 7: Testing
28. [ ] Create `tests/test_accessibility_colors.py`:
    - Test palette definitions
    - Test CVD simulation functions
    - Test distinguishability checks
29. [ ] Create `tests/test_accessibility_formatters.py`:
    - Test screen reader formatter
    - Test high-contrast formatter
    - Test HTML output
30. [ ] Create `tests/test_accessibility_integration.py`:
    - Test matplotlib integration
    - Test plotly integration
    - Test configuration changes
31. [ ] Manual testing with actual screen readers:
    - NVDA (Windows)
    - JAWS (Windows)
    - VoiceOver (macOS/iOS)
    - TalkBack (Android)
    - Orca (Linux)

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/accessibility/__init__.py` | NEW: Public API exports |
| `src/dimtensor/accessibility/colors.py` | NEW: Colorblind-safe palettes, CVD simulation |
| `src/dimtensor/accessibility/formatters.py` | NEW: Screen reader & high-contrast formatters |
| `src/dimtensor/accessibility/html.py` | NEW: HTML/Jupyter output with ARIA |
| `src/dimtensor/accessibility/matplotlib_theme.py` | NEW: Matplotlib theme integration |
| `src/dimtensor/accessibility/plotly_theme.py` | NEW: Plotly theme integration |
| `src/dimtensor/accessibility/jupyter.css` | NEW: High-contrast CSS for Jupyter |
| `src/dimtensor/config.py` | MODIFY: Add `AccessibilityOptions` dataclass |
| `src/dimtensor/core/dimarray.py` | MODIFY: Update `__str__()`, add `_repr_html_()` |
| `src/dimtensor/visualization/matplotlib.py` | MODIFY: Add palette parameter, respect config |
| `src/dimtensor/visualization/plotly.py` | MODIFY: Add palette parameter, respect config |
| `tests/test_accessibility_colors.py` | NEW: Test color palettes and CVD simulation |
| `tests/test_accessibility_formatters.py` | NEW: Test formatters |
| `tests/test_accessibility_integration.py` | NEW: Test visualization integration |
| `docs/guide/accessibility.md` | NEW: Accessibility documentation |
| `examples/accessibility_demo.ipynb` | NEW: Interactive accessibility demo |

**Estimated**: ~1,500 lines of implementation code, ~600 lines of tests, ~800 lines of documentation

---

## Testing Strategy

### Unit Tests
- [ ] Color palette validation (all colors defined, correct format)
- [ ] CVD simulation produces different colors
- [ ] Distinguishability checks work correctly
- [ ] Screen reader formatter produces readable text
- [ ] High-contrast formatter has enhanced spacing
- [ ] HTML formatter produces valid HTML with ARIA attributes
- [ ] Configuration changes propagate correctly

### Integration Tests
- [ ] Matplotlib respects accessibility settings
- [ ] Plotly respects accessibility settings
- [ ] DimArray output changes with screen_reader_mode
- [ ] Jupyter HTML output includes ARIA labels
- [ ] Context managers work for temporary accessibility changes

### Visual Tests (manual)
- [ ] Generate plots with each palette
- [ ] Simulate CVD for each palette (using online tools)
- [ ] Verify colors remain distinguishable
- [ ] Check contrast ratios meet WCAG 2.1 AA (4.5:1)

### Screen Reader Tests (manual)
- [ ] Test with NVDA on Windows
- [ ] Test with VoiceOver on macOS
- [ ] Test Jupyter notebook output
- [ ] Verify equation reading is comprehensible
- [ ] Check table navigation works properly

### User Testing
- [ ] Get feedback from users with CVD (if possible)
- [ ] Get feedback from screen reader users (if possible)
- [ ] Iterate based on real-world usage

---

## Risks / Edge Cases

### Risk 1: Color palette too limited
- **Description**: 8-color colorblind-safe palette may not be enough for complex visualizations
- **Mitigation**:
  - Provide multiple palette options (qualitative, sequential, diverging)
  - Allow palette extension with patterns/markers
  - Document when to use which palette

### Risk 2: Screen reader output too verbose
- **Description**: Fully spelling out all scientific notation could be overwhelming
- **Mitigation**:
  - Make it configurable (verbosity levels)
  - Provide both brief and detailed formats
  - Use ARIA to allow screen reader users to skip details

### Risk 3: HTML output breaks existing notebooks
- **Description**: Adding `_repr_html_()` might change how existing notebooks display
- **Mitigation**:
  - Default to current behavior
  - Only activate when accessibility options are set
  - Provide opt-out mechanism

### Risk 4: Performance impact of CVD simulation
- **Description**: Real-time CVD simulation could slow down plotting
- **Mitigation**:
  - Pre-compute palettes, don't simulate on-the-fly
  - Cache palette transformations
  - Simulation is for validation only, not runtime

### Risk 5: Matplotlib/Plotly theme conflicts
- **Description**: User custom themes might conflict with accessibility themes
- **Mitigation**:
  - Apply themes non-destructively
  - Document how to combine with custom themes
  - Provide override mechanism

### Edge Case 1: Grayscale printing
- **Handling**: GRAYSCALE palette + marker differentiation for plots

### Edge Case 2: Multiple CVD types in classroom
- **Handling**: Default to palette that works for all types (Wong 2011)

### Edge Case 3: Dark mode Jupyter notebooks
- **Handling**: Detect theme, adjust high-contrast CSS accordingly

### Edge Case 4: Non-visual data exploration
- **Handling**: Provide sonification hooks (future work, document API)

### Edge Case 5: Complex multi-panel figures
- **Handling**: Document best practices for accessible multi-panel design

---

## Definition of Done

- [ ] All implementation steps complete (Phase 1-5)
- [ ] Documentation complete (Phase 6)
- [ ] All unit tests pass (>80% coverage for new modules)
- [ ] All integration tests pass
- [ ] Manual testing with at least 2 screen readers
- [ ] Visual testing confirms palettes are distinguishable
- [ ] Contrast ratios measured and meet WCAG 2.1 AA
- [ ] Example notebook demonstrates all features
- [ ] CONTINUITY.md updated with task completion
- [ ] Code reviewed for accessibility best practices

---

## Notes / Log

### Color Palette Research

**Wong 2011 / Okabe & Ito Palette** (most widely used):
- Black: #000000
- Orange: #E69F00
- Sky Blue: #56B4E9
- Bluish Green: #009E73
- Yellow: #F0E442
- Blue: #0072B2
- Vermillion: #D55E00
- Reddish Purple: #CC79A7

Safe for:
- Deuteranopia (most common)
- Protanopia
- Tritanopia
- Grayscale printing

**Alternative: Tol Palette** (Paul Tol's color schemes):
- Bright scheme: 7 colors
- High-contrast scheme: 3 colors
- Muted scheme: 9 colors

**Sequential palettes** (for heatmaps):
- Blue sequential: Safe for all CVD types
- Yellow-Orange-Brown: Viridis-inspired
- Avoid: Rainbow (red-green transitions)

### Screen Reader Best Practices

**Mathematical notation**:
- "3.14 meters" not "3.14m" (screen readers may say "three fourteen m")
- "x squared" not "x²" (superscripts are inconsistent)
- "plus or minus" not "±" (symbol may not be read)

**Table structure**:
- Use `<table>` with `<caption>`, `<thead>`, `<tbody>`
- Add `scope="row"` and `scope="col"` attributes
- Avoid nested tables

**ARIA attributes**:
- `aria-label` for concise description
- `aria-describedby` for detailed explanation
- `role="math"` for mathematical expressions
- `role="img"` with `aria-label` for plots

**Semantic HTML**:
- Use `<strong>` for important values (not `<b>`)
- Use `<em>` for emphasis (not `<i>`)
- Use heading hierarchy (`<h1>`, `<h2>`, etc.)

### WCAG 2.1 Contrast Requirements

**Level AA** (minimum):
- Normal text: 4.5:1
- Large text (18pt+): 3.0:1
- Graphics and UI components: 3.0:1

**Level AAA** (enhanced):
- Normal text: 7.0:1
- Large text: 4.5:1

**Tools for checking**:
- WebAIM Contrast Checker
- Color Oracle (CVD simulator)
- Coblis (Color Blindness Simulator)

### Jupyter Notebook Accessibility

**Current issues**:
- Default output is plain text or matplotlib images
- Images have no alt text
- No ARIA labels on cell outputs
- No semantic structure

**Solutions**:
- Implement `_repr_html_()` with semantic markup
- Add alt text to generated plots
- Use collapsible sections for long output
- Provide "data table" view as fallback

**Best practices**:
- Keep cell outputs concise
- Provide text descriptions before plots
- Use headings to structure notebook
- Include summary statistics in text

---

## References

1. Wong, B. (2011). "Points of view: Color blindness." Nature Methods 8, 441.
2. Okabe, M. & Ito, K. "Color Universal Design." https://jfly.uni-koeln.de/color/
3. W3C. "Web Content Accessibility Guidelines (WCAG) 2.1." https://www.w3.org/TR/WCAG21/
4. Paul Tol. "Colour Schemes." https://personal.sron.nl/~pault/
5. Seaborn colorblind-safe palettes: https://seaborn.pydata.org/tutorial/color_palettes.html
6. matplotlib colorblind considerations: https://matplotlib.org/stable/users/explain/colors/colormaps.html
7. ARIA Authoring Practices Guide: https://www.w3.org/WAI/ARIA/apg/

---

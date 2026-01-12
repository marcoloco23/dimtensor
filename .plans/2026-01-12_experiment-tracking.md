# Plan: Experiment Tracking System (v5.0.0)

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner agent
**Task IDs**: #242-245

---

## Goal

Create a comprehensive experiment tracking system that logs physics experiments with full unit awareness, enabling researchers to track parameters, metrics, and results with their physical units preserved across different unit systems and experimental runs.

---

## Background

### Why is this needed?

Current experiment tracking tools (MLflow, W&B) can log numerical values but don't natively understand physical units. Researchers often:
- Compare experiments across different unit systems (SI vs CGS vs natural units)
- Need to verify dimensional consistency of logged metrics
- Want to track parameters with their physical meaning (not just raw numbers)
- Struggle with unit conversion when comparing historical experiments

dimtensor already has MLflow and W&B integrations (`integrations/mlflow.py`, `integrations/wandb.py`) that log unit metadata as tags. However, these are low-level logging utilities. We need a high-level experiment tracking API that:
1. Organizes runs into experiments
2. Compares runs intelligently across unit systems
3. Queries and filters by physical dimensions
4. Exports/imports full experiment histories with units

### What's the context?

- v4.0.0 added MLflow/W&B integrations with unit tagging
- v4.3.0 added data loaders for real physics data
- v4.5.0 added uncertainty propagation and error budgets
- v5.0.0 aims to provide end-to-end research workflow

This experiment tracking system ties everything together: load data with units, train models, track experiments, compare results.

---

## Approach

### Option A: Wrapper around existing tools (MLflow/W&B)

**Description**: Build `DimExperiment` as a high-level wrapper that uses MLflow or W&B as backend storage.

**Pros**:
- Leverage existing infrastructure (storage, UI, APIs)
- Users can still use native MLflow/W&B tools
- No need to build storage/visualization from scratch
- Integration with existing workflows

**Cons**:
- Limited by backend capabilities
- Need to handle two backends with different APIs
- Unit metadata stored as strings/tags (less queryable)

### Option B: Standalone experiment database

**Description**: Build custom experiment storage using SQLite/HDF5 with native unit support.

**Pros**:
- Full control over data model
- Native unit queries (e.g., "find all runs where force > 10 N")
- Cleaner API without backend abstraction

**Cons**:
- Must build storage, versioning, UI from scratch
- No integration with existing tools
- More implementation work
- Harder to adopt (users already have MLflow/W&B)

### Option C: Hybrid approach with DimExperiment facade

**Description**: Build `DimExperiment` API that works standalone AND can sync with MLflow/W&B.

**Pros**:
- Users can work purely in dimtensor
- Optional sync to MLflow/W&B for visualization
- Local JSON/HDF5 storage for lightweight use
- Best of both worlds

**Cons**:
- More complex implementation
- Need to maintain sync logic
- Potential for inconsistencies between local/remote

### Decision: Option A - Wrapper around existing tools

**Rationale**:
- v4.0.0 already built MLflow/W&B integrations
- Users already have these tools set up
- Focus on providing great unit-aware API, not reinventing infrastructure
- Can add standalone storage later if needed (v5.1.0+)
- Backend abstraction is manageable (similar to `io/` module pattern)

---

## Implementation Steps

### Phase 1: Core DimExperiment API (Task #243)

1. [ ] Create `experiments/` module folder
2. [ ] Create `experiments/experiment.py` with `DimExperiment` class
3. [ ] Implement `DimExperiment.__init__(name, backend='mlflow')`
4. [ ] Implement `experiment.start_run(run_name, tags)` - context manager
5. [ ] Implement `experiment.log_param(name, value_with_unit)`
6. [ ] Implement `experiment.log_metric(name, value_with_unit, step)`
7. [ ] Implement `experiment.log_array(name, dimarray, step)`
8. [ ] Implement `experiment.end_run()` and `experiment.get_run(run_id)`
9. [ ] Create backend abstraction layer: `ExperimentBackend` protocol
10. [ ] Create `MLflowBackend` implementing the protocol
11. [ ] Create `WandbBackend` implementing the protocol

### Phase 2: Run Comparison (Task #244)

12. [ ] Implement `experiment.list_runs(filter_string)` with unit-aware filtering
13. [ ] Implement `experiment.compare_runs(run_ids, metric_names)`
14. [ ] Handle automatic unit conversion in comparison
15. [ ] Create `RunComparison` dataclass with comparison results
16. [ ] Implement `comparison.to_dataframe()` for analysis
17. [ ] Implement `comparison.plot(metric_name)` for visualization
18. [ ] Add support for comparing across unit systems (SI vs CGS)
19. [ ] Add relative/absolute difference calculations

### Phase 3: Visualization (Task #245)

20. [ ] Create `experiments/visualization.py` module
21. [ ] Implement `plot_experiment_history(experiment, metric_name)`
22. [ ] Implement `plot_run_comparison(comparison, metrics)`
23. [ ] Implement `plot_parameter_importance(experiment, target_metric)`
24. [ ] Add unit labels to all plots automatically
25. [ ] Support unit conversion in plot display
26. [ ] Create interactive dashboard using existing `web/` module

### Phase 4: Export/Import (Task #242 continuation)

27. [ ] Implement `experiment.export(path, format='json')`
28. [ ] Implement `experiment.import_from(path)`
29. [ ] Support JSON format (using existing `io/json.py` patterns)
30. [ ] Support HDF5 format (using existing `io/hdf5.py` patterns)
31. [ ] Preserve all unit metadata in export
32. [ ] Handle version compatibility in import

### Phase 5: Testing

33. [ ] Unit tests for `DimExperiment` core API
34. [ ] Unit tests for backend abstraction layer
35. [ ] Integration tests with MLflow (skip if not installed)
36. [ ] Integration tests with W&B (skip if not installed)
37. [ ] Tests for run comparison logic
38. [ ] Tests for unit conversion in comparisons
39. [ ] Tests for export/import round-trip
40. [ ] Tests for visualization functions

---

## Files to Modify

| File | Change |
|------|--------|
| src/dimtensor/experiments/__init__.py | New module, export DimExperiment |
| src/dimtensor/experiments/experiment.py | Main DimExperiment class (400-500 lines) |
| src/dimtensor/experiments/backend.py | Backend protocol and implementations (300 lines) |
| src/dimtensor/experiments/comparison.py | RunComparison class (200 lines) |
| src/dimtensor/experiments/visualization.py | Plotting functions (250 lines) |
| src/dimtensor/__init__.py | Add experiments module to exports |
| tests/test_experiments.py | Core API tests (150 lines) |
| tests/test_experiment_comparison.py | Comparison tests (100 lines) |
| tests/test_experiment_export.py | Export/import tests (80 lines) |

Total estimated: ~1,500 lines of new code

---

## Testing Strategy

### Unit Tests

- [ ] Test DimExperiment initialization with different backends
- [ ] Test logging params with various unit types
- [ ] Test logging metrics with scalars and arrays
- [ ] Test run lifecycle (start, log, end)
- [ ] Test unit conversion in metric logging
- [ ] Test filtering runs by dimension

### Integration Tests

- [ ] Test MLflow backend with real MLflow tracking server
- [ ] Test W&B backend with mock wandb runs
- [ ] Test run comparison across different unit systems
- [ ] Test export to JSON and reimport
- [ ] Test visualization generates valid plots

### Manual Verification

- [ ] Create sample experiment with multiple runs
- [ ] Compare runs in different unit systems (SI vs CGS)
- [ ] Export experiment and import in fresh environment
- [ ] Verify plots display correct unit labels
- [ ] Verify MLflow UI shows unit tags correctly

---

## API Design Examples

### Basic Usage

```python
from dimtensor import DimArray, units
from dimtensor.experiments import DimExperiment

# Create experiment
exp = DimExperiment("heat-equation-pinn", backend="mlflow")

# Run 1: SI units
with exp.start_run("run-si"):
    exp.log_param("learning_rate", DimArray(0.001, 1/units.s))
    exp.log_param("domain_length", DimArray(1.0, units.m))

    for step in range(100):
        loss = compute_loss()  # returns DimArray with units.J
        exp.log_metric("loss", loss, step=step)

# Run 2: CGS units (for comparison)
with exp.start_run("run-cgs"):
    exp.log_param("learning_rate", DimArray(0.001, 1/units.s))
    exp.log_param("domain_length", DimArray(100.0, units.cm))

    for step in range(100):
        loss = compute_loss()  # returns DimArray with units.erg
        exp.log_metric("loss", loss, step=step)

# Compare runs (automatic unit conversion)
comparison = exp.compare_runs(["run-si", "run-cgs"], ["loss"])
print(comparison.to_dataframe())
comparison.plot("loss")  # Shows both in same units
```

### Advanced Usage

```python
# Filter runs by dimension
force_experiments = exp.list_runs(
    dimension_filter={"force": True}  # Only runs logging force quantities
)

# Export for sharing
exp.export("experiment_data.json")

# Import in another environment
exp2 = DimExperiment.load("experiment_data.json")
```

---

## Risks / Edge Cases

### Risk 1: Backend API changes

**Mitigation**:
- Use stable MLflow/W&B APIs only
- Version pin dependencies
- Backend abstraction layer isolates changes

### Risk 2: Unit conversion errors in comparison

**Mitigation**:
- Always check dimension compatibility before conversion
- Raise clear errors for incompatible dimensions
- Log original units in all comparisons

### Risk 3: Large arrays in experiment logs

**Mitigation**:
- Log array statistics (mean, std, min, max) by default
- Provide option to log full arrays to artifacts
- Use HDF5 for large array storage

### Edge Case: Mixed dimensionless and dimensional metrics

**Handling**:
- Treat dimensionless as compatible with any dimension (scaling factor only)
- Warn user when comparing dimensionless vs dimensional
- Document behavior clearly

### Edge Case: Same metric name, different units across runs

**Handling**:
- Allow this (e.g., "energy" in J vs erg)
- Auto-convert to common unit in comparison
- Show warning in comparison results

### Edge Case: Backend not available

**Handling**:
- Clear error message: "MLflow not installed. Install with: pip install mlflow"
- Document backend requirements
- Future: Add standalone JSON backend (v5.1.0)

---

## Definition of Done

- [x] Plan approved and reviewed
- [ ] All implementation steps complete
- [ ] All tests pass (unit + integration)
- [ ] Can create experiment, log runs with units
- [ ] Can compare runs across unit systems
- [ ] Can export and import experiments
- [ ] Visualization works with unit labels
- [ ] Documentation added to docs/guide/experiments.md
- [ ] Examples added to examples/ folder
- [ ] CONTINUITY.md updated with task completion

---

## Notes / Log

**2026-01-12 - Planning complete**
- Researched existing MLflow/W&B integrations
- Decided on wrapper approach (Option A)
- Designed API following dimtensor patterns
- Estimated 1,500 lines of code across 9 files
- Ready for implementation

---

## Future Enhancements (Post v5.0.0)

- Standalone JSON/SQLite backend (v5.1.0)
- Hyperparameter optimization with unit constraints
- Automatic experiment versioning
- Experiment templates for common physics problems
- Integration with paper reproduction framework
- Multi-user collaboration features
- Real-time experiment monitoring dashboard

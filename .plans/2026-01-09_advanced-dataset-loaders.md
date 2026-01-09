# Plan: Advanced Dataset Loaders

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Add real physics dataset loaders with automatic downloading, caching, and dimensional metadata for dimtensor, enabling researchers to quickly load experimental/observational data for physics-informed machine learning.

---

## Background

Currently (v3.0.0), dimtensor has a dataset registry system with metadata for 10 synthetic/generated physics datasets (pendulum, projectile, Navier-Stokes, etc.), but no actual data loaders are implemented. The registry system exists but `load_dataset()` raises errors because no loader functions are registered.

For v3.3.0, we want to add "real physics data downloads" - actual experimental and observational datasets from trusted sources like NASA, NIST, ESA, NOAA, and materials science databases. This will make dimtensor useful for real physics ML workflows.

### Research Summary

Identified major public physics data sources:

**Space/Astronomy:**
- NASA Solar Dynamics Observatory (SDO) - ML-ready solar physics data
- NASA Exoplanet Archive - 5000+ confirmed exoplanets with time series
- ESA Gaia DR3/DR4 - Galactic mapping, billions of stars (DR4 in Dec 2026)
- SDSS - Sloan Digital Sky Survey

**Climate/Atmosphere:**
- NASA NCCS/GISS - Climate simulation data
- NOAA Climate Data - Hundreds of datasets on cloud platforms
- PRISM Climate Data - 1895-present temperature/precipitation

**Materials Science:**
- Materials Project - 600k+ users, API access
- AFLOW - 3.5M+ materials via AFLUX API
- NOMAD - 100M+ atomistic calculations
- OPTIMADE - Unified API for multiple databases

**Physics Reference:**
- NIST CODATA 2022 - Fundamental physical constants
- UCI ML Physics Portal - Various ML-ready physics datasets

---

## Approach

### Option A: Simple URL-Based Loaders
- Description: Each dataset has a URL, `load_dataset()` downloads and caches locally
- Pros: Simple, works for static files, easy to implement
- Cons: Doesn't handle APIs, no authentication support, limited flexibility

### Option B: Pluggable Loader Architecture with API Support
- Description: Create a loader framework supporting both direct downloads and API calls
- Components:
  - Base `DatasetLoader` class with caching
  - Specific loaders: `CSVLoader`, `APILoader`, `HDF5Loader`, `FITSLoader`
  - Cache management system
  - Unit conversion pipeline
- Pros: Extensible, handles diverse sources, professional caching
- Cons: More complex, requires more infrastructure

### Option C: External Dependencies (astropy, pandas, h5py)
- Description: Use existing science libraries for file formats
- Pros: Leverage mature ecosystems, better format support
- Cons: Heavier dependencies, may conflict with minimal install

### Decision: Hybrid Approach (B + selective C)

Use Option B's architecture with Option C's libraries as optional dependencies:
- Core: Simple URL-based downloading with caching
- Optional: `astropy` for FITS (astronomy), `h5py` for HDF5, `netCDF4` for climate
- API wrappers for major sources (NASA, NIST, Materials Project)
- Convert everything to DimArray/DimTensor with proper units

**Why:**
- Balances simplicity and power
- Keeps core lightweight with optional extras
- Allows incremental implementation (start simple, add complexity)

---

## Implementation Steps

### Phase 1: Infrastructure (Priority)
1. [ ] Create `src/dimtensor/datasets/loaders/` module
2. [ ] Implement `BaseLoader` class with caching logic
3. [ ] Create cache directory management (`~/.dimtensor/cache/`)
4. [ ] Add download utilities (progress bars, retry logic, checksums)
5. [ ] Implement `CSVLoader` for simple datasets

### Phase 2: First Real Datasets (Start Here)
6. [ ] **NIST CODATA loader** - Fundamental constants (simple, no API)
7. [ ] **PRISM Climate loader** - CSV time series data
8. [ ] **NASA Exoplanet loader** - CSV bulk download endpoint
9. [ ] Register these 3 datasets with proper dimensional metadata
10. [ ] Write tests for download, caching, and unit conversion

### Phase 3: API-Based Loaders
11. [ ] Create `APILoader` base class (handle auth, rate limits, retries)
12. [ ] **Materials Project API loader** - Requires API key
13. [ ] **NOAA Climate API loader** - Public access
14. [ ] Add API key management system (`~/.dimtensor/config.json` or env vars)

### Phase 4: Advanced Formats (Optional Extensions)
15. [ ] Add `FITSLoader` for astronomy data (requires `astropy`)
16. [ ] Add `HDF5Loader` for large datasets (requires `h5py`)
17. [ ] **Gaia DR3 loader** - FITS format
18. [ ] **NASA SDO loader** - HDF5 format

### Phase 5: Documentation & Examples
19. [ ] Add docstrings to all loaders
20. [ ] Create examples/datasets.ipynb notebook
21. [ ] Update README with dataset examples
22. [ ] Add dataset usage to documentation site

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/datasets/__init__.py` | Export new loader functions |
| `src/dimtensor/datasets/registry.py` | Add loader registration for real datasets |
| `src/dimtensor/datasets/loaders/__init__.py` | New module: base loader classes |
| `src/dimtensor/datasets/loaders/base.py` | New: `BaseLoader`, `CSVLoader`, caching utilities |
| `src/dimtensor/datasets/loaders/api.py` | New: `APILoader`, authentication helpers |
| `src/dimtensor/datasets/loaders/nist.py` | New: NIST CODATA constants loader |
| `src/dimtensor/datasets/loaders/climate.py` | New: PRISM, NOAA climate data loaders |
| `src/dimtensor/datasets/loaders/astronomy.py` | New: NASA Exoplanet, Gaia loaders |
| `src/dimtensor/datasets/loaders/materials.py` | New: Materials Project, AFLOW loaders |
| `src/dimtensor/config.py` | Add cache directory configuration |
| `tests/test_dataset_loaders.py` | New: Tests for all loaders |
| `tests/test_cache.py` | New: Cache management tests |
| `setup.py` | Add optional dependencies group: `[datasets]` |
| `README.md` | Add dataset loading examples |
| `examples/datasets.ipynb` | New: Dataset loading tutorial |

---

## API Design

### User-Facing API

```python
from dimtensor.datasets import load_dataset, list_datasets

# List available datasets (includes both synthetic and real)
datasets = list_datasets(domain="astronomy")
for ds in datasets:
    print(f"{ds.name}: {ds.description}")

# Load a real dataset (downloads and caches automatically)
data = load_dataset(
    "nist_codata_2022",
    cache=True,  # Default: True
    force_download=False,  # Re-download even if cached
)

# Load with parameters
exoplanets = load_dataset(
    "nasa_exoplanets",
    columns=["mass", "radius", "period"],  # Select specific columns
    confirmed_only=True,  # Dataset-specific parameter
)

# Load materials data (requires API key)
materials = load_dataset(
    "materials_project",
    api_key="YOUR_KEY",  # Or set MATERIALS_PROJECT_API_KEY env var
    formula="Fe2O3",
    properties=["energy", "band_gap"],
)
```

### Loader Implementation Pattern

```python
# In src/dimtensor/datasets/loaders/nist.py
from .base import BaseLoader
from dimtensor import DimArray
from dimtensor.units import meter, second, kg

class NISTCODATALoader(BaseLoader):
    """Loader for NIST CODATA fundamental constants."""

    URL = "https://physics.nist.gov/cuu/Constants/Table/allascii.txt"

    def load(self, **kwargs):
        """Download and parse CODATA constants."""
        # Download with caching
        filepath = self.download(self.URL, cache_key="codata_2022")

        # Parse the file
        constants = self._parse_codata_file(filepath)

        # Convert to dimensional format
        return {
            name: DimArray(value, unit=unit)
            for name, (value, unit) in constants.items()
        }

    def _parse_codata_file(self, filepath):
        # Parsing logic...
        pass

# Register the loader
@register_dataset("nist_codata_2022", domain="constants")
def load_nist_codata(**kwargs):
    loader = NISTCODATALoader()
    return loader.load(**kwargs)
```

### Caching Strategy

- Cache location: `~/.dimtensor/cache/` (configurable)
- Cache key: `{dataset_name}_{version}_{hash}`
- Cache file formats: `.csv`, `.parquet`, `.h5`, `.fits`
- Metadata: `.json` sidecar with download date, source URL, checksum
- Cache expiry: Configurable (default: never expire for versioned data)
- Cache management: `clear_cache()`, `cache_info()` utilities

---

## Testing Strategy

### Unit Tests
- [ ] Test cache directory creation and permissions
- [ ] Test download with mocked HTTP requests
- [ ] Test cache hit/miss logic
- [ ] Test checksum verification
- [ ] Test retry logic on network errors
- [ ] Test each loader's parsing logic with fixtures

### Integration Tests (Requires Network)
- [ ] Test actual download of small datasets (marked with `@pytest.mark.network`)
- [ ] Test NIST CODATA loader end-to-end
- [ ] Test PRISM climate data loader
- [ ] Test NASA Exoplanet loader

### Mock API Tests
- [ ] Test API authentication flows
- [ ] Test rate limiting handling
- [ ] Test API error handling (404, 500, etc.)
- [ ] Test Materials Project API wrapper

### Dimensional Correctness Tests
- [ ] Verify loaded data has correct units
- [ ] Test unit conversion on load
- [ ] Verify dimensional metadata matches registry
- [ ] Test operations on loaded DimArrays

---

## Risks / Edge Cases

### Risk 1: Data Source URLs Change
- **Mitigation**: Version datasets, maintain URL mappings, add fallback mirrors

### Risk 2: Large Downloads (GBs)
- **Mitigation**: Progress bars, resume support, streaming for large files, dataset size warnings

### Risk 3: API Rate Limits
- **Mitigation**: Exponential backoff, rate limit detection, caching, batch requests

### Risk 4: Authentication/API Keys
- **Mitigation**: Clear documentation, environment variable support, keychain integration (future)

### Risk 5: Inconsistent Data Formats
- **Mitigation**: Robust parsing with error handling, format version detection, schema validation

### Risk 6: Optional Dependencies
- **Mitigation**: Clear error messages, installation instructions, graceful degradation

### Edge Case: Corrupted Cache
- **Handling**: Checksum verification, automatic re-download on corruption

### Edge Case: Concurrent Downloads
- **Handling**: File locking, atomic writes to cache

### Edge Case: Disk Space
- **Handling**: Check available space before download, configurable cache size limit

### Edge Case: Network Offline
- **Handling**: Graceful fallback to cached data, clear error messages

---

## Definition of Done

- [ ] All Phase 1 & 2 implementation steps complete
- [ ] At least 3 real datasets loadable (NIST, PRISM, NASA Exoplanets)
- [ ] Tests pass (both unit and integration)
- [ ] Cache system working correctly
- [ ] Documentation updated with examples
- [ ] Example notebook created and tested
- [ ] CONTINUITY.md updated
- [ ] Optional: Phase 3 (API loaders) for Materials Project

---

## Notes / Log

### Research Sources

**Machine Learning & Physics Datasets:**
- [UCI Machine Learning Physics Portal](https://mlphysics.ics.uci.edu/)
- [NASA SDO ML Dataset Paper](https://iopscience.iop.org/article/10.3847/1538-4365/ab1005)
- [Physics Datasets for ML Overview](https://reason.town/physics-datasets-for-machine-learning/)

**Climate & Atmospheric Data:**
- [NASA Center for Climate Simulation](https://www.nccs.nasa.gov/services/data-collections)
- [NASA GISS Data Portal](https://data.giss.nasa.gov/)
- [NOAA Open Data Datasets](https://www.noaa.gov/nodd/datasets)
- [PRISM Climate Group](https://prism.oregonstate.edu/)
- [Climate Data Guide (UCAR)](https://climatedataguide.ucar.edu/climate-data)

**Physical Constants:**
- [NIST CODATA Fundamental Constants](https://physics.nist.gov/cuu/Constants/)
- [CODATA 2022 Values](https://www.nist.gov/publications/codata-internationally-recommended-2022-values-fundamental-physical-constants)

**Astronomy Data:**
- [ESA Gaia Archive](https://gea.esac.esa.int/archive/)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [Gaia Sky Datasets](https://gaiasky.space/resources/datasets/)

**Materials Science:**
- [Materials Project](https://www.nature.com/articles/s41563-025-02272-0)
- [AFLOW Materials Database](https://www.aflowlib.org/)
- [NOMAD AI Toolkit](https://www.nature.com/articles/s41524-022-00935-z)
- [OPTIMADE API](https://www.nature.com/articles/s41597-021-00974-z)

### Key Design Decisions

**2026-01-09 15:00** - Decision: Hybrid approach with optional dependencies
- Keeps core lightweight (no heavy deps by default)
- Users can install extras: `pip install dimtensor[datasets]`
- Astronomy users: `pip install dimtensor[astronomy]` (adds astropy, astroquery)
- Materials users: `pip install dimtensor[materials]` (adds mp-api, pymatgen)

**2026-01-09 15:15** - Decision: Start with simple loaders first
- Phase 1-2 focus: NIST CODATA, PRISM Climate, NASA Exoplanets
- These are CSV/text-based, no complex formats
- Get the infrastructure right before tackling APIs and binary formats

**2026-01-09 15:20** - Decision: Cache in home directory
- Standard practice: `~/.dimtensor/cache/`
- Respects XDG on Linux, AppData on Windows
- Configurable via environment variable: `DIMTENSOR_CACHE_DIR`

---

## Priority Implementation Order

1. **CRITICAL (v3.3.0 MVP)**: Phase 1 + Phase 2 (steps 1-10)
   - Basic infrastructure + 3 real datasets
   - Estimated: 2-3 days of work

2. **HIGH PRIORITY (v3.3.0 complete)**: Phase 3 (steps 11-14)
   - API loaders for Materials Project, NOAA
   - Estimated: 1-2 days

3. **MEDIUM PRIORITY (v3.4.0)**: Phase 4 (steps 15-18)
   - Advanced formats (FITS, HDF5)
   - Astronomy datasets
   - Estimated: 2-3 days

4. **LOW PRIORITY (continuous)**: Phase 5 + more datasets
   - Documentation, examples
   - Additional data sources
   - Community contributions

---

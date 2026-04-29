# Testing & Quality Assurance

dimtensor's correctness story is built on multiple layers of testing,
introduced incrementally across releases. This guide describes each layer
and how to run it locally.

## Test layers at a glance

| Layer                | File pattern                | Purpose                                        |
| -------------------- | --------------------------- | ---------------------------------------------- |
| Unit tests           | `tests/test_*.py`           | Verify each function/class against examples    |
| Property-based tests | `tests/test_property_based.py` | Verify *invariants* hold for *all* inputs      |
| Fuzz tests           | `tests/test_fuzz.py`        | Catch crashes on malformed/random inputs       |
| Chaos tests          | `tests/test_chaos.py`       | Verify graceful degradation (missing deps, I/O failures) |
| Load tests           | `tests/test_load.py`        | Catch catastrophic performance regressions     |
| Mutation tests       | `mutmut run`                | Verify the test suite *catches* bugs           |
| Security audit       | `bandit`, `pip-audit`       | Static analysis & dependency vulnerabilities   |

## Running the test suite

```bash
# Install with dev + test extras
pip install -e ".[dev,test]"

# Run everything (excluding slow tests)
pytest -m "not slow"

# Just the property-based tests
pytest tests/test_property_based.py

# Just the fuzz tests
pytest tests/test_fuzz.py

# Just the chaos tests
pytest tests/test_chaos.py

# Load tests (CPU-bound, may take longer)
pytest tests/test_load.py

# Include slow / huge-array load tests
pytest tests/test_load.py -m slow
```

## Property-based testing (Hypothesis)

Property-based tests express *invariants* and let
[Hypothesis](https://hypothesis.readthedocs.io/) generate hundreds of
random inputs to find counterexamples. dimtensor's invariants include:

- Dimension forms an Abelian group: `a * b == b * a`, `a * a^-1 == 1`, etc.
- Unit conversion is symmetric: `a.conversion_factor(b) * b.conversion_factor(a) == 1`
- DimArray addition: `a + a == 2 * a` always
- JSON serialization is a round-trip: `from_json(to_json(arr)) == arr`

When Hypothesis finds a failure it automatically *shrinks* the example
to the minimal failing input.

## Fuzz testing

Fuzz tests stress dimtensor with adversarial inputs:

- Garbage JSON strings (`"@@@}}"`)
- Truncated JSON files
- Random dictionaries with missing keys
- Arrays containing `NaN`, `inf`, subnormals
- Random sequences of operations

The test suite asserts that every failure raises *one of the documented
exception types* (`DimensionError`, `TypeError`, `ValueError`, etc.) -
never a Python `AttributeError` or unexpected stacktrace.

## Mutation testing

Mutation testing modifies the source code in small ways (e.g.,
`x + y` -> `x - y`) and re-runs the tests. Each unkilled mutant is a
gap in test coverage that exact-line coverage doesn't reveal.

```bash
pip install mutmut

# Run mutation testing against the core modules
mutmut run --paths-to-mutate=src/dimtensor/core

# Inspect surviving mutants
mutmut results
mutmut show 3
```

Configuration lives in `.mutmut.toml`.

## Chaos testing

Chaos tests confirm dimtensor degrades gracefully when its environment
misbehaves: missing optional dependencies, corrupt files, unwritable
directories, NaN/inf inputs. They are fast and run as part of the
regular suite.

## Load testing

Load tests assert that key operations finish under a generous time
budget for million-element arrays. They are *not* benchmarks - the
budgets are sized to catch O(n²) regressions, not to enforce particular
microbenchmarks. For real benchmarking see `benchmarks/`.

## Coverage dashboard

Each push to `main` triggers
[`dimtensor-coverage.yml`](https://github.com/marcoloco23/dimtensor/actions/workflows/dimtensor-coverage.yml)
which:

1. Runs the full test suite with coverage.
2. Uploads to [Codecov](https://codecov.io).
3. Publishes an HTML report as a build artifact.
4. Generates a coverage badge JSON (consumable by shields.io endpoint).

To generate the report locally:

```bash
pytest --cov=dimtensor --cov-report=html
open htmlcov/index.html  # or xdg-open on Linux
```

## Security audit

The
[`dimtensor-security.yml`](https://github.com/marcoloco23/dimtensor/actions/workflows/dimtensor-security.yml)
workflow runs four scanners on every PR and weekly on `main`:

| Tool       | What it scans                                      |
| ---------- | -------------------------------------------------- |
| Bandit     | Python source for common security smells (SAST)    |
| pip-audit  | Installed dependencies against the PyPA advisory DB |
| Safety     | Dependencies against the Safety DB                 |
| Trivy      | Filesystem for known CVEs                          |
| CodeQL     | Deep semantic analysis (security & quality queries) |

Results land in the GitHub Security tab and as PR annotations.

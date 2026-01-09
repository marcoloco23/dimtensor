# Ecosystem: Hub, Datasets & Equations

Registry systems for models, datasets, and physics equations with dimensional metadata.

## Overview

The dimtensor ecosystem provides three interconnected registry systems:

- **Hub**: Discover and share pre-trained physics-aware models
- **Datasets**: Access physics datasets with dimensional metadata
- **Equations**: Query a database of physics equations

```python
from dimtensor.hub import list_models, load_model
from dimtensor.datasets import list_datasets, load_dataset
from dimtensor.equations import get_equation, search_equations

# Browse models
models = list_models(domain="fluid_dynamics")

# Load a dataset
data = load_dataset("navier-stokes-2d")

# Query equations
eq = get_equation("Newton's Second Law")
print(eq.formula)  # "F = m * a"
```

## Hub: Model Registry

### ModelInfo

Metadata about a registered physics model.

```python
from dimtensor.hub import ModelInfo, register_model
from dimtensor import Dimension

info = ModelInfo(
    name="velocity-predictor-v1",
    version="1.0.0",
    description="Predicts velocity from position time series",
    input_dims={"position": Dimension(L=1)},
    output_dims={"velocity": Dimension(L=1, T=-1)},
    domain="mechanics",
    architecture="LSTM",
    tags=["time-series", "kinematics"]
)

register_model(info, factory_fn=load_my_model)
```

::: dimtensor.hub.ModelInfo
    options:
      members:
        - name
        - version
        - description
        - input_dims
        - output_dims
        - domain
        - characteristic_scales
        - tags
        - source
        - architecture
        - author
        - license
        - to_dict
        - from_dict

### register_model

Register a model in the hub.

```python
from dimtensor.hub import register_model

def load_my_model(**kwargs):
    # Load and return your model
    return model

register_model(model_info, factory_fn=load_my_model)
```

::: dimtensor.hub.register_model

### load_model

Load a registered model by name.

```python
from dimtensor.hub import load_model

model = load_model("velocity-predictor-v1")
```

::: dimtensor.hub.load_model

### list_models

List all registered models, optionally filtered.

```python
from dimtensor.hub import list_models

# All models
all_models = list_models()

# Filter by domain
fluid_models = list_models(domain="fluid_dynamics")

# Filter by tags
lstm_models = list_models(tags=["LSTM", "time-series"])
```

::: dimtensor.hub.list_models

### get_model_info

Get metadata for a model without loading it.

::: dimtensor.hub.get_model_info

## Datasets: Physics Data Registry

### DatasetInfo

Metadata about a physics dataset.

```python
from dimtensor.datasets import DatasetInfo, register_dataset
from dimtensor import Dimension

info = DatasetInfo(
    name="pendulum-trajectories",
    description="Simple pendulum position and velocity data",
    domain="mechanics",
    features={
        "angle": Dimension(),  # radians (dimensionless)
        "angular_velocity": Dimension(T=-1)
    },
    targets={
        "angular_acceleration": Dimension(T=-2)
    },
    size=10000,
    license="CC-BY-4.0",
    tags=["dynamics", "oscillation"]
)
```

::: dimtensor.datasets.DatasetInfo
    options:
      members:
        - name
        - description
        - domain
        - features
        - targets
        - size
        - source
        - license
        - citation
        - tags

### register_dataset

Register a dataset with a loader function.

```python
from dimtensor.datasets import register_dataset

def load_pendulum_data(**kwargs):
    # Load and return dataset
    return X, y

register_dataset(dataset_info, loader_fn=load_pendulum_data)
```

::: dimtensor.datasets.register_dataset

### load_dataset

Load a registered dataset.

```python
from dimtensor.datasets import load_dataset

X, y = load_dataset("pendulum-trajectories", split="train")
```

::: dimtensor.datasets.load_dataset

### list_datasets

List all registered datasets.

```python
from dimtensor.datasets import list_datasets

# All datasets
datasets = list_datasets()

# Filter by domain
mechanics_data = list_datasets(domain="mechanics")

# Filter by tags
oscillation_data = list_datasets(tags=["oscillation"])
```

::: dimtensor.datasets.list_datasets

### Built-in Loaders

The `dimtensor.datasets.loaders` submodule provides loaders for common physics datasets:

#### Astronomy Datasets

```python
from dimtensor.datasets.loaders import load_exoplanet_data

data = load_exoplanet_data()
# Returns DimArrays with proper astronomical units
```

::: dimtensor.datasets.loaders.load_exoplanet_data

#### Climate Datasets

```python
from dimtensor.datasets.loaders import load_climate_data

data = load_climate_data(variable="temperature", years=(2000, 2020))
```

::: dimtensor.datasets.loaders.load_climate_data

#### NIST Reference Data

```python
from dimtensor.datasets.loaders import load_nist_data

data = load_nist_data(material="water", property="density")
```

::: dimtensor.datasets.loaders.load_nist_data

## Equations: Physics Equation Database

### Equation

A physics equation with dimensional metadata.

```python
from dimtensor.equations import Equation
from dimtensor import Dimension

eq = Equation(
    name="Newton's Second Law",
    formula="F = m * a",
    variables={
        "F": Dimension(M=1, L=1, T=-2),  # force
        "m": Dimension(M=1),              # mass
        "a": Dimension(L=1, T=-2)         # acceleration
    },
    domain="mechanics",
    tags=["fundamental", "dynamics"],
    description="Force equals mass times acceleration",
    latex=r"F = ma"
)
```

::: dimtensor.equations.Equation
    options:
      members:
        - name
        - formula
        - variables
        - domain
        - tags
        - description
        - assumptions
        - latex
        - related
        - to_dict

### register_equation

Add an equation to the database.

```python
from dimtensor.equations import register_equation

register_equation(eq)
```

::: dimtensor.equations.register_equation

### get_equation

Retrieve an equation by name.

```python
from dimtensor.equations import get_equation

newton = get_equation("Newton's Second Law")
print(newton.formula)  # "F = m * a"
print(newton.latex)    # "F = ma"
```

::: dimtensor.equations.get_equation

### get_equations

Get equations, optionally filtered by domain and tags.

```python
from dimtensor.equations import get_equations

# All mechanics equations
mech_eqs = get_equations(domain="mechanics")

# Fundamental laws
fundamental = get_equations(tags=["fundamental"])

# Thermodynamics conservation laws
thermo_conservation = get_equations(
    domain="thermodynamics",
    tags=["conservation"]
)
```

::: dimtensor.equations.get_equations

### search_equations

Search equations by name, formula, or description.

```python
from dimtensor.equations import search_equations

# Search for "energy"
results = search_equations("energy")
# Returns equations with "energy" in name or description
```

::: dimtensor.equations.search_equations

## Examples

### Publishing a Model to the Hub

```python
import torch
from dimtensor.hub import ModelInfo, register_model
from dimtensor import Dimension

# 1. Define model metadata
info = ModelInfo(
    name="fluid-velocity-cnn",
    version="1.0.0",
    description="CNN for predicting fluid velocity fields from pressure",
    input_dims={"pressure": Dimension(M=1, L=-1, T=-2)},
    output_dims={"velocity": Dimension(L=1, T=-1)},
    domain="fluid_dynamics",
    characteristic_scales={
        "pressure": 101325.0,  # Pa (1 atm)
        "velocity": 10.0        # m/s
    },
    architecture="ResNet-18",
    author="Your Name",
    license="MIT",
    tags=["CNN", "fluid-dynamics", "CFD"]
)

# 2. Define loader function
def load_fluid_model(**kwargs):
    model = torch.load("path/to/model.pt")
    return model

# 3. Register
register_model(info, factory_fn=load_fluid_model)

# 4. Users can now load your model
model = load_model("fluid-velocity-cnn")
```

### Creating a Custom Dataset

```python
from dimtensor.datasets import DatasetInfo, register_dataset
from dimtensor import DimArray, units
import numpy as np

# 1. Define dataset metadata
info = DatasetInfo(
    name="projectile-motion",
    description="Projectile trajectories with air resistance",
    domain="mechanics",
    features={
        "initial_velocity": Dimension(L=1, T=-1),
        "launch_angle": Dimension(),  # radians
        "drag_coefficient": Dimension()
    },
    targets={
        "range": Dimension(L=1),
        "max_height": Dimension(L=1),
        "flight_time": Dimension(T=1)
    },
    size=50000,
    license="CC0",
    tags=["kinematics", "simulation", "projectile"]
)

# 2. Define loader
def load_projectile_data(split="train", **kwargs):
    # Load your data
    data = np.load(f"projectile_{split}.npz")

    # Wrap in DimArrays
    X = {
        "initial_velocity": DimArray(data['v0'], units.m / units.s),
        "launch_angle": DimArray(data['theta'], units.rad),
        "drag_coefficient": DimArray(data['Cd'], units.dimensionless)
    }

    y = {
        "range": DimArray(data['range'], units.m),
        "max_height": DimArray(data['h_max'], units.m),
        "flight_time": DimArray(data['t_flight'], units.s)
    }

    return X, y

# 3. Register
register_dataset(info, loader_fn=load_projectile_data)

# 4. Users can now load your dataset
X_train, y_train = load_dataset("projectile-motion", split="train")
```

### Using the Equation Database

```python
from dimtensor.equations import (
    Equation, register_equation, get_equation,
    get_equations, search_equations
)
from dimtensor import Dimension

# Register custom equations
kinetic_energy = Equation(
    name="Kinetic Energy",
    formula="KE = 0.5 * m * v**2",
    variables={
        "KE": Dimension(M=1, L=2, T=-2),
        "m": Dimension(M=1),
        "v": Dimension(L=1, T=-1)
    },
    domain="mechanics",
    tags=["energy", "motion"],
    description="Kinetic energy of a moving object",
    latex=r"KE = \frac{1}{2}mv^2",
    related=["Work-Energy Theorem", "Conservation of Energy"]
)

register_equation(kinetic_energy)

# Query equations
eq = get_equation("Kinetic Energy")
print(f"Formula: {eq.formula}")
print(f"LaTeX: {eq.latex}")

# Search by topic
energy_eqs = search_equations("energy")
for eq in energy_eqs:
    print(f"- {eq.name}: {eq.formula}")

# Filter by domain
mechanics_eqs = get_equations(domain="mechanics")
print(f"Found {len(mechanics_eqs)} mechanics equations")
```

### Integration Example: Model + Dataset + Equation

```python
from dimtensor.hub import load_model
from dimtensor.datasets import load_dataset
from dimtensor.equations import get_equation
from dimtensor import units

# 1. Load equation for validation
eq = get_equation("Newton's Second Law")
print(f"Training to learn: {eq.formula}")

# 2. Load training data
X_train, y_train = load_dataset("particle-dynamics", split="train")

# 3. Load pre-trained model
model = load_model("force-predictor-v2")

# 4. Make prediction
mass = X_train["mass"]
acceleration = X_train["acceleration"]
force_pred = model(mass, acceleration)

# 5. Validate dimensions match equation
assert force_pred.dimension == eq.variables["F"]
print(f"✓ Prediction has correct dimension: {force_pred.dimension}")
```

### Discovering Resources

```python
from dimtensor.hub import list_models
from dimtensor.datasets import list_datasets
from dimtensor.equations import get_equations

# Browse available models
print("Available Models:")
for model in list_models(domain="thermodynamics"):
    print(f"  {model.name} - {model.description}")

# Browse datasets
print("\nAvailable Datasets:")
for ds in list_datasets(tags=["fluid-dynamics"]):
    print(f"  {ds.name} ({ds.size} samples)")

# Browse equations
print("\nFundamental Equations:")
for eq in get_equations(tags=["fundamental"]):
    print(f"  {eq.name}: {eq.formula}")
```

## Model Cards

The hub also supports detailed model cards for documentation:

```python
from dimtensor.hub import ModelCard

card = ModelCard(
    model_name="velocity-predictor-v1",
    description="LSTM-based velocity predictor",
    training_data="10M trajectory samples from physics simulations",
    performance_metrics={
        "mse": 0.001,
        "r2": 0.99
    },
    limitations=[
        "Works best for velocities < 100 m/s",
        "Assumes constant mass"
    ],
    use_cases=[
        "Trajectory prediction",
        "State estimation"
    ],
    ethical_considerations="None - physics simulation only"
)

card.save("velocity-predictor-v1-card.json")
```

::: dimtensor.hub.ModelCard
    options:
      members:
        - model_name
        - description
        - training_data
        - performance_metrics
        - limitations
        - use_cases
        - ethical_considerations
        - save
        - load

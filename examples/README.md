# dimtensor Examples

This directory contains Jupyter notebooks demonstrating real-world usage of dimtensor for scientific computing, machine learning, and data analysis.

## Overview

These notebooks showcase dimtensor's capabilities:
- Unit-aware tensor operations with automatic dimensional checking
- Physical constants from CODATA 2022 with uncertainties
- Integration with NumPy, PyTorch, and JAX
- Physics simulations with conservation law verification
- Machine learning with physics-informed neural networks (PINNs)
- Real-world data analysis with proper unit tracking

## Available Notebooks

### 01_basics.ipynb
**Beginner-friendly introduction to dimtensor fundamentals**

Learn the core concepts:
- Creating DimArrays with units
- Available units (SI, derived, non-SI)
- Unit conversions and automatic simplification
- Arithmetic operations with dimensional safety
- Physical constants with uncertainties
- Uncertainty propagation through calculations
- Array operations and NumPy integration

**Best for:** First-time users, learning dimensional analysis basics

**Duration:** 15-20 minutes

### 02_physics_simulation.ipynb
**Classical mechanics simulations with dimensional validation**

Three physics examples of increasing complexity:
1. **Projectile motion with air resistance** - 2D dynamics, RK4 integration, energy dissipation
2. **Simple pendulum** - Nonlinear oscillations, phase space, period vs amplitude
3. **Orbital mechanics** - Gravitational 2-body problem, Kepler's laws, conservation checks

Demonstrates:
- Catching physics bugs through dimensional analysis
- Using CODATA physical constants (G, c, h, etc.)
- Numerical integration with unit tracking
- Energy and momentum conservation validation
- Automatic plot labeling with units

**Best for:** Physics students, researchers, scientific computing

**Duration:** 30-45 minutes

### 03_pytorch_training.ipynb
**Training a Physics-Informed Neural Network (PINN)**

Solve the 1D heat equation using a PINN:
- Problem: ∂T/∂t = α ∂²T/∂x² (thermal diffusion)
- Build dimension-aware neural networks (DimLinear, DimSequential)
- Combine data loss and physics loss (PDE residual)
- Train with automatic dimensional validation
- Verify solution against analytical ground truth
- Check physical conservation laws

Demonstrates:
- PyTorch integration (DimTensor with autograd)
- Physics-informed loss functions
- Unit tracking through backpropagation
- Model evaluation with dimensional checks

**Best for:** ML researchers, computational physics, scientific ML

**Duration:** 45-60 minutes

**Requirements:** PyTorch, matplotlib

### 04_data_analysis.ipynb
**Real-world data analysis with exoplanet data**

Analyze NASA exoplanet archive data:
- Load physical constants (G, c, masses, etc.)
- Work with exoplanet measurements (mass, radius, orbital parameters)
- Convert between unit systems (AU/km, days/years, Jupiter/Earth masses)
- Compute derived quantities (surface gravity, escape velocity)
- Statistical analysis with units preserved
- Verify Kepler's 3rd law
- Create publication-quality plots
- Export data (JSON, CSV, HDF5) with units

Demonstrates:
- CODATA constants with uncertainties
- Unit conversions across multiple systems
- Statistical analysis with dimensional safety
- Data import/export with metadata
- Visualization with automatic axis labels

**Best for:** Data scientists, astronomers, scientific data analysis

**Duration:** 30-40 minutes

**Requirements:** pandas (optional), matplotlib (optional), h5py (optional)

## Prerequisites

### Required
```bash
pip install dimtensor
```

### Optional (for full functionality)
```bash
# For PyTorch integration (notebook 03)
pip install torch

# For data analysis (notebook 04)
pip install pandas matplotlib h5py

# Install everything at once
pip install dimtensor[all]
```

### Environment
- Python 3.8+
- NumPy >= 1.24.0, < 2.0.0
- Jupyter notebook or JupyterLab

## Running the Notebooks

### Option 1: Jupyter Notebook
```bash
cd examples
jupyter notebook
```

Then open any `.ipynb` file from the browser interface.

### Option 2: JupyterLab
```bash
cd examples
jupyter lab
```

### Option 3: VS Code
Open the folder in VS Code with the Jupyter extension installed, then open any `.ipynb` file.

### Option 4: Google Colab
Upload the notebooks to Google Colab and install dimtensor:
```python
!pip install dimtensor[all]
```

## Learning Path

**New to dimtensor?** Follow this sequence:
1. Start with `01_basics.ipynb` to learn fundamentals
2. Try `04_data_analysis.ipynb` for practical applications
3. Explore `02_physics_simulation.ipynb` for scientific computing
4. Advanced users: `03_pytorch_training.ipynb` for ML integration

**Looking for specific topics?**
- Unit conversions → `01_basics.ipynb`, `04_data_analysis.ipynb`
- Physical constants → `01_basics.ipynb`, `02_physics_simulation.ipynb`, `04_data_analysis.ipynb`
- Numerical simulation → `02_physics_simulation.ipynb`
- Machine learning → `03_pytorch_training.ipynb`
- Data import/export → `04_data_analysis.ipynb`

## Key Features Demonstrated

| Feature | Notebook(s) |
|---------|------------|
| Creating DimArrays | 01, 02, 03, 04 |
| Unit conversions | 01, 04 |
| Dimensional safety | 01, 02, 03 |
| Physical constants | 01, 02, 04 |
| Uncertainty propagation | 01 |
| NumPy integration | 01, 02, 04 |
| PyTorch integration | 03 |
| Physics simulations | 02 |
| Conservation laws | 02, 03 |
| PINNs (physics-informed ML) | 03 |
| Data analysis | 04 |
| Statistical operations | 04 |
| Visualization | 02, 03, 04 |
| Data serialization | 04 |

## Documentation

- **Main documentation**: https://github.com/ArcheausGalacto/dimtensor
- **API reference**: See `docs/` directory
- **Installation guide**: See main `README.md`
- **CODATA constants**: See `src/dimtensor/constants/`

## Troubleshooting

### NumPy version error
If you see errors about NumPy 2.x:
```bash
pip install "numpy>=1.24.0,<2.0.0" --force-reinstall
```

### Missing dependencies
Install optional dependencies:
```bash
pip install dimtensor[torch]  # For PyTorch integration
pip install dimtensor[all]    # For everything
```

### Notebook won't run
Ensure you're in the correct environment:
```bash
python -c "import dimtensor; print(dimtensor.__version__)"
```

### Import errors in notebooks
Restart the Jupyter kernel: `Kernel → Restart Kernel`

## Contributing

Found an issue or have a suggestion for a new example?
- Open an issue: https://github.com/ArcheausGalacto/dimtensor/issues
- Submit a PR with a new notebook
- Share your dimtensor projects!

## License

These examples are part of the dimtensor project and distributed under the same license (see main repository).

## Citation

If you use dimtensor in your research, please cite:
```bibtex
@software{dimtensor,
  title = {dimtensor: Unit-aware tensors for scientific computing},
  author = {dimtensor contributors},
  year = {2025},
  url = {https://github.com/ArcheausGalacto/dimtensor}
}
```

---

**Questions?** Open an issue or consult the main documentation.

**Happy computing with dimensional safety!**

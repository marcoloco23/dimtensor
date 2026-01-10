# Plan: TensorFlow Integration

**Date**: 2026-01-10
**Status**: PLANNING
**Author**: planner agent
**Tasks**: #199, #200

---

## Goal

Implement TensorFlow integration for dimtensor, providing DimTensor and DimVariable classes that wrap tf.Tensor and tf.Variable with physical unit tracking, supporting both eager and graph execution modes.

---

## Background

dimtensor already has excellent integrations with:
- **PyTorch** (`src/dimtensor/torch/`): Full DimTensor with autograd, layers, losses, normalization, scalers, priors
- **JAX** (`src/dimtensor/jax/`): DimArray with pytree registration for JIT/vmap/grad

TensorFlow integration is task #199-200 in the roadmap (v4.2.0 framework integrations). TensorFlow is unique because:
1. It has two execution modes: eager (default in TF2) and graph mode
2. Variables are separate from tensors (`tf.Variable` vs `tf.Tensor`)
3. Keras layers use a different paradigm than PyTorch nn.Module
4. GradientTape is the autograd mechanism

---

## Approach

### Option A: Minimal Integration (like JAX)
- Only implement core DimTensor wrapper
- No Keras layers, losses, etc.
- Pros: Fast to implement, low maintenance
- Cons: Less useful for ML practitioners

### Option B: Full Integration (like PyTorch)
- DimTensor (tf.Tensor wrapper)
- DimVariable (tf.Variable wrapper)
- DimLayer base class for Keras
- Dimension-aware losses, optimizers
- Pros: Full parity with PyTorch, comprehensive
- Cons: More work, more maintenance

### Option C: Staged Integration (Recommended)
- **Phase 1**: Core DimTensor + DimVariable (this plan)
- **Phase 2**: Keras layers, losses (future plan)
- Pros: Get usable integration quickly, can iterate
- Cons: Users must wait for full ML support

### Decision: Option C (Staged)

Start with core tensor/variable wrappers that support all arithmetic, unit conversion, and gradient computation. This gives users immediate value for physics computations while we iterate on ML layers.

---

## Implementation Steps

### Phase 1: Core Classes (Tasks #199, #200)

#### 1. [ ] Create directory structure
```
src/dimtensor/tensorflow/
    __init__.py
    dimtensor.py      # DimTensor class
    dimvariable.py    # DimVariable class (extends DimTensor)
```

#### 2. [ ] Implement DimTensor (tf.Tensor wrapper)
Pattern from `torch/dimtensor.py`:
- `__slots__ = ("_data", "_unit")`
- Properties: `data`, `unit`, `dimension`, `shape`, `ndim`, `dtype`, `device`
- `_from_tensor_and_unit()` internal constructor
- Unit conversion: `to_unit()`, `magnitude()`
- Arithmetic: `+`, `-`, `*`, `/`, `**`, `@` with dimension checking
- Comparisons: `<`, `<=`, `>`, `>=`, `==`
- Indexing: `__getitem__`, `__len__`
- Reductions: `sum`, `mean`, `std`, `var`, `min`, `max`
- Reshaping: `reshape`, `transpose`, `flatten`, `squeeze`, `expand_dims`
- Linear algebra: `matmul`, `dot`

TensorFlow-specific considerations:
- Use `tf.Tensor` instead of `torch.Tensor`
- Handle eager vs graph mode (should work in both)
- Use `tf.identity()` for tensor creation
- Device placement via `tf.device()` context

#### 3. [ ] Implement DimVariable (tf.Variable wrapper)
New class extending DimTensor concept for mutable variables:
- Wraps `tf.Variable` instead of `tf.Tensor`
- Adds `.assign()`, `.assign_add()`, `.assign_sub()` methods
- Trainable flag for gradient computation
- Constraint support (for unit-aware constraints)

#### 4. [ ] GradientTape integration
```python
with tf.GradientTape() as tape:
    tape.watch(dim_var.data)  # Watch underlying tensor
    loss = compute_loss(dim_var)
    grads = tape.gradient(loss.data, dim_var.data)
```
- Gradients strip units (like PyTorch)
- Or: wrap gradients with inverse units (optional feature)

#### 5. [ ] Eager and Graph mode testing
- Verify all operations work in eager mode (default)
- Verify operations work inside `@tf.function`
- Test gradient computation in both modes

#### 6. [ ] Unit tests
```
tests/tensorflow/
    test_dimtensor.py
    test_dimvariable.py
    test_gradient_tape.py
    test_graph_mode.py
```

#### 7. [ ] Update package exports
- Add `tensorflow` to optional dependencies in pyproject.toml
- Export from main `__init__.py` (conditional import)

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/dimtensor/tensorflow/__init__.py` | Module exports |
| `src/dimtensor/tensorflow/dimtensor.py` | DimTensor class for tf.Tensor |
| `src/dimtensor/tensorflow/dimvariable.py` | DimVariable class for tf.Variable |
| `tests/tensorflow/test_dimtensor.py` | DimTensor unit tests |
| `tests/tensorflow/test_dimvariable.py` | DimVariable unit tests |
| `tests/tensorflow/test_gradient_tape.py` | Gradient computation tests |
| `tests/tensorflow/test_graph_mode.py` | @tf.function tests |

## Files to Modify

| File | Change |
|------|--------|
| `pyproject.toml` | Add tensorflow optional dependency |
| `src/dimtensor/__init__.py` | Add conditional tensorflow import |

---

## API Design

### DimTensor

```python
from dimtensor.tensorflow import DimTensor
from dimtensor import units
import tensorflow as tf

# Creation
velocity = DimTensor(tf.constant([1.0, 2.0, 3.0]), units.m / units.s)
time = DimTensor([0.5, 1.0, 1.5], units.s)  # auto-convert to tf.Tensor

# Properties
velocity.data      # tf.Tensor
velocity.unit      # Unit
velocity.dimension # Dimension
velocity.shape     # TensorShape
velocity.dtype     # tf.DType
velocity.device    # string like '/device:CPU:0'

# Arithmetic (dimension-aware)
distance = velocity * time        # DimTensor in meters
kinetic = 0.5 * mass * velocity**2  # DimTensor in joules

# Unit conversion
velocity_kmh = velocity.to_unit(units.km / units.hour)

# Reductions
avg_velocity = velocity.mean()    # DimTensor preserves unit
total_distance = distance.sum()

# Works with @tf.function
@tf.function
def compute_energy(m, v):
    return 0.5 * m * v**2

E = compute_energy(mass, velocity)
```

### DimVariable

```python
from dimtensor.tensorflow import DimVariable
from dimtensor import units
import tensorflow as tf

# Creation (trainable by default)
position = DimVariable(tf.zeros([3]), units.m, name='position')
momentum = DimVariable([0.0, 0.0, 0.0], units.kg * units.m / units.s)

# Variable properties
position.trainable   # True
position.name        # 'position:0'

# Assignment
position.assign([1.0, 2.0, 3.0])         # in-place update
position.assign_add([0.1, 0.1, 0.1])     # increment

# Gradient computation
with tf.GradientTape() as tape:
    energy = 0.5 * mass * (velocity ** 2)
    loss = energy.sum()

grads = tape.gradient(loss.data, [mass.data, velocity.data])
```

---

## Testing Strategy

### Unit Tests
- [ ] DimTensor creation from various inputs (tf.Tensor, list, scalar)
- [ ] Property accessors (data, unit, shape, etc.)
- [ ] All arithmetic operations with dimension checking
- [ ] Unit conversion (to_unit, magnitude)
- [ ] Reduction operations (sum, mean, std, var, min, max)
- [ ] Reshaping operations (reshape, transpose, flatten, etc.)
- [ ] Comparison operations
- [ ] Indexing and slicing

### Integration Tests
- [ ] GradientTape gradient computation
- [ ] @tf.function graph mode execution
- [ ] Mixed DimTensor/tf.Tensor operations
- [ ] GPU device placement (if available)
- [ ] Checkpoint save/restore for DimVariable

### Error Handling Tests
- [ ] DimensionError on incompatible operations
- [ ] UnitConversionError on incompatible conversions
- [ ] Proper error messages with dimension info

---

## Risks / Edge Cases

1. **Risk**: TensorFlow's static graph mode may not support dynamic unit tracking
   - **Mitigation**: Units are compile-time constants, stored as Python objects outside the graph
   - **Note**: Unit is static metadata, not a traced tensor

2. **Risk**: tf.function retracing when units change
   - **Mitigation**: Units stored in `_unit` attribute are not traced (Python object)
   - **Test**: Verify no retracing when only magnitudes change

3. **Edge case**: Operations between DimTensor and raw tf.Tensor
   - **Handling**: Like PyTorch, allow scalar multiplication but require dimensionless for add/sub

4. **Edge case**: tf.Variable constraints
   - **Handling**: Constraints operate on magnitude only (unit-aware constraints in Phase 2)

5. **Risk**: TensorFlow version compatibility
   - **Mitigation**: Target TF 2.x (2.10+), require eager execution by default
   - **Note**: TF 1.x graph mode not supported

6. **Edge case**: Mixed CPU/GPU operations
   - **Handling**: Unit is Python object, only tensor moves between devices

---

## Definition of Done

- [ ] DimTensor class implemented with all operations
- [ ] DimVariable class implemented with assignment methods
- [ ] All tests pass in eager mode
- [ ] All tests pass in graph mode (@tf.function)
- [ ] GradientTape gradient computation works
- [ ] pyproject.toml updated with tensorflow optional dependency
- [ ] Main __init__.py exports tensorflow module (conditional)
- [ ] CONTINUITY.md updated with task completion
- [ ] CHANGELOG.md entry added

---

## Future Work (Phase 2)

After Phase 1 is complete:
- **DimLayer**: Base class for Keras layers with dimension tracking
- **DimDense**: Dense layer with input/output dimensions
- **DimConv1D/2D**: Convolutional layers
- **DimLoss**: MSE, MAE, etc. with dimension checking
- **DimOptimizer**: Optimizer wrapper that respects units
- **Keras Model integration**: Functional and subclass APIs

---

## Notes / Log

**2026-01-10** - Plan created by planner agent. Analyzed existing PyTorch and JAX integrations to establish patterns. Recommended staged approach starting with core tensor classes.

---

## Reference: Existing Patterns

### PyTorch DimTensor Pattern
```python
class DimTensor:
    __slots__ = ("_data", "_unit")

    def __init__(self, data, unit=None, dtype=None, device=None, requires_grad=False):
        # Handle DimTensor, Tensor, or raw data
        # Apply dtype, device, requires_grad
        self._data = tensor
        self._unit = unit or dimensionless

    @classmethod
    def _from_tensor_and_unit(cls, tensor, unit):
        # Internal constructor without copying
        result = object.__new__(cls)
        result._data = tensor
        result._unit = unit
        return result

    def __add__(self, other):
        if isinstance(other, DimTensor):
            if self.dimension != other.dimension:
                raise DimensionError.incompatible(...)
            converted = other.to_unit(self._unit)
            new_tensor = self._data + converted._data
            return DimTensor._from_tensor_and_unit(new_tensor, self._unit)
        else:
            if not self.is_dimensionless:
                raise DimensionError(...)
            ...
```

### JAX DimArray Pattern (Pytree Registration)
```python
def _dimarray_flatten(arr):
    return (arr._data,), (arr._unit,)

def _dimarray_unflatten(aux_data, children):
    (data,) = children
    (unit,) = aux_data
    return DimArray._from_data_and_unit(data, unit)

jax.tree_util.register_pytree_node(DimArray, _dimarray_flatten, _dimarray_unflatten)
```

For TensorFlow, similar approach: unit is static metadata, tensor is the only traced value.

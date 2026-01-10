"""Tests for TensorFlow DimTensor and DimVariable integration."""

import pytest
import os

# Suppress TensorFlow warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    import tensorflow as tf
    # Verify TensorFlow actually works by creating a simple tensor
    _test = tf.constant([1.0])
    TF_AVAILABLE = True
except (ImportError, RuntimeError, SystemError, OSError):
    TF_AVAILABLE = False
except Exception:
    # Catch any other TensorFlow initialization errors
    TF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available or not working")

if TF_AVAILABLE:
    from dimtensor.tensorflow import DimTensor, DimVariable

from dimtensor import units
from dimtensor.errors import DimensionError, UnitConversionError


class TestDimTensorCreation:
    """Tests for TensorFlow DimTensor creation."""

    def test_from_tf_tensor(self):
        """Create DimTensor from tf.Tensor."""
        tensor = tf.constant([1.0, 2.0, 3.0])
        dt = DimTensor(tensor, units.m)
        assert dt.shape == (3,)
        assert dt.unit == units.m

    def test_from_list(self):
        """Create DimTensor from list."""
        dt = DimTensor([1.0, 2.0, 3.0], units.m)
        assert dt.shape == (3,)
        assert dt.unit == units.m

    def test_from_scalar(self):
        """Create DimTensor from scalar."""
        dt = DimTensor(5.0, units.kg)
        assert dt.size == 1
        assert dt.unit == units.kg

    def test_default_dimensionless(self):
        """Default unit is dimensionless."""
        dt = DimTensor(tf.constant([1.0]))
        assert dt.is_dimensionless

    def test_with_dtype(self):
        """Create with specific dtype."""
        dt = DimTensor([1.0, 2.0], units.m, dtype=tf.float64)
        assert dt.dtype == tf.float64


class TestDimTensorProperties:
    """Tests for DimTensor properties."""

    def test_shape(self):
        """Test shape property."""
        dt = DimTensor(tf.zeros((2, 3, 4)), units.m)
        assert dt.shape == (2, 3, 4)

    def test_ndim(self):
        """Test ndim property."""
        dt = DimTensor(tf.zeros((2, 3)), units.m)
        assert dt.ndim == 2

    def test_size(self):
        """Test size property."""
        dt = DimTensor(tf.zeros((2, 3)), units.m)
        assert dt.size == 6

    def test_dimension(self):
        """Test dimension property."""
        dt = DimTensor([1.0], units.m / units.s)
        assert dt.dimension == (units.m / units.s).dimension

    def test_device(self):
        """Test device property."""
        dt = DimTensor([1.0], units.m)
        # Should return a string indicating device
        assert isinstance(dt.device, str)


class TestDimTensorArithmetic:
    """Tests for arithmetic operations."""

    def test_add_same_unit(self):
        """Add tensors with same unit."""
        a = DimTensor([1.0, 2.0], units.m)
        b = DimTensor([3.0, 4.0], units.m)
        c = a + b
        assert tf.reduce_all(tf.abs(c.data - tf.constant([4.0, 6.0])) < 1e-6).numpy()
        assert c.unit == units.m

    def test_add_compatible_units(self):
        """Add tensors with compatible units (auto-convert)."""
        a = DimTensor([1000.0], units.m)
        b = DimTensor([1.0], units.km)
        c = a + b
        assert tf.abs(c.data[0] - 2000.0).numpy() < 1e-6
        assert c.unit == units.m

    def test_add_incompatible_raises(self):
        """Adding incompatible dimensions raises error."""
        a = DimTensor([1.0], units.m)
        b = DimTensor([1.0], units.s)
        with pytest.raises(DimensionError):
            a + b

    def test_subtract(self):
        """Subtract tensors."""
        a = DimTensor([5.0, 6.0], units.m)
        b = DimTensor([1.0, 2.0], units.m)
        c = a - b
        assert tf.reduce_all(tf.abs(c.data - tf.constant([4.0, 4.0])) < 1e-6).numpy()

    def test_multiply_dimensions(self):
        """Multiply tensors - dimensions multiply."""
        length = DimTensor([2.0], units.m)
        width = DimTensor([3.0], units.m)
        area = length * width
        assert area.dimension == (units.m**2).dimension
        assert tf.abs(area.data[0] - 6.0).numpy() < 1e-6

    def test_multiply_scalar(self):
        """Multiply by scalar."""
        dt = DimTensor([1.0, 2.0], units.m)
        result = dt * 3.0
        assert tf.reduce_all(tf.abs(result.data - tf.constant([3.0, 6.0])) < 1e-6).numpy()
        assert result.unit == units.m

    def test_rmul_scalar(self):
        """Right multiply by scalar."""
        dt = DimTensor([1.0, 2.0], units.m)
        result = 3.0 * dt
        assert tf.reduce_all(tf.abs(result.data - tf.constant([3.0, 6.0])) < 1e-6).numpy()
        assert result.unit == units.m

    def test_divide_dimensions(self):
        """Divide tensors - dimensions divide."""
        distance = DimTensor([10.0], units.m)
        time = DimTensor([2.0], units.s)
        velocity = distance / time
        assert velocity.dimension == (units.m / units.s).dimension
        assert tf.abs(velocity.data[0] - 5.0).numpy() < 1e-6

    def test_power(self):
        """Power operation scales dimension."""
        length = DimTensor([2.0], units.m)
        area = length ** 2
        assert area.dimension == (units.m**2).dimension
        assert tf.abs(area.data[0] - 4.0).numpy() < 1e-6

    def test_sqrt(self):
        """Square root halves dimension."""
        area = DimTensor([4.0], units.m**2)
        length = area.sqrt()
        assert length.dimension == units.m.dimension
        assert tf.abs(length.data[0] - 2.0).numpy() < 1e-6

    def test_negation(self):
        """Negation preserves unit."""
        dt = DimTensor([1.0, -2.0], units.m)
        neg = -dt
        assert tf.reduce_all(tf.abs(neg.data - tf.constant([-1.0, 2.0])) < 1e-6).numpy()
        assert neg.unit == units.m

    def test_abs(self):
        """Absolute value preserves unit."""
        dt = DimTensor([-1.0, 2.0], units.m)
        result = abs(dt)
        assert tf.reduce_all(tf.abs(result.data - tf.constant([1.0, 2.0])) < 1e-6).numpy()
        assert result.unit == units.m


class TestTFFunction:
    """Tests for @tf.function graph mode execution."""

    def test_tf_function_basic(self):
        """tf.function preserves units."""
        @tf.function
        def double(x):
            return x * 2.0

        dt = DimTensor([1.0, 2.0], units.m)
        result = double(dt)
        assert tf.reduce_all(tf.abs(result.data - tf.constant([2.0, 4.0])) < 1e-6).numpy()
        assert result.unit == units.m

    def test_tf_function_multiply(self):
        """tf.function multiplication preserves dimension algebra."""
        @tf.function
        def kinetic_energy(mass, velocity):
            return 0.5 * mass * velocity ** 2

        m = DimTensor([1.0, 2.0], units.kg)
        v = DimTensor([3.0, 4.0], units.m / units.s)
        E = kinetic_energy(m, v)

        assert tf.reduce_all(tf.abs(E.data - tf.constant([4.5, 16.0])) < 1e-6).numpy()
        assert E.dimension == units.J.dimension

    def test_tf_function_add(self):
        """tf.function addition preserves units."""
        @tf.function
        def add_tensors(a, b):
            return a + b

        a = DimTensor([1.0, 2.0], units.m)
        b = DimTensor([3.0, 4.0], units.m)
        result = add_tensors(a, b)
        assert tf.reduce_all(tf.abs(result.data - tf.constant([4.0, 6.0])) < 1e-6).numpy()
        assert result.unit == units.m

    def test_tf_function_chain(self):
        """tf.function with multiple operations."""
        @tf.function
        def compute(x, v, t):
            return x + v * t

        x = DimTensor([1.0], units.m)
        v = DimTensor([2.0], units.m / units.s)
        t = DimTensor([3.0], units.s)
        result = compute(x, v, t)
        assert tf.abs(result.data[0] - 7.0).numpy() < 1e-6
        assert result.unit == units.m


class TestGradientTape:
    """Tests for GradientTape gradient computation."""

    def test_gradient_basic(self):
        """Basic gradient computation."""
        dt = DimTensor(tf.constant([3.0]), units.m)

        with tf.GradientTape() as tape:
            tape.watch(dt.data)
            result = dt ** 2
            loss = result.data[0]

        grad = tape.gradient(loss, dt.data)
        # Gradient of x^2 is 2x = 6.0
        assert tf.abs(grad[0] - 6.0).numpy() < 1e-6

    def test_gradient_with_dimvariable(self):
        """Gradient computation with DimVariable."""
        mass = DimVariable([1.0], units.kg, trainable=True)
        velocity = DimVariable([2.0], units.m / units.s, trainable=True)

        with tf.GradientTape() as tape:
            energy = 0.5 * mass * velocity ** 2
            loss = energy.sum().data

        grads = tape.gradient(loss, [mass.variable, velocity.variable])
        assert grads[0] is not None
        assert grads[1] is not None


class TestDimTensorUnitConversion:
    """Tests for unit conversion."""

    def test_to_unit(self):
        """Convert to compatible unit."""
        dt = DimTensor([1000.0], units.m)
        dt_km = dt.to_unit(units.km)
        assert tf.abs(dt_km.data[0] - 1.0).numpy() < 1e-6
        assert dt_km.unit == units.km

    def test_to_unit_incompatible_raises(self):
        """Converting to incompatible unit raises error."""
        dt = DimTensor([1.0], units.m)
        with pytest.raises(UnitConversionError):
            dt.to_unit(units.s)

    def test_magnitude(self):
        """Magnitude returns raw tensor."""
        dt = DimTensor([1.0, 2.0], units.m)
        mag = dt.magnitude()
        assert isinstance(mag, tf.Tensor)
        assert tf.reduce_all(tf.abs(mag - tf.constant([1.0, 2.0])) < 1e-6).numpy()


class TestDimTensorReductions:
    """Tests for reduction operations."""

    def test_sum(self):
        """Sum preserves unit."""
        dt = DimTensor([1.0, 2.0, 3.0], units.m)
        result = dt.sum()
        assert tf.abs(result.data - 6.0).numpy() < 1e-6
        assert result.unit == units.m

    def test_mean(self):
        """Mean preserves unit."""
        dt = DimTensor([1.0, 2.0, 3.0], units.m)
        result = dt.mean()
        assert tf.abs(result.data - 2.0).numpy() < 1e-6
        assert result.unit == units.m

    def test_std(self):
        """Standard deviation preserves unit."""
        dt = DimTensor([1.0, 2.0, 3.0], units.m)
        result = dt.std()
        assert result.unit == units.m

    def test_var_squares_unit(self):
        """Variance squares unit."""
        dt = DimTensor([1.0, 2.0, 3.0], units.m)
        result = dt.var()
        assert result.dimension == (units.m**2).dimension

    def test_min(self):
        """Min preserves unit."""
        dt = DimTensor([3.0, 1.0, 2.0], units.m)
        result = dt.min()
        assert tf.abs(result.data - 1.0).numpy() < 1e-6
        assert result.unit == units.m

    def test_max(self):
        """Max preserves unit."""
        dt = DimTensor([3.0, 1.0, 2.0], units.m)
        result = dt.max()
        assert tf.abs(result.data - 3.0).numpy() < 1e-6
        assert result.unit == units.m

    def test_sum_axis(self):
        """Sum along axis."""
        dt = DimTensor(tf.constant([[1.0, 2.0], [3.0, 4.0]]), units.m)
        result = dt.sum(axis=0)
        assert result.shape == (2,)
        assert result.unit == units.m


class TestDimTensorReshaping:
    """Tests for reshaping operations."""

    def test_reshape(self):
        """Reshape preserves unit."""
        dt = DimTensor(tf.zeros((2, 3)), units.m)
        reshaped = dt.reshape((6,))
        assert reshaped.shape == (6,)
        assert reshaped.unit == units.m

    def test_transpose(self):
        """Transpose preserves unit."""
        dt = DimTensor(tf.zeros((2, 3)), units.m)
        transposed = dt.transpose()
        assert transposed.shape == (3, 2)
        assert transposed.unit == units.m

    def test_flatten(self):
        """Flatten preserves unit."""
        dt = DimTensor(tf.zeros((2, 3)), units.m)
        flat = dt.flatten()
        assert flat.shape == (6,)
        assert flat.unit == units.m

    def test_squeeze(self):
        """Squeeze preserves unit."""
        dt = DimTensor(tf.zeros((2, 1, 3)), units.m)
        squeezed = dt.squeeze(axis=1)
        assert squeezed.shape == (2, 3)
        assert squeezed.unit == units.m

    def test_expand_dims(self):
        """Expand dims preserves unit."""
        dt = DimTensor(tf.zeros((2, 3)), units.m)
        expanded = dt.expand_dims(axis=0)
        assert expanded.shape == (1, 2, 3)
        assert expanded.unit == units.m


class TestDimTensorLinearAlgebra:
    """Tests for linear algebra operations."""

    def test_matmul(self):
        """Matrix multiplication multiplies dimensions."""
        a = DimTensor(tf.ones((2, 3)), units.m)
        b = DimTensor(tf.ones((3, 4)), units.s)
        c = a.matmul(b)
        assert c.shape == (2, 4)
        assert c.dimension == (units.m * units.s).dimension

    def test_matmul_operator(self):
        """@ operator works for matmul."""
        a = DimTensor(tf.ones((2, 3)), units.m)
        b = DimTensor(tf.ones((3, 4)), units.s)
        c = a @ b
        assert c.dimension == (units.m * units.s).dimension

    def test_dot(self):
        """Dot product multiplies dimensions."""
        a = DimTensor([1.0, 2.0, 3.0], units.m)
        b = DimTensor([4.0, 5.0, 6.0], units.s)
        c = a.dot(b)
        assert c.dimension == (units.m * units.s).dimension


class TestDimTensorComparison:
    """Tests for comparison operations."""

    def test_eq_same_unit(self):
        """Equality with same unit."""
        a = DimTensor([1.0, 2.0], units.m)
        b = DimTensor([1.0, 3.0], units.m)
        result = a == b
        assert result[0].numpy() == True
        assert result[1].numpy() == False

    def test_lt(self):
        """Less than comparison."""
        a = DimTensor([1.0, 3.0], units.m)
        b = DimTensor([2.0, 2.0], units.m)
        result = a < b
        assert result[0].numpy() == True
        assert result[1].numpy() == False

    def test_le(self):
        """Less than or equal comparison."""
        a = DimTensor([1.0, 2.0], units.m)
        b = DimTensor([2.0, 2.0], units.m)
        result = a <= b
        assert result[0].numpy() == True
        assert result[1].numpy() == True

    def test_gt(self):
        """Greater than comparison."""
        a = DimTensor([3.0, 1.0], units.m)
        b = DimTensor([2.0, 2.0], units.m)
        result = a > b
        assert result[0].numpy() == True
        assert result[1].numpy() == False

    def test_ge(self):
        """Greater than or equal comparison."""
        a = DimTensor([2.0, 3.0], units.m)
        b = DimTensor([2.0, 2.0], units.m)
        result = a >= b
        assert result[0].numpy() == True
        assert result[1].numpy() == True

    def test_compare_incompatible_raises(self):
        """Comparing incompatible dimensions raises error."""
        a = DimTensor([1.0], units.m)
        b = DimTensor([1.0], units.s)
        with pytest.raises(DimensionError):
            a < b


class TestDimTensorIndexing:
    """Tests for indexing operations."""

    def test_getitem(self):
        """Indexing preserves unit."""
        dt = DimTensor([1.0, 2.0, 3.0], units.m)
        item = dt[1]
        assert tf.abs(item.data - 2.0).numpy() < 1e-6
        assert item.unit == units.m

    def test_slice(self):
        """Slicing preserves unit."""
        dt = DimTensor([1.0, 2.0, 3.0, 4.0], units.m)
        sliced = dt[1:3]
        assert sliced.shape == (2,)
        assert sliced.unit == units.m

    def test_len(self):
        """Length returns first dimension size."""
        dt = DimTensor(tf.zeros((5, 3)), units.m)
        assert len(dt) == 5


class TestDimTensorStrings:
    """Tests for string representations."""

    def test_repr(self):
        """repr includes data and unit."""
        dt = DimTensor([1.0], units.m)
        r = repr(dt)
        assert "DimTensor" in r
        assert "m" in r

    def test_str_with_unit(self):
        """str shows data and unit."""
        dt = DimTensor([1.0], units.m)
        s = str(dt)
        assert "m" in s


class TestDimTensorConversion:
    """Tests for conversion methods."""

    def test_numpy(self):
        """Convert to numpy array."""
        dt = DimTensor([1.0, 2.0], units.m)
        arr = dt.numpy()
        assert arr.shape == (2,)
        assert arr[0] == pytest.approx(1.0)

    def test_item(self):
        """Get single element as Python scalar."""
        dt = DimTensor([5.0], units.m)
        val = dt.item()
        assert val == pytest.approx(5.0)


class TestDimTensorDtype:
    """Tests for dtype operations."""

    def test_float(self):
        """Cast to float32."""
        dt = DimTensor([1.0, 2.0], units.m, dtype=tf.float64)
        dt32 = dt.float()
        assert dt32.dtype == tf.float32
        assert dt32.unit == units.m

    def test_double(self):
        """Cast to float64."""
        dt = DimTensor([1.0, 2.0], units.m, dtype=tf.float32)
        dt64 = dt.double()
        assert dt64.dtype == tf.float64
        assert dt64.unit == units.m


# =============================================================================
# DimVariable Tests
# =============================================================================


class TestDimVariableCreation:
    """Tests for DimVariable creation."""

    def test_from_list(self):
        """Create DimVariable from list."""
        dv = DimVariable([1.0, 2.0, 3.0], units.m)
        assert dv.shape == (3,)
        assert dv.unit == units.m

    def test_from_tf_tensor(self):
        """Create DimVariable from tf.Tensor."""
        tensor = tf.constant([1.0, 2.0])
        dv = DimVariable(tensor, units.kg)
        assert dv.shape == (2,)
        assert dv.unit == units.kg

    def test_with_name(self):
        """Create with name."""
        dv = DimVariable([1.0], units.m, name="position")
        assert "position" in dv.name

    def test_trainable_default(self):
        """Default is trainable."""
        dv = DimVariable([1.0], units.m)
        assert dv.trainable == True

    def test_not_trainable(self):
        """Create non-trainable variable."""
        dv = DimVariable([1.0], units.m, trainable=False)
        assert dv.trainable == False


class TestDimVariableProperties:
    """Tests for DimVariable properties."""

    def test_data(self):
        """Data returns tensor value."""
        dv = DimVariable([1.0, 2.0], units.m)
        data = dv.data
        assert isinstance(data, tf.Tensor)

    def test_variable(self):
        """Variable returns tf.Variable."""
        dv = DimVariable([1.0, 2.0], units.m)
        var = dv.variable
        assert isinstance(var, tf.Variable)


class TestDimVariableAssignment:
    """Tests for in-place assignment operations."""

    def test_assign(self):
        """Assign new value."""
        dv = DimVariable([1.0, 2.0], units.m)
        dv.assign([3.0, 4.0])
        assert tf.reduce_all(tf.abs(dv.data - tf.constant([3.0, 4.0])) < 1e-6).numpy()

    def test_assign_add(self):
        """Add value in-place."""
        dv = DimVariable([1.0, 2.0], units.m)
        dv.assign_add([0.5, 0.5])
        assert tf.reduce_all(tf.abs(dv.data - tf.constant([1.5, 2.5])) < 1e-6).numpy()

    def test_assign_sub(self):
        """Subtract value in-place."""
        dv = DimVariable([2.0, 3.0], units.m)
        dv.assign_sub([0.5, 1.0])
        assert tf.reduce_all(tf.abs(dv.data - tf.constant([1.5, 2.0])) < 1e-6).numpy()

    def test_assign_dimtensor(self):
        """Assign from DimTensor with unit conversion."""
        dv = DimVariable([0.0], units.m)
        dt = DimTensor([1.0], units.km)
        dv.assign(dt)
        assert tf.abs(dv.data[0] - 1000.0).numpy() < 1e-6

    def test_assign_incompatible_raises(self):
        """Assigning incompatible dimension raises error."""
        dv = DimVariable([1.0], units.m)
        dt = DimTensor([1.0], units.s)
        with pytest.raises(DimensionError):
            dv.assign(dt)

    def test_assign_returns_self(self):
        """Assign returns self for chaining."""
        dv = DimVariable([1.0], units.m)
        result = dv.assign([2.0])
        assert result is dv


class TestDimVariableArithmetic:
    """Tests for DimVariable arithmetic (returns DimTensor)."""

    def test_add(self):
        """Add returns DimTensor."""
        dv = DimVariable([1.0, 2.0], units.m)
        dt = DimTensor([3.0, 4.0], units.m)
        result = dv + dt
        assert isinstance(result, DimTensor)
        assert tf.reduce_all(tf.abs(result.data - tf.constant([4.0, 6.0])) < 1e-6).numpy()

    def test_multiply(self):
        """Multiply returns DimTensor."""
        dv = DimVariable([2.0], units.m)
        result = dv * 3.0
        assert isinstance(result, DimTensor)
        assert tf.abs(result.data[0] - 6.0).numpy() < 1e-6

    def test_power(self):
        """Power returns DimTensor with scaled dimension."""
        dv = DimVariable([2.0], units.m)
        result = dv ** 2
        assert isinstance(result, DimTensor)
        assert result.dimension == (units.m**2).dimension


class TestDimVariableConversion:
    """Tests for DimVariable conversion."""

    def test_to_tensor(self):
        """Convert to DimTensor."""
        dv = DimVariable([1.0, 2.0], units.m)
        dt = dv.to_tensor()
        assert isinstance(dt, DimTensor)
        assert dt.unit == units.m

    def test_to_unit(self):
        """Convert to different unit (returns DimTensor)."""
        dv = DimVariable([1000.0], units.m)
        dt = dv.to_unit(units.km)
        assert isinstance(dt, DimTensor)
        assert tf.abs(dt.data[0] - 1.0).numpy() < 1e-6


class TestDimVariableReductions:
    """Tests for DimVariable reductions (return DimTensor)."""

    def test_sum(self):
        """Sum returns DimTensor."""
        dv = DimVariable([1.0, 2.0, 3.0], units.m)
        result = dv.sum()
        assert isinstance(result, DimTensor)
        assert tf.abs(result.data - 6.0).numpy() < 1e-6

    def test_mean(self):
        """Mean returns DimTensor."""
        dv = DimVariable([1.0, 2.0, 3.0], units.m)
        result = dv.mean()
        assert isinstance(result, DimTensor)
        assert tf.abs(result.data - 2.0).numpy() < 1e-6


class TestDimVariableGradients:
    """Tests for gradient computation with DimVariable."""

    def test_gradient_tape(self):
        """Compute gradients through DimVariable."""
        dv = DimVariable([2.0], units.m, trainable=True)

        with tf.GradientTape() as tape:
            result = dv ** 2
            loss = result.sum().data

        grad = tape.gradient(loss, dv.variable)
        # Gradient of x^2 is 2x = 4.0
        assert tf.abs(grad[0] - 4.0).numpy() < 1e-6

    def test_multiple_variables(self):
        """Gradient with multiple variables."""
        mass = DimVariable([2.0], units.kg, trainable=True)
        velocity = DimVariable([3.0], units.m / units.s, trainable=True)

        with tf.GradientTape() as tape:
            # E = 0.5 * m * v^2
            energy = 0.5 * mass * velocity ** 2
            loss = energy.sum().data

        grads = tape.gradient(loss, [mass.variable, velocity.variable])
        # dE/dm = 0.5 * v^2 = 0.5 * 9 = 4.5
        assert tf.abs(grads[0][0] - 4.5).numpy() < 1e-6
        # dE/dv = m * v = 2 * 3 = 6
        assert tf.abs(grads[1][0] - 6.0).numpy() < 1e-6


class TestDimVariableStrings:
    """Tests for string representations."""

    def test_repr(self):
        """repr includes data, unit, and name."""
        dv = DimVariable([1.0], units.m, name="test_var")
        r = repr(dv)
        assert "DimVariable" in r
        assert "m" in r


class TestMixedOperations:
    """Tests for operations between DimTensor and DimVariable."""

    def test_tensor_plus_variable(self):
        """Add DimTensor and DimVariable."""
        dt = DimTensor([1.0, 2.0], units.m)
        dv = DimVariable([3.0, 4.0], units.m)
        result = dt + dv
        assert isinstance(result, DimTensor)
        assert tf.reduce_all(tf.abs(result.data - tf.constant([4.0, 6.0])) < 1e-6).numpy()

    def test_variable_plus_tensor(self):
        """Add DimVariable and DimTensor."""
        dv = DimVariable([1.0, 2.0], units.m)
        dt = DimTensor([3.0, 4.0], units.m)
        result = dv + dt
        assert isinstance(result, DimTensor)
        assert tf.reduce_all(tf.abs(result.data - tf.constant([4.0, 6.0])) < 1e-6).numpy()

    def test_tensor_times_variable(self):
        """Multiply DimTensor and DimVariable."""
        dt = DimTensor([2.0], units.m)
        dv = DimVariable([3.0], units.s)
        result = dt * dv
        assert result.dimension == (units.m * units.s).dimension

    def test_variable_matmul_tensor(self):
        """Matrix multiply DimVariable and DimTensor."""
        dv = DimVariable(tf.ones((2, 3)), units.m)
        dt = DimTensor(tf.ones((3, 4)), units.s)
        result = dv @ dt
        assert result.shape == (2, 4)
        assert result.dimension == (units.m * units.s).dimension

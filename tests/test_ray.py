"""Tests for Ray integration.

These tests skip if Ray is not installed.
"""

import pytest
import numpy as np

# Skip all tests if ray is not installed
ray = pytest.importorskip("ray")

from dimtensor import DimArray, units
from dimtensor.ray.serialization import (
    serialize_dimarray,
    deserialize_dimarray,
    is_serialized_dimarray,
    is_serialized_dimtensor,
    DIMARRAY_MARKER,
)


class TestSerialization:
    """Tests for DimArray/DimTensor serialization."""

    def test_serialize_dimarray_basic(self):
        """Serialize a basic DimArray."""
        arr = DimArray([1.0, 2.0, 3.0], units.m)
        result = serialize_dimarray(arr)

        assert result[DIMARRAY_MARKER] is True
        assert np.allclose(result["data"], [1.0, 2.0, 3.0])
        assert result["unit"]["symbol"] == "m"
        assert result["unit"]["scale"] == 1.0
        assert result["unit"]["dimension"]["length"] == 1.0
        assert result["unit"]["dimension"]["mass"] == 0.0

    def test_serialize_dimarray_with_uncertainty(self):
        """Serialize DimArray with uncertainty."""
        arr = DimArray([1.0, 2.0, 3.0], units.m, uncertainty=[0.1, 0.2, 0.3])
        result = serialize_dimarray(arr)

        assert "uncertainty" in result
        assert np.allclose(result["uncertainty"], [0.1, 0.2, 0.3])

    def test_serialize_dimarray_complex_unit(self):
        """Serialize DimArray with complex unit."""
        arr = DimArray([10.0, 20.0], units.m / units.s)
        result = serialize_dimarray(arr)

        assert result["unit"]["dimension"]["length"] == 1.0
        assert result["unit"]["dimension"]["time"] == -1.0

    def test_deserialize_dimarray_basic(self):
        """Deserialize a basic DimArray."""
        data = {
            DIMARRAY_MARKER: True,
            "data": np.array([1.0, 2.0, 3.0]),
            "unit": {
                "symbol": "m",
                "dimension": {
                    "length": 1.0,
                    "mass": 0.0,
                    "time": 0.0,
                    "current": 0.0,
                    "temperature": 0.0,
                    "amount": 0.0,
                    "luminosity": 0.0,
                },
                "scale": 1.0,
            },
        }

        arr = deserialize_dimarray(data)

        assert isinstance(arr, DimArray)
        assert np.allclose(arr._data, [1.0, 2.0, 3.0])
        assert arr.unit.symbol == "m"
        assert arr.dimension.length == 1

    def test_serialize_deserialize_roundtrip(self):
        """Test roundtrip serialization."""
        original = DimArray([1.0, 2.0, 3.0], units.kg)
        serialized = serialize_dimarray(original)
        recovered = deserialize_dimarray(serialized)

        assert np.allclose(recovered._data, original._data)
        assert recovered.unit.symbol == original.unit.symbol
        assert recovered.dimension == original.dimension

    def test_serialize_deserialize_roundtrip_with_uncertainty(self):
        """Test roundtrip with uncertainty."""
        original = DimArray([1.0, 2.0], units.s, uncertainty=[0.1, 0.2])
        serialized = serialize_dimarray(original)
        recovered = deserialize_dimarray(serialized)

        assert np.allclose(recovered._data, original._data)
        assert np.allclose(recovered._uncertainty, original._uncertainty)

    def test_is_serialized_dimarray(self):
        """Test marker detection."""
        valid = {DIMARRAY_MARKER: True, "data": [1.0]}
        invalid = {"data": [1.0]}
        non_dict = [1.0, 2.0]

        assert is_serialized_dimarray(valid)
        assert not is_serialized_dimarray(invalid)
        assert not is_serialized_dimarray(non_dict)

    def test_deserialize_invalid_raises(self):
        """Deserializing invalid data raises ValueError."""
        with pytest.raises(ValueError, match="Not a serialized DimArray"):
            deserialize_dimarray({"data": [1.0]})


class TestDimTensorSerialization:
    """Tests for DimTensor serialization."""

    @pytest.fixture
    def torch(self):
        """Skip if torch not installed."""
        return pytest.importorskip("torch")

    def test_serialize_dimtensor_basic(self, torch):
        """Serialize a basic DimTensor."""
        from dimtensor.torch import DimTensor
        from dimtensor.ray.serialization import serialize_dimtensor, DIMTENSOR_MARKER

        t = DimTensor(torch.tensor([1.0, 2.0, 3.0]), units.m)
        result = serialize_dimtensor(t)

        assert result[DIMTENSOR_MARKER] is True
        assert np.allclose(result["data"], [1.0, 2.0, 3.0])
        assert result["unit"]["symbol"] == "m"
        assert "dtype" in result

    def test_serialize_deserialize_dimtensor_roundtrip(self, torch):
        """Test DimTensor roundtrip."""
        from dimtensor.torch import DimTensor
        from dimtensor.ray.serialization import serialize_dimtensor, deserialize_dimtensor

        original = DimTensor(torch.tensor([1.0, 2.0, 3.0]), units.kg / units.s)
        serialized = serialize_dimtensor(original)
        recovered = deserialize_dimtensor(serialized)

        assert torch.allclose(recovered._data, original._data)
        assert recovered.unit.symbol == original.unit.symbol


class TestRemoteFunctions:
    """Tests for dim_remote decorator and dim_get."""

    @pytest.fixture(scope="class", autouse=True)
    def ray_init(self):
        """Initialize Ray for tests."""
        if not ray.is_initialized():
            ray.init(local_mode=True, ignore_reinit_error=True)
        yield
        # Don't shutdown - other tests may need ray

    def test_dim_remote_basic(self):
        """Basic dim_remote function."""
        from dimtensor.ray import dim_remote, dim_get

        @dim_remote
        def double_array(arr):
            return arr * 2

        arr = DimArray([1.0, 2.0, 3.0], units.m)
        ref = double_array.remote(arr)
        result = dim_get(ref)

        assert isinstance(result, DimArray)
        assert np.allclose(result._data, [2.0, 4.0, 6.0])
        assert result.unit == units.m

    def test_dim_remote_with_options(self):
        """dim_remote with num_cpus option."""
        from dimtensor.ray import dim_remote, dim_get

        @dim_remote(num_cpus=1)
        def triple_array(arr):
            return arr * 3

        arr = DimArray([1.0, 2.0], units.kg)
        ref = triple_array.remote(arr)
        result = dim_get(ref)

        assert np.allclose(result._data, [3.0, 6.0])
        assert result.unit == units.kg

    def test_dim_remote_multiple_args(self):
        """dim_remote with multiple DimArray arguments."""
        from dimtensor.ray import dim_remote, dim_get

        @dim_remote
        def compute_energy(mass, velocity):
            return 0.5 * mass * velocity**2

        mass = DimArray([1.0, 2.0], units.kg)
        velocity = DimArray([10.0, 20.0], units.m / units.s)

        ref = compute_energy.remote(mass, velocity)
        result = dim_get(ref)

        assert isinstance(result, DimArray)
        # E = 0.5 * m * v^2 -> [50, 400] J
        assert np.allclose(result._data, [50.0, 400.0])
        # Result should have energy dimension
        assert result.dimension == (units.kg * (units.m / units.s) ** 2).dimension

    def test_dim_remote_mixed_args(self):
        """dim_remote with mixed DimArray and scalar arguments."""
        from dimtensor.ray import dim_remote, dim_get

        @dim_remote
        def scale_array(arr, factor):
            return arr * factor

        arr = DimArray([1.0, 2.0], units.m)
        ref = scale_array.remote(arr, 3.0)
        result = dim_get(ref)

        assert np.allclose(result._data, [3.0, 6.0])

    def test_dim_remote_returns_tuple(self):
        """dim_remote returning tuple of DimArrays."""
        from dimtensor.ray import dim_remote, dim_get

        @dim_remote
        def split_and_process(arr):
            return arr[:2], arr[2:]

        arr = DimArray([1.0, 2.0, 3.0, 4.0], units.s)
        ref = split_and_process.remote(arr)
        result = dim_get(ref)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert np.allclose(result[0]._data, [1.0, 2.0])
        assert np.allclose(result[1]._data, [3.0, 4.0])

    def test_dim_get_multiple_refs(self):
        """dim_get with list of refs."""
        from dimtensor.ray import dim_remote, dim_get

        @dim_remote
        def double(arr):
            return arr * 2

        arr1 = DimArray([1.0], units.m)
        arr2 = DimArray([2.0], units.m)

        refs = [double.remote(arr1), double.remote(arr2)]
        results = dim_get(refs)

        assert len(results) == 2
        assert np.allclose(results[0]._data, [2.0])
        assert np.allclose(results[1]._data, [4.0])


class TestDimPut:
    """Tests for dim_put."""

    @pytest.fixture(scope="class", autouse=True)
    def ray_init(self):
        """Initialize Ray for tests."""
        if not ray.is_initialized():
            ray.init(local_mode=True, ignore_reinit_error=True)
        yield

    def test_dim_put_basic(self):
        """Basic dim_put usage."""
        from dimtensor.ray import dim_put, dim_get

        arr = DimArray([1.0, 2.0, 3.0], units.m)
        ref = dim_put(arr)
        result = dim_get(ref)

        assert isinstance(result, DimArray)
        assert np.allclose(result._data, arr._data)
        assert result.unit == arr.unit


class TestDimDataset:
    """Tests for DimDataset."""

    @pytest.fixture(scope="class", autouse=True)
    def ray_init(self):
        """Initialize Ray for tests."""
        if not ray.is_initialized():
            ray.init(local_mode=True, ignore_reinit_error=True)
        yield

    def test_from_dimarrays(self):
        """Create DimDataset from DimArrays."""
        from dimtensor.ray import DimDataset

        positions = DimArray(np.arange(10.0), units.m)
        velocities = DimArray(np.arange(10.0) * 2, units.m / units.s)

        ds = DimDataset.from_dimarrays({
            "position": positions,
            "velocity": velocities,
        })

        assert len(ds) == 10
        assert "position" in ds.columns
        assert "velocity" in ds.columns
        assert ds.units["position"] == units.m
        assert ds.units["velocity"] == units.m / units.s

    def test_to_dimarrays(self):
        """Convert DimDataset back to DimArrays."""
        from dimtensor.ray import DimDataset

        original = DimArray(np.array([1.0, 2.0, 3.0]), units.kg)

        ds = DimDataset.from_dimarrays({"mass": original})
        result = ds.to_dimarrays()

        assert "mass" in result
        assert np.allclose(result["mass"]._data, original._data)
        assert result["mass"].unit == units.kg

    def test_iter_batches(self):
        """Iterate over batches."""
        from dimtensor.ray import DimDataset

        positions = DimArray(np.arange(100.0), units.m)

        ds = DimDataset.from_dimarrays({"position": positions})

        batches = list(ds.iter_batches(batch_size=10))

        # Should have ~10 batches
        assert len(batches) >= 9  # Allow for rounding

        # Each batch should have DimArrays
        for batch in batches:
            assert "position" in batch
            assert isinstance(batch["position"], DimArray)
            assert batch["position"].unit == units.m

    @pytest.mark.skip(reason="Ray Data map_batches can hang in local mode")
    def test_map_batches(self):
        """Apply map_batches transformation."""
        from dimtensor.ray import DimDataset

        positions = DimArray(np.array([1.0, 2.0, 3.0]), units.m)

        ds = DimDataset.from_dimarrays({"position": positions})
        ds_doubled = ds.map_batches(lambda batch: {"position": batch["position"] * 2})

        result = ds_doubled.to_dimarrays()
        assert np.allclose(result["position"]._data, [2.0, 4.0, 6.0])
        # Unit should be preserved
        assert result["position"].unit == units.m

    @pytest.mark.skip(reason="Ray Data filter can hang in local mode")
    def test_filter(self):
        """Filter rows."""
        from dimtensor.ray import DimDataset

        values = DimArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), units.m)

        ds = DimDataset.from_dimarrays({"value": values})
        ds_filtered = ds.filter(lambda row: row["value"] > 2.5)

        result = ds_filtered.to_dimarrays()
        # Should have values > 2.5: 3, 4, 5
        assert len(result["value"]) == 3
        assert result["value"].unit == units.m

    def test_units_property(self):
        """Test units property returns copy."""
        from dimtensor.ray import DimDataset

        ds = DimDataset.from_dimarrays({
            "x": DimArray([1.0], units.m),
            "t": DimArray([1.0], units.s),
        })

        units_map = ds.units
        assert units_map["x"] == units.m
        assert units_map["t"] == units.s

        # Modifying the copy shouldn't affect original
        units_map["x"] = units.kg
        assert ds.units["x"] == units.m


class TestDimDatasetSplit:
    """Tests for DimDataset split operations."""

    @pytest.fixture(scope="class", autouse=True)
    def ray_init(self):
        """Initialize Ray for tests."""
        if not ray.is_initialized():
            ray.init(local_mode=True, ignore_reinit_error=True)
        yield

    @pytest.mark.skip(reason="Ray Data shuffle can hang in local mode")
    def test_random_shuffle(self):
        """Random shuffle preserves units."""
        from dimtensor.ray import DimDataset

        values = DimArray(np.arange(10.0), units.m)
        ds = DimDataset.from_dimarrays({"value": values})

        shuffled = ds.random_shuffle(seed=42)

        assert len(shuffled) == 10
        assert shuffled.units["value"] == units.m

    @pytest.mark.skip(reason="Ray Data split can hang in local mode")
    def test_split(self):
        """Split dataset."""
        from dimtensor.ray import DimDataset

        values = DimArray(np.arange(10.0), units.kg)
        ds = DimDataset.from_dimarrays({"value": values})

        splits = ds.split(2)

        assert len(splits) == 2
        assert all(s.units["value"] == units.kg for s in splits)


class TestRegisterSerializers:
    """Tests for register_serializers."""

    @pytest.fixture(scope="class", autouse=True)
    def ray_init(self):
        """Initialize Ray for tests."""
        if not ray.is_initialized():
            ray.init(local_mode=True, ignore_reinit_error=True)
        yield

    def test_register_and_use(self):
        """Register serializers and use with standard ray.remote."""
        from dimtensor.ray import register_serializers

        register_serializers()

        @ray.remote
        def process_registered(arr):
            # This should work with registered serializers
            return arr

        arr = DimArray([1.0, 2.0], units.m)
        ref = process_registered.remote(arr)
        result = ray.get(ref)

        # With registered serializers, should round-trip
        assert isinstance(result, DimArray)
        assert np.allclose(result._data, arr._data)


class TestDimTrainContext:
    """Tests for DimTrainContext."""

    @pytest.fixture
    def torch(self):
        """Skip if torch not installed."""
        return pytest.importorskip("torch")

    def test_context_creation(self, torch):
        """Create training context."""
        from dimtensor.ray.train import DimTrainContext
        from dimtensor.torch import DimScaler

        scaler = DimScaler(method="characteristic")
        ctx = DimTrainContext(
            scaler=scaler,
            input_units={"position": units.m},
            output_units={"energy": units.J},
        )

        assert ctx.scaler is scaler
        assert ctx.input_units["position"] == units.m
        assert ctx.output_units["energy"] == units.J

    def test_inverse_transform(self, torch):
        """Test inverse transform."""
        from dimtensor.ray.train import DimTrainContext
        from dimtensor.torch import DimScaler, DimTensor

        # Create and fit scaler
        scaler = DimScaler(method="characteristic")
        energies = DimArray([100.0, 200.0, 300.0], units.J)
        scaler.fit(energies)

        ctx = DimTrainContext(
            scaler=scaler,
            output_units={"energy": units.J},
        )

        # Simulate model output (scaled)
        scaled_output = torch.tensor([0.5])

        result = ctx.inverse_transform(scaled_output, "energy")

        assert isinstance(result, DimTensor)
        assert result.unit == units.J
        # Value should be unscaled
        assert result._data.item() == pytest.approx(150.0, rel=0.01)

    def test_inverse_transform_unknown_raises(self, torch):
        """Inverse transform raises for unknown output."""
        from dimtensor.ray.train import DimTrainContext

        ctx = DimTrainContext(output_units={"energy": units.J})

        with pytest.raises(KeyError, match="Unknown output"):
            ctx.inverse_transform(torch.tensor([1.0]), "unknown")


class TestIntegration:
    """Integration tests for full Ray workflows."""

    @pytest.fixture(scope="class", autouse=True)
    def ray_init(self):
        """Initialize Ray for tests."""
        if not ray.is_initialized():
            ray.init(local_mode=True, ignore_reinit_error=True)
        yield

    def test_distributed_physics_computation(self):
        """Full workflow: create data, distribute, compute, collect."""
        from dimtensor.ray import dim_remote, dim_get, DimDataset

        # Create physics data
        masses = DimArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), units.kg)
        velocities = DimArray(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), units.m / units.s)

        @dim_remote
        def compute_momentum(m, v):
            return m * v

        @dim_remote
        def compute_kinetic_energy(m, v):
            return 0.5 * m * v**2

        # Compute in parallel
        momentum_ref = compute_momentum.remote(masses, velocities)
        energy_ref = compute_kinetic_energy.remote(masses, velocities)

        momentum = dim_get(momentum_ref)
        energy = dim_get(energy_ref)

        # Verify units
        assert momentum.dimension == (units.kg * units.m / units.s).dimension
        assert energy.dimension == (units.kg * (units.m / units.s) ** 2).dimension

        # Verify values
        expected_momentum = np.array([10.0, 40.0, 90.0, 160.0, 250.0])
        expected_energy = np.array([50.0, 400.0, 1350.0, 3200.0, 6250.0])

        assert np.allclose(momentum._data, expected_momentum)
        assert np.allclose(energy._data, expected_energy)

    @pytest.mark.skip(reason="Ray Data filter can hang in local mode")
    def test_dataset_pipeline(self):
        """Data pipeline with transformations."""
        from dimtensor.ray import DimDataset

        # Simulate particle data
        n_particles = 100
        positions = DimArray(np.random.randn(n_particles, 3), units.m)
        velocities = DimArray(np.random.randn(n_particles, 3), units.m / units.s)

        ds = DimDataset.from_dimarrays({
            "position": positions,
            "velocity": velocities,
        })

        # Filter particles with positive x-position
        # (first column of position array)
        ds_filtered = ds.filter(lambda row: row["position"][0] > 0)

        # Should have roughly half the particles
        assert len(ds_filtered) < n_particles
        assert len(ds_filtered) > 0

        # Units should be preserved
        assert ds_filtered.units["position"] == units.m
        assert ds_filtered.units["velocity"] == units.m / units.s

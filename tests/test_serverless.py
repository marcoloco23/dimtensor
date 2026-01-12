"""Tests for serverless module."""

import json
import base64
import pytest
import numpy as np
from dimtensor import DimArray, units
from dimtensor.errors import DimensionError, UnitConversionError


class TestSerialization:
    """Test HTTP serialization/deserialization."""

    def test_serialize_for_http(self):
        """Test DimArray serialization for HTTP."""
        from dimtensor.serverless import serialize_for_http

        arr = DimArray([1.0, 2.0, 3.0], units.m / units.s)
        result = serialize_for_http(arr)

        assert "data" in result
        assert "unit" in result
        assert "dimension" in result
        assert "shape" in result
        assert "dtype" in result
        assert "scale" in result

        # Check data is base64
        assert isinstance(result["data"], str)
        decoded = base64.b64decode(result["data"])
        assert len(decoded) > 0

        # Check dimension
        assert result["dimension"] == [1, 0, -1, 0, 0, 0, 0]
        assert result["shape"] == [3]

    def test_serialize_with_uncertainty(self):
        """Test serialization with uncertainty."""
        from dimtensor.serverless import serialize_for_http

        arr = DimArray([10.0], units.m, uncertainty=[0.1])
        result = serialize_for_http(arr)

        assert result["uncertainty"] is not None
        decoded = base64.b64decode(result["uncertainty"])
        assert len(decoded) > 0

    def test_deserialize_from_http(self):
        """Test DimArray deserialization from HTTP."""
        from dimtensor.serverless import serialize_for_http, deserialize_from_http

        original = DimArray([1.0, 2.0, 3.0], units.m / units.s)
        serialized = serialize_for_http(original)
        restored = deserialize_from_http(serialized)

        np.testing.assert_array_equal(restored._data, original._data)
        assert restored.unit.symbol == original.unit.symbol
        assert restored.dimension == original.dimension

    def test_round_trip_with_uncertainty(self):
        """Test round-trip with uncertainty."""
        from dimtensor.serverless import serialize_for_http, deserialize_from_http

        original = DimArray([10.0, 20.0], units.J, uncertainty=[0.1, 0.2])
        serialized = serialize_for_http(original)
        restored = deserialize_from_http(serialized)

        np.testing.assert_array_equal(restored._data, original._data)
        assert restored._uncertainty is not None
        np.testing.assert_array_equal(restored._uncertainty, original._uncertainty)

    def test_simple_response(self):
        """Test simple response formatting."""
        from dimtensor.serverless import simple_response

        arr = DimArray([42.0], units.J)
        response = simple_response(arr)

        assert response["statusCode"] == 200
        assert "body" in response
        assert "headers" in response
        assert response["headers"]["Content-Type"] == "application/json"

        # Body should be JSON string
        body = json.loads(response["body"])
        assert "data" in body

    def test_error_response(self):
        """Test error response formatting."""
        from dimtensor.serverless import error_response

        response = error_response("TestError", "Test message", status_code=400)

        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert body["error"] == "TestError"
        assert body["message"] == "Test message"


class TestAWSHandlers:
    """Test AWS Lambda handler decorators."""

    def test_lambda_handler_basic(self):
        """Test basic Lambda handler with DimArray."""
        from dimtensor.serverless import lambda_handler

        @lambda_handler()
        def handler(event, context):
            mass = event["mass"]
            return mass * 2

        # Create mock event
        from dimtensor.serverless import serialize_for_http

        mass = DimArray([5.0], units.kg)
        event = {"body": json.dumps({"mass": serialize_for_http(mass)})}

        # Mock context
        class MockContext:
            memory_limit_in_mb = 512

        response = handler(event, MockContext())

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert "data" in body

    def test_lambda_handler_dimension_error(self):
        """Test Lambda handler catches DimensionError."""
        from dimtensor.serverless import lambda_handler

        @lambda_handler()
        def handler(event, context):
            mass = event["mass"]
            length = event["length"]
            # This should raise DimensionError
            return mass + length

        from dimtensor.serverless import serialize_for_http

        event = {
            "body": json.dumps(
                {
                    "mass": serialize_for_http(DimArray([1.0], units.kg)),
                    "length": serialize_for_http(DimArray([2.0], units.m)),
                }
            )
        }

        response = handler(event, None)

        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert body["error"] == "DimensionError"

    def test_lambda_handler_dict_result(self):
        """Test Lambda handler with dictionary result."""
        from dimtensor.serverless import lambda_handler

        @lambda_handler()
        def handler(event, context):
            mass = event["mass"]
            return {"result": mass * 2, "status": "success"}

        from dimtensor.serverless import serialize_for_http

        event = {
            "body": json.dumps({"mass": serialize_for_http(DimArray([5.0], units.kg))})
        }

        response = handler(event, None)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert "result" in body
        assert body["status"] == "success"

    def test_lambda_physics_decorator(self):
        """Test lambda_physics convenience decorator."""
        from dimtensor.serverless import lambda_physics

        @lambda_physics
        def handler(event, context):
            return DimArray([1.0], units.J)

        response = handler({}, None)
        assert response["statusCode"] == 200

    def test_lambda_ml_decorator(self):
        """Test lambda_ml convenience decorator."""
        from dimtensor.serverless import lambda_ml

        @lambda_ml
        def handler(event, context):
            return DimArray([1.0], units.J)

        response = handler({}, None)
        assert response["statusCode"] == 200


class TestGCPHandlers:
    """Test Google Cloud Functions handler decorators."""

    def test_cloud_function_basic(self):
        """Test basic Cloud Function handler."""
        from dimtensor.serverless import cloud_function

        @cloud_function()
        def handler(request):
            data = request.get_json()
            mass = data["mass"]
            return mass * 2

        # Mock request
        from dimtensor.serverless import serialize_for_http

        class MockRequest:
            def __init__(self, data):
                self._data = data

            def get_json(self, silent=False):
                return self._data

            @property
            def method(self):
                return "POST"

        mass = DimArray([5.0], units.kg)
        request = MockRequest({"mass": serialize_for_http(mass)})

        result = handler(request)

        # Result should be (body, status, headers) tuple
        assert isinstance(result, tuple)
        assert len(result) >= 2
        assert result[1] == 200

    def test_cloud_function_dimension_error(self):
        """Test Cloud Function catches DimensionError."""
        from dimtensor.serverless import cloud_function

        @cloud_function()
        def handler(request):
            data = request.get_json()
            return data["mass"] + data["length"]

        from dimtensor.serverless import serialize_for_http

        class MockRequest:
            def get_json(self, silent=False):
                return {
                    "mass": serialize_for_http(DimArray([1.0], units.kg)),
                    "length": serialize_for_http(DimArray([2.0], units.m)),
                }

            @property
            def method(self):
                return "POST"

        result = handler(MockRequest())

        assert isinstance(result, tuple)
        assert result[1] == 400
        body = json.loads(result[0])
        assert body["error"] == "DimensionError"


class TestValidation:
    """Test validation decorators."""

    def test_require_positive(self):
        """Test require_positive decorator."""
        from dimtensor.serverless import require_positive

        @require_positive("mass")
        def handler(event, context):
            return event["mass"]

        # Valid case
        event = {"mass": DimArray([5.0], units.kg)}
        result = handler(event, None)
        assert result._data[0] == 5.0

        # Invalid case
        with pytest.raises(ValueError, match="must be positive"):
            event = {"mass": DimArray([-5.0], units.kg)}
            handler(event, None)

    def test_require_dimensionless(self):
        """Test require_dimensionless decorator."""
        from dimtensor.serverless import require_dimensionless

        @require_dimensionless("angle")
        def handler(event, context):
            return event["angle"]

        # Valid case
        from dimtensor.core.units import dimensionless

        event = {"angle": DimArray([1.57], dimensionless)}
        result = handler(event, None)
        assert result.dimension == dimensionless.dimension

        # Invalid case
        with pytest.raises(DimensionError, match="must be dimensionless"):
            event = {"angle": DimArray([1.0], units.m)}
            handler(event, None)

    def test_require_params(self):
        """Test require_params decorator."""
        from dimtensor.serverless import require_params

        @require_params("mass", "velocity")
        def handler(event, context):
            return event["mass"] * event["velocity"]

        # Valid case
        event = {
            "mass": DimArray([1.0], units.kg),
            "velocity": DimArray([10.0], units.m / units.s),
        }
        result = handler(event, None)

        # Invalid case
        with pytest.raises(ValueError, match="Missing required parameters"):
            handler({"mass": DimArray([1.0], units.kg)}, None)

    def test_validate_shape(self):
        """Test validate_shape decorator."""
        from dimtensor.serverless import validate_shape

        @validate_shape("position", (3,))
        def handler(event, context):
            return event["position"]

        # Valid case
        event = {"position": DimArray([1.0, 2.0, 3.0], units.m)}
        result = handler(event, None)

        # Invalid case
        with pytest.raises(ValueError, match="has shape"):
            event = {"position": DimArray([1.0, 2.0], units.m)}
            handler(event, None)


class TestIntegration:
    """Integration tests with example functions."""

    def test_kinetic_energy_calculation(self):
        """Test kinetic energy calculation through handler."""
        from dimtensor.serverless import lambda_physics, serialize_for_http

        @lambda_physics
        def handler(event, context):
            mass = event["mass"]
            velocity = event["velocity"]
            return 0.5 * mass * velocity**2

        event = {
            "body": json.dumps(
                {
                    "mass": serialize_for_http(DimArray([2.0], units.kg)),
                    "velocity": serialize_for_http(DimArray([10.0], units.m / units.s)),
                }
            )
        }

        response = handler(event, None)
        assert response["statusCode"] == 200

        # Check result
        body = json.loads(response["body"])
        from dimtensor.serverless import deserialize_from_http

        result = deserialize_from_http(body)

        # KE = 0.5 * 2 * 10^2 = 100 J
        np.testing.assert_allclose(result._data, [100.0])
        assert result.unit.dimension == units.J.dimension

    def test_momentum_calculation(self):
        """Test momentum calculation."""
        from dimtensor.serverless import lambda_handler, serialize_for_http

        @lambda_handler()
        def handler(event, context):
            mass = event["mass"]
            velocity = event["velocity"]
            return mass * velocity

        event = {
            "body": json.dumps(
                {
                    "mass": serialize_for_http(DimArray([5.0], units.kg)),
                    "velocity": serialize_for_http(DimArray([3.0], units.m / units.s)),
                }
            )
        }

        response = handler(event, None)
        assert response["statusCode"] == 200

        body = json.loads(response["body"])
        from dimtensor.serverless import deserialize_from_http

        result = deserialize_from_http(body)

        # p = m * v = 5 * 3 = 15 kg·m/s
        np.testing.assert_allclose(result._data, [15.0])


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_event(self):
        """Test handler with empty event."""
        from dimtensor.serverless import lambda_handler

        @lambda_handler()
        def handler(event, context):
            return {"status": "ok"}

        response = handler({}, None)
        assert response["statusCode"] == 200

    def test_invalid_json(self):
        """Test handler with invalid JSON."""
        from dimtensor.serverless import lambda_handler

        @lambda_handler()
        def handler(event, context):
            return {"status": "ok"}

        event = {"body": "invalid json {"}
        response = handler(event, None)

        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert body["error"] == "JSONDecodeError"

    def test_scalar_dimarray(self):
        """Test serialization of scalar DimArray."""
        from dimtensor.serverless import serialize_for_http, deserialize_from_http

        scalar = DimArray(42.0, units.J)
        serialized = serialize_for_http(scalar)
        restored = deserialize_from_http(serialized)

        np.testing.assert_allclose(restored._data, scalar._data)
        assert restored.shape == scalar.shape

    def test_multidimensional_array(self):
        """Test serialization of multidimensional array."""
        from dimtensor.serverless import serialize_for_http, deserialize_from_http

        arr = DimArray(np.random.rand(3, 4, 5), units.m)
        serialized = serialize_for_http(arr)
        restored = deserialize_from_http(serialized)

        np.testing.assert_array_equal(restored._data, arr._data)
        assert restored.shape == (3, 4, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

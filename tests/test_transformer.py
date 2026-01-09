"""Tests for dimension-aware transformer attention mechanisms.

Tests multi-head attention, encoder layers, and full encoder stacks with
physical dimension tracking and autograd support.
"""

import pytest
import torch

from dimtensor import Dimension, units
from dimtensor.errors import DimensionError
from dimtensor.torch import (
    DimMultiheadAttention,
    DimTensor,
    DimTransformerEncoder,
    DimTransformerEncoderLayer,
)


class TestDimMultiheadAttention:
    """Tests for DimMultiheadAttention layer."""

    def test_basic_forward_pass(self):
        """Test basic attention forward pass with units."""
        # Attention over particle positions
        attn = DimMultiheadAttention(
            embed_dim=64,
            num_heads=4,
            input_dim=Dimension(length=1),  # meters
            output_dim=Dimension(length=1),  # meters
        )

        # Input: (batch, seq_len, embed_dim)
        x = DimTensor(torch.randn(8, 20, 64), units.m)
        output = attn(x)

        # Check shape and dimension
        assert output.shape == (8, 20, 64)
        assert output.dimension == Dimension(length=1)
        assert not torch.isnan(output.data).any()

    def test_single_head(self):
        """Test attention with single head."""
        attn = DimMultiheadAttention(
            embed_dim=32,
            num_heads=1,
            input_dim=Dimension(length=1),
        )

        x = DimTensor(torch.randn(4, 10, 32), units.m)
        output = attn(x)

        assert output.shape == (4, 10, 32)
        assert output.dimension == Dimension(length=1)

    def test_multihead_attention(self):
        """Test multi-head attention with multiple heads."""
        attn = DimMultiheadAttention(
            embed_dim=512,
            num_heads=8,
            input_dim=Dimension(length=1, time=-1),  # velocity
        )

        x = DimTensor(torch.randn(10, 100, 512), units.m / units.s)
        output = attn(x)

        # Verify shape and dimension
        assert output.shape == (10, 100, 512)
        assert output.dimension == Dimension(length=1, time=-1)

    def test_dimension_transformation(self):
        """Test attention with different input/output dimensions."""
        # Input: position [m], Output: velocity [m/s]
        attn = DimMultiheadAttention(
            embed_dim=64,
            num_heads=4,
            input_dim=Dimension(length=1),
            output_dim=Dimension(length=1, time=-1),
        )

        x = DimTensor(torch.randn(8, 20, 64), units.m)
        output = attn(x)

        # Output should have velocity dimension
        assert output.dimension == Dimension(length=1, time=-1)
        assert output.shape == (8, 20, 64)

    def test_dimensionless_input(self):
        """Test attention with dimensionless input."""
        attn = DimMultiheadAttention(
            embed_dim=32,
            num_heads=2,
        )

        x = DimTensor(torch.randn(4, 10, 32), units.dimensionless)
        output = attn(x)

        assert output.is_dimensionless
        assert output.shape == (4, 10, 32)

    def test_invalid_embed_dim(self):
        """Test that embed_dim must be divisible by num_heads."""
        with pytest.raises(ValueError, match="must be divisible"):
            DimMultiheadAttention(
                embed_dim=65,  # Not divisible by 4
                num_heads=4,
            )

    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        attn = DimMultiheadAttention(
            embed_dim=32,
            num_heads=2,
            input_dim=Dimension(length=1),  # Expects meters
        )

        # Provide velocity instead
        x = DimTensor(torch.randn(4, 10, 32), units.m / units.s)

        with pytest.raises(DimensionError):
            attn(x)

    def test_dropout(self):
        """Test attention with dropout enabled."""
        attn = DimMultiheadAttention(
            embed_dim=32,
            num_heads=2,
            input_dim=Dimension(length=1),
            dropout=0.5,
        )

        # Set to eval mode (dropout off)
        attn.eval()
        x = DimTensor(torch.randn(4, 10, 32), units.m)
        output_eval = attn(x)

        # Set to train mode (dropout on)
        attn.train()
        output_train = attn(x)

        # Both should have correct shape and dimension
        assert output_eval.shape == (4, 10, 32)
        assert output_train.shape == (4, 10, 32)
        assert output_eval.dimension == Dimension(length=1)
        assert output_train.dimension == Dimension(length=1)

    def test_autograd(self):
        """Test gradients flow through attention."""
        attn = DimMultiheadAttention(
            embed_dim=32,
            num_heads=2,
            input_dim=Dimension(length=1),
        )

        x = DimTensor(torch.randn(4, 10, 32), units.m)
        x.requires_grad_(True)

        output = attn(x)
        loss = output.data.sum()
        loss.backward()

        # Check gradients exist for projection weights
        assert attn.q_proj.linear.weight.grad is not None
        assert attn.k_proj.linear.weight.grad is not None
        assert attn.v_proj.linear.weight.grad is not None
        assert attn.out_proj.linear.weight.grad is not None

        # Check input gradients
        assert x.grad is not None

    def test_raw_tensor_input(self):
        """Test attention accepts raw Tensor input."""
        attn = DimMultiheadAttention(
            embed_dim=32,
            num_heads=2,
            input_dim=Dimension(length=1),
            validate_input=False,
        )

        # Pass raw tensor (no unit validation)
        x = torch.randn(4, 10, 32)
        output = attn(x)

        # Should return DimTensor with output dimension
        assert isinstance(output, DimTensor)
        assert output.dimension == Dimension(length=1)
        assert output.shape == (4, 10, 32)

    def test_very_small_values(self):
        """Test attention handles very small values (scale computation stability)."""
        attn = DimMultiheadAttention(
            embed_dim=32,
            num_heads=2,
            input_dim=Dimension(length=1),
        )

        # Very small input values (near zero)
        x = DimTensor(torch.randn(4, 10, 32) * 1e-8, units.m)
        output = attn(x)

        # Should not produce NaN or Inf
        assert not torch.isnan(output.data).any()
        assert not torch.isinf(output.data).any()
        assert output.dimension == Dimension(length=1)


class TestDimTransformerEncoderLayer:
    """Tests for DimTransformerEncoderLayer."""

    def test_basic_encoder_layer(self):
        """Test basic encoder layer forward pass."""
        layer = DimTransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
            dimension=Dimension(length=1),
        )

        x = DimTensor(torch.randn(8, 20, 64), units.m)
        output = layer(x)

        # Output should have same shape and dimension (residual connections)
        assert output.shape == (8, 20, 64)
        assert output.dimension == Dimension(length=1)

    def test_residual_connections(self):
        """Test that residual connections preserve dimension."""
        layer = DimTransformerEncoderLayer(
            d_model=32,
            nhead=2,
            dimension=Dimension(mass=1, length=2, time=-2),  # Energy
        )

        x = DimTensor(torch.randn(4, 10, 32), units.J)
        output = layer(x)

        # Dimension must be preserved (residual connections)
        assert output.dimension == Dimension(mass=1, length=2, time=-2)
        assert output.shape == (4, 10, 32)

    def test_different_activations(self):
        """Test encoder layer with different activation functions."""
        # ReLU activation
        layer_relu = DimTransformerEncoderLayer(
            d_model=32,
            nhead=2,
            dimension=Dimension(length=1),
            activation="relu",
        )

        # GELU activation
        layer_gelu = DimTransformerEncoderLayer(
            d_model=32,
            nhead=2,
            dimension=Dimension(length=1),
            activation="gelu",
        )

        x = DimTensor(torch.randn(4, 10, 32), units.m)

        output_relu = layer_relu(x)
        output_gelu = layer_gelu(x)

        assert output_relu.shape == (4, 10, 32)
        assert output_gelu.shape == (4, 10, 32)
        assert output_relu.dimension == Dimension(length=1)
        assert output_gelu.dimension == Dimension(length=1)

    def test_invalid_activation(self):
        """Test that invalid activation raises error."""
        with pytest.raises(ValueError, match="Unsupported activation"):
            DimTransformerEncoderLayer(
                d_model=32,
                nhead=2,
                activation="tanh",  # Not supported
            )

    def test_dimension_mismatch(self):
        """Test encoder layer rejects wrong dimension input."""
        layer = DimTransformerEncoderLayer(
            d_model=32,
            nhead=2,
            dimension=Dimension(length=1),  # Expects position
        )

        # Provide velocity instead
        x = DimTensor(torch.randn(4, 10, 32), units.m / units.s)

        with pytest.raises(DimensionError):
            layer(x)

    def test_autograd_through_encoder_layer(self):
        """Test gradients flow through encoder layer."""
        layer = DimTransformerEncoderLayer(
            d_model=32,
            nhead=2,
            dimension=Dimension(length=1),
        )

        x = DimTensor(torch.randn(4, 10, 32), units.m)
        x.requires_grad_(True)

        output = layer(x)
        loss = output.data.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert layer.self_attn.q_proj.linear.weight.grad is not None
        assert layer.linear1.linear.weight.grad is not None
        assert layer.linear2.linear.weight.grad is not None

    def test_raw_tensor_input(self):
        """Test encoder layer accepts raw Tensor input."""
        layer = DimTransformerEncoderLayer(
            d_model=32,
            nhead=2,
            dimension=Dimension(length=1),
        )

        # Pass raw tensor
        x = torch.randn(4, 10, 32)
        output = layer(x)

        # Should return DimTensor
        assert isinstance(output, DimTensor)
        assert output.dimension == Dimension(length=1)


class TestDimTransformerEncoder:
    """Tests for DimTransformerEncoder (stack of layers)."""

    def test_basic_encoder_stack(self):
        """Test full encoder stack."""
        layer = DimTransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dimension=Dimension(length=1),
        )
        encoder = DimTransformerEncoder(layer, num_layers=6)

        x = DimTensor(torch.randn(8, 20, 64), units.m)
        output = encoder(x)

        # Output should have same shape and dimension
        assert output.shape == (8, 20, 64)
        assert output.dimension == Dimension(length=1)

    def test_single_layer(self):
        """Test encoder with single layer."""
        layer = DimTransformerEncoderLayer(
            d_model=32,
            nhead=2,
            dimension=Dimension(temperature=1),  # Temperature
        )
        encoder = DimTransformerEncoder(layer, num_layers=1)

        x = DimTensor(torch.randn(4, 10, 32), units.K)
        output = encoder(x)

        assert output.shape == (4, 10, 32)
        assert output.dimension == Dimension(temperature=1)

    def test_deep_encoder(self):
        """Test deep encoder with many layers."""
        layer = DimTransformerEncoderLayer(
            d_model=32,
            nhead=2,
            dimension=Dimension(length=1),
        )
        encoder = DimTransformerEncoder(layer, num_layers=12)

        x = DimTensor(torch.randn(4, 10, 32), units.m)
        output = encoder(x)

        assert output.shape == (4, 10, 32)
        assert output.dimension == Dimension(length=1)
        # Check no gradient explosion/vanishing
        assert not torch.isnan(output.data).any()

    def test_encoder_with_norm(self):
        """Test encoder with final normalization layer."""
        from dimtensor.torch import DimLayerNorm

        layer = DimTransformerEncoderLayer(
            d_model=32,
            nhead=2,
            dimension=Dimension(length=1),
        )
        norm = DimLayerNorm(32, dimension=Dimension(length=1))
        encoder = DimTransformerEncoder(layer, num_layers=4, norm=norm)

        x = DimTensor(torch.randn(4, 10, 32), units.m)
        output = encoder(x)

        assert output.shape == (4, 10, 32)
        assert output.dimension == Dimension(length=1)

    def test_autograd_through_encoder(self):
        """Test gradients flow through full encoder stack."""
        layer = DimTransformerEncoderLayer(
            d_model=32,
            nhead=2,
            dimension=Dimension(length=1),
        )
        encoder = DimTransformerEncoder(layer, num_layers=3)

        x = DimTensor(torch.randn(4, 10, 32), units.m)
        x.requires_grad_(True)

        output = encoder(x)
        loss = output.data.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        for enc_layer in encoder.layers:
            assert enc_layer.self_attn.q_proj.linear.weight.grad is not None

    def test_raw_tensor_input(self):
        """Test encoder accepts raw Tensor input."""
        layer = DimTransformerEncoderLayer(
            d_model=32,
            nhead=2,
            dimension=Dimension(length=1),
        )
        encoder = DimTransformerEncoder(layer, num_layers=2)

        # Pass raw tensor
        x = torch.randn(4, 10, 32)
        output = encoder(x)

        assert isinstance(output, DimTensor)
        assert output.dimension == Dimension(length=1)


class TestPhysicsApplications:
    """Integration tests for physics applications."""

    def test_particle_nbody_simulation(self):
        """Test transformer for N-body particle interactions."""
        # Particle positions in 3D space
        layer = DimTransformerEncoderLayer(
            d_model=3,
            nhead=1,
            dim_feedforward=12,
            dimension=Dimension(length=1),  # meters
        )
        encoder = DimTransformerEncoder(layer, num_layers=2)

        # 32 batches, 50 particles, 3D positions
        positions = DimTensor(torch.randn(32, 50, 3), units.m)
        updated_positions = encoder(positions)

        # Output should be positions (same dimension)
        assert updated_positions.shape == (32, 50, 3)
        assert updated_positions.dimension == Dimension(length=1)

    def test_time_series_forecasting(self):
        """Test transformer for temperature time series."""
        # Temperature measurements over time
        layer = DimTransformerEncoderLayer(
            d_model=16,
            nhead=4,
            dim_feedforward=64,
            dimension=Dimension(temperature=1),  # temperature
        )
        encoder = DimTransformerEncoder(layer, num_layers=4)

        # 8 series, 100 timesteps, 16 features
        temps = DimTensor(torch.randn(8, 100, 16), units.K)
        output = encoder(temps)

        # Output should have temperature dimension
        assert output.shape == (8, 100, 16)
        assert output.dimension == Dimension(temperature=1)

        # Can extract prediction for last timestep
        last_prediction = output[:, -1:, :]
        assert last_prediction.dimension == Dimension(temperature=1)

    def test_velocity_field_dynamics(self):
        """Test transformer for fluid velocity fields."""
        # Velocity field over spatial grid
        layer = DimTransformerEncoderLayer(
            d_model=2,
            nhead=1,
            dim_feedforward=8,
            dimension=Dimension(length=1, time=-1),  # velocity
        )
        encoder = DimTransformerEncoder(layer, num_layers=3)

        # Flattened 2D velocity field (batch, grid_points, 2D_velocity)
        velocity_field = DimTensor(torch.randn(16, 64, 2), units.m / units.s)
        updated_field = encoder(velocity_field)

        assert updated_field.shape == (16, 64, 2)
        assert updated_field.dimension == Dimension(length=1, time=-1)

    def test_mixed_physics_quantities(self):
        """Test attention can handle different physical quantities via dimension transformation."""
        # Attention that converts position to momentum
        attn = DimMultiheadAttention(
            embed_dim=32,
            num_heads=4,
            input_dim=Dimension(length=1),  # position
            output_dim=Dimension(mass=1, length=1, time=-1),  # momentum
        )

        positions = DimTensor(torch.randn(8, 20, 32), units.m)
        momenta = attn(positions)

        # Output should be momentum
        assert momenta.dimension == Dimension(mass=1, length=1, time=-1)
        assert momenta.shape == (8, 20, 32)


class TestEdgeCases:
    """Test edge cases and potential failure modes."""

    def test_very_long_sequence(self):
        """Test attention with long sequences."""
        attn = DimMultiheadAttention(
            embed_dim=32,
            num_heads=2,
            input_dim=Dimension(length=1),
        )

        # Long sequence (memory usage O(seq_len²))
        x = DimTensor(torch.randn(2, 500, 32), units.m)
        output = attn(x)

        assert output.shape == (2, 500, 32)
        assert not torch.isnan(output.data).any()

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        layer = DimTransformerEncoderLayer(
            d_model=32,
            nhead=2,
            dimension=Dimension(length=1),
        )

        x = DimTensor(torch.randn(1, 10, 32), units.m)
        output = layer(x)

        assert output.shape == (1, 10, 32)

    def test_sequence_length_one(self):
        """Test with sequence length of 1."""
        attn = DimMultiheadAttention(
            embed_dim=32,
            num_heads=2,
            input_dim=Dimension(length=1),
        )

        x = DimTensor(torch.randn(4, 1, 32), units.m)
        output = attn(x)

        assert output.shape == (4, 1, 32)

    def test_zero_input(self):
        """Test attention with zero input."""
        attn = DimMultiheadAttention(
            embed_dim=32,
            num_heads=2,
            input_dim=Dimension(length=1),
        )

        x = DimTensor(torch.zeros(4, 10, 32), units.m)
        output = attn(x)

        # Should handle gracefully (not NaN)
        assert not torch.isnan(output.data).any()

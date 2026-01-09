"""Tests for DimGraphConv layer."""

import pytest
import torch

from dimtensor import Dimension, units
from dimtensor.errors import DimensionError
from dimtensor.torch import DimGraphConv, DimSequential, DimTensor, DimLinear


class TestDimGraphConvBasic:
    """Test basic functionality of DimGraphConv."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        layer = DimGraphConv(
            in_features=3,
            out_features=4,
            input_dim=Dimension(length=1),
            output_dim=Dimension(length=1, time=-1),
        )

        # 5 nodes with 3 features each
        x = DimTensor(torch.randn(5, 3), units.m)

        # Simple chain: 0->1, 1->2, 2->3, 3->4
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])

        output = layer(x, edge_index)

        assert isinstance(output, DimTensor)
        assert output.shape == (5, 4)
        assert output.dimension == Dimension(length=1, time=-1)

    def test_ring_topology(self):
        """Test with ring topology (cyclic graph)."""
        layer = DimGraphConv(
            in_features=2, out_features=2, input_dim=Dimension(length=1), output_dim=Dimension(length=1)
        )

        # 4 nodes in a ring
        x = DimTensor(torch.randn(4, 2), units.m)

        # Ring: 0->1, 1->2, 2->3, 3->0
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])

        output = layer(x, edge_index)

        assert output.shape == (4, 2)
        assert output.dimension == Dimension(length=1)

    def test_with_raw_tensor_input(self):
        """Test that raw tensor input works (no validation)."""
        layer = DimGraphConv(
            in_features=3,
            out_features=4,
            input_dim=Dimension(length=1),
            output_dim=Dimension(length=1, time=-1),
        )

        # Pass raw tensor instead of DimTensor
        x = torch.randn(5, 3)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])

        output = layer(x, edge_index)

        assert isinstance(output, DimTensor)
        assert output.shape == (5, 4)
        assert output.dimension == Dimension(length=1, time=-1)

    def test_isolated_nodes(self):
        """Test with isolated nodes (no incoming edges)."""
        layer = DimGraphConv(in_features=2, out_features=2)

        # 5 nodes, but only 2 edges (nodes 3 and 4 are isolated)
        x = DimTensor(torch.randn(5, 2), units.dimensionless)
        edge_index = torch.tensor([[0, 1], [1, 2]])

        output = layer(x, edge_index)

        assert output.shape == (5, 2)
        # Isolated nodes should still get self-transformation

    def test_no_edges(self):
        """Test with no edges (only self-transformation)."""
        layer = DimGraphConv(in_features=3, out_features=4)

        x = DimTensor(torch.randn(5, 3), units.dimensionless)
        edge_index = torch.tensor([[], []]).long()  # Empty edge list

        output = layer(x, edge_index)

        assert output.shape == (5, 4)
        # Should only apply self-transformation

    def test_self_loops(self):
        """Test with self-loops (nodes connected to themselves)."""
        layer = DimGraphConv(in_features=2, out_features=2)

        x = DimTensor(torch.randn(3, 2), units.dimensionless)

        # Include self-loops: 0->0, 0->1, 1->1, 1->2, 2->2
        edge_index = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 1, 2, 2]])

        output = layer(x, edge_index)

        assert output.shape == (3, 2)


class TestDimGraphConvDimensions:
    """Test dimension tracking and validation."""

    def test_dimension_validation_pass(self):
        """Test that correct dimensions pass validation."""
        layer = DimGraphConv(
            in_features=3,
            out_features=3,
            input_dim=Dimension(length=1),
            output_dim=Dimension(length=1, time=-1),
            validate_input=True,
        )

        x = DimTensor(torch.randn(5, 3), units.m)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

        output = layer(x, edge_index)

        assert output.dimension == Dimension(length=1, time=-1)

    def test_dimension_validation_fail(self):
        """Test that incorrect dimensions raise DimensionError."""
        layer = DimGraphConv(
            in_features=3,
            out_features=3,
            input_dim=Dimension(length=1),
            output_dim=Dimension(length=1, time=-1),
            validate_input=True,
        )

        # Wrong dimension (mass instead of length)
        x = DimTensor(torch.randn(5, 3), units.kg)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

        with pytest.raises(DimensionError):
            layer(x, edge_index)

    def test_no_validation(self):
        """Test that validation can be disabled."""
        layer = DimGraphConv(
            in_features=3,
            out_features=3,
            input_dim=Dimension(length=1),
            output_dim=Dimension(length=1, time=-1),
            validate_input=False,
        )

        # Wrong dimension, but validation is off
        x = DimTensor(torch.randn(5, 3), units.kg)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

        # Should not raise error
        output = layer(x, edge_index)
        assert output.dimension == Dimension(length=1, time=-1)

    def test_dimensionless(self):
        """Test with dimensionless inputs and outputs."""
        layer = DimGraphConv(in_features=2, out_features=3)

        x = DimTensor(torch.randn(4, 2), units.dimensionless)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

        output = layer(x, edge_index)

        assert output.is_dimensionless


class TestDimGraphConvAggregation:
    """Test different aggregation methods."""

    def test_mean_aggregation(self):
        """Test mean aggregation."""
        layer = DimGraphConv(in_features=2, out_features=2, aggr="mean", normalize=True,
                             input_dim=Dimension(length=1), output_dim=Dimension(length=1))

        x = DimTensor(torch.randn(4, 2), units.m)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

        output = layer(x, edge_index)

        assert output.shape == (4, 2)

    def test_sum_aggregation(self):
        """Test sum aggregation (no normalization)."""
        layer = DimGraphConv(in_features=2, out_features=2, aggr="sum", normalize=False,
                             input_dim=Dimension(length=1), output_dim=Dimension(length=1))

        x = DimTensor(torch.randn(4, 2), units.m)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

        output = layer(x, edge_index)

        assert output.shape == (4, 2)

    def test_invalid_aggregation(self):
        """Test that invalid aggregation method raises ValueError."""
        with pytest.raises(ValueError, match="aggr must be"):
            DimGraphConv(in_features=2, out_features=2, aggr="invalid")

    def test_normalization_effect(self):
        """Test that normalization affects output."""
        torch.manual_seed(42)
        x = DimTensor(torch.randn(5, 2), units.m)
        # Node 2 has 2 incoming edges
        edge_index = torch.tensor([[0, 1, 3], [1, 2, 2]])

        # With normalization
        layer_norm = DimGraphConv(
            in_features=2, out_features=2, aggr="mean", normalize=True,
            input_dim=Dimension(length=1), output_dim=Dimension(length=1)
        )
        layer_norm.lin_self.weight.data = torch.eye(2)
        layer_norm.lin_neigh.weight.data = torch.eye(2)
        layer_norm.bias.data.zero_()

        output_norm = layer_norm(x, edge_index)

        # Without normalization
        layer_no_norm = DimGraphConv(
            in_features=2, out_features=2, aggr="mean", normalize=False,
            input_dim=Dimension(length=1), output_dim=Dimension(length=1)
        )
        layer_no_norm.lin_self.weight.data = torch.eye(2)
        layer_no_norm.lin_neigh.weight.data = torch.eye(2)
        layer_no_norm.bias.data.zero_()

        output_no_norm = layer_no_norm(x, edge_index)

        # Node 2 should have different values due to normalization
        assert not torch.allclose(output_norm.data[2], output_no_norm.data[2])


class TestDimGraphConvParameters:
    """Test layer parameters and configuration."""

    def test_with_bias(self):
        """Test layer with bias."""
        layer = DimGraphConv(in_features=3, out_features=4, bias=True)

        assert layer.bias is not None
        assert layer.bias.shape == (4,)

    def test_without_bias(self):
        """Test layer without bias."""
        layer = DimGraphConv(in_features=3, out_features=4, bias=False)

        assert layer.bias is None

    def test_parameter_count(self):
        """Test that layer has correct number of parameters."""
        layer = DimGraphConv(in_features=3, out_features=4, bias=True)

        params = list(layer.parameters())
        # Should have: lin_self.weight, lin_neigh.weight, bias
        assert len(params) == 3

        # Check shapes (order may vary, check that all expected shapes are present)
        shapes = [tuple(p.shape) for p in params]
        assert (4, 3) in shapes  # lin_self.weight or lin_neigh.weight
        assert shapes.count((4, 3)) == 2  # both weight matrices
        assert (4,) in shapes  # bias

    def test_extra_repr(self):
        """Test string representation."""
        layer = DimGraphConv(
            in_features=3,
            out_features=4,
            input_dim=Dimension(length=1),
            output_dim=Dimension(length=1, time=-1),
            aggr="mean",
            normalize=True,
            bias=True,
        )

        repr_str = repr(layer)
        assert "in_features=3" in repr_str
        assert "out_features=4" in repr_str
        assert "aggr='mean'" in repr_str
        assert "normalize=True" in repr_str


class TestDimGraphConvGradients:
    """Test gradient flow and autograd compatibility."""

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the layer."""
        layer = DimGraphConv(in_features=3, out_features=4, input_dim=Dimension(length=1), output_dim=Dimension(length=1))

        raw_tensor = torch.randn(5, 3, requires_grad=True)
        x = DimTensor(raw_tensor, units.m)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])

        output = layer(x, edge_index)

        # Create a scalar loss
        loss = output.sum()
        loss.backward()

        # Check that gradients exist (check original tensor, not DimTensor wrapper)
        assert raw_tensor.grad is not None
        assert layer.lin_self.weight.grad is not None
        assert layer.lin_neigh.weight.grad is not None

    def test_backprop_through_aggregation(self):
        """Test backpropagation through aggregation."""
        layer = DimGraphConv(in_features=2, out_features=2,
                             input_dim=Dimension(length=1), output_dim=Dimension(length=1))

        raw_tensor = torch.randn(4, 2, requires_grad=True)
        x = DimTensor(raw_tensor, units.m)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

        output = layer(x, edge_index)
        loss = (output**2).sum()
        loss.backward()

        # All nodes should have gradients (check original tensor, not DimTensor wrapper)
        assert raw_tensor.grad is not None
        assert torch.all(torch.isfinite(raw_tensor.grad))


class TestDimGraphConvIntegration:
    """Integration tests with other layers."""

    def test_in_sequential(self):
        """Test DimGraphConv chained with DimLinear."""
        # Note: DimGraphConv needs special handling in forward pass
        # This tests that dimensions are compatible

        graph_layer = DimGraphConv(
            in_features=3,
            out_features=4,
            input_dim=Dimension(length=1),
            output_dim=Dimension(length=1, time=-1),
        )

        # DimLinear expects [batch, features], same as graph output
        linear_layer = DimLinear(
            in_features=4,
            out_features=2,
            input_dim=Dimension(length=1, time=-1),
            output_dim=Dimension(length=1, mass=1, time=-2),
        )

        x = DimTensor(torch.randn(5, 3), units.m)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])

        # Apply graph conv, then linear
        out1 = graph_layer(x, edge_index)
        out2 = linear_layer(out1)

        assert out2.dimension == Dimension(length=1, mass=1, time=-2)

    def test_physics_example(self):
        """Test realistic physics example: positions -> forces."""
        # Molecular dynamics: atom positions [m] -> forces [N]
        layer = DimGraphConv(
            in_features=3,
            out_features=3,
            input_dim=Dimension(length=1),  # meters
            output_dim=Dimension(length=1, mass=1, time=-2),  # Newtons
        )

        # 4 atoms with 3D positions
        positions = DimTensor(torch.randn(4, 3), units.m)

        # Bonds: 0-1, 1-2, 2-3 (chain of atoms)
        # Bidirectional edges
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])

        forces = layer(positions, edge_index)

        assert forces.dimension == Dimension(length=1, mass=1, time=-2)
        assert forces.shape == (4, 3)

    def test_multiple_graph_convs(self):
        """Test stacking multiple graph convolution layers."""
        layer1 = DimGraphConv(
            in_features=3,
            out_features=4,
            input_dim=Dimension(length=1),
            output_dim=Dimension(length=1),
        )

        layer2 = DimGraphConv(
            in_features=4,
            out_features=2,
            input_dim=Dimension(length=1),
            output_dim=Dimension(length=1, time=-1),
        )

        x = DimTensor(torch.randn(5, 3), units.m)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])

        # Apply two graph convolutions with same edge_index
        out1 = layer1(x, edge_index)
        out2 = layer2(out1, edge_index)

        assert out2.dimension == Dimension(length=1, time=-1)
        assert out2.shape == (5, 2)


class TestDimGraphConvEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_node(self):
        """Test with single node."""
        layer = DimGraphConv(in_features=2, out_features=3,
                             input_dim=Dimension(length=1), output_dim=Dimension(length=1))

        x = DimTensor(torch.randn(1, 2), units.m)
        edge_index = torch.tensor([[], []]).long()

        output = layer(x, edge_index)

        assert output.shape == (1, 3)

    def test_fully_connected(self):
        """Test with fully connected graph."""
        layer = DimGraphConv(in_features=2, out_features=2,
                             input_dim=Dimension(length=1), output_dim=Dimension(length=1))

        # 3 nodes, fully connected
        x = DimTensor(torch.randn(3, 2), units.m)

        # All pairs of edges (including self-loops)
        edge_index = torch.tensor(
            [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]]
        )

        output = layer(x, edge_index)

        assert output.shape == (3, 2)

    def test_large_degree_nodes(self):
        """Test with nodes that have many neighbors."""
        layer = DimGraphConv(in_features=2, out_features=2, normalize=True,
                             input_dim=Dimension(length=1), output_dim=Dimension(length=1))

        # 10 nodes, node 0 receives edges from all others
        x = DimTensor(torch.randn(10, 2), units.m)

        # All nodes connect to node 0
        sources = list(range(1, 10))
        targets = [0] * 9
        edge_index = torch.tensor([sources, targets])

        output = layer(x, edge_index)

        assert output.shape == (10, 2)
        # Node 0 should have aggregated information from all neighbors

    def test_empty_graph(self):
        """Test with no nodes and no edges."""
        layer = DimGraphConv(in_features=2, out_features=3,
                             input_dim=Dimension(length=1), output_dim=Dimension(length=1))

        x = DimTensor(torch.randn(0, 2), units.m)
        edge_index = torch.tensor([[], []]).long()

        output = layer(x, edge_index)

        assert output.shape == (0, 3)

    def test_device_compatibility(self):
        """Test that layer works on different devices."""
        layer = DimGraphConv(in_features=3, out_features=4,
                             input_dim=Dimension(length=1), output_dim=Dimension(length=1))

        # CPU
        x_cpu = DimTensor(torch.randn(5, 3), units.m)
        edge_index_cpu = torch.tensor([[0, 1, 2], [1, 2, 3]])

        output_cpu = layer(x_cpu, edge_index_cpu)

        assert output_cpu.device.type == "cpu"

        # Note: GPU test would require CUDA availability check


class TestDimGraphConvNumerical:
    """Test numerical correctness of operations."""

    def test_self_and_neighbor_contribution(self):
        """Test that self and neighbor contributions are correctly combined."""
        layer = DimGraphConv(in_features=2, out_features=2, bias=False, normalize=True,
                             input_dim=Dimension(length=1), output_dim=Dimension(length=1))

        # Set weights to identity for easy verification
        layer.lin_self.weight.data = torch.eye(2)
        layer.lin_neigh.weight.data = torch.eye(2)

        # Simple graph: 0 -> 1
        x = DimTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), units.m)
        edge_index = torch.tensor([[0], [1]])

        output = layer(x, edge_index)

        # Node 0: only self-transformation = [1, 2]
        assert torch.allclose(output.data[0], torch.tensor([1.0, 2.0]))

        # Node 1: self [3, 4] + neighbor [1, 2] = [4, 6]
        assert torch.allclose(output.data[1], torch.tensor([4.0, 6.0]))

    def test_degree_normalization(self):
        """Test that degree normalization works correctly."""
        layer = DimGraphConv(in_features=1, out_features=1, bias=False, normalize=True,
                             input_dim=Dimension(length=1), output_dim=Dimension(length=1))

        # Set weights to identity
        layer.lin_self.weight.data = torch.ones(1, 1)
        layer.lin_neigh.weight.data = torch.ones(1, 1)

        # Graph: 0 -> 2, 1 -> 2 (node 2 has degree 2)
        x = DimTensor(torch.tensor([[1.0], [2.0], [3.0]]), units.m)
        edge_index = torch.tensor([[0, 1], [2, 2]])

        output = layer(x, edge_index)

        # Node 2: self=3 + (1+2)/2 = 3 + 1.5 = 4.5
        assert torch.allclose(output.data[2], torch.tensor([4.5]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Plan: DimGraphConv Implementation

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner agent

---

## Goal

Implement `DimGraphConv` - a graph convolution layer that tracks physical dimensions through message passing operations in graph neural networks.

---

## Background

Graph Neural Networks (GNNs) are increasingly important in physics simulations where entities (nodes) interact through edges (bonds, forces, etc.). For dimtensor's physics ML capabilities, we need GNN layers that preserve unit tracking.

**Use cases**:
- Molecular dynamics: atoms (nodes with positions [m], velocities [m/s]) connected by bonds
- Particle systems: particles with mass [kg], charge [C], momentum [kg·m/s]
- Graph-based PDEs: mesh nodes with physical field values

Graph convolutions follow a message-passing paradigm:
1. **Message**: Compute messages from neighbors (aggregate neighbor features)
2. **Aggregate**: Sum/mean/max messages
3. **Update**: Combine aggregated messages with node features to update node state

---

## Existing Patterns

From `src/dimtensor/torch/layers.py`:

**DimLayer Pattern**:
```python
class DimLayer(nn.Module, ABC):
    def __init__(self, input_dim, output_dim, validate_input=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._output_unit = Unit(str(output_dim), output_dim, 1.0)

    @abstractmethod
    def _forward_impl(self, x: Tensor) -> Tensor:
        """Raw computation without unit wrapping"""
        pass

    def forward(self, x: DimTensor | Tensor) -> DimTensor:
        """Validates input dimension, calls _forward_impl, wraps output"""
        # Validate input dimension if DimTensor
        # Call _forward_impl with raw tensor
        # Return DimTensor with output_dim
```

**Key principles**:
1. Separate dimension validation from computation (`forward` vs `_forward_impl`)
2. Accept both `DimTensor` and raw `Tensor` inputs
3. Always return `DimTensor` with the layer's `output_dim`
4. Store learned parameters in wrapped PyTorch modules

---

## Approach

### Option A: Minimal Standalone Implementation

Implement graph convolution from scratch using PyTorch operations, without external GNN dependencies.

**Pros**:
- No new dependencies (keeps dimtensor lightweight)
- Full control over implementation
- Easier to understand and maintain
- Can optimize for unit tracking

**Cons**:
- Need to implement message passing logic ourselves
- Missing optimizations from specialized libraries
- Users might want torch_geometric compatibility

### Option B: torch_geometric Integration

Create a wrapper around `torch_geometric.nn.MessagePassing` with dimension tracking.

**Pros**:
- Leverages battle-tested GNN implementation
- Familiar API for GNN users
- More graph operations available

**Cons**:
- Adds dependency on torch_geometric (and its dependencies)
- Harder to customize for unit tracking
- More complex to maintain

### Option C: Hybrid - Standalone with Optional Integration

Implement minimal standalone version, with optional torch_geometric compatibility if installed.

**Pros**:
- Works out of the box without dependencies
- Power users can use torch_geometric if needed
- Best of both worlds

**Cons**:
- More code to maintain
- Testing complexity

### Decision: Option A (Minimal Standalone)

**Rationale**:
1. dimtensor currently has no torch_geometric dependency, and adding one would increase installation complexity
2. Graph convolution can be implemented cleanly in ~100 lines using PyTorch primitives
3. For v3.5.0, we want to demonstrate GNN capability - a simple, working implementation is sufficient
4. Users who need advanced GNN features can still use torch_geometric alongside dimtensor
5. Future versions (v4.x) can add torch_geometric integration if demand exists

**Implementation Strategy**:
- Implement a simple GCN-style convolution (as in Kipf & Welling 2017)
- Message function: linear transformation of neighbor features
- Aggregation: sum or mean
- Update: add self-loop contribution

---

## Design

### Class Structure

```python
class DimGraphConv(DimLayer):
    """Graph convolution layer with dimension tracking.

    Implements a simple graph convolution:
        h_i' = σ(W_self · h_i + W_neigh · Σ_{j∈N(i)} h_j / |N(i)|)

    Where:
    - h_i: node feature vector for node i [input_dim]
    - h_i': updated node feature vector [output_dim]
    - N(i): neighbors of node i
    - W_self, W_neigh: learnable weight matrices

    Args:
        in_features: Number of input features per node
        out_features: Number of output features per node
        input_dim: Physical dimension of node features
        output_dim: Physical dimension of output features
        aggr: Aggregation method ('mean' or 'sum')
        bias: If True, add learnable bias
        normalize: If True, normalize by degree (degree=|N(i)|)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        input_dim: Dimension = DIMENSIONLESS,
        output_dim: Dimension = DIMENSIONLESS,
        aggr: str = "mean",
        bias: bool = True,
        normalize: bool = True,
        device=None,
        dtype=None,
        validate_input: bool = True,
    ):
        super().__init__(input_dim, output_dim, validate_input)

        self.in_features = in_features
        self.out_features = out_features
        self.aggr = aggr
        self.normalize = normalize

        # Linear transformations
        self.lin_self = nn.Linear(in_features, out_features, bias=False, ...)
        self.lin_neigh = nn.Linear(in_features, out_features, bias=False, ...)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, ...))
        else:
            self.register_parameter('bias', None)

    def _forward_impl(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Compute graph convolution.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge connectivity [2, num_edges]
                       edge_index[0]: source nodes
                       edge_index[1]: target nodes

        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Self transformation
        out_self = self.lin_self(x)

        # Neighbor aggregation
        row, col = edge_index
        neighbor_feats = x[row]  # Features of source nodes
        neighbor_msg = self.lin_neigh(neighbor_feats)

        # Aggregate messages to target nodes
        out_neigh = torch.zeros_like(out_self)
        out_neigh.index_add_(0, col, neighbor_msg)

        # Normalize by degree if requested
        if self.normalize:
            deg = torch.zeros(x.size(0), device=x.device)
            deg.index_add_(0, col, torch.ones(col.size(0), device=x.device))
            deg = deg.clamp(min=1.0)
            out_neigh = out_neigh / deg.unsqueeze(-1)

        # Combine
        out = out_self + out_neigh

        if self.bias is not None:
            out = out + self.bias

        return out

    def forward(
        self,
        x: DimTensor | Tensor,
        edge_index: Tensor,
    ) -> DimTensor:
        """Forward pass with dimension tracking.

        Args:
            x: Node features with units
            edge_index: Edge connectivity (unitless indices)

        Returns:
            Updated node features with output_dim
        """
        # Validate input dimension
        if isinstance(x, DimTensor):
            if self.validate_input and x.dimension != self.input_dim:
                raise DimensionError(...)
            tensor = x.data
        else:
            tensor = x

        # Compute (edge_index is always raw tensor)
        result = self._forward_impl(tensor, edge_index)

        # Return with output dimension
        return DimTensor._from_tensor_and_unit(result, self._output_unit)
```

### Unit Propagation Rules

**Key insight**: Graph operations (sum, mean) preserve dimensions when all inputs have the same dimension.

1. **Node features**: Must all have the same `input_dim` (validated on input)
2. **Self transformation**: `W_self · h_i` has implicit dimension `output_dim / input_dim`
3. **Neighbor messages**: `W_neigh · h_j` also has dimension `output_dim / input_dim`
4. **Aggregation**: Sum/mean of same-dimension tensors preserves dimension
5. **Output**: Has dimension `output_dim`

**Edge features**: For v3.5.0, we'll start without edge features. If added later:
- Edge features would be optional `DimTensor` with their own dimension
- Would require a third linear layer: `W_edge · e_{ij}`
- Edge dimension could be different from node dimension
- Example: nodes are positions [m], edges are forces [N]

### Parameters

- `in_features: int` - Input feature dimensionality
- `out_features: int` - Output feature dimensionality
- `input_dim: Dimension` - Physical dimension of input node features
- `output_dim: Dimension` - Physical dimension of output node features
- `aggr: str` - Aggregation method ('mean', 'sum', 'max')
- `bias: bool` - Whether to include bias term
- `normalize: bool` - Whether to normalize by node degree
- `validate_input: bool` - Whether to validate input dimensions (from DimLayer)

---

## Implementation Steps

1. [ ] Create `DimGraphConv` class in `src/dimtensor/torch/layers.py`
   - Inherit from `DimLayer`
   - Add `__init__` with parameters above
   - Create `lin_self` and `lin_neigh` Linear layers
   - Create optional bias parameter

2. [ ] Implement `_forward_impl(x: Tensor, edge_index: Tensor) -> Tensor`
   - Apply self transformation: `out_self = lin_self(x)`
   - Gather neighbor features: `x[edge_index[0]]`
   - Apply neighbor transformation: `out_neigh = lin_neigh(...)`
   - Aggregate to target nodes using `index_add_`
   - Normalize by degree if `normalize=True`
   - Combine: `out = out_self + out_neigh + bias`

3. [ ] Override `forward(x: DimTensor | Tensor, edge_index: Tensor) -> DimTensor`
   - Extract raw tensor from DimTensor input
   - Validate dimension if needed
   - Call `_forward_impl(tensor, edge_index)`
   - Wrap result with `output_dim`
   - Add comprehensive docstring with usage example

4. [ ] Add `extra_repr()` for pretty printing
   - Include in_features, out_features, aggr, normalize
   - Include input_dim, output_dim

5. [ ] Update `src/dimtensor/torch/__init__.py`
   - Export `DimGraphConv`

6. [ ] Add docstring example
   - Show molecular dynamics use case
   - Demonstrate dimension preservation
   - Show edge_index format

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/torch/layers.py` | Add `DimGraphConv` class (~120 lines with docstring) |
| `src/dimtensor/torch/__init__.py` | Export `DimGraphConv` |

---

## Testing Strategy

Create comprehensive tests in `tests/test_torch.py` (or new `tests/test_torch_graph.py`):

### Unit Tests

1. [ ] **Test basic forward pass**
   - Create simple graph (3 nodes, 2 edges)
   - Pass DimTensor with units [m]
   - Verify output shape and dimension

2. [ ] **Test dimension validation**
   - Pass DimTensor with wrong dimension
   - Verify `DimensionError` is raised

3. [ ] **Test aggregation methods**
   - Test aggr='mean'
   - Test aggr='sum'
   - Verify mathematical correctness

4. [ ] **Test normalization**
   - Test normalize=True vs normalize=False
   - Verify degree normalization works correctly

5. [ ] **Test edge cases**
   - Single node (no edges)
   - Disconnected graph
   - Self-loops
   - Empty graph

6. [ ] **Test gradient flow**
   - Create loss function
   - Backpropagate through DimGraphConv
   - Verify gradients exist and are correct

7. [ ] **Test with raw Tensor input**
   - Pass raw tensor instead of DimTensor
   - Verify still returns DimTensor with output_dim

8. [ ] **Test parameter initialization**
   - Check weight shapes
   - Check bias exists when bias=True

### Integration Tests

9. [ ] **Test in DimSequential**
   - Chain DimGraphConv with DimLinear
   - Verify dimension compatibility checking

10. [ ] **Test realistic physics example**
    - Molecular system: positions [m] -> forces [N]
    - Verify physical correctness of unit transformations

### Property Tests

11. [ ] **Test permutation invariance**
    - Reorder nodes and edges consistently
    - Verify output is permuted correctly (graph convolution is permutation-equivariant)

---

## Risks / Edge Cases

### Risk 1: Graph indexing complexity
**Concern**: Edge index handling with `index_add_` can be tricky, especially for large graphs or unusual topologies.

**Mitigation**:
- Start with simple, well-tested implementation
- Add extensive edge case tests
- Document edge_index format clearly (COO format: [2, num_edges])

### Risk 2: Memory efficiency for large graphs
**Concern**: Creating `torch.zeros_like(out_self)` for aggregation might not scale well.

**Mitigation**:
- For v3.5.0, prioritize correctness over performance
- Add TODO comment for future optimization (scatter operations, sparse tensors)
- Document memory considerations in docstring

### Risk 3: Users expect torch_geometric compatibility
**Concern**: Users familiar with torch_geometric might expect compatible APIs.

**Mitigation**:
- Document that this is a standalone implementation
- Keep parameter names similar to torch_geometric where reasonable
- Note in docstring: "For advanced GNN features, see torch_geometric"

### Edge Case 1: Self-loops
**Handling**: Self-loops (edges where source == target) should be handled naturally by the aggregation - a node simply receives its own features as a message.

### Edge Case 2: Isolated nodes
**Handling**: Nodes with no neighbors should still get self-transformation. Degree normalization should handle deg=0 gracefully with `.clamp(min=1.0)`.

### Edge Case 3: Bidirectional edges
**Handling**: Graph should be undirected for most physics applications. Users must provide both (i,j) and (j,i) edges if they want undirected convolution.

### Edge Case 4: Batch processing
**Handling**: For v3.5.0, assume single graph. Batching multiple graphs requires additional metadata (batch indices). Document this limitation and suggest workarounds (disjoint union of graphs).

---

## Definition of Done

- [x] Plan created and reviewed
- [ ] `DimGraphConv` implemented in `layers.py`
- [ ] Exports added to `__init__.py`
- [ ] All unit tests pass (10+ tests)
- [ ] Integration test with physics example passes
- [ ] Docstring includes detailed usage example
- [ ] Type hints complete and mypy passes
- [ ] Code follows existing layer patterns (DimLinear, DimConv2d)
- [ ] CONTINUITY.md updated with task completion

---

## Notes / Log

**2026-01-09 14:00** - Plan created by planner agent
- Researched existing DimLayer patterns
- Chose standalone implementation (no torch_geometric dependency)
- Designed simple GCN-style convolution with unit tracking
- Identified key risks and edge cases
- Ready for implementer agent

---

## References

- **Kipf & Welling (2017)**: Semi-Supervised Classification with Graph Convolutional Networks
  - Standard GCN formulation

- **dimtensor layers.py**: Existing DimLayer pattern
  - `DimLinear`, `DimConv2d` as reference implementations

- **torch_geometric.nn.MessagePassing**: Industry-standard GNN API
  - Reference for parameter naming and method signatures (for future compatibility)

---

## Future Enhancements (Post v3.5.0)

1. **Edge features support**
   - Add optional `edge_attr: DimTensor` parameter
   - Add `lin_edge` layer for edge feature transformation
   - Example: node forces depend on bond strengths

2. **More aggregation functions**
   - Add 'max' aggregation
   - Add 'attention' aggregation (GAT-style)

3. **Batch processing**
   - Support multiple graphs in single forward pass
   - Add `batch` parameter for graph membership

4. **torch_geometric compatibility layer**
   - Optional wrapper: `DimMessagePassing` that wraps `MessagePassing`
   - Allows using torch_geometric's optimized ops with unit tracking

5. **Sparse tensor optimization**
   - Use PyTorch sparse tensors for adjacency matrix
   - Improve memory efficiency for large graphs

6. **Heterogeneous graphs**
   - Different node types with different dimensions
   - Different edge types with different dimensions
   - Example: protein structure (atoms, residues, domains)

# GNN for Physics Research

**Date**: 2026-01-09
**Status**: COMPLETED
**Author**: planner agent

---

## Goal

Research graph neural network architectures used in physics applications and design a unit-aware DimGraphConv layer that tracks physical dimensions through message passing operations while maintaining compatibility with the existing DimLayer framework.

---

## Background

Graph Neural Networks (GNNs) have become essential in physics-informed machine learning, particularly for:
- Molecular dynamics and interatomic potential learning
- Particle physics (jet tagging, event reconstruction)
- Materials science (crystal property prediction)
- Multi-agent physical systems (particles, robots)

dimtensor v3.5.0 aims to provide state-of-the-art ML architectures with full unit tracking, making it crucial to understand how physical dimensions propagate through graph-structured data.

---

## Background Research: GNN Architectures for Physics

### 1. Message Passing Neural Networks (MPNNs)

**Core Framework**: Nodes exchange information with neighbors iteratively:
```
m_ij = φ_msg(h_i, h_j, e_ij)        # Message from j to i
m_i = aggregate({m_ij : j ∈ N(i)})   # Aggregate messages
h_i' = φ_update(h_i, m_i)            # Update node representation
```

**Physical Considerations**:
- Node features `h_i` can have different dimensions (position: [L], velocity: [L/T], mass: [M])
- Edge features `e_ij` often include distances ([L]), angles (dimensionless), or forces ([M·L/T²])
- Aggregation (sum, mean) must preserve dimensional consistency

### 2. SchNet (Continuous-Filter Convolutional Networks)

**Architecture**:
- Uses continuous convolutions on interatomic distances
- Filter-generating networks produce weights based on distance
- Rotationally and translationally invariant

**Key Innovation**:
```
h_i' = Σ_j φ_filter(||r_ij||) ⊙ φ_dense(h_j)
```
where `r_ij` is the distance vector ([L])

**Unit Flow**:
- Input: Atomic positions [L]
- Distances: [L]
- Output: Energies [M·L²/T²], forces [M·L/T²]

### 3. E(3)-Equivariant GNNs (EGNN, NequIP, PaiNN)

**Philosophy**: Respect 3D Euclidean symmetries (rotations, translations, reflections)

**Architecture Components**:
- **Scalar features**: Rotationally invariant (energy, charge, mass)
- **Vector features**: Equivariant to rotations (position, velocity, force)
- **Message passing**: Preserves equivariance through geometric operations

**Example (EGNN)**:
```
m_ij = φ_e(h_i, h_j, ||x_i - x_j||², e_ij)  # Scalar message
x_i' = x_i + Σ_j (x_i - x_j) φ_x(m_ij)      # Coordinate update
h_i' = φ_h(h_i, Σ_j m_ij)                    # Feature update
```

**Physical Units in Equivariant Models**:
- Equivariance (geometric symmetry) is ORTHOGONAL to unit tracking
- Can have unit-aware equivariant networks
- Position vectors [L], velocity vectors [L/T], force vectors [M·L/T²]

### 4. Graph Network Simulators (GNS)

**Application**: Learning physics simulations (cloth, fluids, particles)

**Architecture**:
- Encoder: Maps raw features to latent space
- Processor: GNN message passing (multiple layers)
- Decoder: Maps latent to physical outputs

**Unit Flow**:
- Input: Positions [L], velocities [L/T], material properties
- Output: Accelerations [L/T²] or next-step positions [L]

---

## Unit Flow in GNNs: Design Principles

### Principle 1: Dimensional Consistency in Message Passing

**Rule**: All messages aggregated to a node must have the same dimension.

**Example**:
```python
# VALID: All messages have dimension [M·L/T²]
force_msg_1 = DimTensor([1.0, 0.0], units.N)  # [M·L/T²]
force_msg_2 = DimTensor([0.0, 1.0], units.N)  # [M·L/T²]
total_force = force_msg_1 + force_msg_2       # [M·L/T²]

# INVALID: Cannot aggregate different dimensions
energy_msg = DimTensor([1.0], units.J)        # [M·L²/T²]
force_msg = DimTensor([1.0, 0.0], units.N)    # [M·L/T²]
# total = energy_msg + force_msg  # DimensionError!
```

### Principle 2: Edge Features and Distance-Based Weights

**Edge Features** can have arbitrary dimensions:
- Distance: `||r_ij||` has dimension [L]
- Relative velocity: `v_i - v_j` has dimension [L/T]
- Force magnitude: `F_ij` has dimension [M·L/T²]

**Distance-Based Weighting** (SchNet-style):
```python
# Distance [L] → dimensionless weight
distance = DimTensor(r_ij, units.m)           # [L]
weight = radial_basis_function(distance)       # Dimensionless
# RBF must output dimensionless values!

# Apply weight to message
message = weight * node_feature                # Preserves dimension
```

### Principle 3: Aggregation Operations

**Sum Aggregation**: Preserves dimension
```python
# If all messages are [M·L/T²], sum is [M·L/T²]
total_force = sum([msg_1, msg_2, msg_3])
```

**Mean Aggregation**: Preserves dimension
```python
# If all messages are [M·L/T²], mean is [M·L/T²]
avg_force = mean([msg_1, msg_2, msg_3])
```

**Max Aggregation**: Preserves dimension (but less physically meaningful)

### Principle 4: Node Update Functions

**Pattern**: Combine old features with aggregated messages

**Option A: Residual Connection** (requires same dimension)
```python
h_new = h_old + aggregate(messages)  # Requires matching dimensions
```

**Option B: Learned Transformation** (allows dimension change)
```python
h_new = MLP(concat(h_old, aggregate(messages)))
# MLP can change dimensions via DimLinear layers
```

### Principle 5: Multi-Dimensional Node Features

**Real physical systems have heterogeneous features**:
```python
node_features = {
    'position': DimTensor([x, y, z], units.m),      # [L]
    'velocity': DimTensor([vx, vy, vz], units.m/units.s),  # [L/T]
    'mass': DimTensor([m], units.kg),                # [M]
    'charge': DimTensor([q], units.C),               # [I·T]
}
```

**Approach**: Separate message passing channels per dimension
- Position messages: [L]
- Velocity messages: [L/T]
- Force messages: [M·L/T²] (used to update momentum)

---

## Recommended Architecture for DimGraphConv

### Design Decision: Flexible Message Passing with Unit Tracking

**Architecture**: Extend DimLayer to handle graph-structured data

```python
class DimGraphConv(DimLayer):
    """Graph convolution with physical dimension tracking.

    Performs message passing on graphs while tracking physical dimensions:
    1. Message generation: φ_msg(h_i, h_j, e_ij)
    2. Aggregation: aggregate({m_ij : j ∈ N(i)})
    3. Update: φ_update(h_i, m_i)

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        input_dim: Physical dimension of input node features
        output_dim: Physical dimension of output node features
        edge_dim: Dimension of edge features (optional)
        edge_input_dim: Physical dimension of edge features
        aggr: Aggregation method ('sum', 'mean', 'max')
    """
```

### Key Design Choices

**1. Separate Edge and Node Dimensions**
- Node features have their own dimension (e.g., [L], [M])
- Edge features have independent dimensions (e.g., [L] for distances)
- Message function combines them respecting dimensional analysis

**2. Message Function Structure**
```python
def _message(self, x_i: DimTensor, x_j: DimTensor,
             edge_attr: DimTensor | None) -> DimTensor:
    """Generate messages.

    Must return messages with a CONSISTENT dimension for aggregation.
    """
    # Option A: Simple multiplication (SchNet-style)
    # edge_attr [L] → dimensionless weights
    weight = self.edge_network(edge_attr)  # → dimensionless
    message = weight * x_j  # Preserves x_j dimension

    # Option B: Learned transformation
    # Concatenate and transform
    combined = torch.cat([x_i.data, x_j.data, edge_attr.data], dim=-1)
    message_data = self.message_network(combined)
    message = DimTensor(message_data, message_unit)

    return message
```

**3. Aggregation with Dimension Preservation**
```python
def _aggregate(self, messages: list[DimTensor]) -> DimTensor:
    """Aggregate messages.

    All messages must have the same dimension.
    """
    if self.aggr == 'sum':
        return sum(messages)  # DimTensor.__add__ handles this
    elif self.aggr == 'mean':
        return sum(messages) / len(messages)  # Preserves dimension
    elif self.aggr == 'max':
        # Max-pooling (less common in physics)
        return max(messages)
```

**4. Update Function**
```python
def _update(self, x: DimTensor, aggr_out: DimTensor) -> DimTensor:
    """Update node features.

    Can change dimensions via DimLinear.
    """
    # Combine old and new information
    # Use DimLinear to allow dimension transformations
    combined = torch.cat([x.data, aggr_out.data], dim=-1)
    output_data = self.update_network(combined)
    return DimTensor(output_data, self._output_unit)
```

### Integration with PyTorch Geometric (Optional)

**Decision**: Start with a standalone implementation, then add PyG compatibility

**Rationale**:
- PyG has complex edge_index conventions
- Many users may not have PyG installed
- Standalone version easier to test and understand
- Can add PyG mixin later for advanced users

**Standalone Format**:
```python
# Graph representation
adjacency: torch.Tensor  # [num_edges, 2] - list of (src, dst) pairs
node_features: DimTensor  # [num_nodes, in_channels]
edge_features: DimTensor | None  # [num_edges, edge_dim]
```

---

## Implementation Steps

1. [ ] Create `src/dimtensor/torch/graph.py` module

2. [ ] Implement `DimGraphConv` base class
   - [ ] Inherit from `DimLayer`
   - [ ] Add `edge_input_dim` parameter
   - [ ] Implement `_message()` method signature
   - [ ] Implement `_aggregate()` method (sum/mean)
   - [ ] Implement `_update()` method
   - [ ] Override `_forward_impl()` for graph data

3. [ ] Implement message network components
   - [ ] `DimMessageNetwork`: Transforms (h_i, h_j, e_ij) → message
   - [ ] Respect dimensional consistency in outputs
   - [ ] Support both SchNet-style (distance-based) and learned messages

4. [ ] Add aggregation options
   - [ ] `sum`: Simple summation
   - [ ] `mean`: Average (preserves dimension)
   - [ ] `max`: Maximum (use with caution for physics)

5. [ ] Create edge network for distance-based weighting
   - [ ] `RadialBasisNetwork`: [L] → dimensionless weights
   - [ ] Gaussian RBF, exponential decay, etc.
   - [ ] Cutoff functions for finite interaction range

6. [ ] Add multi-layer GNN support
   - [ ] `DimGNN`: Stack multiple DimGraphConv layers
   - [ ] Validate dimension compatibility between layers
   - [ ] Support residual connections (same dimension)

7. [ ] Add utilities for graph construction
   - [ ] `build_edge_index()`: Construct graph from positions
   - [ ] `compute_edge_distances()`: Calculate ||r_ij|| with units
   - [ ] Support periodic boundary conditions (for materials)

8. [ ] Create examples
   - [ ] Particle system (N-body problem)
   - [ ] Molecular graph (atoms + bonds)
   - [ ] Spring-mass system

---

## Files to Create/Modify

| File | Change |
|------|--------|
| `src/dimtensor/torch/graph.py` | Create new module with `DimGraphConv`, `DimMessageNetwork`, `RadialBasisNetwork` |
| `src/dimtensor/torch/__init__.py` | Export new GNN classes |
| `tests/torch/test_graph.py` | Unit tests for graph convolutions |
| `tests/torch/test_graph_physics.py` | Physics-based integration tests (N-body, molecules) |
| `examples/gnn_particle_physics.py` | Example: Learning particle interactions |
| `examples/gnn_molecular_dynamics.py` | Example: Predicting molecular properties |
| `docs/architecture/gnn.md` | Documentation: GNN architecture and unit flow |

---

## Testing Strategy

### Unit Tests

- [ ] Test message generation with various dimensions
  - Input [L], output [L]
  - Input [M], output [M·L/T²]
  - Edge features [L] → dimensionless weights

- [ ] Test aggregation preserves dimensions
  - Sum of forces [M·L/T²]
  - Mean of velocities [L/T]

- [ ] Test update function changes dimensions
  - Input [L] → Output [L/T] (position → velocity)
  - Input [M] → Output [M·L²/T²] (mass → energy)

- [ ] Test dimension validation
  - Reject incompatible edge dimensions
  - Reject heterogeneous message dimensions

- [ ] Test with PyTorch autograd
  - Gradient flow through message passing
  - Backward pass preserves units

### Integration Tests

- [ ] **N-body gravitational system**
  - Nodes: particles with mass [M] and position [L]
  - Edges: distances [L]
  - Messages: forces [M·L/T²]
  - Output: accelerations [L/T²]
  - Validate: F = G·m₁·m₂/r² dimensional correctness

- [ ] **Spring-mass network**
  - Nodes: masses [M], positions [L], velocities [L/T]
  - Edges: spring constants [M/T²], rest lengths [L]
  - Messages: spring forces [M·L/T²]
  - Output: accelerations [L/T²]
  - Validate: F = -k(x - x₀) dimensional correctness

- [ ] **Molecular property prediction**
  - Nodes: atomic features (charge, electronegativity)
  - Edges: bond types, distances [L]
  - Output: molecular energy [M·L²/T²]
  - Validate: Energy is extensive property

### Manual Verification

- [ ] Print dimension flow through 3-layer GNN
- [ ] Visualize learned interaction functions
- [ ] Compare to known physics (inverse-square law, Hooke's law)

---

## Risks / Edge Cases

### Risk 1: Heterogeneous Node Features
**Problem**: Real systems have nodes with multiple features of different dimensions (position [L], mass [M], charge [I·T])

**Mitigation**:
- Support dictionary-based node features: `{'position': DimTensor(...), 'mass': DimTensor(...)}`
- Or separate message passing channels per feature type
- Document best practices for handling multi-dimensional features

### Risk 2: Dimensionless Ratios in Message Passing
**Problem**: Some physical operations require dimensionless inputs (e.g., exp(-r/r_cutoff))

**Handling**:
- Explicitly require cutoff radius to create dimensionless ratio
- `exp(-distance / cutoff)` where both have [L]
- Document pattern for creating dimensionless weights

### Risk 3: PyG Compatibility
**Problem**: PyTorch Geometric (PyG) is the dominant GNN library, but has complex internals

**Mitigation**:
- Start with standalone implementation (no PyG dependency)
- Create `DimGraphConv` as PyG-compatible but not dependent
- Later: Add `pyg_compatible=True` flag or `DimMessagePassing` base class extending PyG

### Risk 4: Performance Overhead
**Problem**: Unit tracking adds overhead; GNNs are already compute-intensive

**Mitigation**:
- Use `DimTensor._from_tensor_and_unit()` to avoid unnecessary copies
- Profile and optimize hot paths
- Consider optional unit validation (validate_input=False for production)

### Edge Case 1: Self-Loops
**Problem**: Messages from node to itself (i → i)

**Handling**:
- Support optional self-loops in edge construction
- Physically meaningful for self-interactions (e.g., self-energy terms)

### Edge Case 2: Directed vs Undirected Graphs
**Problem**: Physical systems usually have symmetric interactions, but graphs can be directed

**Handling**:
- Default to undirected (add both i→j and j→i edges)
- Support directed graphs with a flag for advanced use cases
- Validate that symmetric forces produce symmetric edge lists

### Edge Case 3: Dynamic Graphs
**Problem**: Graphs change over time (particles move in/out of interaction range)

**Handling**:
- Support dynamic edge construction based on position updates
- Provide utility to rebuild graph from new positions
- Document pattern for simulation loops with graph updates

---

## Definition of Done

- [x] Research completed on GNN architectures for physics
- [x] Unit flow principles documented
- [x] Architecture design finalized
- [x] Implementation steps outlined
- [x] Testing strategy defined
- [x] Risks and edge cases identified
- [x] Plan document created at `.plans/2026-01-09_gnn-physics-research.md`
- [ ] Plan reviewed and approved
- [ ] Implementation task (#163) can proceed with clear guidance

---

## Notes / Log

**2026-01-09 - Initial Research**

Key findings:
1. **Message Passing is Core**: All physics GNNs use variants of message passing
2. **Unit Flow is Aggregation-Critical**: All aggregated messages must have same dimension
3. **Edge Features are Distinct**: Edge dimensions (distance [L]) are separate from node dimensions
4. **Equivariance ≠ Unit Tracking**: These are orthogonal concerns (geometric vs dimensional)
5. **SchNet Pattern is Dominant**: Distance-based weighting [L] → dimensionless is very common

Recommended implementation path:
1. Start with simple MPNN + unit tracking
2. Add distance-based edge networks (SchNet-style)
3. Later: Add equivariant extensions (E(3)-GNN) if needed

**Key Architectural Decision**:
Separate `edge_input_dim` from `input_dim` / `output_dim`. Edge features live in their own dimensional space and are used to generate dimensionless weights or to inform message generation, but do not directly determine message dimensions.

---

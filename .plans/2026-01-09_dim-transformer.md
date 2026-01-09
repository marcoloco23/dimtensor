# Plan: DimTransformer - Attention Mechanism with Unit Tracking

**Date**: 2026-01-09
**Status**: PLANNING
**Author**: planner-agent

---

## Goal

Implement transformer attention mechanism (multi-head attention, encoder layers) that tracks physical units through Query-Key-Value operations, enabling physics-aware sequence modeling for particle interactions, time series, and spatiotemporal dynamics.

---

## Background

### Why This Is Needed

Transformers have revolutionized ML with self-attention mechanisms, but existing implementations don't track physical dimensions. For physics applications like:

- **Particle interactions**: N-body systems where positions/velocities have units
- **Time series forecasting**: Weather, climate, stock prices with physical quantities
- **Spatiotemporal PDEs**: Heat equation, wave equation on spatial grids
- **Molecular dynamics**: Atom trajectories with forces/energies

We need attention that preserves unit safety while learning interactions.

### Existing Patterns

From `src/dimtensor/torch/layers.py`:
- **DimLayer base class**: Tracks `input_dim` and `output_dim`, validates on forward()
- **Pattern**: Subclass implements `_forward_impl(x: Tensor) -> Tensor` for raw computation
- **Pattern**: Public `forward(x: DimTensor | Tensor) -> DimTensor` handles unit validation
- **Construction**: Uses `DimTensor._from_tensor_and_unit(result, self._output_unit)`
- **Examples**: DimLinear (dimension transformation), DimConv1d/2d (convolution with units)

### Transformer Mechanics

Standard transformer attention:
```python
Q = x @ W_q  # Query projection
K = x @ W_k  # Key projection
V = x @ W_v  # Value projection

scores = Q @ K^T / sqrt(d_k)  # Attention scores
attn = softmax(scores)         # Attention weights (dimensionless!)
output = attn @ V              # Weighted sum of values
```

**Key constraint**: `softmax()` requires dimensionless input, so `Q @ K^T` must be dimensionless.

---

## Approach

### Option A: Same-Dimension QKV with Explicit Scaling

**Design**:
- Q, K, V all have same input dimension (e.g., all sequences have units of [m])
- Q @ K^T produces dimension [m²]
- Divide by characteristic scale squared to make dimensionless: `(Q @ K^T) / scale²`
- Characteristic scale: `sqrt(mean(K²))` computed per batch

**Pros**:
- Natural for physics: all sequence elements have same physical quantity
- Explicit about dimensionless conversion (no hidden magic)
- Works for any input dimension

**Cons**:
- Requires computing characteristic scale (small overhead)
- Scale computation may be unstable for very small/large values

### Option B: Conjugate Dimensions (Q and K inverses)

**Design**:
- Input has dimension D
- Q projection: D → D
- K projection: D → D⁻¹ (inverse dimension)
- V projection: D → output_dim
- Q @ K^T is automatically dimensionless

**Pros**:
- No runtime scaling computation
- Mathematically elegant

**Cons**:
- Unnatural for physics (why would keys have inverse units?)
- Hard to interpret: what does "inverse position" mean?
- Breaks symmetry of self-attention

### Option C: All Dimensionless

**Design**:
- Input must be dimensionless (or automatically converted)
- Q, K, V all dimensionless
- Output dimensionless

**Pros**:
- Simplest implementation
- Matches standard PyTorch transformers exactly

**Cons**:
- Loses dimensional safety (defeats the purpose!)
- User must manually non-dimensionalize
- No unit tracking through the model

### Decision: Option A (Same-Dimension with Explicit Scaling)

**Rationale**:
- Preserves physical intuition: all sequence elements have same units
- Maintains dimensional safety throughout forward pass
- Explicit scaling makes the dimensionless conversion clear
- Characteristic scale is a physics-meaningful quantity (typical magnitude)
- Follows dimtensor philosophy: never hide dimensional conversions

**Implementation details**:
- Characteristic scale: `scale = sqrt(mean(K * K) + eps)` (has units of K)
- Dimensionless scores: `scores = (Q @ K^T) / scale²`
- `scale²` has units [K²] = [Q²], cancels Q @ K^T dimension
- Output has units of V (value dimension)

---

## Implementation Steps

### Phase 1: Core Attention Mechanism

1. [x] Research existing patterns in torch/layers.py
2. [ ] Create `torch/attention.py` module
3. [ ] Implement `DimMultiheadAttention` class:
   - Inherits from `DimLayer`
   - Store `embed_dim`, `num_heads`, `dropout`
   - Create Q, K, V projection layers (DimLinear instances)
   - Create output projection layer
4. [ ] Implement `_compute_attention()` helper:
   - Compute Q, K, V from input
   - Calculate characteristic scale from K
   - Compute dimensionless attention scores
   - Apply softmax and dropout
   - Weighted sum with V
5. [ ] Implement `_forward_impl()`:
   - Call `_compute_attention()`
   - Apply output projection
   - Return result tensor

### Phase 2: Encoder Layer

6. [ ] Implement `DimTransformerEncoderLayer`:
   - Multi-head attention block
   - Feedforward block (2x DimLinear with activation)
   - Residual connections (requires same dimension!)
   - Layer normalization (DimLayerNorm)
7. [ ] Add dimension validation:
   - Verify input/output dims match for residual connection
   - Check attention input_dim = output_dim for self-attention

### Phase 3: Encoder Stack

8. [ ] Implement `DimTransformerEncoder`:
   - Stack of N encoder layers
   - Validate dimension consistency across layers
   - Optional final normalization layer

### Phase 4: Testing

9. [ ] Create `tests/test_transformer.py`
10. [ ] Add tests:
    - Basic attention forward pass with units
    - Multi-head attention splits heads correctly
    - Attention weights are dimensionless
    - Output has correct dimension (same as V)
    - Encoder layer with residual connections
    - Full encoder stack
    - Autograd through attention
    - Edge case: single head
    - Edge case: very small/large input values

### Phase 5: Documentation & Integration

11. [ ] Add docstrings with physics examples
12. [ ] Update `torch/__init__.py` exports
13. [ ] Add example to docs/guide/pytorch.md
14. [ ] Update CONTINUITY.md

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/torch/attention.py` | **CREATE**: DimMultiheadAttention, DimTransformerEncoderLayer, DimTransformerEncoder |
| `src/dimtensor/torch/__init__.py` | Add exports for attention classes |
| `tests/test_transformer.py` | **CREATE**: Comprehensive test suite (~20 tests) |
| `docs/guide/pytorch.md` | Add transformer example (particle interactions) |
| `CONTINUITY.md` | Update task #164 status |

---

## Testing Strategy

### Unit Tests

- [x] **Test: Basic attention forward pass**
  - Input: (batch, seq_len, embed_dim) with units [m]
  - Expected: Output with same shape and units [m]

- [ ] **Test: Attention weights dimensionless**
  - Mock softmax to inspect scores
  - Verify scores are DimTensor with dimensionless unit

- [ ] **Test: Multi-head attention**
  - Input: (10, 20, 512) with units [kg·m/s]
  - num_heads=8
  - Verify: Each head processes embed_dim/num_heads=64 dims

- [ ] **Test: Output dimension matches value dimension**
  - Q, K: dimension [L]
  - V: dimension [L·T⁻¹] (velocity)
  - Expected output: [L·T⁻¹]

- [ ] **Test: Encoder layer with residuals**
  - Input/output same dimension (required for residual)
  - Verify no dimension errors on x + attention(x)

- [ ] **Test: Autograd through attention**
  - Input with requires_grad=True
  - Compute loss, call backward()
  - Verify gradients exist for Q, K, V projections

### Integration Tests

- [ ] **Test: Particle N-body simulation**
  - Sequence of particle positions [m]
  - Transformer learns pairwise interactions
  - Verify output has position units

- [ ] **Test: Time series forecasting**
  - Temperature sequence [K]
  - Transformer predicts next timestep
  - Verify output has temperature units

### Edge Cases

- [ ] Single head (num_heads=1)
- [ ] Very large sequences (seq_len=10000)
- [ ] Very small values (scale near zero → add epsilon)
- [ ] Dimensionless input (should work, scale=1.0)

---

## Design Details

### DimMultiheadAttention Architecture

```python
class DimMultiheadAttention(DimLayer):
    """Multi-head attention with physical dimension tracking.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        input_dim: Physical dimension of input (Q, K, V all same)
        output_dim: Physical dimension of output (default: same as input)
        dropout: Dropout probability
        bias: Add bias to projections

    Forward:
        x: (batch, seq_len, embed_dim) DimTensor with input_dim
        Returns: (batch, seq_len, embed_dim) DimTensor with output_dim
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        input_dim: Dimension = DIMENSIONLESS,
        output_dim: Dimension = DIMENSIONLESS,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__(input_dim, output_dim)
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Projections (all same input/output dimension for self-attention)
        self.q_proj = DimLinear(embed_dim, embed_dim, input_dim, input_dim, bias=bias)
        self.k_proj = DimLinear(embed_dim, embed_dim, input_dim, input_dim, bias=bias)
        self.v_proj = DimLinear(embed_dim, embed_dim, input_dim, output_dim, bias=bias)

        # Output projection
        self.out_proj = DimLinear(embed_dim, embed_dim, output_dim, output_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.scale_epsilon = 1e-8

    def _forward_impl(self, x: Tensor) -> Tensor:
        # x: (batch, seq, embed_dim) - raw tensor
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V (these are DimTensors)
        Q = self.q_proj(DimTensor._from_tensor_and_unit(x, self._input_unit))
        K = self.k_proj(DimTensor._from_tensor_and_unit(x, self._input_unit))
        V = self.v_proj(DimTensor._from_tensor_and_unit(x, self._input_unit))

        # Reshape for multi-head: (batch, seq, heads, head_dim)
        Q = Q.data.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.data.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.data.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose: (batch, heads, seq, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute characteristic scale: sqrt(mean(K²)) per head
        # K has units [input_dim], scale has units [input_dim]
        K_squared_mean = (K * K).mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        scale_squared = K_squared_mean + self.scale_epsilon  # [input_dim²]

        # Attention scores: Q @ K^T / scale²
        # Q @ K^T has units [input_dim²], scale² has units [input_dim²]
        # Result is dimensionless
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, heads, seq, seq)
        scores_dimensionless = scores / scale_squared.sqrt()  # Dimensionless!

        # Softmax over keys (dimensionless input/output)
        attn_weights = F.softmax(scores_dimensionless, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum: attn @ V
        # attn is dimensionless, V has units [output_dim]
        # Result has units [output_dim]
        attn_output = torch.matmul(attn_weights, V)  # (batch, heads, seq, head_dim)

        # Reshape back: (batch, seq, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        # Output projection
        output = self.out_proj(DimTensor._from_tensor_and_unit(attn_output, self._output_unit))

        return output.data
```

### DimTransformerEncoderLayer

```python
class DimTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with dimension tracking.

    Components:
    - Multi-head self-attention
    - Feedforward network (2-layer MLP)
    - Residual connections (requires dim consistency!)
    - Layer normalization

    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        dim_feedforward: Feedforward network dimension
        dimension: Physical dimension (same for input/output due to residuals)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dimension: Dimension = DIMENSIONLESS,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention (input_dim = output_dim for residual)
        self.self_attn = DimMultiheadAttention(
            d_model, nhead, dimension, dimension, dropout
        )

        # Feedforward network
        self.linear1 = DimLinear(d_model, dim_feedforward, dimension, dimension)
        self.linear2 = DimLinear(dim_feedforward, d_model, dimension, dimension)

        # Layer norm (dimension-aware)
        self.norm1 = DimLayerNorm(d_model, dimension)
        self.norm2 = DimLayerNorm(d_model, dimension)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.dimension = dimension

    def forward(self, x: DimTensor) -> DimTensor:
        """Forward pass with residual connections.

        Args:
            x: Input (batch, seq, d_model) with self.dimension

        Returns:
            Output (batch, seq, d_model) with self.dimension
        """
        # Verify dimension
        if x.dimension != self.dimension:
            raise DimensionError(
                f"Expected input dimension {self.dimension}, got {x.dimension}"
            )

        # Self-attention block with residual
        attn_output = self.self_attn(x)
        x = x + self.dropout1(attn_output)  # Residual connection (same dims!)
        x = self.norm1(x)

        # Feedforward block with residual
        ff_output = self.linear2(self.activation(self.linear1(x)))
        x = x + self.dropout2(ff_output)  # Residual connection (same dims!)
        x = self.norm2(x)

        return x
```

---

## Risks / Edge Cases

### Risk 1: Scale Computation Instability

**Problem**: If K values are very small, `scale = sqrt(mean(K²))` could be near zero, causing division instability.

**Mitigation**:
- Add epsilon: `scale² = mean(K²) + 1e-8`
- Epsilon has same units as K², so result remains correct dimensionally
- Value chosen to be small enough not to affect normal operation

### Risk 2: Residual Connection Dimension Mismatch

**Problem**: Residual connections `x + f(x)` require same dimensions. If user specifies different input/output dims, will fail.

**Mitigation**:
- Document clearly: encoder layers require `input_dim = output_dim`
- Add validation in `__init__`: `assert input_dim == output_dim`
- Provide clear error message if violated

### Risk 3: Head Dimension Not Divisible

**Problem**: If `embed_dim % num_heads != 0`, reshaping fails.

**Mitigation**:
- Add assertion in `__init__`: `assert embed_dim % num_heads == 0`
- Clear error message: "embed_dim must be divisible by num_heads"

### Edge Case: Dimensionless Input

**Behavior**: Should work normally with scale=1.0 (dimensionless unit has scale 1.0).

**Verification**: Add test with dimensionless input to confirm.

### Edge Case: Very Long Sequences

**Behavior**: Memory usage O(seq_len²) for attention matrix. Standard PyTorch limitation.

**Mitigation**: Document memory requirements. Future work: Flash Attention integration.

### Edge Case: Autograd Through Scale Computation

**Problem**: Scale depends on K values, creates computation graph. Must preserve gradients.

**Verification**:
- Test backward pass
- Verify gradients flow to Q, K, V projection weights
- Check no detach() calls that break autograd

---

## Physics Applications

### Application 1: N-Body Particle Interactions

```python
# Particle positions in 3D space
positions = DimTensor(torch.randn(32, 100, 3), units.m)  # 32 batches, 100 particles

# Transformer learns pairwise interactions
encoder = DimTransformerEncoder(
    d_model=3,
    nhead=1,
    num_layers=4,
    dimension=Dimension(L=1),  # meters
)

# Output: updated positions (still in meters)
updated_positions = encoder(positions)
```

### Application 2: Time Series Forecasting

```python
# Temperature measurements over time
temps = DimTensor(torch.randn(16, 1000, 1), units.K)  # 16 series, 1000 timesteps

# Predict next timestep
model = DimTransformerEncoder(
    d_model=1,
    nhead=1,
    num_layers=6,
    dimension=Dimension(Theta=1),  # temperature
)

prediction = model(temps)[:, -1:, :]  # Last timestep prediction (still in K)
```

### Application 3: Spatiotemporal PDE

```python
# Heat equation on 2D grid over time
# Shape: (batch, time, height*width, 1) - flattened spatial grid
heat_field = DimTensor(torch.randn(8, 50, 64, 1), units.K)

# Learn spatial-temporal dynamics
model = DimTransformerEncoder(
    d_model=1,
    nhead=1,
    num_layers=8,
    dimension=Dimension(Theta=1),
)

predicted_field = model(heat_field)  # Still temperature [K]
```

---

## Definition of Done

- [ ] All implementation steps complete
- [ ] All tests pass (target: 20+ tests for attention)
- [ ] Autograd verified (gradients flow correctly)
- [ ] Docstrings complete with examples
- [ ] Exports added to `torch/__init__.py`
- [ ] Example added to docs/guide/pytorch.md
- [ ] CONTINUITY.md task #164 marked complete
- [ ] mypy passes with no new errors

---

## Notes / Log

**2026-01-09 (planner-agent)** - Created comprehensive plan for DimTransformer. Key design decision: use Option A (same-dimension QKV with explicit characteristic scaling) to preserve physical intuition while maintaining dimensional safety. Characteristic scale computed as `sqrt(mean(K²))` makes attention scores dimensionless before softmax.

**Architecture**:
- `DimMultiheadAttention`: Core attention with unit tracking
- `DimTransformerEncoderLayer`: Attention + feedforward with residuals
- `DimTransformerEncoder`: Stack of encoder layers

**Critical insight**: Residual connections require `input_dim = output_dim`, which means encoder layers preserve dimension throughout. For dimension transformation, user must add separate DimLinear projection layers before/after encoder stack.

**Testing priorities**:
1. Verify attention weights are dimensionless
2. Verify output has correct dimension (same as V)
3. Verify autograd through entire attention mechanism
4. Test with physics applications (particles, time series)

---

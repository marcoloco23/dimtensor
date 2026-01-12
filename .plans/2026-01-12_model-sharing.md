# Plan: Model Sharing Protocol

**Date**: 2026-01-12
**Status**: PLANNING
**Author**: planner-agent
**Task**: #251

---

## Goal

Enable sharing of trained physics ML models with complete unit metadata so they can be loaded and used correctly across different systems, with automatic unit validation at inference time.

---

## Background

### Why This is Needed

Currently, dimtensor has:
- `hub/registry.py` - Model metadata registry with `ModelInfo` dataclass
- `hub/cards.py` - `ModelCard` with training information
- Both support serialization (`.to_dict()`, `.from_dict()`)

However, there's no protocol for:
1. Saving model weights WITH unit metadata in a single package
2. Validating input/output units at inference time
3. Converting between unit systems automatically
4. Sharing models between PyTorch and JAX

Real-world need: A researcher trains a fluid dynamics model expecting velocity in m/s and pressure in Pa. Another researcher wants to use it with velocity in km/h and pressure in bar. The model should handle this automatically or fail with a clear error.

### Existing Patterns

**HuggingFace Transformers**:
- `model.save_pretrained(path)` - saves weights + config
- `model.from_pretrained(path)` - loads from path or hub
- `config.json` contains model architecture metadata

**PyTorch Hub**:
- `torch.hub.load()` - loads from GitHub repos
- `state_dict()` for weights, separate metadata

**dimtensor existing serialization**:
- `io/json.py`, `io/hdf5.py` - serialize DimArray with units
- `hub/registry.py` - model metadata (but no weight saving)
- `hub/cards.py` - training metadata

---

## Approach

### Option A: Extend ModelCard with Save/Load Methods

**Description**: Add `save_with_weights()` and `load_with_weights()` methods to `ModelCard` class.

**Pros**:
- Minimal changes to existing architecture
- ModelCard already has all metadata we need
- Natural extension of current pattern

**Cons**:
- Couples ModelCard (metadata) with model persistence (weights)
- Harder to support multiple frameworks (PyTorch, JAX) from one class

### Option B: New DimModelPackage Class

**Description**: Create a new class that wraps a model + ModelCard + framework-specific serialization.

**Pros**:
- Clean separation: metadata vs weights vs framework
- Easy to add new frameworks
- Can version the package format independently

**Cons**:
- More abstractions
- Need to decide on directory structure for packages

### Option C: Protocol-Based Design

**Description**: Define a `DimModel` protocol that models must implement (`.dim_input`, `.dim_output`, `.forward_with_validation()`). Save/load as standalone functions.

**Pros**:
- Most flexible - works with any model class
- Follows Python protocol pattern (like `__iter__`)
- No inheritance required

**Cons**:
- Requires modifying existing models to implement protocol
- More boilerplate for users

### Decision: Option B (DimModelPackage)

**Rationale**:
1. **Framework agnostic**: Can wrap PyTorch `nn.Module`, JAX `flax.linen.Module`, or even TensorFlow models
2. **Clean separation**: ModelCard for metadata, framework-specific serializer for weights
3. **Forward compatible**: Easy to add new serialization formats (safetensors, ONNX, etc.)
4. **Directory structure**: Natural to use directory-based package (like HuggingFace)

Package structure:
```
my-model/
├── model_card.json       # ModelCard metadata
├── weights.pt            # PyTorch state_dict
├── weights.safetensors   # Optional: safetensors format
└── config.json           # Framework-specific config
```

---

## Implementation Steps

### Phase 1: Core DimModelPackage Class

1. [ ] Create `hub/package.py` with `DimModelPackage` class
   - `__init__(model, card: ModelCard, framework: str)`
   - `save(path: Path)` - save package to directory
   - `load(path: Path)` - class method to load package
   - `validate_inputs(inputs: dict[str, DimTensor])` - check dimensions
   - `validate_outputs(outputs: dict[str, DimTensor])` - check dimensions

2. [ ] Implement PyTorch serialization
   - Save: `torch.save(model.state_dict(), "weights.pt")`
   - Load: `model.load_state_dict(torch.load("weights.pt"))`
   - Save model architecture info in `config.json`

3. [ ] Implement JAX serialization
   - Save: Use `flax.serialization.msgpack_serialize`
   - Load: Use `flax.serialization.msgpack_restore`
   - Handle pytree structure

4. [ ] Add unit conversion helpers
   - `convert_inputs_to_model_units(inputs, target_units)` - auto-convert
   - `convert_outputs_from_model_units(outputs, target_units)` - auto-convert

### Phase 2: Validation at Inference

5. [ ] Create `DimModelWrapper` class
   - Wraps any model with unit validation
   - `forward()` validates inputs, calls model, validates outputs
   - Works with both PyTorch and JAX

6. [ ] Implement dimension checking
   - Compare input dimensions against `ModelCard.info.input_dims`
   - Compare output dimensions against `ModelCard.info.output_dims`
   - Raise `DimensionError` with helpful message on mismatch

7. [ ] Add optional auto-conversion
   - If input has correct dimension but wrong unit (m/s vs km/h), auto-convert
   - Controlled by `auto_convert=True/False` flag
   - Log conversions for transparency

### Phase 3: Sharing Protocol

8. [ ] Add `export_to_hub()` method
   - Upload package to dimtensor hub (future: cloud storage)
   - For now: save to shared directory with unique ID
   - Generate shareable URL/path

9. [ ] Add `import_from_hub()` class method
   - Download package from hub
   - Verify checksum/signature
   - Cache locally (use existing cache system)

10. [ ] Add version management
    - Support semantic versioning (1.0.0, 1.1.0, etc.)
    - Allow loading specific versions
    - Warn on version mismatches

### Phase 4: Framework-Specific Enhancements

11. [ ] PyTorch integration
    - Support `torch.jit.script` models
    - Support `torch.compile` models
    - Handle device placement (CPU/CUDA/MPS)

12. [ ] JAX integration
    - Support JIT-compiled functions
    - Handle device placement (CPU/TPU/GPU)
    - Preserve JAX transformations (vmap, grad)

13. [ ] Optional: ONNX export
    - Export to ONNX format for cross-framework compatibility
    - Embed unit metadata in ONNX metadata

### Phase 5: CLI Integration

14. [ ] Add `dimtensor models save` command
    - CLI wrapper around `DimModelPackage.save()`
    - Interactive prompts for metadata if not provided

15. [ ] Add `dimtensor models load` command
    - Load and inspect model packages
    - Show ModelCard as markdown

16. [ ] Add `dimtensor models validate` command
    - Validate package integrity
    - Check all files present
    - Verify dimensions are self-consistent

---

## Files to Modify

| File | Change |
|------|--------|
| `src/dimtensor/hub/package.py` | NEW - DimModelPackage, DimModelWrapper |
| `src/dimtensor/hub/serializers.py` | NEW - Framework-specific serialization |
| `src/dimtensor/hub/validators.py` | NEW - Unit validation at inference |
| `src/dimtensor/hub/__init__.py` | Export new classes |
| `src/dimtensor/cli/models.py` | NEW - CLI commands for model management |
| `src/dimtensor/__main__.py` | Add models subcommand |
| `tests/test_hub_package.py` | NEW - Test model saving/loading |
| `tests/test_hub_validators.py` | NEW - Test validation logic |

---

## Testing Strategy

### Unit Tests

- [ ] **Test saving PyTorch model**
  - Save model with units
  - Verify all files created
  - Load and verify weights match

- [ ] **Test saving JAX model**
  - Save JAX function with units
  - Verify pytree structure preserved
  - Load and verify outputs match

- [ ] **Test dimension validation**
  - Pass inputs with wrong dimensions → should raise error
  - Pass inputs with correct dimensions → should pass
  - Error messages are clear and actionable

- [ ] **Test unit conversion**
  - Input in km/h, model expects m/s → auto-convert
  - Input in wrong dimension → error (not convertible)
  - Conversions are logged

- [ ] **Test version management**
  - Save multiple versions
  - Load specific version
  - Default to latest version

### Integration Tests

- [ ] **End-to-end PyTorch workflow**
  1. Train small physics model
  2. Create ModelCard
  3. Save with DimModelPackage
  4. Load in new process
  5. Run inference with validation

- [ ] **End-to-end JAX workflow**
  - Same as PyTorch but with JAX

- [ ] **Cross-framework loading**
  - Save PyTorch model
  - Convert to ONNX (if implemented)
  - Load in JAX (if possible)

### Manual Testing

- [ ] Save/load large model (>100MB) - performance
- [ ] Test with real physics models from examples/
- [ ] Test error messages are user-friendly

---

## Risks / Edge Cases

### Risk 1: Framework incompatibility
**Issue**: PyTorch model can't be loaded in JAX directly.
**Mitigation**:
- Document that framework must match
- Provide ONNX export as cross-framework bridge
- Clear error message if framework mismatch

### Risk 2: Unit metadata lost in torch.save()
**Issue**: `torch.save()` doesn't preserve custom metadata easily.
**Mitigation**:
- Save metadata separately in `model_card.json`
- Use directory-based package format (not single file)
- Consider safetensors format which supports metadata

### Risk 3: Large model packages
**Issue**: Models with weights can be GB-sized.
**Mitigation**:
- Support compression in save/load
- Use git-lfs or similar for hub storage
- Stream weights instead of loading all at once

### Risk 4: Dimension inference on raw tensors
**Issue**: Model outputs raw `torch.Tensor`, not `DimTensor`. How to attach units?
**Mitigation**:
- ModelCard specifies output dimensions
- Wrapper automatically wraps outputs in DimTensor
- Works for dict outputs: `{"velocity": tensor, "pressure": tensor}`

### Edge Case 1: Multi-output models
**Handling**: ModelCard has `output_dims: dict[str, Dimension]`. Wrapper expects model to return dict or tuple, matches by key or position.

### Edge Case 2: Optional inputs
**Handling**: Mark in ModelCard which inputs are optional. Validator only checks provided inputs.

### Edge Case 3: Dynamic dimensions
**Handling**: Some models accept variable dimensions (e.g., spatial dimensions L^1, L^2, or L^3). Use `None` or special marker in dimension spec.

### Edge Case 4: Batch dimensions
**Handling**: Validation ignores batch dimension (first dim). Only checks feature dimensions.

---

## Definition of Done

- [ ] DimModelPackage class implemented and tested
- [ ] PyTorch serialization works (save/load with validation)
- [ ] JAX serialization works (save/load with validation)
- [ ] Dimension validation at inference prevents wrong-unit inputs
- [ ] Auto-conversion works when enabled
- [ ] CLI commands for save/load/validate
- [ ] 50+ tests covering all scenarios
- [ ] Documentation in docs/guide/model-sharing.md
- [ ] Example notebook showing workflow
- [ ] CONTINUITY.md updated

---

## Notes / Log

### Key Design Decisions

1. **Directory-based packages** over single files
   - Easier to add new files (ONNX, metadata)
   - Standard pattern (HuggingFace)

2. **Wrapper pattern** for validation
   - Non-invasive: works with any existing model
   - Can be disabled if user wants raw performance

3. **Framework-specific serializers**
   - Clean abstraction
   - Easy to add TensorFlow, MXNet, etc. later

4. **Metadata in ModelCard**
   - Already has input_dims, output_dims
   - No need for new classes

### Future Enhancements (Post-v5.0.0)

- Hub service with REST API
- Model versioning with git
- Model provenance tracking
- Differential privacy guarantees
- Quantization-aware saving
- Distributed model loading (for huge models)

---

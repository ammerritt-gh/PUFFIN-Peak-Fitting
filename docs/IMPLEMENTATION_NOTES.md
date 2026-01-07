# Parameter Linking Implementation Notes

## Overview

This document provides technical implementation notes for the parameter linking feature added to PUFFIN.

## Architecture

The parameter linking feature follows the existing MVVM (Model-View-ViewModel) architecture:

```
Model (model_specs.py)
  ↓
  Stores link_group in Parameter objects
  ↓
ViewModel (fitter_vm.py)
  ↓
  Handles link_group updates and value propagation
  ↓
View (parameters_dock.py)
  ↓
  Displays link spinbox and visual indicators
```

## Key Design Decisions

### 1. Link Group Storage

**Decision**: Store `link_group` directly on `Parameter` objects.

**Rationale**: 
- Parameters already have metadata like `fixed`, `min`, `max`
- Natural extension of the Parameter class
- Automatically flows through `to_spec()` to the UI
- Preserved during parameter cloning in composite models

### 2. Link Group Range

**Decision**: Allow link groups 1-99 (0 = not linked).

**Rationale**:
- Sufficient for most use cases (rarely need >10 link groups)
- Small range prevents UI confusion
- 0 as "unlinked" is intuitive (default state)

### 3. Visual Feedback

**Decision**: Use colored borders on parameter value widgets.

**Rationale**:
- Clear visual indication without cluttering UI
- Different colors distinguish different link groups
- Matches existing UI styling patterns

**Color Palette**:
```python
["#FFD700", "#87CEEB", "#98FB98", "#FFB6C1", "#DDA0DD", "#F0E68C", "#E6E6FA", "#FFA07A"]
```
Cycling through 8 distinct colors for visibility.

### 4. Value Propagation Timing

**Decision**: Propagate values immediately in `apply_parameters()`.

**Rationale**:
- Real-time feedback when user changes values
- Consistent with "Fixed" checkbox behavior
- Simpler than deferred propagation

### 5. Fitting Optimization

**Decision**: Build link group map before fitting, optimize only representatives.

**Rationale**:
- Reduces parameter space (improves fit stability)
- Maintains exact equality of linked parameters
- Transparent to the optimizer (standard curve_fit interface)

**Implementation**:
```python
# Build representative mapping
link_representatives = {}  # param_name -> representative_name
for link_group, param_names in link_groups.items():
    representative = param_names[0]
    for pname in param_names:
        link_representatives[pname] = representative

# Only fit representatives
unique_free_keys = [rep for rep in set(link_representatives.values())]
```

## Integration Points

### Model Specs

The `Parameter` class in `model_specs.py` is the foundation:

```python
class Parameter:
    def __init__(self, ..., link_group=None):
        self.link_group = int(link_group) if link_group else None
```

Composite models inherit linking through parameter cloning:

```python
def _clone_parameter(param, new_name, value):
    return Parameter(
        ...,
        link_group=getattr(param, "link_group", None)
    )
```

### ViewModel

The `FitterViewModel` handles:

1. **Link group updates** via `__link` suffix:
```python
for k, v in params.items():
    if k.endswith("__link"):
        base = k[:-len("__link")]
        model_spec.params[base].link_group = int(v) if v else None
```

2. **Value propagation** during apply:
```python
if model_spec.params[k].link_group:
    for linked_name in linked_params:
        model_spec.params[linked_name].value = v
```

3. **Fit optimization** with link representatives:
```python
unique_free_keys = [rep for rep in set(link_representatives.values())]
```

### View

The `ParametersDock` creates UI elements:

```python
link_spin = QSpinBox()
link_spin.setRange(0, 99)
link_spin.setValue(link_val)
link_spin.setPrefix("Link: ")

# Visual indicator
if link_val > 0:
    color = link_colors[(link_val - 1) % len(link_colors)]
    widget.setStyleSheet(f"border: 2px solid {color};")
```

## Limitations and Future Work

### Current Limitations

1. **Composite Model Snapshots**: Link groups are saved, but full composite structure isn't
2. **Cross-Model Linking**: Parameters can only be linked within a single model state
3. **Type Restrictions**: Only numeric parameters should be linked (no validation yet)

### Future Enhancements

1. **Advanced Linking**: Support for ratio-based linking (e.g., param2 = 2 × param1)
2. **Link Groups UI**: Dedicated panel showing all link groups and their members
3. **Link Validation**: Warn when linking incompatible parameter types
4. **Link Names**: Allow naming link groups for better clarity

## Testing Strategy

### Test Hierarchy

1. **Unit Tests** (`test_parameter_linking.py`):
   - Parameter class with link_group
   - Composite model link propagation
   - Model evaluation with links

2. **UI Tests** (`test_ui_linking.py`):
   - Widget creation
   - Value display
   - Visual indicators

3. **Integration Tests** (`test_linking_integration.py`):
   - Value propagation through viewmodel
   - Fitting structure with links

4. **End-to-End Test** (`test_end_to_end.py`):
   - Complete workflow from setup to evaluation

### Coverage

- ✅ Model layer: Parameter class, cloning, composite models
- ✅ ViewModel layer: apply_parameters, run_fit, value propagation
- ✅ View layer: widget creation, visual feedback
- ✅ State persistence: snapshot/restore of link_groups

## Performance Considerations

**Impact**: Minimal

- Link group map building: O(n) where n = number of parameters
- Value propagation: O(k) where k = parameters in link group (typically 2-5)
- Fitting: Reduces parameter space, may improve convergence

**Memory**: Negligible (one integer per parameter)

## Backward Compatibility

✅ Fully backward compatible:

- Old snapshots without `link_groups` load correctly (defaults to None)
- Models without linked parameters work exactly as before
- No breaking changes to existing APIs

## Maintenance Notes

### When Adding New Parameter Types

Update these locations:
1. `Parameter.__init__()` - add to constructor signature
2. `_clone_parameter()` - preserve in cloning
3. `Parameter.to_spec()` - expose to UI
4. `ModelState.snapshot()` - save in state
5. `ModelState.load_from_snapshot()` - restore from state

### When Modifying Fitting Logic

Be aware that `run_fit()` in `fitter_vm.py`:
1. Builds link_representatives map
2. Creates unique_free_keys list
3. Wraps evaluate() to propagate values
4. Propagates fit results back to linked params

All four steps must remain synchronized.

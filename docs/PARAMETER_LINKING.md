# Parameter Linking Feature

## Overview

The parameter linking feature allows you to link multiple parameters together so they always have the same value and are fit as a single parameter during optimization. This is useful for scenarios like:

- Multiple peaks with the same width (e.g., two Gaussian peaks with identical FWHM)
- Shared resolution parameters across different components
- Any case where you want to enforce that certain parameters remain equal

## How to Use

### Linking Parameters

1. **Add components to your custom model** (e.g., two Gaussian peaks)
2. **Locate the "Link:" spinbox** next to each parameter in the Parameters dock
3. **Enter the same number** (1-99) in the Link spinbox for all parameters you want to link together
   - Use `0` or leave blank to unlink a parameter
   - Parameters with the same link number will be synchronized

### Visual Feedback

When parameters are linked:
- **Colored border**: Linked parameters display a colored border around the value widget
- **Same color = same link group**: Different link groups use different colors
- The color cycles through a palette: gold, sky blue, light green, pink, lavender, etc.

### Example: Two Gaussians with Linked Widths

```python
# In the UI:
# 1. Select "Custom Model" from the Model dropdown
# 2. Add two Gaussian components
# 3. Set Link: 1 for both elem1_Width and elem2_Width
# 4. Now changing either width will update both
```

## Behavior

### Value Synchronization

When you change a linked parameter:
- All parameters with the same link number are immediately updated to the same value
- This happens in real-time as you edit values

### During Fitting

When you run a fit:
- Linked parameters are treated as a **single free parameter**
- Only one parameter per link group is optimized
- The fitted value is automatically propagated to all linked parameters
- This reduces the parameter space and can improve fit stability

### Fixed Parameters

- Linked parameters can also be marked as "Fixed"
- When fixed, all parameters in the link group are excluded from fitting
- The "Fixed" checkbox takes precedence over linking

## Technical Details

### Model Layer

- Each `Parameter` object has a `link_group` attribute (integer or None)
- Parameters with the same non-zero `link_group` are linked
- `link_group=0` or `None` means not linked

### State Persistence

- Link groups are saved when you save your model state
- They are restored when you reload a saved state
- Stored in the `link_groups` section of the snapshot

### Fitting Implementation

- The fitting logic builds a map of link groups before optimization
- For each link group, one parameter is chosen as the "representative"
- Only the representative is passed to the optimizer
- Fitted values are propagated back to all linked parameters

## Limitations

- Link groups are identified by integers 1-99
- All linked parameters must have compatible types (e.g., all float)
- Linking is currently only supported for numeric parameters
- Link groups don't span across different model states (each model has its own link groups)

## Screenshot

![Parameter Linking Demo](parameter_linking_demo.png)

The screenshot shows a custom model with two Gaussian peaks where both `Width` parameters are linked with Link: 1. Notice the golden border around both width value spinboxes, indicating they are linked together.

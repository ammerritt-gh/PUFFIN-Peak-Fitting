# BigFit Interactive Peak Control - Quick Start Guide

## Overview
BigFit now supports interactive mouse controls for adjusting peak parameters in real-time. This guide shows you how to use these features with the Gaussian model.

## Getting Started

1. **Launch BigFit**
   ```bash
   cd BigFit
   python main.py
   ```

2. **Select the Gaussian Model**
   - Look for the "Model:" dropdown in the Parameters panel (right side)
   - Select "Gaussian" from the list

3. **Load or Use Default Data**
   - Click "Load Data" to load your data file, or
   - Use the default synthetic data that's pre-loaded

## Interactive Controls

### Selecting a Peak Parameter

**To enable interactive controls for a parameter:**
1. Click near the parameter's current value on the x-axis
2. The parameter name will appear in the log
3. The ViewBox panning will be disabled (you're now in parameter control mode)

**Example**: If Center = 0.0, click near x=0 on the plot

**Selection Threshold**: 
- The click must be within 2% of the data x-range of the parameter value
- Example: If your data spans x=[-10, 10] (range=20), threshold = 0.4 units

### Adjusting Parameters Interactively

Once a parameter is selected:

#### **Center Parameter** (Peak Position)
- **Control**: Click + Drag Horizontally
- **Effect**: Center follows your mouse x-coordinate
- **Example**: 
  - Select Center by clicking near x=0
  - Drag left → Center moves left
  - Drag right → Center moves right

#### **Width Parameter** (Peak Width/FWHM)
- **Control**: Ctrl + Mouse Wheel
- **Effect**: Width scales by 5% per scroll
- **Example**:
  - Select any parameter (or just have focus on plot)
  - Hold Ctrl, scroll up → Width increases (multiply by 1.05)
  - Hold Ctrl, scroll down → Width decreases (divide by 1.05)

#### **Area Parameter** (Peak Amplitude)
- **Control**: Mouse Wheel (no modifier keys)
- **Effect**: Area scales by 10% per scroll
- **Example**:
  - Select any parameter (or just have focus on plot)
  - Scroll up → Area increases (multiply by 1.1)
  - Scroll down → Area decreases (divide by 1.1)

### Deselecting / Returning to View Panning

**To exit parameter control mode:**
- Press the **Spacebar** key
- The log will show "Selection cleared"
- ViewBox panning and zooming are re-enabled

### View Panning and Zooming

**When no parameter is selected:**
- **Left Mouse Drag**: Pan the view
- **Right Mouse Drag**: Zoom (box zoom)
- **Mouse Wheel**: Zoom in/out
- **Right Click + Zoom**: Reset to auto-range

## Workflow Example

Here's a typical workflow for adjusting a Gaussian peak:

1. **Load your data**
   ```
   Click "Load Data" → Select your file
   ```

2. **Select Gaussian model**
   ```
   Model dropdown → "Gaussian"
   ```

3. **Roughly position the peak**
   ```
   Click near where you want the center
   Drag horizontally to move it to the right position
   ```

4. **Adjust the width**
   ```
   Hold Ctrl + Scroll to make the peak wider or narrower
   ```

5. **Adjust the amplitude/area**
   ```
   Scroll (no modifier) to make the peak taller or shorter
   ```

6. **Fine-tune with parameter panel**
   ```
   Use the spin boxes in the Parameters panel for precise values
   ```

7. **Run a fit**
   ```
   Click "Run Fit" to optimize all parameters
   ```

## Tips and Tricks

### Selection Not Working?
- **Check the threshold**: Click closer to the parameter value on the x-axis
- **Verify input hints**: The parameter must have drag or wheel controls defined
- **Check the log**: Selection attempts are logged

### Drag Not Responding?
- **Ensure a parameter is selected**: Look for "Selected: ..." in the log
- **Check you're dragging**: Hold left mouse button while moving
- **Only horizontal drag works**: Vertical movement is ignored for Center

### Wheel Not Working?
- **Check modifiers**: Width requires Ctrl key held
- **Scroll slowly**: Rapid scrolling may cause events to be missed
- **Check selection**: Some controls work better with a parameter selected

### Want Fine Control?
- **Use parameter panel**: Spin boxes allow precise value entry
- **Scroll incrementally**: Each scroll is a small change
- **Use keyboard**: Arrow keys in spin boxes for fine adjustments

### Want to Start Over?
- **Reset parameters**: 
  1. Change model to something else
  2. Change back to Gaussian
  3. Parameters reset to defaults (area=1, width=1, center=0)

## Parameter Panel Integration

The parameter panel works alongside interactive controls:

- **Spin Boxes**: Show current values, update during interaction
- **Auto-Apply**: Changes in spin boxes immediately update the plot
- **Tooltips**: Hover over parameters to see control hints
- **Apply Button**: Manually apply changes if auto-apply is disabled
- **Refresh Button**: Reload parameters from the model

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Clear selection / Return to view panning |
| F | Run fit (when parameters selected) |
| R | Reset view (auto-range) |
| U | Update plot |

## Model-Specific Notes

### Gaussian Model
- **Default Values**: area=1, width=1, center=0
- **Area**: Integrated area under the curve (not peak height)
- **Width**: Full Width at Half Maximum (FWHM), not standard deviation
- **Center**: Peak position on x-axis

### Other Models
- **Voigt**: Also supports interactive controls for center, widths, and area
- **DHO+Voigt**: Complex model with phonon energy, damping, resolution
- **Lorentzian**: Similar to Gaussian, different line shape

## Troubleshooting

### "No selection" but I clicked
**Solution**: Click closer to the parameter value. The threshold is 2% of x-range.

### Drag moves view instead of parameter
**Solution**: Parameter wasn't selected. Click near the value first.

### Ctrl+Wheel does nothing
**Solution**: 
1. Ensure Width parameter exists in current model
2. Check that plot window has focus
3. Try clicking the plot first

### Parameters jump unexpectedly
**Solution**: Multiple parameters may be mapped to the same control. Use more specific selection.

### View stuck in selection mode
**Solution**: Press Spacebar to force clear selection.

## Advanced Usage

### Defining Custom Interactive Controls

To add interactive controls to a new parameter:

```python
Parameter("my_param", value=1.0, ptype="float",
          input_hint={
              "drag": {"action": "set", "value_from": "x"},
              "wheel": {"modifiers": ["Ctrl"], "action": "scale", "factor": 1.05}
          })
```

Action types:
- `"set"`: Direct assignment from coordinate
- `"scale"`: Multiplicative change
- `"increment"`: Additive change

### Multiple Models/Peaks

For multi-peak models:
1. Parameters are named with indices (e.g., "Center_0", "Center_1")
2. Click near each peak's position to select its parameters
3. Only one parameter can be selected at a time

## Reference

### Complete Control Summary

| Parameter | Control | Modifier | Action | Factor/Step |
|-----------|---------|----------|--------|-------------|
| Center | Drag | None | Set from x | N/A |
| Width | Wheel | Ctrl | Scale | 1.05 |
| Area | Wheel | None | Scale | 1.1 |

### Selection States

| State | Mouse Behavior | Wheel Behavior |
|-------|----------------|----------------|
| No selection | View pan/zoom | View zoom |
| Parameter selected | Drag updates param | Adjusts param (if mapped) |

## Getting Help

If you encounter issues:
1. Check the log panel (bottom) for messages
2. Verify parameter has `input_hint` in model specification
3. Try pressing Spacebar to reset state
4. Restart the application if needed

For development questions or to report bugs, see the main BigFit documentation.

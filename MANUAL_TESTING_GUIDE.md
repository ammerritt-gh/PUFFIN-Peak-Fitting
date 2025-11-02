# Manual Testing Guide for Interactive Peak Controls

## Overview
This guide provides step-by-step instructions for manually testing the interactive peak control features implemented for the Gaussian model in BigFit.

## Implementation Summary

### Changes Made
**File**: `BigFit/models/model_specs.py` (lines 374-382)

Three parameters in `GaussianModelSpec` were enhanced with `input_hint` configurations:

1. **Area Parameter** (line 375-376):
   ```python
   input_hint={"wheel": {"action": "scale", "factor": 1.1}}
   ```
   - Mouse wheel (no modifier) scales Area by 10%
   - Scroll up: multiply by 1.1
   - Scroll down: divide by 1.1

2. **Width Parameter** (lines 377-379):
   ```python
   input_hint={"wheel": {"modifiers": ["Ctrl"], "action": "scale", "factor": 1.05}}
   ```
   - Ctrl+wheel scales Width by 5%
   - Ctrl+scroll up: multiply by 1.05
   - Ctrl+scroll down: divide by 1.05

3. **Center Parameter** (lines 380-382):
   ```python
   input_hint={"drag": {"action": "set", "value_from": "x"}}
   ```
   - Horizontal drag sets Center to mouse x-coordinate
   - Drag left/right: Center follows mouse

### Default Values
The Gaussian model has these defaults (as specified in requirements):
- Area: 1.0
- Width: 1.0
- Center: 0.0

### Existing Infrastructure Used
The implementation leverages the existing MVVM architecture:
- **InputHandler** (`view/input_handler.py`): Captures mouse/keyboard events
- **MainWindow** (`view/main_window.py`): Handles selection logic
- **FitterViewModel** (`viewmodel/fitter_vm.py`): Processes parameter updates

## Prerequisites

### System Requirements
- Python 3.8 or higher
- PySide6 (Qt6 for Python)
- numpy
- scipy
- pyqtgraph

### Installation
```bash
# Install dependencies (if not already installed)
pip install PySide6 numpy scipy pyqtgraph
```

## Testing Procedure

### 1. Launch the Application

```bash
cd /path/to/BigFit/BigFit
python main.py
```

The BigFit window should open with:
- Central plot area
- Left panel: Controls and buttons
- Right panel: Parameters
- Bottom panel: Log messages

### 2. Select the Gaussian Model

1. In the **Parameters** panel (right side), find the "Model:" dropdown
2. Click the dropdown and select **"Gaussian"** from the list
3. Verify in the log panel (bottom) that the model was changed

**Expected log message**: "Model switched to: Gaussian" or similar

### 3. Verify Default Parameter Values

In the Parameters panel, check that the Gaussian parameters have these values:
- **Area**: 1.0
- **Width**: 1.0
- **Center**: 0.0

**✓ Requirement Met**: Default Gaussian with area=1, width=1, center=0

### 4. Test Peak Selection (Simple Click)

**Objective**: Click near the Center parameter value to select it

**Steps**:
1. Look at the plot - the Gaussian peak should be centered at x=0 (since Center=0.0)
2. Click with the **left mouse button** near x=0 on the plot
3. Check the log panel for a selection message

**Expected results**:
- Log message: "Selection started for parameter: Center" or similar
- The ViewBox mouse interactions should be disabled (panning won't work)

**✓ Requirement Met**: Simple clicks select peaks

**Note**: The selection threshold is 2% of the x-range. If your data spans x=[-10, 10], you need to click within 0.4 units of Center's value.

### 5. Test Horizontal Drag to Change Center

**Objective**: Drag horizontally to move the peak

**Precondition**: Center parameter must be selected (from step 4)

**Steps**:
1. With Center selected, click and hold the left mouse button on the plot
2. Drag horizontally (left and right)
3. Release the mouse button
4. Observe the peak position and the Center parameter value in the panel

**Expected results**:
- As you drag, the Center parameter value updates in real-time
- The Gaussian peak moves horizontally following your mouse
- The plot refreshes continuously during the drag
- Log messages show parameter updates

**✓ Requirement Met**: Horizontal mouse drag changes peak center

**Troubleshooting**:
- If dragging doesn't work, make sure Center is selected (step 4)
- If the peak doesn't move, check that the plot is refreshing
- Verify in the log that "Interactive: wheel scaled Center" or similar messages appear

### 6. Test Spacebar to Deselect

**Objective**: Press spacebar to clear selection and re-enable panning

**Precondition**: A parameter is selected

**Steps**:
1. Ensure a parameter is selected (Center from previous steps)
2. Press the **Spacebar** key
3. Check the log panel for a deselection message
4. Try to pan the plot by dragging

**Expected results**:
- Log message: "Selection cleared" or "Selection ended"
- ViewBox mouse interactions are re-enabled
- You can now pan and zoom the plot normally
- The parameter is no longer highlighted

**✓ Requirement Met**: Spacebar deselects peaks

### 7. Test Scroll Wheel to Change Area

**Objective**: Use mouse wheel (no modifiers) to adjust Area

**Steps**:
1. Ensure the mouse cursor is over the plot area
2. **Without holding any keys**, scroll the mouse wheel up
3. Observe the Area parameter value in the Parameters panel
4. Scroll the mouse wheel down
5. Observe the Area parameter value again

**Expected results**:
- Scrolling up: Area increases (multiplied by 1.1)
  - Example: 1.0 → 1.1 → 1.21 → 1.331 → ...
- Scrolling down: Area decreases (divided by 1.1)
  - Example: 1.0 → 0.909 → 0.826 → ...
- The Gaussian peak height changes accordingly
- Log messages show "Interactive: wheel scaled Area -> [value]"

**✓ Requirement Met**: Scroll (no modifier) changes area

**Note**: You don't need to select a parameter first for wheel controls to work on Area.

### 8. Test Ctrl+Scroll to Change Width

**Objective**: Use Ctrl+wheel to adjust Width

**Steps**:
1. Ensure the mouse cursor is over the plot area
2. Hold down the **Ctrl** key (or **Cmd** on Mac)
3. Scroll the mouse wheel up while holding Ctrl
4. Observe the Width parameter value in the Parameters panel
5. Scroll the mouse wheel down while holding Ctrl
6. Observe the Width parameter value again

**Expected results**:
- Ctrl+Scroll up: Width increases (multiplied by 1.05)
  - Example: 1.0 → 1.05 → 1.1025 → ...
- Ctrl+Scroll down: Width decreases (divided by 1.05)
  - Example: 1.0 → 0.952 → 0.907 → ...
- The Gaussian peak width changes accordingly
- Log messages show "Interactive: wheel scaled Width -> [value]"

**✓ Requirement Met**: Ctrl+scroll changes width

**Troubleshooting**:
- Make sure you're holding Ctrl before scrolling
- On Mac, try Cmd if Ctrl doesn't work
- Check the log for "Ctrl+Wheel" messages

### 9. Test Default to View Panning (No Selection)

**Objective**: Verify that without selection, plot reverts to normal panning

**Steps**:
1. Press Spacebar to clear any selection (if needed)
2. Click and drag on the plot
3. Verify that the view pans (moves the visible region)
4. Right-click and drag to create a zoom box
5. Verify that the view zooms to the selected region

**Expected results**:
- Left-click drag: Pan the plot
- Right-click drag: Box zoom
- Mouse wheel: Zoom in/out
- No parameter selection occurs
- Normal PyQtGraph ViewBox behavior

**✓ Requirement Met**: Without selection, defaults to view panning

### 10. Test Click+Drag Only Works with Selected Peak

**Objective**: Verify drag only affects parameters when selected

**Steps**:
1. Clear any selection (Spacebar)
2. Click and drag on the plot
   - **Expected**: View pans (no parameter change)
3. Click near x=0 to select Center
4. Click and drag on the plot
   - **Expected**: Center parameter changes (no panning)
5. Press Spacebar to deselect
6. Click and drag again
   - **Expected**: View pans again

**✓ Requirement Met**: Click+drag only works with selected peak

### 11. Test Separation of Click from Drag

**Objective**: Verify that simple clicks select, while drag modifies

**Steps**:
1. Clear selection (Spacebar)
2. **Single click** near x=0 (don't drag)
   - **Expected**: Center is selected, but value doesn't change
3. **Click and drag** horizontally
   - **Expected**: Center value changes during drag
4. **Single click** away from any parameter value
   - **Expected**: Selection is cleared (or nothing happens)

**✓ Requirement Met**: Separate simple clicks from click+drag interactions

## Comprehensive Test Checklist

Use this checklist to verify all requirements:

- [ ] Default Gaussian model has area=1, width=1, center=0
- [ ] Simple click near Center value selects the parameter
- [ ] Horizontal drag changes Center when selected
- [ ] Mouse wheel (no modifier) changes Area
- [ ] Ctrl+wheel changes Width
- [ ] Spacebar clears selection
- [ ] Without selection, dragging pans the view
- [ ] With selection, dragging modifies parameter (not view)
- [ ] Simple clicks don't modify parameters (only drag does)
- [ ] Log panel shows appropriate messages for all actions

## Troubleshooting

### Mouse Events Not Working
- Ensure the plot widget has focus (click on it once)
- Check that you're clicking within the plot boundaries
- Look for error messages in the log panel

### Selection Not Working
- Verify you're clicking close enough to the parameter value
- Selection threshold is 2% of x-range
- Check log for "Selection started" messages

### Drag Not Changing Parameters
- Ensure the parameter is selected first
- Verify Center has a drag input_hint (check implementation)
- Check log for "Interactive: wheel scaled" or similar messages

### Wheel Controls Not Working
- Make sure modifiers are pressed before scrolling
- Verify Area/Width have wheel input_hints
- Check that plot has focus

### View Panning Not Working After Selection
- Press Spacebar to clear selection
- Check log for "Selection ended" message
- Verify ViewBox mouse is re-enabled

## Expected Log Messages

During testing, you should see messages like:
- "Model switched to: Gaussian"
- "Selection started for parameter: Center"
- "Interactive: wheel scaled Area -> 1.1000"
- "Interactive: wheel scaled Width -> 1.0500"
- "Selection ended" or "Selection cleared"
- "Applied parameters: Area" or similar

## Success Criteria

All 11 tests pass, demonstrating:
1. ✓ Default Gaussian configuration
2. ✓ Peak selection with mouse click
3. ✓ Horizontal drag modifies Center
4. ✓ Wheel modifies Area
5. ✓ Ctrl+wheel modifies Width
6. ✓ Spacebar clears selection
7. ✓ View panning without selection
8. ✓ Parameter modification with selection
9. ✓ Click vs drag distinction
10. ✓ Interactive controls properly separated from view controls
11. ✓ All requirements from problem statement met

## Additional Notes

### Architecture Benefits
This implementation leverages the existing MVVM architecture:
- **Declarative**: Parameters declare their controls via `input_hint`
- **Reusable**: Same pattern works for any model/parameter
- **Maintainable**: No hardcoded bindings in UI code
- **Extensible**: Easy to add new input types or modifiers

### Future Enhancements
Possible extensions:
- Shift+wheel for different parameters
- Alt+drag for vertical parameter control
- Multi-parameter selection with click on multiple peaks
- Visual indicators for selected parameters
- Undo/redo for parameter changes

## Screenshots

*Note: Screenshots should be taken during manual testing to document:*
1. Initial Gaussian peak at center=0
2. Peak after dragging to new center
3. Parameter panel showing updated values
4. Log panel showing interaction messages

## Conclusion

If all tests pass, the implementation successfully meets all requirements:
- Default Gaussian with specified parameters
- Mouse click+drag for horizontal peak movement
- Scroll wheel for area adjustment
- Ctrl+scroll for width adjustment
- Spacebar to deselect
- Proper separation of selection and modification
- Seamless integration with existing view controls

The system is ready for use and can be extended to other models using the same pattern.

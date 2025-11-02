# Gaussian Peak Interaction Implementation

## Overview
This document describes the implementation of mouse click+drag interactions for peak parameters in BigFit, specifically for the Gaussian model.

## Requirements Met

### 1. Default Gaussian Model ✓
- **Requirement**: Default fit should be a Gaussian with area=1, width=1, center=0
- **Implementation**: `GaussianModelSpec` in `models/model_specs.py` (lines 371-389)
  - Area: `value=1.0`
  - Width: `value=1.0`
  - Center: `value=0.0`

### 2. Horizontal Mouse Drag Changes Peak Center ✓
- **Requirement**: Horizontal mouse drag changes peak center
- **Implementation**: 
  - `GaussianModelSpec.Center` parameter has `input_hint={"drag": {"action": "set", "value_from": "x"}}`
  - `FitterViewModel.handle_plot_mouse_move()` processes drag events (lines 637-767)
  - Drag action sets the Center parameter to the x-coordinate of the mouse

### 3. Ctrl+Scroll Changes Width ✓
- **Requirement**: Ctrl+scroll changes width
- **Implementation**:
  - `GaussianModelSpec.Width` parameter has `input_hint={"wheel": {"modifiers": ["Ctrl"], "action": "scale", "factor": 1.05}}`
  - `FitterViewModel.handle_wheel_scroll()` processes wheel events with modifier checking (lines 803-891)
  - Ctrl+wheel scales Width by factor 1.05 (up) or 1/1.05 (down)

### 4. Scroll Changes Area ✓
- **Requirement**: Scroll (no modifier) changes area
- **Implementation**:
  - `GaussianModelSpec.Area` parameter has `input_hint={"wheel": {"action": "scale", "factor": 1.1}}`
  - `FitterViewModel.handle_wheel_scroll()` handles wheel without modifiers
  - Wheel scales Area by factor 1.1 (up) or 1/1.1 (down)

### 5. Click+Drag Only Works with Selected Peak ✓
- **Requirement**: Click+drag only works with a selected peak
- **Implementation**:
  - `FitterViewModel.begin_selection()` starts a selection session (lines 391-409)
  - `FitterViewModel._interactive_drag_info` tracks active selection and filters handlers to selected parameter
  - `MainWindow._try_select_param()` attempts to select a parameter near click location (lines 761-837)
  - Only drag handlers for the selected parameter are active during drag

### 6. Without Selection, Defaults to View Panning ✓
- **Requirement**: Without selection, defaults to view panning
- **Implementation**:
  - `MainWindow._set_selection_active()` disables ViewBox mouse when parameter selected (lines 955-968)
  - `MainWindow._clear_selection()` re-enables ViewBox mouse interactions (lines 970-1023)
  - When no parameter selected, ViewBox handles mouse for pan/zoom

### 7. Simple Clicks Select Peaks ✓
- **Requirement**: Simple clicks select peaks
- **Implementation**:
  - `MainWindow._on_plot_clicked()` handles click events (lines 702-759)
  - `MainWindow._try_select_param()` finds nearby parameters with drag handlers (lines 761-837)
  - Click within threshold (2% of x-range) of a parameter's value selects it
  - Selection criteria: parameter must have drag input_hint and numeric value

### 8. Spacebar Deselects Peaks ✓
- **Requirement**: Spacebar deselects peaks
- **Implementation**:
  - `MainWindow._on_plot_key_pressed()` handles keyboard (lines 1024-1050)
  - `Qt.Key_Space` triggers `_clear_selection()`
  - Clears selection state and re-enables view panning

## Architecture

The implementation follows the MVVM (Model-View-ViewModel) pattern:

```
┌─────────────────────────────────────────────────────────┐
│                    View Layer                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │ MainWindow (main_window.py)                      │  │
│  │ - GUI presentation                               │  │
│  │ - Event routing                                  │  │
│  │ - Selection visualization                        │  │
│  └──────────────────────────────────────────────────┘  │
│            ↑                                ↓            │
│  ┌──────────────────────────────────────────────────┐  │
│  │ InputHandler (input_handler.py)                  │  │
│  │ - Mouse/keyboard event capture                   │  │
│  │ - Signal emission (clicked, moved, wheel, keys)  │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                        ↓ Signals
┌─────────────────────────────────────────────────────────┐
│                 ViewModel Layer                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ FitterViewModel (fitter_vm.py)                   │  │
│  │ - Business logic                                 │  │
│  │ - Input map building (_build_input_map)         │  │
│  │ - Drag/wheel handlers                            │  │
│  │ - Selection lifecycle (begin/end)                │  │
│  │ - Parameter updates                              │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                   Model Layer                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │ GaussianModelSpec (model_specs.py)               │  │
│  │ - Parameter definitions with input_hint          │  │
│  │ - Default values (area=1, width=1, center=0)     │  │
│  │ - Evaluation function                            │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ ModelState (model_state.py)                      │  │
│  │ - Current dataset                                │  │
│  │ - Active model and parameters                    │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Parameter Input Hints
Parameters use structured `input_hint` dictionaries to declare interactive controls:

```python
Parameter("Center", value=0.0, ptype="float",
          hint="Peak center (x-axis)",
          input_hint={"drag": {"action": "set", "value_from": "x"}})
```

Supported event types:
- `drag`: Mouse click+drag actions
- `wheel`: Mouse wheel with optional modifiers
- `hotkey`: Keyboard shortcuts (extensible)

### 2. Input Map
The ViewModel builds an `_input_map` from parameter specs:

```python
{
  "drag": [
    {"param": "Center", "action": {"action": "set", "value_from": "x"}}
  ],
  "wheel": [
    {"param": "Area", "action": {"action": "scale", "factor": 1.1}},
    {"param": "Width", "action": {"modifiers": ["Ctrl"], "action": "scale", "factor": 1.05}}
  ]
}
```

### 3. Selection System
Selection workflow:
1. User clicks near a parameter value (within 2% of x-range threshold)
2. `_try_select_param()` finds closest parameter with drag handler
3. `begin_selection()` sets `_selected_param` and builds filtered handler list
4. ViewBox mouse interactions disabled (pan/zoom off)
5. Drag events only affect the selected parameter
6. Spacebar or explicit deselect clears selection
7. ViewBox mouse re-enabled for panning

### 4. Event Flow

**Click Event:**
```
User clicks plot
  → InputHandler._on_mouse_click()
  → Signal: mouse_clicked(x, y, button)
  → MainWindow._on_plot_clicked()
  → MainWindow._try_select_param()
  → FitterViewModel.begin_selection()
  → ViewBox mouse disabled
```

**Drag Event (with selection):**
```
User drags mouse
  → InputHandler._on_mouse_move()
  → Signal: mouse_moved(x, y, buttons)
  → MainWindow._on_plot_mouse_moved()
  → FitterViewModel.handle_plot_mouse_move()
  → Process drag handlers for selected param
  → Update parameter value
  → Update plot
```

**Wheel Event:**
```
User scrolls wheel
  → InputHandler.eventFilter() catches QWheelEvent
  → Signal: wheel_scrolled(delta, modifiers)
  → MainWindow._on_plot_wheel_scrolled()
  → FitterViewModel.handle_wheel_scroll()
  → Match modifiers, find handlers
  → Scale/increment parameter
  → Update plot
```

**Deselect (Spacebar):**
```
User presses Space
  → InputHandler._on_key_press()
  → Signal: key_pressed(key, modifiers)
  → MainWindow._on_plot_key_pressed()
  → MainWindow._clear_selection()
  → FitterViewModel.end_selection()
  → ViewBox mouse re-enabled
```

## Testing

To verify the implementation:

1. **Start the application**:
   ```bash
   cd BigFit
   python main.py
   ```

2. **Select Gaussian model** from the model dropdown

3. **Verify defaults** in the parameter panel:
   - Area = 1.0
   - Width = 1.0
   - Center = 0.0

4. **Test interactions**:
   - Click near Center value (0.0) on x-axis → parameter selected
   - Drag horizontally → Center follows mouse x-coordinate
   - Ctrl+Scroll → Width scales up/down
   - Scroll (no modifier) → Area scales up/down
   - Spacebar → selection cleared, view panning restored

5. **Test view panning**:
   - Without selection, drag should pan the view
   - Mouse wheel should zoom (unless parameter selected)

## Files Modified

### `/home/runner/work/BigFit/BigFit/BigFit/models/model_specs.py`
- **Lines 371-389**: Updated `GaussianModelSpec` class
- Added `input_hint` to Area, Width, and Center parameters
- Configured wheel/drag actions per requirements

**Changes:**
```python
# Area: scroll (no modifier) to scale
input_hint={"wheel": {"action": "scale", "factor": 1.1}}

# Width: Ctrl+scroll to scale
input_hint={"wheel": {"modifiers": ["Ctrl"], "action": "scale", "factor": 1.05}}

# Center: horizontal drag to set from x
input_hint={"drag": {"action": "set", "value_from": "x"}}
```

## Extensibility

The implementation is designed to be extensible:

1. **New Parameters**: Add parameters with `input_hint` to any ModelSpec
2. **New Actions**: Extend action types in `FitterViewModel` handlers
3. **New Models**: Any model can use the same interaction pattern
4. **Custom Controls**: Add new event types (e.g., double-click, right-click)

Example - adding a background parameter with Alt+wheel control:
```python
self.add(Parameter("background", value=0.0, ptype="float",
                   hint="Constant background offset",
                   input_hint={"wheel": {"modifiers": ["Alt"], 
                                        "action": "increment", 
                                        "step": 0.1}}))
```

## Best Practices

1. **Use structured input_hint**: Always use dict format for actionable hints
2. **Match parameter type**: `drag` for position-like params, `wheel` for scalars
3. **Choose appropriate actions**:
   - `set`: Direct value assignment (positions, centers)
   - `scale`: Multiplicative (widths, amplitudes, positive values)
   - `increment`: Additive (backgrounds, offsets, any sign)
4. **Threshold tuning**: Adjust selection threshold (currently 2% of x-range) if needed
5. **Visual feedback**: Consider adding visual markers for selected parameters

## Known Limitations

1. **Multi-peak models**: Current implementation selects single parameters; multi-peak models need extended selection UI
2. **Y-coordinate drag**: Currently only x-coordinate used; vertical drag could control amplitude
3. **Touch input**: Not tested with touch/tablet input
4. **Undo/redo**: No undo system for interactive parameter changes

## Future Enhancements

1. **Visual parameter markers**: Draw vertical lines at parameter positions
2. **Parameter value display**: Show current value during drag
3. **Snap to data**: Option to snap Center to nearest data point
4. **Keyboard fine-tune**: Arrow keys for small adjustments
5. **Parameter linking**: Link related parameters (e.g., multi-peak centers)
6. **History tracking**: Save parameter change history for undo

## Conclusion

The implementation successfully integrates mouse click+drag interactions for Gaussian peak parameters in BigFit. The architecture leverages the existing MVVM pattern and provides a clean, extensible framework for interactive parameter control across all model types.

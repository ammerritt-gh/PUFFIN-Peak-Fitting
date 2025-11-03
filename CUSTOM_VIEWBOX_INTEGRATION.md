# Custom ViewBox Integration Documentation

## Overview

The custom viewbox module (`view/custom_viewbox.py`) provides an enhanced `InteractiveViewBox` class that extends PyQtGraph's standard `ViewBox` with interactive features for data manipulation and selection. This module was extracted and generalized from the `PeakViewBox` pattern in `PySide_Fitter_PyQtGraph_drag.py`.

## Features

### 1. Object Selection
- Click on plot elements (curves, markers, data points) to select them
- Visual feedback for selected objects
- Clear selection by clicking empty space

### 2. Object Dragging
- Drag selected objects to modify their position
- Dragging is context-aware (only works when object is selected)
- Falls back to default panning when no object is selected

### 3. Exclude Mode
- Toggle individual data points in/out of the dataset
- Click points to toggle their inclusion
- Visual distinction between included and excluded points

### 4. Box Selection
- In exclude mode, drag to draw a rubber-band rectangle
- All points within the box are toggled (included ↔ excluded)
- Semi-transparent orange overlay during selection

### 5. Conditional Wheel Behavior
- Wheel zooming is blocked when:
  - An object is selected (prevents accidental zoom during parameter adjustment)
  - Exclude mode is active (keeps view stable during point selection)
- Default zoom behavior otherwise

## Architecture

### Host Interface

The `InteractiveViewBox` requires a "host" object (typically the main window) that provides:

#### Required Attributes
```python
class Host:
    exclude_mode: bool          # Whether exclude mode is active
    selected_kind: str | None   # Type of selected object ('fit', 'data', 'peak', etc.)
    selected_obj: object | None # Reference to the selected object
    energy: np.ndarray         # X data values (for compatibility)
    counts: np.ndarray         # Y data values (for compatibility)
    excluded_mask: np.ndarray  # Boolean mask for excluded points
```

#### Required Methods
```python
def _toggle_nearest_point_exclusion_xy(self, x: float, y: float):
    """Toggle exclusion of the nearest data point to (x, y)."""
    pass

def _nearest_target_xy(self, x: float, y: float) -> tuple[str | None, object | None]:
    """
    Find nearest selectable target near (x, y).
    
    Returns:
        (kind, obj): Type and reference of target, or (None, None)
    """
    pass

def set_selected(self, kind: str | None, obj: object | None):
    """Set the currently selected object."""
    pass

def _update_data_plot(self, do_range: bool = True):
    """Update the data plot after exclusion changes."""
    pass

def update_previews(self):
    """Update preview overlays during object drag."""
    pass
```

## Integration in BigFit

### MainWindow Integration

The custom viewbox is integrated into `MainWindow` as follows:

```python
# view/main_window.py

from .custom_viewbox import InteractiveViewBox

class MainWindow(QMainWindow):
    def __init__(self, viewmodel=None):
        # ... initialization ...
        
        # Initialize host interface attributes
        self.exclude_mode = False
        self.selected_kind = None
        self.selected_obj = None
        self.energy = None
        self.counts = None
        self.excluded_mask = None
        
        # Create custom viewbox and plot widget
        self.custom_viewbox = InteractiveViewBox(host=self)
        self.plot_widget = pg.PlotWidget(
            title="Data and Fit",
            viewBox=self.custom_viewbox
        )
```

### Method Implementations

The `MainWindow` implements the required host interface:

1. **_toggle_nearest_point_exclusion_xy()**: Finds the nearest data point and toggles its excluded state
2. **_nearest_target_xy()**: Checks for proximity to fit curve or data points
3. **set_selected()**: Updates selection state and visual feedback
4. **_update_data_plot()**: Requests plot update from viewmodel
5. **update_previews()**: Requests live preview update during drag

### Data Synchronization

The `update_plot_data()` method keeps the host attributes synchronized:

```python
def update_plot_data(self, x, y_data, y_fit=None, y_err=None):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y_data, dtype=float)
    
    # Update attributes for custom viewbox
    self.energy = x_arr
    self.counts = y_arr
    
    # ... plotting logic ...
```

## Usage Patterns

### Enabling Exclude Mode

```python
# In MainWindow or via UI control
self.exclude_mode = True  # Enable exclude mode
self.exclude_mode = False # Disable exclude mode
```

### Object Selection Workflow

1. User clicks on plot element
2. `InteractiveViewBox.mouseClickEvent()` captures click
3. Calls `host._nearest_target_xy(x, y)` to find target
4. If target found, calls `host.set_selected(kind, obj)`
5. Host updates visual feedback and state

### Object Dragging Workflow

1. User clicks and drags with object selected
2. `InteractiveViewBox.mouseDragEvent()` captures drag
3. Updates object position via `_update_dragged_object(x)`
4. Calls `host.update_previews()` for live feedback
5. On mouse release, drag completes

### Exclude Box Selection Workflow

1. User enables exclude mode
2. User drags to draw selection box
3. `InteractiveViewBox` creates semi-transparent rectangle overlay
4. On mouse release, `_finish_exclude_box_selection()` is called
5. All points within box boundaries are toggled
6. Host's `_update_data_plot()` is called to refresh display

## Customization

### Pixel Thresholds

Adjust selection sensitivity by modifying thresholds in host methods:

```python
def _nearest_target_xy(self, x, y, pixel_threshold=50):
    # Increase/decrease threshold as needed
    pass

def _toggle_nearest_point_exclusion_xy(self, x, y, pixel_threshold=10):
    # Adjust for point selection sensitivity
    pass
```

### Visual Styling

Modify overlay appearance in `custom_viewbox.py`:

```python
# In mouseDragEvent(), exclude box creation:
pen = pg.mkPen((255, 140, 0), width=2, style=DashLine)  # Orange dashed border
brush = pg.mkBrush(255, 165, 0, 50)                      # Semi-transparent orange fill
```

### Draggable Object Types

Extend `_update_dragged_object()` to support new object types:

```python
def _update_dragged_object(self, x):
    if isinstance(self._drag_obj, dict):
        # Dictionary-like object (peaks, markers)
        self._drag_obj['center'] = x
        # ...
    elif isinstance(self._drag_obj, MyCustomObject):
        # Add support for custom types
        self._drag_obj.set_position(x)
    # ...
```

## Example

See `examples/custom_viewbox_example.py` for a standalone demonstration of the custom viewbox with a minimal host implementation.

## Benefits

### For Users
- Intuitive interaction with plot elements
- Precise control over data inclusion/exclusion
- Visual feedback during manipulation
- Familiar pan/zoom behavior when not interacting

### For Developers
- Clean separation of concerns (ViewBox vs Host)
- Reusable component across different contexts
- Well-defined interface contract
- Minimal coupling with specific data models

## Future Enhancements

Potential areas for extension:

1. **Multi-selection**: Support selecting multiple objects simultaneously
2. **Undo/Redo**: Track exclusion/position changes for undo
3. **Keyboard modifiers**: Use Ctrl/Shift for additive selection
4. **Custom cursors**: Change cursor to indicate available actions
5. **Snap-to-grid**: Optional snapping during drag operations
6. **Selection persistence**: Save/load selection and exclusion state

## Troubleshooting

### Objects Not Selecting

- Verify host implements `_nearest_target_xy()` correctly
- Check pixel threshold values are appropriate
- Ensure `selected_kind` and `selected_obj` are being set

### Dragging Not Working

- Confirm object has `center` or `position` attribute (or add to `_update_dragged_object()`)
- Verify `update_previews()` is implemented in host
- Check that object is actually selected before drag starts

### Exclude Mode Issues

- Ensure `exclude_mode` attribute is accessible and updated
- Verify `energy`, `counts`, and `excluded_mask` arrays have consistent length
- Check `_update_data_plot()` refreshes the plot correctly

## References

- **Original Implementation**: `Minis Testing/PySide_Fitter_PyQtGraph_drag.py` (PeakViewBox)
- **PyQtGraph Documentation**: https://pyqtgraph.readthedocs.io/
- **BigFit Architecture**: See `IMPLEMENTATION_SUMMARY.md` and `INPUT_HANDLER_INTEGRATION.md`

# Input Handler Integration Documentation

## Overview

This document describes the input handler integration that centralizes mouse and keyboard event handling for the PUFFIN application, following patterns from `PySide_Fitter_PyQtGraph.py`.

## Architecture

The integration follows the MVVM (Model-View-ViewModel) pattern with three layers:

```
┌─────────────────┐
│  Input Handler  │  (View Layer - Captures raw events)
│  (Signals)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Main Window    │  (View Layer - Translates to actions)
│  (Event methods)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Fitter VM      │  (Logic Layer - Updates state)
│  (Handlers)     │
└─────────────────┘
```

## Components

### 1. InputHandler (view/input_handler.py)

**Purpose**: Centralized event capture and signal emission for PyQtGraph plots.

**Signals**:
- `mouse_clicked(float x, float y, object button)` - Emitted when plot is clicked
- `mouse_moved(float x, float y)` - Emitted during mouse movement (data coordinates)
- `key_pressed(int key, object modifiers)` - Emitted on keyboard input
- `wheel_scrolled(int delta, object modifiers)` - Emitted on mouse wheel scroll

**Key Methods**:
- `__init__(plot_widget)` - Initialize and connect to PyQtGraph widget
- `_connect_events()` - Wire up event handlers to plot widget
- `map_to_data_coords(scene_x, scene_y)` - Convert scene to data coordinates
- Event handler methods (`_on_mouse_click`, `_on_mouse_move`, etc.)

**Usage**:
```python
# In MainWindow.__init__
self.input_handler = InputHandler(self.plot_widget)
self.input_handler.mouse_clicked.connect(self._on_plot_clicked)
```

### 2. MainWindow Updates (view/main_window.py)

**New Attributes**:
- `self.input_handler` - InputHandler instance
- `parameters_updated` - Signal emitted when parameter panel refreshes

**New Methods**:
- `_connect_input_handler()` - Connect input handler signals to view methods
- `_on_plot_clicked(x, y, button)` - Handle plot click events
- `_on_plot_mouse_moved(x, y)` - Handle mouse movement (for dragging, etc.)
- `_on_plot_key_pressed(key, modifiers)` - Handle keyboard shortcuts
- `_on_plot_wheel_scrolled(delta, modifiers)` - Handle wheel with modifiers

**Keyboard Shortcuts Implemented**:
- `R` - Reset/auto-range the view
- `Space` - Clear selection (delegated to viewmodel)
- Other keys delegated to viewmodel for custom actions

### 3. FitterViewModel Updates (viewmodel/fitter_vm.py)

**New Methods**:
- `handle_plot_click(x, y, button)` - Process click events (future: peak selection)
- `handle_plot_mouse_move(x, y)` - Support dragging operations
- `handle_key_press(key, modifiers)` - Execute keyboard-triggered actions
  - `F` key - Run fit
  - `U` key - Update plot
  - `Space` - Clear selection
- `handle_wheel_scroll(delta, modifiers)` - Adjust parameters interactively
  - `Ctrl+Wheel` - Adjust first parameter (example implementation)

## Event Flow Examples

### Example 1: Mouse Click on Plot

```
User clicks plot at (5.2, 100.3)
    ↓
InputHandler._on_mouse_click() receives event
    ↓
Maps scene coordinates to data coordinates
    ↓
Emits: input_handler.mouse_clicked(5.2, 100.3, LeftButton)
    ↓
MainWindow._on_plot_clicked(5.2, 100.3, LeftButton)
    ↓
Logs click and delegates to viewmodel
    ↓
FitterViewModel.handle_plot_click(5.2, 100.3, LeftButton)
    ↓
Future: Select nearest peak, start dragging, etc.
```

### Example 2: Keyboard Shortcut (F key for Fit)

```
User presses F key
    ↓
InputHandler._on_key_press() captures event
    ↓
Emits: input_handler.key_pressed(Qt.Key_F, modifiers)
    ↓
MainWindow._on_plot_key_pressed(Qt.Key_F, modifiers)
    ↓
Delegates to viewmodel
    ↓
FitterViewModel.handle_key_press(Qt.Key_F, modifiers)
    ↓
Recognizes F key and calls run_fit()
    ↓
Fit executes and plot updates
```

### Example 3: Wheel Scroll with Modifier (Ctrl+Wheel)

```
User scrolls wheel while holding Ctrl
    ↓
InputHandler.eventFilter() captures wheel event
    ↓
Emits: input_handler.wheel_scrolled(120, Qt.ControlModifier)
    ↓
MainWindow._on_plot_wheel_scrolled(120, Qt.ControlModifier)
    ↓
Logs event and delegates to viewmodel
    ↓
FitterViewModel.handle_wheel_scroll(120, Qt.ControlModifier)
    ↓
Detects Ctrl modifier, adjusts parameter by 10%
    ↓
Calls apply_parameters() and update_plot()
```

## Integration with PySide_Fitter_PyQtGraph.py Patterns

The implementation extracts and adapts patterns from the reference implementation:

| PySide_Fitter_PyQtGraph.py | PUFFIN Implementation |
|----------------------------|----------------------|
| `connect_plot_events()` | `InputHandler._connect_events()` |
| `on_mouse_click(event)` | `InputHandler._on_mouse_click()` → `mouse_clicked` signal |
| `on_mouse_move(event)` | `InputHandler._on_mouse_move()` → `mouse_moved` signal |
| `on_key_press(event)` | `InputHandler._on_key_press()` → `key_pressed` signal |
| `eventFilter(obj, ev)` | `InputHandler.eventFilter()` → `wheel_scrolled` signal |
| Direct parameter updates | Delegated through signals to `FitterViewModel` |

**Key Differences**:
- **Separation of Concerns**: Input handling separated from business logic
- **Signal-Based**: Events propagate through Qt signals instead of direct calls
- **Testable**: Logic in viewmodel can be tested without GUI
- **Reusable**: InputHandler can be reused with different views

## Extending the Integration

### Adding New Keyboard Shortcuts

1. **In FitterViewModel.handle_key_press()**:
```python
elif key == Qt.Key_S:  # Save shortcut
    self.log_message.emit("S key: Saving data")
    self.save_data()
```

### Adding Parameter Adjustment with Wheel

1. **In FitterViewModel.handle_wheel_scroll()**:
```python
if is_shift and not is_ctrl:
    # Shift+Wheel adjusts second parameter
    param_name = list(specs.keys())[1]  # Get second param
    # ... adjust logic
```

### Adding Drag Operations

1. **Track drag state in MainWindow**:
```python
self._drag_active = False
self._drag_start_x = None

def _on_plot_mouse_moved(self, x, y):
    if self._drag_active:
        self.viewmodel.handle_drag(self._drag_start_x, x)
```

2. **Handle in FitterViewModel**:
```python
def handle_drag(self, start_x, current_x):
    # Update peak position, region, etc.
    pass
```

## Testing

A comprehensive test suite is available in `/tmp/test_input_handler.py` that validates:

1. ✓ InputHandler module structure and signals
2. ✓ MainWindow integration (import, instantiation, methods)
3. ✓ FitterViewModel handler methods
4. ✓ MVVM architecture compliance
5. ✓ Pattern matching with reference implementation

Run tests with:
```bash
python3 /tmp/test_input_handler.py
```

## Migration Notes

For developers familiar with PySide_Fitter_PyQtGraph.py:

1. **Event Handlers**: Instead of implementing `on_mouse_click()` directly in MainWindow, connect to `input_handler.mouse_clicked` signal
2. **Parameter Updates**: Use `viewmodel.apply_parameters()` instead of directly modifying spinboxes
3. **State Management**: Keep UI state in `ModelState` (accessed via viewmodel) instead of MainWindow attributes
4. **Testing**: Logic in viewmodel can be unit tested without Qt GUI

## Future Enhancements

Potential improvements to consider:

1. **Peak Selection**: Implement click-to-select nearest peak functionality
2. **Drag-to-Move**: Allow dragging peaks horizontally to adjust position
3. **Region Selection**: Add box-select for excluding data points
4. **Touch Support**: Extend InputHandler to support touch gestures
5. **Undo/Redo**: Add command pattern for reversible operations
6. **Multi-Selection**: Support selecting multiple peaks with Ctrl+Click

## See Also

- `PUFFIN/view/input_handler.py` - InputHandler implementation
- `PUFFIN/view/main_window.py` - View integration
- `PUFFIN/viewmodel/fitter_vm.py` - ViewModel handlers
- `Minis Testing/PySide_Fitter_PyQtGraph.py` - Reference implementation

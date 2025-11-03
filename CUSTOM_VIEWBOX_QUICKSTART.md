# Quick Start: Using the Custom ViewBox

## What's New?

BigFit now includes an enhanced interactive viewbox that provides intuitive mouse interactions for plot manipulation. This was extracted and generalized from the `PeakViewBox` pattern in `PySide_Fitter_PyQtGraph_drag.py`.

## Features

### 1. Click to Select
- Click on any plot element (fit curve, data points) to select it
- Selected elements are visually highlighted
- Click empty space to clear selection

### 2. Drag to Move (when object selected)
- With an object selected, drag it left/right to adjust position
- Live preview updates during drag
- Release to finalize position

### 3. Exclude Mode (toggle data points)
- Enable exclude mode (can add UI button in future)
- Click individual points to toggle inclusion/exclusion
- Excluded points are visually distinct

### 4. Box Selection (in exclude mode)
- Drag to draw a selection box
- All points within box are toggled
- Semi-transparent orange overlay shows selection area

### 5. Smart Wheel Behavior
- Wheel zoom blocked when object is selected (prevents accidental zoom)
- Wheel zoom blocked in exclude mode (keeps view stable)
- Normal zoom behavior otherwise

## How It Works

The custom viewbox is **automatically integrated** into BigFit's `MainWindow`. No additional setup required!

### Architecture

```
┌─────────────────────────────────────┐
│         MainWindow (Host)           │
│  - Provides data & state            │
│  - Implements interface methods     │
│  - Handles business logic           │
└──────────────┬──────────────────────┘
               │
               │ host interface
               │
┌──────────────▼──────────────────────┐
│     InteractiveViewBox              │
│  - Captures mouse/keyboard events   │
│  - Manages selection state          │
│  - Draws overlays                   │
│  - Delegates to host                │
└─────────────────────────────────────┘
```

## For End Users

### Basic Workflow

1. **Load your data** as usual
2. **Click** on elements to select them
3. **Drag** selected elements to adjust
4. **Enable exclude mode** to filter points
5. **Save** your results

### Tips

- Selection is indicated by visual changes (highlight, size change)
- Press Space (if bound) to clear selection and return to pan/zoom
- In exclude mode, box selection is faster than clicking individual points

## For Developers

### Adding Exclude Mode Toggle

Add a button to toggle exclude mode:

```python
# In MainWindow._init_left_dock() or similar
exclude_btn = QPushButton("Toggle Exclude Mode")
exclude_btn.setCheckable(True)
exclude_btn.toggled.connect(lambda checked: setattr(self, 'exclude_mode', checked))
layout.addWidget(exclude_btn)
```

### Customizing Selection

Adjust sensitivity in `_nearest_target_xy()`:

```python
def _nearest_target_xy(self, x, y, pixel_threshold=50):
    # Increase threshold for easier selection
    pixel_threshold = 100
    # ... rest of method
```

### Supporting New Object Types

Extend `_update_dragged_object()` in `custom_viewbox.py`:

```python
def _update_dragged_object(self, x):
    # Add support for your custom object
    if isinstance(self._drag_obj, MyPeakClass):
        self._drag_obj.set_center(x)
        # Update widgets, invalidate caches, etc.
```

### Saving Exclusion State

Store `excluded_mask` with your data:

```python
# In save operation
np.save('excluded_mask.npy', self.excluded_mask)

# In load operation
self.excluded_mask = np.load('excluded_mask.npy')
```

## Documentation

- **Full Integration Guide**: See `CUSTOM_VIEWBOX_INTEGRATION.md`
- **Example Code**: See `BigFit/examples/custom_viewbox_example.py`
- **Source**: See `BigFit/view/custom_viewbox.py`

## Troubleshooting

**Q: Selection not working?**
- Check that `_nearest_target_xy()` returns valid targets
- Verify pixel threshold is appropriate for your data scale

**Q: Dragging doesn't update?**
- Ensure `update_previews()` is implemented
- Check that selected object has position attribute

**Q: Box selection not toggling points?**
- Verify `energy`, `counts`, and `excluded_mask` arrays match in length
- Check `_update_data_plot()` refreshes the display

**Q: Want to disable a feature?**
- Override the relevant method in ViewBox subclass
- Or modify behavior in host implementation

## What's Next?

The custom viewbox is fully integrated and ready to use! Consider adding:

1. UI controls for exclude mode toggle
2. Keyboard shortcuts (Space to deselect, E for exclude mode)
3. Status bar showing current mode
4. Visual cursor changes for different modes

## Questions?

- See the comprehensive guide: `CUSTOM_VIEWBOX_INTEGRATION.md`
- Review the example: `BigFit/examples/custom_viewbox_example.py`
- Check the source code: `BigFit/view/custom_viewbox.py`

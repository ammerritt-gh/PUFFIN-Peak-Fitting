"""
Example demonstrating the InteractiveViewBox custom viewbox.

This example shows how the custom viewbox provides:
1. Click-to-select functionality for plot elements
2. Drag-to-move for selected objects
3. Exclude mode for data point selection
4. Rubber-band box selection in exclude mode
5. Conditional wheel blocking when selection is active

The custom viewbox is automatically integrated into BigFit's MainWindow.
This example is for reference and understanding the feature set.
"""

import sys
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
import pyqtgraph as pg
from view.custom_viewbox import InteractiveViewBox


class ExampleHost:
    """
    Example host class that provides the interface required by InteractiveViewBox.
    
    The InteractiveViewBox expects the host to have these attributes and methods:
    
    Attributes:
        exclude_mode (bool): Whether exclude mode is active
        selected_kind (str|None): Type of selected object
        selected_obj (object|None): Reference to selected object
        energy (np.ndarray): X data values
        counts (np.ndarray): Y data values
        excluded_mask (np.ndarray): Boolean mask for excluded points
    
    Methods:
        _toggle_nearest_point_exclusion_xy(x, y): Toggle exclusion of nearest point
        _nearest_target_xy(x, y): Find nearest selectable target
        set_selected(kind, obj): Set currently selected object
        _update_data_plot(do_range): Update plot after exclusion changes
        update_previews(): Update preview overlays during drag
    """
    
    def __init__(self):
        # Initialize required attributes
        self.exclude_mode = False
        self.selected_kind = None
        self.selected_obj = None
        
        # Generate example data
        self.energy = np.linspace(-10, 10, 100)
        self.counts = 100 * np.exp(-self.energy**2 / 10) + 10 * np.random.randn(100)
        self.excluded_mask = np.zeros(len(self.energy), dtype=bool)
    
    def _toggle_nearest_point_exclusion_xy(self, x, y):
        """Toggle exclusion of the nearest data point to (x, y)."""
        idx = np.argmin(np.abs(self.energy - x))
        self.excluded_mask[idx] = not self.excluded_mask[idx]
        print(f"Toggled exclusion for point {idx}: {self.excluded_mask[idx]}")
        self._update_data_plot(do_range=False)
    
    def _nearest_target_xy(self, x, y):
        """Find nearest selectable target near (x, y)."""
        # In this example, we'll return a simple target if close to data
        idx = np.argmin(np.abs(self.energy - x))
        if abs(self.energy[idx] - x) < 0.5:  # threshold in data units
            return ('data_point', {'index': idx, 'x': self.energy[idx], 'y': self.counts[idx]})
        return (None, None)
    
    def set_selected(self, kind, obj):
        """Set the currently selected object."""
        self.selected_kind = kind
        self.selected_obj = obj
        if kind is None:
            print("Selection cleared")
        else:
            print(f"Selected: {kind}, {obj}")
    
    def _update_data_plot(self, do_range=True):
        """Update the data plot."""
        print(f"Update data plot (do_range={do_range})")
        # In a real application, this would redraw the plot
    
    def update_previews(self):
        """Update preview overlays."""
        print("Update previews")
        # In a real application, this would update overlay graphics


def main():
    """
    Main function demonstrating the custom viewbox.
    
    Usage:
    - Click on data points to select them
    - Press 'E' key (or add button) to toggle exclude mode
    - In exclude mode:
      - Click points to toggle their exclusion
      - Drag to draw a rubber-band box for batch exclusion
    - With a point selected, drag to move it (demo only)
    - Wheel scrolling is blocked when something is selected
    """
    app = QApplication(sys.argv)
    
    # Create host with required interface
    host = ExampleHost()
    
    # Create custom viewbox
    viewbox = InteractiveViewBox(host=host)
    
    # Create plot widget with custom viewbox
    plot = pg.PlotWidget(viewBox=viewbox, title="Custom ViewBox Example")
    plot.setBackground('w')
    plot.showGrid(x=True, y=True, alpha=0.3)
    
    # Plot the data
    included_mask = ~host.excluded_mask
    scatter = pg.ScatterPlotItem(
        x=host.energy[included_mask],
        y=host.counts[included_mask],
        size=8,
        pen=pg.mkPen('k'),
        brush=pg.mkBrush('b')
    )
    plot.addItem(scatter)
    
    # Add instructions as text
    instructions = pg.TextItem(
        "Custom ViewBox Demo\n"
        "- Click to select points\n"
        "- Toggle exclude mode (implement button)\n"
        "- Box-select in exclude mode\n"
        "- Drag selected objects",
        anchor=(0, 0),
        color='k'
    )
    instructions.setPos(-9, max(host.counts))
    plot.addItem(instructions)
    
    plot.resize(800, 600)
    plot.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    # Note: This example requires a display/X11 to run
    # It demonstrates the API and integration pattern
    try:
        main()
    except Exception as e:
        print(f"Cannot run GUI example in this environment: {e}")
        print("\nThis example demonstrates the InteractiveViewBox API.")
        print("In BigFit, the custom viewbox is automatically integrated into MainWindow.")
        print("See BigFit/view/custom_viewbox.py for implementation details.")

# view/input_handler.py
from PySide6.QtCore import QObject
import numpy as np


class InputHandler(QObject):
    """
    Connects interactive ViewBox events (peak selection, drag, exclusion)
    to ViewModel logic.

    This merges your eventFilter-based logic with signal-based ViewBox input.
    """

    def __init__(self, viewbox=None, viewmodel=None, parent=None):
        super().__init__(parent)
        self.viewmodel = viewmodel
        self.viewbox = viewbox

        self.dragging_peak = None
        self.drag_offset = 0.0

        if self.viewbox is not None:
            self._connect_viewbox()

    def set_viewmodel(self, vm):
        self.viewmodel = vm

    def _connect_viewbox(self):
        """Connect all custom ViewBox signals."""
        vb = self.viewbox
        vb.peakSelected.connect(self.on_peak_selected)
        vb.peakMoved.connect(self.on_peak_moved)
        vb.excludePointClicked.connect(self.on_exclude_point)
        vb.excludeBoxDrawn.connect(self.on_exclude_box)

    # -----------------------
    # Core handlers
    # -----------------------
    def on_peak_selected(self, x, y):
        """User clicked near a peak → select or start drag."""
        if self.viewmodel is None:
            return
        idx = self.find_nearest_peak(x)
        if idx is not None:
            self.dragging_peak = idx
            self.drag_offset = self.viewmodel.peaks[idx] - x
            self.viewmodel.log_message.emit(f"Selected peak #{idx} at x={self.viewmodel.peaks[idx]:.3f}")

    def on_peak_moved(self, peak_info):
        """User dragged a peak."""
        if self.viewmodel is None or self.dragging_peak is None:
            return
        new_x = float(peak_info.get("center", 0.0))
        self.viewmodel.peaks[self.dragging_peak] = new_x
        if hasattr(self.viewmodel, "update_plot"):
            self.viewmodel.update_plot()

    def on_exclude_point(self, x, y):
        """Toggle exclusion of a single data point."""
        if self.viewmodel is None:
            return
        self.viewmodel.toggle_point_exclusion(x, y)
        self.viewmodel.log_message.emit(f"Toggled exclusion at ({x:.2f}, {y:.2f})")

    def on_exclude_box(self, x0, y0, x1, y1):
        """Box-drag exclusion event."""
        if self.viewmodel is None:
            return
        self.viewmodel.toggle_box_exclusion(x0, y0, x1, y1)
        self.viewmodel.log_message.emit(
            f"Exclusion box: ({x0:.2f},{y0:.2f}) → ({x1:.2f},{y1:.2f})"
        )

    # -----------------------
    # Helpers
    # -----------------------
    def find_nearest_peak(self, x, threshold=0.1):
        """Find the nearest peak in ViewModel within threshold."""
        if self.viewmodel is None or not hasattr(self.viewmodel, "peaks"):
            return None
        peaks = np.array(self.viewmodel.peaks)
        if len(peaks) == 0:
            return None
        idx = np.argmin(np.abs(peaks - x))
        if abs(peaks[idx] - x) <= threshold:
            return idx
        return None

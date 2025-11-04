# view/input_handler.py
from PySide6.QtCore import QObject, Qt, QPointF, QEvent
from PySide6.QtGui import QKeyEvent, QMouseEvent
import numpy as np


class InputHandler(QObject):
    """
    Handles all interactive user input for the main plot:
    - Peak dragging and exclusion events (from ViewBox signals)
    - Optional fallback to raw Qt eventFilter handling
    - Curve selection and deselection via mouse + spacebar
    """

    def __init__(self, viewbox=None, viewmodel=None, parent=None):
        super().__init__(parent)
        self.viewmodel = viewmodel
        self.viewbox = viewbox

        # dragging state
        self.dragging_peak = None
        self.drag_offset = 0.0

        # curve selection state (stores the curve ID or None)
        self.selected_curve_id = None

        if self.viewbox is not None:
            self._connect_viewbox()

    def set_viewmodel(self, vm):
        self.viewmodel = vm

    # -----------------------
    # ViewBox signal hookups
    # -----------------------
    def _connect_viewbox(self):
        """Connect all custom ViewBox signals to local handlers."""
        vb = self.viewbox
        if hasattr(vb, "peakSelected"):
            vb.peakSelected.connect(self.on_peak_selected)
        if hasattr(vb, "peakMoved"):
            vb.peakMoved.connect(self.on_peak_moved)
        if hasattr(vb, "excludePointClicked"):
            vb.excludePointClicked.connect(self.on_exclude_point)
        if hasattr(vb, "excludeBoxDrawn"):
            vb.excludeBoxDrawn.connect(self.on_exclude_box)

    # -----------------------
    # Event Filter (fallback)
    # -----------------------
    def eventFilter(self, obj, event):
        """Fallback for when ViewBox signals are unavailable."""
        if isinstance(event, QMouseEvent):
            # Use QEvent enums for the event.type() comparisons
            if event.type() == QEvent.MouseButtonPress:
                return self.on_mouse_press(obj, event)
            elif event.type() == QEvent.MouseMove:
                return self.on_mouse_move(obj, event)
            elif event.type() == QEvent.MouseButtonRelease:
                return self.on_mouse_release(obj, event)

        if isinstance(event, QKeyEvent) and event.type() == QEvent.KeyPress:
            return self.handle_key(event)

        return super().eventFilter(obj, event)

    # -----------------------
    # ViewBox event handlers
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
        if hasattr(self.viewmodel, "toggle_point_exclusion"):
            self.viewmodel.toggle_point_exclusion(x, y)
        self.viewmodel.log_message.emit(f"Toggled exclusion at ({x:.2f}, {y:.2f})")

    def on_exclude_box(self, x0, y0, x1, y1):
        """Box-drag exclusion event."""
        if self.viewmodel is None:
            return
        if hasattr(self.viewmodel, "toggle_box_exclusion"):
            self.viewmodel.toggle_box_exclusion(x0, y0, x1, y1)
        self.viewmodel.log_message.emit(f"Exclusion box: ({x0:.2f},{y0:.2f}) → ({x1:.2f},{y1:.2f})")

    # -----------------------
    # Mouse event (fallbacks)
    # -----------------------
    def on_mouse_press(self, obj, event):
        if event.button() == Qt.LeftButton:
            x, y = self.mouse_to_data(obj, event.pos())

            # Try selecting a curve
            curve_id = self.detect_curve_at(obj, x, y)
            if curve_id is not None:
                self.set_selected_curve(curve_id)
                return True

            # Otherwise, check for a nearby peak
            peak_idx = self.find_nearest_peak(x)
            if peak_idx is not None:
                self.dragging_peak = peak_idx
                self.drag_offset = self.viewmodel.peaks[peak_idx] - x
                return True
        return False

    def on_mouse_move(self, obj, event):
        if self.dragging_peak is not None:
            x, y = self.mouse_to_data(obj, event.pos())
            new_x = x + self.drag_offset
            self.viewmodel.peaks[self.dragging_peak] = new_x
            if hasattr(self.viewmodel, "update_plot"):
                self.viewmodel.update_plot()
            return True
        return False

    def on_mouse_release(self, obj, event):
        if event.button() == Qt.LeftButton and self.dragging_peak is not None:
            self.dragging_peak = None
            return True
        return False

    # -----------------------
    # Keyboard
    # -----------------------
    def handle_key(self, event):
        key = event.key()

        # Log the key press in a readable format
        key_text = event.text() or ""
        key_name = Qt.Key(key).name if hasattr(Qt.Key(key), "name") else str(key)
        if hasattr(self.viewmodel, "log_message"):
            self.viewmodel.log_message.emit(
                f"KeyPress: key={key} ({key_name}), text='{key_text}'"
            )
        else:
            print(f"KeyPress: key={key} ({key_name}), text='{key_text}'")

        if key == Qt.Key_Plus:
            self.viewmodel.log_message.emit("Increase parameter (placeholder)")
            return True
        elif key == Qt.Key_Minus:
            self.viewmodel.log_message.emit("Decrease parameter (placeholder)")
            return True
        elif key == Qt.Key_Space:
            if self.viewbox is not None:
                self.viewbox.clear_selection()
            self.clear_selected_curve()
            return True
        return False

    # -----------------------
    # Selection management
    # -----------------------
    def set_selected_curve(self, curve_id):
        """Set selected curve ID and inform ViewModel."""
        if self.selected_curve_id != curve_id:
            self.selected_curve_id = curve_id
            if hasattr(self.viewmodel, "set_selected_curve"):
                self.viewmodel.set_selected_curve(curve_id)
            if curve_id is not None:
                self.viewmodel.log_message.emit(f"Curve '{curve_id}' selected.")

    def clear_selected_curve(self):
        """Clear selection and notify ViewModel."""
        # Always clear local state and instruct the ViewModel to clear its selection.
        # This ensures deselect works even if selection was set directly on the ViewModel
        # (e.g. via MainWindow._on_curve_clicked) and InputHandler.selected_curve_id is stale.
        self.selected_curve_id = None
        if self.viewmodel is not None and hasattr(self.viewmodel, "clear_selected_curve"):
            try:
                self.viewmodel.clear_selected_curve()
            except Exception:
                pass
        if self.viewmodel is not None and hasattr(self.viewmodel, "log_message"):
            try:
                self.viewmodel.log_message.emit("Curve deselected (spacebar).")
            except Exception:
                pass

    # -----------------------
    # Utility helpers
    # -----------------------
    def mouse_to_data(self, plot_widget, pos):
        """Convert mouse QPoint to plot data coordinates."""
        vb = plot_widget.getViewBox()
        mouse_point = vb.mapSceneToView(pos)
        return mouse_point.x(), mouse_point.y()

    def find_nearest_peak(self, x, threshold=0.1):
        """Find nearest peak (for dragging)."""
        if self.viewmodel is None or not hasattr(self.viewmodel, "peaks"):
            return None
        peaks = np.array(self.viewmodel.peaks)
        if len(peaks) == 0:
            return None
        idx = np.argmin(np.abs(peaks - x))
        if abs(peaks[idx] - x) <= threshold:
            return idx
        return None

    def detect_curve_at(self, plot_widget, x, y, tol=0.05):
        """
        Attempt to detect which curve (if any) was clicked at (x, y).
        Returns a curve_id or None.
        """
        if self.viewmodel is None or not hasattr(self.viewmodel, "curves"):
            return None
        for cid, (cx, cy) in self.viewmodel.curves.items():
            cx = np.asarray(cx)
            cy = np.asarray(cy)
            if len(cx) == 0:
                continue
            dist = np.min(np.sqrt((cx - x) ** 2 + (cy - y) ** 2))
            if dist < tol:
                return cid
        return None

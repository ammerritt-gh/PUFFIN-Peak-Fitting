# view/input_handler.py
from PySide6.QtCore import QObject, Qt, QPointF
from PySide6.QtGui import QKeyEvent, QMouseEvent

import numpy as np

class InputHandler(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.viewmodel = None
        self.dragging_peak = None
        self.drag_offset = 0.0

    def set_viewmodel(self, vm):
        self.viewmodel = vm

    # -----------------------
    # Event filter
    # -----------------------
    def eventFilter(self, obj, event):
        # Mouse events (only plot)
        if isinstance(event, QMouseEvent):
            if event.type() == QMouseEvent.MouseButtonPress:
                return self.on_mouse_press(obj, event)
            elif event.type() == QMouseEvent.MouseMove:
                return self.on_mouse_move(obj, event)
            elif event.type() == QMouseEvent.MouseButtonRelease:
                return self.on_mouse_release(obj, event)

        # Keyboard events (hotkeys)
        if isinstance(event, QKeyEvent) and event.type() == QKeyEvent.KeyPress:
            return self.handle_key(event)

        return super().eventFilter(obj, event)

    # -----------------------
    # Mouse helpers
    # -----------------------
    def mouse_to_data(self, plot_widget, pos):
        """Convert mouse QPoint to plot data coordinates"""
        vb = plot_widget.getViewBox()
        mouse_point = vb.mapSceneToView(pos)
        return mouse_point.x(), mouse_point.y()

    def find_nearest_peak(self, x, threshold=0.1):
        """Find the nearest peak in viewmodel within threshold"""
        if self.viewmodel is None or not hasattr(self.viewmodel, "peaks"):
            return None
        peaks = np.array(self.viewmodel.peaks)
        if len(peaks) == 0:
            return None
        idx = np.argmin(np.abs(peaks - x))
        if abs(peaks[idx] - x) <= threshold:
            return idx
        return None

    # -----------------------
    # Mouse event handlers
    # -----------------------
    def on_mouse_press(self, obj, event):
        if event.button() == Qt.LeftButton:
            x, y = self.mouse_to_data(obj, event.pos())
            peak_idx = self.find_nearest_peak(x)
            if peak_idx is not None:
                self.dragging_peak = peak_idx
                self.drag_offset = self.viewmodel.peaks[peak_idx] - x
                return True  # event handled
        return False

    def on_mouse_move(self, obj, event):
        if self.dragging_peak is not None:
            x, y = self.mouse_to_data(obj, event.pos())
            new_x = x + self.drag_offset
            # Update the peak in viewmodel
            self.viewmodel.peaks[self.dragging_peak] = new_x
            if hasattr(self.viewmodel, "update_plot"):
                self.viewmodel.update_plot()
            return True
        return False

    def on_mouse_release(self, obj, event):
        if event.button() == Qt.LeftButton and self.dragging_peak is not None:
            # Finish dragging
            self.dragging_peak = None
            return True
        return False

    # -----------------------
    # Keyboard / hotkeys
    # -----------------------
    def handle_key(self, event):
        key = event.key()
        if key == Qt.Key_Plus:
            print("Increase something")  # replace with your logic
            return True
        elif key == Qt.Key_Minus:
            print("Decrease something")  # replace with your logic
            return True
        return False

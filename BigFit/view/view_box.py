# view/peak_viewbox.py
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np


class CustomViewBox(pg.ViewBox):
    """
    Custom ViewBox for interactive data manipulation.
    Handles:
      - Selecting and dragging peaks
      - Box exclusion (click + drag)
      - Point exclusion (click)
    Emits clean, ViewModel-friendly signals instead of touching the model.
    """

    peakSelected = QtCore.Signal(float, float)           # x, y
    peakDeselected = QtCore.Signal()                     # emitted when deselected
    peakMoved = QtCore.Signal(dict)                      # {"center": x, "height": y, ...}
    excludePointClicked = QtCore.Signal(float, float)
    excludeBoxDrawn = QtCore.Signal(float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseMode(self.PanMode)
        self.setAspectLocked(False)
        self.enableAutoRange(True, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable, True)
        self.setFocus()

        # Internal state
        self._dragging = False
        self._exclude_active = False
        self._exclude_start = None
        self._exclude_rect = None
        self._selected_peak = None   # stores (x, y)
        self.exclude_mode = False

    # ---------------------
    # Mouse events
    # ---------------------
    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            pos_v = self.mapSceneToView(ev.scenePos())
            x, y = float(pos_v.x()), float(pos_v.y())

            if self.exclude_mode:
                self.excludePointClicked.emit(x, y)
                ev.accept()
                return

            # If a peak is already selected and we click nearby → deselect it
            if self._selected_peak is not None:
                sel_x, sel_y = self._selected_peak
                dist = np.hypot(x - sel_x, y - sel_y)
                if dist < 0.05:  # configurable tolerance
                    self.clear_selection()
                    ev.accept()
                    return

            # Otherwise, select a new peak
            self._selected_peak = (x, y)
            self.peakSelected.emit(x, y)
            self.setFocus()
            ev.accept()
            return

        super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() != QtCore.Qt.LeftButton:
            return super().mouseDragEvent(ev, axis)

        pos_v = self.mapSceneToView(ev.scenePos())
        x, y = float(pos_v.x()), float(pos_v.y())

        if self.exclude_mode:
            if ev.isStart():
                self._exclude_active = True
                self._exclude_start = (x, y)
                self._create_exclude_rect()
                ev.accept()
                return
            elif self._exclude_active:
                self._update_exclude_rect(x, y)
                ev.accept()
                if ev.isFinish():
                    x0, y0 = self._exclude_start
                    x1, y1 = x, y
                    self.excludeBoxDrawn.emit(x0, y0, x1, y1)
                    self._remove_exclude_rect()
                    self._exclude_active = False
                    ev.accept()
                return
            else:
                return

        if self._dragging and self._selected_peak is not None:
            ev.accept()
            self.peakMoved.emit({"center": x, "height": y})
            if ev.isFinish():
                self._dragging = False
            return

        if ev.isStart() and self._selected_peak is not None:
            self._dragging = True
            ev.accept()
            return

        super().mouseDragEvent(ev, axis)

    def wheelEvent(self, ev, axis=None):
        if self.exclude_mode or self._dragging:
            ev.accept()
            return
        super().wheelEvent(ev, axis)

    # ---------------------
    # External helpers
    # ---------------------
    def clear_selection(self):
        """Deselect the currently selected peak (if any)."""
        if self._selected_peak is not None:
            self._selected_peak = None
            self._dragging = False
            self.peakDeselected.emit()

    def _create_exclude_rect(self):
        rect_item = QtWidgets.QGraphicsRectItem()
        rect_item.setPen(pg.mkPen((255, 140, 0), width=2, style=QtCore.Qt.DashLine))
        rect_item.setBrush(pg.mkBrush(255, 165, 0, 50))
        rect_item.setZValue(1e6)
        if self.childGroup is not None:
            rect_item.setParentItem(self.childGroup)
        else:
            self.addItem(rect_item)
        self._exclude_rect = rect_item

    def _update_exclude_rect(self, x, y):
        if not self._exclude_rect or not self._exclude_start:
            return
        x0, y0 = self._exclude_start
        rx0, rx1 = (x0, x) if x0 <= x else (x, x0)
        ry0, ry1 = (y0, y) if y0 <= y else (y, y0)
        rect = QtCore.QRectF(rx0, ry0, rx1 - rx0, ry1 - ry0)
        self._exclude_rect.setRect(rect)

    def _remove_exclude_rect(self):
        if self._exclude_rect is not None:
            try:
                self.removeItem(self._exclude_rect)
            except Exception:
                pass
            self._exclude_rect = None

    def set_exclude_mode(self, enabled: bool):
        self.exclude_mode = bool(enabled)

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Space:
            self.clear_selection()
            ev.accept()
            return
        super().keyPressEvent(ev)
"""
Custom ViewBox for interactive plot manipulation.

This module provides a custom PyQtGraph ViewBox that enables:
- Custom mouse drag behavior for selected objects
- Exclude mode for data point selection/deselection  
- Click-to-select functionality
- Conditional wheel event handling based on selection state
- Rubber-band box selection in exclude mode

Extracted and generalized from PySide_Fitter_PyQtGraph_drag.py patterns.
"""

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets, QtGui


class InteractiveViewBox(pg.ViewBox):
    """
    Custom ViewBox to manage interactive mouse behavior.
    
    Features:
    - When an object is selected, left-drag moves it (no panning)
    - When no object is selected, default panning/zoom behavior is used
    - Exclude mode for toggling data point inclusion
    - Rubber-band box selection in exclude mode
    - Blocks wheel zoom when a selection is active
    
    The host object should provide:
    - exclude_mode: bool attribute indicating if exclude mode is active
    - selected_kind: str/None indicating type of selected object
    - selected_obj: object/None reference to selected object
    - Methods: _toggle_nearest_point_exclusion_xy(x, y), _nearest_target_xy(x, y),
               set_selected(kind, obj), _update_data_plot(do_range), update_previews()
    - Data attributes: energy, counts, excluded_mask
    """
    
    def __init__(self, host, *args, **kwargs):
        """
        Initialize the custom viewbox.
        
        Args:
            host: The parent window/widget that provides the interface methods
            *args, **kwargs: Additional arguments passed to pg.ViewBox
        """
        super().__init__(*args, **kwargs)
        self.host = host
        
        # Drag state for moving selected objects
        self._dragging_obj = False
        self._drag_obj = None
        
        # Exclude-mode drag box state
        self._exclude_active = False
        self._exclude_start = None  # (x0, y0) in view/data coords
        self._exclude_rect = None   # QGraphicsRectItem overlay in data coords

    def mouseClickEvent(self, ev):
        """
        Handle mouse click events.
        
        In exclude mode, toggles nearest data point.
        Otherwise, attempts to select objects or clears selection.
        """
        try:
            # Get Qt button constants safely
            try:
                LeftButton = QtCore.Qt.MouseButton.LeftButton
            except AttributeError:
                LeftButton = QtCore.Qt.LeftButton
            
            if ev.button() == LeftButton:
                pos_v = self.mapSceneToView(ev.scenePos())
                x, y = float(pos_v.x()), float(pos_v.y())
                
                # Exclude mode: toggle nearest point on click
                if getattr(self.host, 'exclude_mode', False):
                    if hasattr(self.host, '_toggle_nearest_point_exclusion_xy'):
                        self.host._toggle_nearest_point_exclusion_xy(x, y)
                    ev.accept()
                    return
                
                # Selection by click - try to find and select nearest target
                if hasattr(self.host, '_nearest_target_xy') and hasattr(self.host, 'set_selected'):
                    kind, obj = self.host._nearest_target_xy(x, y)
                    if kind is not None:
                        self.host.set_selected(kind, obj)
                        ev.accept()
                        return
                    # Clicked empty space: clear selection to allow panning next drag
                    self.host.set_selected(None, None)
        except Exception as e:
            print(f"Custom ViewBox mouseClickEvent error: {e}")
        
        # Fallback to default behavior
        super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        """
        Handle mouse drag events.
        
        In exclude mode, draws rubber-band selection box.
        With an object selected, drags the object.
        Otherwise, allows default panning/zoom.
        """
        try:
            # Get Qt button and pen style constants safely
            try:
                LeftButton = QtCore.Qt.MouseButton.LeftButton
                DashLine = QtCore.Qt.PenStyle.DashLine
            except AttributeError:
                try:
                    LeftButton = QtCore.Qt.LeftButton
                    DashLine = QtCore.Qt.DashLine
                except AttributeError:
                    LeftButton = 1
                    DashLine = 2
            
            # Exclude mode: handle rubber-band box selection
            if getattr(self.host, 'exclude_mode', False):
                if ev.button() == LeftButton:
                    pos_v = self.mapSceneToView(ev.scenePos())
                    x = float(pos_v.x())
                    y = float(pos_v.y())
                    
                    if ev.isStart():
                        # Begin rubber-band rectangle in data coordinates
                        self._exclude_active = True
                        self._exclude_start = (x, y)
                        
                        # Create rectangle overlay in data coords
                        try:
                            RectItemCls = getattr(QtWidgets, 'QGraphicsRectItem', None) or \
                                         getattr(QtGui, 'QGraphicsRectItem', None)
                            if RectItemCls is not None:
                                self._exclude_rect = RectItemCls()
                                pen = pg.mkPen((255, 140, 0), width=2, style=DashLine)
                                brush = pg.mkBrush(255, 165, 0, 50)
                                self._exclude_rect.setPen(pen)
                                self._exclude_rect.setBrush(brush)
                                self._exclude_rect.setZValue(1e6)
                                
                                # Parent into data transform space
                                if hasattr(self, 'childGroup') and self.childGroup is not None:
                                    self._exclude_rect.setParentItem(self.childGroup)
                                else:
                                    self.addItem(self._exclude_rect)
                        except Exception as e:
                            print(f"Failed to create exclude rect: {e}")
                            self._exclude_rect = None
                        ev.accept()
                        return
                    
                    # Update rectangle while dragging
                    if self._exclude_active and self._exclude_start is not None:
                        try:
                            x0, y0 = self._exclude_start
                            x1, y1 = x, y
                            rx0, rx1 = (x0, x1) if x0 <= x1 else (x1, x0)
                            ry0, ry1 = (y0, y1) if y0 <= y1 else (y1, y0)
                            if self._exclude_rect is not None:
                                r = QtCore.QRectF(rx0, ry0, rx1 - rx0, ry1 - ry0)
                                self._exclude_rect.setRect(r)
                        except Exception as e:
                            print(f"Failed to update exclude rect: {e}")
                        
                        ev.accept()
                        
                        if ev.isFinish():
                            # Toggle inclusion for points within the box
                            self._finish_exclude_box_selection(x, y)
                        return
                else:
                    # In exclude mode, consume other drag gestures to keep view fixed
                    ev.accept()
                    return
            
            # Object drag mode (when an object is selected)
            if ev.button() == LeftButton:
                if ev.isStart():
                    pos_v = self.mapSceneToView(ev.scenePos())
                    x, y = float(pos_v.x()), float(pos_v.y())
                    
                    # Only start dragging if an object is already selected
                    selected_kind = getattr(self.host, 'selected_kind', None)
                    selected_obj = getattr(self.host, 'selected_obj', None)
                    
                    if selected_kind is not None and selected_obj is not None:
                        # Check if this type of object supports dragging
                        # (could be 'phonon', 'peak', 'marker', etc.)
                        self._dragging_obj = True
                        self._drag_obj = selected_obj
                        ev.accept()
                        return
                
                # While dragging an object
                if self._dragging_obj and self._drag_obj is not None:
                    pos_v = self.mapSceneToView(ev.scenePos())
                    x = float(pos_v.x())
                    
                    # Update the position of the dragged object
                    self._update_dragged_object(x)
                    
                    ev.accept()
                    if ev.isFinish():
                        self._dragging_obj = False
                        self._drag_obj = None
                    return
        except Exception as e:
            print(f"Custom ViewBox mouseDragEvent error: {e}")
        
        # Default behavior (panning/zoom) when not handling custom drag
        super().mouseDragEvent(ev, axis=axis)

    def wheelEvent(self, ev, axis=None):
        """
        Handle mouse wheel events.
        
        Blocks wheel zoom when:
        - Exclude mode is active
        - An object is selected
        Otherwise, allows default zoom behavior.
        """
        try:
            # Keep view fixed while excluding points
            if getattr(self.host, 'exclude_mode', False):
                ev.accept()
                return
            
            # Block wheel zoom when something is selected
            if getattr(self.host, 'selected_kind', None) is not None:
                ev.accept()
                return
        except Exception as e:
            print(f"Custom ViewBox wheelEvent error: {e}")
        
        # Allow default wheel zoom behavior
        super().wheelEvent(ev, axis=axis)
    
    def _finish_exclude_box_selection(self, x, y):
        """
        Complete the exclude box selection by toggling points within the box.
        
        Args:
            x: Current x coordinate
            y: Current y coordinate
        """
        try:
            rx0, rx1 = (self._exclude_start[0], x) if self._exclude_start[0] <= x else (x, self._exclude_start[0])
            ry0, ry1 = (self._exclude_start[1], y) if self._exclude_start[1] <= y else (y, self._exclude_start[1])
            
            # Access data from host
            energy = getattr(self.host, 'energy', None)
            counts = getattr(self.host, 'counts', None)
            
            if energy is not None and counts is not None:
                energy = np.asarray(energy, dtype=float)
                counts = np.asarray(counts, dtype=float)
                
                # Find points within the box
                sel = (energy >= rx0) & (energy <= rx1) & (counts >= ry0) & (counts <= ry1)
                
                if sel.any():
                    # Toggle excluded_mask for selected points
                    excluded_mask = getattr(self.host, 'excluded_mask', None)
                    if not isinstance(excluded_mask, np.ndarray) or len(excluded_mask) != len(energy):
                        excluded_mask = np.zeros(len(energy), dtype=bool)
                        setattr(self.host, 'excluded_mask', excluded_mask)
                    
                    self.host.excluded_mask[sel] = ~self.host.excluded_mask[sel]
                    
                    # Update plot
                    if hasattr(self.host, '_update_data_plot'):
                        self.host._update_data_plot(do_range=False)
                    if hasattr(self.host, 'update_previews'):
                        self.host.update_previews()
        except Exception as e:
            print(f"Failed to finish exclude box selection: {e}")
        finally:
            # Cleanup overlay
            try:
                if self._exclude_rect is not None:
                    if hasattr(self, 'removeItem'):
                        self.removeItem(self._exclude_rect)
                    else:
                        pr = self._exclude_rect.parentItem()
                        if pr is not None:
                            pr.removeFromGroup(self._exclude_rect)
                    self._exclude_rect = None
            except Exception as e:
                print(f"Failed to cleanup exclude rect: {e}")
            
            self._exclude_active = False
            self._exclude_start = None
    
    def _update_dragged_object(self, x):
        """
        Update the position of the currently dragged object.
        
        Args:
            x: New x coordinate for the object
        """
        try:
            # Generic update - set 'center' or 'position' attribute
            if isinstance(self._drag_obj, dict):
                # Dictionary-like object (common pattern)
                self._drag_obj['center'] = x
                
                # Update associated widget if present
                widgets = self._drag_obj.get('widgets', {})
                widget = widgets.get('center_spin') or widgets.get('position_spin')
                if widget is not None:
                    was_blocked = widget.blockSignals(True)
                    widget.setValue(x)
                    widget.blockSignals(was_blocked)
                
                # Invalidate cached values that depend on position
                if 'area' in self._drag_obj:
                    self._drag_obj.pop('area', None)
            else:
                # Object with attributes
                if hasattr(self._drag_obj, 'center'):
                    setattr(self._drag_obj, 'center', x)
                elif hasattr(self._drag_obj, 'position'):
                    setattr(self._drag_obj, 'position', x)
            
            # Request update from host
            if hasattr(self.host, 'update_previews'):
                self.host.update_previews()
        except Exception as e:
            print(f"Failed to update dragged object: {e}")

"""
Input handler module for plot interactions.

This module centralizes mouse and keyboard event handling for interactive plot
manipulation, extracted from PySide_Fitter_PyQtGraph.py patterns.
"""
from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtGui import QKeyEvent, QWheelEvent
import pyqtgraph as pg


class InputHandler(QObject):
    """
    Handles user input events for plot interactions.
    
    Signals:
        mouse_clicked: Emitted when plot is clicked with (x, y, button)
        mouse_moved: Emitted when mouse moves with (x, y) in data coordinates
        key_pressed: Emitted when key is pressed with (key_code, modifiers)
        wheel_scrolled: Emitted when wheel scrolled with (delta, modifiers)
    """
    
    mouse_clicked = Signal(float, float, object)  # x, y, button
    mouse_moved = Signal(float, float)  # x, y in data coordinates
    key_pressed = Signal(int, object)  # key code, modifiers
    wheel_scrolled = Signal(int, object)  # delta, modifiers
    
    def __init__(self, plot_widget):
        """
        Initialize input handler.
        
        Args:
            plot_widget: PyQtGraph PlotWidget to handle events for
        """
        super().__init__()
        self.plot_widget = plot_widget
        self._dragging = False
        self._drag_start = None
        
        # Connect to plot events
        self._connect_events()
    
    def _connect_events(self):
        """Connect to plot widget events."""
        try:
            # Mouse events
            self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_click)
            self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_move)
            
            # Keyboard events - override plot widget methods
            self.plot_widget.keyPressEvent = self._on_key_press
            self.plot_widget.keyReleaseEvent = self._on_key_release
            
            # Install event filter for wheel events
            self.plot_widget.viewport().installEventFilter(self)
        except Exception as e:
            print(f"Warning: Failed to connect some events: {e}")
    
    def _on_mouse_click(self, event):
        """Handle mouse click events."""
        try:
            pos = event.scenePos()
            vb = self.plot_widget.getViewBox()
            if vb is None or not vb.sceneBoundingRect().contains(pos):
                return
            
            # Map to data coordinates
            mousePoint = vb.mapSceneToView(pos)
            x = float(mousePoint.x())
            y = float(mousePoint.y())
            
            # Get button
            try:
                button = event.button()
            except:
                button = None
            
            # Emit signal
            self.mouse_clicked.emit(x, y, button)
            
        except Exception as e:
            print(f"Mouse click error: {e}")
    
    def _on_mouse_move(self, event):
        """Handle mouse move events."""
        try:
            # Get position from event
            try:
                pos = event.scenePos() if hasattr(event, 'scenePos') else event
            except:
                pos = event
            
            vb = self.plot_widget.getViewBox()
            if vb is None:
                return
            
            # Map to data coordinates
            mousePoint = vb.mapSceneToView(pos)
            x = float(mousePoint.x())
            y = float(mousePoint.y())
            
            # Emit signal
            self.mouse_moved.emit(x, y)
            
        except Exception as e:
            print(f"Mouse move error: {e}")
    
    def _on_key_press(self, event):
        """Handle key press events."""
        try:
            key = event.key()
            modifiers = event.modifiers()
            
            # Emit signal
            self.key_pressed.emit(key, modifiers)
            
        except Exception as e:
            print(f"Key press error: {e}")
        
        # Call parent implementation
        try:
            return pg.PlotWidget.keyPressEvent(self.plot_widget, event)
        except:
            pass
    
    def _on_key_release(self, event):
        """Handle key release events."""
        # Call parent implementation
        try:
            return pg.PlotWidget.keyReleaseEvent(self.plot_widget, event)
        except:
            pass
    
    def eventFilter(self, obj, event):
        """Filter events, especially for wheel events."""
        try:
            if isinstance(event, QWheelEvent):
                delta = event.angleDelta().y()
                modifiers = event.modifiers()
                
                # Emit signal
                self.wheel_scrolled.emit(delta, modifiers)
                
                # Check if we should handle it (block default zoom behavior)
                # Return True to block, False to allow default
                # For now, let the handler decide via signal
                
        except Exception as e:
            print(f"Event filter error: {e}")
        
        # Call parent implementation
        return super().eventFilter(obj, event)
    
    def map_to_data_coords(self, scene_x, scene_y):
        """
        Map scene coordinates to data coordinates.
        
        Args:
            scene_x: X coordinate in scene
            scene_y: Y coordinate in scene
            
        Returns:
            tuple: (x, y) in data coordinates, or (None, None) if mapping fails
        """
        try:
            vb = self.plot_widget.getViewBox()
            if vb is None:
                return None, None
            
            mousePoint = vb.mapSceneToView(pg.Point(scene_x, scene_y))
            return float(mousePoint.x()), float(mousePoint.y())
        except:
            return None, None

"""
Input handler module for plot interactions.

This module centralizes mouse and keyboard event handling for interactive plot
manipulation, extracted from PySide_Fitter_PyQtGraph.py patterns.
"""
from PySide6.QtCore import QObject, Signal, Qt, QEvent
from PySide6.QtGui import QKeyEvent, QWheelEvent, QMouseEvent
import pyqtgraph as pg


class InputHandler(QObject):
    """
    Handles user input events for plot interactions.
    
    Signals:
        mouse_clicked: Emitted when plot is clicked with (x, y, button)
        mouse_moved: Emitted when mouse moves with (x, y, buttons) in data coordinates
        key_pressed: Emitted when key is pressed with (key_code, modifiers)
        wheel_scrolled: Emitted when wheel scrolled with (delta, modifiers)
    """
    
    mouse_clicked = Signal(float, float, object)  # x, y, button
    mouse_moved = Signal(float, float, object)  # x, y in data coordinates, buttons
    key_pressed = Signal(int, object)  # key code, modifiers
    wheel_scrolled = Signal(int, object)  # delta, modifiers
    # Debug/interaction signals
    mouse_pressed = Signal(float, float, object, object)  # x, y, button, items
    mouse_released = Signal(float, float, object, object)  # x, y, button, items
    mouse_dragged = Signal(float, float, float)  # dx, dy, distance
    
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
            # install event filter on the viewport so we can see raw mouse press/release/move
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
            
            # Determine button state if available on the event
            buttons = 0
            try:
                # QGraphicsSceneMouseEvent has buttons()
                if hasattr(event, "buttons"):
                    buttons = event.buttons()
            except Exception:
                buttons = 0
            
            # Emit signal (x, y, buttons)
            self.mouse_moved.emit(x, y, buttons)
            
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
            # Wheel events
            if isinstance(event, QWheelEvent):
                delta = event.angleDelta().y()
                modifiers = event.modifiers()
                # Emit signal
                self.wheel_scrolled.emit(delta, modifiers)

            # Mouse events from the viewport: map them into scene and data coordinates
            # Use QEvent types so we only handle these event kinds here.
            if isinstance(event, QMouseEvent) or event.type() in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseMove):
                try:
                    views = self.plot_widget.scene().views()
                    if not views:
                        return super().eventFilter(obj, event)
                    view = views[0]
                    # map viewport widget coordinates to scene coordinates
                    scene_pos = view.mapToScene(event.pos())
                    vb = self.plot_widget.getViewBox()
                    if vb is None:
                        return super().eventFilter(obj, event)

                    # Map to data coords
                    try:
                        mousePoint = vb.mapSceneToView(scene_pos)
                        x = float(mousePoint.x())
                        y = float(mousePoint.y())
                    except Exception:
                        x, y = None, None

                    # identify nearby items under the scene position for debugging
                    items = self._identify_items_at(scene_pos)

                    # handle press
                    if event.type() == QEvent.MouseButtonPress:
                        try:
                            btn = event.button()
                        except Exception:
                            btn = None
                        # start drag
                        self._dragging = True
                        self._drag_start = (x, y)
                        # emit pressed debug signal
                        try:
                            self.mouse_pressed.emit(x, y, btn, items)
                        except Exception:
                            pass

                    # handle release
                    if event.type() == QEvent.MouseButtonRelease:
                        try:
                            btn = event.button()
                        except Exception:
                            btn = None
                        # compute drag distance if we started dragging
                        dx = dy = dist = 0.0
                        if self._dragging and self._drag_start is not None and x is not None and y is not None:
                            try:
                                sx, sy = self._drag_start
                                dx = float(x - sx)
                                dy = float(y - sy)
                                from math import hypot
                                dist = float(hypot(dx, dy))
                            except Exception:
                                dx = dy = dist = 0.0
                        # emit released and dragged signals
                        try:
                            self.mouse_released.emit(x, y, btn, items)
                        except Exception:
                            pass
                        try:
                            # emit drag only if distance non-zero
                            if dist is not None:
                                self.mouse_dragged.emit(dx, dy, dist)
                        except Exception:
                            pass
                        # reset drag state
                        self._dragging = False
                        self._drag_start = None

                    # For mouse move: optionally emit existing mouse_moved (keep compatibility)
                    if event.type() == QEvent.MouseMove:
                        # try to retrieve button state
                        buttons = 0
                        try:
                            if hasattr(event, 'buttons'):
                                buttons = event.buttons()
                        except Exception:
                            buttons = 0
                        try:
                            if x is not None and y is not None:
                                self.mouse_moved.emit(x, y, buttons)
                        except Exception:
                            pass

                except Exception as e:
                    print(f"Event filter error (mouse): {e}")
                
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

    def _identify_items_at(self, scene_pos):
        """Return a list of human-readable item descriptions under scene_pos."""
        try:
            items = self.plot_widget.scene().items(scene_pos)
            desc = []
            for it in items:
                try:
                    # Prefer friendly names for common pyqtgraph items
                    if isinstance(it, pg.ScatterPlotItem):
                        desc.append("ScatterPlotItem")
                    elif isinstance(it, pg.PlotDataItem):
                        name = getattr(it, 'name', None)
                        if name:
                            desc.append(f"PlotDataItem({name})")
                        else:
                            desc.append("PlotDataItem")
                    else:
                        nm = getattr(it, 'name', None) or type(it).__name__
                        desc.append(str(nm))
                except Exception:
                    try:
                        desc.append(type(it).__name__)
                    except Exception:
                        desc.append("UnknownItem")
            return desc
        except Exception:
            return []

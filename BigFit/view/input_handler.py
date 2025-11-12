# view/input_handler.py
# type: ignore
from PySide6.QtCore import QObject, Qt, QPointF, QEvent
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
import numpy as np
from models import CompositeModelSpec


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
        self.selected_component_prefix = None

        # control mapping for interactive parameter binding
        # keys: (action:str, modifiers: tuple(sorted modifier names)) -> list of {name, step}
        self.control_map = {}
        # last mouse data coordinate used for mouse_move controls
        self._last_mouse_data_x = None

        if self.viewbox is not None:
            self._connect_viewbox()

    def set_viewmodel(self, vm):
        self.viewmodel = vm

    def set_control_map(self, control_map: dict):
        """Provide control mapping from the view. Expected shape:
        { (action, tuple(modifiers)): [ { 'name': str, 'step': float, ... }, ... ] }
        """
        out = {}
        for k, v in (control_map or {}).items():
            try:
                # accept keys as (action, modifiers) or as a simple action string
                if isinstance(k, (list, tuple)) and len(k) >= 2:
                    action = k[0]
                    mods = tuple(sorted([str(m) for m in k[1]]))
                elif isinstance(k, (list, tuple)) and len(k) == 1:
                    action = k[0]
                    mods = tuple()
                else:
                    action = k
                    mods = tuple()
                out[(action, mods)] = list(v)
            except Exception:
                continue
        self.control_map = out

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

        # Wheel events (map to parameter adjustments)
        if isinstance(event, QWheelEvent) or event.type() == QEvent.Wheel:
            try:
                handled = self.on_wheel(obj, event)
                if handled:
                    return True
            except Exception:
                pass

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
        # Only allow peak selection if a curve is explicitly selected
        if self.selected_curve_id is None:
            # ignore selection and keep UI/plot behavior unchanged
            try:
                if hasattr(self.viewmodel, "log_message"):
                    self.viewmodel.log_message.emit("Peak selected but no curve is active — ignoring.")
            except Exception:
                pass
            return

        idx = self.find_nearest_peak(x)
        if idx is not None:
            self.dragging_peak = idx
            self.drag_offset = self.viewmodel.peaks[idx] - x
            try:
                if hasattr(self.viewmodel, "log_message"):
                    self.viewmodel.log_message.emit(f"Selected peak #{idx} at x={self.viewmodel.peaks[idx]:.3f}")
            except Exception:
                pass

    def on_peak_moved(self, peak_info):
        """User dragged a peak."""
        if self.viewmodel is None or self.dragging_peak is None:
            return
        new_x = float(peak_info.get("center", 0.0))
        # update local peak store (ViewModel or example harness may expose .peaks)
        try:
            self.viewmodel.peaks[self.dragging_peak] = new_x
        except Exception:
            pass

        # Prefer centralized dispatch if available so peak movement is handled
        # consistently by the ViewModel. Fallback to update_plot if needed.
        handled = False
        if hasattr(self.viewmodel, "handle_action"):
            try:
                self.viewmodel.handle_action("on_peak_moved", info=peak_info)
                handled = True
            except Exception:
                handled = False
        if not handled and hasattr(self.viewmodel, "update_plot"):
            try:
                self.viewmodel.update_plot()
            except Exception:
    def on_exclude_point(self, x, y):
        """Toggle exclusion of a single data point."""
        if self.viewmodel is None:
            return

        idx = self._find_nearest_data_index(x, y)
        self._toggle_exclusion(idx, x, y)

        try:
            # Maintain the higher-level log for visual trace; ViewModel also logs index/result.
            if hasattr(self.viewmodel, "log_message"):
                self.viewmodel.log_message.emit(f"Toggled exclusion at ({x:.2f}, {y:.2f})")
        except Exception:
            pass

    def _find_nearest_data_index(self, x, y):
        """Helper to find the nearest data index to (x, y) in scene coordinates."""
        idx = None
        try:
            tol_px = 8
            try:
                from view.constants import CURVE_SELECT_TOL_PIXELS
                tol_px = CURVE_SELECT_TOL_PIXELS
            except Exception:
                pass

            if hasattr(self, "viewbox") and getattr(self, "viewbox", None) is not None:
                vb = self.viewbox
                try:
                    from PySide6.QtCore import QPointF
                    click_scene_pt = vb.mapViewToScene(QPointF(float(x), float(y)))
                    click_scene = (float(click_scene_pt.x()), float(click_scene_pt.y()))
                    xd = getattr(self.viewmodel.state, "x_data", None)
                    yd = getattr(self.viewmodel.state, "y_data", None)
                    if xd is not None and yd is not None:
                        xd = np.asarray(xd)
                        yd = np.asarray(yd)
                        best_i = None
                        best_d2 = None
                        for i, (xi, yi) in enumerate(zip(xd, yd)):
                            try:
                                pt = vb.mapViewToScene(QPointF(float(xi), float(yi)))
                                dx = float(pt.x()) - click_scene[0]
                                dy = float(pt.y()) - click_scene[1]
                                d2 = dx * dx + dy * dy
                                if best_d2 is None or d2 < best_d2:
                                    best_d2 = d2
                                    best_i = i
                            except Exception:
                                continue
                        if best_i is not None and best_d2 is not None:
                            if best_d2 <= (float(tol_px) * float(tol_px)):
                                idx = int(best_i)
                except Exception:
                    idx = None
        except Exception:
            idx = None
        return idx

    def _toggle_exclusion(self, idx, x, y):
        """Helper to toggle exclusion by index or coordinate, with simplified fallback."""
        try:
            if idx is not None:
                # Try index-based exclusion first
                if hasattr(self.viewmodel, "handle_action"):
                    try:
                        self.viewmodel.handle_action("toggle_point_exclusion_by_index", idx=int(idx))
                        return
                    except Exception:
                        pass
                if hasattr(self.viewmodel, "toggle_point_exclusion_by_index"):
                    try:
                        self.viewmodel.toggle_point_exclusion_by_index(int(idx))
                        return
                    except Exception:
                        pass
            # Fallback to coordinate-based exclusion
            if hasattr(self.viewmodel, "handle_action"):
                try:
                    self.viewmodel.handle_action("toggle_point_exclusion", x=x, y=y)
                    return
                except Exception:
                    pass
            if hasattr(self.viewmodel, "toggle_point_exclusion"):
                try:
                    self.viewmodel.toggle_point_exclusion(x, y)
                except Exception:
                    pass
        except Exception:
            pass
        except Exception:
            pass

    def on_exclude_box(self, x0, y0, x1, y1):
        """Box-drag exclusion event."""
        if self.viewmodel is None:
            return
        if hasattr(self.viewmodel, "handle_action"):
            try:
                self.viewmodel.handle_action("toggle_box_exclusion", x0=x0, y0=y0, x1=x1, y1=y1)
            except Exception:
                # fallback
                try:
                    if hasattr(self.viewmodel, "toggle_box_exclusion"):
                        self.viewmodel.toggle_box_exclusion(x0, y0, x1, y1)
                except Exception:
                    pass
        else:
            if hasattr(self.viewmodel, "toggle_box_exclusion"):
                try:
                    self.viewmodel.toggle_box_exclusion(x0, y0, x1, y1)
                except Exception:
                    pass

        try:
            if hasattr(self.viewmodel, "log_message"):
                self.viewmodel.log_message.emit(f"Exclusion box: ({x0:.2f},{y0:.2f}) → ({x1:.2f},{y1:.2f})")
        except Exception:
            pass

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

        # If no peak drag is active, map mouse movement to parameter controls if configured.
        # Only do interactive parameter mapping when a curve is selected to avoid
        # interfering with normal pan/zoom behavior.
        if self.selected_curve_id is None:
            # keep last mouse x in sync so first move after selection initializes correctly
            try:
                x, y = self.mouse_to_data(obj, event.pos())
                self._last_mouse_data_x = x
            except Exception:
                pass
            return False

        if self.control_map and self.viewmodel is not None:
            x, y = self.mouse_to_data(obj, event.pos())
            try:
                mods = event.modifiers()
            except Exception:
                mods = Qt.NoModifier
            mod_names = self._modifiers_to_names(mods)
            key = ("mouse_move", mod_names)
            if key not in self.control_map:
                key = ("mouse_move", tuple())
                if key not in self.control_map:
                    # update last mouse x and return
                    self._last_mouse_data_x = x
                    return False

            last_x = self._last_mouse_data_x
            self._last_mouse_data_x = x
            if last_x is None:
                return False

            dx = x - last_x
            if abs(dx) == 0:
                return False

            updates = {}
            try:
                current_specs = self.viewmodel.get_parameters()
            except Exception:
                current_specs = {}

            for entry in self.control_map.get(key, []):
                try:
                    if not isinstance(entry, dict):
                        continue
                    entry_component = entry.get("component")
                    if entry_component and entry_component != self.selected_component_prefix:
                        continue
                    name = entry.get("name")
                    step_val = float(entry.get("step", 1.0))
                    cur = None
                    if name in current_specs and isinstance(current_specs[name], dict):
                        cur = current_specs[name].get("value")
                    if cur is None:
                        mdl = getattr(self.viewmodel.state, "model", None)
                        if mdl is not None and hasattr(mdl, name):
                            cur = getattr(mdl, name)
                    if cur is None:
                        spec = getattr(self.viewmodel.state, "model_spec", None)
                        if spec is not None and name in getattr(spec, "params", {}):
                            cur = getattr(spec.params[name], "value", None)
                    if cur is None:
                        continue
                    new_val = cur + step_val * float(dx)
                    if name in current_specs and isinstance(current_specs[name], dict):
                        mn = current_specs[name].get("min", None)
                        mx = current_specs[name].get("max", None)
                        if mn is not None:
                            new_val = max(new_val, mn)
                        if mx is not None:
                            new_val = min(new_val, mx)
                    updates[name] = new_val
            if updates:
                try:
                    if hasattr(self.viewmodel, "handle_action"):
                        try:
                            self.viewmodel.handle_action("apply_parameters", params=updates)
                        except Exception:
                            try:
                                if hasattr(self.viewmodel, "apply_parameters"):
                                    self.viewmodel.apply_parameters(updates)
                            except Exception:
                                pass
                    elif hasattr(self.viewmodel, "apply_parameters"):
                        try:
                            self.viewmodel.apply_parameters(updates)
                        except Exception:
                            pass
                    if hasattr(self.viewmodel, "log_message"):
                        try:
                            self.viewmodel.log_message.emit(f"Interactive update: {updates}")
                        except Exception:
                            pass
                except Exception:
                    pass
                return True
                except Exception:
                    pass
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

        if Qt.Key_0 <= key <= Qt.Key_9:
            if self._handle_numeric_hotkey(key):
                return True

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
        elif key == Qt.Key_F:
            if self.viewmodel is not None:
                if hasattr(self.viewmodel, "handle_action"):
                    try:
                        self.viewmodel.handle_action("run_fit")
                    except Exception:
                        # fallback
                        try:
                            if hasattr(self.viewmodel, "run_fit"):
                                self.viewmodel.run_fit()
                        except Exception:
                            pass
                else:
                    try:
                        if hasattr(self.viewmodel, "run_fit"):
                            self.viewmodel.run_fit()
                    except Exception:
                        pass
            return True
        elif key in (Qt.Key_E, Qt.Key_Q):
            step = 1 if key == Qt.Key_E else -1
            if self._cycle_selected_component(step):
                return True
        elif key == Qt.Key_D:
            # Toggle exclude mode on the ViewBox
            try:
                if self.viewbox is not None and hasattr(self.viewbox, "set_exclude_mode"):
                    current = getattr(self.viewbox, "exclude_mode", False)
                    new = not bool(current)
                    try:
                        self.viewbox.set_exclude_mode(new)
                    except Exception:
                        pass
                    # notify via ViewModel log if available
                    try:
                        if self.viewmodel is not None and hasattr(self.viewmodel, "log_message"):
                            self.viewmodel.log_message.emit(f"Exclude mode {'enabled' if new else 'disabled'} (hotkey)")
                    except Exception:
                        pass
                    return True
            except Exception:
                pass
        return False

    # -----------------------
    # Selection management
    # -----------------------
    def set_selected_curve(self, curve_id, notify_vm: bool = True):
        """Set selected curve ID locally. If notify_vm is True, also inform the ViewModel.

        When called from the ViewModel (e.g. external selection change), pass notify_vm=False
        to avoid cycles.
        """
        normalized_id = curve_id
        if curve_id == "fit" and self._composite_has_components():
            normalized_id = None

        prefix = None
        if normalized_id and str(normalized_id).startswith("component:"):
            prefix = str(normalized_id).split(":", 1)[1]
        self.selected_component_prefix = prefix

        if self.selected_curve_id != normalized_id:
            self.selected_curve_id = normalized_id
            if notify_vm and hasattr(self.viewmodel, "set_selected_curve"):
                try:
                    self.viewmodel.set_selected_curve(normalized_id)
                except Exception:
                    pass

        if normalized_id is not None and self.viewmodel is not None and hasattr(self.viewmodel, "log_message"):
            try:
                self.viewmodel.log_message.emit(f"Curve '{normalized_id}' selected.")
            except Exception:
                pass

    def clear_selected_curve(self):
        """Clear selection and notify ViewModel."""
        # Always clear local state and instruct the ViewModel to clear its selection.
        # This ensures deselect works even if selection was set directly on the ViewModel
        # (e.g. via MainWindow._on_curve_clicked) and InputHandler.selected_curve_id is stale.
        self.selected_curve_id = None
        self.selected_component_prefix = None
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

    def _cycle_selected_component(self, step: int) -> bool:
        if step == 0:
            return False
        if self.viewmodel is None:
            return False

        descriptors = []
        try:
            if hasattr(self.viewmodel, "get_component_descriptors"):
                descriptors = self.viewmodel.get_component_descriptors() or []
        except Exception:
            descriptors = []

        curve_order = []
        for desc in descriptors:
            prefix = desc.get("prefix")
            if not prefix:
                continue
            curve_order.append(f"component:{prefix}")

        if not curve_order:
            try:
                existing = list(getattr(self.viewmodel, "curves", {}).keys())
            except Exception:
                existing = []
            curve_order = [cid for cid in existing if cid]
            if not curve_order:
                return False

        current = self.selected_curve_id if self.selected_curve_id in curve_order else None
        if current is None:
            target = curve_order[0] if step > 0 else curve_order[-1]
        else:
            idx = curve_order.index(current)
            target = curve_order[(idx + step) % len(curve_order)]

        self.set_selected_curve(target)
        return True

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

    # Use shared constant default for selection tolerance so it's easy to tune
    from view.constants import CURVE_SELECT_TOL_PIXELS

    def detect_curve_at(self, plot_widget, x, y, tol_pixels: int = CURVE_SELECT_TOL_PIXELS):
        """
        Attempt to detect which curve (if any) was clicked at data coords (x, y).

        Uses a pixel-space distance test (scene coordinates) so selection is
        independent of data scaling. Returns a curve_id or None.

        tol_pixels: maximum distance in screen pixels for a click to count as
                    "on the curve" (default 8 px).
        """
        if self.viewmodel is None or not hasattr(self.viewmodel, "curves"):
            return None

        composite_has_components = self._composite_has_components()

        try:
            vb = plot_widget.getViewBox()
        except Exception:
            vb = None

        # Map the clicked data coord to scene (pixel) coords if possible.
        click_scene = None
        if vb is not None:
            try:
                from PySide6.QtCore import QPointF

                click_scene_pt = vb.mapViewToScene(QPointF(float(x), float(y)))
                click_scene = (float(click_scene_pt.x()), float(click_scene_pt.y()))
            except Exception:
                click_scene = None

        # Fallback: if we couldn't map to scene coords, use a conservative
        # data-space distance test (small tolerance) so we don't select
        # everything. This happens rarely but keeps behavior safe.
        if click_scene is None:
            for cid, (cx, cy) in self.viewmodel.curves.items():
                if composite_has_components and cid == "fit":
                    continue
                cx = np.asarray(cx)
                cy = np.asarray(cy)
                if len(cx) == 0:
                    continue
                dist = np.min(np.sqrt((cx - x) ** 2 + (cy - y) ** 2))
                if dist < 1e-6:  # extremely small fallback
                    return cid
            return None

        # Otherwise compute pixel distances. This is a bit more expensive
        # (maps curve points to scene coords) but remains fast for typical
        # plotted arrays. We return the first curve within tol_pixels.
        tpx = float(tol_pixels)
        from PySide6.QtCore import QPointF

        for cid, (cx, cy) in self.viewmodel.curves.items():
            if composite_has_components and cid == "fit":
                continue
            cx = np.asarray(cx)
            cy = np.asarray(cy)
            if len(cx) == 0:
                continue
            try:
                # Map each data point to scene coords and compute min distance
                # to the click. We'll bail out early if any point is close enough.
                for xi, yi in zip(cx, cy):
                    pt = vb.mapViewToScene(QPointF(float(xi), float(yi)))
                    dx = float(pt.x()) - click_scene[0]
                    dy = float(pt.y()) - click_scene[1]
                    if (dx * dx + dy * dy) <= (tpx * tpx):
                        return cid
            except Exception:
                # If mapping fails for this curve, skip it.
                continue
        return None

    # -----------------------
    # Wheel / mouse-move control handlers
    # -----------------------
    def _modifiers_to_names(self, mods) -> tuple:
        names = []
        try:
            if mods & Qt.ControlModifier:
                names.append("Control")
            if mods & Qt.ShiftModifier:
                names.append("Shift")
            if mods & Qt.AltModifier:
                names.append("Alt")
        except Exception:
            pass
        return tuple(sorted(names))

    def _handle_numeric_hotkey(self, key) -> bool:
        """Support number-key shortcuts for selecting components."""
        try:
            descriptors = []
            if self.viewmodel and hasattr(self.viewmodel, "get_component_descriptors"):
                descriptors = self.viewmodel.get_component_descriptors() or []
        except Exception:
            descriptors = []

        composite_has_components = self._composite_has_components()

        if not descriptors:
            if key == Qt.Key_0:
                self.clear_selected_curve()
                return True
            if key == Qt.Key_1 and not composite_has_components:
                self.set_selected_curve("fit")
                return True
            return False

        if key == Qt.Key_0:
            self.clear_selected_curve()
            return True

        index = key - Qt.Key_1
        if index < 0 or index >= len(descriptors):
            return False

        prefix = descriptors[index].get("prefix") if isinstance(descriptors[index], dict) else None
        if not prefix:
            return False
        curve_id = f"component:{prefix}"
        self.set_selected_curve(curve_id)
        return True

    def _is_composite_active(self) -> bool:
        try:
            if self.viewmodel is None or not hasattr(self.viewmodel, "state"):
                return False
            spec = getattr(self.viewmodel.state, "model_spec", None)
            return isinstance(spec, CompositeModelSpec)
        except Exception:
            return False

    def _composite_has_components(self) -> bool:
        if not self._is_composite_active():
            return False
        try:
            if self.viewmodel and hasattr(self.viewmodel, "get_component_descriptors"):
                descriptors = self.viewmodel.get_component_descriptors() or []
                return bool(descriptors)
        except Exception:
            return False
        return False

    def on_wheel(self, obj, event):
        """Handle wheel events and map to parameter changes based on control_map."""
        # Only intercept wheel for parameter control when a curve is selected
        if self.selected_curve_id is None:
            return False
        if not self.control_map or self.viewmodel is None:
            return False
        try:
            mods = event.modifiers()
        except Exception:
            mods = Qt.NoModifier
        mod_names = self._modifiers_to_names(mods)

        key = ("wheel", mod_names)
        if key not in self.control_map:
            key = ("wheel", tuple())
            if key not in self.control_map:
                return False

        try:
            delta_units = event.angleDelta().y() / 120.0
        except Exception:
            try:
                delta_units = event.delta() / 120.0
            except Exception:
                delta_units = 0.0

        updates = {}
        try:
            current_specs = self.viewmodel.get_parameters()
        except Exception:
            current_specs = {}

        for entry in self.control_map.get(key, []):
            try:
                if not isinstance(entry, dict):
                    continue
                entry_component = entry.get("component")
                if entry_component and entry_component != self.selected_component_prefix:
                    continue
                name = entry.get("name")
                step_val = float(entry.get("step", 1.0))
                cur = None
                if name in current_specs and isinstance(current_specs[name], dict):
                    cur = current_specs[name].get("value")
                if cur is None:
                    mdl = getattr(self.viewmodel.state, "model", None)
                    if mdl is not None and hasattr(mdl, name):
                        cur = getattr(mdl, name)
                if cur is None:
                    spec = getattr(self.viewmodel.state, "model_spec", None)
                    if spec is not None and name in getattr(spec, "params", {}):
                        cur = getattr(spec.params[name], "value", None)
                if cur is None:
                    continue
                new_val = cur + step_val * float(delta_units)
                if name in current_specs and isinstance(current_specs[name], dict):
                    mn = current_specs[name].get("min", None)
                    mx = current_specs[name].get("max", None)
                    if mn is not None:
                        new_val = max(new_val, mn)
                    if mx is not None:
                        new_val = min(new_val, mx)
                updates[name] = new_val
        if updates:
            try:
                if hasattr(self.viewmodel, "handle_action"):
                    try:
                        self.viewmodel.handle_action("apply_parameters", params=updates)
                    except Exception:
                        try:
                            if hasattr(self.viewmodel, "apply_parameters"):
                                self.viewmodel.apply_parameters(updates)
                        except Exception:
                            pass
                elif hasattr(self.viewmodel, "apply_parameters"):
                    try:
                        self.viewmodel.apply_parameters(updates)
                    except Exception:
                        pass
                # accept/consume the Qt event so the ViewBox does not perform zoom
                try:
                    event.accept()
                except Exception:
                    pass
                if hasattr(self.viewmodel, "log_message"):
                    try:
                        # emit a concise interactive message for visibility
                        self.viewmodel.log_message.emit(f"Interactive update: {updates}")
                    except Exception:
                        pass
            except Exception:
                pass
            return True
        return False
                pass
            return True
        return False

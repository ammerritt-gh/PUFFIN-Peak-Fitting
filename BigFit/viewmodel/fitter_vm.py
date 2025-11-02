# viewmodel/fitter_vm.py
from PySide6.QtCore import QObject, Signal
import numpy as np
from types import SimpleNamespace

from models import ModelState
from dataio.data_loader import select_and_load_files
from dataio.data_saver import save_dataset
import typing as _typing


class FitterViewModel(QObject):
    """
    Central logic layer: handles loading/saving, fitting, and updates to the plot.
    """

    plot_updated = Signal(object, object, object, object)  # x, y_data, y_fit, y_err
    log_message = Signal(str)
    parameters_updated = Signal()

    def __init__(self, model_state=None):
        super().__init__()
        self.state = model_state or ModelState()
        # mapping of input event -> list of handlers built from parameter specs
        self._input_map = {}

    # --------------------------
    # Data I/O
    # --------------------------
    def load_data(self):
        """Open file dialog and load a dataset."""
        loaded = select_and_load_files(None)
        if not loaded:
            return
        x, y, err, info = loaded[0]
        # Ensure numeric numpy arrays and a well-formed error array
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if err is None:
            err = np.sqrt(np.clip(np.abs(y), 1e-12, np.inf))
        else:
            err = np.asarray(err, dtype=float)
        # Trim to the shortest common length to avoid mismatched arrays
        common_len = min(len(x), len(y), len(err))
        x = x[:common_len]
        y = y[:common_len]
        err = err[:common_len]
        self.state.x_data = x
        self.state.y_data = y
        self.state.errors = err
        self.state.file_info = info
        self.log_message.emit(f"Loaded data file: {info['name']}")
        self.update_plot()

    def save_data(self):
        """Save current data and fit to file."""
        y_fit = self.state.evaluate() if hasattr(self.state, "evaluate") else None
        save_dataset(self.state.x_data, self.state.y_data, y_fit=y_fit)
        self.log_message.emit("Data saved successfully.")

    # --------------------------
    # Configuration accessors (ViewModel handles logic/persistence)
    # --------------------------
    def get_config(self) -> dict:
        """Return a minimal dict of configuration values for the view to edit."""
        try:
            # lazy import to avoid circular import
            from dataio import get_config as _get_cfg
            cfg = _get_cfg()
            return {
                "default_load_folder": cfg.default_load_folder,
                "default_save_folder": cfg.default_save_folder,
                "config_path": str(cfg.config_path),
            }
        except Exception as e:
            raise RuntimeError(f"Unable to read configuration: {e}") from e

    def save_config(self, default_load_folder: _typing.Optional[str] = None, default_save_folder: _typing.Optional[str] = None) -> bool:
        """Save provided configuration fields to disk. Returns True on success."""
        try:
            from dataio import get_config as _get_cfg
            # get singleton, update fields, save
            cfg = _get_cfg()
            if default_load_folder is not None:
                cfg.default_load_folder = str(default_load_folder)
            if default_save_folder is not None:
                cfg.default_save_folder = str(default_save_folder)
            cfg.save()
            # force reload of singleton so other callers see changes
            _get_cfg(recreate=True)
            self.log_message.emit("Configuration saved.")
            return True
        except Exception as e:
            self.log_message.emit(f"Failed to save configuration: {e}")
            return False

    def reload_config(self):
        """Reload the project's configuration from disk and notify UI."""
        try:
            # import here to avoid circular import at module import time
            from dataio import get_config
            cfg = get_config(recreate=True)
            self.log_message.emit(f"Config reloaded from: {cfg.config_path}")
        except Exception as e:
            self.log_message.emit(f"Failed to reload config: {e}")
        # refresh plot in case default folders or parameters changed
        try:
            self.update_plot()
        except Exception:
            pass

    # --------------------------
    # Fit + Plot logic
    # --------------------------
    def run_fit(self):
        """Start an asynchronous fit using FitWorker (non-blocking)."""
        # Prevent multiple concurrent fits
        if getattr(self, "_fit_worker", None) is not None:
            self.log_message.emit("A fit is already running.")
            return

        # basic data checks
        x = getattr(self.state, "x_data", None)
        y = getattr(self.state, "y_data", None)
        if x is None or y is None:
            self.log_message.emit("No data available to fit.")
            return

        err = getattr(self.state, "errors", None)

        # obtain model_spec (use existing or create one)
        model_spec = getattr(self.state, "model_spec", None)
        if model_spec is None:
            try:
                from models import get_model_spec
                model_spec = get_model_spec(getattr(self.state, "model_name", "voigt"))
                setattr(self.state, "model_spec", model_spec)
            except Exception as e:
                self.log_message.emit(f"Unable to obtain model specification: {e}")
                return

        # ensure model_spec provides an evaluate function
        model_func = getattr(model_spec, "evaluate", None)
        if model_func is None:
            self.log_message.emit("Selected model does not provide an evaluate(x, **params) function.")
            return

        # build initial parameters and bounds from model_spec.params
        try:
            params = {k: getattr(v, "value", v) for k, v in getattr(model_spec, "params", {}).items()}
            lower = [v.min if getattr(v, "min", None) is not None else -np.inf for v in getattr(model_spec, "params", {}).values()]
            upper = [v.max if getattr(v, "max", None) is not None else np.inf for v in getattr(model_spec, "params", {}).values()]
            bounds = (lower, upper)
        except Exception as e:
            self.log_message.emit(f"Failed to build parameter/bounds list: {e}")
            return

        # wrap evaluate(x, **params) into curve_fit-style f(x, *args)
        param_keys = list(params.keys())

        def wrapped_func(xx, *args):
            # build param dict from positional args (curve_fit provides positional params)
            p = dict(zip(param_keys, args))
            # IMPORTANT: some ModelSpec.evaluate implementations expect a single 'params' dict
            # as the second positional argument (evaluate(x, params)). Others accept **kwargs.
            # Pass the params dict positionally to support evaluate(x, params).
            try:
                return model_func(xx, p)
            except TypeError:
                # fallback: try as kwargs (in case evaluate accepts keyword args)
                return model_func(xx, **p)

        # lazy import of FitWorker to reduce module-time coupling
        try:
            from worker.fit_worker import FitWorker
        except Exception as e:
            self.log_message.emit(f"Failed to import FitWorker: {e}")
            return

        worker = FitWorker(x, y, wrapped_func, params, err, bounds)
        self._fit_worker = worker

        # connect progress updates
        worker.progress.connect(lambda p: self.log_message.emit(f"Fit progress: {int(p*100)}%"))

        # finished handler
        def on_finished(result, y_fit):
            try:
                if result is not None:
                    # store fit result on state and update plot
                    self.state.fit_result = result

                    # Apply fit result back into model_spec.params where possible so UI will reflect fitted values
                    try:
                        spec = getattr(self.state, "model_spec", None)
                        if spec is not None and hasattr(spec, "params"):
                            for k, v in result.items():
                                if k in spec.params:
                                    try:
                                        spec.params[k].value = v
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                    # Also set attributes on any concrete model object for consistency
                    try:
                        mdl = getattr(self.state, "model", None)
                        if mdl is not None:
                            for k, v in result.items():
                                try:
                                    setattr(mdl, k, v)
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    # Notify views that parameters changed so UI can refresh widgets
                    try:
                        self.parameters_updated.emit()
                    except Exception:
                        pass

                    self.plot_updated.emit(x, y, y_fit, err)
                    self.log_message.emit("Fit completed successfully.")
                else:
                    self.log_message.emit("Fit failed.")
            finally:
                # always clear the worker reference
                try:
                    self._fit_worker = None
                except Exception:
                    pass

        worker.finished.connect(on_finished)
        worker.start()
        self.log_message.emit("Fit started in background...")

    def update_plot(self):
        """Update plot without running a fit."""
        y_fit = None
        if hasattr(self.state, "evaluate"):
            try:
                y_fit = self.state.evaluate()
            except Exception:
                y_fit = None
        errs = getattr(self.state, "errors", None)
        errs = None if errs is None else np.asarray(errs, dtype=float)
        self.plot_updated.emit(self.state.x_data, self.state.y_data, y_fit, errs)

    def get_parameters(self) -> dict:
        """Return parameter specs for the current model.

        This implementation:
          - determines a canonical model key (normalizes UI names like "DHO+Voigt"),
          - fetches a ModelSpec via get_model_spec(key),
          - builds a dict directly from model_spec.params (Parameter objects),
          - overrides values from a concrete state.model if present.
        """
        try:
            # local import to avoid circular import problems at module import time
            from models import get_model_spec, canonical_model_key
            # Helper to log safely
            def _log(msg):
                try:
                    self.log_message.emit(msg)
                except Exception:
                    print(msg)

            # Determine model_name (prefer explicit state.model_name)
            model_name = getattr(self.state, "model_name", None)
            if model_name is None:
                mdl = getattr(self.state, "model", None)
                if mdl is not None:
                    model_name = getattr(mdl, "name", None) or mdl.__class__.__name__
            if model_name is None:
                model_name = "voigt"

            _log(f"get_parameters: requested model_name raw='{model_name}'")

            # Normalize the requested model name to the canonical registry key
            key = str(model_name).strip()
            canonical = canonical_model_key(key)
            _log(f"get_parameters: normalized model key='{canonical}' (from '{model_name}')")

            # Obtain spec instance (using local get_model_spec)
            try:
                model_spec = get_model_spec(canonical)
            except Exception as e:
                _log(f"get_parameters: get_model_spec('{canonical}') raised: {e}")
                return {}

            # If model_spec has initialize hook, call it with data
            try:
                model_spec.initialize(getattr(self.state, "x_data", None), getattr(self.state, "y_data", None))
            except Exception as e:
                _log(f"get_parameters: model_spec.initialize() raised: {e}")

            # Build specs dict from Parameter objects if present
            specs = {}
            try:
                params_map = getattr(model_spec, "params", None) or {}
                for pname, pobj in params_map.items():
                    try:
                        # Parameter.to_spec() expected
                        if hasattr(pobj, "to_spec"):
                            specs[pname] = pobj.to_spec()
                        else:
                            # fall back to a minimal spec using the value
                            specs[pname] = {"value": getattr(pobj, "value", None)}
                    except Exception:
                        specs[pname] = {"value": getattr(pobj, "value", None)}
            except Exception as e:
                _log(f"get_parameters: failed to build specs from model_spec.params: {e}")
                specs = {}

            # If a concrete state.model exists, override spec values with current model attrs
            mdl = getattr(self.state, "model", None)
            if mdl is not None:
                for name in list(specs.keys()):
                    try:
                        if hasattr(mdl, name):
                            if isinstance(specs[name], dict):
                                specs[name]["value"] = getattr(mdl, name)
                            else:
                                specs[name] = getattr(mdl, name)
                    except Exception:
                        pass

            _log(f"get_parameters: returning {len(specs)} parameters: {list(specs.keys())}")
            # ensure we store the canonical name/spec for future calls
            setattr(self.state, "model_name", canonical)
            setattr(self.state, "model_spec", model_spec)

            # Build input-action map for interactive controls
            try:
                self._input_map = self._build_input_map(specs)
            except Exception:
                self._input_map = {}

            return specs

        except Exception as e:
            try:
                self.log_message.emit(f"Failed to build parameter specs: {e}")
            except Exception:
                print(f"Failed to build parameter specs: {e}")
            return {}

    def get_addable_model_names(self) -> list:
        """Return a list of model display names that are addable (exclude constructed models).

        This is a convenience wrapper for the view when presenting a list of elements
        that can be added to composites.
        """
        try:
            from models import get_available_model_names
            return get_available_model_names(addable_only=True)
        except Exception:
            return []

    def _build_input_map(self, specs: dict) -> dict:
        """
        Build a mapping of input events -> handlers from parameter specs.
        Each handler is a dict: { 'param': name, 'action': {...} }
        Recognized event keys: 'click', 'drag', 'wheel', 'key'
        The 'input' entry in a spec may be a str, list or a dict describing those keys.
        """
        from collections import defaultdict
        mapping = defaultdict(list)
        for pname, pspec in (specs or {}).items():
            try:
                inp = None
                if isinstance(pspec, dict):
                    # Check both 'input' (from Parameter.to_spec()) and 'input_hint' (fallback)
                    inp = pspec.get("input") or pspec.get("input_hint")
                # allow older string hints as label-only (skip)
                if not inp:
                    continue
                # if it's a string, keep as hint-only and skip actionable mapping
                if isinstance(inp, str):
                    continue
                # assume dict-like structure mapping event -> action-spec
                if isinstance(inp, dict):
                    for ev, act in inp.items():
                        mapping[ev].append({"param": pname, "action": act})
            except Exception:
                continue
        return dict(mapping)

    # Selection lifecycle API
    def begin_selection(self, pname: str, x: float, y: float) -> bool:
        """Begin a selection/interactive session for parameter `pname`.
        Returns True if a session was started."""
        try:
            # find drag handlers for this parameter
            drag_handlers = [h for h in (self._input_map.get("drag", []) or []) if h.get("param") == pname]
            if not drag_handlers:
                return False
            # store selected param and an interactive drag session with filtered handlers
            setattr(self.state, "_selected_param", pname)
            setattr(self.state, "_interactive_drag_info", {"handlers": drag_handlers, "last_x": float(x), "last_y": float(y)})
            self.log_message.emit(f"Selection started for parameter: {pname}")
            return True
        except Exception as e:
            try:
                self.log_message.emit(f"Failed to begin selection for {pname}: {e}")
            except Exception:
                pass
            return False

    def end_selection(self):
        """End any current selection / interactive session."""
        try:
            if hasattr(self.state, "_interactive_drag_info"):
                try:
                    del self.state._interactive_drag_info
                except Exception:
                    try:
                        setattr(self.state, "_interactive_drag_info", None)
                    except Exception:
                        pass
            if hasattr(self.state, "_selected_param"):
                try:
                    del self.state._selected_param
                except Exception:
                    try:
                        setattr(self.state, "_selected_param", None)
                    except Exception:
                        pass
            self.log_message.emit("Selection ended.")
        except Exception:
            pass

    # --- helpers for interactive drag updates ---
    def _get_param_value(self, pname):
        """Return the current numeric value for a parameter from state.model or model_spec, or None."""
        try:
            mdl = getattr(self.state, "model", None)
            if mdl is not None and hasattr(mdl, pname):
                return getattr(mdl, pname)
            ms = getattr(self.state, "model_spec", None)
            if ms is not None and pname in ms.params:
                return ms.params[pname].value
        except Exception:
            pass
        return None

    def _set_param_value(self, pname, value):
        """Set parameter value via apply_parameters (keeps single codepath/update)."""
        try:
            # coerce numeric types where sensible
            if isinstance(value, (np.floating, np.integer)):
                value = float(value)
            self.apply_parameters({pname: value})
        except Exception as e:
            try:
                self.log_message.emit(f"Failed to set {pname}: {e}")
            except Exception:
                pass

    def _modifiers_match(self, modifiers, required_mods):
        """Return True if all required_mods (list of names) are present in modifiers."""
        try:
            from PySide6.QtCore import Qt
            if not required_mods:
                return True
            for r in required_mods:
                rn = str(r).lower()
                if rn == "ctrl" and not bool(modifiers & Qt.ControlModifier):
                    return False
                if rn == "shift" and not bool(modifiers & Qt.ShiftModifier):
                    return False
                if rn == "alt" and not bool(modifiers & Qt.AltModifier):
                    return False
            return True
        except Exception:
            return False

    def set_model(self, model_name: str):
        """Switch the active model specification to `model_name`."""
        try:
            model_name = (model_name or "").strip()
            if not model_name:
                return
            # Create and attach a model_spec for the requested model (local import)
            from models import get_model_spec
            model_spec = get_model_spec(model_name)
            # Allow the spec to initialize from current data
            try:
                model_spec.initialize(getattr(self.state, "x_data", None), getattr(self.state, "y_data", None))
            except Exception:
                pass
            # store name and spec on state
            setattr(self.state, "model_name", model_name)
            setattr(self.state, "model_spec", model_spec)
            # clear any concrete state.model so get_parameters will reflect spec values
            if hasattr(self.state, "model"):
                try:
                    delattr = lambda obj, name: obj.__delattr__(name)
                    # try deleting attribute if possible (SimpleNamespace or similar)
                    try:
                        del self.state.model
                    except Exception:
                        # fallback: replace with a fresh simple namespace
                        from types import SimpleNamespace
                        self.state.model = SimpleNamespace()
                except Exception:
                    pass
            self.log_message.emit(f"Model switched to: {model_name}")
            # Notify View to refresh parameters/plot
            try:
                self.update_plot()
            except Exception:
                pass
        except Exception as e:
            self.log_message.emit(f"Failed to set model '{model_name}': {e}")

    def apply_parameters(self, params: dict):
        """Apply parameters from the UI.

        - params: dict mapping parameter-name -> value.
        The function will set attributes on state.model if present, otherwise update
        the attached state.model_spec parameter values. Finally triggers update_plot().
        """
        if not isinstance(params, dict):
            # try to coerce mapping-like objects to dict
            try:
                params = dict(params)
            except Exception:
                # nothing sensible to do
                self.log_message.emit("apply_parameters: expected a dict")
                return

        # Apply into state.model where possible, otherwise into state.model_spec
        mdl = getattr(self.state, "model", None)
        model_spec = getattr(self.state, "model_spec", None)

        # Ensure we have a model_spec to persist defaults if needed
        if model_spec is None:
            # try to create one from state.model_name or fallback
            model_name = getattr(self.state, "model_name", None)
            if model_name is None and mdl is not None:
                model_name = getattr(mdl, "name", None) or mdl.__class__.__name__
            if model_name is None:
                model_name = "voigt"
            # local import for get_model_spec
            try:
                from models import get_model_spec
                model_spec = get_model_spec(model_name)
                setattr(self.state, "model_spec", model_spec)
            except Exception:
                model_spec = None

        # If no concrete model object exists, create a thin namespace to hold attributes
        if mdl is None:
            mdl = SimpleNamespace()
            setattr(self.state, "model", mdl)

        applied = []
        for k, v in params.items():
            try:
                # prefer to set attribute on the actual model object
                try:
                    setattr(mdl, k, v)
                    applied.append(k)
                except Exception:
                    # if model doesn't accept attribute, fallback to model_spec if available
                    if model_spec is not None and k in model_spec.params:
                        model_spec.params[k].value = v
                        applied.append(k)
                    else:
                        # create attribute on model as last resort
                        setattr(mdl, k, v)
                        applied.append(k)
            except Exception:
                # ignore per-parameter failures
                pass

        # Also update any model_spec values for consistency if both exist
        if model_spec is not None:
            for name in applied:
                if name in model_spec.params:
                    try:
                        model_spec.params[name].value = getattr(mdl, name)
                    except Exception:
                        pass

        # Notify and refresh plot
        self.log_message.emit(f"Applied parameters: {', '.join(applied) if applied else 'none'}")
        try:
            self.update_plot()
        except Exception:
            pass
        # Notify any views that parameter values have changed so they can refresh their widgets.
        try:
            self.parameters_updated.emit()
        except Exception:
            # avoid raising if no listeners or signal fails
            pass
    
    # --------------------------
    # Input event handlers (integrated from PySide_Fitter_PyQtGraph.py patterns)
    # --------------------------
    def handle_plot_click(self, x, y, button):
        """
        Handle plot click events from the view.

        Args:
            x: X coordinate in data space
            y: Y coordinate in data space
            button: Mouse button used
        """
        try:
            self.log_message.emit(f"Plot clicked at data coords: ({x:.3f}, {y:.3f})")
            # Only use click here to start a drag session if a "drag" handler exists.
            try:
                from PySide6.QtCore import Qt
                if button is not None and int(button) == int(Qt.LeftButton):
                    drag_handlers = (self._input_map or {}).get("drag", [])
                    if drag_handlers:
                        # Start an interactive drag session storing handlers and initial coords.
                        self.state._interactive_drag_info = {
                            "handlers": drag_handlers,
                            "last_x": float(x),
                            "last_y": float(y)
                        }
                        self.log_message.emit(f"Interactive: started drag session for {[h.get('param') for h in drag_handlers]}")
                # Do not perform immediate 'set' on plain click — clicks reserved for other UI parts.
            except Exception:
                pass

            # Future: Add logic for selecting/manipulating peaks or markers
            # For now, just log the event
        except Exception as e:
            self.log_message.emit(f"Error in handle_plot_click: {e}")

    def handle_plot_mouse_move(self, x, y, buttons=None):
        """
        Handle plot mouse move events from the view.
        If a button is pressed and the model declared a 'drag' action for a parameter,
        update that parameter continuously. When no buttons are pressed, stop drag.
        """
        try:
            from PySide6.QtCore import Qt
            # Determine if left-button (or any) is pressed
            left_pressed = False
            try:
                if buttons is not None:
                    left_pressed = bool(int(buttons) & int(Qt.LeftButton))
            except Exception:
                left_pressed = False

            # If a drag session was started, honor it
            try:
                drag_info = getattr(self.state, "_interactive_drag_info", None)
                # If left button released, stop drag session (but keep selection until explicit end_selection)
                if not left_pressed:
                    if drag_info is not None:
                        try:
                            del self.state._interactive_drag_info
                        except Exception:
                            try:
                                setattr(self.state, "_interactive_drag_info", None)
                            except Exception:
                                pass
                    return

                if drag_info is None:
                    # No explicit session, but still allow declared handlers (fallback)
                    handlers = (self._input_map or {}).get("drag", [])
                    # If there is an explicit selected param, filter to it
                    selected = getattr(self.state, "_selected_param", None)
                    if selected:
                        handlers = [h for h in handlers if h.get("param") == selected]
                    if not handlers:
                        return
                    # Build a temporary session using handlers with last positions = current
                    drag_info = {"handlers": handlers, "last_x": float(x), "last_y": float(y)}

                # Compute deltas
                last_x = float(drag_info.get("last_x", x))
                last_y = float(drag_info.get("last_y", y))
                dx = float(x) - last_x
                dy = float(y) - last_y
                # update last positions
                drag_info["last_x"] = float(x)
                drag_info["last_y"] = float(y)
                # persist back if real session
                if getattr(self.state, "_interactive_drag_info", None) is not None:
                    self.state._interactive_drag_info = drag_info

                # Apply handlers: allow separate horizontal ('h') and vertical ('v') actions
                for h in drag_info.get("handlers", []):
                    pname = h.get("param")
                    act = h.get("action", {}) or {}
                    # If action specifies separate horizontal/vertical sub-actions
                    if isinstance(act, dict) and ("h" in act or "v" in act):
                        # Horizontal part
                        if "h" in act and abs(dx) > 0:
                            ah = act["h"]
                            # support 'set' (absolute), 'increment' (by step * sign), 'scale' (multiply by factor^sign)
                            try:
                                cur = self._get_param_value(pname)
                                if ah.get("action") == "set":
                                    val = x if ah.get("value_from", "x") == "x" else y
                                    self._set_param_value(pname, float(val))
                                elif ah.get("action") == "increment" and cur is not None:
                                    step = float(ah.get("step", 0.01))
                                    new = float(cur) + step * np.sign(dx)
                                    self._set_param_value(pname, new)
                                elif ah.get("action") == "scale" and cur is not None:
                                    factor = float(ah.get("factor", 1.02))
                                    mult = factor if dx > 0 else (1.0 / factor)
                                    self._set_param_value(pname, float(cur) * mult)
                            except Exception:
                                pass
                        # Vertical part — may target same param or different field via 'v.param'
                        if "v" in act and abs(dy) > 0:
                            av = act["v"]
                            try:
                                # by default vertical modifies the same parameter; handler may specify 'param_v'
                                target = av.get("param", pname)
                                cur = self._get_param_value(target)
                                if av.get("action") == "set":
                                    val = y if av.get("value_from", "y") == "y" else x
                                    self._set_param_value(target, float(val))
                                elif av.get("action") == "increment" and cur is not None:
                                    step = float(av.get("step", 0.01))
                                    new = float(cur) + step * np.sign(dy)
                                    self._set_param_value(target, new)
                                elif av.get("action") == "scale" and cur is not None:
                                    factor = float(av.get("factor", 1.02))
                                    mult = factor if dy > 0 else (1.0 / factor)
                                    self._set_param_value(target, float(cur) * mult)
                            except Exception:
                                pass
                    else:
                        # Simple legacy behavior: treat action as a set-from-x on horizontal movement
                        try:
                            if abs(dx) > 0:
                                if act.get("action") == "set":
                                    vfrom = act.get("value_from", "x")
                                    val = x if vfrom == "x" else y
                                    self._set_param_value(pname, float(val))
                                elif act.get("action") == "increment":
                                    step = float(act.get("step", 0.01))
                                    cur = self._get_param_value(pname)
                                    if cur is not None:
                                        new = float(cur) + step * np.sign(dx)
                                        self._set_param_value(pname, new)
                                elif act.get("action") == "scale":
                                    factor = float(act.get("factor", 1.02))
                                    cur = self._get_param_value(pname)
                                    if cur is not None:
                                        mult = factor if dx > 0 else (1.0 / factor)
                                        self._set_param_value(pname, float(cur) * mult)
                                else:
                                    # fallback: absolute set to x
                                    self._set_param_value(pname, float(x))
                        except Exception:
                            pass
            except Exception:
                pass

        except Exception:
            # be quiet for mouse-move errors to avoid spam
            return

    def handle_key_press(self, key, modifiers):
        """
        Handle keyboard events from the plot.
        
        Args:
            key: Qt key code
            modifiers: Qt keyboard modifiers
        """
        try:
            from PySide6.QtCore import Qt
            
            # Handle specific keys
            if key == Qt.Key_Space:
                self.log_message.emit("Space key: Clear selection")
                # Future: Clear any active selections
            elif key == Qt.Key_F:
                # Example: Trigger fit
                self.log_message.emit("F key: Run fit")
                try:
                    self.run_fit()
                except Exception as e:
                    self.log_message.emit(f"Fit failed: {e}")
            elif key == Qt.Key_U:
                # Example: Update plot
                self.log_message.emit("U key: Update plot")
                try:
                    self.update_plot()
                except Exception:
                    pass
            # Add more key handlers as needed
                
        except Exception as e:
            self.log_message.emit(f"Error in handle_key_press: {e}")
    
    def handle_wheel_scroll(self, delta, modifiers):
        """
        Handle mouse wheel scroll events with keyboard modifiers.

        Can be used to adjust parameters interactively.
        """
        try:
            from PySide6.QtCore import Qt

            is_ctrl = bool(modifiers & Qt.ControlModifier)
            is_shift = bool(modifiers & Qt.ShiftModifier)
            is_alt = bool(modifiers & Qt.AltModifier)

            # Determine scroll direction
            step_sign = 1 if delta > 0 else -1

            # First consult input_map 'wheel' handlers
            try:
                handlers = (self._input_map or {}).get("wheel", [])
                # if a parameter is selected, limit handlers to that parameter
                selected = getattr(self.state, "_selected_param", None)
                if selected:
                    handlers = [h for h in handlers if h.get("param") == selected]
                for h in handlers:
                    pname = h.get("param")
                    act = h.get("action", {}) or {}
                    req_mods = [m for m in (act.get("modifiers") or [])]
                    if not self._modifiers_match(modifiers, req_mods):
                        continue
                    # action types: scale (multiply by factor^sign), increment (add step * sign)
                    if act.get("action") == "scale":
                        factor = float(act.get("factor", 1.05))
                        try:
                            cur = self._get_param_value(pname)
                            if isinstance(cur, (int, float)):
                                mult = factor if step_sign > 0 else (1.0 / factor)
                                new_val = float(cur) * mult
                                self.apply_parameters({pname: new_val})
                                self.log_message.emit(f"Interactive: wheel scaled {pname} -> {new_val:.4g}")
                        except Exception as e:
                            self.log_message.emit(f"Wheel scale failed for {pname}: {e}")
                    elif act.get("action") == "increment":
                        step = float(act.get("step", 0.1))
                        try:
                            cur = self._get_param_value(pname)
                            if isinstance(cur, (int, float)):
                                new_val = float(cur) + step * step_sign
                                self.apply_parameters({pname: new_val})
                                self.log_message.emit(f"Interactive: wheel increment {pname} -> {new_val:.4g}")
                        except Exception as e:
                            self.log_message.emit(f"Wheel increment failed for {pname}: {e}")
                # if any handler consumed, return
            except Exception:
                pass

            # Fallback: previous behavior - adjust first numeric parameter with Ctrl only (but if selected, try selected param)
            selected = getattr(self.state, "_selected_param", None)
            if selected:
                try:
                    cur = self._get_param_value(selected)
                    if isinstance(cur, (int, float)):
                        factor = 1.1 if step_sign > 0 else 0.9
                        new_val = float(cur) * factor
                        self.apply_parameters({selected: new_val})
                        self.log_message.emit(f"Wheel on selected: Adjusted {selected} to {new_val:.3f}")
                        return
                except Exception:
                    pass

            if is_ctrl and not is_shift and not is_alt:
                try:
                    specs = self.get_parameters()
                    if specs:
                        param_name = next(iter(specs.keys()))
                        spec = specs[param_name]
                        if isinstance(spec, dict):
                            current_val = spec.get('value', 0)
                        else:
                            current_val = spec
                        if isinstance(current_val, (int, float)):
                            factor = 1.1 if step_sign > 0 else 0.9
                            new_val = current_val * factor
                            self.apply_parameters({param_name: new_val})
                            self.log_message.emit(f"Ctrl+Wheel: Adjusted {param_name} to {new_val:.3f}")
                except Exception as e:
                    self.log_message.emit(f"Could not adjust parameter (fallback): {e}")

        except Exception as e:
            self.log_message.emit(f"Error in handle_wheel_scroll: {e}")

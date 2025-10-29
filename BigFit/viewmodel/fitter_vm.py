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

    def __init__(self, model_state=None):
        super().__init__()
        self.state = model_state or ModelState()

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
        """Placeholder for fitting routine."""
        y_fit = self.state.evaluate()
        errs = getattr(self.state, "errors", None)
        errs = None if errs is None else np.asarray(errs, dtype=float)
        self.plot_updated.emit(self.state.x_data, self.state.y_data, y_fit, errs)
        self.log_message.emit("Fit completed (mock).")

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
            from models import get_model_spec
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

            # Normalize common UI labels to canonical keys expected by get_model_spec
            _map = {
                "voigt": "voigt",
                "voigtmodel": "voigt",
                "gaussian": "gaussian",
                "gauss": "gaussian",
                "dho": "dho",
                "dho+voigt": "dho+voigt",
                "dho_voigt": "dho+voigt",
            }
            key = str(model_name).strip()
            lower = key.lower()
            # try direct map, then fall back to cleaned lowercase
            if lower in _map:
                canonical = _map[lower]
            else:
                # clean up common variants (remove spaces/underscores)
                clean = lower.replace(" ", "").replace("_", "")
                canonical = _map.get(clean, lower)

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
            return specs

        except Exception as e:
            try:
                self.log_message.emit(f"Failed to build parameter specs: {e}")
            except Exception:
                print(f"Failed to build parameter specs: {e}")
            return {}

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

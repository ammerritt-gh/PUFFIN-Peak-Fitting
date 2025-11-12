# viewmodel/fitter_vm.py
from PySide6.QtCore import QObject, Signal
import numpy as np
import os
from types import SimpleNamespace

from models import ModelState, CompositeModelSpec, get_model_spec, get_atomic_component_names
from dataio.data_loader import select_and_load_files
from dataio.data_saver import save_dataset
import typing as _typing


class FitterViewModel(QObject):
    """
    Central logic layer: handles loading/saving, fitting, and updates to the plot.
    """

    plot_updated = Signal(object, object, object, object)   # x, y_data, y_fit, y_err
    curve_selection_changed = Signal(object)                # emits curve_id or None
    log_message = Signal(str)
    parameters_updated = Signal()
    files_updated = Signal(object)                          # list of queued file metadata


    def __init__(self, model_state=None):
        super().__init__()
        self.state = model_state or ModelState()
        self._fit_worker = None
        self._selected_curve_id = None  # currently selected curve ID (str or None)
        self._datasets = []  # queued datasets [(dict)]
        self._active_dataset_index = None
        self.curves: dict = {}
        # Attempt to restore previously queued files from configuration
        try:
            self._load_queue_from_config()
        except Exception as exc:
            # Failed to restore previous file queue; continue with empty queue.
            self.log_message.emit(f"Could not restore previous file queue: {exc}")

    # --------------------------
    # Data I/O
    # --------------------------
    def load_data(self):
        """Open file dialog and append one or more datasets to the queue."""
        loaded = select_and_load_files(None)
        if not loaded:
            return

        added = 0
        last_info = None
        for x, y, err, info in loaded:
            try:
                x_arr, y_arr, err_arr = self._prepare_dataset(x, y, err)
            except Exception as exc:
                name = (info or {}).get("name") if isinstance(info, dict) else None
                label = name or "dataset"
                self.log_message.emit(f"Skipped {label}: {exc}")
                continue

            dataset = {
                "x": x_arr,
                "y": y_arr,
                "err": err_arr,
                "info": info or {},
            }
            self._datasets.append(dataset)
            added += 1
            last_info = info or last_info

        if added == 0:
            return

        if isinstance(last_info, dict):
            self._persist_last_loaded(last_info)

        plural = "file" if added == 1 else "files"
        self.log_message.emit(f"Queued {added} {plural} for viewing.")
        self._emit_file_queue()

        # If nothing is currently active, auto-activate the first queued dataset
        try:
            if self._active_dataset_index is None and len(self._datasets) > 0:
                # activate_file will apply to state, emit updated queue, and update plot
                try:
                    self.activate_file(0)
                except Exception:
                    # fallback: set active index and emit queue
                    self._active_dataset_index = 0
                    try:
                        self._emit_file_queue()
                    except Exception:
                        pass
        except Exception:
            pass

    def _prepare_dataset(self, x, y, err):
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if err is None:
            err_arr = np.sqrt(np.clip(np.abs(y_arr), 1e-12, np.inf))
        else:
            err_arr = np.asarray(err, dtype=float)

        if len(x_arr) == 0 or len(y_arr) == 0 or len(err_arr) == 0:
            raise ValueError("Dataset is empty.")

        common_len = min(len(x_arr), len(y_arr), len(err_arr))
        if common_len == 0:
            raise ValueError("Dataset has no overlapping values.")

        return (
            np.array(x_arr[:common_len], dtype=float, copy=True),
            np.array(y_arr[:common_len], dtype=float, copy=True),
            np.array(err_arr[:common_len], dtype=float, copy=True),
        )

    def _persist_last_loaded(self, info: dict):
        try:
            path = info.get("path")
            if not path:
                return
            from dataio import get_config

            cfg = get_config()
            cfg.last_loaded_file = path
            folder = os.path.dirname(path)
            if folder:
                cfg.default_load_folder = folder
            try:
                cfg.save()
            except Exception:
                pass
        except Exception:
            pass

    def _emit_file_queue(self):
        files = []
        for idx, dataset in enumerate(self._datasets):
            info = dataset.get("info") if isinstance(dataset, dict) else {}
            info = info if isinstance(info, dict) else {}
            entry = {
                "index": idx,
                "name": info.get("name") or f"Dataset {idx + 1}",
                "path": info.get("path"),
                "size": info.get("size"),
                "active": idx == self._active_dataset_index,
                "info": info,
            }
            files.append(entry)
        try:
            self.files_updated.emit(files)
        except Exception:
            pass
        # persist queue to config so it survives restarts
        try:
            self._save_queue_to_config()
        except Exception:
            pass

    def notify_file_queue(self):
        self._emit_file_queue()

    def _save_queue_to_config(self):
        try:
            from dataio import get_config
            cfg = get_config()
            # Save as simple list of file paths and optional metadata
            queued = []
            for ds in self._datasets:
                info = ds.get("info") if isinstance(ds, dict) else {}
                path = None
                if isinstance(info, dict):
                    path = info.get("path")
                if not path:
                    path = ds.get("info", {}).get("path") if isinstance(ds.get("info"), dict) else None
                queued.append({"path": path, "name": (info.get("name") if isinstance(info, dict) else None)})
            cfg.queued_files = queued
            # store active index if present
            cfg.queued_active = self._active_dataset_index
            try:
                cfg.save()
            except Exception:
                pass
        except Exception:
            pass

    def _load_queue_from_config(self):
        try:
            from dataio import get_config
            cfg = get_config()
            q = getattr(cfg, "queued_files", None)
            active = getattr(cfg, "queued_active", None)
            if not q:
                return
            # Attempt to load each path into the queue
            from dataio.data_loader import load_data_from_file
            added = 0
            for entry in q:
                try:
                    path = entry.get("path") if isinstance(entry, dict) else None
                    if not path or not os.path.isfile(path):
                        continue
                    x, y, err, info = load_data_from_file(path)
                    x_arr, y_arr, err_arr = self._prepare_dataset(x, y, err)
                    dataset = {"x": x_arr, "y": y_arr, "err": err_arr, "info": info}
                    self._datasets.append(dataset)
                    added += 1
                except Exception:
                    continue
            # restore active index if valid
            if isinstance(active, int) and 0 <= active < len(self._datasets):
                self._active_dataset_index = int(active)
            else:
                self._active_dataset_index = None
            if added:
                # notify UI
                try:
                    self._emit_file_queue()
                except Exception:
                    pass
        except Exception:
            pass

    def _apply_dataset_to_state(self, dataset: dict):
        x = dataset.get("x")
        y = dataset.get("y")
        err = dataset.get("err")
        info = dataset.get("info") if isinstance(dataset.get("info"), dict) else {}

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        try:
            self.state.set_data(np.array(x_arr, copy=True), np.array(y_arr, copy=True))
        except Exception:
            self.state.x_data = np.array(x_arr, copy=True)
            self.state.y_data = np.array(y_arr, copy=True)
            try:
                self.state.model_spec.initialize(self.state.x_data, self.state.y_data)
            except Exception:
                pass
            try:
                self.state.excluded = np.zeros_like(self.state.x_data, dtype=bool)
            except Exception:
                self.state.excluded = np.array([], dtype=bool)

        try:
            self.state.errors = np.array(err, dtype=float, copy=True)
        except Exception:
            self.state.errors = np.sqrt(np.clip(np.abs(self.state.y_data), 1e-12, np.inf))

        if info:
            try:
                try:
                    setattr(self.state, "file_info", info)
                except Exception:
                    # fallback: ignore if state doesn't accept dynamic attributes
                    pass
            except Exception:
                pass

    def activate_file(self, index: int):
        try:
            idx = int(index)
        except Exception:
            return

        if idx < 0 or idx >= len(self._datasets):
            return

        dataset = self._datasets[idx]
        self._apply_dataset_to_state(dataset)
        self._active_dataset_index = idx

        info = dataset.get("info") if isinstance(dataset, dict) else {}
        if not isinstance(info, dict):
            info = {}
        name = info.get("name")
        label = name or f"Dataset {idx + 1}"
        self.log_message.emit(f"Loaded dataset: {label}")

        try:
            self.parameters_updated.emit()
        except Exception:
            pass

        self._emit_file_queue()
        try:
            self.update_plot()
        except Exception:
            pass

    def remove_file_at(self, index: int):
        try:
            idx = int(index)
        except Exception:
            return

        if idx < 0 or idx >= len(self._datasets):
            return

        dataset = self._datasets.pop(idx)
        info_obj = dataset.get("info") if isinstance(dataset, dict) else {}
        info = info_obj if isinstance(info_obj, dict) else {}
        name = info.get("name") or f"Dataset {idx + 1}"
        self.log_message.emit(f"Removed dataset: {name}")

        if self._active_dataset_index == idx:
            self._active_dataset_index = None
            if self._datasets:
                next_idx = idx if idx < len(self._datasets) else len(self._datasets) - 1
                self.activate_file(next_idx)
            else:
                self.clear_loaded_files(emit_log=False)
                self.log_message.emit("Dataset queue is now empty; reverted to default data.")
        else:
            if self._active_dataset_index is not None and idx < self._active_dataset_index:
                self._active_dataset_index -= 1
            self._emit_file_queue()

    def clear_loaded_files(self, emit_log=True):
        self._datasets.clear()
        self._active_dataset_index = None
        model_name = getattr(self.state, "model_name", "Voigt") if hasattr(self, "state") else "Voigt"
        try:
            self.state = ModelState(model_name=model_name)
        except Exception:
            self.state = ModelState()
        self._fit_worker = None
        self._emit_file_queue()
        try:
            self.parameters_updated.emit()
        except Exception:
            pass
        try:
            self.update_plot()
        except Exception:
            pass
        if emit_log:
            self.log_message.emit("Cleared queued datasets and restored initial synthetic data.")

    def save_data(self):
        """Save current data and fit to file."""
        y_fit = self.state.evaluate() if hasattr(self.state, "evaluate") else None
        save_dataset(self.state.x_data, self.state.y_data, y_fit=y_fit)
        self.log_message.emit("Data saved successfully.")

    def clear_plot(self):
        """Reset to initial synthetic dataset and clear stored last-loaded file in config."""
        try:
            self.clear_loaded_files(emit_log=False)

            # clear saved last-loaded file in config if available
            try:
                from dataio import get_config
                cfg = get_config()
                cfg.last_loaded_file = None
                try:
                    cfg.save()
                except Exception:
                    pass
            except Exception:
                pass

            self.log_message.emit("Cleared queued datasets and restored initial synthetic data.")
        except Exception as e:
            try:
                self.log_message.emit(f"Failed to clear plot: {e}")
            except Exception:
                pass

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

    def handle_action(self, action: str, **kwargs):
        """
        Central dispatcher for view-driven actions.

        Examples:
            handle_action('run_fit')
            handle_action('toggle_point_exclusion', x=1.2, y=3.4)
            handle_action('apply_parameters', params={...})

        Returns:
            The result of the called action function, or None if the action is not recognized or an error occurs.
        """
        if not action:
            try:
                self.log_message.emit("handle_action: no action provided")
            except Exception:
                pass
            return None

        a = str(action).strip()
        # helper to safely attempt calling functions that expect an integer index
        def _call_with_index(keys, func):
            for k in keys:
                if k in kwargs and kwargs.get(k) is not None:
                    v = kwargs.get(k)
                    # prefer to pass the raw value; the target methods already
                    # perform their own int(...) conversions and validations.
                    try:
                        return func(v)
                    except Exception:
                        return None
            return None

        # mapping of common action names to callables
        mapping = {
            "run_fit": self.run_fit,
            "update_plot": self.update_plot,
            "load_data": self.load_data,
            "save_data": self.save_data,
            "clear_plot": self.clear_plot,
            "toggle_point_exclusion": lambda: self.toggle_point_exclusion(kwargs.get("x"), kwargs.get("y"), tol=kwargs.get("tol", 0.05)),
            "toggle_point_exclusion_by_index": lambda: _call_with_index(("idx", "index"), self.toggle_point_exclusion_by_index),
            "toggle_box_exclusion": lambda: self.toggle_box_exclusion(kwargs.get("x0"), kwargs.get("y0"), kwargs.get("x1"), kwargs.get("y1")),
            "clear_exclusions": self.clear_exclusions,
            "apply_parameters": lambda: self.apply_parameters(kwargs.get("params") or kwargs.get("updates") or {}),
            "set_selected_curve": lambda: self.set_selected_curve(kwargs.get("curve_id") or kwargs.get("curve")),
            "clear_selected_curve": self.clear_selected_curve,
            "on_peak_moved": lambda: self.on_peak_moved(kwargs.get("info") or kwargs.get("peak_info") or {}),
            "on_peak_selected": lambda: self.on_peak_selected(kwargs.get("x"), kwargs.get("y")),
            "activate_file": lambda: _call_with_index(("index", "idx"), self.activate_file),
            "remove_file_at": lambda: _call_with_index(("index", "idx"), self.remove_file_at),
        }

        try:
            if a in mapping:
                func = mapping[a]
                try:
                    return func()
                except TypeError:
                    # fallback: try passing kwargs directly
                    try:
                        return func(**kwargs)
                    except Exception:
                        return None
            # dynamic dispatch: try to call a same-named method on self
            if hasattr(self, a) and callable(getattr(self, a)):
                try:
                    return getattr(self, a)(**kwargs)
                except TypeError:
                    return getattr(self, a)()
        except Exception as e:
            try:
                self.log_message.emit(f"handle_action('{action}') failed: {e}")
            except Exception:
                pass
        try:
            self.log_message.emit(f"Unknown action requested: '{action}'")
        except Exception:
            pass
        return None

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
        # Use masked (included) data for fitting so excluded points are ignored
        try:
            x_incl, y_incl, err_incl = self.state.get_masked_data()
        except Exception:
            x_incl, y_incl, err_incl = None, None, None
        x = x_incl if x_incl is not None else getattr(self.state, "x_data", None)
        y = y_incl if y_incl is not None else getattr(self.state, "y_data", None)
        if x is None or y is None:
            self.log_message.emit("No data available to fit.")
            return

        err = err_incl if err_incl is not None else getattr(self.state, "errors", None)

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

        # build initial parameter value map and separate free (optimised) parameters
        try:
            full_values = {k: getattr(v, "value", None) for k, v in getattr(model_spec, "params", {}).items()}
            # Determine which parameters are free (not fixed)
            free_keys = [k for k, v in getattr(model_spec, "params", {}).items() if not bool(getattr(v, "fixed", False))]
            if not free_keys:
                self.log_message.emit("No free parameters to fit (all parameters are fixed).")
                return

            # initial values for free params
            params = {k: full_values.get(k) for k in free_keys}
            # bounds for free params (ordered to match free_keys)
            lower = [getattr(model_spec.params[k], "min", None) if getattr(model_spec.params[k], "min", None) is not None else -np.inf for k in free_keys]
            upper = [getattr(model_spec.params[k], "max", None) if getattr(model_spec.params[k], "max", None) is not None else np.inf for k in free_keys]
            bounds = (lower, upper)
        except Exception as e:
            self.log_message.emit(f"Failed to build parameter/bounds list: {e}")
            return

        # wrap evaluate(x, **params) into curve_fit-style f(x, *args)
        param_keys = list(params.keys())

        def wrapped_func(xx, *args):
            # start from the full map (including fixed values) and overwrite free keys
            p = dict(full_values)
            try:
                for k, val in zip(param_keys, args):
                    p[k] = val
            except Exception:
                pass
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
                        if spec is not None:
                            for k, v in result.items():
                                try:
                                    if isinstance(spec, CompositeModelSpec) and hasattr(spec, "set_param_value"):
                                        spec.set_param_value(k, v)
                                    elif hasattr(spec, "params") and k in spec.params:
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

                    plot_refreshed = False
                    try:
                        self.update_plot()
                        plot_refreshed = True
                    except Exception:
                        plot_refreshed = False

                    if not plot_refreshed:
                        # Fallback: emit direct plot update if update_plot failed
                        full_y_fit = None
                        try:
                            # build a full param dict (merge fixed/full values with fitted results)
                            spec = getattr(self.state, "model_spec", None)
                            if spec is not None and hasattr(spec, "get_param_values"):
                                full_params = spec.get_param_values()
                            else:
                                # fall back to merging captured full_values with result
                                full_params = dict(full_values)
                                try:
                                    full_params.update(result or {})
                                except Exception:
                                    pass

                            try:
                                full_y_fit = model_func(getattr(self.state, "x_data", np.array([])), full_params)
                            except TypeError:
                                try:
                                    full_y_fit = model_func(getattr(self.state, "x_data", np.array([])), **full_params)
                                except Exception:
                                    full_y_fit = None
                        except Exception:
                            full_y_fit = None

                        full_errs = getattr(self.state, "errors", None)
                        self.plot_updated.emit(getattr(self.state, "x_data", x), getattr(self.state, "y_data", y), full_y_fit, full_errs)

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
        x = getattr(self.state, "x_data", None)
        y = getattr(self.state, "y_data", None)
        if x is None or y is None:
            return

        model_spec = getattr(self.state, "model_spec", None)
        y_fit_payload = None
        curves_payload = {}

        if isinstance(model_spec, CompositeModelSpec):
            try:
                component_outputs = model_spec.evaluate_components(x)
            except Exception:
                component_outputs = []

            total = np.zeros_like(np.asarray(x, dtype=float), dtype=float)
            components_for_view = []

            for component, values in component_outputs:
                arr = np.asarray(values, dtype=float)
                try:
                    total += arr
                except Exception as e:
                    # Non-fatal: if a component's output cannot be summed, skip it but show in plot.
                    self.log_message.emit(
                        f"Could not add component '{component.prefix}' to total fit: {e}"
                    )
                components_for_view.append(
                    {
                        "prefix": component.prefix,
                        "label": component.label,
                        "color": component.color,
                        "y": arr,
                    }
                )
                curves_payload[f"component:{component.prefix}"] = (
                    np.asarray(x, dtype=float),
                    arr,
                )

            if components_for_view:
                curves_payload["fit"] = (np.asarray(x, dtype=float), total)
                y_fit_payload = {"total": total, "components": components_for_view}
            else:
                curves_payload["fit"] = (np.asarray(x, dtype=float), total)
                y_fit_payload = total
        else:
            y_fit = None
            if hasattr(self.state, "evaluate"):
                try:
                    y_fit = self.state.evaluate()
                except Exception:
                    y_fit = None
            if y_fit is not None:
                arr = np.asarray(y_fit, dtype=float)
                curves_payload["fit"] = (np.asarray(x, dtype=float), arr)
            y_fit_payload = y_fit

        self.curves = curves_payload
        errs = getattr(self.state, "errors", None)
        errs = None if errs is None else np.asarray(errs, dtype=float)
        self.plot_updated.emit(x, y, y_fit_payload, errs)

    def compute_statistics(self, y_fit=None, n_params: int = 0) -> dict:
        """Compute common fit statistics (reduced chi-squared, Cash) for current state.

        Returns a dict with keys: 'reduced_chi2', 'cash', 'reduced_cash'. Values may be None on failure.
        This method performs a lazy import of models.metrics to avoid module-time coupling.
        """
        try:
            # import locally to avoid circular imports at module import time
            from models.metrics import compute_statistics_from_state
            stats = compute_statistics_from_state(self.state, y_fit, n_params)
            return stats
        except Exception as e:
            try:
                self.log_message.emit(f"Failed to compute statistics: {e}")
            except Exception:
                pass
            return {"reduced_chi2": None, "cash": None, "reduced_cash": None}

    # --------------------------
    # Exclusion management (called from InputHandler/View)
    # --------------------------
    def toggle_point_exclusion(self, x, y, tol=0.05):
        """Toggle exclusion for the nearest point to (x,y)."""
        try:
            if hasattr(self.state, "toggle_point_exclusion"):
                idx = self.state.toggle_point_exclusion(x, y, tol=tol)
                self.log_message.emit(f"Toggled exclusion for point index: {idx}")
            else:
                self.log_message.emit("State does not support point exclusion.")
        except Exception as e:
            self.log_message.emit(f"Failed to toggle point exclusion: {e}")
        try:
            self.update_plot()
        except Exception:
            pass

    def toggle_point_exclusion_by_index(self, idx: int):
        """Toggle exclusion state by index (if supported by state). Logs the index and updates plot."""
        try:
            if hasattr(self.state, "excluded"):
                try:
                    if idx is None:
                        self.log_message.emit("No index provided for toggle_point_exclusion_by_index.")
                        return
                    idx = int(idx)
                    # ensure index in range
                    xd = getattr(self.state, "x_data", None)
                    if xd is None:
                        self.log_message.emit("No data present to toggle exclusion.")
                        return
                    if idx < 0 or idx >= len(xd):
                        self.log_message.emit(f"Index {idx} out of range for exclusion toggle.")
                        return
                    self.state.excluded[idx] = not bool(self.state.excluded[idx])
                    self.log_message.emit(f"Toggled exclusion for point index: {idx}")
                except Exception as e:
                    self.log_message.emit(f"Failed to toggle exclusion by index: {e}")
            else:
                self.log_message.emit("State does not support exclusions by index.")
        except Exception as e:
            self.log_message.emit(f"Failed to toggle point exclusion by index: {e}")
        try:
            self.update_plot()
        except Exception:
            pass

    def toggle_box_exclusion(self, x0, y0, x1, y1):
        """Toggle exclusion for points inside the given box."""
        try:
            if hasattr(self.state, "toggle_box_exclusion"):
                inds = self.state.toggle_box_exclusion(x0, y0, x1, y1)
                self.log_message.emit(f"Toggled exclusion for {len(inds)} points.")
            else:
                self.log_message.emit("State does not support box exclusion.")
        except Exception as e:
            self.log_message.emit(f"Failed to toggle box exclusion: {e}")
        try:
            self.update_plot()
        except Exception:
            pass

    def clear_exclusions(self):
        """Include all points (clear any exclusion mask) without touching model parameters."""
        try:
            if hasattr(self.state, "excluded"):
                try:
                    self.state.excluded = np.zeros_like(getattr(self.state, "x_data", np.array([])), dtype=bool)
                except Exception:
                    self.state.excluded = np.array([], dtype=bool)
                self.log_message.emit("Cleared all exclusions (included all points).")
            else:
                self.log_message.emit("No exclusion mask present on state.")
        except Exception as e:
            self.log_message.emit(f"Failed to clear exclusions: {e}")
        try:
            self.update_plot()
        except Exception:
            pass

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
                "custom": "custom model",
                "custommodel": "custom model",
                "custom model": "custom model",
                "composite": "custom model",
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

            # Obtain spec instance: prefer existing state.model_spec when available
            model_spec = getattr(self.state, "model_spec", None)
            if model_spec is None:
                try:
                    model_spec = get_model_spec(canonical)
                    setattr(self.state, "model_spec", model_spec)
                    setattr(self.state, "model_name", model_name)
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
                if hasattr(model_spec, "get_parameters"):
                    specs = dict(model_spec.get_parameters() or {})
                else:
                    params_map = getattr(model_spec, "params", None) or {}
                    for pname, pobj in params_map.items():
                        try:
                            if hasattr(pobj, "to_spec"):
                                specs[pname] = pobj.to_spec()
                            else:
                                specs[pname] = {"value": getattr(pobj, "value", None)}
                        except Exception:
                            specs[pname] = {"value": getattr(pobj, "value", None)}
            except Exception as e:
                _log(f"get_parameters: failed to build specs from model_spec: {e}")
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
            # Create and attach a model_spec for the requested model
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
            # If a concrete model attribute exists on state, try to remove it so
            # get_parameters() will prefer the model_spec values. Use attribute
            # helpers to avoid static type-checker errors about unknown attributes.
            try:
                if hasattr(self.state, "model"):
                    try:
                        delattr(self.state, "model")
                    except Exception:
                        # fallback: replace with a fresh simple namespace
                        try:
                            from types import SimpleNamespace
                            setattr(self.state, "model", SimpleNamespace())
                        except Exception:
                            pass
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

    # --------------------------
    # Composite model helpers
    # --------------------------
    def _require_composite(self, action: str) -> _typing.Optional[CompositeModelSpec]:
        spec = getattr(self.state, "model_spec", None)
        if not isinstance(spec, CompositeModelSpec):
            self.log_message.emit(f"{action} requires the Custom model to be active.")
            return None
        return spec

    def get_available_component_names(self) -> _typing.List[str]:
        return get_atomic_component_names()

    def get_component_descriptors(self) -> _typing.List[dict]:
        spec = getattr(self.state, "model_spec", None)
        if not isinstance(spec, CompositeModelSpec):
            return []
        descriptors = []
        for idx, component in enumerate(spec.list_components()):
            descriptors.append(
                {
                    "index": idx,
                    "prefix": component.prefix,
                    "label": component.label,
                    "color": component.color,
                    "spec_name": component.spec.__class__.__name__,
                }
            )
        return descriptors

    def add_component_to_model(self, component_name: str, initial_params: _typing.Optional[dict] = None) -> bool:
        spec = self._require_composite("Adding elements")
        if spec is None:
            return False
        try:
            component = spec.add_component(
                component_name,
                initial_params or {},
                data_x=getattr(self.state, "x_data", None),
                data_y=getattr(self.state, "y_data", None),
            )
        except Exception as exc:
            self.log_message.emit(f"Failed to add component '{component_name}': {exc}")
            return False
        self.log_message.emit(f"Added component {component.label} ({component.spec.__class__.__name__})")
        try:
            self.parameters_updated.emit()
        except Exception:
            pass
        try:
            self.update_plot()
        except Exception:
            pass
        return True

    def remove_component_at(self, index: int) -> bool:
        spec = self._require_composite("Removing elements")
        if spec is None:
            return False
        try:
            removed = spec.remove_component_at(index)
        except Exception as exc:
            self.log_message.emit(f"Failed to remove component at index {index}: {exc}")
            return False
        if removed is None:
            self.log_message.emit(f"No component found at index {index}.")
            return False
        self.log_message.emit(f"Removed component {removed.label}")
        try:
            self.parameters_updated.emit()
        except Exception:
            pass
        try:
            self.update_plot()
        except Exception:
            pass
        return True

    def remove_last_component(self) -> bool:
        spec = self._require_composite("Removing elements")
        if spec is None or not spec.list_components():
            return False
        return self.remove_component_at(len(spec.list_components()) - 1)

    def reorder_component(self, old_index: int, new_index: int) -> bool:
        spec = self._require_composite("Reordering elements")
        if spec is None:
            return False
        try:
            changed = spec.reorder_component(old_index, new_index)
        except Exception as exc:
            self.log_message.emit(f"Failed to reorder components: {exc}")
            return False
        if not changed:
            return False
        self.log_message.emit(f"Reordered component from {old_index} to {new_index}")
        try:
            self.parameters_updated.emit()
        except Exception:
            pass
        try:
            self.update_plot()
        except Exception:
            pass
        return True

    def reorder_components_by_prefix(self, prefix_order: _typing.List[str]) -> bool:
        spec = self._require_composite("Reordering elements")
        if spec is None:
            return False
        try:
            changed = spec.reorder_by_prefix(prefix_order)
        except Exception as exc:
            self.log_message.emit(f"Failed to reorder components: {exc}")
            return False
        if not changed:
            return False
        try:
            self.parameters_updated.emit()
        except Exception:
            pass
        try:
            self.update_plot()
        except Exception:
            pass
        return True

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

        is_composite = isinstance(model_spec, CompositeModelSpec)
        # split fixed toggles (keys ending with '__fixed') from value updates
        fixed_suffix = "__fixed"
        fixed_updates = {}
        value_updates = {}
        for k, v in params.items():
            try:
                if isinstance(k, str) and k.endswith(fixed_suffix):
                    base = k[: -len(fixed_suffix)]
                    fixed_updates[base] = bool(v)
                else:
                    value_updates[k] = v
            except Exception:
                # fallback: treat as value update
                value_updates[k] = v

        applied = []
        # First apply fixed-state updates so UI and fit logic see changes immediately
        for base, fv in fixed_updates.items():
            try:
                if model_spec is not None and base in model_spec.params:
                    try:
                        model_spec.params[base].fixed = bool(fv)
                    except Exception:
                        pass
                    # If composite, also propagate fixed state back to the underlying component param
                    try:
                        if isinstance(model_spec, CompositeModelSpec) and hasattr(model_spec, 'get_link'):
                            link = model_spec.get_link(base)
                            if link and isinstance(link, tuple) and len(link) >= 2:
                                component, pname = link
                                try:
                                    if pname in getattr(component.spec, 'params', {}):
                                        component.spec.params[pname].fixed = bool(fv)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                # also record on the concrete model object if present
                try:
                    setattr(mdl, f"{base}{fixed_suffix}", bool(fv))
                except Exception:
                    pass
                applied.append(f"{base}{fixed_suffix}")
            except Exception:
                pass

        # Now apply value updates (regular parameter assignments)
        for k, v in value_updates.items():
            try:
                # prefer to set attribute on the actual model object
                try:
                    setattr(mdl, k, v)
                    if is_composite and hasattr(model_spec, "set_param_value"):
                        model_spec.set_param_value(k, v)
                    applied.append(k)
                except Exception:
                    # if model doesn't accept attribute, fallback to model_spec if available
                    if model_spec is not None and k in model_spec.params:
                        if is_composite and hasattr(model_spec, "set_param_value"):
                            model_spec.set_param_value(k, v)
                        else:
                            model_spec.params[k].value = v
                        applied.append(k)
                    else:
                        # create attribute on model as last resort
                        try:
                            setattr(mdl, k, v)
                        except Exception:
                            pass
                        if is_composite and hasattr(model_spec, "set_param_value"):
                            try:
                                model_spec.set_param_value(k, v)
                            except Exception:
                                pass
                        applied.append(k)
            except Exception:
                # ignore per-parameter failures
                pass

        # Also update any model_spec values for consistency if both exist
        if model_spec is not None:
            for name in applied:
                if name in model_spec.params:
                    try:
                        value = getattr(mdl, name)
                        if is_composite and hasattr(model_spec, "set_param_value"):
                            model_spec.set_param_value(name, value)
                        else:
                            model_spec.params[name].value = value
                    except Exception:
                        pass

        # Notify and refresh plot
        self.log_message.emit(f"Applied parameters: {', '.join(applied) if applied else 'none'}")
        try:
            self.update_plot()
        except Exception:
            pass
        # Notify views that parameters changed so UI can refresh widgets
        try:
            self.parameters_updated.emit()
        except Exception:
            pass

    # --------------------------
    # Curve selection management
    # --------------------------
    def set_selected_curve(self, curve_id: _typing.Optional[str]):
        """Set the active curve selection. Emits a signal if changed."""
        old = self._selected_curve_id
        if old != curve_id:
            self._selected_curve_id = curve_id
            self.curve_selection_changed.emit(curve_id)
            self.log_message.emit(
                f"Curve selection changed: {curve_id if curve_id else 'None'}"
            )

    def get_selected_curve(self) -> _typing.Optional[str]:
        """Return the currently selected curve ID (or None if nothing selected)."""
        return self._selected_curve_id

    def clear_selected_curve(self):
        """Deselect any active curve."""
        if self._selected_curve_id is not None:
            self._selected_curve_id = None
            self.curve_selection_changed.emit(None)
            self.log_message.emit("Curve selection cleared.")

    # --------------------------
    # Peak interaction hooks (called from the view)
    # --------------------------
    def on_peak_selected(self, x, y):
        """Optional hook called when the View announces a peak was selected.

        Default: no-op. Implementations may choose to select a curve or
        prepare for a drag.
        """
        # default: do nothing, but keep method present so the view can call it
        return

    def on_peak_moved(self, info: dict):
        """Called when the View reports a peak was moved.

        This will map a movement of the selected peak to the model's 'center'
        parameter if present.
        """
        if not isinstance(info, dict):
            return
        new_center = info.get("center", None)
        if new_center is None:
            return

        try:
            val = float(new_center)
        except Exception:
            return

        spec = getattr(self.state, "model_spec", None)
        updates = {}

        # If a composite model component is selected, scope the update to that component
        if isinstance(spec, CompositeModelSpec):
            selected_id = self.get_selected_curve()
            if isinstance(selected_id, str) and selected_id.startswith("component:"):
                prefix = selected_id.split(":", 1)[1]
                for component in spec.list_components():
                    if component.prefix == prefix and hasattr(component, "spec"):
                        params = getattr(component.spec, "params", {}) or {}
                        for name in params.keys():
                            if name.lower() == "center":
                                updates[f"{prefix}{name}"] = val
                                break
                        break

        # Fallback: global parameter names on the active spec
        if not updates:
            params = getattr(spec, "params", {}) if spec is not None else {}
            for candidate in ("center", "Center"):
                if isinstance(params, dict) and candidate in params:
                    updates[candidate] = val
                    break

        if not updates:
            return

        try:
            self.apply_parameters(updates)
            target = ", ".join(updates.keys())
            self.log_message.emit(f"Peak center updated -> {val} ({target})")
        except Exception:
            pass

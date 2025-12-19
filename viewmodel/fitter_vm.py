# viewmodel/fitter_vm.py
from PySide6.QtCore import QObject, Signal, QTimer
import numpy as np
import os
import math
from types import SimpleNamespace

from models import ModelState, CompositeModelSpec, get_model_spec, get_atomic_component_names
from dataio.data_loader import select_and_load_files
from dataio.data_saver import save_dataset
import typing as _typing
from .logging_helpers import log_exception, log_message

# Constants for resolution convolution
DEFAULT_RESOLUTION_RANGE = 20.0  # Range for resolution kernel evaluation (symmetric around 0)
RESOLUTION_PREVIEW_RANGE = 5.0  # Range for preview plot display
PREVIEW_VISIBLE_MARGIN = 2.0  # Visible margin beyond data range (smaller than calculation padding)


class FitterViewModel(QObject):
    """
    Central logic layer: handles loading/saving, fitting, and updates to the plot.
    """

    plot_updated = Signal(object, object, object, object)   # x, y_data, y_fit, y_err
    curve_selection_changed = Signal(object)                # emits curve_id or None
    log_message = Signal(str)
    parameters_updated = Signal()
    files_updated = Signal(object)                          # list of queued file metadata
    resolution_updated = Signal()                           # emitted when resolution changes
    fit_progress = Signal(float)                            # fit progress 0.0 to 1.0
    fit_step_completed = Signal(int, object, object)        # step_num, fit_result, y_fit
    fit_started = Signal()                                  # emitted when fit begins
    fit_finished = Signal(bool)                             # emitted when fit ends (success)
    revert_available_changed = Signal(bool)                 # emitted when revert state changes


    def __init__(self, model_state=None):
        super().__init__()
        self.state = model_state or ModelState()
        self._fit_worker = None
        self._selected_curve_id = None  # currently selected curve ID (str or None)
        self._datasets = []  # queued datasets [(dict)]
        self._active_dataset_index = None
        self.curves: dict = {}
        self._last_blocked_drag = None  # tracks last drag ignored due to fixed params
        self._last_blocked_value_update = None
        
        # Resolution model state
        self._resolution_model_name = "None"  # "None" means no resolution convolution
        self._resolution_spec = None  # BaseModelSpec or None
        
        # Pre-fit state storage for revert functionality (stack for multiple undos)
        self._pre_fit_stack = []  # list of dicts with "params" and "resolution" keys
        self._max_revert_stack_size = 20  # Maximum number of states to keep
        
        # Debounce timer for auto-saving fit state after parameter changes
        self._fit_save_timer = QTimer()
        self._fit_save_timer.setSingleShot(True)
        self._fit_save_timer.setInterval(500)  # 500ms debounce
        self._fit_save_timer.timeout.connect(self._save_current_fit)
        
        # Attempt to restore previously queued files from configuration
        try:
            self._load_queue_from_config()
        except Exception as exc:
            # Failed to restore previous file queue; continue with empty queue.
            log_exception("Could not restore previous file queue", exc, vm=self)

    def _log_message(self, message: str) -> None:
        """Emit a message via the shared logging helper."""
        log_message(message, vm=self)

    def _log_exception(self, context: str, exc: Exception) -> None:
        """Emit an exception context via the shared logging helper."""
        log_exception(context, exc, vm=self)

    # --------------------------
    # Data I/O
    # --------------------------
    def load_data(self):
        """Open file dialog and append one or more datasets to the queue."""
        loaded = select_and_load_files(None)
        if not loaded:
            return

        def _canon_path(p: _typing.Optional[str]) -> _typing.Optional[str]:
            if not p:
                return None
            try:
                return os.path.normcase(os.path.abspath(os.path.normpath(str(p))))
            except Exception:
                return str(p)

        existing_paths = set()
        for ds in self._datasets:
            try:
                info = ds.get("info") if isinstance(ds, dict) else None
                info = info if isinstance(info, dict) else {}
                p = _canon_path(info.get("path"))
                if p:
                    existing_paths.add(p)
            except Exception:
                continue

        added = 0
        skipped_duplicates = 0
        last_info = None
        for x, y, err, info in loaded:
            try:
                x_arr, y_arr, err_arr = self._prepare_dataset(x, y, err)
            except Exception as exc:
                name = (info or {}).get("name") if isinstance(info, dict) else None
                label = name or "dataset"
                log_exception(f"Skipped {label}", exc, vm=self)
                continue

            info_dict = info or {}
            if not isinstance(info_dict, dict):
                info_dict = {}

            p = _canon_path(info_dict.get("path"))
            if p and p in existing_paths:
                skipped_duplicates += 1
                continue
            if p:
                existing_paths.add(p)

            dataset = {
                "x": x_arr,
                "y": y_arr,
                "err": err_arr,
                "info": info_dict,
            }
            self._datasets.append(dataset)
            added += 1
            last_info = info_dict or last_info

        if added == 0:
            return

        if isinstance(last_info, dict):
            self._persist_last_loaded(last_info)

        if skipped_duplicates:
            self._log_message(f"Skipped {skipped_duplicates} duplicate file(s) already in the queue.")

        plural = "file" if added == 1 else "files"
        self._log_message(f"Queued {added} {plural} for viewing.")
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

    def has_queued_files(self) -> bool:
        """Return True if there are queued files."""
        return len(self._datasets) > 0

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
        from .logging_helpers import safe_call
        
        try:
            from dataio import get_config
            from dataio.data_loader import load_data_from_file
            
            cfg = get_config()
            q = getattr(cfg, "queued_files", None)
            active = getattr(cfg, "queued_active", None)
            if not q:
                return
            
            def _canon_path(p: _typing.Optional[str]) -> _typing.Optional[str]:
                if not p:
                    return None
                try:
                    return os.path.normcase(os.path.abspath(os.path.normpath(str(p))))
                except Exception:
                    return str(p)

            existing_paths = set()
            for ds in self._datasets:
                try:
                    info = ds.get("info") if isinstance(ds, dict) else None
                    info = info if isinstance(info, dict) else {}
                    p = _canon_path(info.get("path"))
                    if p:
                        existing_paths.add(p)
                except Exception:
                    continue

            # Attempt to load each path into the queue
            added = 0
            for entry in q:
                path = entry.get("path") if isinstance(entry, dict) else None
                if not path or not os.path.isfile(path):
                    continue

                p = _canon_path(path)
                if p and p in existing_paths:
                    continue
                
                try:
                    x, y, err, info = load_data_from_file(path)
                    x_arr, y_arr, err_arr = self._prepare_dataset(x, y, err)
                    dataset = {"x": x_arr, "y": y_arr, "err": err_arr, "info": info}
                    self._datasets.append(dataset)
                    if p:
                        existing_paths.add(p)
                    added += 1
                except Exception:
                    continue
            
            # Determine desired active index, but do not set _active_dataset_index yet.
            # We want activate_file() to actually apply the dataset and load its fit.
            if isinstance(active, int) and 0 <= active < len(self._datasets):
                desired_active = int(active)
            elif self._datasets:
                desired_active = 0
            else:
                desired_active = None
            self._active_dataset_index = None
            
            # Notify UI and activate if needed
            if added:
                safe_call(self._emit_file_queue, context="emit file queue", vm=self)
                if desired_active is not None:
                    safe_call(self.activate_file, desired_active, 
                             context="activate file in _load_queue_from_config", vm=self)
        except Exception:
            pass

    def _apply_dataset_to_state(self, dataset: dict):
        from .logging_helpers import safe_call
        
        x = dataset.get("x")
        y = dataset.get("y")
        err = dataset.get("err")
        info = dataset.get("info") if isinstance(dataset.get("info"), dict) else {}

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        # Set data using set_data method if available, otherwise set directly
        if hasattr(self.state, 'set_data'):
            try:
                self.state.set_data(np.array(x_arr, copy=True), np.array(y_arr, copy=True))
            except Exception:
                # Fallback: set data directly and initialize
                self.state.x_data = np.array(x_arr, copy=True)
                self.state.y_data = np.array(y_arr, copy=True)
                safe_call(self.state.model_spec.initialize, self.state.x_data, self.state.y_data,
                         context="model_spec.initialize", vm=self)
        else:
            self.state.x_data = np.array(x_arr, copy=True)
            self.state.y_data = np.array(y_arr, copy=True)
        
        # Ensure exclusion mask exists and matches data length
        self.state.excluded = np.zeros_like(self.state.x_data, dtype=bool)

        # Set errors with fallback to sqrt of data
        try:
            self.state.errors = np.array(err, dtype=float, copy=True)
        except Exception:
            self.state.errors = np.sqrt(np.clip(np.abs(self.state.y_data), 1e-12, np.inf))

        # Set file_info if provided
        if info:
            safe_call(setattr, self.state, "file_info", info, 
                     context="set file_info", vm=self)

    def activate_file(self, index: int):
        from .logging_helpers import safe_call, safe_emit
        
        try:
            idx = int(index)
        except (ValueError, TypeError):
            return

        if idx < 0 or idx >= len(self._datasets):
            return

        # Avoid reloading the currently active dataset *only* when it's already
        # applied to state (same file path). This prevents skipping activation
        # during startup queue restore.
        try:
            if self._active_dataset_index is not None and idx == self._active_dataset_index:
                current_path = self._get_current_file_path()
                ds = self._datasets[idx]
                info = ds.get("info") if isinstance(ds, dict) else {}
                info = info if isinstance(info, dict) else {}
                ds_path = info.get("path")
                if current_path and ds_path:
                    try:
                        current_norm = os.path.normcase(os.path.abspath(os.path.normpath(str(current_path))))
                        ds_norm = os.path.normcase(os.path.abspath(os.path.normpath(str(ds_path))))
                    except Exception:
                        current_norm = str(current_path)
                        ds_norm = str(ds_path)
                    if current_norm == ds_norm:
                        return
        except Exception:
            pass

        # Reset any selected curve when swapping datasets so UI doesn't carry state.
        try:
            self.clear_selected_curve()
        except Exception:
            pass

        # Always start from a clean model/fit state before applying new data.
        # This prevents stale model_spec/model_name leaking between files.
        try:
            from models import get_model_spec
            setattr(self.state, "fit_result", None)
            try:
                if hasattr(self.state, "model"):
                    delattr(self.state, "model")
            except Exception:
                pass
            setattr(self.state, "model_name", "Voigt")
            setattr(self.state, "model_spec", get_model_spec("Voigt"))
        except Exception:
            pass

        # Clear resolution state on file swap; file-specific fits may restore it.
        try:
            self._resolution_model_name = "None"
            self._resolution_spec = None
            try:
                self.resolution_updated.emit()
            except Exception:
                pass
        except Exception:
            pass

        dataset = self._datasets[idx]
        self._apply_dataset_to_state(dataset)
        self._active_dataset_index = idx

        # Extract dataset name for logging
        info = dataset.get("info") if isinstance(dataset, dict) else {}
        info = info if isinstance(info, dict) else {}
        name = info.get("name") or f"Dataset {idx + 1}"
        self._log_message(f"Loaded dataset: {name}")

        # Try to load saved fit for this file. If none exists, only apply a
        # default model/fit if the user explicitly configured it.
        fit_loaded = safe_call(
            self._load_fit_for_current_file,
            default=False,
            context="load fit for current file",
            vm=self,
        )

        if not fit_loaded:
            default_choice = None
            try:
                default_choice = self._get_default_model_choice()
            except Exception:
                default_choice = None

            if default_choice and default_choice not in ("(None - User Select)",):
                safe_call(self._load_default_fit, context="load default fit", vm=self)
            else:
                # Ensure the clean default model is initialized for this dataset.
                try:
                    spec = getattr(self.state, "model_spec", None)
                    if spec is not None and hasattr(spec, "initialize"):
                        spec.initialize(getattr(self.state, "x_data", None), getattr(self.state, "y_data", None))
                except Exception:
                    pass

        # Ensure model_name matches model_spec before emitting signals
        safe_call(self._synchronize_model_state, context="synchronize model state", vm=self)

        # Notify UI
        safe_emit(self.parameters_updated, vm=self, signal_name="parameters_updated")
        self._emit_file_queue()
        safe_call(self.update_plot, context="update plot", vm=self)

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
        self._log_message(f"Removed dataset: {name}")

        if self._active_dataset_index == idx:
            self._active_dataset_index = None
            if self._datasets:
                next_idx = idx if idx < len(self._datasets) else len(self._datasets) - 1
                self.activate_file(next_idx)
            else:
                self.clear_loaded_files(emit_log=False)
                self._log_message("Dataset queue is now empty; reverted to default data.")
        else:
            if self._active_dataset_index is not None and idx < self._active_dataset_index:
                self._active_dataset_index -= 1
            self._emit_file_queue()

    def clear_loaded_files(self, emit_log=True):
        from .logging_helpers import safe_emit, safe_call
        
        self._datasets.clear()
        self._active_dataset_index = None
        model_name = getattr(self.state, "model_name", "Voigt") if hasattr(self, "state") else "Voigt"
        try:
            self.state = ModelState(model_name=model_name)
        except Exception:
            self.state = ModelState()
        self._fit_worker = None
        self._emit_file_queue()
        safe_emit(self.parameters_updated, vm=self, signal_name="parameters_updated")
        safe_call(self.update_plot, context="update plot", vm=self)
        if emit_log:
            self._log_message("Cleared queued datasets and restored initial synthetic data.")

    def save_data(self):
        """Save current data and fit to file with multiple export options."""
        try:
            # Import required modules
            from view.dialogs.save_data_dialog import SaveDataDialog
            from dataio.data_exporter import save_as_image, save_as_ascii, save_parameters
            from dataio import get_config
            from PySide6.QtWidgets import QDialog
            from pathlib import Path
            
            # Get default save folder and delimiter from config
            cfg = get_config()
            default_folder = cfg.default_save_folder or str(Path.home())
            
            # Get default filename from loaded file
            default_filename = "data"
            try:
                file_info = getattr(self.state, "file_info", None)
                if file_info and isinstance(file_info, dict):
                    file_path = file_info.get("path")
                    if file_path:
                        default_filename = Path(file_path).stem
            except Exception:
                pass
            
            # Show save dialog
            dialog = SaveDataDialog(parent=None, default_folder=default_folder, default_filename=default_filename)
            
            # Set delimiter from config
            try:
                dialog.set_delimiter_from_config(cfg.save_delimiter)
            except Exception:
                pass
            
            # Connect button signals to perform saves
            def perform_save():
                """Perform the actual save operation based on dialog options."""
                base_path = dialog.get_base_path()
                options = dialog.get_save_options()
                
                if not base_path:
                    dialog.set_status("Error: No path selected.", False)
                    return
                
                # Gather data for export
                x_data = getattr(self.state, "x_data", None)
                y_data = getattr(self.state, "y_data", None)
                y_errors = getattr(self.state, "errors", None)
                excluded_mask = getattr(self.state, "excluded", None)
                file_info = getattr(self.state, "file_info", None)
                model_name = getattr(self.state, "model_name", None)
                
                if x_data is None or y_data is None:
                    dialog.set_status("Error: No data available to save.", False)
                    self._log_message("No data available to save.")
                    return
                
                # Get fit data
                y_fit_dict = None
                try:
                    # Try to get detailed fit with components
                    y_fit_dict = self._get_plot_fit_data()
                except Exception:
                    # Fallback to simple evaluation
                    try:
                        y_fit = self.state.evaluate() if hasattr(self.state, "evaluate") else None
                        if y_fit is not None:
                            y_fit_dict = y_fit
                    except Exception:
                        y_fit_dict = None
                
                # Get parameters and fit result
                parameters = self.get_parameters() if hasattr(self, "get_parameters") else {}
                fit_result = getattr(self.state, "fit_result", None)
                
                # Get delimiter
                delimiter = options.get('delimiter', ',')
                if delimiter == 'comma':
                    delimiter = ','
                elif delimiter == 'tab':
                    delimiter = '\t'
                elif delimiter == 'space':
                    delimiter = ' '
                
                # Perform exports based on options
                margin_percent = options.get('margin_percent', 10.0)
                save_image_flag = options.get('save_image', False)
                save_ascii_flag = options.get('save_ascii', False)
                save_params_flag = options.get('save_parameters', False)
                
                success_count = 0
                failed_exports = []
                saved_files = []
                
                if save_image_flag:
                    image_path = f"{base_path}_image.png"
                    if save_as_image(x_data, y_data, y_fit_dict, y_errors, image_path, 
                                    margin_percent, excluded_mask, file_info):
                        self._log_message(f"Image saved to: {image_path}")
                        saved_files.append("image")
                        success_count += 1
                    else:
                        failed_exports.append("image")
                
                if save_ascii_flag:
                    ascii_path = f"{base_path}_ASCII.txt"
                    if save_as_ascii(x_data, y_data, y_fit_dict, y_errors, ascii_path, excluded_mask, delimiter):
                        self._log_message(f"ASCII data saved to: {ascii_path}")
                        saved_files.append("ASCII")
                        success_count += 1
                    else:
                        failed_exports.append("ASCII data")
                
                if save_params_flag:
                    params_path = f"{base_path}_params.txt"
                    if save_parameters(parameters, fit_result, params_path, model_name, delimiter):
                        self._log_message(f"Parameters saved to: {params_path}")
                        saved_files.append("parameters")
                        success_count += 1
                    else:
                        failed_exports.append("parameters")
                
                # Update status in dialog
                if success_count > 0:
                    status_msg = f"Saved: {', '.join(saved_files)}"
                    dialog.set_status(status_msg, True)
                    self._log_message(f"Save completed: {success_count} file(s) exported successfully.")
                    
                    # Save delimiter preference to config
                    try:
                        delimiter_name = dialog.get_delimiter_name()
                        cfg.save_delimiter = delimiter_name
                        cfg.save()
                    except Exception:
                        pass
                
                if failed_exports:
                    error_msg = f"Failed: {', '.join(failed_exports)}"
                    dialog.set_status(error_msg, False)
                    self._log_message(f"Failed to save: {', '.join(failed_exports)}")
            
            # Connect each button to perform_save
            dialog.save_image_btn.clicked.connect(perform_save)
            dialog.save_ascii_btn.clicked.connect(perform_save)
            dialog.save_params_btn.clicked.connect(perform_save)
            dialog.save_all_btn.clicked.connect(perform_save)
            
            # Show dialog (it won't close on save, only on Close button)
            dialog.exec()
        
        except Exception as e:
            log_exception("Failed to save data", e, vm=self)

    def clear_plot(self):
        """Reset to initial synthetic dataset and clear stored last-loaded file in config."""
        from .logging_helpers import safe_call
        
        try:
            self.clear_loaded_files(emit_log=False)

            # Clear saved last-loaded file in config if available
            def _clear_config():
                from dataio import get_config
                cfg = get_config()
                cfg.last_loaded_file = None
                safe_call(cfg.save, context="save config", vm=self)
            
            safe_call(_clear_config, context="clear config", vm=self)
            self._log_message("Cleared queued datasets and restored initial synthetic data.")
        except Exception as e:
            log_exception("Failed to clear plot", e, vm=self)

    def _synchronize_model_state(self):
        """Ensure model_name matches model_spec type.
        
        This is critical for UI consistency when switching files or loading fits.
        Composite models must show as "Custom Model", atomic models show their spec name.
        """
        try:
            from models.model_specs import CompositeModelSpec
            
            model_spec = getattr(self.state, "model_spec", None)
            if model_spec is None:
                return
            
            if isinstance(model_spec, CompositeModelSpec):
                # All composite models display as "Custom Model"
                setattr(self.state, "model_name", "Custom Model")
            else:
                # Atomic models: use spec's name if available
                spec_name = getattr(model_spec, "name", None)
                if spec_name:
                    setattr(self.state, "model_name", spec_name)
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
            self._log_message("Configuration saved.")
            return True
        except Exception as e:
            log_exception("Failed to save configuration", e, vm=self)
            return False

    def reload_config(self):
        """Reload the project's configuration from disk and notify UI."""
        try:
            # import here to avoid circular import at module import time
            from dataio import get_config
            cfg = get_config(recreate=True)
            self._log_message(f"Config reloaded from: {cfg.config_path}")
        except Exception as e:
            log_exception("Failed to reload config", e, vm=self)
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
            self._log_message("handle_action: no action provided")
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
            "reset_fit": self.reset_fit,
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
            log_exception(f"handle_action('{action}') failed", e, vm=self)
        self._log_message(f"Unknown action requested: '{action}'")
        return None

    # --------------------------
    # Fit + Plot logic
    # --------------------------
    def run_fit(self):
        """Start an asynchronous fit using FitWorker (non-blocking)."""
        # Prevent multiple concurrent fits
        if getattr(self, "_fit_worker", None) is not None:
            self._log_message("A fit is already running.")
            return

        # Store pre-fit state so Run Fit behaves like Fit Dock and can be reverted
        try:
            self._store_pre_fit_state()
        except Exception:
            pass

        # Emit fit_started signal so UI can disable buttons
        try:
            self.fit_started.emit()
        except Exception:
            pass

        # basic data checks
        # Use masked (included) data for fitting so excluded points are ignored
        try:
            x_incl, y_incl, err_incl = self.state.get_masked_data()
        except Exception:
            x_incl, y_incl, err_incl = None, None, None
        x = x_incl if x_incl is not None else getattr(self.state, "x_data", None)
        y = y_incl if y_incl is not None else getattr(self.state, "y_data", None)
        if x is None or y is None:
            self._log_message("No data available to fit.")
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
                log_exception("Unable to obtain model specification", e, vm=self)
                return

        # ensure model_spec provides an evaluate function
        model_func = getattr(model_spec, "evaluate", None)
        if model_func is None:
            self._log_message("Selected model does not provide an evaluate(x, **params) function.")
            return

        # build initial parameter value map and separate free (optimised) parameters
        try:
            model_params_map = getattr(model_spec, "params", {}) or {}
            model_full_values = {k: getattr(v, "value", None) for k, v in model_params_map.items()}
            # Determine which parameters are free (not fixed)
            model_free_keys = [k for k, v in model_params_map.items() if not bool(getattr(v, "fixed", False))]
            # Build link_group mapping: for linked parameters, only fit one representative
            model_link_groups = {}  # link_group -> [param_names]
            model_link_representatives = {}  # param_name -> representative_name (for linked params)
            
            for k in model_free_keys:
                try:
                    lg = getattr(model_params_map[k], "link_group", None)
                    if lg and lg > 0:
                        model_link_groups.setdefault(lg, []).append(k)
                except Exception:
                    pass
            
            # For each link group, choose the first parameter as the representative
            for lg, param_names in model_link_groups.items():
                if len(param_names) > 1:
                    representative = param_names[0]
                    for pname in param_names:
                        model_link_representatives[pname] = representative
            
            # Build unique free parameter list (only one param per link group)
            model_unique_free_keys = []
            seen_representatives = set()
            for k in model_free_keys:
                rep = model_link_representatives.get(k, k)
                if rep not in seen_representatives:
                    model_unique_free_keys.append(rep)
                    seen_representatives.add(rep)
            
            if not model_unique_free_keys and not self.has_resolution():
                self._log_message("No free model parameters to fit (all model parameters are fixed or linked).")

            # bounds for unique free params (ordered to match model_unique_free_keys)
            model_lower = [getattr(model_params_map[k], "min", None) if getattr(model_params_map[k], "min", None) is not None else -np.inf for k in model_unique_free_keys]
            model_upper = [getattr(model_params_map[k], "max", None) if getattr(model_params_map[k], "max", None) is not None else np.inf for k in model_unique_free_keys]
        except Exception as e:
            log_exception("Failed to build parameter/bounds list", e, vm=self)
            return

        # Resolution parameter handling (optional)
        res_spec = self._resolution_spec if self.has_resolution() else None
        res_params_map = getattr(res_spec, "params", {}) or {}
        res_name_map = {}  # prefixed name -> base name
        res_full_values = {}
        res_link_groups = {}
        res_link_representatives = {}
        res_unique_free_keys = []
        res_lower = []
        res_upper = []
        if res_spec is not None:
            try:
                # Build prefixed names to avoid collisions with model parameters
                res_full_values = {f"res__{k}": getattr(v, "value", None) for k, v in res_params_map.items()}
                for base_name, param in res_params_map.items():
                    prefixed = f"res__{base_name}"
                    res_name_map[prefixed] = base_name
                    if not bool(getattr(param, "fixed", False)):
                        res_unique_free_keys.append(prefixed)
                        lg = getattr(param, "link_group", None)
                        if lg and lg > 0:
                            res_link_groups.setdefault(lg, []).append(base_name)
                # pick representatives per link group (prefixed)
                for lg, names in res_link_groups.items():
                    if len(names) > 1:
                        rep_pref = f"res__{names[0]}"
                        for n in names:
                            res_link_representatives[f"res__{n}"] = rep_pref
                res_lower = [getattr(res_params_map[res_name_map[k]], "min", None) if getattr(res_params_map[res_name_map[k]], "min", None) is not None else -np.inf for k in res_unique_free_keys]
                res_upper = [getattr(res_params_map[res_name_map[k]], "max", None) if getattr(res_params_map[res_name_map[k]], "max", None) is not None else np.inf for k in res_unique_free_keys]
            except Exception as e:
                log_exception("Failed to build resolution parameter list", e, vm=self)
                res_unique_free_keys = []
                res_lower = []
                res_upper = []

        # Combine model + resolution parameters for fitting
        param_keys = list(model_unique_free_keys) + list(res_unique_free_keys)
        if not param_keys:
            self._log_message("No free parameters to fit (model and resolution are fixed).")
            return
        combined_full_values = dict(model_full_values)
        combined_full_values.update(res_full_values)
        params = {k: combined_full_values.get(k) for k in param_keys}
        bounds = (list(model_lower) + list(res_lower), list(model_upper) + list(res_upper))

        # Capture resolution state for use inside wrapped_func
        has_res = res_spec is not None
        evaluate_with_res = self.evaluate_with_resolution if has_res else None

        def wrapped_func(xx, *args):
            # start from the full map (including fixed values) and overwrite free keys
            p = dict(combined_full_values)
            try:
                for k, val in zip(param_keys, args):
                    p[k] = val
                    # If this parameter is a representative for linked params, propagate value
                    for orig_k, rep_k in model_link_representatives.items():
                        if rep_k == k:
                            p[orig_k] = val
                    for orig_k, rep_k in res_link_representatives.items():
                        if rep_k == k:
                            p[orig_k] = val
            except Exception:
                pass
            # IMPORTANT: some ModelSpec.evaluate implementations expect a single 'params' dict
            # as the second positional argument (evaluate(x, params)). Others accept **kwargs.
            # Pass the params dict positionally to support evaluate(x, params).
            model_param_values = {k: p.get(k) for k in model_full_values.keys()}
            try:
                y_out = model_func(xx, model_param_values)
            except TypeError:
                # fallback: try as kwargs (in case evaluate accepts keyword args)
                y_out = model_func(xx, **model_param_values)
            
            # Apply resolution convolution if enabled
            if has_res and evaluate_with_res is not None:
                # Update resolution spec values with current params for this evaluation
                try:
                    res_spec_local = getattr(self, "_resolution_spec", None)
                    if res_spec_local is not None and hasattr(res_spec_local, "params"):
                        for pref_name, base_name in res_name_map.items():
                            if pref_name in p and base_name in res_spec_local.params:
                                res_spec_local.params[base_name].value = p[pref_name]
                except Exception:
                    pass
                try:
                    y_out = evaluate_with_res(xx, y_out)
                except Exception:
                    pass
            
            return y_out

        # lazy import of FitWorker to reduce module-time coupling
        try:
            from worker.fit_worker import FitWorker
        except Exception as e:
            log_exception("Failed to import FitWorker", e, vm=self)
            return

        worker = FitWorker(x, y, wrapped_func, params, err, bounds)
        self._fit_worker = worker

        # connect progress updates
        worker.progress.connect(lambda p: self._log_message(f"Fit progress: {int(p*100)}%"))

        # finished handler
        def on_finished(result, y_fit):
            try:
                combined_result = result or {}
                model_result = {k: v for k, v in combined_result.items() if not (isinstance(k, str) and k.startswith("res__"))}
                resolution_result = {res_name_map.get(k, k): v for k, v in combined_result.items() if isinstance(k, str) and k.startswith("res__")}

                if combined_result:
                    # store fit result on state and update plot
                    self.state.fit_result = model_result

                    # Apply fit result back into model_spec.params where possible so UI will reflect fitted values
                    # Also propagate values to linked parameters
                    try:
                        spec = getattr(self.state, "model_spec", None)
                        if spec is not None:
                            for k, v in model_result.items():
                                try:
                                    if isinstance(spec, CompositeModelSpec) and hasattr(spec, "set_param_value"):
                                        spec.set_param_value(k, v)
                                    elif hasattr(spec, "params") and k in spec.params:
                                        spec.params[k].value = v
                                    
                                    # Propagate to linked parameters
                                    for orig_k, rep_k in model_link_representatives.items():
                                        if rep_k == k and orig_k != k:
                                            try:
                                                if isinstance(spec, CompositeModelSpec) and hasattr(spec, "set_param_value"):
                                                    spec.set_param_value(orig_k, v)
                                                elif hasattr(spec, "params") and orig_k in spec.params:
                                                    spec.params[orig_k].value = v
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    # Also set attributes on any concrete model object for consistency
                    try:
                        mdl = getattr(self.state, "model", None)
                        if mdl is not None:
                            for k, v in model_result.items():
                                try:
                                    setattr(mdl, k, v)
                                    # Propagate to linked parameters
                                    for orig_k, rep_k in model_link_representatives.items():
                                        if rep_k == k and orig_k != k:
                                            try:
                                                setattr(mdl, orig_k, v)
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    # Apply fitted resolution parameters back to the resolution spec
                    try:
                        if resolution_result and res_spec is not None:
                            res_params = getattr(res_spec, "params", {}) or {}
                            for base_name, val in resolution_result.items():
                                if base_name in res_params:
                                    try:
                                        res_params[base_name].value = val
                                        lg = getattr(res_params[base_name], "link_group", None)
                                        if lg and lg in res_link_groups:
                                            for linked_base in res_link_groups[lg]:
                                                if linked_base != base_name and linked_base in res_params:
                                                    res_params[linked_base].value = val
                                    except Exception:
                                        pass
                            try:
                                self.resolution_updated.emit()
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
                                full_params = dict(model_full_values)
                                try:
                                    full_params.update(model_result or {})
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

                    self._log_message("Fit completed successfully.")

                    # Auto-save fit after successful fit
                    try:
                        self._schedule_fit_save()
                    except Exception as e:
                        log_exception("Auto-save fit failed", e)
                    
                    # Emit fit_finished signal so UI can re-enable buttons
                    try:
                        self.fit_finished.emit(True)
                    except Exception:
                        pass
                else:
                    self._log_message("Fit failed.")
                    # Emit fit_finished signal even on failure so UI can re-enable buttons
                    try:
                        self.fit_finished.emit(False)
                    except Exception:
                        pass
            finally:
                # always clear the worker reference
                try:
                    self._fit_worker = None
                except Exception:
                    pass

        worker.finished.connect(on_finished)
        worker.start()
        self._log_message("Fit started in background...")

    def _store_pre_fit_state(self):
        """Store the current parameter state before fitting for revert functionality."""
        try:
            model_spec = getattr(self.state, "model_spec", None)
            if model_spec is None:
                return
            
            # Store current parameter values
            pre_fit_params = {}
            params = getattr(model_spec, "params", {}) or {}
            for name, param in params.items():
                try:
                    pre_fit_params[name] = {
                        "value": getattr(param, "value", None),
                        "fixed": bool(getattr(param, "fixed", False)),
                        "link_group": getattr(param, "link_group", None),
                        "min": getattr(param, "min", None),
                        "max": getattr(param, "max", None),
                    }
                except Exception:
                    pass
            
            # Store resolution state
            pre_fit_resolution = self.get_resolution_state()
            
            # Push to stack
            state_entry = {
                "params": pre_fit_params,
                "resolution": pre_fit_resolution
            }
            self._pre_fit_stack.append(state_entry)
            
            # Trim stack if too large
            while len(self._pre_fit_stack) > self._max_revert_stack_size:
                self._pre_fit_stack.pop(0)
            
            try:
                self.revert_available_changed.emit(True)
            except Exception:
                pass
            
            self._log_message(f"Pre-fit parameters stored ({len(self._pre_fit_stack)} states available).")
        except Exception as e:
            log_exception("Failed to store pre-fit state", e, vm=self)

    def has_revert_state(self) -> bool:
        """Check if there's a pre-fit state available for revert."""
        return len(self._pre_fit_stack) > 0
    
    def get_revert_stack_depth(self) -> int:
        """Get the number of states available for revert."""
        return len(self._pre_fit_stack)

    def revert_to_pre_fit(self) -> bool:
        """Revert parameters to their pre-fit values.
        
        Returns:
            True if revert was successful
        """
        try:
            if not self.has_revert_state():
                self._log_message("No pre-fit state available to revert to.")
                return False
            
            # Pop the most recent state from the stack
            state_entry = self._pre_fit_stack.pop()
            pre_fit_params = state_entry.get("params", {})
            pre_fit_resolution = state_entry.get("resolution")
            
            model_spec = getattr(self.state, "model_spec", None)
            if model_spec is None:
                self._log_message("No model spec available for revert.")
                return False
            
            params = getattr(model_spec, "params", {}) or {}
            mdl = getattr(self.state, "model", None)
            is_composite = isinstance(model_spec, CompositeModelSpec)
            
            # Restore parameter values
            for name, stored in pre_fit_params.items():
                try:
                    if name in params:
                        params[name].value = stored.get("value")
                        params[name].fixed = stored.get("fixed", False)
                        params[name].link_group = stored.get("link_group")
                        if stored.get("min") is not None:
                            params[name].min = stored.get("min")
                        if stored.get("max") is not None:
                            params[name].max = stored.get("max")
                        
                        # Update model object if present
                        if mdl is not None:
                            try:
                                setattr(mdl, name, stored.get("value"))
                            except Exception:
                                pass
                        
                        # Update composite spec if applicable
                        if is_composite and hasattr(model_spec, "set_param_value"):
                            try:
                                model_spec.set_param_value(name, stored.get("value"))
                            except Exception:
                                pass
                except Exception:
                    pass
            
            # Restore resolution state
            if pre_fit_resolution is not None:
                self.set_resolution_state(pre_fit_resolution)
            
            # Update revert availability
            try:
                self.revert_available_changed.emit(self.has_revert_state())
            except Exception:
                pass
            
            # Notify views
            try:
                self.parameters_updated.emit()
            except Exception:
                pass
            
            try:
                self.update_plot()
            except Exception:
                pass
            
            remaining = len(self._pre_fit_stack)
            self._log_message(f"Reverted to pre-fit parameters ({remaining} more states available).")
            return True
        except Exception as e:
            log_exception("Failed to revert to pre-fit state", e, vm=self)
            return False

    def run_fit_steps(self, num_steps: int = 1, live_preview: bool = True):
        """Run a limited number of fit iterations with optional live preview.
        
        Args:
            num_steps: Number of fitting iterations to run
            live_preview: If True, update plot after each step
        """
        # Prevent multiple concurrent fits
        if getattr(self, "_fit_worker", None) is not None:
            self._log_message("A fit is already running.")
            return
        
        # Store pre-fit state
        self._store_pre_fit_state()
        
        # Emit fit started signal
        try:
            self.fit_started.emit()
        except Exception:
            pass

        # Get data
        try:
            x_incl, y_incl, err_incl = self.state.get_masked_data()
        except Exception:
            x_incl, y_incl, err_incl = None, None, None
        x = x_incl if x_incl is not None else getattr(self.state, "x_data", None)
        y = y_incl if y_incl is not None else getattr(self.state, "y_data", None)
        if x is None or y is None:
            self._log_message("No data available to fit.")
            try:
                self.fit_finished.emit(False)
            except Exception:
                pass
            return

        err = err_incl if err_incl is not None else getattr(self.state, "errors", None)

        # Get model spec
        model_spec = getattr(self.state, "model_spec", None)
        if model_spec is None:
            self._log_message("No model available to fit.")
            try:
                self.fit_finished.emit(False)
            except Exception:
                pass
            return

        # Build parameter info (same as run_fit)
        model_func = getattr(model_spec, "evaluate", None)
        if model_func is None:
            self._log_message("Selected model does not provide an evaluate function.")
            try:
                self.fit_finished.emit(False)
            except Exception:
                pass
            return

        try:
            # Build model parameter maps
            model_params_map = getattr(model_spec, "params", {}) or {}
            model_full_values = {k: getattr(v, "value", None) for k, v in model_params_map.items()}
            model_free_keys = [k for k, v in model_params_map.items() if not bool(getattr(v, "fixed", False))]

            # Link groups for model params
            model_link_groups = {}
            model_link_representatives = {}
            for k in model_free_keys:
                try:
                    lg = getattr(model_params_map[k], "link_group", None)
                    if lg and lg > 0:
                        model_link_groups.setdefault(lg, []).append(k)
                except Exception:
                    pass
            for lg, names in model_link_groups.items():
                if len(names) > 1:
                    rep = names[0]
                    for n in names:
                        model_link_representatives[n] = rep

            model_unique_free_keys = []
            seen = set()
            for k in model_free_keys:
                rep = model_link_representatives.get(k, k)
                if rep not in seen:
                    model_unique_free_keys.append(rep)
                    seen.add(rep)

            model_lower = [getattr(model_params_map[k], "min", None) if getattr(model_params_map[k], "min", None) is not None else -np.inf for k in model_unique_free_keys]
            model_upper = [getattr(model_params_map[k], "max", None) if getattr(model_params_map[k], "max", None) is not None else np.inf for k in model_unique_free_keys]

            # Resolution parameters (optional)
            res_spec = self._resolution_spec if self.has_resolution() else None
            res_params_map = getattr(res_spec, "params", {}) or {}
            res_name_map = {}
            res_full_values = {}
            res_link_groups = {}
            res_link_representatives = {}
            res_unique_free_keys = []
            res_lower = []
            res_upper = []
            if res_spec is not None:
                try:
                    res_full_values = {f"res__{k}": getattr(v, "value", None) for k, v in res_params_map.items()}
                    for base_name, param in res_params_map.items():
                        pref = f"res__{base_name}"
                        res_name_map[pref] = base_name
                        if not bool(getattr(param, "fixed", False)):
                            res_unique_free_keys.append(pref)
                            lg = getattr(param, "link_group", None)
                            if lg and lg > 0:
                                res_link_groups.setdefault(lg, []).append(base_name)
                    for lg, names in res_link_groups.items():
                        if len(names) > 1:
                            rep_pref = f"res__{names[0]}"
                            for n in names:
                                res_link_representatives[f"res__{n}"] = rep_pref
                    res_lower = [getattr(res_params_map[res_name_map[k]], "min", None) if getattr(res_params_map[res_name_map[k]], "min", None) is not None else -np.inf for k in res_unique_free_keys]
                    res_upper = [getattr(res_params_map[res_name_map[k]], "max", None) if getattr(res_params_map[res_name_map[k]], "max", None) is not None else np.inf for k in res_unique_free_keys]
                except Exception as e:
                    log_exception("Failed to build resolution parameter list", e, vm=self)
                    res_unique_free_keys = []
                    res_lower = []
                    res_upper = []

            # Combine model + resolution free keys
            param_keys = list(model_unique_free_keys) + list(res_unique_free_keys)
            if not param_keys:
                self._log_message("No free parameters to fit.")
                try:
                    self.fit_finished.emit(False)
                except Exception:
                    pass
                return

            combined_full_values = dict(model_full_values)
            combined_full_values.update(res_full_values)
            params = {k: combined_full_values.get(k) for k in param_keys}
            bounds = (list(model_lower) + list(res_lower), list(model_upper) + list(res_upper))
        except Exception as e:
            log_exception("Failed to build parameter list", e, vm=self)
            try:
                self.fit_finished.emit(False)
            except Exception:
                pass
            return

        # Create wrapped function that maps positional args into the combined
        # parameter map and evaluates model (+ resolution convolution if active)
        has_res = self.has_resolution()

        def wrapped_func(xx, *args):
            p = dict(combined_full_values)
            try:
                for k, val in zip(param_keys, args):
                    p[k] = val
                # propagate model-linked representatives
                for linked_param, representative in model_link_representatives.items():
                    if linked_param != representative and representative in p:
                        p[linked_param] = p[representative]
                # propagate resolution-linked representatives
                for linked_pref, rep_pref in res_link_representatives.items():
                    if linked_pref != rep_pref and rep_pref in p:
                        p[linked_pref] = p[rep_pref]
            except Exception:
                pass

            # Build model-only param dict to pass to evaluate(x, params)
            model_param_values = {k: p.get(k) for k in model_full_values.keys()}
            try:
                y_out = model_func(xx, model_param_values)
            except TypeError:
                try:
                    y_out = model_func(xx, **model_param_values)
                except Exception:
                    y_out = np.zeros_like(xx)

            # Update resolution spec with current trial params before convolution
            if has_res:
                try:
                    res_spec_local = getattr(self, "_resolution_spec", None)
                    if res_spec_local is not None and hasattr(res_spec_local, "params"):
                        for pref_name, base_name in res_name_map.items():
                            if pref_name in p and base_name in res_spec_local.params:
                                try:
                                    res_spec_local.params[base_name].value = p[pref_name]
                                except Exception:
                                    pass
                except Exception:
                    pass

                try:
                    y_out = self.evaluate_with_resolution(xx, y_out)
                except Exception:
                    pass

            return y_out

        # Import iterative worker
        try:
            from worker.fit_worker import IterativeFitWorker
        except Exception as e:
            log_exception("Failed to import IterativeFitWorker", e, vm=self)
            try:
                self.fit_finished.emit(False)
            except Exception:
                pass
            return

        worker = IterativeFitWorker(
            x, y, wrapped_func, params, err, bounds,
            max_steps=num_steps,
            step_mode=live_preview
        )
        self._fit_worker = worker

        # Connect progress
        worker.progress.connect(lambda p: self.fit_progress.emit(p))

        # Connect error signal for user feedback
        def on_error(error_msg):
            self._log_message(f"Fit error: {error_msg}")
        worker.error_occurred.connect(on_error)

        # Connect step completed for live preview
        if live_preview:
            def on_step_completed(step_num, result, y_fit, chi_squared=None):
                if result is None:
                    return
                try:
                    self.fit_step_completed.emit(step_num, result, y_fit)
                    # Log chi-squared for user feedback
                    if chi_squared is not None:
                        self._log_message(f"Step {step_num}: χ² = {chi_squared:.4g}")

                    # Separate model vs resolution results (resolution keys start with 'res__')
                    combined_result = result or {}
                    model_result = {k: v for k, v in combined_result.items() if not (isinstance(k, str) and k.startswith("res__"))}
                    resolution_result = {res_name_map.get(k, k): v for k, v in combined_result.items() if isinstance(k, str) and k.startswith("res__")}

                    # Update model spec parameters for preview
                    spec = getattr(self.state, "model_spec", None)
                    if spec is not None:
                        is_comp = isinstance(spec, CompositeModelSpec)
                        for k, v in model_result.items():
                            try:
                                if is_comp and hasattr(spec, "set_param_value"):
                                    try:
                                        spec.set_param_value(k, v)
                                    except Exception:
                                        if k in getattr(spec, "params", {}):
                                            spec.params[k].value = v
                                else:
                                    if k in getattr(spec, "params", {}):
                                        spec.params[k].value = v
                            except Exception:
                                pass

                    # Also update concrete model attributes so evaluate() sees the temporary change
                    mdl = getattr(self.state, "model", None)
                    if mdl is not None:
                        for k, v in model_result.items():
                            try:
                                setattr(mdl, k, v)
                            except Exception:
                                pass

                    # Propagate linked model params
                    for linked_param, representative in model_link_representatives.items():
                        if linked_param != representative and representative in model_result:
                            try:
                                val = model_result[representative]
                                if spec is not None:
                                    if isinstance(spec, CompositeModelSpec) and hasattr(spec, "set_param_value"):
                                        try:
                                            spec.set_param_value(linked_param, val)
                                        except Exception:
                                            if linked_param in getattr(spec, "params", {}):
                                                spec.params[linked_param].value = val
                                    else:
                                        if linked_param in getattr(spec, "params", {}):
                                            spec.params[linked_param].value = val
                                if mdl is not None:
                                    try:
                                        setattr(mdl, linked_param, val)
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                    # Apply resolution preview updates to resolution spec
                    try:
                        if resolution_result and res_spec is not None:
                            res_params = getattr(res_spec, "params", {}) or {}
                            for base_name, val in resolution_result.items():
                                if base_name in res_params:
                                    try:
                                        res_params[base_name].value = val
                                        lg = getattr(res_params[base_name], "link_group", None)
                                        if lg and lg in res_link_groups:
                                            for linked_base in res_link_groups[lg]:
                                                if linked_base != base_name and linked_base in res_params:
                                                    res_params[linked_base].value = val
                                    except Exception:
                                        pass
                            try:
                                self.resolution_updated.emit()
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Update plot
                    try:
                        self.update_plot()
                    except Exception:
                        pass
                except Exception as e:
                    log_exception("Error in step completed handler", e, vm=self)

            worker.step_completed.connect(on_step_completed)

        # Connect finished
        def on_finished(result, y_fit):
            try:
                if result is not None:
                    self.state.fit_result = result
                    # Apply final result including linked parameters
                    spec = getattr(self.state, "model_spec", None)
                    if spec is not None:
                        # Apply to fitted params
                        for k, v in result.items():
                            try:
                                if isinstance(spec, CompositeModelSpec) and hasattr(spec, "set_param_value"):
                                    spec.set_param_value(k, v)
                                elif hasattr(spec, "params") and k in spec.params:
                                    spec.params[k].value = v
                            except Exception:
                                pass
                        # Propagate to linked params
                        for linked_param, representative in model_link_representatives.items():
                            if linked_param != representative and representative in result:
                                try:
                                    if isinstance(spec, CompositeModelSpec) and hasattr(spec, "set_param_value"):
                                        spec.set_param_value(linked_param, result[representative])
                                    elif hasattr(spec, "params") and linked_param in spec.params:
                                        spec.params[linked_param].value = result[representative]
                                except Exception:
                                    pass
                    
                    # Update model object
                    mdl = getattr(self.state, "model", None)
                    if mdl is not None:
                        for k, v in result.items():
                            try:
                                setattr(mdl, k, v)
                            except Exception:
                                pass
                        # Also set linked params on model
                        for linked_param, representative in model_link_representatives.items():
                            if linked_param != representative and representative in result:
                                try:
                                    setattr(mdl, linked_param, result[representative])
                                except Exception:
                                    pass
                    
                    try:
                        self.parameters_updated.emit()
                    except Exception:
                        pass
                    
                    try:
                        self.update_plot()
                    except Exception:
                        pass
                    
                    self._log_message(f"Fit completed ({num_steps} step(s)).")
                    
                    try:
                        self._schedule_fit_save()
                    except Exception:
                        pass
                    
                    try:
                        self.fit_finished.emit(True)
                    except Exception:
                        pass
                else:
                    self._log_message("Fit failed.")
                    try:
                        self.fit_finished.emit(False)
                    except Exception:
                        pass
            finally:
                try:
                    self._fit_worker = None
                except Exception:
                    pass

        worker.finished.connect(on_finished)
        worker.start()
        self._log_message(f"Fit started ({num_steps} step(s))...")

    def run_fit_to_completion(self, live_preview: bool = True):
        """Run fit until convergence with optional live preview.
        
        Args:
            live_preview: If True, update plot during fitting
        """
        # Store pre-fit state first
        self._store_pre_fit_state()

        # If live preview is requested, prefer the iterative worker so each step
        # can be emitted and the GUI updated. Otherwise fall back to the fast
        # non-iterative `run_fit` implementation.
        if live_preview:
            try:
                # Choose a reasonably large max steps; the IterativeFitWorker will
                # stop early on convergence. This keeps the behavior responsive.
                max_steps = 1000
                self.run_fit_steps(num_steps=max_steps, live_preview=True)
            except Exception:
                # Fallback to original behavior on error
                try:
                    self.run_fit()
                except Exception:
                    pass
        else:
            # run_fit emits fit_started and fit_finished in its worker callback
            self.run_fit()

    def set_parameter_bounds(self, name: str, min_val, max_val):
        """Set min/max bounds for a parameter.
        
        Args:
            name: Parameter name
            min_val: Minimum value or None for no limit
            max_val: Maximum value or None for no limit
        """
        try:
            model_spec = getattr(self.state, "model_spec", None)
            if model_spec is None:
                return
            
            params = getattr(model_spec, "params", {}) or {}
            if name not in params:
                return
            
            param = params[name]
            if min_val is not None:
                param.min = float(min_val)
            else:
                param.min = None
            
            if max_val is not None:
                param.max = float(max_val)
            else:
                param.max = None
            
            # For CompositeModelSpec, also update the component's underlying spec
            # to ensure bounds persist across _rebuild_flat_params calls
            if isinstance(model_spec, CompositeModelSpec):
                try:
                    param_links = getattr(model_spec, "_param_links", {})
                    if name in param_links:
                        component, orig_name = param_links[name]
                        if hasattr(component, "spec") and hasattr(component.spec, "params"):
                            orig_param = component.spec.params.get(orig_name)
                            if orig_param is not None:
                                orig_param.min = param.min
                                orig_param.max = param.max
                except (AttributeError, KeyError, TypeError):
                    # Component spec structure may vary; silently skip if bounds can't be propagated
                    pass
            
            self._log_message(f"Bounds for '{name}' set to [{min_val}, {max_val}]")
        except Exception as e:
            log_exception(f"Failed to set bounds for '{name}'", e, vm=self)

    def get_parameter_bounds(self) -> dict:
        """Get the current bounds for all parameters.
        
        Returns:
            Dict mapping parameter names to (min, max) tuples
        """
        bounds = {}
        try:
            model_spec = getattr(self.state, "model_spec", None)
            if model_spec is None:
                return bounds
            
            params = getattr(model_spec, "params", {}) or {}
            for name, param in params.items():
                try:
                    min_val = getattr(param, "min", None)
                    max_val = getattr(param, "max", None)
                    bounds[name] = (min_val, max_val)
                except Exception:
                    pass
        except Exception:
            pass
        return bounds

    def is_fit_running(self) -> bool:
        """Check if a fit is currently running."""
        return getattr(self, "_fit_worker", None) is not None

    def update_plot(self):
        """Update plot without running a fit."""
        x = getattr(self.state, "x_data", None)
        y = getattr(self.state, "y_data", None)
        if x is None or y is None:
            return

        try:
            x_arr = np.asarray(x, dtype=float)
        except Exception:
            x_arr = np.array([], dtype=float)

        model_spec = getattr(self.state, "model_spec", None)
        y_fit_payload = None
        curves_payload = {}

        preview_grid = self._build_preview_grid(x_arr)
        preview_enabled = preview_grid.size > x_arr.size

        if isinstance(model_spec, CompositeModelSpec):
            try:
                component_outputs = model_spec.evaluate_components(x_arr)
            except Exception:
                component_outputs = []

            preview_lookup = {}
            if preview_enabled:
                try:
                    preview_outputs = model_spec.evaluate_components(preview_grid)
                    for component, values in preview_outputs:
                        try:
                            preview_lookup[component.prefix] = np.asarray(values, dtype=float)
                        except Exception:
                            preview_lookup[component.prefix] = np.array([], dtype=float)
                except Exception:
                    preview_lookup = {}
                    preview_enabled = False

            total = np.zeros_like(x_arr, dtype=float)
            total_preview = np.zeros_like(preview_grid, dtype=float) if preview_enabled else None
            preview_complete = bool(preview_enabled)
            components_for_view = []

            for component, values in component_outputs:
                arr = np.asarray(values, dtype=float)
                if arr.shape != x_arr.shape:
                    continue
                try:
                    total += arr
                except Exception as e:
                    # Non-fatal: if a component's output cannot be summed, skip it but show in plot.
                    log_exception(
                        f"Could not add component '{component.prefix}' to total fit",
                        e,
                        vm=self,
                    )
                disp_x = x_arr
                disp_y = arr
                if preview_enabled:
                    preview_arr = np.asarray(preview_lookup.get(component.prefix, np.array([])), dtype=float)
                    if preview_arr.size == preview_grid.size:
                        disp_x = preview_grid
                        disp_y = preview_arr
                        try:
                            if total_preview is not None:
                                total_preview += preview_arr
                        except Exception:
                            preview_complete = False
                    else:
                        preview_complete = False
                components_for_view.append(
                    {
                        "prefix": component.prefix,
                        "label": component.label,
                        "color": component.color,
                        "x": np.asarray(disp_x, dtype=float),
                        "y": np.asarray(disp_y, dtype=float),
                    }
                )
                curves_payload[f"component:{component.prefix}"] = (
                    np.asarray(disp_x, dtype=float),
                    np.asarray(disp_y, dtype=float),
                )

            if preview_complete and total_preview is not None and total_preview.size == preview_grid.size:
                fit_x = np.asarray(preview_grid, dtype=float)
                fit_y = np.asarray(total_preview, dtype=float)
            else:
                fit_x = np.asarray(x_arr, dtype=float)
                fit_y = np.asarray(total, dtype=float)
                preview_complete = False

            # Apply resolution convolution if enabled
            if self.has_resolution():
                try:
                    fit_y = self.evaluate_with_resolution(fit_x, fit_y)
                    # Also apply to component displays
                    for comp_data in components_for_view:
                        comp_y = comp_data.get("y")
                        comp_x = comp_data.get("x")
                        if comp_y is not None and comp_x is not None:
                            comp_data["y"] = self.evaluate_with_resolution(comp_x, comp_y)
                    # Update total for data_y
                    total = self.evaluate_with_resolution(x_arr, total)
                    
                    # Trim preview to visible range (data range + small margin)
                    # to hide edge artifacts while keeping the plot focused on data
                    data_min, data_max = float(np.min(x_arr)), float(np.max(x_arr))
                    fit_x, fit_y = self._trim_to_visible_range(fit_x, fit_y, data_min, data_max)
                    for comp_data in components_for_view:
                        comp_x = comp_data.get("x")
                        comp_y = comp_data.get("y")
                        if comp_x is not None and comp_y is not None:
                            comp_data["x"], comp_data["y"] = self._trim_to_visible_range(
                                comp_x, comp_y, data_min, data_max
                            )
                except Exception as e:
                    log_exception("Failed to apply resolution in update_plot", e, vm=self)

            # Update curves_payload for components with trimmed data
            for comp_data in components_for_view:
                prefix = comp_data.get("prefix")
                if prefix:
                    curves_payload[f"component:{prefix}"] = (
                        np.asarray(comp_data.get("x", np.array([])), dtype=float),
                        np.asarray(comp_data.get("y", np.array([])), dtype=float),
                    )
            
            curves_payload["fit"] = (fit_x, fit_y)
            y_fit_payload = {
                "total": {"x": fit_x, "y": fit_y, "data_y": total},
                "components": components_for_view,
            }
        else:
            y_fit = None
            if hasattr(self.state, "evaluate"):
                try:
                    y_fit = self.state.evaluate()
                except Exception:
                    y_fit = None
            if y_fit is not None:
                arr = np.asarray(y_fit, dtype=float)
                # Apply resolution convolution if enabled
                if self.has_resolution():
                    try:
                        arr = self.evaluate_with_resolution(np.asarray(x, dtype=float), arr)
                    except Exception as e:
                        log_exception("Failed to apply resolution in update_plot (non-composite)", e, vm=self)
                curves_payload["fit"] = (np.asarray(x, dtype=float), arr)
            # Determine fit payload: use convolved array if resolution is active, otherwise original
            if self.has_resolution() and y_fit is not None:
                y_fit_payload = arr
            else:
                y_fit_payload = y_fit

        self.curves = curves_payload
        errs = getattr(self.state, "errors", None)
        errs = None if errs is None else np.asarray(errs, dtype=float)
        self.plot_updated.emit(x, y, y_fit_payload, errs)

    def _build_preview_grid(self, x: np.ndarray) -> np.ndarray:
        """Return a smoother grid for previewing fits/components.
        
        When resolution convolution is active, the grid is extended beyond the
        data range to avoid edge artifacts in the convolution.
        """
        try:
            x_arr = np.asarray(x, dtype=float)
        except Exception:
            return np.array([], dtype=float)

        finite = x_arr[np.isfinite(x_arr)]
        if finite.size < 2:
            return x_arr

        span_min = float(np.min(finite))
        span_max = float(np.max(finite))
        if not np.isfinite(span_min) or not np.isfinite(span_max) or span_max <= span_min:
            return x_arr

        # Extend grid beyond data range if resolution is active to avoid edge effects
        padding = 0.0
        if self.has_resolution():
            # Add padding equal to the resolution range on each side
            padding = DEFAULT_RESOLUTION_RANGE
        
        extended_min = span_min - padding
        extended_max = span_max + padding
        
        # Calculate target samples for the extended range
        data_range = span_max - span_min
        extended_range = extended_max - extended_min
        base_samples = max(400, min(8000, int(finite.size * 4)))
        
        # Scale samples proportionally for extended range
        if data_range > 0:
            target_samples = int(base_samples * extended_range / data_range)
            target_samples = max(base_samples, min(12000, target_samples))
        else:
            target_samples = base_samples
        
        if target_samples <= finite.size:
            return x_arr

        try:
            return np.linspace(extended_min, extended_max, target_samples)
        except Exception:
            return x_arr

    def _trim_to_visible_range(self, x_arr: np.ndarray, y_arr: np.ndarray, 
                                data_min: float, data_max: float) -> _typing.Tuple[np.ndarray, np.ndarray]:
        """Trim preview arrays to visible range (data range plus small margin).
        
        This hides edge artifacts from convolution while keeping previews
        slightly larger than the data for visual appeal.
        
        Args:
            x_arr: X values of preview
            y_arr: Y values of preview
            data_min: Minimum of original data range
            data_max: Maximum of original data range
            
        Returns:
            Tuple of (trimmed_x, trimmed_y)
        """
        try:
            visible_min = data_min - PREVIEW_VISIBLE_MARGIN
            visible_max = data_max + PREVIEW_VISIBLE_MARGIN
            
            mask = (x_arr >= visible_min) & (x_arr <= visible_max)
            return x_arr[mask], y_arr[mask]
        except Exception:
            return x_arr, y_arr

    def _get_plot_fit_data(self) -> _typing.Optional[dict]:
        """Get the current fit data structure for export (similar to update_plot output).
        
        Returns:
            Dict with 'total' and 'components' keys, or None if no fit available
        """
        try:
            x = getattr(self.state, "x_data", None)
            if x is None:
                return None
            
            x_arr = np.asarray(x, dtype=float)
            model_spec = getattr(self.state, "model_spec", None)
            
            preview_grid = self._build_preview_grid(x_arr)
            preview_enabled = preview_grid.size > x_arr.size
            
            if isinstance(model_spec, CompositeModelSpec):
                try:
                    component_outputs = model_spec.evaluate_components(x_arr)
                except Exception:
                    component_outputs = []
                
                preview_lookup = {}
                if preview_enabled:
                    try:
                        preview_outputs = model_spec.evaluate_components(preview_grid)
                        for component, values in preview_outputs:
                            try:
                                preview_lookup[component.prefix] = np.asarray(values, dtype=float)
                            except Exception:
                                pass
                    except Exception:
                        preview_lookup = {}
                        preview_enabled = False
                
                total = np.zeros_like(x_arr, dtype=float)
                total_preview = np.zeros_like(preview_grid, dtype=float) if preview_enabled else None
                preview_complete = bool(preview_enabled)
                components_for_export = []
                
                for component, values in component_outputs:
                    arr = np.asarray(values, dtype=float)
                    if arr.shape != x_arr.shape:
                        continue
                    try:
                        total += arr
                    except Exception:
                        pass
                    
                    disp_x = x_arr
                    disp_y = arr
                    if preview_enabled:
                        preview_arr = np.asarray(preview_lookup.get(component.prefix, np.array([])), dtype=float)
                        if preview_arr.size == preview_grid.size:
                            disp_x = preview_grid
                            disp_y = preview_arr
                            try:
                                if total_preview is not None:
                                    total_preview += preview_arr
                            except Exception:
                                preview_complete = False
                        else:
                            preview_complete = False
                    
                    components_for_export.append({
                        "prefix": component.prefix,
                        "label": component.label,
                        "color": component.color,
                        "x": np.asarray(disp_x, dtype=float),
                        "y": np.asarray(disp_y, dtype=float),
                    })
                
                if preview_complete and total_preview is not None:
                    fit_x = np.asarray(preview_grid, dtype=float)
                    fit_y = np.asarray(total_preview, dtype=float)
                else:
                    fit_x = np.asarray(x_arr, dtype=float)
                    fit_y = np.asarray(total, dtype=float)
                
                # Apply resolution if enabled
                if self.has_resolution():
                    try:
                        fit_y = self.evaluate_with_resolution(fit_x, fit_y)
                        for comp_data in components_for_export:
                            comp_y = comp_data.get("y")
                            comp_x = comp_data.get("x")
                            if comp_y is not None and comp_x is not None:
                                comp_data["y"] = self.evaluate_with_resolution(comp_x, comp_y)
                        total = self.evaluate_with_resolution(x_arr, total)
                        
                        # Trim to visible range
                        data_min, data_max = float(np.min(x_arr)), float(np.max(x_arr))
                        fit_x, fit_y = self._trim_to_visible_range(fit_x, fit_y, data_min, data_max)
                        for comp_data in components_for_export:
                            comp_x = comp_data.get("x")
                            comp_y = comp_data.get("y")
                            if comp_x is not None and comp_y is not None:
                                comp_data["x"], comp_data["y"] = self._trim_to_visible_range(
                                    comp_x, comp_y, data_min, data_max
                                )
                    except Exception:
                        pass
                
                return {
                    "total": {"x": fit_x, "y": fit_y, "data_y": total},
                    "components": components_for_export,
                }
            else:
                # Non-composite model
                y_fit = None
                if hasattr(self.state, "evaluate"):
                    try:
                        y_fit = self.state.evaluate()
                    except Exception:
                        pass
                
                if y_fit is not None:
                    arr = np.asarray(y_fit, dtype=float)
                    if self.has_resolution():
                        try:
                            arr = self.evaluate_with_resolution(x_arr, arr)
                        except Exception:
                            pass
                    return arr
                
                return None
        except Exception:
            return None

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
    # Fit Persistence
    # --------------------------
    def _get_current_file_path(self) -> _typing.Optional[str]:
        """Get the file path of the currently loaded data file, if any."""
        try:
            file_info = getattr(self.state, "file_info", None)
            if isinstance(file_info, dict):
                return file_info.get("path")
        except Exception as e:
            self.log_message.emit(f"Failed to get current file path: {e}")
        return None

    def _get_default_model_choice(self) -> _typing.Optional[str]:
        """Get the default model choice from config."""
        try:
            from dataio import get_config
            cfg = get_config()
            default_model = getattr(cfg, "default_model_for_new_files", None)
            # Safer default: don't apply the last fit/model automatically.
            # Users can opt into "(Use Last Fit)" via the UI selector.
            return default_model if default_model else "(None - User Select)"
        except Exception:
            return "(None - User Select)"

    def set_default_model(self, model_name: str):
        """Set the default model for newly loaded files without saved fits.
        
        Args:
            model_name: The model name, or special values:
                       "(Use Last Fit)" - use saved default fit
                       "(None - User Select)" - no auto-load
        """
        try:
            from dataio import get_config
            cfg = get_config()
            cfg.default_model_for_new_files = model_name
            try:
                cfg.save()
            except Exception:
                pass
            self._log_message(f"Default model set to: {model_name}")
        except Exception as e:
            log_exception("Failed to save default model choice", e, vm=self)

    def _save_current_fit(self):
        """Save the current fit state to both default and file-specific locations."""
        try:
            from dataio import save_default_fit, save_fit_for_file

            filepath = self._get_current_file_path()
            resolution_state = self.get_resolution_state()

            if filepath:
                # When working with a real data file, keep the default fit untouched
                try:
                    if save_fit_for_file(self.state, filepath, resolution_state):
                        self._log_message(f"Saved fit for: {os.path.basename(filepath)}")
                except Exception as e:
                    self._log_message(f"Failed to save fit for file: {e}")
            else:
                # Only update the default fit when no external file is active
                try:
                    if save_default_fit(self.state, resolution_state):
                        self._log_message("Saved default fit.")
                except Exception as e:
                    self._log_message(f"Failed to save default fit: {e}")
        except Exception as e:
            log_exception("Failed to save fit", e, vm=self)

    def _schedule_fit_save(self):
        """Schedule a debounced save of the current fit state.
        
        This restarts the timer so that rapid changes only trigger one save
        after the user stops making changes (500ms debounce).
        """
        try:
            self._fit_save_timer.stop()
            self._fit_save_timer.start()
        except Exception:
            # If timer doesn't work, save immediately
            try:
                self._save_current_fit()
            except Exception as e:
                log_exception("Failed to save fit in fallback", e, vm=self)

    def _load_fit_for_current_file(self) -> bool:
        """Try to load a saved fit for the currently loaded file.

        Returns True if a file-specific fit was loaded, False otherwise.
        """
        try:
            from dataio import load_fit_for_file, has_fit_for_file

            filepath = self._get_current_file_path()
            if not filepath:
                return False

            if not has_fit_for_file(filepath):
                return False

            result = load_fit_for_file(self.state, filepath, apply_excluded=True)
            # Handle both old (bool) and new (tuple) return formats
            if isinstance(result, tuple):
                success, resolution_state = result
            else:
                success, resolution_state = result, None
            
            if success:
                self._log_message(f"Restored fit for: {os.path.basename(filepath)}")
                # Restore resolution state if present
                if resolution_state:
                    self.set_resolution_state(resolution_state)
                return True
        except Exception as e:
            log_exception("Failed to load fit for file", e, vm=self)
        return False

    def _load_default_fit(self) -> bool:
        """Try to load the default fit for the current data.

        Returns True if the default fit was loaded, False otherwise.
        """
        try:
            # Check what the default model selector is set to
            default_model_choice = self._get_default_model_choice()
            
            if default_model_choice == "(None - User Select)":
                # User doesn't want auto-loading
                return False
            elif default_model_choice and default_model_choice not in ("(Use Last Fit)", "(None - User Select)"):
                # User selected a specific model as default
                try:
                    self.set_model(default_model_choice)
                    self._log_message(f"Loaded default model: {default_model_choice}")
                    return True
                except Exception as e:
                    log_exception(f"Failed to load default model '{default_model_choice}'", e, vm=self)
                    # Fall through to try loading saved fit
            
            # Default behavior: try to load the saved default fit
            from dataio import load_default_fit

            result = load_default_fit(self.state, apply_excluded=False)
            # Handle both old (bool) and new (tuple) return formats
            if isinstance(result, tuple):
                success, resolution_state = result
            else:
                success, resolution_state = result, None
            
            if success:
                self._log_message("Loaded default fit parameters.")
                # Restore resolution state if present
                if resolution_state:
                    self.set_resolution_state(resolution_state)
                
                # Debug: Log what was loaded
                model_name = getattr(self.state, "model_name", "Unknown")
                model_spec = getattr(self.state, "model_spec", None)
                if isinstance(model_spec, CompositeModelSpec):
                    num_components = len(model_spec.list_components()) if hasattr(model_spec, "list_components") else 0
                    self._log_message(f"Loaded Custom Model with {num_components} component(s)")
                else:
                    self._log_message(f"Loaded model: {model_name}")
                
                return True
        except Exception as e:
            log_exception("Failed to load default fit", e, vm=self)
        return False

    def reset_fit(self):
        """Reset the fit for the currently active dataset.

        Deletes any file-specific saved fit for the active file (if any), clears
        fit_result and exclusions, and reinitializes the current model parameters.
        The global default fit is only reset when no real file is active.
        """
        try:
            from dataio import reset_fit_for_file, reset_default_fit

            # Reset file-specific fit if applicable
            filepath = self._get_current_file_path()
            if filepath:
                try:
                    if reset_fit_for_file(filepath):
                        self._log_message(f"Cleared saved fit for: {os.path.basename(filepath)}")
                except Exception as e:
                    log_exception(f"Failed to clear saved fit for file: {filepath}", e, vm=self)
            else:
                # Only reset the default fit when no external file is active.
                try:
                    if reset_default_fit():
                        self._log_message("Cleared default fit.")
                except Exception:
                    pass

            # Reinitialize model spec with current data
            model_spec = getattr(self.state, "model_spec", None)
            if model_spec is not None:
                try:
                    model_spec.initialize(
                        getattr(self.state, "x_data", None),
                        getattr(self.state, "y_data", None)
                    )
                except Exception as e:
                    log_exception("Error during model_spec.initialize in reset_fit", e, vm=self)

            # Clear fit result
            try:
                self.state.fit_result = None
            except Exception:
                pass

            # Clear exclusions
            try:
                xd = getattr(self.state, "x_data", None)
                if xd is not None:
                    self.state.excluded = np.zeros_like(np.asarray(xd), dtype=bool)
            except Exception:
                pass

            # Clear resolution for this dataset
            try:
                self._resolution_model_name = "None"
                self._resolution_spec = None
                try:
                    self.resolution_updated.emit()
                except Exception:
                    pass
            except Exception:
                pass

            # Notify views and update plot
            try:
                self.parameters_updated.emit()
            except Exception:
                pass
            try:
                self.update_plot()
            except Exception:
                pass

            self._log_message("Fit has been reset.")
        except Exception as e:
            log_exception("Failed to reset fit", e, vm=self)

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
            
            # Check if this is a saved custom model first
            is_saved_custom = self._is_saved_custom_model(model_name)
            
            if is_saved_custom:
                # For saved custom models, set up Custom Model spec FIRST
                # This ensures any immediate get_parameters() calls work correctly
                spec = get_model_spec("Custom Model")
                setattr(self.state, "model_spec", spec)
                setattr(self.state, "model_name", "Custom Model")
                self.log_message.emit(f"Model switched to: Custom Model")
                
                # Now load the components from the saved model
                # Pass emit_signals=False to avoid duplicate signal emissions
                success = self.load_saved_custom_model(model_name, emit_signals=False)
                if success:
                    # Notify View to refresh parameters/plot
                    try:
                        self.parameters_updated.emit()
                    except Exception:
                        pass
                    try:
                        self.update_plot()
                    except Exception:
                        pass
                    # Immediately persist the new model/fit state for the active file
                    try:
                        self._save_current_fit()
                    except Exception:
                        try:
                            self.log_message.emit("Failed to auto-save fit after model change.")
                        except Exception:
                            pass
                    return
                # If loading failed, Custom Model is still active but empty
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
            # Persist the new model/fit immediately when switching models
            try:
                self._save_current_fit()
            except Exception:
                try:
                    self.log_message.emit("Failed to auto-save fit after model change.")
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
                    "tooltip": component.spec_label,
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
        # Save fit immediately when component structure changes
        try:
            self._save_current_fit()
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
        # Save fit immediately when component structure changes
        try:
            self._save_current_fit()
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
        # Save fit immediately when component structure changes
        try:
            self._save_current_fit()
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
        # Save fit immediately when component structure changes
        try:
            self._save_current_fit()
        except Exception:
            pass
        return True

    # --------------------------
    # Save custom model
    # --------------------------
    def get_model_data_for_save(self) -> _typing.Optional[_typing.Dict[str, _typing.Any]]:
        """
        Extract current composite model structure for saving to YAML.
        
        Returns a dictionary containing:
        - name: suggested model name
        - components: list of component dictionaries with parameters
        - default_save_path: default location to save the model
        
        Returns None if current model is not a composite model.
        """
        spec = getattr(self.state, "model_spec", None)
        if not isinstance(spec, CompositeModelSpec):
            self.log_message.emit("Only custom composite models can be saved.")
            return None
        
        components_list = []
        try:
            for component in spec.list_components():
                comp_data = {
                    'type': component.spec.__class__.__name__.replace("ModelSpec", ""),
                    'prefix': component.prefix,
                    'label': component.label,
                    'color': component.color,
                    'parameters': {}
                }
                
                # Get parameters for this component
                for name, param in component.spec.params.items():
                    param_data = {
                        'value': getattr(param, 'value', None),
                        'fixed': bool(getattr(param, 'fixed', False)),
                        'link_group': getattr(param, 'link_group', None),
                        'min': getattr(param, 'min', None),
                        'max': getattr(param, 'max', None),
                        'type': getattr(param, 'ptype', 'float'),
                        'decimals': getattr(param, 'decimals', None),
                        'step': getattr(param, 'step', None),
                        'hint': getattr(param, 'hint', ''),
                    }
                    comp_data['parameters'][name] = param_data
                
                components_list.append(comp_data)
        except Exception as exc:
            self.log_message.emit(f"Error extracting model data: {exc}")
            return None
        
        # Get default save path (models/custom_models/)
        from pathlib import Path
        try:
            repo_root = Path(__file__).resolve().parent.parent
            default_path = repo_root / "models" / "custom_models"
            # Ensure the directory exists
            default_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Fallback to current working directory if we can't determine repo root
            try:
                default_path = Path.cwd() / "models" / "custom_models"
                default_path.mkdir(parents=True, exist_ok=True)
            except Exception:
                default_path = Path.cwd()
        
        return {
            'name': 'CustomModel',
            'components': components_list,
            'default_save_path': str(default_path)
        }

    def save_custom_model_to_yaml(self, filepath: str, model_name: str, 
                                   description: str = "") -> bool:
        """
        Save the current composite model to a YAML file.
        
        Args:
            filepath: Full path where to save the YAML file
            model_name: Display name for the model
            description: Optional description of the model
            
        Returns:
            True if save succeeded, False otherwise
        """
        from pathlib import Path
        import yaml
        
        model_data = self.get_model_data_for_save()
        if model_data is None:
            return False

        try:
            components = model_data.get('components', [])
            
            # Build YAML structure for saved custom model
            # Note: This is NOT a model element, but a saved configuration
            # It will be loaded by switching to Custom Model and adding components
            yaml_data = {
                'name': model_name,
                'description': description or f"Saved custom model: {model_name}",
                'version': 1,
                'author': 'BigFit User',
                'category': 'saved_custom_model',  # Mark as saved custom model, not element
                'components': []
            }
            
            # Add each component as a sub-model
            for comp_idx, comp in enumerate(components):
                comp_type = comp.get('type', 'Unknown')
                prefix = comp.get('prefix', f'elem{comp_idx + 1}_')
                
                component_def = {
                    'element': comp_type,
                    'prefix': prefix,
                    'default_parameters': {}
                }
                
                # Add parameters with their default values and metadata
                params = comp.get('parameters', {})
                for param_name, param_info in params.items():
                    param_def = {
                        'value': param_info.get('value'),
                        'type': param_info.get('type', 'float'),
                    }
                    
                    # Add optional metadata
                    if param_info.get('fixed'):
                        param_def['fixed'] = True
                    
                    link_group = param_info.get('link_group')
                    if link_group is not None and link_group != 0:
                        param_def['link_group'] = link_group
                    
                    if param_info.get('min') is not None:
                        param_def['min'] = param_info['min']
                    if param_info.get('max') is not None:
                        param_def['max'] = param_info['max']
                    
                    if param_info.get('decimals') is not None:
                        param_def['decimals'] = param_info['decimals']
                    if param_info.get('step') is not None:
                        param_def['step'] = param_info['step']
                    if param_info.get('hint'):
                        param_def['hint'] = param_info['hint']
                    
                    component_def['default_parameters'][param_name] = param_def
                
                yaml_data['components'].append(component_def)
            
            # Write to file
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, 
                         allow_unicode=True, indent=2)
            
            self.log_message.emit(f"Model saved to: {filepath}")
            self.log_message.emit(f"Model '{model_name}' can be loaded from the 'Load Custom Model' menu.")
            
            return True
            
        except Exception as exc:
            self.log_message.emit(f"Failed to save model: {exc}")
            import traceback
            traceback.print_exc()
            return False

    def _scan_saved_custom_models(self) -> _typing.Dict[str, object]:
        """Scan the models/custom_models directory for saved custom-model YAML files.

        Returns a mapping of saved-model display name -> pathlib.Path to the YAML file.
        Only files with 'category' == 'saved_custom_model' are included. If no
        directory exists or an error occurs, an empty dict is returned.
        """
        try:
            from pathlib import Path
            import yaml

            repo_root = Path(__file__).resolve().parent.parent
            custom_models_dir = repo_root / "models" / "custom_models"
            if not custom_models_dir.exists():
                return {}

            found = {}
            for yaml_file in custom_models_dir.glob("*.yaml"):
                try:
                    with open(yaml_file, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    if data and data.get('category') == 'saved_custom_model':
                        name = data.get('name', yaml_file.stem)
                        found[name] = yaml_file
                except Exception:
                    # Skip unreadable/invalid files
                    continue
            return found
        except Exception:
            return {}

    def _is_saved_custom_model(self, model_name: str) -> bool:
        """Check if a model name corresponds to a saved custom model."""
        try:
            models = self._scan_saved_custom_models()
            return model_name in models
        except Exception:
            return False

    def list_saved_custom_models(self) -> _typing.List[str]:
        """
        List all available saved custom models.
        
        Returns:
            List of model names (without .yaml extension)
        """
        try:
            models = list(self._scan_saved_custom_models().keys())
            return sorted(models)
        except Exception:
            return []

    def load_saved_custom_model(self, model_name: str, emit_signals: bool = True) -> bool:
        """
        Load a saved custom model by switching to Custom Model and adding its components.
        
        Args:
            model_name: Name of the saved model to load
            emit_signals: Whether to emit parameters_updated and update_plot signals (default: True)
            
        Returns:
            True if successful, False otherwise
        """
        import yaml

        try:
            models_map = self._scan_saved_custom_models()
            yaml_file = models_map.get(model_name)
            if yaml_file is None:
                self.log_message.emit(f"Could not find saved model: {model_name}")
                return False

            # Load the model data
            with open(str(yaml_file), 'r', encoding='utf-8') as f:
                model_data = yaml.safe_load(f)
            
            # Get the Custom Model spec - either from state (if set_model already created it)
            # or create it now (if called directly via Load Custom Model button)
            spec = getattr(self.state, "model_spec", None)
            if not isinstance(spec, CompositeModelSpec):
                from models import get_model_spec
                spec = get_model_spec("Custom Model")
                if not isinstance(spec, CompositeModelSpec):
                    self.log_message.emit("Failed to create Custom Model spec")
                    return False
                
                # Set the spec on state
                setattr(self.state, "model_spec", spec)
                setattr(self.state, "model_name", "Custom Model")
            
            # Clear any existing components
            spec.clear_components()
            
            # Add components from saved model
            components_data = model_data.get('components', [])
            if not components_data:
                self.log_message.emit(f"Warning: No components found in saved model '{model_name}'")
                return False
                
            components_added = 0
            for comp_data in components_data:
                element_type = comp_data.get('element')
                prefix = comp_data.get('prefix')
                default_params = comp_data.get('default_parameters', {})
                
                if not element_type:
                    continue
                
                try:
                    # Add the component
                    component = spec.add_component(
                        element_type,
                        prefix=prefix,
                        data_x=getattr(self.state, "x_data", None),
                        data_y=getattr(self.state, "y_data", None),
                    )
                    
                    if component:
                        components_added += 1
                        # Apply saved parameter values and properties
                        for param_name, param_data in default_params.items():
                            if param_name in component.spec.params:
                                param_obj = component.spec.params[param_name]
                                
                                # Apply value
                                if isinstance(param_data, dict):
                                    value = param_data.get('value')
                                    if value is not None:
                                        param_obj.value = value
                                    
                                    # Apply fixed state
                                    if param_data.get('fixed'):
                                        param_obj.fixed = True
                                    
                                    # Apply link group
                                    link_group = param_data.get('link_group')
                                    if link_group is not None and link_group != 0:
                                        param_obj.link_group = link_group
                                    
                                    # Apply bounds
                                    min_val = param_data.get('min')
                                    if min_val is not None:
                                        param_obj.min = min_val
                                    max_val = param_data.get('max')
                                    if max_val is not None:
                                        param_obj.max = max_val
                                else:
                                    # Simple value
                                    param_obj.value = param_data
                        
                        # Rebuild flat params to propagate changes
                        spec._rebuild_flat_params()
                        
                except Exception as e:
                    self.log_message.emit(f"Warning: Could not add component '{element_type}': {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if components_added == 0:
                self.log_message.emit(f"Error: No components could be added from saved model '{model_name}'")
                return False
            
            # Save the loaded model as fit for current data
            # (Do this before emitting signals so the fit is saved with the right state)
            try:
                self._save_current_fit()
            except Exception:
                pass
            
            self.log_message.emit(f"Loaded custom model: {model_name}")
            
            # Notify UI to refresh - only if caller requests it
            if emit_signals:
                try:
                    self.parameters_updated.emit()
                except Exception:
                    pass
                try:
                    self.update_plot()
                except Exception:
                    pass
            
            return True
            
        except Exception as exc:
            self.log_message.emit(f"Failed to load custom model '{model_name}': {exc}")
            import traceback
            traceback.print_exc()
            return False

    def _is_parameter_fixed(self, name: _typing.Optional[str]) -> bool:
        """Return True if the given parameter is flagged as fixed in the current spec."""
        if not name or not isinstance(name, str):
            return False

        spec = getattr(self.state, "model_spec", None)
        params = {}
        if spec is not None:
            try:
                params = getattr(spec, "params", {}) or {}
            except Exception:
                params = {}

        param = params.get(name)
        if param is not None:
            try:
                if bool(getattr(param, "fixed", False)):
                    return True
            except Exception:
                pass

        if isinstance(spec, CompositeModelSpec) and hasattr(spec, "get_link"):
            try:
                link = spec.get_link(name)
            except Exception:
                link = None
            if link and isinstance(link, tuple) and len(link) >= 2:
                component, base_name = link
                try:
                    comp_params = getattr(component.spec, "params", {}) if hasattr(component, "spec") else {}
                except Exception:
                    comp_params = {}
                comp_param = comp_params.get(base_name)
                if comp_param is not None:
                    try:
                        if bool(getattr(comp_param, "fixed", False)):
                            return True
                    except Exception:
                        pass

        mdl = getattr(self.state, "model", None)
        if mdl is not None:
            try:
                if bool(getattr(mdl, f"{name}__fixed", False)):
                    return True
            except Exception:
                pass

        return False

    def _collect_link_groups(self, model_spec) -> dict:
        groups = {}
        if model_spec is None:
            return groups
        try:
            params = getattr(model_spec, "params", {}) or {}
        except Exception:
            return groups

        for pname, param in params.items():
            try:
                lg = getattr(param, "link_group", None)
            except Exception:
                lg = None
            if lg and lg > 0:
                groups.setdefault(lg, []).append(pname)
        return groups

    def _set_param_fixed_state(self, model_spec, model_obj, name: str, fixed_value: bool):
        if not name:
            return
        try:
            if model_spec is not None and name in getattr(model_spec, "params", {}):
                try:
                    model_spec.params[name].fixed = bool(fixed_value)
                except Exception:
                    pass
                if isinstance(model_spec, CompositeModelSpec) and hasattr(model_spec, "get_link"):
                    try:
                        link = model_spec.get_link(name)
                    except Exception:
                        link = None
                    if link and isinstance(link, tuple) and len(link) >= 2:
                        component, pname = link
                        try:
                            comp_params = getattr(component.spec, "params", {}) if hasattr(component, "spec") else {}
                        except Exception:
                            comp_params = {}
                        if pname in comp_params:
                            try:
                                comp_params[pname].fixed = bool(fixed_value)
                            except Exception:
                                pass
        except Exception:
            pass

        if model_obj is not None:
            try:
                setattr(model_obj, f"{name}__fixed", bool(fixed_value))
            except Exception:
                pass

    def _apply_fixed_state_to_group(self, model_spec, model_obj, base: str, fixed_value: bool, link_groups: dict):
        targets = set()
        if base:
            targets.add(base)
        try:
            params = getattr(model_spec, "params", {}) if model_spec is not None else {}
            lg = None
            if base in params:
                lg = getattr(params[base], "link_group", None)
        except Exception:
            lg = None
        if lg and lg in link_groups:
            for name in link_groups[lg]:
                targets.add(name)

        for name in targets:
            self._set_param_fixed_state(model_spec, model_obj, name, fixed_value)

    def _get_current_param_value(self, name: str):
        mdl = getattr(self.state, "model", None)
        if mdl is not None and hasattr(mdl, name):
            try:
                return getattr(mdl, name)
            except Exception:
                pass
        spec = getattr(self.state, "model_spec", None)
        if spec is not None and name in getattr(spec, "params", {}):
            try:
                return spec.params[name].value
            except Exception:
                pass
        return None

    def _values_close(self, a, b) -> bool:
        if a is None or b is None:
            return a is b
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
        except Exception:
            return a == b

    def _split_parameter_updates(self, params: dict):
        """Split parameter dict into fixed, link, and value updates.
        
        Returns:
            Tuple of (fixed_updates, link_updates, value_updates) dicts
        """
        fixed_suffix = "__fixed"
        link_suffix = "__link"
        fixed_updates = {}
        link_updates = {}
        value_updates = {}
        
        for k, v in params.items():
            if isinstance(k, str) and k.endswith(fixed_suffix):
                base = k[:-len(fixed_suffix)]
                fixed_updates[base] = bool(v)
            elif isinstance(k, str) and k.endswith(link_suffix):
                base = k[:-len(link_suffix)]
                try:
                    link_updates[base] = int(v) if v else None
                except (ValueError, TypeError):
                    link_updates[base] = None
            else:
                value_updates[k] = v
        
        return fixed_updates, link_updates, value_updates
    
    def _ensure_model_and_spec(self):
        """Ensure state has both model and model_spec, creating them if needed.
        
        Returns:
            Tuple of (model, model_spec, is_composite)
        """
        mdl = getattr(self.state, "model", None)
        model_spec = getattr(self.state, "model_spec", None)

        # Ensure we have a model_spec
        if model_spec is None:
            model_name = getattr(self.state, "model_name", None)
            if model_name is None and mdl is not None:
                model_name = getattr(mdl, "name", None) or mdl.__class__.__name__
            if model_name is None:
                model_name = "voigt"
            try:
                from models import get_model_spec
                model_spec = get_model_spec(model_name)
                setattr(self.state, "model_spec", model_spec)
            except Exception:
                model_spec = None

        # Ensure we have a model object
        if mdl is None:
            mdl = SimpleNamespace()
            setattr(self.state, "model", mdl)

        is_composite = isinstance(model_spec, CompositeModelSpec)
        return mdl, model_spec, is_composite

    def apply_parameters(self, params: dict):
        """Apply parameters from the UI.

        - params: dict mapping parameter-name -> value.
        The function will set attributes on state.model if present, otherwise update
        the attached state.model_spec parameter values. Finally triggers update_plot().
        """
        if not isinstance(params, dict):
            try:
                params = dict(params)
            except Exception:
                self.log_message.emit("apply_parameters: expected a dict")
                return

        # Ensure model and spec exist
        mdl, model_spec, is_composite = self._ensure_model_and_spec()
        
        # Split parameters by type
        fixed_updates, link_updates, value_updates = self._split_parameter_updates(params)


        applied = []
        blocked_values = []
        
        # Apply link_group updates
        for base, link_val in link_updates.items():
            if model_spec is not None and base in getattr(model_spec, 'params', {}):
                try:
                    model_spec.params[base].link_group = link_val
                    # Propagate to composite component if applicable
                    if isinstance(model_spec, CompositeModelSpec) and hasattr(model_spec, 'get_link'):
                        link = model_spec.get_link(base)
                        if link and isinstance(link, tuple) and len(link) >= 2:
                            component, pname = link
                            if hasattr(component, 'spec') and pname in getattr(component.spec, 'params', {}):
                                component.spec.params[pname].link_group = link_val
                    applied.append(f"{base}__link")
                except Exception:
                    pass
        
        # Refresh link groups after updates
        link_groups = self._collect_link_groups(model_spec)
        
        # Apply fixed-state updates
        for base, fv in fixed_updates.items():
            try:
                self._apply_fixed_state_to_group(model_spec, mdl, base, bool(fv), link_groups)
                applied.append(f"{base}__fixed")
            except Exception:
                pass

        # Apply value updates (regular parameter assignments)
        for k, v in value_updates.items():
            # Skip if parameter is fixed
            if self._is_parameter_fixed(k):
                current_val = self._get_current_param_value(k)
                if not self._values_close(current_val, v):
                    blocked_values.append(k)
                continue
            
            # Set parameter value on model and spec
            try:
                setattr(mdl, k, v)
                if is_composite and hasattr(model_spec, "set_param_value"):
                    model_spec.set_param_value(k, v)
                elif model_spec and k in getattr(model_spec, 'params', {}):
                    model_spec.params[k].value = v
                applied.append(k)
            except Exception:
                pass
            
            # Propagate value to linked parameters
            if model_spec and k in getattr(model_spec, 'params', {}):
                try:
                    lg = getattr(model_spec.params[k], "link_group", None)
                    if lg and lg > 0 and lg in link_groups:
                        for linked_name in link_groups[lg]:
                            if linked_name != k:
                                try:
                                    setattr(mdl, linked_name, v)
                                    if is_composite and hasattr(model_spec, "set_param_value"):
                                        model_spec.set_param_value(linked_name, v)
                                    elif linked_name in model_spec.params:
                                        model_spec.params[linked_name].value = v
                                    if linked_name not in applied:
                                        applied.append(linked_name)
                                except Exception:
                                    pass
                except Exception:
                    pass

        if blocked_values:
            key = tuple(sorted(blocked_values))
            if getattr(self, "_last_blocked_value_update", None) != key:
                self._last_blocked_value_update = key
                try:
                    if len(blocked_values) == 1:
                        msg = f"Skipped update: '{blocked_values[0]}' is fixed."
                    else:
                        names = ", ".join(blocked_values)
                        msg = f"Skipped updates; fixed parameters: {names}."
                    self.log_message.emit(msg)
                except Exception:
                    pass
        else:
            self._last_blocked_value_update = None

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
        # Schedule debounced save of fit state
        try:
            self._schedule_fit_save()
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

        free_updates = {}
        blocked = []
        for pname, pval in updates.items():
            if self._is_parameter_fixed(pname):
                blocked.append(pname)
            else:
                free_updates[pname] = pval

        if not free_updates:
            if blocked:
                blocked_key = tuple(sorted(blocked))
                if getattr(self, "_last_blocked_drag", None) != blocked_key:
                    self._last_blocked_drag = blocked_key
                    try:
                        if len(blocked) == 1:
                            msg = f"Ignored peak drag: '{blocked[0]}' is fixed."
                        else:
                            msg = "Ignored peak drag: all targeted centers are fixed."
                        self.log_message.emit(msg)
                    except Exception:
                        pass
            return

        self._last_blocked_drag = None
        updates = free_updates

        try:
            self.apply_parameters(updates)
            target = ", ".join(updates.keys())
            self.log_message.emit(f"Peak center updated -> {val} ({target})")
        except Exception:
            pass

    # --------------------------
    # Resolution model management
    # --------------------------
    def set_resolution_model(self, model_name: str):
        """Set the resolution model to use for convolution.
        
        Args:
            model_name: Name of the model to use, or "None" for no resolution
        """
        try:
            model_name = (model_name or "").strip()
            if not model_name or model_name.lower() == "none":
                self._resolution_model_name = "None"
                self._resolution_spec = None
                self._log_message("Resolution convolution disabled.")
            else:
                # Create a model spec for the resolution
                spec = get_model_spec(model_name)
                if spec is None:
                    self._log_message(f"Failed to create resolution model: {model_name}")
                    return
                
                # Initialize the resolution spec centered at 0 (since it's a kernel)
                try:
                    # Create a symmetric x range for initialization
                    x_init = np.linspace(-10, 10, 100)
                    y_init = np.ones_like(x_init)
                    spec.initialize(x_init, y_init)
                except Exception:
                    pass
                
                # Set default center to 0 for resolution functions
                try:
                    if hasattr(spec, "params") and "Center" in spec.params:
                        spec.params["Center"].value = 0.0
                except Exception:
                    pass
                
                self._resolution_model_name = model_name
                self._resolution_spec = spec
                self._log_message(f"Resolution model set to: {model_name}")
            
            try:
                self.resolution_updated.emit()
            except Exception:
                pass
            
            # Schedule fit save when resolution model changes
            try:
                self._schedule_fit_save()
            except Exception:
                pass
        except Exception as e:
            log_exception(f"Failed to set resolution model '{model_name}'", e, vm=self)

    def get_resolution_model_name(self) -> str:
        """Get the current resolution model name."""
        return self._resolution_model_name or "None"

    def get_resolution_parameters(self) -> dict:
        """Get the resolution model parameters for the UI.
        
        Returns:
            Dict of parameter specs (same format as get_parameters())
        """
        if self._resolution_spec is None:
            return {}
        
        try:
            if hasattr(self._resolution_spec, "get_parameters"):
                return self._resolution_spec.get_parameters()
            elif hasattr(self._resolution_spec, "params"):
                return {name: p.to_spec() for name, p in self._resolution_spec.params.items()}
        except Exception as e:
            log_exception("Failed to get resolution parameters", e, vm=self)
        return {}

    def apply_resolution_parameters(self, params: dict):
        """Apply parameter updates to the resolution model.
        
        Args:
            params: Dict mapping parameter names to values
        """
        if self._resolution_spec is None or not params:
            return
        
        try:
            fixed_suffix = "__fixed"
            link_suffix = "__link"
            
            for k, v in params.items():
                try:
                    if isinstance(k, str) and k.endswith(fixed_suffix):
                        base = k[: -len(fixed_suffix)]
                        if base in self._resolution_spec.params:
                            self._resolution_spec.params[base].fixed = bool(v)
                    elif isinstance(k, str) and k.endswith(link_suffix):
                        base = k[: -len(link_suffix)]
                        if base in self._resolution_spec.params:
                            try:
                                self._resolution_spec.params[base].link_group = int(v) if v else None
                            except Exception:
                                self._resolution_spec.params[base].link_group = None
                    else:
                        if k in self._resolution_spec.params:
                            self._resolution_spec.params[k].value = v
                except Exception:
                    pass
            
            try:
                self.resolution_updated.emit()
            except Exception:
                pass
            
            # Schedule fit save when resolution parameters change
            try:
                self._schedule_fit_save()
            except Exception:
                pass
        except Exception as e:
            log_exception("Failed to apply resolution parameters", e, vm=self)

    def get_resolution_preview(self) -> _typing.Tuple[np.ndarray, np.ndarray]:
        """Get the resolution function preview data.
        
        Returns:
            Tuple of (x_data, y_data) for the preview plot
        """
        if self._resolution_spec is None:
            return np.array([]), np.array([])
        
        try:
            # Create a symmetric grid centered at 0
            x = np.linspace(-RESOLUTION_PREVIEW_RANGE, RESOLUTION_PREVIEW_RANGE, 200)
            
            # Evaluate the resolution function
            if hasattr(self._resolution_spec, "evaluate"):
                y = self._resolution_spec.evaluate(x)
            else:
                y = np.zeros_like(x)
            
            # Normalize so peak is at 1 for display
            y_max = np.max(np.abs(y))
            if y_max > 0:
                y = y / y_max
            
            return x, y
        except Exception as e:
            log_exception("Failed to generate resolution preview", e, vm=self)
            return np.array([]), np.array([])

    def evaluate_with_resolution(self, x: np.ndarray, y_model: np.ndarray) -> np.ndarray:
        """Apply resolution convolution to a model output.
        
        The convolution is performed on a uniform grid centered at 0 with proper
        padding to avoid edge effects. The result is interpolated back to the
        original x values. This approach matches `convolute_voigt_dho()`.
        
        Args:
            x: X values (used to determine spacing for convolution)
            y_model: Model output to convolve
            
        Returns:
            Convolved model output, or original if no resolution set
        """
        if self._resolution_spec is None:
            return y_model
        
        try:
            from scipy.signal import fftconvolve
            from scipy.interpolate import interp1d
            
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y_model, dtype=float)
            
            if x_arr.size < 2 or y_arr.size != x_arr.size:
                return y_model
            
            # Use a fixed step size for the uniform grid (like convolute_voigt_dho)
            dx = 0.05
            pad_range = DEFAULT_RESOLUTION_RANGE
            
            # Create a uniform grid centered at 0 that spans the data range plus padding
            x_min, x_max = np.min(x_arr), np.max(x_arr)
            x_span = max(abs(x_min), abs(x_max)) + pad_range
            x_uniform = np.arange(-x_span, x_span, dx)
            
            # Interpolate the input signal onto the uniform grid
            signal_interp = interp1d(x_arr, y_arr, kind='linear',
                                     bounds_error=False, fill_value=0.0)
            signal_uniform = signal_interp(x_uniform)
            
            # Evaluate the resolution function on the uniform grid (centered at 0)
            try:
                res_y = self._resolution_spec.evaluate(x_uniform)
            except Exception:
                return y_model
            
            # Normalize the resolution function (area = 1 for proper convolution)
            area = np.trapz(res_y, x_uniform)
            if abs(area) > 1e-12:
                res_y = res_y / area
            
            # Perform convolution on the uniform grid
            convolved = fftconvolve(signal_uniform, res_y, mode='same') * dx
            
            # Interpolate the convolved result back to original x values
            result_interp = interp1d(x_uniform, convolved, kind='linear',
                                     bounds_error=False, fill_value=0.0)
            result = result_interp(x_arr)
            
            return result
        except Exception as e:
            log_exception("Failed to apply resolution convolution", e, vm=self)
            return y_model

    def has_resolution(self) -> bool:
        """Check if a resolution model is active."""
        return self._resolution_spec is not None

    def get_resolution_state(self) -> _typing.Optional[dict]:
        """Get the current resolution state for persistence.
        
        Returns:
            Dict containing resolution model name and parameters, or None if no resolution
        """
        if self._resolution_spec is None or self._resolution_model_name == "None":
            return None
        
        try:
            params = {}
            spec_params = getattr(self._resolution_spec, "params", {}) or {}
            for name, param in spec_params.items():
                try:
                    lg = getattr(param, "link_group", None)
                    params[name] = {
                        "value": getattr(param, "value", None),
                        "fixed": bool(getattr(param, "fixed", False)),
                        "link_group": int(lg) if lg else None
                    }
                except Exception:
                    params[name] = {"value": None, "fixed": False, "link_group": None}
            
            return {
                "model_name": self._resolution_model_name,
                "parameters": params
            }
        except Exception as e:
            log_exception("Failed to get resolution state", e, vm=self)
            return None

    def set_resolution_state(self, state: _typing.Optional[dict]) -> bool:
        """Restore resolution state from persistence.
        
        Args:
            state: Dict containing resolution model name and parameters, or None to clear
            
        Returns:
            True if state was applied successfully
        """
        if state is None:
            self._resolution_model_name = "None"
            self._resolution_spec = None
            try:
                self.resolution_updated.emit()
            except Exception:
                pass
            return True
        
        try:
            model_name = state.get("model_name", "None")
            if not model_name or model_name == "None":
                self._resolution_model_name = "None"
                self._resolution_spec = None
                try:
                    self.resolution_updated.emit()
                except Exception:
                    pass
                return True
            
            # Set the model (this creates the spec)
            self.set_resolution_model(model_name)
            
            if self._resolution_spec is None:
                return False
            
            # Apply saved parameters
            params = state.get("parameters", {})
            spec_params = getattr(self._resolution_spec, "params", {}) or {}
            
            for name, param_info in params.items():
                if name in spec_params:
                    try:
                        # Apply value
                        value = param_info.get("value")
                        if value is not None:
                            spec_params[name].value = value
                        
                        # Apply fixed state
                        spec_params[name].fixed = bool(param_info.get("fixed", False))
                        
                        # Apply link group
                        link_group = param_info.get("link_group")
                        spec_params[name].link_group = int(link_group) if link_group else None
                    except Exception:
                        pass
            
            try:
                self.resolution_updated.emit()
            except Exception:
                pass
            
            return True
        except Exception as e:
            log_exception("Failed to set resolution state", e, vm=self)
            return False

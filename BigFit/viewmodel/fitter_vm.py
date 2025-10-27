# viewmodel/fitter_vm.py
from PySide6.QtCore import QObject, Signal
import numpy as np

from models.model_state import ModelState
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

    def apply_parameters(self, gauss=None, lorentz=None, temp=None):
        """Apply new model parameters from the GUI."""
        if gauss is not None:
            self.state.model.gauss_fwhm = gauss
        if lorentz is not None:
            self.state.model.lorentz_fwhm = lorentz
        if temp is not None:
            self.state.model.T = temp

        self.log_message.emit(f"Updated parameters: G={gauss}, L={lorentz}, T={temp}")
        self.update_plot()

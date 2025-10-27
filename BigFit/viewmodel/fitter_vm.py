# viewmodel/fitter_vm.py
from PySide6.QtCore import QObject, Signal
import numpy as np

from models.model_state import ModelState
from dataio.data_loader import select_and_load_files
from dataio.data_saver import save_dataset


class FitterViewModel(QObject):
    """
    Central logic layer: handles loading/saving, fitting, and updates to the plot.
    """

    plot_updated = Signal(object, object, object)  # x, y_data, y_fit
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
    # Fit + Plot logic
    # --------------------------
    def run_fit(self):
        """Placeholder for fitting routine."""
        y_fit = self.state.evaluate()
        self.plot_updated.emit(self.state.x_data, self.state.y_data, y_fit)
        self.log_message.emit("Fit completed (mock).")

    def update_plot(self):
        """Update plot without running a fit."""
        y_fit = self.state.evaluate()
        self.plot_updated.emit(self.state.x_data, self.state.y_data, y_fit)

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

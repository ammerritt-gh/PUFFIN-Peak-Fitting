# viewmodel/fitter_vm.py
from PySide6.QtCore import QObject, Signal
import numpy as np
from models.model_state import ModelState


class FitterViewModel(QObject):
    """
    Acts as the bridge between the GUI (View) and the model state (ModelState).
    """

    plot_updated = Signal(object, object, object)  # x, y_data, y_fit
    log_message = Signal(str)

    def __init__(self, model_state=None):
        super().__init__()
        self.state = model_state or ModelState()

    # --- UI triggers ---
    def load_data(self):
        """Example: generate synthetic data (later can load from file)."""
        self.state.x_data = np.linspace(-20, 20, 801)
        self.state.y_data = np.exp(-0.5 * (self.state.x_data / 3) ** 2)
        self.log_message.emit("Loaded example data.")
        self.update_plot()

    def run_fit(self):
        """Placeholder fit routine — for now just evaluate current model."""
        y_fit = self.state.evaluate()
        self.log_message.emit("Fit completed (mock).")
        self.plot_updated.emit(self.state.x_data, self.state.y_data, y_fit)

    def update_plot(self):
        y_fit = self.state.evaluate()
        self.plot_updated.emit(self.state.x_data, self.state.y_data, y_fit)

    def apply_parameters(self, gauss=None, lorentz=None, temp=None):
        """Called when Apply button is pressed in the UI."""
        if gauss is not None and lorentz is not None:
            self.state.update_resolution(gauss, lorentz)
        if temp is not None:
            self.state.update_temperature(temp)
        self.log_message.emit(f"Updated parameters: G={gauss}, L={lorentz}, T={temp}")
        self.update_plot()


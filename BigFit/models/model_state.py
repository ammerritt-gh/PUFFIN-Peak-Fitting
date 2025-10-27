# model/model_state.py
import numpy as np
from models.dho_voigt_model import DhoVoigtComposite


class ModelState:
    """
    Holds current dataset, model, and parameters for the fitting session.
    """

    def __init__(self):
        self.x_data = np.linspace(-20, 20, 801)
        self.y_data = np.exp(-0.5 * (self.x_data / 3) ** 2) + np.random.normal(0, 0.02, len(self.x_data))
        self.model = DhoVoigtComposite(gauss_fwhm=1.14, lorentz_fwhm=0.28, bg=0.0, elastic_height=200.0)
        self.fit_result = None

    # -- API for ViewModel --
    def evaluate(self):
        """Return model prediction for current parameters."""
        return self.model.evaluate(self.x_data)

    def add_phonon(self, center, height, damping=0.1):
        self.model.add_peak(center=center, height=height, damping=damping)

    def update_resolution(self, gauss, lorentz):
        self.model.gauss_fwhm = gauss
        self.model.lorentz_fwhm = lorentz

    def update_temperature(self, T):
        self.model.T = T

    def snapshot(self):
        """Serialize model state to a dictionary (for saving or thread-safe transfer)."""
        return {
            "x": self.x_data.tolist(),
            "y": self.y_data.tolist(),
            "model": self.model.snapshot(),
        }

    def load_from_snapshot(self, snap):
        self.x_data = np.array(snap["x"])
        self.y_data = np.array(snap["y"])
        self.model = DhoVoigtComposite.from_snapshot(snap["model"])


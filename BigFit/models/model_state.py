# models/model_state.py
import numpy as np
from typing import Any, Dict, Optional
from models.model_specs import get_model_spec, BaseModelSpec


class ModelState:
    """
    Holds the current dataset, active model type, and its parameters.
    This class is model-agnostic and does not depend on any specific
    implementation like DhoVoigtComposite.
    """

    def __init__(self, model_name: str = "DHO+Voigt"):
        # Experimental data
        self.x_data = np.linspace(-20, 20, 801)
        self.y_data = np.exp(-0.5 * (self.x_data / 3) ** 2) + np.random.normal(0, 0.02, len(self.x_data))

        # Current model and fit result
        self.model_name = model_name
        self.model_spec: BaseModelSpec = get_model_spec(model_name)
        self.fit_result: Optional[Dict[str, Any]] = None

        # Initialize parameters based on current data
        self.model_spec.initialize(self.x_data, self.y_data)

    # ------------------------------------------------------------------
    # Data accessors
    # ------------------------------------------------------------------
    def set_data(self, x: np.ndarray, y: np.ndarray):
        self.x_data = np.asarray(x)
        self.y_data = np.asarray(y)
        self.model_spec.initialize(x, y)

    def get_data(self):
        return self.x_data, self.y_data

    # ------------------------------------------------------------------
    # Model parameter handling
    # ------------------------------------------------------------------
    def get_parameters(self) -> Dict[str, Any]:
        """Return parameter dict for UI/viewmodel."""
        return self.model_spec.get_parameters()

    def update_parameters(self, params: Dict[str, Any]):
        """Update the internal parameter set."""
        for name, spec in self.model_spec.params.items():
            if name in params:
                spec.value = params[name]

    # ------------------------------------------------------------------
    # Evaluation (placeholder — to be linked to model factory)
    # ------------------------------------------------------------------
    def evaluate(self, model_func=None):
        """
        Evaluate current model. The callable model_func should be provided
        by the fitting layer or external function. If not, returns zeros.
        """
        if callable(model_func):
            return model_func(self.x_data, self.model_spec)
        return np.zeros_like(self.x_data)

    # ------------------------------------------------------------------
    # Snapshot/save helpers
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Serialize current state for saving."""
        return {
            "model_name": self.model_name,
            "x": self.x_data.tolist(),
            "y": self.y_data.tolist(),
            "parameters": {k: v.value for k, v in self.model_spec.params.items()},
        }

    def load_from_snapshot(self, snap: Dict[str, Any]):
        """Restore model state from a snapshot."""
        self.model_name = snap.get("model_name", self.model_name)
        self.x_data = np.array(snap["x"])
        self.y_data = np.array(snap["y"])

        # rebuild the model spec
        self.model_spec = get_model_spec(self.model_name)
        for k, v in snap.get("parameters", {}).items():
            if k in self.model_spec.params:
                self.model_spec.params[k].value = v

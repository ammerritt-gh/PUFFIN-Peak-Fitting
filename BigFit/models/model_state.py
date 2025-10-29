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

    def __init__(self, model_name: str = "Voigt"):
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
        by the fitting layer or external function. If not provided, try common
        places for a concrete model or model_spec to produce the prediction.

        This implementation builds a plain params dict from model_spec.params
        (and any attributes on state.model) and attempts to call callables
        with signature (x, params) first, then falls back to (x).
        """
        # build a simple params dict from model_spec.params
        params = {}
        spec = getattr(self, "model_spec", None)
        if spec is not None:
            try:
                # use the spec helper if available
                if hasattr(spec, "get_param_values"):
                    params = spec.get_param_values()
                else:
                    for n, p in getattr(spec, "params", {}).items():
                        try:
                            params[n] = getattr(p, "value", None)
                        except Exception:
                            params[n] = None
            except Exception:
                params = {}

        # overlay any attributes stored on a concrete model object
        mdl = getattr(self, "model", None)
        if mdl is not None:
            try:
                for k in list(params.keys()):
                    if hasattr(mdl, k):
                        try:
                            params[k] = getattr(mdl, k)
                        except Exception:
                            pass
                # also include any extra attrs on the model that aren't in spec
                for k in dir(mdl):
                    if k.startswith("_"):
                        continue
                    if k not in params:
                        try:
                            v = getattr(mdl, k)
                            # skip methods
                            if not callable(v):
                                params[k] = v
                        except Exception:
                            pass
            except Exception:
                pass

        # 1) explicit callable wins: try with (x, params) then (x)
        if callable(model_func):
            try:
                return model_func(self.x_data, params)
            except Exception:
                try:
                    return model_func(self.x_data)
                except Exception:
                    pass

        # 2) concrete model instance on state (callable or has evaluate())
        if mdl is not None:
            if callable(mdl):
                try:
                    return mdl(self.x_data, params)
                except Exception:
                    try:
                        return mdl(self.x_data)
                    except Exception:
                        pass
            if hasattr(mdl, "evaluate") and callable(getattr(mdl, "evaluate")):
                try:
                    return mdl.evaluate(self.x_data, params)
                except Exception:
                    try:
                        return mdl.evaluate(self.x_data)
                    except Exception:
                        pass

        # 3) attached model_spec: try evaluate(x, params) then evaluate(x)
        if spec is not None:
            if hasattr(spec, "evaluate") and callable(getattr(spec, "evaluate")):
                try:
                    return spec.evaluate(self.x_data, params)
                except Exception:
                    try:
                        return spec.evaluate(self.x_data)
                    except Exception:
                        pass
            # try common maker helpers that may return a callable accepting params
            for maker in ("to_callable", "to_model", "get_callable", "get_model"):
                if hasattr(spec, maker) and callable(getattr(spec, maker)):
                    try:
                        fn = getattr(spec, maker)()
                        if callable(fn):
                            try:
                                return fn(self.x_data, params)
                            except Exception:
                                try:
                                    return fn(self.x_data)
                                except Exception:
                                    pass
                    except Exception:
                        pass

        # nothing usable: return zeros matching x
        try:
            return np.zeros_like(self.x_data)
        except Exception:
            return np.array([])

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

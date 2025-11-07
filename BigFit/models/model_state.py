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
        # x values centered on zero
        self.x_data = np.linspace(-20, 20, 161)
        # Create a Gaussian-shaped expectation (center 0, sigma=3) and sample
        # Poisson-distributed counts from it to emulate counting data. Use a
        # reasonable peak count so the Poisson noise is visible but not
        # overwhelming.
        peak_counts = 50.0
        expectation = peak_counts * np.exp(-0.5 * (self.x_data / 3) ** 2) + 1
        # Poisson sample — returns integers; cast to float for downstream code
        try:
            self.y_data = np.random.poisson(expectation).astype(float)
        except Exception:
            # fallback to Gaussian + small noise if Poisson sampling fails
            self.y_data = expectation + np.random.normal(0, 0.02, len(self.x_data))
        # Poisson errors: sqrt(counts). Avoid zero uncertainties by flooring to 1.0
        try:
            self.errors = np.sqrt(np.maximum(self.y_data, 1.0)).astype(float)
        except Exception:
            self.errors = np.full_like(self.x_data, 1.0, dtype=float)

        # Current model and fit result
        self.model_name = model_name
        self.model_spec: BaseModelSpec = get_model_spec(model_name)
        self.fit_result: Optional[Dict[str, Any]] = None

        # Initialize parameters based on current data
        self.model_spec.initialize(self.x_data, self.y_data)
        # Exclusion mask (False = included, True = excluded)
        try:
            self.excluded = np.zeros_like(self.x_data, dtype=bool)
        except Exception:
            self.excluded = np.array([], dtype=bool)

    # ------------------------------------------------------------------
    # Data accessors
    # ------------------------------------------------------------------
    def set_data(self, x: np.ndarray, y: np.ndarray):
        self.x_data = np.asarray(x)
        self.y_data = np.asarray(y)
        # Default error handling for user-provided data: if data look like
        # counting data (non-negative integers), use sqrt(counts); otherwise
        # fall back to a small constant uncertainty.
        try:
            y_arr = np.asarray(y, dtype=float)
            if np.all(y_arr >= 0) and np.all(np.isfinite(y_arr)):
                # treat as counts (or non-negative measurements)
                self.errors = np.sqrt(np.maximum(y_arr, 1.0)).astype(float)
            else:
                self.errors = np.full_like(self.x_data, 0.02, dtype=float)
        except Exception:
            self.errors = np.full_like(self.x_data, 0.02, dtype=float)
        self.model_spec.initialize(x, y)
        # reset exclusion mask to match new data length
        try:
            self.excluded = np.zeros_like(self.x_data, dtype=bool)
        except Exception:
            self.excluded = np.array([], dtype=bool)

    def get_data(self):
        return self.x_data, self.y_data

    # ------------------------------------------------------------------
    # Exclusion helpers
    # ------------------------------------------------------------------
    def get_included_mask(self):
        """Return boolean mask where True means included (not excluded)."""
        try:
            return ~np.asarray(self.excluded, dtype=bool)
        except Exception:
            try:
                return np.ones_like(self.x_data, dtype=bool)
            except Exception:
                return np.array([], dtype=bool)

    def get_masked_data(self):
        """Return (x_included, y_included, err_included or None) for fitting.

        err array is optional and returned if available on state as `errors`.
        """
        mask = self.get_included_mask()
        try:
            x = np.asarray(self.x_data)[mask]
            y = np.asarray(self.y_data)[mask]
        except Exception:
            return np.array([]), np.array([]), None
        err = getattr(self, "errors", None)
        if err is None:
            return x, y, None
        try:
            err = np.asarray(err)[mask]
        except Exception:
            err = None
        return x, y, err

    def toggle_point_exclusion(self, x, y, tol=0.05):
        """Toggle exclusion state of the nearest point to (x,y) within tol.
        Returns index toggled or None if none found.
        """
        try:
            xd = np.asarray(self.x_data)
            yd = np.asarray(self.y_data)
            if len(xd) == 0:
                return None
            dists = np.hypot(xd - float(x), yd - float(y))
            idx = int(np.argmin(dists))
            if dists[idx] <= float(tol):
                try:
                    self.excluded[idx] = not bool(self.excluded[idx])
                except Exception:
                    pass
                return idx
        except Exception:
            pass
        return None

    def toggle_box_exclusion(self, x0, y0, x1, y1):
        """Toggle exclusion for points inside the axis-aligned box.

        If all points inside are currently excluded, they will be set to included.
        Otherwise all points within the box will be set to excluded.
        Returns array of indices affected (may be empty).
        """
        try:
            xd = np.asarray(self.x_data)
            yd = np.asarray(self.y_data)
            if len(xd) == 0:
                return np.array([], dtype=int)
            lx, hx = (min(x0, x1), max(x0, x1))
            ly, hy = (min(y0, y1), max(y0, y1))
            inside = (xd >= lx) & (xd <= hx) & (yd >= ly) & (yd <= hy)
            inds = np.nonzero(inside)[0]
            if inds.size == 0:
                return inds
            # if all inside are excluded, un-exclude them, else exclude them
            if np.all(self.excluded[inds]):
                self.excluded[inds] = False
            else:
                self.excluded[inds] = True
            return inds
        except Exception:
            return np.array([], dtype=int)

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
            "excluded": np.asarray(self.excluded).astype(bool).tolist(),
        }

    def load_from_snapshot(self, snap: Dict[str, Any]):
        """Restore model state from a snapshot."""
        self.model_name = snap.get("model_name", self.model_name)
        self.x_data = np.array(snap["x"])
        self.y_data = np.array(snap["y"])

        # restore exclusion mask if present, otherwise reset
        exc = snap.get("excluded", None)
        try:
            if exc is None:
                self.excluded = np.zeros_like(self.x_data, dtype=bool)
            else:
                arr = np.asarray(exc, dtype=bool)
                # if length mismatches, attempt to pad/truncate
                if len(arr) != len(self.x_data):
                    a = np.zeros_like(self.x_data, dtype=bool)
                    a[: min(len(a), len(arr))] = arr[: len(a)]
                    self.excluded = a
                else:
                    self.excluded = arr
        except Exception:
            try:
                self.excluded = np.zeros_like(self.x_data, dtype=bool)
            except Exception:
                self.excluded = np.array([], dtype=bool)

        # rebuild the model spec
        self.model_spec = get_model_spec(self.model_name)
        for k, v in snap.get("parameters", {}).items():
            if k in self.model_spec.params:
                self.model_spec.params[k].value = v

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

        # Information about the last-loaded file (path, name, size, ...)
        # Initialized here so callers can safely assign `model_state.file_info`.
        self.file_info: Optional[Dict[str, Any]] = None

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
                self.excluded[idx] = not bool(self.excluded[idx])
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
            self.excluded[inds] = not np.all(self.excluded[inds])
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
    def _try_call_with_fallback(self, func, x, params):
        """Helper to try calling func(x, params) then fallback to func(x)."""
        if not callable(func):
            return None
        try:
            return func(x, params)
        except (TypeError, Exception):
            try:
                return func(x)
            except Exception:
                return None
    
    def _get_params_dict(self):
        """Build parameter dict from model_spec and model attributes."""
        params = {}
        spec = getattr(self, "model_spec", None)
        
        # Get params from spec
        if spec is not None and hasattr(spec, "get_param_values"):
            try:
                params = spec.get_param_values()
            except Exception:
                pass
        elif spec is not None:
            for n, p in getattr(spec, "params", {}).items():
                try:
                    params[n] = getattr(p, "value", None)
                except Exception:
                    params[n] = None
        
        # Overlay model object attributes
        mdl = getattr(self, "model", None)
        if mdl is not None:
            for k in list(params.keys()):
                if hasattr(mdl, k):
                    try:
                        params[k] = getattr(mdl, k)
                    except Exception:
                        pass
            # Include extra non-private non-callable attributes
            for k in dir(mdl):
                if not k.startswith("_") and k not in params:
                    try:
                        v = getattr(mdl, k)
                        if not callable(v):
                            params[k] = v
                    except Exception:
                        pass
        
        return params
    
    def evaluate(self, model_func=None):
        """
        Evaluate current model. The callable model_func should be provided
        by the fitting layer or external function. If not provided, try common
        places for a concrete model or model_spec to produce the prediction.

        This implementation builds a plain params dict from model_spec.params
        (and any attributes on state.model) and attempts to call callables
        with signature (x, params) first, then falls back to (x).
        """
        params = self._get_params_dict()
        
        # 1) Try explicit callable first
        if callable(model_func):
            result = self._try_call_with_fallback(model_func, self.x_data, params)
            if result is not None:
                return result
        
        # 2) Try concrete model instance
        mdl = getattr(self, "model", None)
        if mdl is not None:
            # Try evaluate method
            if hasattr(mdl, "evaluate"):
                result = self._try_call_with_fallback(mdl.evaluate, self.x_data, params)
                if result is not None:
                    return result
            # Try calling model directly
            result = self._try_call_with_fallback(mdl, self.x_data, params)
            if result is not None:
                return result
        
        # 3) Try attached model_spec
        spec = getattr(self, "model_spec", None)
        if spec is not None:
            # Try evaluate method
            if hasattr(spec, "evaluate"):
                result = self._try_call_with_fallback(spec.evaluate, self.x_data, params)
                if result is not None:
                    return result
            # Try maker helpers
            for maker_name in ("to_callable", "to_model", "get_callable", "get_model"):
                if hasattr(spec, maker_name):
                    try:
                        maker_func = getattr(spec, maker_name)
                        if callable(maker_func):
                            fn = maker_func()
                            result = self._try_call_with_fallback(fn, self.x_data, params)
                            if result is not None:
                                return result
                    except Exception:
                        pass
        
        # Fallback: return zeros matching x
        return np.zeros_like(self.x_data) if len(self.x_data) > 0 else np.array([])

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
            "fixed": {k: bool(getattr(v, 'fixed', False)) for k, v in self.model_spec.params.items()},
            "link_groups": {k: getattr(v, 'link_group', None) for k, v in self.model_spec.params.items()},
            "excluded": np.asarray(self.excluded).astype(bool).tolist(),
        }

    def load_from_snapshot(self, snap: Dict[str, Any]):
        """Restore model state from a snapshot."""
        self.model_name = snap.get("model_name", self.model_name)
        self.x_data = np.array(snap["x"])
        self.y_data = np.array(snap["y"])

        # Restore exclusion mask with length matching
        exc = snap.get("excluded", None)
        try:
            if exc is None:
                self.excluded = np.zeros_like(self.x_data, dtype=bool)
            else:
                arr = np.asarray(exc, dtype=bool)
                if len(arr) != len(self.x_data):
                    # Pad or truncate to match data length
                    self.excluded = np.zeros_like(self.x_data, dtype=bool)
                    min_len = min(len(self.excluded), len(arr))
                    self.excluded[:min_len] = arr[:min_len]
                else:
                    self.excluded = arr
        except Exception:
            self.excluded = np.zeros_like(self.x_data, dtype=bool)

        # Rebuild the model spec
        self.model_spec = get_model_spec(self.model_name)
        
        # Restore parameters
        for k, v in snap.get("parameters", {}).items():
            if k in self.model_spec.params:
                self.model_spec.params[k].value = v
        
        # Restore fixed state
        for k, fv in snap.get("fixed", {}).items():
            if k in self.model_spec.params:
                try:
                    self.model_spec.params[k].fixed = bool(fv)
                except Exception:
                    pass
        
        # Restore link groups
        for k, lg in snap.get("link_groups", {}).items():
            if k in self.model_spec.params:
                try:
                    self.model_spec.params[k].link_group = int(lg) if lg else None
                except Exception:
                    pass
